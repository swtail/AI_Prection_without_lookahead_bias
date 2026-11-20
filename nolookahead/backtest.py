"""Walk-forward backtester that is lookahead-free by construction.

At every decision time ``T`` the engine:

1. Takes a point-in-time snapshot (only data with ``available_time <= T``).
2. Builds the training set from rows whose *label is fully realized* by ``T``
   (the label for event ``t`` with horizon ``h`` needs the target at ``t+h``,
   and that value must itself have been published by ``T``). This is the
   "purge" step - overlapping labels that peek past ``T`` are dropped.
3. Uses *as-was* features: the feature row for event ``t`` is computed from
   the snapshot as of ``t``, so training rows look exactly like the row the
   model will be served live (no training/serving skew from data revisions).
4. Fits a brand-new model (scalers included) on that training set only, and
   predicts the row for ``T``.

Realized outcomes are scored afterwards from the fully revised data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .store import PointInTimeStore

FeatureFn = Callable[[pd.DataFrame], pd.DataFrame]
ModelFactory = Callable[[], object]


class LookaheadError(RuntimeError):
    """Raised when a configuration would let future information leak in."""


@dataclass
class WalkForwardBacktester:
    store: PointInTimeStore
    feature_fn: FeatureFn
    model_factory: ModelFactory
    target_field: str
    horizon: int = 1
    min_train: int = 100
    max_train: int | None = None
    refit_every: int = 1
    embargo: int = 0
    knowledge_cutoff: object | None = None
    _feature_cache: dict = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------ public API
    def with_store(self, store: PointInTimeStore) -> "WalkForwardBacktester":
        return WalkForwardBacktester(
            store=store,
            feature_fn=self.feature_fn,
            model_factory=self.model_factory,
            target_field=self.target_field,
            horizon=self.horizon,
            min_train=self.min_train,
            max_train=self.max_train,
            refit_every=self.refit_every,
            embargo=self.embargo,
            knowledge_cutoff=self.knowledge_cutoff,
        )

    def decision_times(self) -> pd.Index:
        """Event times of the target, skipping the first ``min_train + horizon``."""
        times = self.store.first_available(self.target_field).index
        return times[self.min_train + self.horizon :]

    def predict_at(self, T) -> float:
        model = self._fit(T)
        if model is None:
            return np.nan
        x = self._features_as_of(T)
        if x is None or x.isna().any():
            return np.nan
        return float(model.predict(x.to_frame().T)[0])

    def run(self, decision_times=None) -> pd.DataFrame:
        times = self.decision_times() if decision_times is None else pd.Index(decision_times)
        realized = self._labels(self.store.latest())
        rows, model = [], None
        for i, T in enumerate(times):
            if model is None or i % self.refit_every == 0:
                model = self._fit(T)
            x = self._features_as_of(T)
            pred = np.nan
            if model is not None and x is not None and not x.isna().any():
                pred = float(model.predict(x.to_frame().T)[0])
            rows.append((T, pred, realized.get(T, np.nan)))
        out = pd.DataFrame(rows, columns=["decision_time", "prediction", "realized"])
        return out.set_index("decision_time")

    # ------------------------------------------------------------ internals
    def _check_cutoff(self, T) -> None:
        if self.knowledge_cutoff is not None and T <= self.knowledge_cutoff:
            raise LookaheadError(
                f"decision time {T} is on or before the model's knowledge cutoff "
                f"{self.knowledge_cutoff}: a pretrained model has already seen what "
                "happened next. Only backtest on periods after the cutoff."
            )

    def _features_as_of(self, t) -> pd.Series | None:
        """Feature row for event ``t`` computed from the snapshot as of ``t`` (cached)."""
        if t not in self._feature_cache:
            snap = self.store.snapshot(t)
            feats = self.feature_fn(snap) if not snap.empty else pd.DataFrame()
            self._feature_cache[t] = feats.loc[t] if t in feats.index else None
        return self._feature_cache[t]

    def _labels(self, snap: pd.DataFrame) -> pd.Series:
        """Forward return over ``horizon`` steps of the target, indexed by event time."""
        y = snap[self.target_field].dropna()
        return (y.shift(-self.horizon) / y - 1.0).dropna()

    def _label_realized_time(self) -> pd.Series:
        """For event t: when every input to label(t) had been published."""
        first = self.store.first_available(self.target_field)
        # Max over positions t .. t+horizon. Done with shifts rather than a
        # numeric rolling window so any orderable time type (datetimes of any
        # resolution/timezone, ints) is handled without unit conversions.
        # A label whose window runs off the end gets NaN/NaT (never realized).
        shifted = pd.concat([first.shift(-k) for k in range(self.horizon + 1)], axis=1)
        return shifted.max(axis=1, skipna=False)

    def _fit(self, T):
        self._check_cutoff(T)
        snap = self.store.snapshot(T)
        if snap.empty or self.target_field not in snap:
            return None
        labels = self._labels(snap)
        realized_at = self._label_realized_time().reindex(labels.index)
        target_times = self.store.first_available(self.target_field).index
        # Purge: only labels fully known at T. Embargo: drop a further `embargo`
        # target periods right before T.
        cutoff_pos = target_times.searchsorted(T, side="right") - self.embargo
        cutoff_time = target_times[cutoff_pos - 1] if cutoff_pos > 0 else None
        ok = realized_at.notna() & (realized_at <= T)
        if cutoff_time is not None:
            ok &= realized_at <= cutoff_time
        labels = labels[ok.to_numpy()]
        if self.max_train is not None:
            labels = labels.iloc[-self.max_train :]
        if len(labels) < self.min_train:
            return None
        rows = {t: self._features_as_of(t) for t in labels.index}
        rows = {t: r for t, r in rows.items() if r is not None}
        if len(rows) < self.min_train:
            return None
        X = pd.DataFrame(rows).T.apply(pd.to_numeric)
        labels = labels.loc[X.index]
        mask = X.notna().all(axis=1).to_numpy()
        if mask.sum() < self.min_train:
            return None
        model = self.model_factory()
        model.fit(X[mask], labels[mask].to_numpy())
        return model

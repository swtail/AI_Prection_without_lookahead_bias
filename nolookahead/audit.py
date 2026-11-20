"""Automated lookahead audits.

Two complementary tests:

* :func:`check_future_invariance` - black-box. Scramble every record that was
  unknown at ``T`` and re-run the prediction for ``T``. Any change means
  something from the future reached the model.

* :func:`check_feature_causality` - white-box for a feature function. A
  causal feature's value for event ``t`` must be the same whether it is
  computed on data up to ``t`` or on a longer history. Centered windows,
  ``shift(-k)``, full-sample normalisation, ``bfill`` etc. all fail this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import FeatureFn, WalkForwardBacktester
from .store import PointInTimeStore


@dataclass
class InvarianceResult:
    decision_time: object
    original: float
    perturbed: float

    @property
    def passed(self) -> bool:
        if np.isnan(self.original) and np.isnan(self.perturbed):
            return True
        return bool(np.isclose(self.original, self.perturbed, rtol=1e-9, atol=1e-12))


def check_future_invariance(
    backtester: WalkForwardBacktester,
    decision_times,
    seed: int = 0,
) -> list[InvarianceResult]:
    results = []
    rng = np.random.default_rng(seed)
    for T in decision_times:
        original = backtester.with_store(backtester.store).predict_at(T)
        perturbed_bt = backtester.with_store(backtester.store.perturb_after(T, rng))
        results.append(InvarianceResult(T, original, perturbed_bt.predict_at(T)))
    return results


def check_feature_causality(
    feature_fn: FeatureFn,
    store: PointInTimeStore,
    times,
    rtol: float = 1e-9,
) -> pd.DataFrame:
    """Return the (time, feature) pairs whose value depends on later events.

    For each ``t`` the feature function is run twice: on all events up to
    ``t``, and on the full history. A causal function gives the same row for
    ``t`` both times. Publication lags and revisions are the store's job, not
    this check's: the check uses first-release values with every value
    treated as published at its event time, so that only "looking forward in
    event time" (centered windows, ``shift(-k)``, ``bfill``, full-sample
    scaling...) is tested.

    An empty frame means the feature function passed.
    """
    rec = store.first_release().records
    rec["available_time"] = rec["event_time"]
    everything = PointInTimeStore(rec).latest()
    full = feature_fn(everything)
    bad = []
    for t in times:
        local = _row(feature_fn(everything[everything.index <= t]), t)
        with_future = _row(full, t)
        cols = local.index.union(with_future.index)
        for col in cols:
            x, y = local.get(col, np.nan), with_future.get(col, np.nan)
            if pd.isna(x) and pd.isna(y):
                continue
            if pd.isna(x) or pd.isna(y) or not np.isclose(x, y, rtol=rtol):
                bad.append((t, col, x, y))
    return pd.DataFrame(bad, columns=["time", "feature", "value_as_of_t", "value_with_future"])


def _row(frame: pd.DataFrame, t) -> pd.Series:
    if t in frame.index:
        return frame.loc[t]
    return pd.Series(np.nan, index=frame.columns)

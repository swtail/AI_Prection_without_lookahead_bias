import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from nolookahead import (
    LookaheadError,
    PointInTimeStore,
    WalkForwardBacktester,
    check_feature_causality,
    check_future_invariance,
    purged_walk_forward_splits,
)


def _price_store(n=200, seed=0, dates=True):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n) if dates else pd.RangeIndex(n)
    price = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    return PointInTimeStore.from_series(price, "price")


def _feats(snap):
    p = snap["price"]
    return pd.DataFrame({"r1": p.pct_change(), "r5": p.pct_change(5)})


class RecordingModel:
    """Remembers what it was trained on so tests can inspect it."""

    fits: list = []

    def fit(self, X, y):
        RecordingModel.fits.append(X.index.copy())
        self._m = Ridge().fit(X, y)
        return self

    def predict(self, X):
        return self._m.predict(X)


# ---------------------------------------------------------------- store
def test_snapshot_respects_publication_lag_and_revisions():
    idx = pd.bdate_range("2021-01-01", periods=3)
    lag = pd.offsets.BDay(2)
    store = PointInTimeStore.from_series(pd.Series([1.0, 2.0, 3.0], idx), "gdp", lag)
    rev = pd.DataFrame(
        {"event_time": [idx[0]], "available_time": [idx[0] + pd.offsets.BDay(4)],
         "field": ["gdp"], "value": [1.5]}
    )
    store = store.with_revisions(rev)

    assert store.snapshot(idx[1]).empty  # nothing published yet
    snap = store.snapshot(idx[0] + lag)
    assert list(snap["gdp"]) == [1.0]
    assert store.snapshot(idx[0] + pd.offsets.BDay(4)).loc[idx[0], "gdp"] == 1.5
    assert store.first_release().latest().loc[idx[0], "gdp"] == 1.0


def test_store_rejects_values_known_before_their_period():
    with pytest.raises(ValueError):
        PointInTimeStore(pd.DataFrame(
            {"event_time": [5], "available_time": [4], "field": ["x"], "value": [1.0]}
        ))


# ------------------------------------------------------------- backtest
@pytest.mark.parametrize("horizon", [1, 5])
def test_training_labels_are_fully_realized_at_decision_time(horizon):
    store = _price_store()
    RecordingModel.fits = []
    bt = WalkForwardBacktester(store, _feats, RecordingModel, "price",
                               horizon=horizon, min_train=50, refit_every=1)
    all_times = store.latest().index
    checked = 0
    for T in bt.decision_times()[:20]:
        RecordingModel.fits = []
        bt.predict_at(T)
        if not RecordingModel.fits:
            continue
        checked += 1
        last = all_times.get_loc(RecordingModel.fits[0].max())
        # label(t) needs price at t + horizon, which must be <= T
        assert all_times[last + horizon] <= T
        assert all_times[last + horizon + 1] > T  # and nothing usable was thrown away
    assert checked >= 10


def test_embargo_drops_extra_rows():
    store = _price_store()
    RecordingModel.fits = []
    WalkForwardBacktester(store, _feats, RecordingModel, "price", horizon=1,
                          min_train=50, embargo=3).run(_price_store().latest().index[[120]])
    all_times = store.latest().index
    assert all_times.get_loc(RecordingModel.fits[0].max()) == 120 - 1 - 3


def test_future_invariance_passes_for_engine():
    bt = WalkForwardBacktester(_price_store(), _feats, Ridge, "price", horizon=2, min_train=50)
    results = check_future_invariance(bt, bt.decision_times()[::40])
    assert results and all(r.passed for r in results)
    assert any(not np.isnan(r.original) for r in results)


def test_works_with_integer_time():
    bt = WalkForwardBacktester(_price_store(dates=False), _feats, Ridge, "price",
                               horizon=3, min_train=50, refit_every=10)
    res = bt.run()
    assert res["prediction"].notna().sum() > 50
    assert all(r.passed for r in check_future_invariance(bt, bt.decision_times()[::50]))


def test_knowledge_cutoff_guard():
    store = _price_store()
    times = store.latest().index
    bt = WalkForwardBacktester(store, _feats, Ridge, "price", min_train=50,
                               knowledge_cutoff=times[150])
    with pytest.raises(LookaheadError):
        bt.predict_at(times[100])
    assert not np.isnan(bt.predict_at(times[160]))


# ---------------------------------------------------------------- audit
def test_feature_causality_flags_forward_looking_features():
    store = _price_store()
    times = store.latest().index[20:-20:10]

    def leaky(snap):
        p = snap["price"]
        return pd.DataFrame({
            "ok": p.pct_change(),
            "next": p.shift(-1),
            "centered": p.rolling(5, center=True).mean(),
            "zscore": (p - p.mean()) / p.std(),  # full-sample normalisation
        })

    bad = check_feature_causality(leaky, store, times)
    assert set(bad["feature"]) == {"next", "centered", "zscore"}
    assert check_feature_causality(_feats, store, times).empty


# --------------------------------------------------------------- splits
@pytest.mark.parametrize("horizon,embargo", [(1, 0), (5, 2)])
def test_purged_splits(horizon, embargo):
    splits = list(purged_walk_forward_splits(100, 4, horizon, embargo))
    assert len(splits) == 4
    for train, test in splits:
        assert train.max() + horizon < test.min() - embargo + 1
        assert not set(train) & set(test)

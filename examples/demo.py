"""Demo: how much lookahead bias inflates a backtest, and how the toolkit removes it.

Synthetic world:
  * A daily price series.
  * A macro indicator that genuinely predicts the next day's return, but it is
    published 3 business days late, and the first release is noisy; it gets
    revised to the true value 20 days later.

A naive backtest uses today's revised macro numbers as if they had been known
on the event date, and a centered smoothing window. It looks great. The
point-in-time backtest shows the skill you could actually have had.

Run:  python examples/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nolookahead import (  # noqa: E402
    PointInTimeStore,
    WalkForwardBacktester,
    check_feature_causality,
    check_future_invariance,
)

N = 600
LAG = pd.offsets.BDay(3)
REVISION = pd.offsets.BDay(20)


def make_world(seed: int = 7) -> PointInTimeStore:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=N)
    macro = pd.Series(rng.normal(size=N), index=dates)
    ret = 0.004 * macro.shift(1).fillna(0) + rng.normal(0, 0.01, N)
    price = 100 * (1 + ret).cumprod()

    first_release = macro + rng.normal(0, 1.0, N)  # noisy flash estimate
    store = PointInTimeStore.concat(
        [
            PointInTimeStore.from_series(price, "price"),
            PointInTimeStore.from_series(first_release, "macro", publication_lag=LAG),
        ]
    )
    revisions = pd.DataFrame(
        {
            "event_time": dates,
            "available_time": dates + LAG + REVISION,
            "field": "macro",
            "value": macro.to_numpy(),
        }
    )
    return store.with_revisions(revisions)


def causal_features(snap: pd.DataFrame) -> pd.DataFrame:
    price = snap["price"].dropna()
    macro = snap["macro"].ffill().reindex(price.index)
    return pd.DataFrame(
        {
            "ret1": price.pct_change(),
            "mom5": price.pct_change(5),
            "macro": macro,
        }
    )


def leaky_features(snap: pd.DataFrame) -> pd.DataFrame:
    """Typical mistakes: centered window + ignoring that macro was published late."""
    price = snap["price"].dropna()
    ret = price.pct_change()
    return pd.DataFrame(
        {
            "ret1": ret,
            "ret_smooth": ret.rolling(5, center=True).mean(),  # uses t+1, t+2
            "macro": snap["macro"].reindex(price.index),
        }
    )


def model():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def naive_backtest(store: PointInTimeStore, min_train=150) -> pd.Series:
    """What people often do: compute features once on today's data, then walk forward."""
    data = store.latest()
    X = leaky_features(data)
    y = (data["price"].shift(-1) / data["price"] - 1).reindex(X.index)
    X = X.dropna()
    y = y.reindex(X.index)
    preds = {}
    for i in range(min_train, len(X) - 1):
        m = model().fit(X.iloc[:i], y.iloc[:i])  # y[i-1] already uses price[i]: ok here
        preds[X.index[i]] = m.predict(X.iloc[[i]])[0]
    return pd.Series(preds), y


def ic(pred: pd.Series, realized: pd.Series) -> float:
    df = pd.concat([pred, realized], axis=1, sort=True).dropna()
    return df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman")


def hit_rate(pred, realized) -> float:
    df = pd.concat([pred, realized], axis=1, sort=True).dropna()
    return float((np.sign(df.iloc[:, 0]) == np.sign(df.iloc[:, 1])).mean())


def main() -> None:
    store = make_world()

    print("1) Feature-causality audit")
    sample = store.latest().index[50:-50:25]
    for name, fn in [("leaky_features", leaky_features), ("causal_features", causal_features)]:
        bad = check_feature_causality(fn, store, sample)
        status = "FAIL" if len(bad) else "pass"
        detail = f" -> leaking columns: {sorted(bad['feature'].unique())}" if len(bad) else ""
        print(f"   {name:16s} {status}{detail}")

    print("\n2) Backtests (rank IC / hit rate on next-day return)")
    naive_pred, naive_y = naive_backtest(store)
    print(f"   naive (revised data, leaky features): IC={ic(naive_pred, naive_y):+.3f}  "
          f"hit={hit_rate(naive_pred, naive_y):.1%}")

    bt = WalkForwardBacktester(
        store=store,
        feature_fn=causal_features,
        model_factory=model,
        target_field="price",
        horizon=1,
        min_train=150,
        refit_every=20,
    )
    res = bt.run()
    print(f"   point-in-time walk-forward:            IC={ic(res.prediction, res.realized):+.3f}  "
          f"hit={hit_rate(res.prediction, res.realized):.1%}")

    print("\n3) Future-invariance audit on the point-in-time backtester")
    checks = check_future_invariance(bt, res.index[::100])
    for c in checks:
        print(f"   {c.decision_time.date()}  pred={c.original:+.6f}  "
              f"with scrambled future={c.perturbed:+.6f}  {'pass' if c.passed else 'FAIL'}")


if __name__ == "__main__":
    main()

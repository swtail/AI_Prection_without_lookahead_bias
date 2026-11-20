# AI prediction without lookahead bias

`nolookahead` is a small Python toolkit for backtesting machine-learning forecasts so that, at every decision time, the model only sees what was actually known at that moment.

Lookahead bias usually gets in through four doors:

| Leak | Example | How this toolkit closes it |
|---|---|---|
| **Publication lag** | Using Q1 GDP on 31 March when it was released on 28 April | `PointInTimeStore` keeps `event_time` and `available_time` for every value. `snapshot(T)` returns only rows with `available_time <= T`. |
| **Revisions** | Training on today's revised figures instead of the first estimate | Revisions are stored as extra rows (vintages). A snapshot returns the vintage known at `T`. |
| **Forward-looking features** | `shift(-1)`, `rolling(center=True)`, `bfill`, scaling by full-sample mean/std | `check_feature_causality` reruns your feature function on truncated history and flags every column that changes. |
| **Label overlap / fitting on the future** | An h-day label at `T-1` uses prices after `T`; a scaler fit on the whole dataset | `WalkForwardBacktester` trains only on labels fully realized by `T` (purge + optional embargo) and refits the whole model from scratch at each refit. |

Two more safeguards:

- **As-was features.** The training row for event `t` is computed from the snapshot as of `t`, so training rows look exactly like what the model is served live.
- **Future-invariance audit.** `check_future_invariance` scrambles every record unknown at `T` and checks that the prediction for `T` doesn't change. This is a black-box end-to-end test, so it also catches leaks in your own code.
- **Pretrained models / LLMs.** A foundation model has already "seen" everything before its training cutoff. Pass `knowledge_cutoff=` and the backtester raises `LookaheadError` for any decision time on or before it.

## Quick start

```python
from nolookahead import PointInTimeStore, WalkForwardBacktester, check_future_invariance

store = PointInTimeStore.concat([
    PointInTimeStore.from_series(prices, "price"),                       # known at close
    PointInTimeStore.from_series(gdp_flash, "gdp", publication_lag=lag), # published late
]).with_revisions(gdp_revisions)   # columns: event_time, available_time, field, value

def features(snap):                # snap = what was known at one point in time
    p = snap["price"]
    return pd.DataFrame({"r1": p.pct_change(), "gdp": snap["gdp"].ffill()})

bt = WalkForwardBacktester(store, features, lambda: make_pipeline(StandardScaler(), Ridge()),
                           target_field="price", horizon=5, min_train=250, refit_every=21)
results = bt.run()                 # columns: prediction, realized
assert all(r.passed for r in check_future_invariance(bt, results.index[::50]))
```

`model_factory` can return anything that has `fit(X, y)` and `predict(X)`. Model selection inside a training window should use `purged_walk_forward_splits`, not shuffled K-fold.

## Demo

```bash
pip install -e ".[dev]"
```

```bash
python examples/demo.py
```

The demo builds a synthetic market where a macro indicator predicts next-day returns, but it is published 3 days late as a noisy flash estimate and revised 20 days after that:

```
1) Feature-causality audit
   leaky_features   FAIL -> leaking columns: ['ret_smooth']
   causal_features  pass

2) Backtests (rank IC / hit rate on next-day return)
   naive (revised data, leaky features): IC=+0.571  hit=69.4%
   point-in-time walk-forward:            IC=+0.051  hit=51.6%
```

The naive backtest looks like a money machine. The honest one shows the signal was not usable in time.

## Tests

```bash
python -m pytest -q
```

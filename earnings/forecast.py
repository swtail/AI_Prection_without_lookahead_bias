"""Quarterly EPS forecasts for 1-4 quarters ahead under three information sets.

Model I   - firm only. Classic quarterly time-series models from the accounting
            literature, each fitted on the last 5 and 10 years of the firm's own
            point-in-time EPS history:
              * seasonal random walk (Foster 1977)
              * seasonal random walk with drift (Foster 1977)
              * Brown & Rozeff (1979) SARIMA (1,0,0)x(0,1,1)_4
            The reported forecast is the equal-weight average (combination
            forecasts beat individual models; Clemen 1989).
Model II  - firm + industry. Pooled cross-sectional regression of future ROE on
            the firm's own ROE, loss indicators and Fama-French-48 industry ROE,
            estimated on the firm's industry peers (in the spirit of Fama & French
            2000; Hou, van Dijk & Zhang 2012; Li & Mohanram 2014).
Model III - firm + industry + country. Model II plus country-wide earnings
            (aggregate ROE level and change; Kothari, Lewellen & Warner 2006) and
            real-time macro data (real GDP growth, term spread, 10y yield; Philly
            Fed RTDSM / Treasury, each as known at the time).

Every training label is a realized outcome available at the forecast date, and
estimation windows end at the forecast date, so no future data enters a fit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .panel import calendar_quarter
from .public import macro_state

HORIZONS = (1, 2, 3, 4)


# ============================================================ Model I: firm only
def _srw(y: np.ndarray, k: int) -> float:
    # E[y_{t+k}] = y_{t+k-4}; for k<=4 that observation is in-sample.
    return float(y[len(y) - 4 + (k - 1)])


def _srw_drift(y: np.ndarray, k: int) -> float:
    d = y[4:] - y[:-4]
    return _srw(y, k) + float(np.mean(d)) if len(d) else _srw(y, k)


def _brown_rozeff(y: np.ndarray) -> np.ndarray | None:
    if len(y) < 16:
        return None
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = SARIMAX(y, order=(1, 0, 0), seasonal_order=(0, 1, 1, 4), trend="n").fit(disp=False)
            return np.asarray(res.forecast(4))
        except Exception:  # noqa: BLE001 - non-convergence on short/odd series
            return None


def firm_model(eps: pd.Series, years: int) -> dict:
    """Forecasts for the next 4 fiscal quarters from the last ``years`` years of EPS."""
    y = eps.dropna().iloc[-4 * years:].to_numpy(dtype=float)
    if len(y) < 8:
        return {}
    br = _brown_rozeff(y)
    out = {}
    for k in HORIZONS:
        parts = {"srw": _srw(y, k), "srw_drift": _srw_drift(y, k)}
        if br is not None and np.isfinite(br[k - 1]):
            parts["brown_rozeff"] = float(br[k - 1])
        out[k] = {"eps": float(np.mean(list(parts.values()))), **{f"eps_{n}": v for n, v in parts.items()}}
    return out


# ======================================================= Models II & III: panel
def _winsor(s: pd.Series, q=0.01) -> pd.Series:
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lo, hi)


def build_design(panel: pd.DataFrame, k: int, macro: pd.DataFrame | None) -> pd.DataFrame:
    """Rows (cik, cq) with features at cq and label ROE at cq+k."""
    p = panel[["cik", "cq", "roe", "ind_med", "cty_med"]].copy()
    p["roe"] = _winsor(p["roe"])
    idx = p.set_index(["cik", "cq"])["roe"]

    def look(shift):
        keys = list(zip(p["cik"], p["cq"] + shift))
        return idx.reindex(keys).to_numpy()

    d = p.copy()
    d["roe_seas"] = look(k - 4)  # same fiscal quarter last year, relative to the target
    d["y"] = look(k)
    d["loss"] = (d["roe"] < 0).astype(float)
    d["loss_x_roe"] = d["loss"] * d["roe"]
    ind = panel.groupby("cq")["ind_med"].first()
    cty = panel.groupby("cq")["cty_med"].first()
    d["ind_seas"] = (d["cq"] + (k - 4)).map(ind)
    d["cty_chg"] = d["cty_med"] - (d["cq"] - 4).map(cty)
    if macro is not None:
        d = d.merge(macro, left_on="cq", right_index=True, how="left")
    return d


FEATURES_II = ["roe", "roe_seas", "loss", "loss_x_roe", "ind_med", "ind_seas"]
FEATURES_III = FEATURES_II + ["cty_med", "cty_chg", "gdp_yoy", "term_spread", "rf_10y"]


def macro_by_quarter(quarters, report_lag_days: int = 45) -> pd.DataFrame:
    """Macro state as known ``report_lag_days`` after each quarter end (when its earnings were out)."""
    rows = {}
    for q in quarters:
        m = macro_state(q.end_time.normalize() + pd.Timedelta(days=report_lag_days))
        rows[q] = {"gdp_yoy": m.get("real_gdp_growth_yoy"), "term_spread": m.get("term_spread"),
                   "rf_10y": m.get("rf_10y")}
    return pd.DataFrame(rows).T.astype(float)


def panel_model(design: pd.DataFrame, features: list[str], train_end: pd.Period, k: int,
                years: int) -> tuple[np.ndarray, dict] | tuple[None, dict]:
    """OLS of ROE_{t+k} on features; trains only on labels realized by ``train_end``."""
    tr = design[(design["cq"] + k <= train_end) & (design["cq"] > train_end - 4 * years - k)]
    tr = tr.dropna(subset=features + ["y"])
    if len(tr) < 30 + 5 * len(features):
        return None, {"n_obs": len(tr)}
    X = np.column_stack([np.ones(len(tr)), tr[features].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(X, tr["y"].to_numpy(float), rcond=None)
    resid = tr["y"].to_numpy(float) - X @ beta
    r2 = 1 - resid.var() / tr["y"].var()
    return beta, {"n_obs": int(len(tr)), "n_firms": int(tr["cik"].nunique()), "r2": float(r2),
                  "y_lo": float(tr["y"].quantile(0.01)), "y_hi": float(tr["y"].quantile(0.99))}


def firm_features(firm_q: pd.DataFrame, panel: pd.DataFrame, k: int, macro_now: dict) -> dict:
    """Feature row for the firm's latest quarter (firm_q is calendar-indexed, with roe)."""
    t = firm_q.index[-1]
    roe = firm_q["roe"]
    ind = panel.groupby("cq")["ind_med"].first()
    cty = panel.groupby("cq")["cty_med"].first()
    f = {"roe": roe.iloc[-1], "roe_seas": roe.get(t + (k - 4), np.nan)}
    f["loss"] = float(f["roe"] < 0)
    f["loss_x_roe"] = f["loss"] * f["roe"]
    f["ind_med"] = ind.get(t, ind.iloc[-1] if len(ind) else np.nan)
    f["ind_seas"] = ind.get(t + (k - 4), np.nan)
    f["cty_med"] = cty.get(t, cty.iloc[-1] if len(cty) else np.nan)
    f["cty_chg"] = f["cty_med"] - cty.get(t - 4, np.nan)
    f["gdp_yoy"] = macro_now.get("real_gdp_growth_yoy")
    f["term_spread"] = macro_now.get("term_spread")
    f["rf_10y"] = macro_now.get("rf_10y")
    return f


def calendarize(firm_snap: pd.DataFrame) -> pd.DataFrame:
    """Firm quarterly NI/equity indexed by calendar quarter, with ROE on lagged equity."""
    d = firm_snap[["ni", "equity"]].dropna(subset=["ni"]).copy()
    d.index = [calendar_quarter(t) for t in d.index]
    d = d[~d.index.duplicated(keep="last")]
    full = pd.period_range(d.index.min(), d.index.max(), freq="Q")
    eq = firm_snap["equity"].dropna()
    eq.index = [calendar_quarter(t) for t in eq.index]
    eq = eq[~eq.index.duplicated(keep="last")].reindex(full)
    d = d.reindex(full)
    d["be_lag"] = eq.shift(1)
    d["roe"] = d["ni"] / d["be_lag"]
    d.loc[~(d["be_lag"] > 0), "roe"] = np.nan
    return d

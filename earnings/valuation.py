"""Per-share intrinsic value (target: stock price) under four information sets.

Models
------
1. Residual income model, RIM (Edwards-Bell-Ohlson; Ohlson 1995), implemented as
   in Gebhardt, Lee & Swaminathan (2001): explicit year-1 earnings, ROE fading
   linearly to a target ROE by year 12, then a no-growth terminal value.
2. Discounted cash flow (free cash flow to equity): FCFE_t = E_t - g_t * B_{t-1}
   (the value-driver identity; Penman 2013), five explicit years with growth
   fading to long-run nominal growth, then Gordon growth.
3. Own model - Empirical-Persistence RIM (EP-RIM). Combines the linear
   information dynamics of Ohlson (1995) and Dechow, Hutton & Sloan (1999) with
   the evidence that ROE mean-reverts toward an industry level (Fama & French
   2000; Nissim & Penman 2001) and that losses are less persistent than profits
   (Hayn 1995; Li & Mohanram 2014):
       ROE_t - ROE* = omega^(t-1) * (ROE_1 - ROE*)
   where omega is *estimated* (AR(1) of annual ROE deviations in the relevant
   information set, separately for loss firms), ROE* is the information set's
   long-run ROE, and book value grows by clean surplus. 30 explicit years, then
   a no-growth terminal value. The final "composite" target price is the
   median of RIM, DCF and EP-RIM.

Cost of equity = 10y Treasury + Blume-adjusted CAPM beta x Damodaran implied ERP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ValuationInputs:
    bvps0: float            # book value per share now
    eps_fy1: float          # sum of next 4 quarterly EPS forecasts
    eps_ttm: float          # trailing 4 quarters EPS
    payout: float           # dividend payout ratio (0..1)
    r: float                # cost of equity
    roe_target: float       # long-run ROE for the information set (annual)
    g_long: float           # long-run growth for the information set
    omega: float            # ROE persistence for the information set
    fcfe_ttm: float | None = None


def _clip(x, lo, hi):
    return float(min(max(x, lo), hi))


def _book_growth(roe: float, v: "ValuationInputs") -> float:
    """Clean-surplus book growth, capped at long-run growth (steady state).

    Retaining (1 - payout) of a high ROE would compound book value faster than
    the discount rate and make the sum explode; in steady state a firm cannot
    reinvest at above-normal rates forever, so earnings beyond what sustains
    growth at g_long are treated as paid out (Penman 2013, ch. 6).
    """
    g_cap = min(max(v.g_long, 0.0), v.r - 0.01)
    return min(roe * (1 - v.payout), g_cap)


def rim_gls(v: ValuationInputs, fade_years: int = 12) -> float:
    b, value = v.bvps0, v.bvps0
    if b <= 0:
        return np.nan
    roe1 = v.eps_fy1 / b
    for t in range(1, fade_years + 1):
        roe = roe1 if t <= 2 else roe1 + (v.roe_target - roe1) * (t - 2) / (fade_years - 2)
        value += (roe - v.r) * b / (1 + v.r) ** t
        b *= 1 + _book_growth(roe, v)
    value += (roe - v.r) * b / (v.r * (1 + v.r) ** fade_years)
    return float(value)


def dcf_fcfe(v: ValuationInputs, explicit: int = 5) -> float:
    # A growth-based DCF is not meaningful when earnings are negative (no growth
    # base); the RIM-type models handle loss firms through book value.
    if v.eps_fy1 <= 0 or v.eps_ttm <= 0 or v.bvps0 <= 0:
        return np.nan
    g_lr = _clip(v.g_long, 0.0, v.r - 0.01)
    g1 = _clip(v.eps_fy1 / v.eps_ttm - 1, -0.3, 0.3)
    e, b, value = v.eps_fy1, max(v.bvps0, 0.0), 0.0
    for t in range(1, explicit + 1):
        g = g1 + (g_lr - g1) * (t - 1) / (explicit - 1)
        if t > 1:
            e *= 1 + g
        fcfe = e - g * b  # reinvestment needed to grow book value at g
        value += fcfe / (1 + v.r) ** t
        b *= 1 + g
    fcfe_next = e * (1 + g_lr) - g_lr * b
    value += fcfe_next / (v.r - g_lr) / (1 + v.r) ** explicit
    return float(value)


def ep_rim(v: ValuationInputs, years: int = 30) -> float:
    b, value = v.bvps0, v.bvps0
    if b <= 0:
        return np.nan
    roe1 = v.eps_fy1 / b
    omega = _clip(v.omega, 0.0, 0.95)
    for t in range(1, years + 1):
        roe = v.roe_target + omega ** (t - 1) * (roe1 - v.roe_target)
        value += (roe - v.r) * b / (1 + v.r) ** t
        b *= 1 + _book_growth(roe, v)
    value += (roe - v.r) * b / (v.r * (1 + v.r) ** years)
    return float(value)


def value_all(v: ValuationInputs) -> dict:
    out = {"rim": rim_gls(v), "dcf": dcf_fcfe(v), "ep_rim": ep_rim(v)}
    vals = [x for x in out.values() if np.isfinite(x)]
    out["composite"] = float(np.median(vals)) if vals else np.nan
    return out


def roe_persistence(annual_roe: "pd.DataFrame", target: float) -> float:  # noqa: F821
    """AR(1) slope of (ROE_{y+1} - target) on (ROE_y - target); pooled over firms."""
    import pandas as pd  # local import keeps this module dependency-light

    a = annual_roe.dropna()
    if len(a) < 20:
        return 0.6  # literature-typical annual ROE persistence (Nissim & Penman 2001)
    x = (a["roe"] - target).to_numpy()
    y = (a["roe_next"] - target).to_numpy()
    w = float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 0 else 0.6
    return _clip(w, 0.0, 0.95)

"""Analyst coverage and consensus from Nasdaq's public analyst API (estimates by Zacks).

I/B/E/S (the research standard) is fee-based; this public source gives, per
fiscal quarter and fiscal year, the consensus EPS, high/low estimates, number
of estimates and recent revisions. It is a *snapshot at fetch time*, so the
pipeline fetches it at the scheduled run (23:59 ET) and stores it; stored
snapshots form a point-in-time history going forward.
"""

from __future__ import annotations

import pandas as pd

from .http import get_json


def earnings_forecast(ticker: str, max_age: float = 6 * 3600) -> dict:
    url = f"https://api.nasdaq.com/api/analyst/{ticker.replace('-', '.')}/earnings-forecast"
    try:
        d = get_json(url, max_age=max_age).get("data") or {}
    except Exception:  # noqa: BLE001
        return {"quarterly": [], "yearly": []}
    q = ((d.get("quarterlyForecast") or {}).get("rows")) or []
    y = ((d.get("yearlyForecast") or {}).get("rows")) or []
    return {"quarterly": q, "yearly": y}


def coverage(ticker: str | None) -> dict:
    """Number of analysts (max estimates over upcoming periods) and consensus fields."""
    if not ticker:
        return {"n_analysts": 0}
    ef = earnings_forecast(ticker)
    out = {"n_analysts": 0}
    for kind, rows in (("q", ef["quarterly"]), ("fy", ef["yearly"])):
        for i, r in enumerate(rows[:4] if kind == "q" else rows[:2], start=1):
            try:
                n = int(r.get("noOfEstimates") or 0)
            except (TypeError, ValueError):
                n = 0
            out["n_analysts"] = max(out["n_analysts"], n)
            p = f"consensus_{kind}{i}"
            out[f"{p}_period"] = r.get("fiscalEnd")
            out[f"{p}_eps"] = _f(r.get("consensusEPSForecast"))
            out[f"{p}_high"] = _f(r.get("highEPSForecast"))
            out[f"{p}_low"] = _f(r.get("lowEPSForecast"))
            out[f"{p}_n"] = n
            out[f"{p}_revisions_up"] = r.get("up")
            out[f"{p}_revisions_down"] = r.get("down")
    out["consensus_fetched_utc"] = str(pd.Timestamp.now(tz="UTC"))
    return out


def _f(x):
    try:
        return float(str(x).replace("$", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def earnings_calendar(day) -> list[dict]:
    """Firms reporting earnings on ``day`` (Nasdaq earnings calendar; data by Zacks).

    The press release date, not the 8-K filing date: firms may file the Item 2.02
    8-K up to four business days after the release. ``time`` is
    time-pre-market / time-after-hours / time-not-supplied.
    """
    d = pd.Timestamp(day).date()
    data = get_json(f"https://api.nasdaq.com/api/calendar/earnings?date={d}", max_age=3600).get("data") or {}
    return data.get("rows") or []

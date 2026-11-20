"""Cross-sectional panel (all SEC filers) by calendar quarter, from XBRL frames.

Quarterly ROE_it = NI_it / BE_i,t-1 is the scale-free earnings measure used in
the industry and country models (Fama & French 2000; Hou, van Dijk & Zhang 2012
use the same idea in levels). Fiscal Q4 is derived as annual - (Q1+Q2+Q3).

Point-in-time: frames carry the latest value per filer, without filing dates.
For a live run at ``as_of`` we drop filings accepted after ``as_of`` (from
EDGAR's current-filings feed). Panels for historical dates should use
``lag_days`` instead (a value is assumed known ``lag_days`` after period end;
the 10-Q deadline is 40-45 days, the 10-K deadline 60-90 days).
"""

from __future__ import annotations

import html
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from .http import get
from .sec import frame


def calendar_quarter(ts) -> pd.Period:
    """Map a fiscal period end to the calendar quarter it mostly covers.

    A quarter ending 2 Jan 2027 covers Oct-Dec 2026, so shift back two weeks first.
    """
    return (pd.Timestamp(ts) - pd.Timedelta(days=14)).to_period("Q")


def filings_after(as_of: pd.Timestamp, forms=("10-Q", "10-K", "10-Q/A", "10-K/A")) -> set[str]:
    """Accession numbers of periodic reports accepted after ``as_of`` (US/Eastern)."""
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("America/New_York")
    out = set()
    for form in forms:
        for start in range(0, 4000, 100):
            url = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent"
                   f"&type={form}&company=&dateb=&owner=include&start={start}&count=100&output=atom")
            xml = get(url, max_age=1800).decode("latin-1")
            entries = xml.split("<entry>")[1:]
            if not entries:
                break
            reached = False
            for e in entries:
                upd = pd.Timestamp(re.search(r"<updated>(.*?)</updated>", e).group(1))
                if upd <= as_of:
                    reached = True
                    continue
                m = re.search(r"AccNo:</b>\s*([\d-]+)", html.unescape(e))
                if m:
                    out.add(m.group(1))
            if reached:
                break
    return out


@lru_cache(maxsize=None)
def _q(tag, unit, period):
    return frame(tag, unit, period)


def quarterly_panel(first: pd.Period, last: pd.Period, exclude_accns: frozenset = frozenset(),
                    as_of: pd.Timestamp | None = None, lag_days: int | None = None) -> pd.DataFrame:
    """Long panel: cik, cq (calendar quarter), loc, country, ni, be_lag, roe."""
    rows = []
    for p in pd.period_range(first - 1, last, freq="Q"):
        tag = f"CY{p.year}Q{p.quarter}"
        ni = _q("NetIncomeLoss", "USD", tag)
        be = _q("StockholdersEquity", "USD", tag + "I")
        if p.quarter == 4:  # many filers only report the fiscal year; derive Q4
            ann = _q("NetIncomeLoss", "USD", f"CY{p.year}")
            parts = [_q("NetIncomeLoss", "USD", f"CY{p.year}Q{k}") for k in (1, 2, 3)]
            if len(ann):
                d = ann[["cik", "loc", "end", "val", "accn"]].copy()
                for k, pt in enumerate(parts):
                    d = d.merge(pt[["cik", "val"]].rename(columns={"val": f"q{k}"}), on="cik", how="inner")
                d["val"] = d["val"] - d[["q0", "q1", "q2"]].sum(axis=1)
                d = d[~d.cik.isin(ni.cik)][["cik", "loc", "end", "val", "accn"]]
                ni = pd.concat([ni, d], ignore_index=True)
        for df, name in ((ni, "ni"), (be, "be")):
            if df.empty:
                continue
            d = df.copy()
            if exclude_accns:
                d = d[~d["accn"].isin(exclude_accns)]
            if lag_days is not None and as_of is not None:
                d = d[d["end"] + pd.Timedelta(days=lag_days) <= as_of]
            rows.append(pd.DataFrame({"cik": d["cik"], "cq": p, "loc": d.get("loc"), "field": name,
                                      "val": d["val"].astype(float)}))
    long = pd.concat(rows, ignore_index=True).drop_duplicates(["cik", "cq", "field"], keep="first")
    wide = long.pivot_table(index=["cik", "cq"], columns="field", values="val", aggfunc="first").reset_index()
    locs = long.dropna(subset=["loc"]).drop_duplicates(["cik"], keep="last").set_index("cik")["loc"]
    wide["loc"] = wide["cik"].map(locs)
    wide["country"] = wide["loc"].fillna("US").str.slice(0, 2)
    wide = wide.sort_values(["cik", "cq"])
    prev_be = wide.groupby("cik")["be"].shift(1)
    consecutive = wide["cq"] == wide.groupby("cik")["cq"].shift(1).apply(lambda x: x + 1 if pd.notna(x) else x)
    wide["be_lag"] = prev_be.where(consecutive)
    wide["roe"] = wide["ni"] / wide["be_lag"]
    ok = (wide["be_lag"] > 0) & wide["roe"].between(-1.0, 1.0)
    wide.loc[~ok, "roe"] = np.nan
    return wide[wide["cq"] >= first].reset_index(drop=True)

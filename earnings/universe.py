"""Russell 3000 proxy from public SEC data.

FTSE Russell's constituent list is proprietary and iShares blocks automated
downloads of IWV holdings, so we rebuild the universe with Russell's own rule
(the 3,000 largest U.S. companies by market value), using the public float
from each filer's latest 10-K cover page (XBRL dei:EntityPublicFloat, measured
at the end of the second fiscal quarter). Exclusions follow Russell's ineligible
security types: royalty trusts, ETFs/commodity pools, funds.

Point-in-time: only 10-K filings accepted at or before the run cutoff are used.
"""

from __future__ import annotations

import re

import pandas as pd

from .sec import frame

_EXCLUDE = re.compile(
    r"(?i)royalty trust|\bETF\b|\bfund\b|portfolio|commodity|gold trust|silver trust|bitcoin|ether(?:eum)? trust|"
    r"\bshares trust\b|currencyshares|index trust|\bspdr\b|ishares|proshares|invesco db|grayscale")


def _is_us(profile, cik) -> bool:
    try:
        return profile(cik).country == "US"
    except Exception:  # noqa: BLE001
        return False


def russell3000_proxy(as_of: pd.Timestamp, exclude_accns=frozenset(), n: int = 3000) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of).tz_localize(None) if pd.Timestamp(as_of).tzinfo else pd.Timestamp(as_of)
    last = (as_of.normalize() - pd.Timedelta(days=1)).to_period("Q")
    parts = []
    # Float is measured mid fiscal year and filed with the 10-K up to ~a year later.
    for q in pd.period_range(last - 9, last, freq="Q"):
        f = frame("EntityPublicFloat", "USD", f"CY{q.year}Q{q.quarter}I", taxonomy="dei")
        if len(f):
            parts.append(f)
    df = pd.concat(parts, ignore_index=True)
    df = df[~df["accn"].isin(exclude_accns)]
    df = df.sort_values("end").drop_duplicates("cik", keep="last")
    df = df[(df["val"] > 0) & ~df["entityName"].str.contains(_EXCLUDE)]

    # Location: frames give "US-xx"; "-" means unknown -> use the business address.
    unknown = df["loc"].fillna("-").eq("-")
    us = df["loc"].fillna("").str.startswith("US")
    if unknown.any():
        from .sec import profile
        cand = df[unknown].nlargest(400, "val")  # only those large enough to matter
        us_extra = {c for c in cand["cik"] if _is_us(profile, c)}
        us |= df["cik"].isin(us_extra)
    df = df[us]

    # Unit errors: some filers tag float in thousands-as-dollars x1000. A float above
    # 300x book equity is implausible; rescale by 1/1000.
    be = []
    for q in pd.period_range(last - 3, last, freq="Q"):
        f = frame("StockholdersEquity", "USD", f"CY{q.year}Q{q.quarter}I")
        if len(f):
            be.append(f[["cik", "end", "val"]])
    if be:
        bev = pd.concat(be).sort_values("end").drop_duplicates("cik", keep="last").set_index("cik")["val"]
        for _ in range(3):  # errors seen in the wild are x1,000 and x1,000,000
            ratio = df["val"] / df["cik"].map(bev)
            fix = (ratio > 300) & (df["val"] > 5e10)
            if not fix.any():
                break
            df.loc[fix, "val"] = df.loc[fix, "val"] / 1000
    df = df.sort_values("val", ascending=False).head(n).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df[["cik", "entityName", "loc", "end", "val", "rank"]].rename(
        columns={"entityName": "name", "val": "public_float", "end": "float_date"})

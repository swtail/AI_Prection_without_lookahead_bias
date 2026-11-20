"""Public, literature-standard non-SEC data, each with explicit point-in-time availability.

* U.S. Treasury daily par yield curve (risk-free rate)
    https://home.treasury.gov/resource-center/data-chart-center/interest-rates
* Kenneth R. French Data Library - Fama-French 48 industry definitions (SIC -> FF48)
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
* Aswath Damodaran (NYU Stern) - implied equity risk premium, S&P 500, by year-end
    https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/implpr.html
* Federal Reserve Bank of Philadelphia Real-Time Data Set for Macroeconomists
  (Croushore & Stark 2001) - real GDP and GDP deflator as they were known at each vintage
    https://www.philadelphiafed.org/surveys-and-data/real-time-data-research
* Nasdaq public quote API - daily closing prices (no free official source exists;
  CRSP is the research standard but is fee-based)
"""

from __future__ import annotations

import io
import re
import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd

from .http import get, get_json

PHIL = "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/real-time-data/data-files/xlsx/"


# ----------------------------------------------------------------- risk-free
@lru_cache(maxsize=None)
def _treasury_year(year: int) -> pd.DataFrame:
    url = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
           f"daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
           f"&field_tdr_date_value={year}&page&_format=csv")
    this_year = year >= pd.Timestamp.today().year
    df = pd.read_csv(io.BytesIO(get(url, max_age=3600 if this_year else None)))
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


def treasury_yield(as_of, tenor: str = "10 Yr") -> float:
    """Latest published par yield (decimal) on or before ``as_of``."""
    as_of = pd.Timestamp(as_of)
    for year in (as_of.year, as_of.year - 1):
        df = _treasury_year(year)
        s = df.loc[df.index <= as_of.normalize(), tenor].dropna()
        if len(s):
            return float(s.iloc[-1]) / 100
    raise LookupError("no Treasury yield found")


# ------------------------------------------------------------------- FF48
@lru_cache(maxsize=1)
def ff48_table() -> pd.DataFrame:
    raw = get("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes48.zip", max_age=None)
    text = zipfile.ZipFile(io.BytesIO(raw)).read(
        zipfile.ZipFile(io.BytesIO(raw)).namelist()[0]).decode("latin-1")
    rows, cur = [], None
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)\s+(\S+)\s+(.*\S)\s*$", line)
        r = re.match(r"^\s+(\d{4})-(\d{4})", line)
        if r and cur:
            rows.append((int(r.group(1)), int(r.group(2)), *cur))
        elif m:
            cur = (int(m.group(1)), m.group(2), m.group(3))
    return pd.DataFrame(rows, columns=["sic_lo", "sic_hi", "ff48", "ff48_short", "ff48_name"])


def ff48(sic: int | None) -> tuple[int, str, str]:
    """Fama-French 48 industry for a SIC code; 48 'Other' when unmapped."""
    if sic is not None:
        t = ff48_table()
        hit = t[(t.sic_lo <= sic) & (t.sic_hi >= sic)]
        if len(hit):
            r = hit.iloc[0]
            return int(r.ff48), r.ff48_short, r.ff48_name
    return 48, "Other", "Almost Nothing"


def ff48_sic_codes(industry: int) -> list[int]:
    t = ff48_table()
    t = t[t.ff48 == industry]
    return sorted({s for lo, hi in zip(t.sic_lo, t.sic_hi) for s in range(lo, hi + 1)})


# ------------------------------------------------------------- Damodaran ERP
@lru_cache(maxsize=1)
def _damodaran_erp() -> pd.Series:
    raw = get("https://pages.stern.nyu.edu/~adamodar/pc/datasets/histimpl.xls", max_age=30 * 86400)
    x = pd.read_excel(io.BytesIO(raw), header=None)
    hdr = x.index[x.iloc[:, 0].astype(str).str.strip().eq("Year")][0]
    df = x.iloc[hdr + 1:].copy()
    df.columns = [str(c).strip() for c in x.iloc[hdr]]
    df = df[pd.to_numeric(df["Year"], errors="coerce").notna()]
    s = pd.to_numeric(df["Implied ERP (FCFE)"], errors="coerce")
    s.index = df["Year"].astype(int)
    return s.dropna()


def equity_risk_premium(as_of) -> float:
    """Damodaran implied ERP for the last year-end published by ``as_of``.

    The year-Y estimate is published in the first days of January Y+1; we
    treat it as available from 15 January Y+1.
    """
    as_of = pd.Timestamp(as_of)
    s = _damodaran_erp()
    usable = s[[pd.Timestamp(year=y + 1, month=1, day=15) <= as_of for y in s.index]]
    return float(usable.iloc[-1]) if len(usable) else 0.05


# --------------------------------------------- Philadelphia Fed real-time data
@lru_cache(maxsize=None)
def _rtdsm(name: str) -> pd.DataFrame:
    raw = get(PHIL + name, max_age=7 * 86400)
    df = pd.read_excel(io.BytesIO(raw), engine="calamine")
    df["DATE"] = pd.PeriodIndex(df["DATE"].str.replace(":", ""), freq="Q").to_timestamp(how="end").normalize()
    return df.set_index("DATE")


def _vintage_date(col: str) -> pd.Timestamp:
    """RTDSM vintages are dated in the middle of the quarter's second month."""
    m = re.search(r"(\d{2})Q(\d)$", col)
    yy, q = int(m.group(1)), int(m.group(2))
    year = 1900 + yy if yy >= 65 else 2000 + yy
    return pd.Timestamp(year=year, month=3 * q - 1, day=15)


def real_time_series(as_of, variable: str = "ROUTPUT") -> pd.Series:
    """Quarterly level series as it was published at the latest vintage <= ``as_of``."""
    df = _rtdsm(f"{variable}QvQd.xlsx")
    cols = [c for c in df.columns if c.startswith(variable) and _vintage_date(c) <= pd.Timestamp(as_of)]
    if not cols:
        raise LookupError("no vintage available")
    return pd.to_numeric(df[cols[-1]], errors="coerce").dropna()


def macro_state(as_of) -> dict:
    """US macro variables exactly as known at ``as_of`` (annualised decimals)."""
    out = {}
    try:
        g = real_time_series(as_of, "ROUTPUT")
        out["real_gdp_growth_yoy"] = float(g.iloc[-1] / g.iloc[-5] - 1)
        out["real_gdp_growth_qoq_ann"] = float((g.iloc[-1] / g.iloc[-2]) ** 4 - 1)
        out["real_gdp_growth_10y"] = float((g.iloc[-1] / g.iloc[-41]) ** (1 / 10) - 1)
        out["gdp_last_quarter"] = str(g.index[-1].date())
    except Exception as e:  # noqa: BLE001 - macro data is optional; record why it is missing
        out["gdp_error"] = str(e)
    try:
        p = real_time_series(as_of, "P")
        out["inflation_yoy"] = float(p.iloc[-1] / p.iloc[-5] - 1)
        out["inflation_10y"] = float((p.iloc[-1] / p.iloc[-41]) ** (1 / 10) - 1)
    except Exception as e:  # noqa: BLE001
        out["inflation_error"] = str(e)
    try:
        out["rf_10y"] = treasury_yield(as_of, "10 Yr")
        out["rf_3m"] = treasury_yield(as_of, "3 Mo")
        out["term_spread"] = out["rf_10y"] - out["rf_3m"]
    except Exception as e:  # noqa: BLE001
        out["rf_error"] = str(e)
    out["erp"] = equity_risk_premium(as_of)
    if "real_gdp_growth_10y" in out and "inflation_10y" in out:
        out["nominal_growth_lr"] = (1 + out["real_gdp_growth_10y"]) * (1 + out["inflation_10y"]) - 1
    return out


# ------------------------------------------------------------------ prices
def daily_prices(ticker: str, start="2010-01-01", end=None, asset="stocks") -> pd.Series:
    """Unadjusted daily closes (Nasdaq public API). Splits are NOT adjusted."""
    end = pd.Timestamp(end or pd.Timestamp.today()).date()
    url = (f"https://api.nasdaq.com/api/quote/{ticker.replace('-', '.')}/historical?assetclass={asset}"
           f"&fromdate={pd.Timestamp(start).date()}&todate={end}&limit=9999")
    d = get_json(url, max_age=3600)
    rows = ((d.get("data") or {}).get("tradesTable") or {}).get("rows") or []
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(r["date"]): float(str(r["close"]).replace("$", "").replace(",", ""))
                   for r in rows})
    return s.sort_index()


def price_as_of(ticker: str, as_of) -> tuple[float, pd.Timestamp] | tuple[None, None]:
    s = daily_prices(ticker, start=pd.Timestamp(as_of) - pd.Timedelta(days=14), end=as_of)
    s = s[s.index <= pd.Timestamp(as_of)]
    return (float(s.iloc[-1]), s.index[-1]) if len(s) else (None, None)


def capm_beta(ticker: str, as_of, years: int = 5) -> float | None:
    """Monthly-return beta vs. SPY over the previous ``years`` (min. 24 months).

    Uses price returns only; split days show up as outliers, so months with
    |return| > 60% are dropped.
    """
    as_of = pd.Timestamp(as_of)
    start = as_of - pd.DateOffset(years=years) - pd.Timedelta(days=40)
    p = daily_prices(ticker, start, as_of)
    m = daily_prices("SPY", start, as_of, asset="etf")
    if p.empty or m.empty:
        return None
    r = pd.concat([p.resample("ME").last().pct_change(), m.resample("ME").last().pct_change()],
                  axis=1, keys=["i", "m"]).dropna()
    r = r[r["i"].abs() < 0.6]
    if len(r) < 24:
        return None
    return float(np.cov(r["i"], r["m"])[0, 1] / r["m"].var())

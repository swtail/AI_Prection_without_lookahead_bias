"""SEC EDGAR: point-in-time fundamentals, earnings-announcement detection, peers.

Sources (all public, free):
* XBRL company facts   https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
  Every fact carries ``filed`` - the date it became public - so restatements
  in later filings become later vintages instead of overwriting history.
* XBRL frames          https://data.sec.gov/api/xbrl/frames/...  (cross-section per calendar quarter)
* Submissions          https://data.sec.gov/submissions/CIK##########.json (SIC, address, filings)
* Current/daily 8-K feeds; Item 2.02 = "Results of Operations and Financial Condition".
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .http import get, get_json

# --------------------------------------------------------------------------- tags
# First available tag wins, in this order (common US-GAAP alternatives).
TAGS = {
    "eps": (["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted",
             "EarningsPerShareBasic"], "USD/shares", "duration"),
    "ni": (["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
           "USD", "duration"),
    "revenue": (["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                 "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
                "USD", "duration"),
    "cfo": (["NetCashProvidedByUsedInOperatingActivities",
             "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"], "USD", "duration"),
    "capex": (["PaymentsToAcquirePropertyPlantAndEquipment",
               "PaymentsToAcquireProductiveAssets"], "USD", "duration"),
    "dps": (["CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid"],
            "USD/shares", "duration"),
    "shares_diluted": (["WeightedAverageNumberOfDilutedSharesOutstanding"], "shares", "average"),
    "equity": (["StockholdersEquity",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
               "USD", "instant"),
    "assets": (["Assets"], "USD", "instant"),
    "debt": (["LongTermDebt", "LongTermDebtNoncurrent"], "USD", "instant"),
    "cash": (["CashAndCashEquivalentsAtCarryingValue"], "USD", "instant"),
}
_FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT", "10-QT"}


def cik10(cik) -> str:
    return f"{int(cik):010d}"


# ------------------------------------------------------------------ raw fetchers
def company_facts(cik, max_age=6 * 3600) -> dict:
    return get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10(cik)}.json", max_age=max_age)


def submissions(cik, max_age=6 * 3600) -> dict:
    return get_json(f"https://data.sec.gov/submissions/CIK{cik10(cik)}.json", max_age=max_age)


def ticker_map(max_age=86400) -> pd.DataFrame:
    d = get_json("https://www.sec.gov/files/company_tickers.json", max_age=max_age)
    df = pd.DataFrame(d.values()).rename(columns={"cik_str": "cik", "title": "name"})
    return df


def frame(tag: str, unit: str, period: str, taxonomy="us-gaap") -> pd.DataFrame:
    """Cross-section for one calendar period, e.g. period='CY2024Q2' or 'CY2024Q2I' (instant).

    Frames hold the *latest* value per company, so for historical backtests the
    caller must apply a conservative availability lag (see ``industry.py``).
    """
    closed = pd.Timestamp.today() - pd.Timestamp(_period_end(period)) > pd.Timedelta(days=400)
    try:
        d = get_json(f"https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json",
                     max_age=None if closed else 86400)
    except FileNotFoundError:
        return pd.DataFrame(columns=["cik", "loc", "end", "val"])
    df = pd.DataFrame(d["data"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def _period_end(period: str) -> str:
    m = re.match(r"CY(\d{4})(Q(\d))?", period)
    y, q = int(m.group(1)), int(m.group(3) or 4)
    return f"{y}-{3 * q:02d}-28"


# --------------------------------------------------- company facts -> PIT records
def _entries(facts: dict, tags: list[str], unit: str) -> pd.DataFrame:
    """All facts for any of ``tags``; where tags overlap, the earlier tag wins.

    Firms switch tags over time (e.g. ``Revenues`` -> ``RevenueFromContract...``
    after ASC 606), so the union is needed to get a continuous history.
    """
    parts = []
    g = facts.get("facts", {}).get("us-gaap", {})
    for prio, t in enumerate(tags):
        if t in g and unit in g[t]["units"]:
            df = pd.DataFrame(g[t]["units"][unit])
            df = df[df["form"].isin(_FORMS)]
            if not df.empty:
                parts.append(df.assign(_prio=prio))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True).sort_values("_prio", kind="stable")
    keys = [c for c in ("start", "end", "filed") if c in df.columns]
    return df.drop_duplicates(keys, keep="first").drop(columns="_prio")


def quarterly_records(facts: dict) -> pd.DataFrame:
    """Point-in-time records (event_time, available_time, field, value) of fiscal-quarter data.

    Duration items: 3-month values are taken directly; otherwise a quarter is the
    difference of two year-to-date values with the same fiscal-year start
    (6M-3M, 9M-6M, FY-9M), using the vintage of the earlier YTD known when the
    later one was filed. Fiscal Q4 therefore appears when the 10-K is filed.
    """
    out = []
    for field, (tags, unit, kind) in TAGS.items():
        df = _entries(facts, tags, unit)
        if df.empty:
            continue
        df = df.copy()
        df["end"] = pd.to_datetime(df["end"])
        df["filed"] = pd.to_datetime(df["filed"])
        if kind == "instant":
            out.append(pd.DataFrame({"event_time": df["end"], "available_time": df["filed"],
                                     "field": field, "value": df["val"].astype(float)}))
            continue
        if "start" not in df.columns:
            continue
        df = df.dropna(subset=["start"])
        df["start"] = pd.to_datetime(df["start"])
        if kind == "average":
            # Averages (e.g. weighted shares) cannot be differenced: use the 3-month
            # figure when reported, otherwise the shortest period ending on that date.
            df["days"] = (df["end"] - df["start"]).dt.days
            d = df[df["days"] >= 80].sort_values(["end", "filed", "days"])
            d = d.drop_duplicates(["end", "filed"], keep="first")
            out.append(pd.DataFrame({"event_time": d["end"], "available_time": d["filed"],
                                     "field": field, "value": d["val"].astype(float)}))
            continue
        df["days"] = (df["end"] - df["start"]).dt.days
        q = df[df["days"].between(80, 100)]
        rows = [pd.DataFrame({"event_time": q["end"], "available_time": q["filed"],
                              "field": field, "value": q["val"].astype(float)})]
        ytd = df[df["days"] > 100].sort_values("filed")
        by_start = {s: g for s, g in df[df["days"] >= 80].groupby("start")}
        derived = []
        for r in ytd.itertuples():
            g = by_start.get(r.start)
            prev = g[(g["end"] < r.end) & ((r.end - g["end"]).dt.days.between(80, 100))
                     & (g["filed"] <= r.filed)]
            if prev.empty:
                continue
            p = prev.sort_values("filed").iloc[-1]
            derived.append((r.end, r.filed, field, float(r.val) - float(p["val"])))
        if derived:
            rows.append(pd.DataFrame(derived, columns=["event_time", "available_time", "field", "value"]))
        rec = pd.concat(rows, ignore_index=True)
        # A direct 3-month figure beats a derived one filed the same day.
        rec = rec.drop_duplicates(["event_time", "available_time", "field"], keep="first")
        out.append(rec)
    if not out:
        return pd.DataFrame(columns=["event_time", "available_time", "field", "value"])
    rec = pd.concat(out, ignore_index=True)
    rec = rec[rec["available_time"] >= rec["event_time"]]
    return rec.sort_values(["available_time", "event_time"]).reset_index(drop=True)


# ------------------------------------------------------------------ firm profile
@dataclass
class FirmProfile:
    cik: int
    name: str
    ticker: str | None
    sic: int | None
    sic_description: str
    country: str
    state: str | None
    fiscal_year_end: str | None
    exchanges: list


def profile(cik) -> FirmProfile:
    s = submissions(cik, max_age=7 * 86400)
    biz = s.get("addresses", {}).get("business", {}) or {}
    foreign = bool(biz.get("isForeignLocation"))
    country = (biz.get("countryCode") or biz.get("stateOrCountry") or "US") if foreign else "US"
    sic = int(s["sic"]) if str(s.get("sic") or "").isdigit() else None
    tickers = s.get("tickers") or []
    return FirmProfile(int(cik), s.get("name", ""), tickers[0] if tickers else None, sic,
                       s.get("sicDescription", ""), country,
                       None if foreign else biz.get("stateOrCountry"),
                       s.get("fiscalYearEnd"), s.get("exchanges") or [])


def peers_by_sic(sic: int, max_pages: int = 10) -> list[int]:
    """CIKs registered under a SIC code (EDGAR company browse)."""
    ciks = []
    for page in range(max_pages):
        url = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
               f"&SIC={sic:04d}&type=10-Q&dateb=&owner=include&count=100&start={100 * page}&output=atom")
        txt = get(url, max_age=30 * 86400).decode("latin-1")
        found = re.findall(r"<cik>(\d+)</cik>", txt) or re.findall(r"CIK=(\d{10})", txt)
        ciks += [int(c) for c in found]
        if len(found) < 100:
            break
    return sorted(set(ciks))


# ------------------------------------------------------- earnings announcements
@dataclass
class Announcement:
    cik: int
    name: str
    accession: str | None          # 8-K Item 2.02 accepted by the cutoff, if any
    accepted: pd.Timestamp          # announcement time (calendar) or 8-K acceptance time
    index_url: str | None
    source: str = "8-K"             # "calendar" when the date comes from the earnings calendar
    ticker: str | None = None
    calendar: dict | None = None    # raw earnings-calendar row (timing, street EPS, forecast)


def announcements_on(day: date) -> list[Announcement]:
    """8-K filings with Item 2.02 accepted on ``day`` (US/Eastern)."""
    day = pd.Timestamp(day).date()
    out = []
    if (pd.Timestamp.now(tz="America/New_York").date() - day).days <= 1:
        out = _from_current_feed(day)
    if not out:
        out = _from_daily_index(day)
    seen, uniq = set(), []
    for a in out:
        if a.accession not in seen:
            seen.add(a.accession)
            uniq.append(a)
    return uniq


def _from_current_feed(day: date) -> list[Announcement]:
    out = []
    for start in range(0, 2000, 100):
        url = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb="
               f"&owner=include&start={start}&count=100&output=atom")
        xml = get(url, max_age=600).decode("latin-1")
        entries = xml.split("<entry>")[1:]
        if not entries:
            break
        older = False
        for e in entries:
            updated = pd.Timestamp(re.search(r"<updated>(.*?)</updated>", e).group(1))
            if updated.tz_convert("America/New_York").date() < day:
                older = True
                continue
            if updated.tz_convert("America/New_York").date() != day:
                continue
            summ = html.unescape(re.search(r"<summary[^>]*>(.*?)</summary>", e, re.S).group(1))
            if "Item 2.02" not in summ:
                continue
            title = html.unescape(re.search(r"<title>(.*?)</title>", e).group(1))
            m = re.match(r"8-K(?:/A)? - (.*) \((\d{10})\) \((\w+)\)", title)
            if not m or m.group(3) != "Filer":
                continue
            acc = re.search(r"AccNo:</b>\s*([\d-]+)", summ).group(1)
            link = re.search(r'href="([^"]+)"', e).group(1)
            out.append(Announcement(int(m.group(2)), m.group(1), acc, updated, link))
        if older:
            break
    return out


def _from_daily_index(day: date) -> list[Announcement]:
    q = (day.month - 1) // 3 + 1
    url = f"https://www.sec.gov/Archives/edgar/daily-index/{day.year}/QTR{q}/form.{day:%Y%m%d}.idx"
    try:
        txt = get(url, max_age=3600).decode("latin-1")
    except FileNotFoundError:
        return []
    out = []
    for line in txt.splitlines():
        if not line.startswith("8-K "):
            continue
        parts = re.split(r"\s{2,}", line.strip())
        cik, path = int(parts[-3]), parts[-1]
        acc = path.rsplit("/", 1)[-1].replace(".txt", "")
        sub = submissions(cik, max_age=3600)["filings"]["recent"]
        for i, a in enumerate(sub["accessionNumber"]):
            if a == acc and "2.02" in (sub["items"][i] or ""):
                accepted = pd.Timestamp(sub["acceptanceDateTime"][i])
                link = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{acc}-index.htm"
                out.append(Announcement(cik, parts[1], acc, accepted, link))
    return out


# --------------------------------------------- press-release (EX-99.1) parsing
_MONEY = r"\(?\$?\s*(-?\d{1,4}(?:\.\d{1,4}))\)?"


@dataclass
class ReleaseNumbers:
    eps_diluted: float | None
    period_end: pd.Timestamp | None
    snippet: str
    exhibit_url: str | None


def press_release_numbers(ann: Announcement) -> ReleaseNumbers:
    """Best-effort extraction of GAAP diluted EPS and quarter-end date from EX-99.1.

    Press releases are unstructured; the value is flagged *preliminary* and is
    replaced by the XBRL 10-Q/10-K value once that is filed.
    """
    if not ann.index_url:
        return ReleaseNumbers(None, None, "", None)
    idx = get(ann.index_url, max_age=None).decode("latin-1")
    ex = None
    for row in re.findall(r"<tr.*?</tr>", idx, re.S):
        if re.search(r">\s*EX-99(\.1|\.01)?\s*<", row):
            m = re.search(r'href="([^"]+\.html?)"', row)
            if m:
                ex = "https://www.sec.gov" + m.group(1).replace("/ix?doc=", "")
                break
    if ex is None:
        return ReleaseNumbers(None, None, "", None)
    text = _html_text(get(ex, max_age=None).decode("latin-1", errors="ignore"))
    eps, snippet = _find_eps(text)
    return ReleaseNumbers(eps, _find_period_end(text, ann.accepted), snippet, ex)


def _html_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)</(td|th)>", " | ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</(p|div|tr)>", "\n", raw)
    txt = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    return re.sub(r"[ \t\xa0]+", " ", txt)


_NONGAAP = re.compile(r"(?i)non-gaap|adjusted|excluding|core |operating earnings|pro forma"
                      r"|outlook|guidance|expects?|forecast|full[- ]year|fiscal (?:year )?20\d\d (?:eps|outlook)|\d to \$")


def _signed(num: str, ctx: str) -> float:
    neg = "(" in num or num.strip().startswith("-")
    val = abs(float(re.sub(r"[^\d.]", "", num)))
    if not neg and (re.search(r"(?i)loss per (?:basic|diluted|common)?\s*share", ctx)
                    or (re.search(r"(?i)\bnet loss\b", ctx) and not re.search(r"(?i)net income", ctx))):
        neg = True  # some releases print a loss without parentheses after "net loss of"
    return -val if neg else val


def _find_eps(text: str) -> tuple[float | None, str]:
    """GAAP diluted EPS for the most recent quarter, or None. Narrative first, then tables."""
    num = r"(\$?\s*\(?\s*-?\$?\s*\d{1,4}\.\d{2,4}\s*\)?)"
    sentences = [
        rf"(?i)(?<!non-)gaap diluted (?:earnings|net income|eps)[^.$]{{0,60}}?(?:of|was|were|totaled)?\s*{num}",
        rf"(?i)diluted (?:earnings|net income \(loss\)|net income|net loss|loss|eps)(?: per (?:common )?share)?"
        rf"[^.$|]{{0,40}}?(?:of|was|were|totaled|:)\s*{num}",
        rf"(?i){num}\s*per (?:basic and )?diluted (?:common )?share",
    ]
    for p in sentences:
        for m in re.finditer(p, text):
            ctx = text[max(0, m.start() - 120): m.end() + 20]
            if _NONGAAP.search(ctx):
                continue
            return _signed(m.group(1), ctx), " ".join(ctx.split())
    # Table rows: label cell mentions "diluted"; first numeric cell is the latest quarter.
    for line in text.split("\n"):
        cells = [c.strip() for c in line.split("|")]
        label = cells[0] if cells else ""
        if not re.search(r"(?i)diluted", label) or re.search(r"(?i)weighted|shares outstanding|average", label):
            continue
        if _NONGAAP.search(label):
            continue
        context_before = text[max(0, text.find(line) - 300): text.find(line)]
        if not re.search(r"(?i)per (?:common )?share|per share|eps", label + context_before):
            continue
        if _NONGAAP.search(line):
            continue
        for cell in cells[1:]:
            if cell in ("", "$", "%"):
                continue
            if cell in ("—", "–", "-", "--"):
                return 0.0, " ".join(line.split())[:200]  # a dash in an EPS column means zero
            m = re.search(r"\(?\s*-?\$?\s*\(?\d{1,4}\.\d{2,4}\s*\)?", cell)
            if m:
                return _signed(m.group(0), label), " ".join(line.split())[:200]
            break
    return None, ""


def _find_period_end(text: str, accepted: pd.Timestamp) -> pd.Timestamp | None:
    m = re.search(r"(?i)(?:quarter|three months|period) ended\s+([A-Z][a-z]+\.? \d{1,2},? \d{4})", text)
    if m:
        try:
            ts = pd.Timestamp(m.group(1).replace(".", ""))
            if pd.Timedelta(0) < accepted.tz_localize(None) - ts < pd.Timedelta(days=120):
                return ts
        except ValueError:
            pass
    return None

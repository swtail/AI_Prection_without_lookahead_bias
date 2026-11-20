"""Daily run: for every firm that announced earnings on ``day``, forecast and value it
using only information public at ``day`` 19:00 US/Eastern.

    python -m earnings.pipeline --date 2026-11-19 --out outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from nolookahead import PointInTimeStore

from . import forecast as F
from . import public, sec
from .panel import calendar_quarter, filings_after, quarterly_panel
from .state import StateStore
from .valuation import ValuationInputs, roe_persistence, value_all

log = logging.getLogger("earnings")
WINDOWS = (5, 10)
CUTOFF = "23:59"  # scheduled run time, US/Eastern; EDGAR stops accepting filings at 22:00
INFO_SETS = ("firm", "industry", "country", "all")


# ------------------------------------------------------------------ helpers
def _blume(beta):
    return 0.67 * beta + 0.33 if beta is not None and np.isfinite(beta) else 1.0


def _annual_roe(panel: pd.DataFrame) -> pd.DataFrame:
    """Annual ROE per firm at each Q4 = sum of 4 quarterly NI / equity 4 quarters earlier."""
    p = panel.sort_values(["cik", "cq"]).copy()
    g = p.groupby("cik")
    p["ni4"] = g["ni"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    p["be4"] = g["be"].shift(4)
    p = p[p["cq"].map(lambda q: q.quarter == 4)]
    p["roe"] = (p["ni4"] / p["be4"]).where(p["be4"] > 0)
    p = p[p["roe"].between(-1, 1)]
    p["roe_next"] = p.groupby("cik")["roe"].shift(-1)
    p["be_growth"] = p.groupby("cik")["be"].pct_change()
    return p


def _info_params(annual: pd.DataFrame, years: int = 10) -> dict:
    if annual.empty:
        return {"roe_target": np.nan, "omega": 0.6, "be_growth": np.nan, "n_firms": 0}
    last = annual["cq"].max()
    a = annual[annual["cq"] > last - 4 * years]
    latest = a[a["cq"] == last]
    # Long-run ROE target: median over profitable firms (Gebhardt, Lee & Swaminathan 2001).
    prof = a[a["roe"] > 0]
    target = float(prof.groupby("cq")["roe"].median().tail(years).median()) if len(prof) else np.nan
    return {
        "roe_target": target,
        "roe_latest_median": float(latest["roe"].median()) if len(latest) else np.nan,
        "omega": roe_persistence(a, target),
        "be_growth": float(a.groupby("cq")["be_growth"].median().tail(years).median()),
        "n_firms": int(latest["cik"].nunique()),
    }


def _plausible(eps_new: float, hist: pd.Series) -> bool:
    h = hist.dropna().iloc[-8:]
    if len(h) < 4:
        return True
    med, mad = h.median(), (h - h.median()).abs().median()
    return abs(eps_new - med) <= max(6 * mad, 2 * abs(med), 0.25)


# ---------------------------------------------------------------- per firm
class Context:
    """Data shared by all firms in one run (built once)."""

    def __init__(self, as_of: pd.Timestamp, state: StateStore | None = None):
        self.as_of = as_of
        self.as_of_date = as_of.tz_localize(None).normalize()
        self.state = state
        self.reuse_log: list[str] = []
        log.info("macro state as of %s", as_of)
        self.macro = public.macro_state(self.as_of_date)
        excl = frozenset(filings_after(as_of))
        last_q = (self.as_of_date - pd.Timedelta(days=1)).to_period("Q") - 1
        first_q = last_q - 4 * max(WINDOWS) - 8
        self.panel = self._panel(first_q, last_q, excl)
        counts = self.panel.dropna(subset=["roe"]).groupby("cq").size()
        self.train_end = counts[counts >= 0.5 * counts.max()].index.max()
        self.macro_q = self._macro_q(sorted(self.panel["cq"].unique()))
        self.annual = _annual_roe(self.panel)
        self._peers: dict[int, set] = {}
        self.designs: dict = {}

    def _state_usable(self, meta: dict) -> bool:
        # Never reuse state saved after this run's cutoff (would leak later restatements).
        return bool(meta) and pd.Timestamp(meta.get("as_of")) <= self.as_of

    def _panel(self, first_q, last_q, excl) -> pd.DataFrame:
        old, meta = self.state.load_panel() if self.state else (None, {})
        if old is not None and self._state_usable(meta) and old["cq"].min() <= first_q:
            keep_before = last_q - 1  # the two newest quarters are still being filed: refresh
            fresh = quarterly_panel(keep_before, last_q, excl)
            panel = pd.concat([old[old["cq"] < keep_before], fresh], ignore_index=True)
            panel = panel[panel["cq"] >= first_q]
            self.reuse_log.append(f"panel: reused {old['cq'].min()}..{keep_before - 1} from state, "
                                  f"refreshed {keep_before}..{last_q}")
        else:
            log.info("building panel from SEC frames (first run or state not usable)")
            panel = quarterly_panel(first_q, last_q, excl)
            self.reuse_log.append(f"panel: built {first_q}..{last_q} from SEC frames")
        if self.state:
            self.state.save_panel(panel, {"as_of": str(self.as_of), "first_q": str(first_q),
                                          "last_q": str(last_q), "rows": len(panel)})
        return panel

    def _macro_q(self, quarters) -> pd.DataFrame:
        old = self.state.load_macro_q() if self.state else None
        have = set(old.index) if old is not None else set()
        cutoff = self.as_of_date - pd.Timedelta(days=45)
        # A quarter's macro row is final once its "known at" date (end + 45d) has passed.
        need = [q for q in quarters if q not in have or q.end_time.normalize() > cutoff - pd.Timedelta(days=1)]
        new = F.macro_by_quarter(need) if need else pd.DataFrame()
        parts = [df for df in (old, new) if df is not None and len(df)]
        merged = pd.concat(parts) if parts else pd.DataFrame()
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        self.reuse_log.append(f"macro_by_quarter: reused {len(have) - len([q for q in need if q in have])}, "
                              f"computed {len(need)}")
        if self.state:
            self.state.save_macro_q(merged)
        return merged.reindex(quarters)

    def peers(self, industry: int) -> set:
        if industry not in self._peers:
            saved = self.state.read_json("industry", f"ff48_{industry}.json", default={}) if self.state else {}
            fresh = saved.get("peers_date") and \
                (self.as_of_date - pd.Timestamp(saved["peers_date"])).days < 30
            if fresh:
                ciks = set(saved["peers"])
                self.reuse_log.append(f"FF48 {industry} peers: reused ({len(ciks)})")
            else:
                ciks = set()
                for s in public.ff48_sic_codes(industry):
                    try:
                        ciks |= set(sec.peers_by_sic(s, max_pages=3))
                    except Exception:  # noqa: BLE001
                        continue
                saved.update({"ff48": industry, "peers_date": str(self.as_of_date.date()),
                              "peers": sorted(ciks)})
                if self.state:
                    self.state.write_json(saved, "industry", f"ff48_{industry}.json")
                self.reuse_log.append(f"FF48 {industry} peers: fetched ({len(ciks)})")
            self._peers[industry] = ciks
        return self._peers[industry]

    def save_params(self, kind: str, key, params: dict) -> None:
        if not self.state:
            return
        name = f"ff48_{key}.json" if kind == "industry" else f"{key}.json"
        saved = self.state.read_json(kind, name, default={})
        saved.setdefault("params_by_date", {})[str(self.as_of_date.date())] = params
        self.state.write_json(saved, kind, name)


def run_firm(ctx: Context, ann: sec.Announcement) -> dict:
    prof = sec.profile(ann.cik)
    ind_no, ind_short, ind_name = public.ff48(prof.sic)
    store = PointInTimeStore(sec.quarterly_records(sec.company_facts(ann.cik)))
    snap = store.snapshot(ctx.as_of_date)
    rec: dict = {
        "as_of": str(ctx.as_of), "cik": ann.cik, "ticker": prof.ticker, "name": prof.name,
        "accession": ann.accession, "announced_at": str(ann.accepted),
        "announcement_source": ann.source,
        "announcement_time": (ann.calendar or {}).get("time"),
        "eps_street_actual_zacks": (ann.calendar or {}).get("eps"),
        "eps_street_forecast_zacks": (ann.calendar or {}).get("epsForecast"),
        "eps_surprise_pct_zacks": (ann.calendar or {}).get("surprise"),
        "fiscal_quarter_ending_calendar": (ann.calendar or {}).get("fiscalQuarterEnding"),
        "sic": prof.sic, "ff48": ind_no, "ff48_name": ind_name, "country": prof.country,
        "state": prof.state, "fiscal_year_end": prof.fiscal_year_end,
    }
    if "eps" not in snap or snap["eps"].dropna().size < 8:
        rec["status"] = "insufficient EPS history in XBRL"
        return rec

    eps = snap["eps"].dropna()
    rec["last_xbrl_quarter_end"] = str(eps.index[-1].date())
    # ---- today's press release (preliminary, replaced by the 10-Q/10-K later)
    rel = sec.press_release_numbers(ann)
    rec.update({"release_eps_diluted": rel.eps_diluted, "release_snippet": rel.snippet[:300],
                "release_url": rel.exhibit_url})
    new_end = rel.period_end or (eps.index[-1] + pd.DateOffset(months=3))
    used_release = False
    # If the latest XBRL quarter was filed around the announcement (10-Q/10-K filed with or
    # just before the release), the release describes that quarter: do not append it again.
    last_filed = store.first_available("eps").get(eps.index[-1])
    ann_day = pd.Timestamp(ann.accepted).tz_localize(None).normalize() if pd.Timestamp(ann.accepted).tzinfo \
        else pd.Timestamp(ann.accepted).normalize()
    already_in_xbrl = last_filed is not None and abs((ann_day - pd.Timestamp(last_filed)).days) <= 7
    rec["release_quarter_already_in_xbrl"] = bool(already_in_xbrl)
    if rel.eps_diluted is not None and not already_in_xbrl \
            and new_end > eps.index[-1] + pd.Timedelta(days=45) and _plausible(rel.eps_diluted, eps):
        eps = pd.concat([eps, pd.Series([rel.eps_diluted], index=[pd.Timestamp(new_end)])])
        used_release = True
    rec["release_eps_used"] = used_release
    rec["latest_quarter_end"] = str(eps.index[-1].date())
    rec["eps_latest"] = float(eps.iloc[-1])
    rec["eps_ttm"] = float(eps.iloc[-4:].sum())

    # ---- fundamentals for valuation
    last = lambda f: float(snap[f].dropna().iloc[-1]) if f in snap and snap[f].notna().any() else np.nan  # noqa: E731
    ttm = lambda f: float(snap[f].dropna().iloc[-4:].sum()) if f in snap and snap[f].notna().sum() >= 4 else np.nan  # noqa: E731
    shares = last("shares_diluted")
    equity = last("equity")
    rec.update({"equity": equity, "shares_diluted": shares, "revenue_ttm": ttm("revenue"),
                "ni_ttm": ttm("ni"), "cfo_ttm": ttm("cfo"), "capex_ttm": ttm("capex"),
                "dps_ttm": ttm("dps"), "assets": last("assets"), "debt": last("debt"),
                "cash": last("cash")})
    bvps = equity / shares if shares and shares > 0 else np.nan
    rec["bvps"] = bvps
    payout = rec["dps_ttm"] / rec["eps_ttm"] if rec["eps_ttm"] > 0 and np.isfinite(rec["dps_ttm"]) else 0.0
    payout = float(min(max(payout, 0.0), 1.0))
    rec["payout"] = payout

    # ---- market data
    price, pdate = (public.price_as_of(prof.ticker, ctx.as_of_date) if prof.ticker else (None, None))
    beta = public.capm_beta(prof.ticker, ctx.as_of_date) if prof.ticker else None
    rec.update({"price": price, "price_date": str(pdate.date()) if pdate is not None else None,
                "beta_raw": beta, "beta_blume": _blume(beta)})
    if price and shares:
        rec["market_cap"] = price * shares
        rec["book_to_market"] = equity / rec["market_cap"] if equity == equity else np.nan
        rec["earnings_yield_ttm"] = rec["eps_ttm"] / price
    rf, erp = ctx.macro.get("rf_10y", 0.04), ctx.macro["erp"]
    r = rf + rec["beta_blume"] * erp
    rec.update({"rf_10y": rf, "erp": erp, "cost_of_equity": r})
    rec.update({f"macro_{k}": v for k, v in ctx.macro.items() if isinstance(v, (int, float))})

    # ---- Model I
    forecasts = []
    for w in WINDOWS:
        for k, f in F.firm_model(eps, w).items():
            forecasts.append({"info_set": "firm", "window_years": w, "horizon_q": k, **f})

    # ---- Models II / III on the industry panel
    panel = ctx.panel.copy()
    peers = ctx.peers(ind_no)
    panel["ind_member"] = panel["cik"].isin(peers)
    ind_med = panel[panel["ind_member"]].groupby("cq")["roe"].median()
    cty_med = panel[panel["country"] == prof.country].groupby("cq")["roe"].median()
    panel["ind_med"] = panel["cq"].map(ind_med)
    panel["cty_med"] = panel["cq"].map(cty_med)
    est = panel[panel["ind_member"]]
    rec["industry_panel_firms"] = int(est["cik"].nunique())
    if not (equity > 0):
        rec["panel_note"] = "negative or missing book equity: ROE-based panel and RIM models not applicable"
    if est["cik"].nunique() < 15:  # thin industry: fall back to all firms, keep industry features
        est = panel
        rec["industry_panel_note"] = "fewer than 15 FF48 peers; estimated on all firms"

    firm_snap = snap.copy()
    if used_release and np.isfinite(shares):
        firm_snap.loc[pd.Timestamp(new_end), "ni"] = rel.eps_diluted * shares
    fq = F.calendarize(firm_snap)
    rec["roe_latest_q"] = float(fq["roe"].iloc[-1]) if len(fq) else np.nan
    t0 = fq.index[-1] if len(fq) else None
    for info, feats in (("industry", F.FEATURES_II), ("country", F.FEATURES_III)):
        if t0 is None or not np.isfinite(rec["roe_latest_q"]):
            break
        be = float(fq["be_lag"].dropna().iloc[-1]) if fq["be_lag"].notna().any() else equity
        be = equity if np.isfinite(equity) else be
        for w in WINDOWS:
            b_running = be
            for k in F.HORIZONS:
                key = (ind_no, prof.country, k, info, len(est))
                if key not in ctx.designs:
                    ctx.designs[key] = F.build_design(est, k, ctx.macro_q if info == "country" else None)
                design = ctx.designs[key]
                beta_k, diag = F.panel_model(design, feats, ctx.train_end, k, w)
                if beta_k is None:
                    continue
                x = F.firm_features(fq, panel, k, ctx.macro)
                xv = np.array([1.0] + [x.get(c, np.nan) for c in feats], dtype=float)
                if not np.all(np.isfinite(xv)):
                    xv = np.where(np.isfinite(xv), xv, 0.0)  # missing seasonal lag -> 0 (documented)
                roe_hat = float(np.clip(xv @ beta_k, diag["y_lo"], diag["y_hi"]))
                ni_hat = roe_hat * b_running
                b_running += ni_hat * (1 - payout)
                forecasts.append({"info_set": info, "window_years": w, "horizon_q": k,
                                  "roe": roe_hat, "eps": ni_hat / shares if shares else np.nan,
                                  "n_obs": diag["n_obs"], "r2": diag.get("r2")})

    fc = pd.DataFrame(forecasts)
    if fc.empty:
        rec["status"] = "no forecasts"
        return rec
    # "all" = equal-weight combination of the three information sets
    comb = fc.dropna(subset=["eps"]).groupby(["window_years", "horizon_q"])["eps"].mean().reset_index().assign(info_set="all")
    fc = pd.concat([fc, comb], ignore_index=True)
    step = pd.DateOffset(months=3)
    fc["period_end_est"] = [str((eps.index[-1] + step * int(k)).date()) for k in fc["horizon_q"]]
    for (info, w), g in fc.groupby(["info_set", "window_years"]):
        for _, row in g.iterrows():
            rec[f"eps_{info}_{w}y_q{int(row.horizon_q)}"] = row.eps
        rec[f"eps_{info}_{w}y_4q_sum"] = float(g.sort_values("horizon_q")["eps"].sum(min_count=4))

    # ---- valuation per information set
    firm_annual = F.calendarize(snap)
    fa = pd.DataFrame({"cik": ann.cik, "cq": firm_annual.index, "ni": firm_annual["ni"].to_numpy(),
                       "be": firm_snap["equity"].dropna().groupby(
                           [calendar_quarter(t) for t in firm_snap["equity"].dropna().index]).last()
                       .reindex(firm_annual.index).to_numpy()})
    params = {
        "firm": _info_params(_annual_roe(fa)),
        "industry": _info_params(ctx.annual[ctx.annual["cik"].isin(peers)]),
        "country": _info_params(ctx.annual[ctx.annual["country"] == prof.country]),
    }
    p_all = {k: np.nanmean([params["industry"][k], params["country"][k]]) for k in ("roe_target", "omega", "be_growth")}
    params["all"] = p_all
    ctx.save_params("industry", ind_no, params["industry"])
    ctx.save_params("country", prof.country, params["country"])
    if params["firm"]["n_firms"]:  # shrink a single firm's persistence toward the literature value
        params["firm"]["omega"] = 0.5 * params["firm"]["omega"] + 0.5 * 0.6
    nominal = ctx.macro.get("nominal_growth_lr", 0.04)
    g_long = {"firm": params["firm"].get("be_growth"), "industry": params["industry"].get("be_growth"),
              "country": nominal, "all": nominal}
    vals = []
    for info in INFO_SETS:
        for w in WINDOWS:
            key = f"eps_{info}_{w}y_4q_sum"
            if key not in rec or not np.isfinite(rec[key]) or not np.isfinite(bvps):
                continue
            prm = params[info]
            target = prm["roe_target"] if np.isfinite(prm.get("roe_target", np.nan)) else r
            g = g_long[info] if g_long[info] is not None and np.isfinite(g_long[info]) else nominal
            vi = ValuationInputs(bvps0=bvps, eps_fy1=rec[key], eps_ttm=rec["eps_ttm"], payout=payout,
                                 r=r, roe_target=target, g_long=min(max(g, 0.0), nominal, r - 0.01),
                                 omega=prm.get("omega", 0.6))
            v = value_all(vi)
            row = {"info_set": info, "window_years": w, **{f"value_{m}": x for m, x in v.items()},
                   "roe_target": vi.roe_target, "omega": vi.omega, "g_long": vi.g_long,
                   "eps_fy1": vi.eps_fy1, "r": r}
            if price:
                row.update({f"v_to_p_{m}": x / price for m, x in v.items()})
            vals.append(row)
            rec[f"target_price_{info}_{w}y"] = v["composite"]
    rec["status"] = "ok"
    rec["_forecasts"] = fc.to_dict("records")
    rec["_valuations"] = vals
    rec["_params"] = params
    return rec


# ---------------------------------------------------------------------- run
def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=Path(__file__).resolve().parent).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_spec(ctx: "Context", as_of, n: int) -> dict:
    import os
    import platform
    return {
        "as_of": str(as_of),
        "run_started_utc": str(pd.Timestamp.now(tz="UTC")),
        "code_commit": _git_commit(),
        "python": platform.python_version(),
        # Filled in by whoever launches the run (e.g. the scheduled agent session).
        "ai_agent": {"tool": os.environ.get("EARNINGS_AI_TOOL", "unspecified"),
                     "model": os.environ.get("EARNINGS_AI_MODEL", "unspecified"),
                     "effort": os.environ.get("EARNINGS_AI_EFFORT", "unspecified")},
        "announcement_rule": f"firms on the Nasdaq/Zacks earnings calendar for the run date (press-release date), plus any 8-K Item 2.02 accepted by {CUTOFF} US/Eastern; 8-K text used only if accepted by the cutoff",
        "sample_filter": "Russell 3000 proxy (top 3,000 US filers by 10-K public float) and >=1 analyst estimate (Nasdaq/Zacks)",
        "estimation_windows_years": list(WINDOWS),
        "horizons_quarters": list(F.HORIZONS),
        "information_sets": {
            "firm": "seasonal random walk, SRW with drift, Brown-Rozeff SARIMA(1,0,0)(0,1,1)4; equal weights",
            "industry": "pooled OLS of ROE_t+k on " + ", ".join(F.FEATURES_II) + " over FF48 peers",
            "country": "industry model plus " + ", ".join(F.FEATURES_III[len(F.FEATURES_II):]),
            "all": "equal-weight average of firm, industry and country EPS forecasts",
        },
        "valuation": {"models": ["RIM (GLS 2001, 12-year fade)", "DCF FCFE (5-year fade, Gordon)",
                                 "EP-RIM (estimated AR(1) ROE persistence, 30 years)"],
                      "composite": "median of the three",
                      "cost_of_equity": "10y Treasury + Blume beta (60 monthly returns vs SPY) x Damodaran implied ERP"},
        "train_end_quarter": str(ctx.train_end),
        "state_reuse": ctx.reuse_log,
        "n_announcements": n,
        "macro": ctx.macro,
    }


_TIME = {"time-pre-market": "08:00", "time-after-hours": "16:30"}


def announcements(day: str, as_of: pd.Timestamp) -> list[sec.Announcement]:
    """Earnings announcements on ``day``.

    The announcement date comes from the earnings calendar (the press-release
    date), because the Item 2.02 8-K can be filed up to four business days later.
    An 8-K accepted by the cutoff, if any, is attached only as the source of the
    press-release text. 8-K filers missing from the calendar are kept as well.
    """
    from .analysts import earnings_calendar
    tickers = sec.ticker_map()
    by_ticker = {str(t).upper(): int(c) for t, c in zip(tickers["ticker"], tickers["cik"])}
    eightk = {a.cik: a for a in sec.announcements_on(pd.Timestamp(day).date()) if a.accepted <= as_of}
    out, seen = [], set()
    for row in earnings_calendar(day):
        tic = str(row.get("symbol", "")).upper()
        cik = by_ticker.get(tic) or by_ticker.get(tic.replace(".", "-"))
        if cik is None or cik in seen:
            continue
        seen.add(cik)
        when = pd.Timestamp(f"{day} {_TIME.get(row.get('time'), '12:00')}", tz="America/New_York")
        k = eightk.get(cik)
        out.append(sec.Announcement(cik, row.get("name", ""), k.accession if k else None, when,
                                    k.index_url if k else None, "calendar", tic, row))
    for cik, k in eightk.items():
        if cik not in seen:
            k.source = "8-K only"
            out.append(k)
    return out


def run(day: str, out_dir: str = "outputs", limit: int | None = None,
        state_dir: str | None = "state") -> Path:
    as_of = pd.Timestamp(f"{day} {CUTOFF}", tz="America/New_York")
    anns = announcements(day, as_of)
    n_all = len(anns)
    from .analysts import coverage
    from .universe import russell3000_proxy
    universe = russell3000_proxy(as_of, frozenset(filings_after(as_of)))
    r3 = universe.set_index("cik")
    kept, cov = [], {}
    for a in anns:
        if a.cik not in r3.index:
            continue
        c = coverage(sec.profile(a.cik).ticker)
        if c.get("n_analysts", 0) >= 1:
            kept.append(a)
            cov[a.cik] = c
    log.info("sample: %d announcements -> %d in Russell 3000 proxy -> %d with analyst coverage",
             n_all, sum(a.cik in r3.index for a in anns), len(kept))
    anns = kept[:limit] if limit else kept
    log.info("%d earnings announcements (8-K Item 2.02) on %s up to 19:00 ET", len(anns), day)
    state = StateStore(state_dir) if state_dir else None
    ctx = Context(as_of, state)
    out = Path(out_dir) / day
    out.mkdir(parents=True, exist_ok=True)
    summaries, fcs, vals = [], [], []
    for a in anns:
        log.info("  %s (%s)", a.name, a.cik)
        try:
            rec = run_firm(ctx, a)
            rec.update({"r3000_rank": int(r3.loc[a.cik, "rank"]),
                        "public_float": float(r3.loc[a.cik, "public_float"]), **cov[a.cik]})
        except Exception as e:  # noqa: BLE001 - one bad filer must not stop the run
            rec = {"cik": a.cik, "name": a.name, "status": f"error: {e}",
                   "trace": traceback.format_exc(limit=3)}
        for f in rec.pop("_forecasts", []):
            fcs.append({"cik": a.cik, "ticker": rec.get("ticker"), "name": rec.get("name"), **f})
        for v in rec.pop("_valuations", []):
            vals.append({"cik": a.cik, "ticker": rec.get("ticker"), "name": rec.get("name"),
                         "price": rec.get("price"), **v})
        params = rec.pop("_params", None)
        if params:
            for info, prm in params.items():
                for k, v in prm.items():
                    rec[f"param_{info}_{k}"] = v
        summaries.append(rec)
    pd.DataFrame(summaries).to_csv(out / "firm_summary.csv", index=False)
    pd.DataFrame(fcs).to_csv(out / "eps_forecasts.csv", index=False)
    pd.DataFrame(vals).to_csv(out / "valuations.csv", index=False)
    meta = {"as_of": str(as_of), "n_announcements": len(anns), "n_announcements_all": n_all, "train_end_quarter": str(ctx.train_end),
            "panel_rows": int(len(ctx.panel)), "macro": ctx.macro,
            "sources": ["SEC EDGAR XBRL companyfacts/frames/submissions, 8-K Item 2.02",
                        "U.S. Treasury par yield curve", "Philadelphia Fed RTDSM (ROUTPUT, P)",
                        "Damodaran implied ERP", "Kenneth French FF48 SIC definitions",
                        "Nasdaq public quote API (daily close)"]}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    spec = run_spec(ctx, as_of, len(anns))
    (out / "run_spec.json").write_text(json.dumps(spec, indent=2, default=str))
    if state:
        state.write_json(spec, "runs", day, "run_spec.json")
        sm = pd.DataFrame(summaries)
        state.append(sm, "history", "firm_summary.csv")
        state.append(pd.DataFrame(fcs).assign(as_of=str(as_of)), "history", "eps_forecasts.csv")
        state.append(pd.DataFrame(vals).assign(as_of=str(as_of)), "history", "valuations.csv")
        for _, r in sm.iterrows():
            state.append(pd.DataFrame([r]), "firms", str(r["cik"]), "history.csv")
    from .report import write_report
    write_report(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(pd.Timestamp.now(tz="America/New_York").date()))
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--state", default="state")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from .calendar import is_trading_day
    if not is_trading_day(pd.Timestamp(args.date)):
        log.info("%s is not an NYSE trading day; nothing to do", args.date)
        return
    print(run(args.date, args.out, args.limit, args.state))


if __name__ == "__main__":
    main()

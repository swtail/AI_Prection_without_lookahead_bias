"""Human-readable daily report (markdown) from a run directory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(x, nd=2):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{nd}f}"


def write_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    s = pd.read_csv(run_dir / "firm_summary.csv")
    lines = [f"# Earnings forecasts and valuations - as of {meta['as_of']}", "",
             f"Announcements (8-K Item 2.02) processed: {meta['n_announcements']}  ",
             f"Panel training data through calendar quarter: {meta['train_end_quarter']}  ",
             f"10y Treasury {_fmt(100 * meta['macro'].get('rf_10y', np.nan))}%, "
             f"Damodaran implied ERP {_fmt(100 * meta['macro'].get('erp', np.nan))}%, "
             f"real GDP growth (yoy, real-time vintage) {_fmt(100 * meta['macro'].get('real_gdp_growth_yoy', np.nan))}%",
             "", "## EPS forecasts (diluted, 10-year estimation window) and target prices", "",
             "| Ticker | Name | FF48 | Last EPS | Release used | Firm 4Q | Firm+Ind 4Q | +Country 4Q | All 4Q "
             "| Price | Target (all) | V/P |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        if r.get("status") != "ok":
            continue
        tp = r.get("target_price_all_10y", np.nan)
        price = r.get("price", np.nan)
        vp = tp / price if price and np.isfinite(price) and np.isfinite(tp) else np.nan
        lines.append(
            f"| {r.get('ticker', '')} | {str(r.get('name', ''))[:30]} | {r.get('ff48', '')} "
            f"| {_fmt(r.get('eps_latest'))} | {'yes' if r.get('release_eps_used') else 'no'} "
            f"| {_fmt(r.get('eps_firm_10y_4q_sum'))} | {_fmt(r.get('eps_industry_10y_4q_sum'))} "
            f"| {_fmt(r.get('eps_country_10y_4q_sum'))} | {_fmt(r.get('eps_all_10y_4q_sum'))} "
            f"| {_fmt(price)} | {_fmt(tp)} | {_fmt(vp)} |")
    skipped = s[s.get("status", pd.Series(dtype=str)) != "ok"]
    if len(skipped):
        lines += ["", "## Not forecast", ""]
        lines += [f"- {r.get('name')} (CIK {r.get('cik')}): {r.get('status')}" for _, r in skipped.iterrows()]
    lines += ["", "## Notes", "",
              "- Values for firms with negative book equity are not computed (ROE-based models undefined).",
              "- Residual income book growth is capped at long-run growth (steady-state payout).",
              "- 5-year-window panel models (especially the country model) are estimated on a short, "
              "post-2020 sample and can extrapolate macro effects; compare with the 10-year window and "
              "analyst consensus in firm_summary.csv."]
    lines += ["", "## Sources", ""] + [f"- {x}" for x in meta["sources"]]
    lines += ["", "Files: `firm_summary.csv` (all variables), `eps_forecasts.csv` (by information set, "
              "window and horizon), `valuations.csv` (RIM, DCF, EP-RIM, composite, V/P), `run_meta.json`, "
              "`run_spec.json`."]
    p = run_dir / "report.md"
    p.write_text("\n".join(lines) + "\n")
    return p

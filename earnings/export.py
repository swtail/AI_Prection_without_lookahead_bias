"""Slim, human-sized copies of a run's outputs (for Dropbox upload).

The full outputs stay on the ``analysis-state`` branch; these files keep the
variables a reader needs and stay small enough to upload as text.

    python -m earnings.export outputs/2026-11-20 upload/
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

SUMMARY_COLS = [
    "ticker", "name", "cik", "ff48", "r3000_rank", "n_analysts", "announced_at", "latest_quarter_end",
    "eps_latest", "release_eps_used", "eps_ttm", "consensus_q1_period", "consensus_q1_eps",
    "consensus_fy1_period", "consensus_fy1_eps",
    "eps_firm_10y_4q_sum", "eps_industry_10y_4q_sum", "eps_country_10y_4q_sum", "eps_all_10y_4q_sum",
    "eps_firm_5y_4q_sum", "eps_industry_5y_4q_sum", "eps_country_5y_4q_sum", "eps_all_5y_4q_sum",
    "bvps", "payout", "price", "market_cap", "book_to_market", "beta_blume", "cost_of_equity",
    "target_price_firm_10y", "target_price_industry_10y", "target_price_country_10y",
    "target_price_all_10y", "status",
]
VAL_COLS = ["ticker", "info_set", "eps_fy1", "r", "roe_target", "omega", "g_long", "value_rim",
            "value_dcf", "value_ep_rim", "value_composite", "price", "v_to_p_composite"]


def export(run_dir: str | Path, dest: str | Path) -> list[Path]:
    run_dir, dest = Path(run_dir), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    day = run_dir.name
    written = []

    s = pd.read_csv(run_dir / "firm_summary.csv", low_memory=False)
    p = dest / f"{day}_firm_summary.csv"
    s[[c for c in SUMMARY_COLS if c in s]].round(3).to_csv(p, index=False)
    written.append(p)

    f = pd.read_csv(run_dir / "eps_forecasts.csv")
    if len(f):
        w = f.pivot_table(index=["ticker", "info_set", "window_years"], columns="horizon_q",
                          values="eps").round(3)
        w.columns = [f"eps_q{c}" for c in w.columns]
        w["eps_4q_sum"] = w.sum(axis=1, min_count=4).round(3)
        w = w.reset_index()
        q1_end = f[f["horizon_q"] == 1].groupby("ticker")["period_end_est"].first()
        w.insert(1, "q1_period_end_est", w["ticker"].map(q1_end))
        p = dest / f"{day}_eps_forecasts.csv"
        w.to_csv(p, index=False)
        written.append(p)

    v = pd.read_csv(run_dir / "valuations.csv")
    if len(v):
        v = v[v["window_years"] == 10]
        p = dest / f"{day}_valuations_10y.csv"
        v[[c for c in VAL_COLS if c in v]].round(2).to_csv(p, index=False)
        written.append(p)

    for name in ("report.md", "run_spec.json"):
        if (run_dir / name).exists():
            p = dest / f"{day}_{name}"
            shutil.copy(run_dir / name, p)
            written.append(p)
    return written


if __name__ == "__main__":
    for path in export(sys.argv[1], sys.argv[2]):
        print(path)

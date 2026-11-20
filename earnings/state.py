"""Persistent analysis state, reused across daily runs.

Layout under ``root`` (default ``state/``, kept on the ``analysis-state`` branch):

    global/panel.parquet           cross-sectional panel (all filers, calendar quarters)
    global/panel_meta.json         last quarter built, build time
    global/macro_by_quarter.csv    macro features per quarter, as known ~45 days after quarter end
    industry/ff48_<n>.json         peer CIKs + date, industry parameters by run date
    country/<cc>.json              country parameters by run date
    firms/<cik>/history.csv        every forecast/valuation summary ever produced for the firm
    history/*.csv                  append-only log of all runs (summaries, forecasts, valuations)
    runs/<date>/run_spec.json      full specification of each run

Closed quarters are not re-downloaded or recomputed: only the two most recent
calendar quarters (still being filed) are refreshed on each run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class StateStore:
    def __init__(self, root: str | Path = "state"):
        self.root = Path(root)

    def _p(self, *parts) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ---------------------------------------------------------------- json
    def read_json(self, *parts, default=None):
        p = self.root.joinpath(*parts)
        return json.loads(p.read_text()) if p.exists() else default

    def write_json(self, obj, *parts) -> None:
        self._p(*parts).write_text(json.dumps(obj, indent=2, default=str))

    # --------------------------------------------------------------- panel
    def load_panel(self) -> tuple[pd.DataFrame | None, dict]:
        p = self.root / "global" / "panel.parquet"
        if not p.exists():
            return None, {}
        df = pd.read_parquet(p)
        df["cq"] = pd.PeriodIndex(df["cq"], freq="Q")
        return df, self.read_json("global", "panel_meta.json", default={})

    def save_panel(self, panel: pd.DataFrame, meta: dict) -> None:
        d = panel.copy()
        d["cq"] = d["cq"].astype(str)
        d.to_parquet(self._p("global", "panel.parquet"), index=False)
        self.write_json(meta, "global", "panel_meta.json")

    def load_macro_q(self) -> pd.DataFrame | None:
        p = self.root / "global" / "macro_by_quarter.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p, index_col=0)
        df.index = pd.PeriodIndex(df.index, freq="Q")
        return df

    def save_macro_q(self, df: pd.DataFrame) -> None:
        d = df.copy()
        d.index = d.index.astype(str)
        d.to_csv(self._p("global", "macro_by_quarter.csv"))

    # ------------------------------------------------------- append-only logs
    def append(self, df: pd.DataFrame, *parts, keys=("as_of", "cik")) -> None:
        if df.empty:
            return
        p = self._p(*parts)
        if p.exists():
            old = pd.read_csv(p, low_memory=False)
            df = pd.concat([old, df], ignore_index=True)
            k = [c for c in keys if c in df.columns]
            if k:
                df = df.drop_duplicates(subset=k + [c for c in ("info_set", "window_years", "horizon_q")
                                                    if c in df.columns], keep="last")
        df.to_csv(p, index=False)

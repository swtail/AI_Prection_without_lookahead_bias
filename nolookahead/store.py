"""Point-in-time (bitemporal) data store.

Every observation carries two timestamps:

* ``event_time``     - the period the value describes (e.g. GDP for Q1).
* ``available_time`` - when the value became known to the world
  (e.g. GDP for Q1 is first published in late April, then revised in May).

A model making a decision at time ``T`` may only see records whose
``available_time <= T``. :meth:`PointInTimeStore.snapshot` enforces this, and
because revisions are stored as additional rows, a snapshot returns the
*vintage* that was known at ``T`` rather than today's revised number.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

_COLUMNS = ["event_time", "available_time", "field", "value"]


class PointInTimeStore:
    def __init__(self, records: pd.DataFrame):
        missing = set(_COLUMNS) - set(records.columns)
        if missing:
            raise ValueError(f"records missing columns: {sorted(missing)}")
        rec = records[_COLUMNS].copy()
        if (rec["available_time"] < rec["event_time"]).any():
            raise ValueError(
                "a record is available before the period it describes "
                "(available_time < event_time); this is almost always a data bug"
            )
        # Stable sort so that, for identical available_time, later rows win.
        self._records = rec.sort_values("available_time", kind="stable").reset_index(drop=True)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_series(
        cls,
        series: pd.Series,
        field: str,
        publication_lag: pd.Timedelta | int | None = None,
    ) -> "PointInTimeStore":
        """Build a store from a plain series indexed by event time.

        ``publication_lag`` says how long after the event the value is
        published. Leaving it at ``None`` (lag 0) is only correct for data that
        is truly known at the event time, such as a closing price.
        """
        return cls(_series_records(series, field, publication_lag))

    @classmethod
    def concat(cls, stores: Iterable["PointInTimeStore"]) -> "PointInTimeStore":
        return cls(pd.concat([s._records for s in stores], ignore_index=True))

    def with_revisions(self, revisions: pd.DataFrame) -> "PointInTimeStore":
        """Return a new store with extra (later-vintage) rows appended."""
        return PointInTimeStore(pd.concat([self._records, revisions[_COLUMNS]], ignore_index=True))

    # ------------------------------------------------------------------ query
    @property
    def records(self) -> pd.DataFrame:
        return self._records.copy()

    @property
    def fields(self) -> list[str]:
        return sorted(self._records["field"].unique())

    def snapshot(self, as_of) -> pd.DataFrame:
        """Wide frame (index=event_time, columns=fields) of what was known at ``as_of``."""
        rec = self._records
        known = rec[(rec["available_time"] <= as_of) & (rec["event_time"] <= as_of)]
        if known.empty:
            return pd.DataFrame(columns=self.fields)
        latest = known.groupby(["event_time", "field"], sort=False)["value"].last()
        wide = latest.unstack("field").sort_index()
        return wide.reindex(columns=self.fields)

    def latest(self) -> pd.DataFrame:
        """Today's fully revised view. Use ONLY for scoring, never for features."""
        return self.snapshot(self._records["available_time"].max())

    def first_available(self, field: str) -> pd.Series:
        """For each event_time, the first time any value for ``field`` became known."""
        rec = self._records[self._records["field"] == field]
        return rec.groupby("event_time")["available_time"].min().sort_index()

    def first_release(self) -> "PointInTimeStore":
        """Store containing only the first vintage of each (event_time, field)."""
        rec = self._records.groupby(["event_time", "field"], sort=False).head(1)
        return PointInTimeStore(rec)

    def perturb_after(self, as_of, rng: np.random.Generator | None = None) -> "PointInTimeStore":
        """Copy of the store where everything unknown at ``as_of`` is replaced by noise.

        Used by the future-invariance audit: a prediction made at ``as_of``
        must not change when information from after ``as_of`` is scrambled.
        """
        rng = rng or np.random.default_rng(0)
        rec = self._records.copy()
        future = rec["available_time"] > as_of
        scale = rec["value"].abs().mean() or 1.0
        rec.loc[future, "value"] = rng.normal(0, 10 * scale, future.sum())
        return PointInTimeStore(rec)


def _series_records(series: pd.Series, field: str, lag) -> pd.DataFrame:
    s = series.dropna()
    event = s.index
    available = event if lag is None else event + lag
    return pd.DataFrame(
        {"event_time": event, "available_time": available, "field": field, "value": s.to_numpy()}
    )

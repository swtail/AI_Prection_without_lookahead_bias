"""NYSE trading-day check (regular full-day closures, rules as published by NYSE)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def _easter(y: int) -> date:
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return date(y, month, day)


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nth(y, month, weekday, n):
    d = date(y, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last(y, month, weekday):
    d = date(y, month + 1, 1) - timedelta(days=1) if month < 12 else date(y, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def nyse_holidays(y: int) -> set[date]:
    h = {
        _observed(date(y, 1, 1)),
        _nth(y, 1, 0, 3),             # Martin Luther King Jr. Day
        _nth(y, 2, 0, 3),             # Washington's Birthday
        _easter(y) - timedelta(days=2),  # Good Friday
        _last(y, 5, 0),               # Memorial Day
        _observed(date(y, 7, 4)),
        _nth(y, 9, 0, 1),             # Labor Day
        _nth(y, 11, 3, 4),            # Thanksgiving
        _observed(date(y, 12, 25)),
    }
    if y >= 2022:
        h.add(_observed(date(y, 6, 19)))  # Juneteenth
    if date(y, 1, 1).weekday() == 5:  # NYSE does not observe New Year's on the prior Friday
        h.discard(date(y - 1, 12, 31))
    return h


def is_trading_day(ts) -> bool:
    d = pd.Timestamp(ts).date()
    return d.weekday() < 5 and d not in nyse_holidays(d.year)

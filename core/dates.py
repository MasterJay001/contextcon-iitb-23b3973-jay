"""Fuzzy-date handling (leak class #5).

Crustdata returns role dates as `YYYY-MM-DD` strings with no precision flag, so a role that
started "in 2025" is indistinguishable from one that started on 2025-01-01. We therefore
*infer* precision conservatively:

    day != 1                -> day precision    (exact)
    day == 1, month != 1    -> month precision  (could be any day in that month)
    day == 1, month == 1    -> year precision   (could be any day in that year)

A date is only trusted relative to the as-of date T when its whole uncertainty interval
`[lo, hi]` falls on one side of T. Straddling dates are "ambiguous" and the Time Guard drops
them (conservative), which can only remove information, never leak it.
"""
from __future__ import annotations

import calendar
import datetime as dt
import re
from typing import Any

from pydantic import BaseModel, ConfigDict

_DATE_RE = re.compile(r"^\s*(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?")
DAYS_PER_MONTH = 30.4375
_NULLISH = {"", "null", "none", "present", "current", "ongoing", "now"}


class FuzzyDate(BaseModel):
    model_config = ConfigDict(frozen=True)

    date: dt.date
    precision: str = "day"          # day | month | year
    inferred: bool = False          # precision was inferred from a full YYYY-MM-DD string

    @property
    def lo(self) -> dt.date:
        if self.precision == "year":
            return dt.date(self.date.year, 1, 1)
        if self.precision == "month":
            return dt.date(self.date.year, self.date.month, 1)
        return self.date

    @property
    def hi(self) -> dt.date:
        if self.precision == "year":
            return dt.date(self.date.year, 12, 31)
        if self.precision == "month":
            return dt.date(self.date.year, self.date.month, calendar.monthrange(self.date.year, self.date.month)[1])
        return self.date

    def label(self) -> str:
        """Display string at the precision we actually trust."""
        if self.precision == "year":
            return f"{self.date.year:04d}"
        if self.precision == "month":
            return f"{self.date.year:04d}-{self.date.month:02d}"
        return self.date.isoformat()


def _from_ymd(y: int, m: int, d: int) -> FuzzyDate:
    date = dt.date(y, m, d)
    if d == 1 and m == 1:
        return FuzzyDate(date=date, precision="year", inferred=True)
    if d == 1:
        return FuzzyDate(date=date, precision="month", inferred=True)
    return FuzzyDate(date=date, precision="day")


def parse_date(value: Any) -> FuzzyDate | None:
    """Parse API dates (`2017-01-01`, `2024-03-15T00:00:00Z`, `2024-03`, `2024`, date objects)."""
    if value is None:
        return None
    if isinstance(value, FuzzyDate):
        return value
    if isinstance(value, dt.datetime):
        return _from_ymd(value.year, value.month, value.day)
    if isinstance(value, dt.date):
        return _from_ymd(value.year, value.month, value.day)
    s = str(value).strip()
    if s.lower() in _NULLISH:
        return None
    m = _DATE_RE.match(s)
    if not m:
        return None
    y = int(m.group(1))
    if not 1950 <= y <= 2100:
        return None
    mo, d = m.group(2), m.group(3)
    if mo is None:
        return FuzzyDate(date=dt.date(y, 1, 1), precision="year")
    month = int(mo)
    if not 1 <= month <= 12:
        return None
    if d is None:
        return FuzzyDate(date=dt.date(y, month, 1), precision="month")
    try:
        return _from_ymd(y, month, int(d))
    except ValueError:
        return None


def relation_to(fd: FuzzyDate, T: dt.date, *, trust: bool = False) -> str:
    """'before' (definitely <= T), 'after' (definitely > T) or 'ambiguous'.

    `trust=True` is used when T is at/after the data snapshot (live mode): every date in the
    data is by construction <= the snapshot, so there is nothing to be ambiguous about.
    """
    if trust:
        return "before" if fd.date <= T else "after"
    if fd.hi <= T:
        return "before"
    if fd.lo > T:
        return "after"
    return "ambiguous"


def months_between(a: dt.date, b: dt.date) -> float:
    """Signed fractional months from a to b."""
    return (b - a).days / DAYS_PER_MONTH


def add_days(d: dt.date, n: int) -> dt.date:
    return d + dt.timedelta(days=n)


def overlap_months(s1: dt.date, e1: dt.date, s2: dt.date, e2: dt.date) -> float:
    """Months two closed intervals overlap (0 if disjoint)."""
    lo, hi = max(s1, s2), min(e1, e2)
    return max(0.0, (hi - lo).days / DAYS_PER_MONTH)


def month_label(d: dt.date | None) -> str:
    return "ongoing" if d is None else f"{d.year:04d}-{d.month:02d}"

import datetime as dt

import pytest

from core.dates import FuzzyDate, overlap_months, parse_date, relation_to


@pytest.mark.parametrize("raw,prec,iso", [
    ("2025-07-17", "day", "2025-07-17"),
    ("2025-07-01", "month", "2025-07-01"),
    ("2025-01-01", "year", "2025-01-01"),
    ("2025-07", "month", "2025-07-01"),
    ("2025", "year", "2025-01-01"),
    ("2024-03-15T00:00:00Z", "day", "2024-03-15"),
])
def test_precision_inference(raw, prec, iso):
    fd = parse_date(raw)
    assert fd is not None and fd.precision == prec and fd.date.isoformat() == iso


@pytest.mark.parametrize("raw", [None, "", "null", "Present", "abc", "1899-01-01", "2025-13-01", "2025-02-31"])
def test_unparseable_is_none(raw):
    assert parse_date(raw) is None


def test_intervals():
    m = parse_date("2025-02-01")
    assert (m.lo, m.hi) == (dt.date(2025, 2, 1), dt.date(2025, 2, 28))
    y = parse_date("2025-01-01")
    assert (y.lo, y.hi) == (dt.date(2025, 1, 1), dt.date(2025, 12, 31))


@pytest.mark.parametrize("raw,T,expected", [
    ("2025-04-01", dt.date(2025, 5, 12), "before"),       # whole April <= T
    ("2025-05-01", dt.date(2025, 5, 12), "ambiguous"),    # May straddles T
    ("2025-06-01", dt.date(2025, 5, 12), "after"),
    ("2025-05-10", dt.date(2025, 5, 12), "before"),       # exact day
    ("2025-05-13", dt.date(2025, 5, 12), "after"),
    ("2025-01-01", dt.date(2025, 5, 12), "ambiguous"),    # year precision inside T's year
    ("2024-01-01", dt.date(2025, 5, 12), "before"),
])
def test_relation(raw, T, expected):
    assert relation_to(parse_date(raw), T) == expected


def test_trust_mode_uses_reported_date():
    fd = parse_date("2025-01-01")
    assert relation_to(fd, dt.date(2025, 5, 12), trust=True) == "before"
    assert relation_to(parse_date("2025-06-01"), dt.date(2025, 5, 12), trust=True) == "after"


def test_overlap_months():
    a, b = dt.date(2020, 1, 1), dt.date(2021, 1, 1)
    assert overlap_months(a, b, dt.date(2020, 7, 1), dt.date(2022, 1, 1)) == pytest.approx(6.0, abs=0.2)
    assert overlap_months(a, b, dt.date(2022, 1, 1), dt.date(2023, 1, 1)) == 0.0

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_config  # noqa: E402
from crust.client import CrustClient, FaultInjector  # noqa: E402
from crust.sim import SimTransport  # noqa: E402
from crust.sim_world import SimWorld  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(overrides={"run": {"now": "2026-10-31"}, "backend": {"crust": "sim", "llm": "mock"}})


@pytest.fixture(scope="session")
def world():
    return SimWorld(seed=7, n_people=1500, n_target=24, n_other=40)


@pytest.fixture()
def client(cfg, world):
    c = CrustClient(cfg, SimTransport(world), api_key="sim-key", use_cache=False, faults=FaultInjector({}),
                    sleep=lambda s: None)
    yield c
    c.close()


def make_role(idx, title, company, start, end=None, cid=None, desc=None, from_current=None, loc=None):
    from core.dates import parse_date
    from core.models import Role

    return Role(idx=idx, src=f"x[{idx}]", title=title, company_name=company, company_id=cid, start=parse_date(start),
                end=parse_date(end), description=desc, location=loc, from_current=(end is None) if from_current is None else from_current)


def make_profile(pid, roles, **kw):
    from core.models import RawProfile

    return RawProfile(person_id=pid, roles=roles, **kw)


T0 = dt.date(2025, 5, 12)

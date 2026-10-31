"""The Evaluator: holds the answer key and scores ranked lists. No agent can read this module (rule 1, enforced by a test).

An agent hands over only an ordered list of person ids and its pool; the evaluator looks up the real hires, finds their
ranks, and returns a `CaseScore`. It never sends anything back into the pipeline except through the explicitly separate,
train-only Failure Analyst channel (`eval/self_improve.py`).
"""
from __future__ import annotations

import json
from pathlib import Path

from core.config import data_path
from core.models import Case
from eval.metrics import CaseScore


def load_cases(cfg: dict, split: str | None = None, crust_backend: str = "live") -> list[Case]:
    p = data_path(cfg, "cases", crust_backend)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run `python -m data.build_cases` (add `--crust-backend sim` for the simulator)")
    cases = [Case.model_validate_json(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [c for c in cases if split in (None, "all") or c.split == split]


class Evaluator:
    def __init__(self, cfg: dict, crust_backend: str = "live"):
        p = data_path(cfg, "hires", crust_backend)
        if not p.exists():
            raise FileNotFoundError(f"answer key {p} not found - run `python -m data.build_cases` first")
        self.hires: dict[str, list[dict]] = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                self.hires[row["case_id"]] = row["hires"]

    def hire_ids(self, case_id: str) -> list[int]:
        return [int(h["person_id"]) for h in self.hires.get(case_id, [])]

    def hire_records(self, case_id: str) -> list[dict]:
        return self.hires.get(case_id, [])

    def score(self, case_id: str, ranked_ids: list[int], pool_ids: list[int], *, injected: bool = False) -> CaseScore:
        pos = {pid: i + 1 for i, pid in enumerate(ranked_ids)}
        ranks = [pos.get(h) for h in self.hire_ids(case_id)]
        return CaseScore(case_id=case_id, pool_size=len(pool_ids), n_hires=len(ranks), ranks=ranks, injected=injected)

    def shuffled_mapping(self, case_ids: list[str], seed: int) -> dict[str, list[int]]:
        """Cross-case reassignment of hires (derangement): case i is scored against the hires of case pi(i)."""
        import random

        rng = random.Random(seed)
        ids = list(case_ids)
        for _ in range(100):
            perm = ids[:]
            rng.shuffle(perm)
            if all(a != b for a, b in zip(ids, perm)) or len(ids) < 2:
                break
        return {a: self.hire_ids(b) for a, b in zip(ids, perm)}

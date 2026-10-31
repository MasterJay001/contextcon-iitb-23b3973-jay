"""Rules that a machine can check (CLAUDE.md #1, #4, #5): the evaluation stays honest by construction."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLIND_DIRS = ["agents", "crust", "llm", "rag", "core"]
BLIND_FILES = ["app/live.py"]
FORBIDDEN = re.compile(r"eval_private|hires\.jsonl")


def _py_files():
    for d in BLIND_DIRS:
        yield from (ROOT / d).rglob("*.py")
    for f in BLIND_FILES:
        if (ROOT / f).exists():
            yield ROOT / f


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                ids.add(id(body[0].value))
    return ids


def test_blind_packages_never_touch_the_answer_key():
    """AST check: no import of `eval.*` and no string literal naming the answer key (docstrings are prose, ignored)."""
    offenders = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docs = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and (node.module == "eval" or node.module.startswith("eval.")):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}: from {node.module} import ...")
            elif isinstance(node, ast.Import) and any(a.name == "eval" or a.name.startswith("eval.") for a in node.names):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}: import eval")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs and FORBIDDEN.search(node.value):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}: string literal {node.value[:40]!r}")
    assert not offenders, "answer-key access outside eval/:\n" + "\n".join(offenders)


def test_neutral_sort_is_the_default_and_semantic_search_is_unused():
    src = (ROOT / "crust" / "people.py").read_text(encoding="utf-8")
    assert 'NEUTRAL_SORT = [{"field": "crustdata_person_id", "order": "asc"}]' in src
    offenders = []
    for path in _py_files():
        if path.name in {"sim.py", "sim_world.py"}:      # the simulator ignores/validates these keys
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"""["']search["']\s*:""", code) or re.search(r"""mode["']\s*:\s*["'](semantic|hybrid)""", code):
                offenders.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not offenders, "semantic/relevance retrieval leaks post-T data (rule 4):\n" + "\n".join(offenders)


def test_every_crustdata_request_goes_through_the_cached_client():
    """Only crust/client.py may import an HTTP library; everything else uses CrustClient (cache, limits, caps)."""
    http_libs = {"httpx", "httpx2", "requests", "urllib3", "aiohttp", "urllib"}
    for path in (ROOT / "crust").glob("*.py"):
        if path.name == "client.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module.split(".")[0]]
            assert not (set(mods) & http_libs), f"{path.name} imports an HTTP library directly and bypasses CrustClient"


def test_secrets_are_not_committed():
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gi and "cache/" in gi and "eval_private/hires.jsonl" in gi
    for p in ROOT.rglob("*.py"):
        if ".venv" in p.parts:
            continue
        assert not re.search(r"sk-ant-[A-Za-z0-9_-]{20,}", p.read_text(encoding="utf-8", errors="ignore")), p

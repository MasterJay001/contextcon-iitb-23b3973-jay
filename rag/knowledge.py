"""Knowledge base: every document under knowledge/ (+ README/NOTES/docs when asked) chunked and indexed.

Chunking is structure-aware: markdown by heading/paragraph (~900 chars, small overlap); the title taxonomy by
family; the location file by metro; other JSON generically. Every hit carries its source so prompts can cite
"[title_taxonomy.json#swe_backend]" and a reviewer can trace a normalisation back to a document.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from core.config import ROOT
from rag.index import Doc, HybridIndex, Hit


def chunk_markdown(text: str, source: str, max_chars: int = 900, overlap: int = 120) -> list[Doc]:
    docs: list[Doc] = []
    heading = ""
    buf: list[str] = []
    size = 0

    def flush():
        nonlocal buf, size
        body = "\n".join(buf).strip()
        if body:
            docs.append(Doc(f"{source}#{len(docs)}", (f"{heading}\n" if heading else "") + body, {"source": source, "heading": heading}))
        tail = body[-overlap:] if overlap and len(body) > overlap else ""
        buf, size = ([tail] if tail else []), len(tail)

    for line in text.splitlines():
        if re.match(r"^#{1,4}\s", line):
            flush()
            buf, size = [], 0
            heading = line.lstrip("# ").strip()
            continue
        if size + len(line) > max_chars and buf:
            flush()
        buf.append(line)
        size += len(line) + 1
    flush()
    return docs


def _taxonomy_docs(path: Path) -> list[Doc]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    out = []
    for f in raw["families"]:
        text = (f"{f['canonical']} ({f['department']} / {f['sub_department']}). Aliases: {', '.join(f['aliases'])}. "
                f"Keywords: {', '.join(f.get('keywords', []))}. Adjacent roles: {', '.join(f.get('adjacent', []))}.")
        out.append(Doc(f"{path.name}#{f['id']}", text, {"source": path.name, "heading": f["canonical"], "family": f["id"]}))
    return out


def _locations_docs(path: Path) -> list[Doc]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    out = []
    for m in raw["metros"]:
        text = f"{m['canonical']} metro, {m['state']}, {m['country']}. Also written: {', '.join(m['aliases'])}."
        out.append(Doc(f"{path.name}#{m['id']}", text, {"source": path.name, "heading": m["canonical"]}))
    return out


def _generic_json_docs(path: Path) -> list[Doc]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, dict):
        return [Doc(f"{path.name}#{k}", f"{k}: {', '.join(map(str, v)) if isinstance(v, list) else v}", {"source": path.name, "heading": k})
                for k, v in raw.items() if not str(k).startswith("_")]
    return [Doc(f"{path.name}#0", json.dumps(raw)[:900], {"source": path.name, "heading": path.stem})]


class KnowledgeBase:
    def __init__(self, cfg: dict | None = None, extra_paths: list[Path] | None = None, backend: str | None = None):
        self.cfg = cfg
        self.root = ROOT / (cfg["paths"]["knowledge"] if cfg else "knowledge")
        self.extra = extra_paths or []
        rag = (cfg or {}).get("fit", {}).get("rag", {}) if cfg else {}
        self.index = HybridIndex(backend or str(rag.get("backend", "auto")), int(rag.get("rrf_k", 60)), cfg=cfg)
        self.docs: list[Doc] = []

    def load(self) -> "KnowledgeBase":
        docs: list[Doc] = []
        for p in sorted(self.root.glob("*")):
            if p.suffix == ".md" or p.suffix == ".txt":
                docs += chunk_markdown(p.read_text(encoding="utf-8-sig"), p.name)
            elif p.name == "title_taxonomy.json":
                docs += _taxonomy_docs(p)
            elif p.name == "locations.json":
                docs += _locations_docs(p)
            elif p.suffix == ".json":
                docs += _generic_json_docs(p)
        for p in self.extra:
            if p.exists() and p.suffix in {".md", ".txt"}:
                docs += chunk_markdown(p.read_text(encoding="utf-8-sig"), p.name)
        self.docs = docs
        self.index.fit(docs)
        return self

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return self.index.search(query, k)

    def context(self, query: str, k: int = 6, max_chars: int = 3000) -> str:
        """Prompt-ready reference block with source tags."""
        parts, used = [], 0
        for h in self.search(query, k):
            snippet = h.doc.text.strip()
            block = f"[{h.doc.id}] {snippet}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    def stats(self) -> dict:
        srcs: dict[str, int] = {}
        for d in self.docs:
            srcs[d.meta.get("source", "?")] = srcs.get(d.meta.get("source", "?"), 0) + 1
        return {"chunks": len(self.docs), "sources": srcs, "dense": self.index.dense_name}

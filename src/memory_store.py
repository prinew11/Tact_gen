"""
Memory store with RAG-like retrieval for fabrication knowledge.

Two memory types:
  - **Semantic memory**: tactile design principles, material knowledge,
    machining best practices.  Soft guidance that shapes correction strategy.
  - **Procedural memory**: hard constraints (max slope, tool radius, min
    feature size).  Each entry carries a machine-readable ``constraint`` dict
    that the corrector can act on directly.

Retrieval uses TF-IDF cosine similarity over entry text (title + content +
tags).  This is lightweight, requires no external model, and works well for
a small, domain-specific knowledge base.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DEFAULT_KB = Path(__file__).resolve().parents[1] / "data" / "fabrication_knowledge.json"


@dataclass
class MemoryEntry:
    id: str
    type: str  # "semantic" or "procedural"
    tags: list[str]
    title: str
    content: str
    constraint: dict[str, Any] | None = None

    def full_text(self) -> str:
        return " ".join(self.tags) + " " + self.title + " " + self.content


@dataclass
class RetrievalResult:
    entry: MemoryEntry
    score: float


_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into through during before after above below between "
    "and or but not no nor so yet both either neither each every all "
    "any few more most other some such that this these those it its".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP_WORDS]


class MemoryStore:
    """In-memory knowledge base with TF-IDF retrieval."""

    def __init__(self, kb_path: str | Path | None = None) -> None:
        self.entries: list[MemoryEntry] = []
        self._idf: dict[str, float] = {}
        self._entry_tfidf: list[dict[str, float]] = []
        if kb_path is None:
            kb_path = _DEFAULT_KB
        self._load(Path(kb_path))

    def _load(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for e in data["entries"]:
            self.entries.append(
                MemoryEntry(
                    id=e["id"],
                    type=e["type"],
                    tags=e.get("tags", []),
                    title=e["title"],
                    content=e["content"],
                    constraint=e.get("constraint"),
                )
            )
        self._build_index()

    def _build_index(self) -> None:
        n = len(self.entries)
        if n == 0:
            return
        doc_tokens = [_tokenize(e.full_text()) for e in self.entries]
        df: Counter[str] = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))
        self._idf = {
            term: math.log((n + 1) / (count + 1)) + 1.0
            for term, count in df.items()
        }
        self._entry_tfidf = []
        for tokens in doc_tokens:
            tf = Counter(tokens)
            total = len(tokens) or 1
            vec = {t: (c / total) * self._idf.get(t, 1.0) for t, c in tf.items()}
            self._entry_tfidf.append(vec)

    def _query_vec(self, text: str) -> dict[str, float]:
        tokens = _tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) or 1
        return {t: (c / total) * self._idf.get(t, 1.0) for t, c in tf.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[k] * b[k] for k in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        type_filter: str | None = None,
        min_score: float = 0.05,
    ) -> list[RetrievalResult]:
        """Retrieve entries most relevant to *query* via TF-IDF cosine similarity."""
        qvec = self._query_vec(query)
        results: list[RetrievalResult] = []
        for entry, evec in zip(self.entries, self._entry_tfidf):
            if type_filter and entry.type != type_filter:
                continue
            score = self._cosine(qvec, evec)
            if score >= min_score:
                results.append(RetrievalResult(entry=entry, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def get_all_procedural(self) -> list[MemoryEntry]:
        """Return all procedural (hard constraint) entries."""
        return [e for e in self.entries if e.type == "procedural"]

    def get_all_semantic(self) -> list[MemoryEntry]:
        """Return all semantic (soft guidance) entries."""
        return [e for e in self.entries if e.type == "semantic"]

    def get_by_id(self, entry_id: str) -> MemoryEntry | None:
        for e in self.entries:
            if e.id == entry_id:
                return e
        return None

    def get_constraints_dict(self) -> dict[str, Any]:
        """Collect all procedural constraints into a flat dict keyed by parameter name."""
        out: dict[str, Any] = {}
        for e in self.entries:
            if e.constraint:
                param = e.constraint.get("parameter")
                if param:
                    out[param] = e.constraint
        return out

    def retrieve_for_material(self, material_hint: str) -> list[RetrievalResult]:
        """Retrieve material-specific guidance."""
        return self.retrieve(f"{material_hint} material machining texture", top_k=3)

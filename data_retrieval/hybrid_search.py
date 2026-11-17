"""Hybrid retrieval: BM25 + dense (Qwen3) -> Reciprocal Rank Fusion -> cross-encoder rerank.

Extras for research papers:
- Visual intent: if the question is about a figure/chart/table/diagram, a second
  retrieval pass restricted to visual chunks is fused in, so figure chunks are not
  drowned out by long text chunks.
- Explicit references: "Figure 3 of LightRAG" / "Table 2 in P02" pins that exact
  figure chunk to the top of the results.
- Paper diversity: at most MAX_CHUNKS_PER_PAPER chunks per paper in the final list,
  so cross-paper questions see evidence from several papers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from config import settings
from data_ingestion.schemas import Chunk
from data_retrieval.reranker import get_reranker
from data_retrieval.store import IndexStore

VISUAL_INTENT_RE = re.compile(
    r"\b(fig(ure)?s?|tables?|charts?|graphs?|plots?|diagrams?|axis|axes|legend|architecture|curve|bar|visual|image)\b",
    re.IGNORECASE,
)
EXPLICIT_REF_RE = re.compile(r"\b(fig(?:ure)?\.?|table)\s*(\d+[a-z]?)\b", re.IGNORECASE)


@dataclass
class Hit:
    chunk: Chunk
    fused_score: float
    rerank_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    pinned: bool = False
    debug: dict = field(default_factory=dict)


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def resolve_papers(query: str, papers: dict[str, dict]) -> list[str]:
    """Find papers named in the question by id (P03), short name (LightRAG) or long title words."""
    q = query.lower()
    found = []
    for pid, meta in papers.items():
        names = [pid.lower(), str(meta.get("short_name", "")).lower()] + [a.lower() for a in meta.get("aliases", [])]
        if any(n and re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", q) for n in names):
            found.append(pid)
    return found


class HybridRetriever:
    def __init__(self, store: IndexStore):
        self.store = store

    def search(self, query: str, top_k: int | None = None, paper_ids: list[str] | None = None,
               chunk_types: list[str] | None = None, per_paper_cap: int | None = None) -> list[Hit]:
        top_k = top_k or settings.final_top_k
        cap = per_paper_cap or settings.max_chunks_per_paper

        dense = self.store.dense_search(query, settings.dense_top_k, paper_ids, chunk_types)
        sparse = self.store.sparse_search(query, settings.sparse_top_k, paper_ids, chunk_types)
        rankings = [[c for c, _ in dense], [c for c, _ in sparse]]

        if VISUAL_INTENT_RE.search(query) and not chunk_types:
            vt = ["figure", "table"]
            rankings.append([c for c, _ in self.store.dense_search(query, 10, paper_ids, vt)])
            rankings.append([c for c, _ in self.store.sparse_search(query, 10, paper_ids, vt)])

        fused = rrf_fuse(rankings, settings.rrf_k)
        dense_rank = {c: i + 1 for i, (c, _) in enumerate(dense)}
        sparse_rank = {c: i + 1 for i, (c, _) in enumerate(sparse)}
        cands = sorted(fused, key=fused.get, reverse=True)[: settings.rerank_top_n]
        hits = [
            Hit(self.store.chunks[c], fused[c], None, dense_rank.get(c), sparse_rank.get(c))
            for c in cands if c in self.store.chunks
        ]

        reranker = get_reranker()
        if reranker is not None and hits:
            scores = reranker.score(query, [h.chunk.text for h in hits])
            for h, s in zip(hits, scores):
                h.rerank_score = s
            hits.sort(key=lambda h: h.rerank_score, reverse=True)

        pinned = self._pinned(query, paper_ids)
        pinned_ids = {p.chunk.chunk_id for p in pinned}
        hits = pinned + [h for h in hits if h.chunk.chunk_id not in pinned_ids]

        out, per_paper = [], {}
        for h in hits:
            n = per_paper.get(h.chunk.paper_id, 0)
            if n >= cap and not h.pinned:
                continue
            per_paper[h.chunk.paper_id] = n + 1
            out.append(h)
            if len(out) >= top_k:
                break
        return out

    def _pinned(self, query: str, paper_ids: list[str] | None) -> list[Hit]:
        refs = EXPLICIT_REF_RE.findall(query)
        if not refs:
            return []
        targets = paper_ids or resolve_papers(query, self.store.papers)
        if not targets:
            return []
        out = []
        for word, num in refs:
            kind = "table" if word.lower().startswith("tab") else "figure"
            for pid in targets:
                c = self.store.find_visual(pid, kind, num)
                if c is not None:
                    out.append(Hit(c, 1.0, None, pinned=True))
        return out


def evidence_is_weak(hits: list[Hit]) -> bool:
    """True when nothing relevant was found (used to refuse instead of hallucinating)."""
    if not hits:
        return True
    if any(h.pinned for h in hits):
        return False
    scored = [h.rerank_score for h in hits if h.rerank_score is not None]
    if scored:
        return max(scored) < settings.min_rerank_score
    return False

"""Retrieval and answer metrics (all deterministic; LLM-judge scores are computed separately)."""
from __future__ import annotations

import math
import re

from data_retrieval.store import tokenize


def _is_rel(hit: dict, rel_papers: set[str], rel_pages: dict[str, list[int]]) -> bool:
    if hit["paper_id"] not in rel_papers:
        return False
    pages = rel_pages.get(hit["paper_id"])
    if not pages:  # paper-level label
        return True
    return any(hit["page_start"] <= p <= hit["page_end"] for p in pages)


def retrieval_metrics(hits: list[dict], rel_papers: list[str], k: int, rel_pages: dict | None = None) -> dict:
    """Relevance = chunk comes from a relevant paper (and from a labelled page, when page labels exist).

    recall@k    = share of relevant papers that appear at least once in the top-k chunks
    precision@k = share of the top-k chunks that are relevant
    mrr         = 1 / rank of the first relevant chunk
    ndcg@k      = binary-gain nDCG over the top-k chunks
    """
    rel = set(rel_papers)
    rel_pages = rel_pages or {}
    top = hits[:k]
    flags = [_is_rel(h, rel, rel_pages) for h in top]
    found = {h["paper_id"] for h, f in zip(top, flags) if f}
    recall = len(found) / len(rel) if rel else None
    precision = sum(flags) / len(top) if top else 0.0
    mrr = next((1.0 / (i + 1) for i, f in enumerate(flags) if f), 0.0)
    dcg = sum(1.0 / math.log2(i + 2) for i, f in enumerate(flags) if f)
    ideal_n = min(k, len(top)) if rel else 0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    ndcg = dcg / idcg if idcg else 0.0
    return {"recall_at_k": recall, "precision_at_k": precision, "mrr": mrr, "ndcg_at_k": ndcg}


def key_fact_coverage(answer: str, key_facts: list[str], threshold: float = 0.6) -> float:
    """Share of expected key facts whose content words mostly appear in the answer (cheap proxy, not a judge)."""
    if not key_facts:
        return 0.0
    a = set(tokenize(answer))
    a |= set(re.findall(r"\d+%?", answer))
    hit = 0
    for fact in key_facts:
        ft = set(tokenize(fact)) | set(re.findall(r"\d+%?", fact))
        if ft and len(ft & a) / len(ft) >= threshold:
            hit += 1
    return hit / len(key_facts)


def citation_paper_accuracy(citations: list[dict], rel_papers: list[str]) -> float | None:
    """Share of citations that point at a paper labelled relevant for the question."""
    if not citations or not rel_papers:
        return None
    return sum(1 for c in citations if c["paper_id"] in set(rel_papers)) / len(citations)


def claim_support_rate(claims: list[dict]) -> float | None:
    """Share of the answer's claims whose cited passage supports them (citations.verify_claims)."""
    scored = [c for c in claims if c.get("support") in {"supported", "weak", "unsupported", "no_citation"}]
    if not scored:
        return None
    return sum(1 for c in scored if c["support"] == "supported") / len(scored)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

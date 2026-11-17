"""Source formatting for prompts and citation verification for answers."""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from backend.guardrails import escape_source, neutralize
from data_ingestion.schemas import Chunk
from data_retrieval.store import tokenize

SID_RE = re.compile(r"\[(S\d+)\]")


def build_sources(chunks: list[Chunk], max_chars: int = 2200) -> tuple[str, dict[str, Chunk], list[str]]:
    """Returns (prompt_block, sid->chunk, flagged_sids). Flagged text is neutralised before it reaches the model."""
    lines, mapping, flagged = [], {}, []
    for i, c in enumerate(chunks, 1):
        sid = f"S{i}"
        mapping[sid] = c
        text, was_flagged = neutralize(c.text[:max_chars])
        if was_flagged or c.injection_suspected:
            flagged.append(sid)
        pages = f"{c.page_start}" if c.page_start == c.page_end else f"{c.page_start}-{c.page_end}"
        flag_attr = ' flagged="possible-injection"' if sid in flagged else ""
        section = escape_source(c.section).replace('"', "'")
        lines.append(
            f'<source id="{sid}" paper="{c.paper_id}" section="{section}" pages="{pages}" '
            f'type="{c.chunk_type}"{flag_attr}>\n'
            f"{escape_source(text)}\n</source>"
        )
    return "\n".join(lines), mapping, flagged


def citation_record(sid: str, c: Chunk, papers: dict[str, dict]) -> dict[str, Any]:
    meta = papers.get(c.paper_id, {})
    return {
        "sid": sid,
        "chunk_id": c.chunk_id,
        "paper_id": c.paper_id,
        "paper_title": c.paper_title,
        "short_name": meta.get("short_name", ""),
        "section": c.section,
        "page_start": c.page_start,
        "page_end": c.page_end,
        "chunk_type": c.chunk_type,
        "element_id": c.element_id,
        "label": c.citation_label(),
        "has_image": bool(c.image_path),
    }


def clean_answer_citations(answer: str, valid: set[str]) -> tuple[str, list[str]]:
    """Remove citation markers that point at sources the model was never given."""
    invalid = [m for m in SID_RE.findall(answer or "") if m not in valid]
    cleaned = SID_RE.sub(lambda m: m.group(0) if m.group(1) in valid else "", answer or "")
    return cleaned, invalid


def _content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if len(t) > 2}


def lexical_support(claim: str, source: str) -> float:
    ct = _content_tokens(claim)
    if not ct:
        return 0.0
    return len(ct & _content_tokens(source)) / len(ct)


def verify_claims(claims: list[dict], mapping: dict[str, Chunk], embedder=None, llm=None) -> list[dict]:
    """Score every (claim, cited source) pair.

    Default method (cheap, deterministic): max of lexical token recall and embedding cosine.
    Optional: an LLM judge (VERIFY_CITATIONS_WITH_LLM=true) - recorded separately because
    LLM judges are not ground truth.
    """
    results = []
    for cl in claims or []:
        text = str(cl.get("claim", ""))
        sids = [s for s in cl.get("sources", []) if s in mapping]
        best, per = 0.0, []
        for s in sids:
            lex = lexical_support(text, mapping[s].text)
            cos = 0.0
            if embedder is not None:
                a = np.asarray(embedder.embed_query(text))
                b = np.asarray(embedder.embed_documents([mapping[s].text[:2000]])[0])
                cos = float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))
            score = max(lex, cos)
            per.append({"sid": s, "lexical": round(lex, 3), "cosine": round(cos, 3)})
            best = max(best, score)
        status = "no_citation" if not sids else ("supported" if best >= 0.55 else "weak" if best >= 0.35 else "unsupported")
        results.append({**cl, "sources": sids, "support_score": round(best, 3), "support": status, "detail": per})

    if llm is not None and getattr(llm, "available", False):
        from backend.prompts import JUDGE_SUPPORT_PROMPT

        pairs, idx = [], []
        for i, r in enumerate(results):
            for s in r["sources"]:
                pairs.append(f"{len(pairs) + 1}. CLAIM: {r['claim']}\n   SOURCE: {mapping[s].text[:1200]}")
                idx.append(i)
        if pairs:
            try:
                verdicts = llm.generate_json(JUDGE_SUPPORT_PROMPT.format(pairs="\n".join(pairs))).get("verdicts", [])
                for i, v in zip(idx, verdicts):
                    results[i].setdefault("llm_verdicts", []).append(v)
            except Exception:
                pass
    return results

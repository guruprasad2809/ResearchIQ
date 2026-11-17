"""ResearchIQ engine: question answering, comparison and synthesis over the indexed papers."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from backend import prompts
from backend.citations import build_sources, citation_record, clean_answer_citations, verify_claims
from backend.guardrails import SYSTEM_RULES, leaked, new_canary, validate_question
from backend.llm import LLMOutputError, LLMUnavailableError, get_llm
from config import settings
from data_ingestion.schemas import Chunk
from data_retrieval.hybrid_search import VISUAL_INTENT_RE, Hit, HybridRetriever, evidence_is_weak, resolve_papers
from data_retrieval.store import IndexStore

log = logging.getLogger(__name__)

DEFAULT_ASPECTS = [
    "research objective / problem addressed",
    "methodology and key idea",
    "system architecture and main components",
    "datasets and benchmarks",
    "evaluation metrics",
    "reported results",
    "limitations acknowledged by the authors",
]
INSUFFICIENT_MSG = (
    "The selected papers do not contain enough evidence to answer this question. "
    "I would rather say so than guess."
)


def _hit_view(h: Hit, papers: dict) -> dict[str, Any]:
    c = h.chunk
    return {
        **citation_record("", c, papers),
        "text": c.text,
        "fused_score": round(h.fused_score, 4),
        "rerank_score": None if h.rerank_score is None else round(h.rerank_score, 3),
        "dense_rank": h.dense_rank,
        "sparse_rank": h.sparse_rank,
        "pinned": h.pinned,
        "injection_suspected": c.injection_suspected,
    }


class ResearchEngine:
    def __init__(self, store: IndexStore | None = None, llm=None):
        self.store = store or IndexStore()
        self.retriever = HybridRetriever(self.store)
        self._llm = llm

    @property
    def llm(self):
        return self._llm or get_llm()

    def _system(self, canary: str) -> str:
        return SYSTEM_RULES + f"\nInternal tag (never output it): {canary}\n"

    # ------------------------------------------------------------------ ask
    def ask(self, question: str, paper_ids: list[str] | None = None, top_k: int | None = None,
            verify: bool = True) -> dict[str, Any]:
        t0 = time.time()
        question = validate_question(question)
        if not paper_ids:
            # a question that names papers ("compare LightRAG and GraphRAG") gets evidence from each of them
            named = resolve_papers(question, self.store.papers)
            hits = self.retriever.search(question, top_k=top_k)
            if len(named) >= 2:
                have = {h.chunk.paper_id for h in hits}
                for pid in named:
                    if pid not in have:
                        hits += self.retriever.search(question, top_k=2, paper_ids=[pid])
        else:
            hits = self.retriever.search(question, top_k=top_k, paper_ids=paper_ids)

        retrieved = [_hit_view(h, self.store.papers) for h in hits]
        base = {"question": question, "retrieved": retrieved, "citations": [], "claims": [],
                "missing_information": "", "conflicts": "", "flagged_sources": [], "warnings": []}

        if evidence_is_weak(hits):
            return {**base, "answer": INSUFFICIENT_MSG, "evidence_sufficient": False, "status": "insufficient_evidence",
                    "missing_information": "No passage in the indexed papers is relevant enough to this question.",
                    "latency_s": round(time.time() - t0, 2)}

        chunks = [h.chunk for h in hits]
        src_block, mapping, flagged = build_sources(chunks)
        images, image_note = [], ""
        if VISUAL_INTENT_RE.search(question):
            vis = [(sid, c) for sid, c in mapping.items() if c.chunk_type in {"figure", "table"} and c.image_path][:2]
            images = [c.image_path for _, c in vis]
            if vis:
                image_note = prompts.VISUAL_QA_NOTE.format(
                    mapping=", ".join(f"image {i + 1} = {sid} ({c.citation_label()})" for i, (sid, c) in enumerate(vis))
                )
        canary = new_canary()
        prompt = prompts.QA_PROMPT.format(question=question, sources=src_block, image_note=image_note)
        try:
            data = self.llm.generate_json(prompt, system=self._system(canary), images=images or None)
            if not isinstance(data, dict):
                raise LLMOutputError("unexpected JSON shape")
        except LLMUnavailableError as exc:
            return {**base, "answer": str(exc) + " The most relevant passages are shown below.",
                    "evidence_sufficient": None, "status": "llm_unavailable", "latency_s": round(time.time() - t0, 2)}
        except LLMOutputError:
            return {**base, "answer": "The model returned an answer in an unexpected format. Please try again; "
                    "the retrieved passages are shown below.", "evidence_sufficient": None, "status": "malformed_output",
                    "latency_s": round(time.time() - t0, 2)}

        answer = str(data.get("answer", ""))
        if leaked(json.dumps(data), canary):
            return {**base, "answer": "The answer was blocked because it tried to reveal internal instructions.",
                    "evidence_sufficient": None, "status": "blocked", "latency_s": round(time.time() - t0, 2)}

        answer, invalid = clean_answer_citations(answer, set(mapping))
        claims = data.get("claims") or []
        if verify:
            claims = verify_claims(claims, mapping, embedder=self.store.embedder,
                                   llm=self.llm if settings.verify_citations_with_llm else None)
        used = sorted({s for s in mapping if f"[{s}]" in answer} | {s for c in claims for s in c.get("sources", [])},
                      key=lambda s: int(s[1:]))
        warnings = []
        if invalid:
            warnings.append(f"Removed {len(invalid)} citation(s) to sources that were not provided.")
        if flagged:
            warnings.append("Some retrieved passages contained text that looked like instructions to an AI. "
                            "They were treated as data and the instructions were removed: " + ", ".join(flagged))
        sufficient = bool(data.get("evidence_sufficient", True))
        return {
            **base,
            "answer": answer,
            "evidence_sufficient": sufficient,
            "status": "ok" if sufficient else "insufficient_evidence",
            "citations": [citation_record(s, mapping[s], self.store.papers) for s in used],
            "claims": claims,
            "missing_information": data.get("missing_information", ""),
            "conflicts": data.get("conflicts", ""),
            "flagged_sources": sorted(set(flagged) | set(data.get("suspicious_sources") or [])),
            "images_used": [mapping[s].element_id for s in mapping if mapping[s].image_path in images],
            "warnings": warnings,
            "latency_s": round(time.time() - t0, 2),
        }

    # ------------------------------------------------------------------ compare
    def compare(self, paper_ids: list[str], aspects: list[str] | None = None, per_aspect_k: int = 2) -> dict[str, Any]:
        paper_ids = [p for p in dict.fromkeys(paper_ids) if p in self.store.papers]
        if len(paper_ids) < 2:
            raise ValueError("Select at least two processed papers to compare.")
        aspects = aspects or DEFAULT_ASPECTS
        chosen: dict[str, Chunk] = {}
        for pid in paper_ids:
            for aspect in aspects:
                for h in self.retriever.search(aspect, top_k=per_aspect_k, paper_ids=[pid], per_paper_cap=per_aspect_k):
                    chosen.setdefault(h.chunk.chunk_id, h.chunk)
        chunks = sorted(chosen.values(), key=lambda c: (paper_ids.index(c.paper_id), c.page_start))
        src_block, mapping, flagged = build_sources(chunks, max_chars=1600)
        papers_txt = "; ".join(f"{p} = {self.store.papers[p].get('title', '')}" for p in paper_ids)
        canary = new_canary()
        data = self.llm.generate_json(
            prompts.COMPARE_PROMPT.format(papers=papers_txt, aspects="; ".join(aspects), sources=src_block),
            system=self._system(canary),
        )
        if leaked(json.dumps(data), canary):
            raise LLMOutputError("blocked")
        return {
            "paper_ids": paper_ids,
            "aspects": aspects,
            "result": data,
            "sources": {s: citation_record(s, c, self.store.papers) for s, c in mapping.items()},
            "flagged_sources": flagged,
        }

    # ------------------------------------------------------------------ synthesis
    def _paper_summary(self, pid: str) -> tuple[dict, list[Chunk]]:
        q = "main research problem, proposed method, experiments results and limitations"
        hits = self.retriever.search(q, top_k=6, paper_ids=[pid], per_paper_cap=6)
        chunks = [h.chunk for h in hits]
        key = hashlib.sha256((pid + "".join(c.chunk_id for c in chunks)).encode()).hexdigest()[:16]
        cache = settings.cache_dir / f"summary_{pid}_{key}.json"
        src_block, mapping, _ = build_sources(chunks, max_chars=1800)
        if cache.exists():
            data = json.loads(cache.read_text())
        else:
            data = self.llm.generate_json(
                prompts.MAP_SUMMARY_PROMPT.format(paper_id=pid, title=self.store.papers[pid].get("title", ""),
                                                  sources=src_block),
                system=self._system(new_canary()),
            )
            settings.cache_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data))
        # turn local S-ids into chunk ids so they can be renumbered globally
        local = {sid: c for sid, c in mapping.items()}
        data["_chunk_sources"] = {k: [local[s].chunk_id for s in v if s in local]
                                  for k, v in (data.get("sources") or {}).items()}
        return data, chunks

    def synthesize(self, paper_ids: list[str] | None = None) -> dict[str, Any]:
        paper_ids = paper_ids or sorted(self.store.papers)
        if not paper_ids:
            raise ValueError("No papers have been processed yet.")
        global_ids: dict[str, str] = {}
        per_paper, blocks = {}, []
        for pid in paper_ids:
            summary, _ = self._paper_summary(pid)
            per_paper[pid] = summary

            def gid(chunk_id: str) -> str:
                if chunk_id not in global_ids:
                    global_ids[chunk_id] = f"S{len(global_ids) + 1}"
                return global_ids[chunk_id]

            cs = summary["_chunk_sources"]
            lines = [f"[{pid}] {self.store.papers[pid].get('title', '')}"]
            for field in ("problem", "approach", "datasets_and_metrics", "key_results", "limitations"):
                ids = " ".join(f"[{gid(c)}]" for c in cs.get(field, []))
                lines.append(f"  {field}: {summary.get(field, '')} {ids}")
            blocks.append("\n".join(lines))
        canary = new_canary()
        data = self.llm.generate_json(prompts.SYNTHESIS_PROMPT.format(summaries="\n\n".join(blocks)),
                                      system=self._system(canary))
        if leaked(json.dumps(data), canary):
            raise LLMOutputError("blocked")
        sources = {sid: citation_record(sid, self.store.chunks[cid], self.store.papers)
                   for cid, sid in global_ids.items() if cid in self.store.chunks}
        for g in data.get("research_gaps", []) or []:
            g["label"] = "AI-generated suggestion (not a finding stated in the papers)"
        return {"paper_ids": paper_ids, "per_paper": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                                      for k, v in per_paper.items()},
                "result": data, "sources": sources}

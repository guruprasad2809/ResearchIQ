"""Semantic title-based chunking (same strategy as the Healthcare RAG app), extended for papers.

- A new chunk always starts at a section heading, so a chunk never mixes two sections.
- Long sections are split on sentence boundaries into ~CHUNK_MAX_WORDS pieces with a
  small word overlap; each chunk keeps its page range and section path.
- The section title and paper title are prefixed to the chunk text ("contextual" chunk
  header) so that both BM25 and the embedding see where the text comes from.
- Every figure and table becomes its own "visual chunk": caption + model description
  + labels found inside the graphic + the sentences in the paper that refer to it.
- The "References" section is dropped (it adds noise to retrieval).
"""
from __future__ import annotations

import re

from backend.guardrails import looks_like_injection
from data_ingestion.pdf_parser import XREF_RE
from data_ingestion.schemas import Chunk, ParsedDocument, VisualElement

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
STOP_SECTIONS = {"references", "bibliography"}


def _clean_heading(h: str) -> str:
    return re.sub(r"\s+", " ", h).strip()[:120]


def _refs(text: str) -> list[str]:
    out = []
    for m in XREF_RE.finditer(text):
        kind = "Table" if m.group(1).lower().startswith("tab") else "Figure"
        ref = f"{kind} {m.group(2)}"
        if ref not in out:
            out.append(ref)
    return out


class SemanticTitleChunker:
    def __init__(self, max_words: int = 320, overlap_words: int = 40):
        self.max_words = max_words
        self.overlap_words = overlap_words

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        sections = self._sections(doc)
        chunks: list[Chunk] = []
        for section, pieces in sections:
            chunks.extend(self._split_section(doc, section, pieces, len(chunks)))
        for v in doc.visuals:
            chunks.append(self._visual_chunk(doc, v))
        return chunks

    # (section title, [(page, text), ...])
    def _sections(self, doc: ParsedDocument) -> list[tuple[str, list[tuple[int, str]]]]:
        sections: list[tuple[str, list[tuple[int, str]]]] = []
        current, buf = "Front matter", []
        skipping = False
        for b in doc.blocks:
            if b.is_heading:
                if buf and not skipping:
                    sections.append((current, buf))
                current, buf = _clean_heading(b.text), []
                plain = re.sub(r"^(\d+(\.\d+)*|[A-H]|[IVX]+)\.?\s+", "", current).strip().lower()
                if plain in STOP_SECTIONS:
                    skipping = True
                elif skipping and (plain.startswith("appendix") or re.match(r"^[A-H](\.\d+)*\.?\s", current)):
                    skipping = False  # appendix after references is useful again
                continue
            if not skipping:
                buf.append((b.page, b.text))
        if buf and not skipping:
            sections.append((current, buf))
        return sections

    def _split_section(self, doc, section, pieces, start_idx) -> list[Chunk]:
        # sentence stream with page tags
        sents: list[tuple[int, str]] = []
        for page, text in pieces:
            for s in _SENT.split(text):
                if s.strip():
                    sents.append((page, s.strip()))
        chunks: list[Chunk] = []
        cur: list[tuple[int, str]] = []
        cur_words = 0

        def emit():
            if not cur:
                return
            body = " ".join(s for _, s in cur)
            if len(body.split()) < 12 and chunks:  # tiny tail -> merge into previous chunk
                prev = chunks[-1]
                prev.text += " " + body
                prev.page_end = max(prev.page_end, cur[-1][0])
                prev.figure_refs = list(dict.fromkeys(prev.figure_refs + _refs(body)))
                return
            idx = start_idx + len(chunks)
            header = f"[{doc.title} | {section}]"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.paper_id}_c{idx:04d}",
                    paper_id=doc.paper_id,
                    paper_title=doc.title,
                    text=f"{header}\n{body}",
                    chunk_type="text",
                    section=section,
                    page_start=cur[0][0],
                    page_end=cur[-1][0],
                    figure_refs=_refs(body),
                    injection_suspected=looks_like_injection(body),
                )
            )

        for page, s in sents:
            w = len(s.split())
            if cur and cur_words + w > self.max_words:
                emit()
                # overlap: carry the last sentences (up to overlap_words) into the next chunk
                carry, cw = [], 0
                for item in reversed(cur):
                    n = len(item[1].split())
                    if cw + n > self.overlap_words:
                        break
                    carry.insert(0, item)
                    cw += n
                cur, cur_words = carry, cw
            cur.append((page, s))
            cur_words += w
        emit()
        return chunks

    @staticmethod
    def _visual_chunk(doc: ParsedDocument, v: VisualElement) -> Chunk:
        label = "Figure" if v.kind == "figure" else "Table"
        parts = [f"[{doc.title} | {label} {v.number}, page {v.page}]", f"Caption: {v.caption}"]
        if v.table_markdown:
            parts.append("Table content (extracted):\n" + v.table_markdown)
        if v.description:
            parts.append(f"Visual description (model-generated, not from the paper text): {v.description}")
        if v.nearby_text:
            parts.append(f"Nearby text: {v.nearby_text}")
        for r in v.referenced_in:
            parts.append(f"Referenced on page {r['page']}: {r['text']}")
        text = "\n".join(parts)
        return Chunk(
            chunk_id=f"{v.element_id}",
            paper_id=doc.paper_id,
            paper_title=doc.title,
            text=text,
            chunk_type=v.kind,
            section=f"{label} {v.number}",
            page_start=v.page,
            page_end=v.page,
            element_id=v.element_id,
            figure_refs=[f"{label} {v.number}"],
            image_path=v.image_path,
            injection_suspected=looks_like_injection(text),
            metadata={
                "number": v.number,
                "caption": v.caption,
                "description_source": v.description_source,
                "legible": v.legible,
            },
        )

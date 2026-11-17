"""End-to-end ingestion: validate -> parse -> describe visuals -> chunk -> index."""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

from config import ROOT_DIR, settings
from data_ingestion.chunker import SemanticTitleChunker
from data_ingestion.figure_describer import describe_visuals
from data_ingestion.pdf_parser import PDFParser
from data_ingestion.validators import validate_pdf_bytes
from data_retrieval.store import IndexStore

log = logging.getLogger(__name__)
CATALOGUE_PATH = ROOT_DIR / "docs" / "paper_catalogue.csv"


def load_catalogue(path: Path = CATALOGUE_PATH) -> dict[str, dict[str, Any]]:
    """filename -> catalogue row"""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["filename"]: r for r in rows if r.get("filename")}


def next_paper_id(store: IndexStore) -> str:
    n = 1
    while f"U{n:02d}" in store.papers:
        n += 1
    return f"U{n:02d}"  # user-uploaded papers that are not in the catalogue


def ingest_pdf(pdf_path: Path, store: IndexStore, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    pdf_path = Path(pdf_path)
    t0 = time.time()
    validate_pdf_bytes(pdf_path.read_bytes(), pdf_path.name)
    meta = dict(meta or load_catalogue().get(pdf_path.name, {}))
    paper_id = meta.get("paper_id") or next_paper_id(store)

    parser = PDFParser(settings.figures_dir, enable_ocr=settings.enable_ocr)
    parsed = parser.parse(pdf_path, paper_id, title=meta.get("title") or None)
    describe_visuals(parsed.visuals, settings.cache_dir, enabled=settings.describe_figures)
    chunks = SemanticTitleChunker(settings.chunk_max_words, settings.chunk_overlap_words).chunk(parsed)

    aliases = [a.strip() for a in str(meta.get("aliases", "")).split(";") if a.strip()]
    paper_meta = {
        "paper_id": paper_id,
        "title": parsed.title,
        "short_name": meta.get("short_name", "") or parsed.title[:40],
        "aliases": aliases,
        "authors": meta.get("authors", ""),
        "year": meta.get("year", ""),
        "source_website": meta.get("source_website", "user upload"),
        "identifier": meta.get("identifier", ""),
        "file_name": pdf_path.name,
        "num_pages": parsed.num_pages,
        "num_chunks": sum(1 for c in chunks if c.chunk_type == "text"),
        "num_figures": sum(1 for v in parsed.visuals if v.kind == "figure"),
        "num_tables": sum(1 for v in parsed.visuals if v.kind == "table"),
        "figures_described": sum(1 for v in parsed.visuals if v.description_source == "gemini-vision"),
        "ocr_pages": parsed.ocr_pages,
        "warnings": parsed.warnings,
        "injection_flagged_chunks": sum(1 for c in chunks if c.injection_suspected),
        "ingest_seconds": round(time.time() - t0, 1),
    }
    store.add_paper(paper_meta, chunks)
    log.info("Ingested %s as %s: %s chunks, %s figures, %s tables", pdf_path.name, paper_id,
             paper_meta["num_chunks"], paper_meta["num_figures"], paper_meta["num_tables"])
    return paper_meta

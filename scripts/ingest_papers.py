"""Ingest every PDF in data/papers/ (catalogue metadata is attached by file name).

Usage:  python scripts/ingest_papers.py [--with-injection-test]
--with-injection-test also builds and ingests the adversarial test document (paper id T99).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from data_ingestion.pipeline import ingest_pdf, load_catalogue  # noqa: E402
from data_retrieval.store import IndexStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-injection-test", action="store_true")
    args = ap.parse_args()
    settings.ensure_dirs()
    store = IndexStore()
    catalogue = load_catalogue()
    pdfs = sorted(settings.papers_dir.glob("*.pdf"))
    if args.with_injection_test:
        from scripts.make_injection_pdf import build

        pdfs.append(build(settings.data_dir / "test_docs" / "T99_injection_test.pdf"))
    if not pdfs:
        print("No PDFs found. Run: python scripts/download_papers.py")
        return
    for pdf in pdfs:
        meta = catalogue.get(pdf.name)
        if pdf.name.startswith("T99"):
            meta = {"paper_id": "T99", "short_name": "Injection test doc", "title": "Prompt-injection test document",
                    "source_website": "synthetic test file", "aliases": "injection test"}
        try:
            m = ingest_pdf(pdf, store, meta)
            print(f"OK  {m['paper_id']:4} {m['num_pages']:3} pages  {m['num_chunks']:3} text chunks  "
                  f"{m['num_figures']:2} figures  {m['num_tables']:2} tables  ({m['ingest_seconds']}s)  {pdf.name}")
            for w in m["warnings"]:
                print("    warning:", w)
        except Exception as exc:
            print(f"ERR {pdf.name}: {exc}")


if __name__ == "__main__":
    main()

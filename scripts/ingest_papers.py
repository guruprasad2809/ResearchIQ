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
    ap.add_argument("--with-scanned", metavar="PAPER_ID",
                    help="also ingest an image-only (scanned) copy of this catalogue paper to demo OCR, e.g. P05")
    args = ap.parse_args()
    settings.ensure_dirs()
    store = IndexStore()
    catalogue = load_catalogue()
    pdfs = sorted(settings.papers_dir.glob("*.pdf"))
    if args.with_injection_test:
        from scripts.make_injection_pdf import build

        pdfs.append(build(settings.data_dir / "test_docs" / "T99_injection_test.pdf"))
    if args.with_scanned:
        from data_ingestion.pdf_parser import PDFParser
        from scripts.make_scanned_pdf import make_scanned

        if not PDFParser.ocr_available():
            print("OCR is not available: install Tesseract (and set TESSERACT_CMD on Windows). Skipping scanned copy.")
        else:
            src = next((p for p in pdfs if p.name.startswith(args.with_scanned)), None)
            if src is None:
                print(f"No PDF starting with {args.with_scanned} in {settings.papers_dir}")
            else:
                pdfs.append(make_scanned(src, settings.data_dir / "test_docs" / f"SCAN_{src.name}", dpi=150, pages="1-4"))
    if not pdfs:
        print("No PDFs found. Run: python scripts/download_papers.py")
        return
    for pdf in pdfs:
        meta = catalogue.get(pdf.name)
        if pdf.name.startswith("SCAN_"):
            base = catalogue.get(pdf.name[5:], {})
            meta = {"paper_id": "S" + base.get("paper_id", "00")[1:], "short_name": "Scanned " + base.get("short_name", ""),
                    "title": "[Scanned copy] " + base.get("title", pdf.stem), "source_website": "image-only copy for OCR test",
                    "aliases": "scanned copy"}
        if pdf.name.startswith("T99"):
            meta = {"paper_id": "T99", "short_name": "Injection test doc", "title": "Prompt-injection test document",
                    "source_website": "synthetic test file", "aliases": "injection test"}
        try:
            m = ingest_pdf(pdf, store, meta)
            print(f"OK  {m['paper_id']:4} {m['num_pages']:3} pages  {m['num_chunks']:3} text chunks  "
                  f"{m['num_figures']:2} figures  {m['num_tables']:2} tables  ({m['ingest_seconds']}s)  {pdf.name}")
            if m["ocr_pages"]:
                print(f"    OCR used on pages: {m['ocr_pages']}")
            for w in m["warnings"]:
                print("    warning:", w)
        except Exception as exc:
            print(f"ERR {pdf.name}: {exc}")


if __name__ == "__main__":
    main()

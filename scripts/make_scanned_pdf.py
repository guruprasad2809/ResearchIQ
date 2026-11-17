"""Make a 'scanned' copy of a PDF: every page becomes a picture with NO text layer.

Use it to test and demo the OCR path:
    python scripts/make_scanned_pdf.py data/papers/P05_lightrag.pdf --pages 1-3
    -> data/test_docs/SCAN_P05_lightrag.pdf   (then upload it in the app, or run ingest_papers.py --with-scanned)

Options: --dpi (image quality, default 150), --pages (e.g. "1-3" or "2,5"), --out (output path).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf


def parse_pages(spec: str | None, n: int) -> list[int]:
    if not spec:
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a) - 1, min(int(b), n)))
        else:
            out.append(int(part) - 1)
    return [p for p in out if 0 <= p < n]


def make_scanned(src: Path, dest: Path, dpi: int = 150, pages: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(src) as doc, pymupdf.open() as out:
        for pno in parse_pages(pages, doc.page_count):
            page = doc[pno]
            pix = page.get_pixmap(dpi=dpi)
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, pixmap=pix)  # image only -> no extractable text
        out.save(dest)
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--pages", default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    dest = a.out or Path("data/test_docs") / f"SCAN_{a.pdf.stem}.pdf"
    print(make_scanned(a.pdf, dest, a.dpi, a.pages))

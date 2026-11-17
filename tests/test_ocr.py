"""OCR path: a page with no text layer is read with Tesseract and keeps its layout.
Skipped automatically when Tesseract is not installed."""
import pytest

from config import settings
from data_ingestion.chunker import SemanticTitleChunker
from data_ingestion.pdf_parser import PDFParser
from scripts.make_scanned_pdf import make_scanned

pytestmark = pytest.mark.skipif(not PDFParser.ocr_available(), reason="Tesseract is not installed")


@pytest.fixture(scope="module")
def scanned_parsed(sample_pdfs, tmp_dir):
    scan = make_scanned(sample_pdfs[0], tmp_dir / "SCAN_A.pdf", dpi=200)
    return PDFParser(settings.figures_dir, enable_ocr=True).parse(scan, "S01", title="GraphWeave Retrieval")


def test_scanned_copy_really_has_no_text(sample_pdfs, tmp_dir):
    import pymupdf

    scan = make_scanned(sample_pdfs[0], tmp_dir / "SCAN_B.pdf", dpi=100)
    with pymupdf.open(scan) as d:
        assert all(len(p.get_text().strip()) == 0 for p in d)


def test_ocr_reads_every_page_with_page_numbers(scanned_parsed):
    assert scanned_parsed.ocr_pages == [1, 2, 3]
    text = " ".join(b.text for b in scanned_parsed.blocks)
    assert "GraphWeave" in text and "community summaries" in text
    assert {b.page for b in scanned_parsed.blocks} >= {1, 2}


def test_ocr_keeps_layout_headings_columns_and_captions(scanned_parsed):
    headings = [b.text for b in scanned_parsed.blocks if b.is_heading]
    assert any("Introduction" in h for h in headings) and any("Method" in h for h in headings)
    p1 = [b.text for b in scanned_parsed.blocks if b.page == 1]
    left = next(i for i, t in enumerate(p1) if "LEFTCOL" in t)
    right = next(i for i, t in enumerate(p1) if "RIGHTCOL" in t)
    assert left < right
    kinds = {(v.kind, v.number) for v in scanned_parsed.visuals}
    assert ("figure", "1") in kinds and ("table", "1") in kinds


def test_ocr_chunks_are_citable(scanned_parsed):
    chunks = SemanticTitleChunker().chunk(scanned_parsed)
    assert chunks and all(c.page_start >= 1 for c in chunks)
    assert not any("Some reference about retrieval" in c.text for c in chunks if c.chunk_type == "text")


def test_ocr_off_gives_warning_not_crash(sample_pdfs, tmp_dir):
    scan = make_scanned(sample_pdfs[0], tmp_dir / "SCAN_C.pdf", dpi=100)
    parsed = PDFParser(settings.figures_dir, enable_ocr=False).parse(scan, "S02")
    assert parsed.ocr_pages == [] and any("OCR is unavailable" in w for w in parsed.warnings)

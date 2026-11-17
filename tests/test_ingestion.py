from pathlib import Path

import pytest

from config import settings
from data_ingestion.chunker import SemanticTitleChunker
from data_ingestion.pdf_parser import PDFParser
from data_ingestion.validators import FileValidationError, safe_filename, validate_pdf_bytes


@pytest.fixture(scope="module")
def parsed(sample_pdfs):
    return PDFParser(settings.figures_dir, enable_ocr=False).parse(sample_pdfs[0], "P01", title="GraphWeave Retrieval")


def test_page_numbers_and_headings(parsed):
    assert parsed.num_pages == 3
    headings = [b.text for b in parsed.blocks if b.is_heading]
    assert "Abstract" in headings and "1 Introduction" in headings and "2 Method" in headings
    assert {b.page for b in parsed.blocks} >= {1, 2}


def test_two_column_reading_order(parsed):
    texts = [b.text for b in parsed.blocks if b.page == 1]
    left = next(i for i, t in enumerate(texts) if t.startswith("LEFTCOL"))
    right = next(i for i, t in enumerate(texts) if t.startswith("RIGHTCOL"))
    assert left < right


def test_figure_and_table_extracted_with_caption_page_and_image(parsed):
    kinds = {(v.kind, v.number): v for v in parsed.visuals}
    fig = kinds[("figure", "1")]
    tab = kinds[("table", "1")]
    assert fig.page == 2 and "architecture" in fig.caption
    assert Path(fig.image_path).exists()
    assert tab.page == 2 and "HotpotQA" in tab.caption
    # labels drawn inside the figure are linked to the figure, not left in body text
    assert "Indexer" in fig.nearby_text
    # the in-text reference on page 1 is linked back to the figure
    assert any(r["page"] == 1 for r in fig.referenced_in)


def test_chunker_respects_sections_and_drops_references(parsed):
    chunks = SemanticTitleChunker(max_words=80, overlap_words=10).chunk(parsed)
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    assert all(c.section != "References" for c in text_chunks)
    assert not any("Some reference about retrieval" in c.text for c in text_chunks)
    assert all(c.page_start <= c.page_end for c in chunks)
    assert any(c.chunk_type == "figure" and c.element_id == "P01_fig_1" for c in chunks)
    assert any(c.chunk_type == "table" for c in chunks)
    intro = [c for c in text_chunks if c.section == "1 Introduction"]
    assert intro and any("Figure 1" in c.figure_refs for c in intro)
    assert all(c.text.startswith("[GraphWeave Retrieval | ") for c in text_chunks)


def test_validator_rejects_bad_files(sample_pdfs):
    with pytest.raises(FileValidationError):
        validate_pdf_bytes(b"hello", "notes.txt")
    with pytest.raises(FileValidationError):
        validate_pdf_bytes(b"not a pdf at all", "fake.pdf")
    with pytest.raises(FileValidationError):
        validate_pdf_bytes(b"%PDF-1.7\n garbage garbage", "broken.pdf")
    assert validate_pdf_bytes(Path(sample_pdfs[0]).read_bytes(), "A.pdf") == 3


def test_safe_filename_blocks_path_traversal():
    assert safe_filename("../../etc/passwd") == "passwd.pdf"
    assert "/" not in safe_filename("..\\..\\x.pdf")

"""Offline test setup: hash embedder, in-memory Qdrant, fake LLM. No network or API key needed."""
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="researchiq_test_")
os.environ.update({"EMBEDDING_BACKEND": "hash", "GEMINI_API_KEY": "", "DATA_DIR": _TMP, "USE_RERANKER": "false",
                   "DESCRIBE_FIGURES": "false", "ENABLE_OCR": "false"})
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402
import re  # noqa: E402

import pymupdf  # noqa: E402
import pytest  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from config import settings  # noqa: E402

settings.ensure_dirs()

PARA_A = (
    "We propose GraphWeave, a retrieval method that builds an entity graph from the corpus and summarises "
    "communities of related entities. The community summaries are used to answer global questions. "
) * 3
PARA_B = (
    "Dense retrieval embeds every chunk with a bi-encoder and ranks chunks by cosine similarity. "
    "BM25 ranks chunks by keyword overlap and term frequency. "
) * 3


def make_paper(path: Path, title: str, method: str, results_value: str, two_column: bool = True) -> Path:
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 60), title, fontsize=17)
    p1.insert_text((72, 95), "Abstract", fontsize=12)
    p1.insert_textbox(pymupdf.Rect(72, 105, 540, 190), f"{method} " + PARA_A, fontsize=9)
    p1.insert_text((72, 215), "1 Introduction", fontsize=12)
    if two_column:
        p1.insert_textbox(pymupdf.Rect(72, 225, 295, 700), "LEFTCOL " + PARA_B + " As shown in Figure 1, the pipeline has three stages.", fontsize=9)
        p1.insert_textbox(pymupdf.Rect(315, 225, 540, 700), "RIGHTCOL " + PARA_A, fontsize=9)
    else:
        p1.insert_textbox(pymupdf.Rect(72, 225, 540, 700), PARA_B + " As shown in Figure 1, the pipeline has three stages.", fontsize=9)
    p2 = doc.new_page()
    p2.insert_text((72, 60), "2 Method", fontsize=12)
    p2.draw_rect(pymupdf.Rect(120, 90, 480, 260), color=(0, 0, 0), fill=(0.9, 0.9, 1))
    p2.draw_line((150, 175), (450, 175))
    p2.insert_text((160, 170), "Indexer", fontsize=8)
    p2.insert_text((380, 170), "Generator", fontsize=8)
    p2.insert_textbox(pymupdf.Rect(72, 268, 540, 300), f"Figure 1: Overview of the {method} architecture with indexer and generator.", fontsize=9)
    p2.insert_textbox(pymupdf.Rect(72, 310, 540, 420), PARA_B, fontsize=9)
    p2.insert_text((72, 445), "3 Experiments", fontsize=12)
    p2.insert_textbox(pymupdf.Rect(72, 455, 540, 480), "Table 1: Main results on the HotpotQA benchmark.", fontsize=9)
    y = 490
    for row in (["Method", "Recall@5", "EM"], [method, results_value, "41.2"], ["BM25", "55.0", "30.1"]):
        p2.insert_text((90, y), "        ".join(row), fontsize=9)
        y += 14
    p2.insert_text((72, 600), "4 Limitations", fontsize=12)
    p2.insert_textbox(pymupdf.Rect(72, 610, 540, 680), f"{method} is expensive to index because every chunk needs an LLM call. "
                      "We only evaluate on English benchmarks.", fontsize=9)
    p3 = doc.new_page()
    p3.insert_text((72, 60), "References", fontsize=12)
    p3.insert_textbox(pymupdf.Rect(72, 75, 540, 300), "[1] Some reference about retrieval systems. 2020. " * 6, fontsize=9)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture(scope="session")
def tmp_dir():
    return Path(_TMP)


@pytest.fixture(scope="session")
def sample_pdfs(tmp_dir):
    d = tmp_dir / "pdfs"
    d.mkdir(exist_ok=True)
    a = make_paper(d / "A.pdf", "GraphWeave Retrieval", "GraphWeave", "78.5")
    b = make_paper(d / "B.pdf", "DenseFlow Retrieval", "DenseFlow", "64.0", two_column=False)
    return a, b


class FakeLLM:
    """Returns canned JSON; records prompts so tests can inspect what the model would see."""

    available = True
    model = "fake"

    def __init__(self, response=None):
        self.prompts = []
        self.systems = []
        self.response = response

    def generate_json(self, prompt, system="", images=None, model=None):
        self.prompts.append(prompt)
        self.systems.append(system)
        if callable(self.response):
            return self.response(prompt, system)
        if self.response is not None:
            return json.loads(json.dumps(self.response))
        # cite the source that really contains the fact, like a well-behaved model would
        sid = "S1"
        for m in re.finditer(r'<source id="(S\d+)"[^>]*>(.*?)</source>', prompt, re.S):
            if "entity graph" in m.group(2):
                sid = m.group(1)
                break
        return {"evidence_sufficient": True, "answer": f"GraphWeave builds an entity graph [{sid}].",
                "claims": [{"claim": "GraphWeave builds an entity graph from the corpus", "sources": [sid],
                            "kind": "paper_evidence"}], "missing_information": "", "conflicts": "", "suspicious_sources": []}


@pytest.fixture()
def store(tmp_path):
    from data_retrieval.embedder import HashEmbedder
    from data_retrieval.store import IndexStore

    return IndexStore(index_dir=tmp_path / "index", qdrant_client=QdrantClient(":memory:"), embedder=HashEmbedder())


@pytest.fixture()
def loaded_store(store, sample_pdfs):
    from data_ingestion.pipeline import ingest_pdf

    a, b = sample_pdfs
    ingest_pdf(a, store, {"paper_id": "P01", "short_name": "GraphWeave", "title": "GraphWeave Retrieval", "aliases": "graphweave"})
    ingest_pdf(b, store, {"paper_id": "P02", "short_name": "DenseFlow", "title": "DenseFlow Retrieval", "aliases": "denseflow"})
    return store

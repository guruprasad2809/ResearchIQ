"""Central configuration. Every secret and setting comes from environment variables
(.env is loaded for local development). Nothing secret is hard-coded here."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional
    pass

ROOT_DIR = Path(__file__).resolve().parent


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- LLM (Gemini) ---
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.5-flash"))
    gemini_vision_model: str = field(
        default_factory=lambda: _env("GEMINI_VISION_MODEL", _env("GEMINI_MODEL", "gemini-2.5-flash"))
    )
    llm_timeout_s: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT_S", 60))
    llm_max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 3))
    # requests per minute allowed by your Gemini plan (free tier is ~15). 0 = no pacing.
    llm_rpm: int = field(default_factory=lambda: _env_int("LLM_RPM", 14))
    # longest wait (seconds) we accept when Gemini says "retry in N s" after a 429
    llm_max_wait_s: int = field(default_factory=lambda: _env_int("LLM_MAX_WAIT_S", 90))

    # --- Embeddings / reranker ---
    # "qwen" = Qwen/Qwen3-Embedding-0.6B via sentence-transformers; "hash" = offline test embedder
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "qwen"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"))
    reranker_model: str = field(
        default_factory=lambda: _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
    )
    use_reranker: bool = field(default_factory=lambda: _env_bool("USE_RERANKER", True))

    # --- Qdrant ---
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL"))  # empty = local file mode
    qdrant_api_key: str = field(default_factory=lambda: _env("QDRANT_API_KEY"))
    qdrant_path: str = field(default_factory=lambda: _env("QDRANT_PATH", str(ROOT_DIR / "data" / "qdrant")))
    qdrant_collection: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", "researchiq_chunks"))

    # --- Storage ---
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT_DIR / "data"))))

    # --- Retrieval knobs ---
    dense_top_k: int = field(default_factory=lambda: _env_int("DENSE_TOP_K", 30))
    sparse_top_k: int = field(default_factory=lambda: _env_int("SPARSE_TOP_K", 30))
    rerank_top_n: int = field(default_factory=lambda: _env_int("RERANK_TOP_N", 24))
    final_top_k: int = field(default_factory=lambda: _env_int("FINAL_TOP_K", 8))
    max_chunks_per_paper: int = field(default_factory=lambda: _env_int("MAX_CHUNKS_PER_PAPER", 4))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    # Below this cross-encoder score the evidence is treated as "not relevant"
    min_rerank_score: float = field(default_factory=lambda: _env_float("MIN_RERANK_SCORE", -4.0))

    # --- Chunking ---
    chunk_max_words: int = field(default_factory=lambda: _env_int("CHUNK_MAX_WORDS", 320))
    chunk_overlap_words: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_WORDS", 40))

    # --- Upload limits ---
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 40))
    max_pages: int = field(default_factory=lambda: _env_int("MAX_PAGES", 120))

    # --- Features ---
    describe_figures: bool = field(default_factory=lambda: _env_bool("DESCRIBE_FIGURES", True))
    enable_ocr: bool = field(default_factory=lambda: _env_bool("ENABLE_OCR", True))
    # OCR (Tesseract). TESSERACT_CMD is only needed when tesseract is not on PATH (common on Windows)
    tesseract_cmd: str = field(default_factory=lambda: _env("TESSERACT_CMD"))
    ocr_lang: str = field(default_factory=lambda: _env("OCR_LANG", "eng"))
    ocr_dpi: int = field(default_factory=lambda: _env_int("OCR_DPI", 300))
    ocr_min_conf: int = field(default_factory=lambda: _env_int("OCR_MIN_CONF", 40))
    # a page with fewer extractable characters than this is treated as scanned
    ocr_text_threshold: int = field(default_factory=lambda: _env_int("OCR_TEXT_THRESHOLD", 40))
    # force OCR on every page (for testing / PDFs with a broken text layer)
    ocr_force: bool = field(default_factory=lambda: _env_bool("OCR_FORCE", False))
    verify_citations_with_llm: bool = field(default_factory=lambda: _env_bool("VERIFY_CITATIONS_WITH_LLM", False))

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def figures_dir(self) -> Path:
        return self.data_dir / "figures"

    @property
    def papers_dir(self) -> Path:
        return self.data_dir / "papers"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.index_dir, self.figures_dir, self.papers_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()

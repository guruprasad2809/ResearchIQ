"""Cross-encoder reranker: cross-encoder/ms-marco-MiniLM-L6-v2 (same as the Healthcare app)."""
from __future__ import annotations

from functools import lru_cache

from config import settings


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=512)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts], batch_size=16)]


@lru_cache(maxsize=1)
def get_reranker():
    if not settings.use_reranker or settings.embedding_backend == "hash":
        return None
    try:
        return CrossEncoderReranker(settings.reranker_model)
    except Exception:  # model not downloadable -> hybrid search still works without reranking
        return None

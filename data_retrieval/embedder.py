"""Embedding models.

QwenEmbedder  - Qwen/Qwen3-Embedding-0.6B via sentence-transformers (same as the Healthcare app).
                Queries use the model's "query" prompt; documents are embedded as-is.
HashEmbedder  - deterministic, dependency-free embedder used by the offline test-suite
                (EMBEDDING_BACKEND=hash). Not for real use.
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache

import numpy as np

from config import settings


class QwenEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False))

    def embed_query(self, text: str) -> np.ndarray:
        try:
            v = self.model.encode([text], prompt_name="query", normalize_embeddings=True)
        except (KeyError, ValueError):
            v = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(v[0])


class HashEmbedder:
    """Bag-of-words feature hashing (unigrams + bigrams), L2-normalised."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = re.findall(r"[a-z0-9]+", text.lower())
        feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        n = math.sqrt(float((v * v).sum())) or 1.0
        return v / n

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


@lru_cache(maxsize=1)
def get_embedder():
    if settings.embedding_backend == "hash":
        return HashEmbedder()
    return QwenEmbedder(settings.embedding_model)

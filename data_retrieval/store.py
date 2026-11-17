"""Index storage.

- chunks.jsonl  : source of truth for chunk text + metadata (used for evidence view & BM25)
- papers.json   : paper catalogue entries for processed papers
- Qdrant        : dense vectors (Qwen3 embeddings) with paper_id / chunk_type payload for filtering
- BM25          : built in memory from chunks.jsonl at start-up (fast for tens of papers;
                  at thousands of papers switch to Qdrant sparse vectors / OpenSearch)
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from config import settings
from data_ingestion.schemas import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_STOP = set(
    "a an the of to in and or for on with by is are was were be been as at from that this these those it its "
    "we our their which what how does do did can not no into than then also using use used".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class IndexStore:
    def __init__(self, index_dir: Path | None = None, qdrant_client=None, embedder=None):
        self.index_dir = Path(index_dir or settings.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_file = self.index_dir / "chunks.jsonl"
        self.papers_file = self.index_dir / "papers.json"
        self._lock = threading.RLock()
        if embedder is None:
            from data_retrieval.embedder import get_embedder

            embedder = get_embedder()
        self.embedder = embedder
        self.client = qdrant_client or self._make_client()
        self.collection = settings.qdrant_collection
        self.chunks: dict[str, Chunk] = {}
        self.papers: dict[str, dict[str, Any]] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._load()

    @staticmethod
    def _make_client():
        from qdrant_client import QdrantClient

        if settings.qdrant_url:
            return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=settings.qdrant_path)

    def _ensure_collection(self):
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection, vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE)
            )
            if settings.qdrant_url:  # payload indexes only matter on a Qdrant server
                for field in ("paper_id", "chunk_type"):
                    self.client.create_payload_index(self.collection, field, PayloadSchemaType.KEYWORD)

    # ---------------------------------------------------------------- persistence
    def _load(self):
        if self.chunks_file.exists():
            for line in self.chunks_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    c = Chunk.from_dict(json.loads(line))
                    self.chunks[c.chunk_id] = c
        if self.papers_file.exists():
            self.papers = json.loads(self.papers_file.read_text(encoding="utf-8"))
        self._rebuild_bm25()

    def _save(self):
        tmp = self.chunks_file.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for c in self.chunks.values():
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        tmp.replace(self.chunks_file)
        self.papers_file.write_text(json.dumps(self.papers, indent=2, ensure_ascii=False), encoding="utf-8")

    def _rebuild_bm25(self):
        self._bm25_ids = list(self.chunks.keys())
        corpus = [tokenize(self.chunks[i].text) for i in self._bm25_ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    # ---------------------------------------------------------------- writes
    def add_paper(self, paper_meta: dict[str, Any], chunks: list[Chunk]) -> None:
        from qdrant_client.models import PointStruct

        with self._lock:
            self._ensure_collection()
            self.delete_paper(paper_meta["paper_id"], save=False)
            vecs = self.embedder.embed_documents([c.text for c in chunks])
            points = [
                PointStruct(
                    id=_point_id(c.chunk_id),
                    vector=np.asarray(v, dtype=float).tolist(),
                    payload={"chunk_id": c.chunk_id, "paper_id": c.paper_id, "chunk_type": c.chunk_type},
                )
                for c, v in zip(chunks, vecs)
            ]
            for i in range(0, len(points), 128):
                self.client.upsert(self.collection, points=points[i : i + 128])
            for c in chunks:
                self.chunks[c.chunk_id] = c
            self.papers[paper_meta["paper_id"]] = paper_meta
            self._rebuild_bm25()
            self._save()

    def delete_paper(self, paper_id: str, save: bool = True) -> None:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        with self._lock:
            ids = [cid for cid, c in self.chunks.items() if c.paper_id == paper_id]
            if self.client.collection_exists(self.collection):
                self.client.delete(
                    self.collection,
                    points_selector=FilterSelector(
                        filter=Filter(must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))])
                    ),
                )
            for cid in ids:
                self.chunks.pop(cid, None)
            self.papers.pop(paper_id, None)
            if save:
                self._rebuild_bm25()
                self._save()

    # ---------------------------------------------------------------- reads
    def dense_search(self, query: str, top_k: int, paper_ids: list[str] | None = None,
                     chunk_types: list[str] | None = None) -> list[tuple[str, float]]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        if not self.chunks or not self.client.collection_exists(self.collection):
            return []
        must = []
        if paper_ids:
            must.append(FieldCondition(key="paper_id", match=MatchAny(any=list(paper_ids))))
        if chunk_types:
            must.append(FieldCondition(key="chunk_type", match=MatchAny(any=list(chunk_types))))
        qv = np.asarray(self.embedder.embed_query(query), dtype=float).tolist()
        res = self.client.query_points(
            self.collection, query=qv, limit=top_k, query_filter=Filter(must=must) if must else None,
            with_payload=True,
        )
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points if p.payload.get("chunk_id") in self.chunks]

    def sparse_search(self, query: str, top_k: int, paper_ids: list[str] | None = None,
                      chunk_types: list[str] | None = None) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1]
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            c = self.chunks[self._bm25_ids[i]]
            if paper_ids and c.paper_id not in paper_ids:
                continue
            if chunk_types and c.chunk_type not in chunk_types:
                continue
            out.append((c.chunk_id, float(scores[i])))
            if len(out) >= top_k:
                break
        return out

    def get(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def visuals(self, paper_id: str | None = None) -> list[Chunk]:
        return [
            c for c in self.chunks.values()
            if c.chunk_type in {"figure", "table"} and (paper_id is None or c.paper_id == paper_id)
        ]

    def find_visual(self, paper_id: str, kind: str, number: str) -> Chunk | None:
        for c in self.visuals(paper_id):
            if c.chunk_type == kind and str(c.metadata.get("number", "")).lower() == number.lower():
                return c
        return None

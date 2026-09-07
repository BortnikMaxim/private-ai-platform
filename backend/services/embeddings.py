"""Embedding and reranking models.

The heavy sentence-transformers models are loaded once per process and reused
by every request. Importing this module does not import torch: the model
libraries are imported lazily inside :meth:`EmbeddingService.load`.
"""

import asyncio
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, embedding_model: str, reranker_model: str) -> None:
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model

        self._encoder: Any = None
        self._reranker: Any = None
        # A threading.Lock rather than an asyncio.Lock on purpose: a Celery
        # worker runs each task in its own `asyncio.run` loop, and an
        # asyncio.Lock binds to the first loop that uses it. This keeps one
        # instance reusable across loops, and across threads.
        self._lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._encoder is not None and self._reranker is not None

    def load(self) -> None:
        """Blocking, thread safe model load. Call through :meth:`ensure_loaded`."""
        with self._lock:
            if self.is_loaded:
                return

            from sentence_transformers import CrossEncoder, SentenceTransformer

            if self._encoder is None:
                logger.info("loading embedding model: %s", self.embedding_model_name)
                self._encoder = SentenceTransformer(self.embedding_model_name)

            if self._reranker is None:
                logger.info("loading reranker model: %s", self.reranker_model_name)
                self._reranker = CrossEncoder(self.reranker_model_name)

            logger.info("embedding and reranker models are ready")

    async def ensure_loaded(self) -> None:
        if self.is_loaded:
            return

        # Concurrent callers all reach the thread, but load() itself only runs
        # the import and construction once.
        await asyncio.to_thread(self.load)

    # -- embeddings ------------------------------------------------------

    def _encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._encoder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in embeddings]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        await self.ensure_loaded()
        # e5 models expect the "passage: " / "query: " instruction prefixes.
        prefixed = [f"passage: {text}" for text in texts]
        return await asyncio.to_thread(self._encode, prefixed)

    async def embed_query(self, question: str) -> list[float]:
        await self.ensure_loaded()
        vectors = await asyncio.to_thread(self._encode, [f"query: {question}"])
        return vectors[0]

    # -- reranking -------------------------------------------------------

    def _predict(self, pairs: list[list[str]]) -> list[float]:
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        return [float(score) for score in scores]

    async def rerank(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Score candidates with the cross-encoder and keep the best ``top_k``."""
        if not candidates:
            return []

        await self.ensure_loaded()

        pairs = [[question, candidate.get("text") or ""] for candidate in candidates]
        scores = await asyncio.to_thread(self._predict, pairs)

        reranked: list[dict[str, Any]] = []

        for candidate, rerank_score in zip(candidates, scores, strict=True):
            enriched = dict(candidate)
            enriched.setdefault("vector_score", candidate.get("score"))
            enriched["rerank_score"] = rerank_score
            reranked.append(enriched)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

        return reranked[:top_k]

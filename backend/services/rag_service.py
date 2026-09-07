"""Retrieval orchestration: embed -> vector search -> rerank -> context."""

import logging
from typing import Any

from backend.config import Settings
from backend.prompts import NO_CONTEXT_PLACEHOLDER
from backend.services.embeddings import EmbeddingService
from backend.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RagService:
    def __init__(
        self,
        embeddings: EmbeddingService,
        vector_store: VectorStore,
        settings: Settings,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.settings = settings

    async def vector_retrieve(
        self,
        question: str,
        limit: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        vector = await self.embeddings.embed_query(question)

        return await self.vector_store.search(
            vector=vector,
            limit=limit or self.settings.rag_candidate_k,
            document_ids=document_ids,
        )

    async def rerank(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        return await self.embeddings.rerank(
            question=question,
            candidates=candidates,
            top_k=top_k or self.settings.rag_top_k,
        )

    async def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Full retrieval pipeline used by both /rag/ask and conversations."""
        candidates = await self.vector_retrieve(
            question=question,
            limit=candidate_k or self.settings.rag_candidate_k,
            document_ids=document_ids,
        )

        logger.info(
            "rag_retrieved candidates=%d scoped=%s",
            len(candidates),
            bool(document_ids),
        )

        return await self.rerank(
            question=question,
            candidates=candidates,
            top_k=top_k or self.settings.rag_top_k,
        )

    def build_context(self, chunks: list[dict[str, Any]]) -> str:
        """Render retrieved chunks as a numbered, size-bounded context block."""
        if not chunks:
            return NO_CONTEXT_PLACEHOLDER

        budget = self.settings.max_context_chars
        parts: list[str] = []
        used = 0

        for index, chunk in enumerate(chunks, start=1):
            part = (
                f"[SOURCE {index}]\n"
                f"File: {chunk.get('filename')}\n"
                f"Page: {chunk.get('page')}\n"
                f"Text:\n{chunk.get('text') or ''}"
            )

            if used + len(part) > budget:
                # Keep whole chunks; drop the ones that no longer fit.
                if not parts:
                    parts.append(part[:budget])
                break

            parts.append(part)
            used += len(part) + 2

        return "\n\n".join(parts)

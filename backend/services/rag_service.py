"""Retrieval orchestration.

    query ─┬─ dense  (multilingual-e5 → Qdrant, tenant filter in the engine)
           └─ lexical (BM25 over document_chunks, tenant filter in SQL)
                ↓
            RRF fusion
                ↓
          CrossEncoder rerank
                ↓
             top_k chunks

``dense`` mode skips the lexical branch and the fusion step entirely, which is
the pipeline this project had before hybrid retrieval existed.
"""

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Settings
from backend.prompts import NO_CONTEXT_PLACEHOLDER
from backend.services.embeddings import EmbeddingService
from backend.services.fusion import reciprocal_rank_fusion
from backend.services.lexical_index import LexicalRetriever
from backend.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

MODE_DENSE = "dense"
MODE_HYBRID = "hybrid"
RETRIEVAL_MODES = (MODE_DENSE, MODE_HYBRID)


class RagService:
    def __init__(
        self,
        embeddings: EmbeddingService,
        vector_store: VectorStore,
        settings: Settings,
        lexical: LexicalRetriever | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.settings = settings
        self.lexical = lexical

    # -- branches ---------------------------------------------------------

    async def vector_retrieve(
        self,
        question: str,
        user_id: str,
        limit: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Tenant-scoped dense search.

        ``user_id`` has no default on purpose: every call site must supply it,
        and it always comes from the authenticated principal, never from a
        request body.
        """
        vector = await self.embeddings.embed_query(question)

        return await self.vector_store.search(
            vector=vector,
            limit=limit or self.settings.rag_candidate_k,
            user_id=user_id,
            document_ids=document_ids,
        )

    async def lexical_retrieve(
        self,
        session: AsyncSession,
        question: str,
        user_id: str,
        limit: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Tenant-scoped BM25 search over the stored chunks."""
        if self.lexical is None:
            return []

        return await self.lexical.search(
            session=session,
            question=question,
            user_id=user_id,
            limit=limit or self.settings.rag_lexical_candidate_k,
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

    # -- pipeline ---------------------------------------------------------

    def resolve_mode(self, mode: str | None, session: AsyncSession | None) -> str:
        """Pick the effective retrieval mode.

        Hybrid needs a database session for the lexical branch. Rather than
        silently returning half a pipeline, a hybrid request without a session
        degrades to dense and says so in the log.
        """
        requested = (mode or self.settings.retrieval_mode or MODE_DENSE).lower()

        if requested not in RETRIEVAL_MODES:
            logger.warning("unknown retrieval mode %r; falling back to dense", requested)
            return MODE_DENSE

        if requested == MODE_HYBRID and (session is None or self.lexical is None):
            logger.warning(
                "hybrid retrieval requested without a %s; using dense only",
                "database session" if session is None else "lexical index",
            )
            return MODE_DENSE

        return requested

    async def retrieve(
        self,
        question: str,
        user_id: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        document_ids: list[str] | None = None,
        session: AsyncSession | None = None,
        mode: str | None = None,
        lexical_k: int | None = None,
        rrf_k: int | None = None,
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        """Full retrieval pipeline used by /rag/ask, conversations and the agent."""
        effective = self.resolve_mode(mode, session)
        dense_limit = candidate_k or self.settings.rag_candidate_k

        dense = await self.vector_retrieve(
            question=question,
            user_id=user_id,
            limit=dense_limit,
            document_ids=document_ids,
        )

        if effective == MODE_HYBRID:
            lexical = await self.lexical_retrieve(
                session=session,
                question=question,
                user_id=user_id,
                limit=lexical_k or self.settings.rag_lexical_candidate_k,
                document_ids=document_ids,
            )
            pool = reciprocal_rank_fusion(
                [dense, lexical],
                rrf_k=rrf_k or self.settings.rag_rrf_k,
            )
            logger.info(
                "rag_retrieved mode=hybrid dense=%d lexical=%d fused=%d scoped=%s",
                len(dense),
                len(lexical),
                len(pool),
                bool(document_ids),
            )
        else:
            pool = dense
            logger.info(
                "rag_retrieved mode=dense candidates=%d scoped=%s",
                len(pool),
                bool(document_ids),
            )

        # The reranker is the expensive stage, so it only ever sees a bounded
        # slice of the fused pool.
        pool = pool[: self.settings.rag_rerank_candidate_k]

        if not rerank:
            return pool[: top_k or self.settings.rag_top_k]

        return await self.rerank(
            question=question,
            candidates=pool,
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

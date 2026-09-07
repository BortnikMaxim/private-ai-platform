"""Per-process resources for a Celery worker.

The embedding and reranker models are the expensive part, so they live in a
process global and are reused by every task. Everything that is bound to an
event loop — the SQLAlchemy engine and the Qdrant client — is created and
disposed per task, because each task runs in its own ``asyncio.run`` loop and
loop-bound objects must not outlive it.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from backend.config import Settings
from backend.db import create_engine
from backend.services.document_processor import DocumentProcessor
from backend.services.embeddings import EmbeddingService
from backend.services.storage import DocumentStorage
from backend.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

_embeddings: EmbeddingService | None = None


def get_embeddings(settings: Settings) -> EmbeddingService:
    """The process-wide model holder; models load lazily on first use."""
    global _embeddings

    if _embeddings is None:
        logger.info("worker_embedding_service_created")
        _embeddings = EmbeddingService(
            embedding_model=settings.embedding_model,
            reranker_model=settings.reranker_model,
        )

    return _embeddings


def reset_embeddings() -> None:
    """Forget the cached models (called after a prefork)."""
    global _embeddings
    _embeddings = None


@asynccontextmanager
async def worker_resources(
    settings: Settings,
) -> AsyncIterator[tuple[async_sessionmaker, DocumentProcessor]]:
    engine = create_engine(settings.database_url)
    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)

    try:
        session_factory = async_sessionmaker(
            bind=engine,
            expire_on_commit=False,
            autoflush=False,
        )

        processor = DocumentProcessor(
            embeddings=get_embeddings(settings),
            vector_store=VectorStore(
                client=qdrant_client,
                collection_name=settings.qdrant_collection,
                vector_size=settings.embedding_dim,
            ),
            storage=DocumentStorage(settings.upload_dir),
            settings=settings,
        )

        yield session_factory, processor

    finally:
        await qdrant_client.close()
        await engine.dispose()

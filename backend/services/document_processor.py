"""The heavy half of ingestion: read -> parse -> chunk -> embed -> index.

Runs inside a Celery worker (and, for the deprecated synchronous upload
endpoint, inline in the request). It is deliberately written against a
``session_factory`` rather than a single session: embedding a large PDF takes
minutes, and holding one PostgreSQL transaction open for that long would pin a
connection and block vacuum.

Idempotency
-----------
Re-running the task for the same document must not duplicate anything, which
matters because ``task_acks_late`` re-delivers tasks from a crashed worker.
Two mechanisms combine:

1. Qdrant point ids are deterministic: ``uuid5(namespace, "<doc>:<index>")``.
   Re-processing upserts the *same* ids, so vectors are overwritten in place.
2. Every run deletes the document's existing vectors and chunk rows before it
   writes new ones, so a re-run that produces fewer chunks leaves no stale
   leftovers behind.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker

from backend.config import Settings
from backend.errors import (
    AppError,
    DocumentGoneError,
    DocumentSourceMissingError,
    InvalidDocumentError,
    TransientProcessingError,
)
from backend.models import Document, DocumentChunk
from backend.observability import (
    DOCUMENT_PROCESSING_DURATION_SECONDS,
    DOCUMENT_PROCESSING_FAILURES_TOTAL,
    DOCUMENTS_PROCESSING_TOTAL,
    stage,
)
from backend.services.chunking import build_chunks, extract_pdf_pages
from backend.services.embeddings import EmbeddingService
from backend.services.storage import DocumentStorage
from backend.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Fixed namespace so point ids stay stable across processes and restarts.
POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

# Connection level problems only. Broader bases such as DBAPIError would also
# match ProgrammingError, and retrying a schema bug is pointless.
TRANSIENT_DB_ERRORS = (OperationalError, InterfaceError)


def point_id_for(document_id: uuid.UUID | str, chunk_index: int) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{document_id}:{chunk_index}"))


class DocumentProcessor:
    def __init__(
        self,
        embeddings: EmbeddingService,
        vector_store: VectorStore,
        storage: DocumentStorage,
        settings: Settings,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.storage = storage
        self.settings = settings

    # -- public API ------------------------------------------------------

    async def process(
        self,
        session_factory: async_sessionmaker,
        document_id: uuid.UUID,
        task_id: str | None = None,
    ) -> Document:
        """Ingest one document. Raises on failure; never marks status itself."""
        DOCUMENTS_PROCESSING_TOTAL.inc()
        started = time.perf_counter()
        outcome = "failed"

        try:
            document = await self._process(session_factory, document_id, task_id)
            outcome = "ready"
            return document
        except DocumentGoneError:
            # Neither of these is a failure: one was deleted on purpose, the
            # other is a duplicate delivery of an already finished job.
            outcome = "deleted"
            raise
        except AlreadyProcessedError:
            outcome = "skipped"
            raise
        finally:
            DOCUMENT_PROCESSING_DURATION_SECONDS.labels(outcome=outcome).observe(
                time.perf_counter() - started
            )

    async def _process(
        self,
        session_factory: async_sessionmaker,
        document_id: uuid.UUID,
        task_id: str | None,
    ) -> Document:
        key = str(document_id)

        with stage("load", key, task_id) as info:
            filename = await self._claim(session_factory, document_id)
            info["filename_len"] = len(filename)

        with stage("read_source", key, task_id) as info:
            file_bytes = await self.storage.read(document_id)
            info["size_bytes"] = len(file_bytes)

        with stage("parse", key, task_id) as info:
            total_pages, pages = await asyncio.to_thread(extract_pdf_pages, file_bytes)
            info["total_pages"] = total_pages
            info["extracted_pages"] = len(pages)

        with stage("chunk", key, task_id) as info:
            records = build_chunks(
                pages,
                chunk_size=self.settings.chunk_size_words,
                overlap=self.settings.chunk_overlap_words,
            )
            info["chunks"] = len(records)

        if not records:
            raise InvalidDocumentError("No text could be extracted from the PDF")

        # Cheap guard before the expensive part: bail out early if the document
        # was deleted while the PDF was being parsed.
        await self._assert_still_wanted(session_factory, document_id)

        with stage("embed", key, task_id) as info:
            vectors = await self.embeddings.embed_passages(
                [record["text"] for record in records]
            )
            info["vectors"] = len(vectors)

        points = [
            {
                "point_id": point_id_for(document_id, record["chunk_index"]),
                "vector": vector,
                "document_id": key,
                "filename": filename,
                "page": record["page"],
                "chunk_index": record["chunk_index"],
                "text": record["text"],
            }
            for record, vector in zip(records, vectors, strict=True)
        ]

        await self._assert_still_wanted(session_factory, document_id)

        with stage("index", key, task_id) as info:
            # Clear first so a re-run that yields fewer chunks leaves nothing
            # stale; the deterministic ids then overwrite the rest in place.
            await self._vector_call(self.vector_store.delete_document, key)
            await self._vector_call(self.vector_store.upsert_chunks, points)
            info["points"] = len(points)

        with stage("persist", key, task_id) as info:
            document = await self._save_results(
                session_factory,
                document_id,
                points=points,
                total_pages=total_pages,
                extracted_pages=len(pages),
            )
            info["chunks"] = len(points)

        if self.settings.delete_source_after_processing:
            with stage("cleanup_source", key, task_id):
                await self.storage.safe_delete(document_id)

        logger.info(
            "document_processed document_id=%s task_id=%s pages=%d chunks=%d",
            key,
            task_id,
            len(pages),
            len(points),
        )

        return document

    # -- phases ----------------------------------------------------------

    async def _claim(
        self,
        session_factory: async_sessionmaker,
        document_id: uuid.UUID,
    ) -> str:
        """Verify the document exists and is still queued; return its filename."""
        async with self._session(session_factory) as session:
            document = await session.get(Document, document_id)

            if document is None:
                raise DocumentGoneError()

            if document.status != "processing":
                # A completed document is not re-processed by an accidental
                # duplicate delivery.
                raise AlreadyProcessedError(
                    f"Document is already in status '{document.status}'"
                )

            return document.filename

    async def _assert_still_wanted(
        self,
        session_factory: async_sessionmaker,
        document_id: uuid.UUID,
    ) -> None:
        async with self._session(session_factory) as session:
            status = await session.scalar(
                select(Document.status).where(Document.id == document_id)
            )

        if status is None:
            raise DocumentGoneError()

        if status != "processing":
            raise AlreadyProcessedError(f"Document moved to status '{status}'")

    async def _save_results(
        self,
        session_factory: async_sessionmaker,
        document_id: uuid.UUID,
        points: list[dict[str, Any]],
        total_pages: int,
        extracted_pages: int,
    ) -> Document:
        async with self._session(session_factory) as session:
            document = await session.get(Document, document_id)

            if document is None:
                # Deleted while we were indexing: undo the vectors we just
                # wrote so the deletion stays complete.
                raise DocumentGoneError()

            # Idempotency: drop any chunk rows from an earlier attempt.
            await session.execute(
                delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
            )

            session.add_all(
                [
                    DocumentChunk(
                        document_id=document_id,
                        qdrant_point_id=point["point_id"],
                        page=point["page"],
                        chunk_index=point["chunk_index"],
                        text=point["text"],
                    )
                    for point in points
                ]
            )

            document.status = "ready"
            document.total_pages = total_pages
            document.extracted_pages = extracted_pages
            document.chunks_count = len(points)
            document.error_message = None

            await session.commit()
            await session.refresh(document)

            return document

    # -- failure handling -------------------------------------------------

    async def mark_failed(
        self,
        session_factory: async_sessionmaker,
        document_id: uuid.UUID,
        message: str,
    ) -> None:
        """Record a permanent failure and clean up whatever was written."""
        DOCUMENT_PROCESSING_FAILURES_TOTAL.labels(reason=_reason_label(message)).inc()

        try:
            async with self._session(session_factory) as session:
                await session.execute(
                    update(Document)
                    .where(Document.id == document_id)
                    .values(status="failed", error_message=message[:1000])
                )
                await session.commit()
        except Exception:
            logger.exception("document_status_update_failed document_id=%s", document_id)

        await self.cleanup(session_factory, document_id)

    async def cleanup(
        self,
        session_factory: async_sessionmaker,
        document_id: uuid.UUID,
        drop_chunks: bool = True,
    ) -> None:
        """Remove partial vectors, chunk rows and the source file."""
        try:
            await self.vector_store.delete_document(str(document_id))
        except Exception:
            logger.exception("vector_cleanup_failed document_id=%s", document_id)

        if drop_chunks:
            try:
                async with self._session(session_factory) as session:
                    await session.execute(
                        delete(DocumentChunk).where(
                            DocumentChunk.document_id == document_id
                        )
                    )
                    await session.commit()
            except Exception:
                logger.exception("chunk_cleanup_failed document_id=%s", document_id)

        await self.storage.safe_delete(document_id)

    # -- helpers ----------------------------------------------------------

    def _session(self, session_factory: async_sessionmaker):
        return session_factory()

    async def _vector_call(self, call, *args):
        """Qdrant hiccups are worth retrying; surface them as transient."""
        try:
            return await call(*args)
        except AppError:
            raise
        except Exception as exc:
            raise TransientProcessingError(
                f"Vector store call failed: {type(exc).__name__}"
            ) from exc


class AlreadyProcessedError(AppError):
    """A duplicate delivery for a document that is no longer queued."""

    status_code = 409
    default_detail = "Document is not awaiting processing"


def is_transient(exc: BaseException) -> bool:
    """Classify an exception for the Celery retry policy."""
    if isinstance(exc, TransientProcessingError):
        return True

    if isinstance(
        exc,
        InvalidDocumentError
        | DocumentSourceMissingError
        | DocumentGoneError
        | AlreadyProcessedError,
    ):
        return False

    return isinstance(exc, TRANSIENT_DB_ERRORS)


def safe_error_message(exc: BaseException) -> str:
    """A message safe to store in the database: never a traceback."""
    if isinstance(exc, AppError):
        return exc.detail

    return f"Processing failed ({type(exc).__name__})"


def _reason_label(message: str) -> str:
    lowered = message.lower()

    if "no text" in lowered or "pdf" in lowered:
        return "invalid_document"
    if "no longer" in lowered:
        return "source_missing"
    if "vector" in lowered:
        return "vector_store"
    if "retr" in lowered:
        return "retries_exhausted"

    return "internal"

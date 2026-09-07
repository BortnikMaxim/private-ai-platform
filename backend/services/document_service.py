"""Document ingestion: validate -> parse -> chunk -> embed -> store."""

import asyncio
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.config import Settings
from backend.errors import (
    AppError,
    DocumentNotFoundError,
    FileTooLargeError,
    InvalidDocumentError,
    VectorStoreError,
)
from backend.models import Document, DocumentChunk
from backend.services.chunking import build_chunks, extract_pdf_pages
from backend.services.embeddings import EmbeddingService
from backend.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf"}
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",
    "binary/octet-stream",
    "",
}
PDF_MAGIC = b"%PDF"
# \w is unicode-aware, so Cyrillic and other non-ASCII names survive intact;
# only path separators and control characters are replaced.
_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w.\- ]+")


def sanitize_filename(name: str) -> str:
    stem = Path(name).name
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", stem).strip()
    return (cleaned or "document.pdf")[:255]


class DocumentService:
    def __init__(
        self,
        embeddings: EmbeddingService,
        vector_store: VectorStore,
        settings: Settings,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.settings = settings

    # -- queries ---------------------------------------------------------

    async def list_documents(
        self,
        session: AsyncSession,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Document], int]:
        total = await session.scalar(select(func.count()).select_from(Document)) or 0

        result = await session.execute(
            select(Document)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        return list(result.scalars().all()), total

    async def get_document(
        self,
        session: AsyncSession,
        document_id: uuid.UUID,
        with_chunks: bool = False,
    ) -> Document:
        query = select(Document).where(Document.id == document_id)

        if with_chunks:
            query = query.options(selectinload(Document.chunks))

        document = (await session.execute(query)).scalar_one_or_none()

        if document is None:
            raise DocumentNotFoundError()

        return document

    async def existing_document_ids(
        self,
        session: AsyncSession,
        document_ids: list[uuid.UUID],
    ) -> list[uuid.UUID]:
        if not document_ids:
            return []

        result = await session.execute(
            select(Document.id).where(Document.id.in_(document_ids))
        )
        return list(result.scalars().all())

    # -- upload ----------------------------------------------------------

    async def read_upload(self, upload: UploadFile) -> bytes:
        """Validate the upload and read it with a hard size limit."""
        filename = upload.filename or ""

        if not filename:
            raise InvalidDocumentError("Filename is required")

        if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise InvalidDocumentError("Only PDF files are supported")

        content_type = (upload.content_type or "").lower().split(";")[0].strip()

        if content_type not in ALLOWED_CONTENT_TYPES:
            raise InvalidDocumentError(
                f"Unsupported content type: {content_type or 'unknown'}"
            )

        max_bytes = self.settings.max_upload_size_bytes
        buffer = bytearray()

        while True:
            chunk = await upload.read(1024 * 1024)

            if not chunk:
                break

            buffer.extend(chunk)

            if len(buffer) > max_bytes:
                raise FileTooLargeError(
                    f"File exceeds the {self.settings.max_upload_size_mb} MB limit"
                )

        if not buffer:
            raise InvalidDocumentError("Uploaded file is empty")

        if bytes(buffer[: len(PDF_MAGIC)]) != PDF_MAGIC:
            raise InvalidDocumentError("File does not look like a PDF")

        return bytes(buffer)

    async def ingest(
        self,
        session: AsyncSession,
        upload: UploadFile,
    ) -> Document:
        file_bytes = await self.read_upload(upload)
        original_filename = upload.filename or "document.pdf"

        document = Document(
            filename=sanitize_filename(original_filename),
            original_filename=original_filename[:255],
            content_type=(upload.content_type or "application/pdf")[:127],
            size_bytes=len(file_bytes),
            status="processing",
        )

        session.add(document)
        await session.commit()
        await session.refresh(document)

        document_id = document.id
        logger.info(
            "document_ingest_started document_id=%s size_bytes=%d",
            document_id,
            len(file_bytes),
        )

        try:
            return await self._process(session, document, file_bytes)

        except AppError as exc:
            await self._mark_failed(session, document_id, exc.detail)
            raise

        except Exception as exc:
            # Never leak internals to the caller; the traceback goes to the log.
            logger.exception("document_ingest_failed document_id=%s", document_id)
            await self._mark_failed(session, document_id, "Internal ingestion error")
            raise AppError("Document ingestion failed") from exc

    async def _process(
        self,
        session: AsyncSession,
        document: Document,
        file_bytes: bytes,
    ) -> Document:
        total_pages, pages = await asyncio.to_thread(extract_pdf_pages, file_bytes)

        records = build_chunks(
            pages,
            chunk_size=self.settings.chunk_size_words,
            overlap=self.settings.chunk_overlap_words,
        )

        if not records:
            raise InvalidDocumentError("No text could be extracted from the PDF")

        vectors = await self.embeddings.embed_passages(
            [record["text"] for record in records]
        )

        document_id = str(document.id)
        point_records: list[dict[str, Any]] = []

        for record, vector in zip(records, vectors, strict=True):
            point_records.append(
                {
                    "point_id": str(uuid.uuid4()),
                    "vector": vector,
                    "document_id": document_id,
                    "filename": document.filename,
                    "page": record["page"],
                    "chunk_index": record["chunk_index"],
                    "text": record["text"],
                }
            )

        await self.vector_store.upsert_chunks(point_records)

        try:
            session.add_all(
                [
                    DocumentChunk(
                        document_id=document.id,
                        qdrant_point_id=record["point_id"],
                        page=record["page"],
                        chunk_index=record["chunk_index"],
                        text=record["text"],
                    )
                    for record in point_records
                ]
            )

            document.status = "ready"
            document.total_pages = total_pages
            document.extracted_pages = len(pages)
            document.chunks_count = len(point_records)
            document.error_message = None

            await session.commit()

        except Exception:
            # Roll the vectors back so Qdrant never outlives its metadata.
            await session.rollback()
            await self._safe_delete_vectors(document_id)
            raise

        await session.refresh(document)

        logger.info(
            "document_ingest_finished document_id=%s pages=%d chunks=%d",
            document_id,
            len(pages),
            len(point_records),
        )

        return document

    async def _mark_failed(
        self,
        session: AsyncSession,
        document_id: uuid.UUID,
        error_message: str,
    ) -> None:
        await session.rollback()

        try:
            await session.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(status="failed", error_message=error_message[:1000])
            )
            await session.commit()
        except Exception:
            logger.exception(
                "document_status_update_failed document_id=%s", document_id
            )
            await session.rollback()

        await self._safe_delete_vectors(str(document_id))

    async def _safe_delete_vectors(self, document_id: str) -> None:
        try:
            await self.vector_store.delete_document(document_id)
        except Exception:
            logger.exception("vector_cleanup_failed document_id=%s", document_id)

    # -- delete ----------------------------------------------------------

    async def delete_document(
        self,
        session: AsyncSession,
        document_id: uuid.UUID,
    ) -> None:
        document = await self.get_document(session, document_id)

        # Remove the vectors first: orphaned rows are recoverable, orphaned
        # vectors would keep showing up in retrieval results.
        try:
            await self.vector_store.delete_document(str(document.id))
        except Exception as exc:
            logger.exception("vector_delete_failed document_id=%s", document_id)
            raise VectorStoreError(
                "Could not remove document vectors; document was not deleted"
            ) from exc

        # DocumentChunk rows go away through ON DELETE CASCADE.
        await session.delete(document)
        await session.commit()

        logger.info("document_deleted document_id=%s", document_id)

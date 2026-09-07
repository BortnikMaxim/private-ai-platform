"""Document CRUD and the synchronous half of ingestion.

The API only does cheap work here — validate the upload, persist the bytes and
record a ``processing`` row. Parsing, embedding and indexing happen in a Celery
worker; see :mod:`backend.services.document_processor`.
"""

import logging
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import UploadFile
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.config import Settings
from backend.errors import (
    DocumentNotFoundError,
    FileTooLargeError,
    InvalidDocumentError,
    VectorStoreError,
)
from backend.models import Document
from backend.services.storage import DocumentStorage
from backend.services.vector_store import VectorStore

if TYPE_CHECKING:
    from backend.services.task_queue import TaskDispatcher

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
        vector_store: VectorStore,
        storage: DocumentStorage,
        settings: Settings,
    ) -> None:
        self.vector_store = vector_store
        self.storage = storage
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

    async def create_pending(
        self,
        session: AsyncSession,
        upload: UploadFile,
    ) -> Document:
        """Validate and persist an upload, then record it as ``processing``.

        Everything here is cheap and bounded: the request never waits for
        parsing or embedding. The heavy work is picked up by a Celery worker.
        """
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

        try:
            await self.storage.save(document.id, file_bytes)
        except Exception:
            # Without its source file the document could never be processed,
            # so do not leave a permanently stuck row behind.
            logger.exception("document_source_store_failed document_id=%s", document.id)
            await session.delete(document)
            await session.commit()
            raise

        logger.info(
            "document_accepted document_id=%s size_bytes=%d",
            document.id,
            len(file_bytes),
        )

        return document

    async def attach_task(
        self,
        session: AsyncSession,
        document_id: uuid.UUID,
        task_id: str,
    ) -> None:
        await session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(celery_task_id=task_id)
        )
        await session.commit()

    # -- delete ----------------------------------------------------------

    async def delete_document(
        self,
        session: AsyncSession,
        document_id: uuid.UUID,
        dispatcher: "TaskDispatcher | None" = None,
    ) -> None:
        """Delete a document, whether or not a worker is still processing it.

        Ordering matters. The row is removed *last* but its disappearance is
        what stops an in-flight task: every heavy stage re-checks that the
        document still exists, and the final persist step refuses to write
        results for a row that is gone. Revoking the task is best effort on top
        of that — a worker that already dequeued the job ignores a revoke.
        """
        document = await self.get_document(session, document_id)
        was_processing = document.status == "processing"
        task_id = document.celery_task_id

        if dispatcher is not None and was_processing and task_id:
            dispatcher.revoke(task_id)

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

        await self.storage.safe_delete(document_id)

        logger.info(
            "document_deleted document_id=%s was_processing=%s task_id=%s",
            document_id,
            was_processing,
            task_id,
        )

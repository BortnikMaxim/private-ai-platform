"""Local filesystem storage for uploaded documents.

Large PDFs never travel through PostgreSQL or a RabbitMQ message: the API
writes the bytes here and the Celery task is handed only a document id.

Path traversal is structurally impossible rather than filtered: the filename is
built from a parsed ``uuid.UUID``, so no caller-supplied string ever reaches the
filesystem. The resolved parent is still checked as defence in depth.
"""

import asyncio
import logging
import uuid
from pathlib import Path

from backend.errors import DocumentSourceMissingError

logger = logging.getLogger(__name__)

SUFFIX = ".pdf"


class DocumentStorage:
    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()

    def ensure_ready(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, document_id: uuid.UUID | str) -> Path:
        # uuid.UUID() rejects anything containing a separator, a dot segment or
        # a null byte, so the resulting name is always a bare 36-char id.
        identifier = uuid.UUID(str(document_id))
        path = (self.base_dir / f"{identifier}{SUFFIX}").resolve()

        if path.parent != self.base_dir:
            raise ValueError(f"Refusing to use a path outside {self.base_dir}")

        return path

    def exists(self, document_id: uuid.UUID | str) -> bool:
        return self.path_for(document_id).is_file()

    # -- IO --------------------------------------------------------------

    def _write(self, path: Path, data: bytes) -> None:
        # Write to a temporary neighbour and rename, so a crash mid-write never
        # leaves a half-written PDF that a worker would then try to parse.
        temporary = path.with_suffix(f"{SUFFIX}.part")
        temporary.write_bytes(data)
        temporary.replace(path)

    async def save(self, document_id: uuid.UUID | str, data: bytes) -> Path:
        path = self.path_for(document_id)

        await asyncio.to_thread(self.ensure_ready)
        await asyncio.to_thread(self._write, path, data)

        logger.info(
            "document_source_stored document_id=%s size_bytes=%d",
            document_id,
            len(data),
        )
        return path

    async def read(self, document_id: uuid.UUID | str) -> bytes:
        path = self.path_for(document_id)

        try:
            return await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as exc:
            raise DocumentSourceMissingError(
                "The uploaded file for this document is no longer on disk"
            ) from exc

    async def delete(self, document_id: uuid.UUID | str) -> bool:
        path = self.path_for(document_id)

        def _unlink() -> bool:
            try:
                path.unlink()
                return True
            except FileNotFoundError:
                return False

        removed = await asyncio.to_thread(_unlink)

        if removed:
            logger.info("document_source_deleted document_id=%s", document_id)

        return removed

    async def safe_delete(self, document_id: uuid.UUID | str) -> bool:
        """Delete without ever raising; used on cleanup paths."""
        try:
            return await self.delete(document_id)
        except Exception:
            logger.exception("document_source_delete_failed document_id=%s", document_id)
            return False

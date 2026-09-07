"""Celery tasks.

A Celery task is a synchronous entrypoint, so each one opens its own event loop
with ``asyncio.run``. The FastAPI loop is never touched, and no loop-bound
object is shared between tasks.

Error policy
------------
*Permanent* failures (corrupt PDF, no extractable text, missing source file)
are a normal business outcome: the task records ``status="failed"`` with a safe
message and succeeds, so the broker is not filled with retries that can never
pass. *Transient* failures (Qdrant unreachable, dropped database connection)
propagate and Celery retries them with exponential backoff and jitter; when the
retries are exhausted ``on_failure`` marks the document failed.
"""

import asyncio
import logging
import uuid
from typing import Any

from celery import Task
from sqlalchemy.exc import InterfaceError, OperationalError

from backend.config import Settings, get_settings
from backend.errors import DocumentGoneError, TransientProcessingError
from backend.services.document_processor import (
    AlreadyProcessedError,
    is_transient,
    safe_error_message,
)
from backend.worker.celery_app import celery_app
from backend.worker.context import worker_resources

logger = logging.getLogger(__name__)

settings = get_settings()

RETRYABLE = (TransientProcessingError, OperationalError, InterfaceError)


class DocumentTask(Task):
    """Marks the document failed when Celery gives up on it."""

    def on_failure(self, exc, task_id, args, kwargs, einfo) -> None:
        document_id = (args[0] if args else kwargs.get("document_id")) or None

        if document_id is None:
            return

        logger.error(
            "document_task_failed document_id=%s task_id=%s error=%s",
            document_id,
            task_id,
            type(exc).__name__,
        )

        try:
            asyncio.run(
                _mark_failed(
                    get_settings(),
                    uuid.UUID(str(document_id)),
                    safe_error_message(exc),
                )
            )
        except Exception:
            logger.exception(
                "document_failure_bookkeeping_failed document_id=%s", document_id
            )


@celery_app.task(
    bind=True,
    base=DocumentTask,
    name="documents.process",
    autoretry_for=RETRYABLE,
    retry_backoff=settings.celery_retry_backoff_seconds,
    retry_backoff_max=settings.celery_retry_backoff_max_seconds,
    retry_jitter=True,
    max_retries=settings.celery_max_retries,
)
def process_document_task(self, document_id: str) -> dict[str, Any]:
    return asyncio.run(
        run_document_processing(
            get_settings(),
            uuid.UUID(str(document_id)),
            task_id=self.request.id,
        )
    )


async def run_document_processing(
    settings: Settings,
    document_id: uuid.UUID,
    task_id: str | None = None,
) -> dict[str, Any]:
    """The task body, importable and testable without a broker."""
    async with worker_resources(settings) as (session_factory, processor):
        return await process_with(processor, session_factory, document_id, task_id)


async def process_with(
    processor,
    session_factory,
    document_id: uuid.UUID,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Run one ingestion attempt against already built resources."""
    key = str(document_id)

    try:
        document = await processor.process(session_factory, document_id, task_id)

    except DocumentGoneError:
        # Deleted while we were working. Undo whatever was written so the
        # deletion stays complete; the document must not come back to life.
        logger.info("document_deleted_during_processing document_id=%s", key)
        await processor.cleanup(session_factory, document_id)
        return {"document_id": key, "status": "deleted"}

    except AlreadyProcessedError as exc:
        # Duplicate delivery of a job that already finished.
        logger.info("document_task_skipped document_id=%s reason=%s", key, exc.detail)
        return {"document_id": key, "status": "skipped", "reason": exc.detail}

    except Exception as exc:
        if is_transient(exc):
            raise

        message = safe_error_message(exc)
        logger.exception("document_processing_failed document_id=%s", key)
        await processor.mark_failed(session_factory, document_id, message)
        return {"document_id": key, "status": "failed", "error": message}

    return {
        "document_id": key,
        "status": document.status,
        "chunks": document.chunks_count,
        "extracted_pages": document.extracted_pages,
    }


async def _mark_failed(
    settings: Settings,
    document_id: uuid.UUID,
    message: str,
) -> None:
    async with worker_resources(settings) as (session_factory, processor):
        await processor.mark_failed(session_factory, document_id, message)

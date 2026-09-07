"""Celery backed implementation of :class:`~backend.services.task_queue.TaskDispatcher`."""

import logging
import uuid

from backend.worker.celery_app import celery_app
from backend.worker.tasks import process_document_task

logger = logging.getLogger(__name__)


class CeleryTaskDispatcher:
    def enqueue_document_processing(self, document_id: uuid.UUID) -> str:
        result = process_document_task.delay(str(document_id))

        logger.info(
            "document_task_enqueued document_id=%s task_id=%s",
            document_id,
            result.id,
        )
        return result.id

    def revoke(self, task_id: str) -> None:
        """Best effort: a task already running keeps going.

        Cancellation of in-flight work relies on the document row disappearing,
        which every heavy stage re-checks.
        """
        try:
            celery_app.control.revoke(task_id)
            logger.info("document_task_revoked task_id=%s", task_id)
        except Exception as exc:  # noqa: BLE001 - a broker outage must not break DELETE
            logger.warning(
                "document_task_revoke_failed task_id=%s error=%s",
                task_id,
                type(exc).__name__,
            )

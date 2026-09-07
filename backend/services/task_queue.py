"""Queue abstraction used by the API layer.

The routers depend on this Protocol rather than on Celery directly, which keeps
``backend.api`` free of broker imports and lets the tests substitute an
in-memory dispatcher.
"""

import uuid
from typing import Protocol


class TaskDispatcher(Protocol):
    def enqueue_document_processing(self, document_id: uuid.UUID) -> str:
        """Queue ingestion for a document and return the task id."""
        ...

    def revoke(self, task_id: str) -> None:
        """Best effort cancellation of a queued or running task."""
        ...

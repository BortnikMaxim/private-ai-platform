"""Domain level exceptions.

Services raise these instead of ``HTTPException`` so that business logic stays
framework agnostic. ``backend.app`` translates them into JSON responses.
"""


class AppError(Exception):
    """Base class for errors that map onto a specific HTTP status code."""

    status_code: int = 500
    default_detail: str = "Internal server error"

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.default_detail
        super().__init__(self.detail)


class NotFoundError(AppError):
    status_code = 404
    default_detail = "Resource not found"


class ConversationNotFoundError(NotFoundError):
    default_detail = "Conversation not found"


class DocumentNotFoundError(NotFoundError):
    default_detail = "Document not found"


class InvalidDocumentError(AppError):
    status_code = 400
    default_detail = "Invalid document"


class FileTooLargeError(AppError):
    status_code = 413
    default_detail = "Uploaded file is too large"


class InferenceUnavailableError(AppError):
    status_code = 502
    default_detail = "Inference service is unavailable"


class VectorStoreError(AppError):
    status_code = 503
    default_detail = "Vector store is unavailable"


class DocumentSourceMissingError(AppError):
    """The uploaded file backing a document is gone. Permanent, do not retry."""

    status_code = 410
    default_detail = "Uploaded file is no longer available"


class TransientProcessingError(AppError):
    """A failure worth retrying: Qdrant down, database connection dropped, ...

    Celery retries tasks that raise this with exponential backoff and jitter;
    every other exception is treated as permanent.
    """

    status_code = 503
    default_detail = "Temporary processing failure"


class DocumentGoneError(AppError):
    """The document disappeared mid-processing (deleted by a concurrent request)."""

    status_code = 404
    default_detail = "Document was deleted while it was being processed"

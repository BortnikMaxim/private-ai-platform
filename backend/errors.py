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

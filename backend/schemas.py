import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    detail: str


class Source(BaseModel):
    """A single grounded chunk handed to the model."""

    document_id: str | None = None
    filename: str | None = None
    page: int | None = None
    chunk_index: int | None = None
    vector_score: float | None = None
    rerank_score: float | None = None
    # Kept for backward compatibility with the original /rag/ask response.
    score: float | None = None


class RetrievedChunk(Source):
    """Source plus the chunk body, exposed by the debug retrieval endpoint."""

    text: str | None = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    api: str
    postgres: str
    redis: str
    qdrant: str
    inference: str


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    original_filename: str
    content_type: str
    size_bytes: int
    status: str
    total_pages: int
    extracted_pages: int
    chunks_count: int
    error_message: str | None = None
    created_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentRead]
    total: int


class DocumentChunkRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    page: int | None
    chunk_index: int
    text: str


class DocumentDetailResponse(DocumentRead):
    chunks: list[DocumentChunkRead] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    id: uuid.UUID
    deleted: bool = True


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


class ConversationCreate(BaseModel):
    title: str = Field(default="New conversation", min_length=1, max_length=255)
    user_id: uuid.UUID | None = None


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID | None = None
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    items: list[ConversationRead]
    total: int


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str
    content: str
    created_at: datetime


class ConversationDetailResponse(ConversationRead):
    messages: list[MessageRead] = Field(default_factory=list)


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=5000)
    use_rag: bool = False
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)
    top_k: int | None = Field(default=None, ge=1, le=50)
    candidate_k: int | None = Field(default=None, ge=1, le=200)


class MessageResponse(BaseModel):
    message: MessageRead
    used_rag: bool = False
    sources: list[Source] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int | None = Field(default=None, ge=1, le=200)
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=15, ge=1, le=50)
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)


class RetrieveResponse(BaseModel):
    question: str
    vector_results: list[RetrievedChunk]
    reranked_results: list[RetrievedChunk]


# ---------------------------------------------------------------------------
# Inference wire format
# ---------------------------------------------------------------------------

ChatRole = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: ChatRole
    content: str


# ---------------------------------------------------------------------------
# Legacy
# ---------------------------------------------------------------------------


class LegacyUploadResponse(BaseModel):
    """Response shape of the original POST /documents/upload endpoint."""

    document_id: uuid.UUID
    filename: str
    total_pages: int
    extracted_pages: int
    chunks: int


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


def _round(value: Any) -> float | None:
    return None if value is None else round(float(value), 4)


def to_source(chunk: dict[str, Any]) -> Source:
    return Source(
        document_id=_as_str(chunk.get("document_id")),
        filename=chunk.get("filename"),
        page=chunk.get("page"),
        chunk_index=chunk.get("chunk_index"),
        vector_score=_round(chunk.get("vector_score", chunk.get("score"))),
        rerank_score=_round(chunk.get("rerank_score")),
        score=_round(chunk.get("score")),
    )


def to_retrieved_chunk(chunk: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        **to_source(chunk).model_dump(),
        text=chunk.get("text"),
    )


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)

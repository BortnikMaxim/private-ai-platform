import uuid

from fastapi import APIRouter, File, Query, UploadFile, status

from backend.dependencies import (
    DbSession,
    DocumentProcessorDep,
    DocumentServiceDep,
    SessionFactoryDep,
    TaskDispatcherDep,
)
from backend.schemas import (
    DeleteResponse,
    DocumentAcceptedResponse,
    DocumentChunkRead,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentRead,
    LegacyUploadResponse,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_document(
    session: DbSession,
    documents: DocumentServiceDep,
    dispatcher: TaskDispatcherDep,
    file: UploadFile = File(...),
) -> DocumentAcceptedResponse:
    """Accept a PDF and queue it for ingestion.

    Returns as soon as the file is stored and the row exists; parsing,
    embedding and indexing happen in a Celery worker. Poll
    ``GET /documents/{document_id}`` until ``status`` is ``ready`` or
    ``failed``.
    """
    document = await documents.create_pending(session, file)

    task_id = dispatcher.enqueue_document_processing(document.id)
    await documents.attach_task(session, document.id, task_id)

    return DocumentAcceptedResponse(
        document_id=document.id,
        status=document.status,
        task_id=task_id,
    )


@router.post(
    "/upload",
    response_model=LegacyUploadResponse,
    deprecated=True,
    summary="Deprecated: synchronous upload, use POST /documents",
)
async def upload_document(
    session: DbSession,
    session_factory: SessionFactoryDep,
    documents: DocumentServiceDep,
    processor: DocumentProcessorDep,
    file: UploadFile = File(...),
) -> LegacyUploadResponse:
    """Compatibility wrapper that keeps the original blocking semantics.

    Its response reports page and chunk counts, which only exist once ingestion
    has finished, so this endpoint cannot become asynchronous without changing
    its contract. It runs the very same pipeline the worker runs, just inline,
    and blocks the request for the whole parse/embed cycle. New clients should
    use ``POST /documents``.
    """
    document = await documents.create_pending(session, file)

    try:
        document = await processor.process(session_factory, document.id)
    except Exception as exc:
        from backend.services.document_processor import safe_error_message

        await processor.mark_failed(session_factory, document.id, safe_error_message(exc))
        raise

    return LegacyUploadResponse(
        document_id=document.id,
        filename=document.filename,
        total_pages=document.total_pages,
        extracted_pages=document.extracted_pages,
        chunks=document.chunks_count,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    session: DbSession,
    documents: DocumentServiceDep,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> DocumentListResponse:
    items, total = await documents.list_documents(session, limit=limit, offset=offset)

    return DocumentListResponse(
        items=[DocumentRead.model_validate(item) for item in items],
        total=total,
    )


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: uuid.UUID,
    session: DbSession,
    documents: DocumentServiceDep,
    include_chunks: bool = Query(
        default=False,
        description="Include chunk metadata and text in the response",
    ),
) -> DocumentDetailResponse:
    """Current state of a document: processing, ready or failed.

    This is the endpoint to poll after ``POST /documents``. It carries the
    status, the ingestion metadata, ``celery_task_id`` and, when the status is
    ``failed``, ``error_message``.
    """
    document = await documents.get_document(
        session,
        document_id,
        with_chunks=include_chunks,
    )

    chunks = []

    if include_chunks:
        chunks = [
            DocumentChunkRead.model_validate(chunk)
            for chunk in sorted(document.chunks, key=lambda item: item.chunk_index)
        ]

    return DocumentDetailResponse(
        **DocumentRead.model_validate(document).model_dump(),
        chunks=chunks,
    )


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: uuid.UUID,
    session: DbSession,
    documents: DocumentServiceDep,
    dispatcher: TaskDispatcherDep,
) -> DeleteResponse:
    """Remove the document, its chunks, its Qdrant points and its source file.

    Safe to call while ingestion is still running: the task is revoked and, if
    it is already past the point of no revoke, it detects the missing row and
    cleans up after itself instead of resurrecting the document.
    """
    await documents.delete_document(session, document_id, dispatcher=dispatcher)
    return DeleteResponse(id=document_id)

import uuid

from fastapi import APIRouter, File, Query, UploadFile, status

from backend.dependencies import DbSession, DocumentServiceDep
from backend.schemas import (
    DeleteResponse,
    DocumentChunkRead,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentRead,
    LegacyUploadResponse,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def create_document(
    session: DbSession,
    documents: DocumentServiceDep,
    file: UploadFile = File(...),
) -> DocumentRead:
    """Upload and ingest a PDF (parse -> chunk -> embed -> index)."""
    document = await documents.ingest(session, file)
    return DocumentRead.model_validate(document)


@router.post(
    "/upload",
    response_model=LegacyUploadResponse,
    deprecated=True,
    summary="Deprecated: use POST /documents",
)
async def upload_document(
    session: DbSession,
    documents: DocumentServiceDep,
    file: UploadFile = File(...),
) -> LegacyUploadResponse:
    document = await documents.ingest(session, file)

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
) -> DeleteResponse:
    """Remove the document, its chunks and its Qdrant points."""
    await documents.delete_document(session, document_id)
    return DeleteResponse(id=document_id)

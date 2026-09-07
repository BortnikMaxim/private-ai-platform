"""Document tools.

Both delegate to the services that already exist. There is deliberately no
retrieval, embedding or Qdrant logic here — ``search_documents`` is a thin
orchestration wrapper around :class:`~backend.services.rag_service.RagService`.
"""

from typing import Any

from pydantic import BaseModel, Field

from backend.agent.tools.base import Tool, ToolContext, ToolError, parse_uuid
from backend.errors import DocumentNotFoundError


class SearchDocumentsInput(BaseModel):
    query: str = Field(min_length=1, max_length=1000, description="What to look for")
    document_ids: list[str] | None = Field(
        default=None,
        max_length=50,
        description="Optional: restrict the search to these document ids",
    )
    top_k: int | None = Field(default=None, ge=1, le=20)


class SearchDocumentsTool(Tool):
    name = "search_documents"
    description = (
        "Ищет релевантные фрагменты в корпоративных документах "
        "(векторный поиск + переранжирование). Возвращает фрагменты и источники."
    )
    input_schema = SearchDocumentsInput

    async def execute(
        self,
        arguments: SearchDocumentsInput,
        context: ToolContext,
    ) -> dict[str, Any]:
        if context.rag_service is None:
            raise ToolError("document search is not available")

        # An explicit argument wins; otherwise inherit the request's scope.
        document_ids = arguments.document_ids or context.document_ids or None

        if document_ids:
            document_ids = [str(parse_uuid(value)) for value in document_ids]

        chunks = await context.rag_service.retrieve(
            question=arguments.query,
            top_k=arguments.top_k,
            document_ids=document_ids,
        )

        # Hand the citations back so the endpoint can report them.
        context.collect_sources(chunks)

        return {
            "matches": len(chunks),
            "chunks": [
                {
                    "filename": chunk.get("filename"),
                    "page": chunk.get("page"),
                    "chunk_index": chunk.get("chunk_index"),
                    "text": chunk.get("text"),
                }
                for chunk in chunks
            ],
        }


class DocumentMetadataInput(BaseModel):
    document_id: str = Field(min_length=1, max_length=64)


class DocumentMetadataTool(Tool):
    name = "get_document_metadata"
    description = (
        "Возвращает метаданные документа по его id: имя файла, статус "
        "обработки, число страниц и фрагментов."
    )
    input_schema = DocumentMetadataInput

    async def execute(
        self,
        arguments: DocumentMetadataInput,
        context: ToolContext,
    ) -> dict[str, Any]:
        if context.document_service is None or context.session is None:
            raise ToolError("document metadata is not available")

        document_id = parse_uuid(arguments.document_id)

        try:
            document = await context.document_service.get_document(
                context.session,
                document_id,
            )
        except DocumentNotFoundError as exc:
            raise ToolError(f"document {document_id} was not found") from exc

        return {
            "document_id": str(document.id),
            "filename": document.filename,
            "status": document.status,
            "total_pages": document.total_pages,
            "extracted_pages": document.extracted_pages,
            "chunks_count": document.chunks_count,
            "size_bytes": document.size_bytes,
            "error_message": document.error_message,
        }

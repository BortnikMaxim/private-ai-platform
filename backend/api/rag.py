"""Stateless RAG endpoints (no conversation persistence)."""

from fastapi import APIRouter, HTTPException

from backend.dependencies import InferenceDep, RagDep
from backend.prompts import GROUNDED_SYSTEM_PROMPT
from backend.schemas import (
    AskRequest,
    AskResponse,
    RetrieveRequest,
    RetrieveResponse,
    to_retrieved_chunk,
    to_source,
)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    rag: RagDep,
    inference: InferenceDep,
) -> AskResponse:
    document_ids = [str(value) for value in payload.document_ids or []] or None

    retrieved = await rag.retrieve(
        question=payload.question,
        top_k=payload.top_k,
        candidate_k=payload.candidate_k,
        document_ids=document_ids,
    )

    if not retrieved:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found",
        )

    context = rag.build_context(retrieved)

    answer = await inference.chat(
        [
            {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"КОНТЕКСТ:\n\n{context}\n\nВОПРОС:\n{payload.question}",
            },
        ],
        temperature=0.1,
    )

    return AskResponse(
        answer=answer,
        sources=[to_source(chunk) for chunk in retrieved],
    )


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    payload: RetrieveRequest,
    rag: RagDep,
) -> RetrieveResponse:
    """Debug endpoint: inspect vector hits and reranked results side by side."""
    document_ids = [str(value) for value in payload.document_ids or []] or None

    vector_results = await rag.vector_retrieve(
        question=payload.question,
        limit=payload.candidate_k,
        document_ids=document_ids,
    )

    reranked_results = await rag.rerank(
        question=payload.question,
        candidates=vector_results,
        top_k=payload.top_k,
    )

    return RetrieveResponse(
        question=payload.question,
        vector_results=[to_retrieved_chunk(item) for item in vector_results],
        reranked_results=[to_retrieved_chunk(item) for item in reranked_results],
    )

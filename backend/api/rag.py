"""Stateless RAG endpoints (no conversation persistence)."""

from fastapi import APIRouter, HTTPException

from backend.dependencies import CurrentUser, DbSession, InferenceDep, RagDep
from backend.prompts import GROUNDED_SYSTEM_PROMPT
from backend.schemas import (
    AskRequest,
    AskResponse,
    RetrieveRequest,
    RetrieveResponse,
    to_retrieved_chunk,
    to_source,
)
from backend.services.fusion import reciprocal_rank_fusion
from backend.services.rag_service import MODE_HYBRID

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    rag: RagDep,
    inference: InferenceDep,
    user: CurrentUser,
    session: DbSession,
) -> AskResponse:
    # document_ids narrows the search; the tenant filter still applies on top,
    # so a foreign id in the body simply matches nothing.
    document_ids = [str(value) for value in payload.document_ids or []] or None

    retrieved = await rag.retrieve(
        question=payload.question,
        user_id=str(user.id),
        top_k=payload.top_k,
        candidate_k=payload.candidate_k,
        document_ids=document_ids,
        session=session,
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
    user: CurrentUser,
    session: DbSession,
) -> RetrieveResponse:
    """Debug endpoint: inspect every retrieval stage side by side.

    In hybrid mode this exposes the dense list, the lexical list, the RRF-fused
    pool and the reranked result, so a drop in answer quality can be traced to
    the stage that caused it.
    """
    document_ids = [str(value) for value in payload.document_ids or []] or None
    mode = rag.resolve_mode(payload.mode, session)

    vector_results = await rag.vector_retrieve(
        question=payload.question,
        user_id=str(user.id),
        limit=payload.candidate_k,
        document_ids=document_ids,
    )

    lexical_results: list = []
    fused_results = vector_results

    if mode == MODE_HYBRID:
        lexical_results = await rag.lexical_retrieve(
            session=session,
            question=payload.question,
            user_id=str(user.id),
            limit=payload.lexical_k,
            document_ids=document_ids,
        )
        fused_results = reciprocal_rank_fusion(
            [vector_results, lexical_results],
            rrf_k=payload.rrf_k or rag.settings.rag_rrf_k,
        )

    reranked_results = await rag.rerank(
        question=payload.question,
        candidates=fused_results[: rag.settings.rag_rerank_candidate_k],
        top_k=payload.top_k,
    )

    return RetrieveResponse(
        question=payload.question,
        mode=mode,
        vector_results=[to_retrieved_chunk(item) for item in vector_results],
        lexical_results=[to_retrieved_chunk(item) for item in lexical_results],
        fused_results=[to_retrieved_chunk(item) for item in fused_results],
        reranked_results=[to_retrieved_chunk(item) for item in reranked_results],
    )

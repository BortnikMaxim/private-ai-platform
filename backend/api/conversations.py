import uuid

from fastapi import APIRouter, Query, status

from backend.agent.state import tools_used as summarize_tools
from backend.dependencies import ConversationServiceDep, DbSession
from backend.schemas import (
    AgentMessageCreate,
    AgentMessageResponse,
    ConversationCreate,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationRead,
    DeleteResponse,
    MessageCreate,
    MessageRead,
    MessageResponse,
    ToolUsed,
    to_source,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreate,
    session: DbSession,
    conversations: ConversationServiceDep,
) -> ConversationRead:
    conversation = await conversations.create(
        session,
        title=payload.title,
        user_id=payload.user_id,
    )
    return ConversationRead.model_validate(conversation)


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    session: DbSession,
    conversations: ConversationServiceDep,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ConversationListResponse:
    items, total = await conversations.list_conversations(
        session, limit=limit, offset=offset
    )

    return ConversationListResponse(
        items=[ConversationRead.model_validate(item) for item in items],
        total=total,
    )


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: uuid.UUID,
    session: DbSession,
    conversations: ConversationServiceDep,
) -> ConversationDetailResponse:
    conversation = await conversations.get(session, conversation_id)
    messages = await conversations.get_messages(session, conversation_id)

    return ConversationDetailResponse(
        **ConversationRead.model_validate(conversation).model_dump(),
        messages=[MessageRead.model_validate(message) for message in messages],
    )


@router.delete("/{conversation_id}", response_model=DeleteResponse)
async def delete_conversation(
    conversation_id: uuid.UUID,
    session: DbSession,
    conversations: ConversationServiceDep,
) -> DeleteResponse:
    await conversations.delete(session, conversation_id)
    return DeleteResponse(id=conversation_id)


@router.get("/{conversation_id}/messages", response_model=list[MessageRead])
async def list_messages(
    conversation_id: uuid.UUID,
    session: DbSession,
    conversations: ConversationServiceDep,
) -> list[MessageRead]:
    await conversations.get(session, conversation_id)
    messages = await conversations.get_messages(session, conversation_id)

    return [MessageRead.model_validate(message) for message in messages]


@router.post(
    "/{conversation_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_message(
    conversation_id: uuid.UUID,
    payload: MessageCreate,
    session: DbSession,
    conversations: ConversationServiceDep,
) -> MessageResponse:
    """Append a user turn and return the assistant reply.

    With ``use_rag=true`` the question is answered from retrieved document
    chunks; ``document_ids`` restricts retrieval to those documents.
    """
    message, sources = await conversations.post_message(
        session,
        conversation_id=conversation_id,
        content=payload.content,
        use_rag=payload.use_rag,
        document_ids=payload.document_ids,
        top_k=payload.top_k,
        candidate_k=payload.candidate_k,
    )

    return MessageResponse(
        message=MessageRead.model_validate(message),
        used_rag=payload.use_rag,
        sources=[to_source(chunk) for chunk in sources],
    )


@router.post(
    "/{conversation_id}/agent",
    response_model=AgentMessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_agent_message(
    conversation_id: uuid.UUID,
    payload: AgentMessageCreate,
    session: DbSession,
    conversations: ConversationServiceDep,
) -> AgentMessageResponse:
    """Answer a turn through the agent graph.

    The agent classifies the request and takes one of four branches: answering
    directly, searching the document base, calling a whitelisted tool, or a
    safe fallback. It is a fixed state machine, not an autonomous loop.

    ``POST /conversations/{id}/messages`` remains the plain chat/RAG path and is
    unaffected.
    """
    message, state = await conversations.run_agent_turn(
        session,
        conversation_id=conversation_id,
        content=payload.content,
        use_rag=payload.use_rag,
        document_ids=payload.document_ids,
    )

    return AgentMessageResponse(
        message=MessageRead.model_validate(message),
        route=state.get("route") or "fallback",
        tools_used=[ToolUsed(**entry) for entry in summarize_tools(state)],
        sources=[to_source(chunk) for chunk in state.get("retrieved_sources", [])],
    )

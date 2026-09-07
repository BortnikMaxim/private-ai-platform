"""Conversation persistence and the chat / RAG turn orchestration."""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Settings
from backend.errors import (
    AgentUnavailableError,
    ConversationNotFoundError,
    DocumentNotFoundError,
)
from backend.models import Conversation, Message
from backend.prompts import CHAT_SYSTEM_PROMPT, GROUNDED_SYSTEM_PROMPT
from backend.services.document_service import DocumentService
from backend.services.inference_client import InferenceClient
from backend.services.rag_service import RagService

logger = logging.getLogger(__name__)

HISTORY_ROLES = ("user", "assistant")


class ConversationService:
    def __init__(
        self,
        inference: InferenceClient,
        rag: RagService,
        documents: DocumentService,
        settings: Settings,
        agent: Any = None,
    ) -> None:
        self.inference = inference
        self.rag = rag
        self.documents = documents
        self.settings = settings
        # Optional so the plain chat/RAG paths keep working without an agent.
        self.agent = agent

    # -- CRUD ------------------------------------------------------------

    async def create(
        self,
        session: AsyncSession,
        title: str,
        user_id: uuid.UUID | None = None,
    ) -> Conversation:
        conversation = Conversation(title=title, user_id=user_id)

        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)

        return conversation

    async def list_conversations(
        self,
        session: AsyncSession,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Conversation], int]:
        total = await session.scalar(select(func.count()).select_from(Conversation)) or 0

        result = await session.execute(
            select(Conversation)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )

        return list(result.scalars().all()), total

    async def get(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> Conversation:
        conversation = (
            await session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
        ).scalar_one_or_none()

        if conversation is None:
            raise ConversationNotFoundError()

        return conversation

    async def delete(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> None:
        conversation = await self.get(session, conversation_id)

        await session.delete(conversation)
        await session.commit()

        logger.info("conversation_deleted conversation_id=%s", conversation_id)

    async def get_messages(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        limit: int | None = None,
    ) -> list[Message]:
        """Return messages oldest-first, optionally only the newest ``limit``."""
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
        )

        if limit is not None:
            query = query.limit(limit)

        result = await session.execute(query)

        return list(reversed(result.scalars().all()))

    # -- chat turn -------------------------------------------------------

    async def post_message(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        content: str,
        use_rag: bool = False,
        document_ids: list[uuid.UUID] | None = None,
        top_k: int | None = None,
        candidate_k: int | None = None,
    ) -> tuple[Message, list[dict[str, Any]]]:
        conversation = await self.get(session, conversation_id)

        if document_ids:
            await self._assert_documents_exist(session, document_ids)

        # 1. persist the user turn so history survives a restart even if the
        #    model call fails afterwards.
        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=content,
        )
        session.add(user_message)
        await session.commit()

        # 2. replay a bounded slice of history
        history = await self.get_messages(
            session,
            conversation_id,
            limit=self.settings.chat_history_limit,
        )

        # 3. optional retrieval
        sources: list[dict[str, Any]] = []
        context: str | None = None

        if use_rag:
            sources = await self.rag.retrieve(
                question=content,
                top_k=top_k,
                candidate_k=candidate_k,
                document_ids=[str(value) for value in document_ids or []] or None,
            )
            context = self.rag.build_context(sources)

        llm_messages = self._build_llm_messages(history, context)

        logger.info(
            "conversation_turn conversation_id=%s history=%d rag=%s sources=%d",
            conversation_id,
            len(history),
            use_rag,
            len(sources),
        )

        # 4. call the model
        answer = await self.inference.chat(llm_messages)

        # 5. persist the assistant turn
        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
        )
        session.add(assistant_message)

        conversation.updated_at = datetime.now(UTC)
        session.add(conversation)

        await session.commit()
        await session.refresh(assistant_message)

        return assistant_message, sources

    # -- agent turn -------------------------------------------------------

    async def run_agent_turn(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        content: str,
        use_rag: bool = True,
        document_ids: list[uuid.UUID] | None = None,
    ) -> tuple[Message, dict[str, Any]]:
        """Persist the user turn, run the agent graph, persist the answer.

        Only the final assistant text reaches the database — the agent's
        internal state, routing rationale and tool payloads never become
        Message rows.
        """
        if self.agent is None:
            raise AgentUnavailableError()

        conversation = await self.get(session, conversation_id)

        if document_ids:
            await self._assert_documents_exist(session, document_ids)

        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=content,
        )
        session.add(user_message)
        await session.commit()

        history = await self.get_messages(
            session,
            conversation_id,
            limit=self.settings.chat_history_limit,
        )

        state = await self.agent.run(
            conversation_id=conversation_id,
            user_message=content,
            chat_history=[
                {"role": message.role, "content": message.content}
                for message in history
                if message.role in HISTORY_ROLES
            ],
            use_rag=use_rag,
            document_ids=[str(value) for value in document_ids or []] or None,
            session=session,
        )

        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=state.get("final_answer", ""),
        )
        session.add(assistant_message)

        conversation.updated_at = datetime.now(UTC)
        session.add(conversation)

        await session.commit()
        await session.refresh(assistant_message)

        return assistant_message, state

    def _build_llm_messages(
        self,
        history: list[Message],
        context: str | None,
    ) -> list[dict[str, str]]:
        system_prompt = GROUNDED_SYSTEM_PROMPT if context else CHAT_SYSTEM_PROMPT
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        turns = [message for message in history if message.role in HISTORY_ROLES]

        for index, message in enumerate(turns):
            body = message.content

            # Ground only the current question, not the whole transcript.
            is_last_user_turn = index == len(turns) - 1 and message.role == "user"

            if context and is_last_user_turn:
                body = f"КОНТЕКСТ:\n\n{context}\n\nВОПРОС:\n{message.content}"

            if body.strip():
                messages.append({"role": message.role, "content": body})

        return messages

    async def _assert_documents_exist(
        self,
        session: AsyncSession,
        document_ids: list[uuid.UUID],
    ) -> None:
        found = set(await self.documents.existing_document_ids(session, document_ids))
        missing = [str(value) for value in document_ids if value not in found]

        if missing:
            raise DocumentNotFoundError(
                f"Unknown document ids: {', '.join(sorted(missing))}"
            )

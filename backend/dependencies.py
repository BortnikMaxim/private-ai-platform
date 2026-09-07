"""FastAPI dependency providers.

Long lived resources are created once in the application lifespan and stored on
``app.state``; these providers hand them to the routers. Keeping the lookup in
dependencies (rather than importing globals) is what lets the tests swap in
lightweight fakes through ``app.dependency_overrides``.
"""

from typing import Annotated

from fastapi import Depends, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from backend.config import Settings, get_settings
from backend.db import get_db
from backend.services.conversation_service import ConversationService
from backend.services.document_service import DocumentService
from backend.services.inference_client import InferenceClient
from backend.services.rag_service import RagService
from backend.services.vector_store import VectorStore


def get_engine(request: Request) -> AsyncEngine:
    return request.app.state.engine


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_inference_client(request: Request) -> InferenceClient:
    return request.app.state.inference_client


def get_rag_service(request: Request) -> RagService:
    return request.app.state.rag_service


def get_document_service(request: Request) -> DocumentService:
    return request.app.state.document_service


def get_conversation_service(request: Request) -> ConversationService:
    return request.app.state.conversation_service


DbSession = Annotated[AsyncSession, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
EngineDep = Annotated[AsyncEngine, Depends(get_engine)]
RedisDep = Annotated[Redis, Depends(get_redis)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
InferenceDep = Annotated[InferenceClient, Depends(get_inference_client)]
RagDep = Annotated[RagService, Depends(get_rag_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
ConversationServiceDep = Annotated[
    ConversationService, Depends(get_conversation_service)
]

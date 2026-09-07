"""FastAPI dependency providers.

Long lived resources are created once in the application lifespan and stored on
``app.state``; these providers hand them to the routers. Keeping the lookup in
dependencies (rather than importing globals) is what lets the tests swap in
lightweight fakes through ``app.dependency_overrides``.
"""

from typing import Annotated

from fastapi import Depends, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.config import Settings, get_settings
from backend.db import get_db
from backend.services.broker import BrokerClient
from backend.services.conversation_service import ConversationService
from backend.services.document_processor import DocumentProcessor
from backend.services.document_service import DocumentService
from backend.services.inference_client import InferenceClient
from backend.services.rag_service import RagService
from backend.services.task_queue import TaskDispatcher
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


def get_document_processor(request: Request) -> DocumentProcessor:
    return request.app.state.document_processor


def get_task_dispatcher(request: Request) -> TaskDispatcher:
    return request.app.state.task_dispatcher


def get_broker(request: Request) -> BrokerClient:
    return request.app.state.broker


def get_session_factory(request: Request) -> async_sessionmaker:
    return request.app.state.session_factory


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
DocumentProcessorDep = Annotated[DocumentProcessor, Depends(get_document_processor)]
TaskDispatcherDep = Annotated[TaskDispatcher, Depends(get_task_dispatcher)]
BrokerDep = Annotated[BrokerClient, Depends(get_broker)]
SessionFactoryDep = Annotated[async_sessionmaker, Depends(get_session_factory)]

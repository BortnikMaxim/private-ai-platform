"""FastAPI dependency providers.

Long lived resources are created once in the application lifespan and stored on
``app.state``; these providers hand them to the routers. Keeping the lookup in
dependencies (rather than importing globals) is what lets the tests swap in
lightweight fakes through ``app.dependency_overrides``.
"""

from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.config import Settings, get_settings
from backend.db import get_db
from backend.errors import InactiveUserError, InvalidTokenError, PermissionDeniedError
from backend.models import ROLE_ADMIN, User
from backend.security.tokens import decode_access_token
from backend.services.auth_service import AuthService
from backend.services.broker import BrokerClient
from backend.services.conversation_service import ConversationService
from backend.services.document_processor import DocumentProcessor
from backend.services.document_service import DocumentService
from backend.services.inference_client import InferenceClient
from backend.services.rag_service import RagService
from backend.services.rate_limiter import RateLimiter
from backend.services.task_queue import TaskDispatcher
from backend.services.vector_store import VectorStore
from backend.tracing import NULL_TRACER, Tracer

# auto_error=False so a missing header raises our own 401 with the standard
# WWW-Authenticate treatment instead of FastAPI's 403.
bearer_scheme = HTTPBearer(auto_error=False, description="JWT access token")


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


def get_tracer(request: Request) -> Tracer:
    # NULL_TRACER keeps routes working when the lifespan has not run, which is
    # how the offline tests drive the app.
    return getattr(request.app.state, "tracer", NULL_TRACER)


def get_session_factory(request: Request) -> async_sessionmaker:
    return request.app.state.session_factory


def get_auth_service(request: Request) -> AuthService:
    return request.app.state.auth_service


def get_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.rate_limiter


def get_agent_service(request: Request):
    return request.app.state.agent_service


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


async def get_current_user(
    session: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    auth: Annotated[AuthService, Depends(get_auth_service)],
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
) -> User:
    """Resolve the bearer token to a user row.

    Token parsing lives here and nowhere else, so no router ever touches a JWT.
    """
    if credentials is None or not credentials.credentials:
        raise InvalidTokenError("Not authenticated")

    claims = decode_access_token(settings, credentials.credentials)
    user = await auth.get_by_id(session, claims.user_id)

    if user is None:
        # The signature was valid but the account is gone. Same 401 as a bad
        # token: a deleted account must not be distinguishable.
        raise InvalidTokenError()

    return user


async def get_current_active_user(
    user: Annotated[User, Depends(get_current_user)],
) -> User:
    if not user.is_active:
        raise InactiveUserError()

    return user


async def require_admin(
    user: Annotated[User, Depends(get_current_active_user)],
) -> User:
    if user.role != ROLE_ADMIN:
        raise PermissionDeniedError("Administrator privileges are required")

    return user


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
TracerDep = Annotated[Tracer, Depends(get_tracer)]
SessionFactoryDep = Annotated[async_sessionmaker, Depends(get_session_factory)]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
RateLimiterDep = Annotated[RateLimiter, Depends(get_rate_limiter)]
CurrentUser = Annotated[User, Depends(get_current_active_user)]
AdminUser = Annotated[User, Depends(require_admin)]

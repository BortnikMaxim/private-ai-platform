"""Test fixtures.

Every fixture here is offline: the sentence-transformers models, Qdrant, Redis
and the inference service are replaced by deterministic in-process fakes, and
PostgreSQL is replaced by in-memory SQLite. No test downloads a model or opens
a network connection.
"""

import hashlib
import uuid
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend import dependencies as deps
from backend.agent.graph import AgentService
from backend.agent.tools.registry import ToolRegistry, default_registry
from backend.app import create_app
from backend.config import Settings, get_settings
from backend.db import Base, get_db
from backend.errors import InferenceUnavailableError
from backend.models import ROLE_ADMIN, ROLE_USER, User
from backend.security.passwords import hash_password
from backend.security.tokens import create_access_token
from backend.services.auth_service import AuthService
from backend.services.conversation_service import ConversationService
from backend.services.document_processor import DocumentProcessor
from backend.services.document_service import DocumentService
from backend.services.lexical_index import LexicalRetriever
from backend.services.rag_service import RagService
from backend.services.rate_limiter import RateLimiter
from backend.services.storage import DocumentStorage

VECTOR_SIZE = 8
DEFAULT_PASSWORD = "correct-horse-battery-staple"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _deterministic_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [byte / 255.0 for byte in digest[:VECTOR_SIZE]]


class FakeEmbeddingService:
    """Stand-in for EmbeddingService with the same async interface."""

    def __init__(self) -> None:
        self.is_loaded = True
        self.embed_calls: list[list[str]] = []
        self.rerank_calls: list[str] = []

    async def ensure_loaded(self) -> None:
        return None

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [_deterministic_vector(text) for text in texts]

    async def embed_query(self, question: str) -> list[float]:
        return _deterministic_vector(question)

    async def rerank(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Score by word overlap so ordering is predictable and explainable."""
        self.rerank_calls.append(question)

        query_words = set(question.lower().split())
        scored: list[dict[str, Any]] = []

        for candidate in candidates:
            enriched = dict(candidate)
            text_words = set((candidate.get("text") or "").lower().split())
            enriched.setdefault("vector_score", candidate.get("score"))
            enriched["rerank_score"] = float(len(query_words & text_words))
            scored.append(enriched)

        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored[:top_k]


class FakeVectorStore:
    """In-memory Qdrant replacement supporting document_id filtering."""

    def __init__(self) -> None:
        self.points: dict[str, dict[str, Any]] = {}
        self.searches: list[dict[str, Any]] = []
        self.fail_on_delete = False

    async def ensure_collection(self) -> None:
        return None

    async def health(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None

    async def upsert_chunks(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            if not record.get("user_id"):
                raise ValueError("refusing to index a chunk without a user_id")
            self.points[record["point_id"]] = dict(record)

    async def delete_document(
        self,
        document_id: str,
        user_id: str | None = None,
    ) -> None:
        if self.fail_on_delete:
            raise RuntimeError("qdrant is down")

        for point_id in [
            key
            for key, value in self.points.items()
            if value["document_id"] == document_id
            and (user_id is None or str(value.get("user_id")) == str(user_id))
        ]:
            del self.points[point_id]

    async def search(
        self,
        vector: list[float],
        limit: int,
        user_id: str,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Mirrors the real store: the tenant filter is applied before limiting."""
        if not user_id:
            raise ValueError("search requires a user_id")

        self.searches.append(
            {"limit": limit, "user_id": user_id, "document_ids": document_ids}
        )

        matches = [
            point
            for point in self.points.values()
            if str(point.get("user_id")) == str(user_id)
            and (document_ids is None or point["document_id"] in document_ids)
        ]
        matches.sort(key=lambda point: (point["document_id"], point["chunk_index"]))

        results = []

        for offset, point in enumerate(matches[:limit]):
            score = round(1.0 - offset * 0.01, 4)
            results.append(
                {
                    # Mirrors the real store: the point id is the key the
                    # dense and lexical branches are fused on.
                    "point_id": point["point_id"],
                    "score": score,
                    "vector_score": score,
                    "dense_score": score,
                    "dense_rank": offset + 1,
                    "user_id": point.get("user_id"),
                    "document_id": point["document_id"],
                    "filename": point["filename"],
                    "page": point["page"],
                    "chunk_index": point["chunk_index"],
                    "text": point["text"],
                }
            )

        return results


class FakeInferenceClient:
    def __init__(self, answer: str = "Тестовый ответ модели") -> None:
        self.answer = answer
        self.calls: list[list[dict[str, str]]] = []
        self.available = True
        # Scripted replies, consumed in order; falls back to `answer` when empty.
        self.responses: list[str] = []

    def script(self, *responses: str) -> None:
        self.responses.extend(responses)

    async def health(self) -> bool:
        return self.available

    async def aclose(self) -> None:
        return None

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self.calls.append(messages)

        if not self.available:
            raise InferenceUnavailableError()

        if self.responses:
            return self.responses.pop(0)

        return self.answer


class FakeRedis:
    """In-memory stand-in that also serves the rate limiter's Lua script."""

    def __init__(self) -> None:
        self.available = True
        self.counters: dict[str, int] = {}
        self.expiries: dict[str, int] = {}

    async def ping(self) -> bool:
        if not self.available:
            raise ConnectionError("redis is down")
        return True

    async def eval(self, script: str, numkeys: int, *args):
        """Mimic the INCR + EXPIRE script atomically enough for tests."""
        if not self.available:
            raise ConnectionError("redis is down")

        key = args[0]
        ttl = int(args[numkeys]) if len(args) > numkeys else 60

        self.counters[key] = self.counters.get(key, 0) + 1

        if self.counters[key] == 1:
            self.expiries[key] = ttl

        return self.counters[key]

    async def delete(self, key: str) -> int:
        self.counters.pop(key, None)
        self.expiries.pop(key, None)
        return 1

    def expire_window(self) -> None:
        """Simulate every TTL elapsing, without waiting for wall-clock time."""
        self.counters.clear()
        self.expiries.clear()

    async def aclose(self) -> None:
        return None


class FakeTaskDispatcher:
    """Records enqueues instead of talking to RabbitMQ."""

    def __init__(self) -> None:
        self.enqueued: list[tuple[str, str]] = []
        self.revoked: list[str] = []
        self.fail_on_enqueue = False

    def enqueue_document_processing(self, document_id) -> str:
        if self.fail_on_enqueue:
            raise RuntimeError("broker is unreachable")

        task_id = str(uuid.uuid4())
        self.enqueued.append((str(document_id), task_id))
        return task_id

    def revoke(self, task_id: str) -> None:
        self.revoked.append(task_id)


class FakeBroker:
    def __init__(self) -> None:
        self.available = True
        self.workers = ["celery@test"]

    async def health(self) -> bool:
        return self.available

    async def ping_workers(self) -> list[str]:
        return list(self.workers) if self.available else []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        app_env="test",
        # Deterministic and long enough to pass the strength check.
        jwt_secret_key="test-secret-key-that-is-long-enough-for-the-policy",
        database_url="sqlite+aiosqlite:///:memory:",
        inference_api_key="test-key",
        embedding_dim=VECTOR_SIZE,
        preload_models=False,
        rag_top_k=3,
        rag_candidate_k=10,
        chat_history_limit=4,
        max_upload_size_mb=1,
        chunk_size_words=25,
        chunk_overlap_words=5,
        upload_dir=tmp_path / "uploads",
    )


@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # SQLite ignores ON DELETE CASCADE unless foreign keys are switched on.
    @event.listens_for(engine.sync_engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(db_engine):
    return async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)


@pytest.fixture
def embeddings() -> FakeEmbeddingService:
    return FakeEmbeddingService()


@pytest.fixture
def vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def inference() -> FakeInferenceClient:
    return FakeInferenceClient()


@pytest.fixture
def redis_client() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def auth_service(settings) -> AuthService:
    return AuthService(settings=settings)


@pytest.fixture
def rate_limiter(redis_client, settings) -> RateLimiter:
    return RateLimiter(
        redis=redis_client,
        window_seconds=settings.rate_limit_window_seconds,
        enabled=settings.rate_limit_enabled,
    )


@pytest.fixture
def task_dispatcher() -> FakeTaskDispatcher:
    return FakeTaskDispatcher()


@pytest.fixture
def broker() -> FakeBroker:
    return FakeBroker()


@pytest.fixture
def storage(settings) -> DocumentStorage:
    store = DocumentStorage(settings.upload_dir)
    store.ensure_ready()
    return store


@pytest.fixture
def lexical_index(settings) -> LexicalRetriever:
    """Real BM25 retriever — it reads the SQLite test corpus, no mocking needed."""
    return LexicalRetriever(
        k1=settings.bm25_k1,
        b=settings.bm25_b,
        stemming=settings.bm25_stemming,
    )


@pytest.fixture
def rag_service(embeddings, vector_store, settings, lexical_index) -> RagService:
    return RagService(
        embeddings=embeddings,
        vector_store=vector_store,
        settings=settings,
        lexical=lexical_index,
    )


@pytest.fixture
def document_service(vector_store, storage, settings) -> DocumentService:
    return DocumentService(
        vector_store=vector_store,
        storage=storage,
        settings=settings,
    )


@pytest.fixture
def document_processor(embeddings, vector_store, storage, settings) -> DocumentProcessor:
    return DocumentProcessor(
        embeddings=embeddings,
        vector_store=vector_store,
        storage=storage,
        settings=settings,
    )


@pytest.fixture
def tool_registry() -> ToolRegistry:
    return default_registry()


@pytest.fixture
def agent_service(
    inference,
    rag_service,
    document_service,
    tool_registry,
    settings,
) -> AgentService:
    return AgentService(
        inference=inference,
        rag=rag_service,
        documents=document_service,
        registry=tool_registry,
        settings=settings,
    )


@pytest.fixture
def conversation_service(
    inference,
    rag_service,
    document_service,
    agent_service,
    settings,
) -> ConversationService:
    return ConversationService(
        inference=inference,
        rag=rag_service,
        documents=document_service,
        settings=settings,
        agent=agent_service,
    )


@pytest.fixture
def app(
    settings,
    db_engine,
    session_factory,
    redis_client,
    vector_store,
    inference,
    rag_service,
    document_service,
    document_processor,
    conversation_service,
    task_dispatcher,
    broker,
    auth_service,
    rate_limiter,
):
    application = create_app(settings=settings)

    async def override_get_db():
        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    application.dependency_overrides[get_db] = override_get_db
    application.dependency_overrides[deps.get_engine] = lambda: db_engine
    application.dependency_overrides[deps.get_session_factory] = lambda: session_factory
    application.dependency_overrides[deps.get_redis] = lambda: redis_client
    application.dependency_overrides[deps.get_vector_store] = lambda: vector_store
    application.dependency_overrides[deps.get_inference_client] = lambda: inference
    application.dependency_overrides[deps.get_rag_service] = lambda: rag_service
    application.dependency_overrides[deps.get_document_service] = lambda: document_service
    application.dependency_overrides[deps.get_document_processor] = (
        lambda: document_processor
    )
    application.dependency_overrides[deps.get_conversation_service] = (
        lambda: conversation_service
    )
    application.dependency_overrides[deps.get_task_dispatcher] = lambda: task_dispatcher
    application.dependency_overrides[deps.get_broker] = lambda: broker
    application.dependency_overrides[deps.get_auth_service] = lambda: auth_service
    application.dependency_overrides[deps.get_rate_limiter] = lambda: rate_limiter
    application.dependency_overrides[get_settings] = lambda: settings

    return application


@pytest_asyncio.fixture
async def anonymous_client(app):
    """A client with no Authorization header."""
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


# ---------------------------------------------------------------------------
# Users and authenticated clients
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def make_user(session_factory, settings):
    """Factory creating an active user straight in the database."""

    async def _make(
        email: str = "user@example.com",
        password: str = DEFAULT_PASSWORD,
        role: str = ROLE_USER,
        is_active: bool = True,
    ) -> User:
        async with session_factory() as session:
            user = User(
                email=email.lower(),
                password_hash=hash_password(password),
                role=role,
                is_active=is_active,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    return _make


@pytest.fixture
def token_for(settings):
    def _token(user: User) -> str:
        token, _expires = create_access_token(settings, user.id, user.role)
        return token

    return _token


@pytest.fixture
def make_client(app, token_for):
    """Build an httpx client that authenticates as a given user."""

    def _make(user: User | None) -> AsyncClient:
        headers = {}

        if user is not None:
            headers["Authorization"] = f"Bearer {token_for(user)}"

        return AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers=headers,
        )

    return _make


@pytest_asyncio.fixture
async def user(make_user) -> User:
    return await make_user(email="alice@example.com")


@pytest_asyncio.fixture
async def other_user(make_user) -> User:
    return await make_user(email="bob@example.com")


@pytest_asyncio.fixture
async def admin_user(make_user) -> User:
    return await make_user(email="admin@example.com", role=ROLE_ADMIN)


@pytest_asyncio.fixture
async def client(make_client, user):
    """The default client: authenticated as ``user``.

    Existing tests keep working unchanged; they simply run as a real tenant now.
    """
    async with make_client(user) as authed:
        yield authed


@pytest_asyncio.fixture
async def other_client(make_client, other_user):
    async with make_client(other_user) as authed:
        yield authed


@pytest_asyncio.fixture
async def admin_client(make_client, admin_user):
    async with make_client(admin_user) as authed:
        yield authed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def make_document(session_factory, vector_store):
    """Factory for a ready, indexed Document owned by a given user."""
    from backend.models import Document, DocumentChunk

    async def _make(
        owner: User,
        filename: str = "projects.pdf",
        texts: list[str] | None = None,
    ) -> uuid.UUID:
        document_id = uuid.uuid4()
        bodies = texts or [
            "Проект Атлас описывает миграцию биллинга",
            "Проект Борей отвечает за складскую логистику",
            "Финансовый отчёт за третий квартал",
        ]

        async with session_factory() as session:
            session.add(
                Document(
                    id=document_id,
                    user_id=owner.id,
                    filename=filename,
                    original_filename=filename,
                    content_type="application/pdf",
                    size_bytes=1024,
                    status="ready",
                    total_pages=1,
                    extracted_pages=1,
                    chunks_count=len(bodies),
                )
            )

            records = []

            for index, text in enumerate(bodies):
                point_id = str(uuid.uuid4())
                session.add(
                    DocumentChunk(
                        document_id=document_id,
                        qdrant_point_id=point_id,
                        page=1,
                        chunk_index=index,
                        text=text,
                    )
                )
                records.append(
                    {
                        "point_id": point_id,
                        "vector": _deterministic_vector(text),
                        "user_id": str(owner.id),
                        "document_id": str(document_id),
                        "filename": filename,
                        "page": 1,
                        "chunk_index": index,
                        "text": text,
                    }
                )

            await session.commit()

        await vector_store.upsert_chunks(records)

        return document_id

    return _make


@pytest_asyncio.fixture
async def seeded_document(make_document, user) -> uuid.UUID:
    """A ready Document owned by the default ``user`` fixture."""
    return await make_document(user)

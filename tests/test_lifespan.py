"""Lifespan startup/shutdown behaviour, with every external resource mocked.

These tests open no sockets: the engine, Qdrant client, Redis client, inference
client and the embedding models are all replaced with recording fakes.
"""

import pytest

from backend import app as app_module
from backend.app import create_app, lifespan


class FakeEngine:
    def __init__(self) -> None:
        self.disposed = False

    async def dispose(self) -> None:
        self.disposed = True


class FakeQdrantClient:
    def __init__(self, *_args, **_kwargs) -> None:
        self.closed = False
        self.collection_exists_calls = 0

    async def collection_exists(self, _name: str) -> bool:
        self.collection_exists_calls += 1
        return True

    async def create_payload_index(self, **_kwargs) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


class FakeRedisClient:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeInferenceClient:
    def __init__(self, *_args, **_kwargs) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeEmbeddingService:
    def __init__(self, *_args, **_kwargs) -> None:
        self.load_calls = 0

    async def ensure_loaded(self) -> None:
        self.load_calls += 1


@pytest.fixture
def mocked_resources(monkeypatch):
    """Patch every resource constructor used by the lifespan."""
    created: dict[str, object] = {}

    def fake_create_engine(_url):
        engine = FakeEngine()
        created["engine"] = engine
        return engine

    def fake_qdrant(*args, **kwargs):
        client = FakeQdrantClient(*args, **kwargs)
        created["qdrant"] = client
        return client

    def fake_redis_from_url(*_args, **_kwargs):
        client = FakeRedisClient()
        created["redis"] = client
        return client

    def fake_inference(*args, **kwargs):
        client = FakeInferenceClient(*args, **kwargs)
        created["inference"] = client
        return client

    def fake_embeddings(*args, **kwargs):
        service = FakeEmbeddingService(*args, **kwargs)
        created["embeddings"] = service
        return service

    monkeypatch.setattr(app_module, "create_engine", fake_create_engine)
    monkeypatch.setattr(app_module, "AsyncQdrantClient", fake_qdrant)
    monkeypatch.setattr(app_module.redis, "from_url", fake_redis_from_url)
    monkeypatch.setattr(app_module, "InferenceClient", fake_inference)
    monkeypatch.setattr(app_module, "EmbeddingService", fake_embeddings)

    return created


async def test_startup_populates_state_and_shutdown_closes_everything(
    settings,
    mocked_resources,
):
    application = create_app(settings=settings)

    async with lifespan(application):
        state = application.state

        assert state.engine is mocked_resources["engine"]
        assert state.session_factory is not None
        assert state.redis is mocked_resources["redis"]
        assert state.inference_client is mocked_resources["inference"]
        assert state.vector_store.client is mocked_resources["qdrant"]
        assert state.rag_service is not None
        assert state.document_service is not None
        assert state.conversation_service is not None

        # The collection is checked exactly once, at startup.
        assert mocked_resources["qdrant"].collection_exists_calls == 1

        # Nothing is torn down while the app is serving.
        assert mocked_resources["engine"].disposed is False

    assert mocked_resources["engine"].disposed is True
    assert mocked_resources["redis"].closed is True
    assert mocked_resources["qdrant"].closed is True
    assert mocked_resources["inference"].closed is True


async def test_preload_models_disabled_does_not_load_models(settings, mocked_resources):
    settings.preload_models = False
    application = create_app(settings=settings)

    async with lifespan(application):
        assert mocked_resources["embeddings"].load_calls == 0


async def test_preload_models_enabled_loads_once(settings, mocked_resources):
    settings.preload_models = True
    application = create_app(settings=settings)

    async with lifespan(application):
        assert mocked_resources["embeddings"].load_calls == 1


async def test_unreachable_qdrant_does_not_abort_startup(
    settings,
    mocked_resources,
    monkeypatch,
):
    async def failing_collection_exists(_self, _name):
        raise ConnectionError("qdrant is down")

    monkeypatch.setattr(
        FakeQdrantClient, "collection_exists", failing_collection_exists
    )

    application = create_app(settings=settings)

    # Startup still completes so that /health can report what is broken.
    async with lifespan(application):
        assert application.state.vector_store is not None

    assert mocked_resources["qdrant"].closed is True


async def test_a_failure_mid_startup_still_closes_earlier_resources(
    settings,
    mocked_resources,
    monkeypatch,
):
    def exploding_inference(*_args, **_kwargs):
        raise RuntimeError("cannot build the inference client")

    monkeypatch.setattr(app_module, "InferenceClient", exploding_inference)

    application = create_app(settings=settings)

    with pytest.raises(RuntimeError):
        async with lifespan(application):
            pass

    # The engine, Qdrant and Redis clients were created before the failure and
    # must not be leaked.
    assert mocked_resources["engine"].disposed is True
    assert mocked_resources["qdrant"].closed is True
    assert mocked_resources["redis"].closed is True

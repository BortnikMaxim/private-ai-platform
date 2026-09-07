import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from backend.api import conversations, documents, health, rag
from backend.config import Settings
from backend.config import settings as default_settings
from backend.db import create_engine
from backend.errors import AppError
from backend.services.conversation_service import ConversationService
from backend.services.document_service import DocumentService
from backend.services.embeddings import EmbeddingService
from backend.services.inference_client import InferenceClient
from backend.services.rag_service import RagService
from backend.services.vector_store import VectorStore

logger = logging.getLogger("backend")


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the long lived resources once and tear them down on shutdown."""
    settings: Settings = app.state.settings

    configure_logging(settings)

    if not settings.inference_api_key:
        logger.warning(
            "INFERENCE_API_KEY is not set; calls to the inference service "
            "will be rejected with 401"
        )

    # Every resource is registered with the exit stack as soon as it exists, so
    # a failure while building a later one still tears down the earlier ones.
    async with AsyncExitStack() as stack:
        db_engine = create_engine(settings.database_url)
        stack.push_async_callback(db_engine.dispose)

        qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)
        vector_store = VectorStore(
            client=qdrant_client,
            collection_name=settings.qdrant_collection,
            vector_size=settings.embedding_dim,
        )
        stack.push_async_callback(vector_store.aclose)

        redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        stack.push_async_callback(redis_client.aclose)

        inference_client = InferenceClient(
            base_url=settings.inference_url,
            api_key=settings.inference_api_key,
            timeout=settings.inference_timeout_seconds,
            health_timeout=settings.inference_health_timeout_seconds,
            default_max_tokens=settings.inference_max_tokens,
            default_temperature=settings.inference_temperature,
        )
        stack.push_async_callback(inference_client.aclose)

        embeddings = EmbeddingService(
            embedding_model=settings.embedding_model,
            reranker_model=settings.reranker_model,
        )

        rag_service = RagService(
            embeddings=embeddings,
            vector_store=vector_store,
            settings=settings,
        )

        document_service = DocumentService(
            embeddings=embeddings,
            vector_store=vector_store,
            settings=settings,
        )

        conversation_service = ConversationService(
            inference=inference_client,
            rag=rag_service,
            documents=document_service,
            settings=settings,
        )

        app.state.engine = db_engine
        app.state.session_factory = async_sessionmaker(
            bind=db_engine,
            expire_on_commit=False,
            autoflush=False,
        )
        app.state.redis = redis_client
        app.state.qdrant_client = qdrant_client
        app.state.vector_store = vector_store
        app.state.inference_client = inference_client
        app.state.embeddings = embeddings
        app.state.rag_service = rag_service
        app.state.document_service = document_service
        app.state.conversation_service = conversation_service

        # A missing collection or an unreachable Qdrant must not stop the API
        # from serving /health, which is how an operator finds out what broke.
        try:
            await vector_store.ensure_collection()
        except Exception:
            logger.exception("qdrant_collection_setup_failed")

        if settings.preload_models:
            try:
                await embeddings.ensure_loaded()
            except Exception:
                logger.exception("model_preload_failed")

        logger.info("backend_started version=%s", settings.app_version)

        try:
            yield
        finally:
            logger.info("backend_stopping")

    logger.info("backend_stopped")


async def app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Defensive: an `assert` here would be stripped under `python -O`.
    if not isinstance(exc, AppError):
        return await unhandled_error_handler(request, exc)

    if exc.status_code >= 500:
        logger.exception("app_error path=%s", request.url.path)
    else:
        logger.info(
            "app_error path=%s status=%d detail=%s",
            request.url.path,
            exc.status_code,
            exc.detail,
        )

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # The traceback belongs in the log, never in the response body.
    logger.exception("unhandled_error path=%s", request.url.path)

    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or default_settings

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )

    app.state.settings = settings

    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(conversations.router)
    app.include_router(rag.router)

    return app


app = create_app()

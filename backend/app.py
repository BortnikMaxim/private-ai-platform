import logging

import httpx
import redis.asyncio as redis
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from backend.config import settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")


app = FastAPI(
    title="Private AI Platform API",
    version="0.1.1",
)


engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
)

redis_client = redis.from_url(
    settings.redis_url,
    decode_responses=True,
)

qdrant_client = AsyncQdrantClient(
    url=settings.qdrant_url,
)


@app.get("/health")
async def health():
    results = {
        "api": "ok",
        "postgres": "unknown",
        "redis": "unknown",
        "qdrant": "unknown",
        "inference": "unknown",
    }

    try:
        async with engine.connect() as connection:
            result = await connection.execute(text("SELECT 1"))
            result.scalar_one()

        results["postgres"] = "ok"

    except Exception as exc:
        logger.exception("PostgreSQL health check failed: %s", exc)
        results["postgres"] = "error"

    try:
        await redis_client.ping()
        results["redis"] = "ok"

    except Exception as exc:
        logger.exception("Redis health check failed: %s", exc)
        results["redis"] = "error"

    try:
        await qdrant_client.get_collections()
        results["qdrant"] = "ok"

    except Exception as exc:
        logger.exception("Qdrant health check failed: %s", exc)
        results["qdrant"] = "error"

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{settings.inference_url}/health"
            )
            response.raise_for_status()

        results["inference"] = "ok"

    except Exception as exc:
        logger.exception("Inference health check failed: %s", exc)
        results["inference"] = "error"

    return results

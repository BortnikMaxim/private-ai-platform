import logging

from fastapi import APIRouter
from sqlalchemy import text

from backend.dependencies import EngineDep, InferenceDep, RedisDep, VectorStoreDep
from backend.schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    engine: EngineDep,
    redis_client: RedisDep,
    vector_store: VectorStoreDep,
    inference: InferenceDep,
) -> HealthResponse:
    """Aggregate liveness of every backing service.

    Always returns 200 so that a partially degraded stack is still readable.
    """
    results = {
        "api": "ok",
        "postgres": "error",
        "redis": "error",
        "qdrant": "error",
        "inference": "error",
    }

    try:
        async with engine.connect() as connection:
            result = await connection.execute(text("SELECT 1"))
            result.scalar_one()
        results["postgres"] = "ok"
    except Exception as exc:  # noqa: BLE001 - health checks never raise
        logger.warning("postgres_health_failed error=%s", type(exc).__name__)

    try:
        await redis_client.ping()
        results["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        logger.warning("redis_health_failed error=%s", type(exc).__name__)

    if await vector_store.health():
        results["qdrant"] = "ok"

    if await inference.health():
        results["inference"] = "ok"

    return HealthResponse(**results)

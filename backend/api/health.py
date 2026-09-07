import asyncio
import logging

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from backend.dependencies import (
    BrokerDep,
    EngineDep,
    InferenceDep,
    RedisDep,
    VectorStoreDep,
)
from backend.schemas import HealthResponse, WorkerHealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    engine: EngineDep,
    redis_client: RedisDep,
    vector_store: VectorStoreDep,
    inference: InferenceDep,
    broker: BrokerDep,
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
        "rabbitmq": "error",
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

    # Independent probes, so a slow one does not serialise behind the others.
    qdrant_ok, inference_ok, broker_ok = await asyncio.gather(
        vector_store.health(),
        inference.health(),
        broker.health(),
    )

    if qdrant_ok:
        results["qdrant"] = "ok"

    if inference_ok:
        results["inference"] = "ok"

    if broker_ok:
        results["rabbitmq"] = "ok"

    return HealthResponse(**results)


@router.get("/health/workers", response_model=WorkerHealthResponse)
async def worker_health(broker: BrokerDep) -> WorkerHealthResponse:
    """Broadcast ping to the Celery workers.

    Kept out of ``GET /health`` because it waits out its timeout whenever no
    worker answers, which would make the main health check slow exactly when
    the system is degraded.
    """
    workers = await broker.ping_workers()

    return WorkerHealthResponse(workers=workers, available=bool(workers))


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus metrics for the API process.

    Worker metrics are served separately by the worker itself; see the README.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

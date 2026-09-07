"""Celery application: RabbitMQ broker, Redis result backend.

Start a worker with (see README for the Apple Silicon caveats):

    celery -A backend.worker.celery_app worker --loglevel=info --pool=solo
"""

import logging

from celery import Celery
from celery.signals import worker_process_init, worker_ready

from backend.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

celery_app = Celery("private_ai", include=["backend.worker.tasks"])

celery_app.conf.update(
    broker_url=settings.celery_broker_url,
    result_backend=settings.celery_result_backend or None,
    task_default_queue=settings.celery_task_queue,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Acknowledge only after the task finishes, so a killed worker's job is
    # redelivered. That redelivery is safe because ingestion is idempotent.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # One long task at a time per worker; ingestion is CPU/GPU bound.
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    task_time_limit=settings.celery_task_time_limit,
    broker_connection_retry_on_startup=True,
    result_expires=3600,
    # Set by tests so tasks execute inline without a broker.
    task_always_eager=settings.celery_task_always_eager,
    task_eager_propagates=False,
)


@worker_process_init.connect
def reset_process_state(**_kwargs) -> None:
    """Drop inherited model handles after a prefork.

    Torch/MPS state must never be shared across a fork. Clearing the cached
    EmbeddingService makes each child load its own copy lazily instead of
    touching whatever the parent had already initialised.
    """
    from backend.worker import context

    context.reset_embeddings()


@worker_ready.connect
def start_metrics_server(**_kwargs) -> None:
    """Expose worker metrics when WORKER_METRICS_PORT is configured.

    Correct for a single-process pool (solo/threads). With a prefork pool of
    more than one child this would only report the parent's counters — see the
    observability notes in the README.
    """
    port = settings.worker_metrics_port

    if not port:
        return

    from prometheus_client import start_http_server

    try:
        start_http_server(port, addr="127.0.0.1")
        logger.info("worker_metrics_server_started port=%d", port)
    except OSError:
        logger.exception("worker_metrics_server_failed port=%d", port)

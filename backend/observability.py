"""Prometheus metrics and structured stage logging for document processing.

The metrics below are registered in whichever process imports this module. The
API process serves them at ``GET /metrics``; a Celery worker can serve its own
copy on ``WORKER_METRICS_PORT``. See the README for why the two are not merged.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import Counter, Histogram

logger = logging.getLogger("backend.processing")

DOCUMENTS_PROCESSING_TOTAL = Counter(
    "documents_processing_total",
    "Document processing attempts started",
)

DOCUMENT_PROCESSING_FAILURES_TOTAL = Counter(
    "document_processing_failures_total",
    "Document processing attempts that ended in failure",
    ["reason"],
)

DOCUMENT_PROCESSING_DURATION_SECONDS = Histogram(
    "document_processing_duration_seconds",
    "End to end document processing duration",
    ["outcome"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600),
)

DOCUMENT_PROCESSING_STAGE_SECONDS = Histogram(
    "document_processing_stage_seconds",
    "Duration of an individual document processing stage",
    ["stage"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 300),
)


@contextmanager
def stage(name: str, document_id: str, task_id: str | None = None) -> Iterator[dict]:
    """Time one processing stage and log its outcome.

    Yields a mutable dict for stage specific counters (pages, chunks, ...).
    Document text is deliberately never logged — only counts and sizes.
    """
    started = time.perf_counter()
    extra: dict = {}

    try:
        yield extra
    except Exception as exc:
        duration = time.perf_counter() - started
        DOCUMENT_PROCESSING_STAGE_SECONDS.labels(stage=name).observe(duration)
        logger.warning(
            "document_stage_failed stage=%s document_id=%s task_id=%s "
            "duration_ms=%.1f error=%s",
            name,
            document_id,
            task_id,
            duration * 1000,
            type(exc).__name__,
        )
        raise

    duration = time.perf_counter() - started
    DOCUMENT_PROCESSING_STAGE_SECONDS.labels(stage=name).observe(duration)

    details = " ".join(f"{key}={value}" for key, value in sorted(extra.items()))
    logger.info(
        "document_stage stage=%s document_id=%s task_id=%s duration_ms=%.1f%s",
        name,
        document_id,
        task_id,
        duration * 1000,
        f" {details}" if details else "",
    )

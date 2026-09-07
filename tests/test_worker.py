"""Celery wiring: task configuration, retry policy and the dispatcher.

Runs entirely without RabbitMQ — the task is invoked with ``apply()`` in eager
mode and its async body is stubbed out.
"""

import uuid

import pytest

from backend.errors import InvalidDocumentError, TransientProcessingError
from backend.worker import tasks as tasks_module
from backend.worker.celery_app import celery_app
from backend.worker.tasks import process_document_task


@pytest.fixture
def eager():
    previous = celery_app.conf.task_always_eager
    celery_app.conf.task_always_eager = True
    yield
    celery_app.conf.task_always_eager = previous


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_broker_is_rabbitmq_and_results_go_to_redis():
    assert celery_app.conf.broker_url.startswith("amqp://")
    assert celery_app.conf.result_backend.startswith("redis://")


def test_task_is_registered_under_a_stable_name():
    assert "documents.process" in celery_app.tasks
    assert process_document_task.name == "documents.process"


def test_late_acknowledgement_is_enabled():
    # Redelivery after a worker crash is only safe because ingestion is
    # idempotent; both settings belong together.
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True
    assert celery_app.conf.worker_prefetch_multiplier == 1


def test_retry_policy_uses_bounded_backoff_with_jitter():
    assert process_document_task.retry_jitter is True
    assert process_document_task.retry_backoff >= 1
    assert process_document_task.retry_backoff_max >= process_document_task.retry_backoff
    assert 0 < process_document_task.max_retries <= 20


def test_only_transient_errors_are_auto_retried():
    retryable = tasks_module.RETRYABLE

    assert TransientProcessingError in retryable
    # A corrupt PDF will never become valid, so it must not be retried.
    assert not issubclass(InvalidDocumentError, retryable)


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


async def _succeed(_settings, document_id, task_id=None):
    return {"document_id": str(document_id), "status": "ready", "chunks": 3}


def test_transient_failures_are_retried_until_they_pass(monkeypatch, eager):
    attempts = {"count": 0}

    async def flaky(_settings, document_id, task_id=None):
        attempts["count"] += 1

        if attempts["count"] < 3:
            raise TransientProcessingError("qdrant is unreachable")

        return {"document_id": str(document_id), "status": "ready", "chunks": 3}

    monkeypatch.setattr(tasks_module, "run_document_processing", flaky)
    # Retry immediately instead of sleeping through the real backoff.
    monkeypatch.setattr(process_document_task, "retry_backoff", 0, raising=False)
    monkeypatch.setattr(process_document_task, "default_retry_delay", 0, raising=False)

    result = process_document_task.apply(args=[str(uuid.uuid4())])

    assert attempts["count"] == 3
    assert result.successful()
    assert result.result["status"] == "ready"


def test_permanent_failures_are_not_retried(monkeypatch, eager):
    attempts = {"count": 0}

    async def permanent(_settings, document_id, task_id=None):
        # The task body handles permanent errors itself and reports failure
        # as a normal result, so the broker is not filled with dead retries.
        attempts["count"] += 1
        return {"document_id": str(document_id), "status": "failed", "error": "bad pdf"}

    monkeypatch.setattr(tasks_module, "run_document_processing", permanent)

    result = process_document_task.apply(args=[str(uuid.uuid4())])

    assert attempts["count"] == 1
    assert result.successful()
    assert result.result["status"] == "failed"


def test_task_passes_only_the_document_id_to_the_worker(monkeypatch, eager):
    seen = {}

    async def capture(_settings, document_id, task_id=None):
        seen["document_id"] = document_id
        seen["task_id"] = task_id
        return {"document_id": str(document_id), "status": "ready", "chunks": 0}

    monkeypatch.setattr(tasks_module, "run_document_processing", capture)

    document_id = uuid.uuid4()
    process_document_task.apply(args=[str(document_id)])

    assert seen["document_id"] == document_id
    assert isinstance(seen["document_id"], uuid.UUID)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_enqueues_the_id_and_returns_a_task_id(monkeypatch):
    from backend.worker.dispatch import CeleryTaskDispatcher

    sent = {}

    class Result:
        id = "task-42"

    def fake_delay(document_id):
        sent["document_id"] = document_id
        return Result()

    monkeypatch.setattr(
        "backend.worker.dispatch.process_document_task.delay", fake_delay
    )

    document_id = uuid.uuid4()
    task_id = CeleryTaskDispatcher().enqueue_document_processing(document_id)

    assert task_id == "task-42"
    # A string id travels through RabbitMQ, never the PDF payload.
    assert sent["document_id"] == str(document_id)


def test_dispatcher_revoke_never_raises(monkeypatch):
    from backend.worker.dispatch import CeleryTaskDispatcher

    def explode(_task_id):
        raise ConnectionError("broker down")

    monkeypatch.setattr("backend.worker.dispatch.celery_app.control.revoke", explode)

    # Revoking is best effort: a broker outage must not break DELETE.
    CeleryTaskDispatcher().revoke("task-42")

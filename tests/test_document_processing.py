"""The Celery ingestion pipeline, exercised without a broker.

``process_with`` is the task body with its resources injected, so every case
below runs against the same in-memory fakes as the API tests: no RabbitMQ, no
model downloads, no network.
"""

import io
import uuid

import pytest
from pypdf import PdfWriter
from sqlalchemy import select

from backend.errors import (
    DocumentGoneError,
    DocumentSourceMissingError,
    InvalidDocumentError,
    TransientProcessingError,
)
from backend.models import Document, DocumentChunk
from backend.services import document_processor as processor_module
from backend.services.document_processor import (
    AlreadyProcessedError,
    is_transient,
    point_id_for,
    safe_error_message,
)
from backend.worker.tasks import process_with

PAGES = [
    {"page": 1, "text": " ".join(f"альфа{index}" for index in range(60))},
    {"page": 2, "text": " ".join(f"бета{index}" for index in range(60))},
]


@pytest.fixture
def pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def extractable(monkeypatch):
    """Replace real PDF text extraction so the pipeline can run offline."""

    def fake_extract(_file_bytes):
        return 2, PAGES

    monkeypatch.setattr(processor_module, "extract_pdf_pages", fake_extract)


@pytest.fixture
async def queued_document(session_factory, storage, pdf_bytes):
    """A processing Document whose source PDF is on disk, as after POST."""
    document_id = uuid.uuid4()

    async with session_factory() as session:
        session.add(
            Document(
                id=document_id,
                filename="queued.pdf",
                original_filename="queued.pdf",
                content_type="application/pdf",
                size_bytes=len(pdf_bytes),
                status="processing",
                celery_task_id="task-1",
            )
        )
        await session.commit()

    await storage.save(document_id, pdf_bytes)

    return document_id


async def load_document(session_factory, document_id):
    async with session_factory() as session:
        return await session.get(Document, document_id)


async def load_chunks(session_factory, document_id):
    async with session_factory() as session:
        result = await session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


async def test_task_success_marks_the_document_ready(
    document_processor,
    session_factory,
    vector_store,
    queued_document,
    extractable,
):
    result = await process_with(
        document_processor, session_factory, queued_document, task_id="task-1"
    )

    assert result["status"] == "ready"
    assert result["chunks"] > 0

    document = await load_document(session_factory, queued_document)
    assert document.status == "ready"
    assert document.total_pages == 2
    assert document.extracted_pages == 2
    assert document.chunks_count == result["chunks"]
    assert document.error_message is None

    chunks = await load_chunks(session_factory, queued_document)
    assert len(chunks) == result["chunks"]
    assert len(vector_store.points) == result["chunks"]

    payload = next(iter(vector_store.points.values()))
    assert set(payload) >= {"document_id", "filename", "page", "chunk_index", "text"}
    assert payload["document_id"] == str(queued_document)
    assert payload["filename"] == "queued.pdf"


async def test_source_file_is_removed_after_success(
    document_processor,
    session_factory,
    storage,
    queued_document,
    extractable,
):
    assert storage.exists(queued_document)

    await process_with(document_processor, session_factory, queued_document)

    assert not storage.exists(queued_document)


async def test_source_file_is_kept_when_the_policy_says_so(
    document_processor,
    session_factory,
    storage,
    queued_document,
    extractable,
    settings,
):
    settings.delete_source_after_processing = False

    await process_with(document_processor, session_factory, queued_document)

    assert storage.exists(queued_document)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_point_ids_are_deterministic():
    document_id = uuid.uuid4()

    assert point_id_for(document_id, 3) == point_id_for(str(document_id), 3)
    assert point_id_for(document_id, 3) != point_id_for(document_id, 4)
    assert point_id_for(document_id, 3) != point_id_for(uuid.uuid4(), 3)


async def test_rerunning_the_task_does_not_duplicate_anything(
    document_processor,
    session_factory,
    vector_store,
    queued_document,
    extractable,
):
    first = await process_with(document_processor, session_factory, queued_document)

    chunks_after_first = await load_chunks(session_factory, queued_document)
    points_after_first = dict(vector_store.points)

    # A duplicate delivery of a finished job is skipped outright.
    second = await process_with(document_processor, session_factory, queued_document)
    assert second["status"] == "skipped"

    # Force a genuine re-run by putting the document back in the queue.
    async with session_factory() as session:
        document = await session.get(Document, queued_document)
        document.status = "processing"
        await session.commit()

    await document_processor.storage.save(queued_document, b"%PDF-1.4 reprocessed")
    third = await process_with(document_processor, session_factory, queued_document)

    assert third["status"] == "ready"
    assert third["chunks"] == first["chunks"]

    chunks_after_third = await load_chunks(session_factory, queued_document)
    assert len(chunks_after_third) == len(chunks_after_first)
    assert len(vector_store.points) == len(points_after_first)
    # Same deterministic ids, overwritten in place rather than added.
    assert set(vector_store.points) == set(points_after_first)


async def test_a_shorter_rerun_leaves_no_stale_vectors(
    document_processor,
    session_factory,
    vector_store,
    queued_document,
    extractable,
    monkeypatch,
):
    await process_with(document_processor, session_factory, queued_document)
    assert len(vector_store.points) > 1

    # Re-process the same document, this time yielding a single chunk.
    monkeypatch.setattr(
        processor_module,
        "extract_pdf_pages",
        lambda _bytes: (1, [{"page": 1, "text": "короткий документ"}]),
    )

    async with session_factory() as session:
        document = await session.get(Document, queued_document)
        document.status = "processing"
        await session.commit()

    await document_processor.storage.save(queued_document, b"%PDF-1.4 short")
    await process_with(document_processor, session_factory, queued_document)

    assert len(vector_store.points) == 1
    assert len(await load_chunks(session_factory, queued_document)) == 1


# ---------------------------------------------------------------------------
# Permanent failures
# ---------------------------------------------------------------------------


async def test_pdf_without_text_marks_the_document_failed(
    document_processor,
    session_factory,
    queued_document,
    monkeypatch,
):
    monkeypatch.setattr(processor_module, "extract_pdf_pages", lambda _b: (3, []))

    result = await process_with(document_processor, session_factory, queued_document)

    assert result["status"] == "failed"

    document = await load_document(session_factory, queued_document)
    assert document.status == "failed"
    assert "No text" in document.error_message
    # A traceback must never reach the database.
    assert "Traceback" not in document.error_message
    assert "File \"" not in document.error_message


async def test_corrupt_pdf_marks_the_document_failed(
    document_processor,
    session_factory,
    storage,
    queued_document,
):
    await storage.save(queued_document, b"%PDF-1.4\nnot really a pdf")

    result = await process_with(document_processor, session_factory, queued_document)

    assert result["status"] == "failed"
    document = await load_document(session_factory, queued_document)
    assert document.status == "failed"
    assert document.error_message


async def test_permanent_failure_cleans_up_partial_state(
    document_processor,
    session_factory,
    vector_store,
    storage,
    queued_document,
    monkeypatch,
):
    monkeypatch.setattr(processor_module, "extract_pdf_pages", lambda _b: (1, []))

    await process_with(document_processor, session_factory, queued_document)

    assert vector_store.points == {}
    assert await load_chunks(session_factory, queued_document) == []
    assert not storage.exists(queued_document)


async def test_missing_source_file_is_a_permanent_failure(
    document_processor,
    session_factory,
    storage,
    queued_document,
):
    await storage.delete(queued_document)

    result = await process_with(document_processor, session_factory, queued_document)

    assert result["status"] == "failed"
    document = await load_document(session_factory, queued_document)
    assert document.status == "failed"


# ---------------------------------------------------------------------------
# Transient failures and retry classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error,expected",
    [
        (TransientProcessingError("qdrant down"), True),
        (InvalidDocumentError("corrupt"), False),
        (DocumentSourceMissingError(), False),
        (DocumentGoneError(), False),
        (AlreadyProcessedError(), False),
        (ValueError("bug"), False),
    ],
)
def test_error_classification(error, expected):
    assert is_transient(error) is expected


def test_safe_error_message_never_contains_a_traceback():
    assert safe_error_message(InvalidDocumentError("bad pdf")) == "bad pdf"
    assert safe_error_message(RuntimeError("secret internals")) == (
        "Processing failed (RuntimeError)"
    )
    assert "secret internals" not in safe_error_message(RuntimeError("secret internals"))


async def test_qdrant_outage_raises_a_transient_error_for_retry(
    document_processor,
    session_factory,
    vector_store,
    queued_document,
    extractable,
):
    async def explode(_document_id):
        raise ConnectionError("qdrant is unreachable")

    vector_store.delete_document = explode

    with pytest.raises(TransientProcessingError):
        await process_with(document_processor, session_factory, queued_document)

    # The document stays queued so the retry can pick it up again.
    document = await load_document(session_factory, queued_document)
    assert document.status == "processing"
    assert document.error_message is None


async def test_transient_failure_does_not_consume_the_source_file(
    document_processor,
    session_factory,
    vector_store,
    storage,
    queued_document,
    extractable,
):
    async def explode(_records):
        raise ConnectionError("qdrant is unreachable")

    vector_store.upsert_chunks = explode

    with pytest.raises(TransientProcessingError):
        await process_with(document_processor, session_factory, queued_document)

    # The retry needs the PDF to still be there.
    assert storage.exists(queued_document)


# ---------------------------------------------------------------------------
# Deletion racing with processing
# ---------------------------------------------------------------------------


async def test_document_deleted_during_embedding_stops_before_indexing(
    document_processor,
    session_factory,
    vector_store,
    embeddings,
    queued_document,
    extractable,
):
    """DELETE lands mid-embedding; the guard before indexing must catch it."""
    original_embed = embeddings.embed_passages

    async def delete_then_embed(texts):
        vectors = await original_embed(texts)

        async with session_factory() as session:
            document = await session.get(Document, queued_document)
            await session.delete(document)
            await session.commit()

        return vectors

    embeddings.embed_passages = delete_then_embed

    result = await process_with(document_processor, session_factory, queued_document)

    assert result["status"] == "deleted"
    # The expensive embeddings ran, but nothing was written to Qdrant.
    assert vector_store.points == {}
    assert await load_chunks(session_factory, queued_document) == []


async def test_document_deleted_during_indexing_is_not_resurrected(
    document_processor,
    session_factory,
    vector_store,
    storage,
    queued_document,
    extractable,
):
    original_upsert = vector_store.upsert_chunks

    async def delete_during_upsert(records):
        await original_upsert(records)

        async with session_factory() as session:
            document = await session.get(Document, queued_document)
            await session.delete(document)
            await session.commit()

    vector_store.upsert_chunks = delete_during_upsert

    result = await process_with(document_processor, session_factory, queued_document)

    assert result["status"] == "deleted"

    # Everything the task wrote has been rolled back.
    assert vector_store.points == {}
    assert await load_document(session_factory, queued_document) is None
    assert await load_chunks(session_factory, queued_document) == []
    assert not storage.exists(queued_document)


async def test_task_for_an_unknown_document_is_a_no_op(
    document_processor,
    session_factory,
    vector_store,
):
    result = await process_with(document_processor, session_factory, uuid.uuid4())

    assert result["status"] == "deleted"
    assert vector_store.points == {}


async def test_task_skips_a_document_that_is_already_ready(
    document_processor,
    session_factory,
    seeded_document,
):
    result = await process_with(document_processor, session_factory, seeded_document)

    assert result["status"] == "skipped"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _duration_count(outcome: str) -> float:
    from prometheus_client import REGISTRY

    value = REGISTRY.get_sample_value(
        "document_processing_duration_seconds_count",
        {"outcome": outcome},
    )
    return value or 0.0


@pytest.mark.parametrize(
    "scenario,expected_outcome",
    [("ready", "ready"), ("skipped", "skipped"), ("deleted", "deleted")],
)
async def test_duration_is_labelled_with_the_real_outcome(
    document_processor,
    session_factory,
    storage,
    queued_document,
    seeded_document,
    extractable,
    scenario,
    expected_outcome,
):
    """A skipped duplicate and a deleted document are not failures."""
    before = _duration_count(expected_outcome)

    if scenario == "ready":
        await process_with(document_processor, session_factory, queued_document)
    elif scenario == "skipped":
        await process_with(document_processor, session_factory, seeded_document)
    else:
        await process_with(document_processor, session_factory, uuid.uuid4())

    assert _duration_count(expected_outcome) == before + 1

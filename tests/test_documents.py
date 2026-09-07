"""Document API tests.

POST /documents is now asynchronous: it validates, stores and enqueues, then
returns 202. The pipeline itself is covered by test_document_processing.py.
"""

import io
import uuid

import pytest
from pypdf import PdfWriter
from sqlalchemy import select

from backend.models import Document, DocumentChunk
from backend.services import document_service as document_service_module


@pytest.fixture
def blank_pdf_bytes() -> bytes:
    """A structurally valid PDF whose single page carries no extractable text."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)

    buffer = io.BytesIO()
    writer.write(buffer)

    return buffer.getvalue()


def upload(content: bytes, filename="doc.pdf", content_type="application/pdf"):
    return {"file": (filename, content, content_type)}


# ---------------------------------------------------------------------------
# Filename handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("report.pdf", "report.pdf"),
        # Non-ASCII names must survive: the sanitised name is what shows up as
        # `filename` in RAG sources.
        ("отчёт за Q3.pdf", "отчёт за Q3.pdf"),
        ("Ünïcödé.pdf", "Ünïcödé.pdf"),
        # Path traversal is stripped down to the bare name.
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("/absolute/path/doc.pdf", "doc.pdf"),
        # Control and separator characters are replaced.
        ("a\x00b:c.pdf", "a_b_c.pdf"),
        ("", "document.pdf"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert document_service_module.sanitize_filename(raw) == expected


def test_sanitize_filename_caps_length():
    assert len(document_service_module.sanitize_filename("x" * 400 + ".pdf")) == 255


# ---------------------------------------------------------------------------
# Accepting an upload
# ---------------------------------------------------------------------------


async def test_upload_is_accepted_and_queued(
    client,
    blank_pdf_bytes,
    task_dispatcher,
    storage,
    session_factory,
):
    response = await client.post("/documents", files=upload(blank_pdf_bytes))
    body = response.json()

    assert response.status_code == 202
    assert body["status"] == "processing"
    assert body["task_id"]
    assert uuid.UUID(body["document_id"])

    document_id = uuid.UUID(body["document_id"])

    # The task was queued with the id, never with the file contents.
    assert task_dispatcher.enqueued == [(str(document_id), body["task_id"])]

    # The bytes went to storage, not to PostgreSQL or the broker message.
    assert storage.exists(document_id)
    assert storage.path_for(document_id).read_bytes() == blank_pdf_bytes

    async with session_factory() as session:
        document = (await session.execute(select(Document))).scalar_one()

    assert document.status == "processing"
    assert document.chunks_count == 0
    assert document.celery_task_id == body["task_id"]
    assert document.size_bytes == len(blank_pdf_bytes)


async def test_upload_returns_before_any_indexing_happens(
    client,
    blank_pdf_bytes,
    vector_store,
    session_factory,
):
    await client.post("/documents", files=upload(blank_pdf_bytes))

    # Nothing is embedded or indexed while the request is being served.
    assert vector_store.points == {}

    async with session_factory() as session:
        chunks = (await session.execute(select(DocumentChunk))).scalars().all()

    assert chunks == []


async def test_uploaded_cyrillic_filename_is_preserved(client, blank_pdf_bytes, session_factory):
    response = await client.post(
        "/documents",
        files=upload(blank_pdf_bytes, filename="квартальный отчёт.pdf"),
    )

    assert response.status_code == 202

    async with session_factory() as session:
        document = (await session.execute(select(Document))).scalar_one()

    assert document.original_filename == "квартальный отчёт.pdf"
    assert document.filename == "квартальный отчёт.pdf"


# ---------------------------------------------------------------------------
# Validation that must stay synchronous, before anything is queued
# ---------------------------------------------------------------------------


async def test_non_pdf_extension_is_rejected_without_queuing(client, task_dispatcher):
    response = await client.post(
        "/documents",
        files=upload(b"%PDF-1.4 fake", filename="notes.txt", content_type="text/plain"),
    )

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]
    assert task_dispatcher.enqueued == []


async def test_wrong_content_type_is_rejected_without_queuing(client, task_dispatcher):
    response = await client.post(
        "/documents",
        files=upload(b"%PDF-1.4 fake", filename="doc.pdf", content_type="image/png"),
    )

    assert response.status_code == 400
    assert task_dispatcher.enqueued == []


async def test_file_without_pdf_magic_is_rejected_without_queuing(client, task_dispatcher):
    response = await client.post("/documents", files=upload(b"just some plain text"))

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]
    assert task_dispatcher.enqueued == []


async def test_empty_file_is_rejected_without_queuing(client, task_dispatcher):
    response = await client.post("/documents", files=upload(b""))

    assert response.status_code == 400
    assert task_dispatcher.enqueued == []


async def test_size_limit_is_enforced_before_enqueue(
    client,
    settings,
    task_dispatcher,
    storage,
    session_factory,
):
    """413 must happen in the request, not after a worker picks the job up."""
    oversized = b"%PDF-1.4" + b"0" * (settings.max_upload_size_bytes + 1024)

    response = await client.post("/documents", files=upload(oversized))

    assert response.status_code == 413
    assert task_dispatcher.enqueued == []

    # Nothing was stored and no row was created.
    assert list(storage.base_dir.glob("*.pdf")) == []

    async with session_factory() as session:
        assert (await session.execute(select(Document))).scalars().all() == []


# ---------------------------------------------------------------------------
# Reading document state
# ---------------------------------------------------------------------------


async def test_status_is_visible_while_processing(client, blank_pdf_bytes):
    accepted = (
        await client.post("/documents", files=upload(blank_pdf_bytes))
    ).json()

    response = await client.get(f"/documents/{accepted['document_id']}")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "processing"
    assert body["celery_task_id"] == accepted["task_id"]
    assert body["error_message"] is None


async def test_list_and_get_documents(client, seeded_document):
    listing = await client.get("/documents")

    assert listing.status_code == 200
    assert listing.json()["total"] == 1

    detail = await client.get(f"/documents/{seeded_document}")

    assert detail.status_code == 200
    assert detail.json()["status"] == "ready"
    assert detail.json()["chunks"] == []

    with_chunks = await client.get(f"/documents/{seeded_document}?include_chunks=true")
    chunks = with_chunks.json()["chunks"]

    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1, 2]


async def test_get_unknown_document_returns_404(client):
    response = await client.get(f"/documents/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


# ---------------------------------------------------------------------------
# Deleting
# ---------------------------------------------------------------------------


async def test_delete_document_removes_chunks_and_vectors(
    client,
    seeded_document,
    vector_store,
    session_factory,
):
    assert vector_store.points

    response = await client.delete(f"/documents/{seeded_document}")

    assert response.status_code == 200
    assert vector_store.points == {}

    async with session_factory() as session:
        assert (await session.execute(select(Document))).scalars().all() == []
        assert (await session.execute(select(DocumentChunk))).scalars().all() == []


async def test_delete_unknown_document_returns_404(client):
    response = await client.delete(f"/documents/{uuid.uuid4()}")

    assert response.status_code == 404


async def test_document_survives_a_failed_vector_deletion(
    client,
    seeded_document,
    vector_store,
    session_factory,
):
    vector_store.fail_on_delete = True

    response = await client.delete(f"/documents/{seeded_document}")

    assert response.status_code == 503

    # Nothing was removed, so the document and its vectors stay consistent.
    async with session_factory() as session:
        assert (await session.execute(select(Document))).scalar_one()


async def test_delete_also_removes_the_stored_source_file(
    client,
    blank_pdf_bytes,
    storage,
):
    accepted = (
        await client.post("/documents", files=upload(blank_pdf_bytes))
    ).json()
    document_id = uuid.UUID(accepted["document_id"])

    assert storage.exists(document_id)

    await client.delete(f"/documents/{document_id}")

    assert not storage.exists(document_id)


async def test_delete_while_processing_revokes_the_task(
    client,
    blank_pdf_bytes,
    task_dispatcher,
    storage,
    session_factory,
):
    accepted = (
        await client.post("/documents", files=upload(blank_pdf_bytes))
    ).json()
    document_id = uuid.UUID(accepted["document_id"])

    response = await client.delete(f"/documents/{document_id}")

    assert response.status_code == 200
    assert task_dispatcher.revoked == [accepted["task_id"]]

    # Row, file and vectors are all gone.
    assert not storage.exists(document_id)

    async with session_factory() as session:
        assert (await session.execute(select(Document))).scalars().all() == []


async def test_delete_of_a_finished_document_does_not_revoke_anything(
    client,
    seeded_document,
    task_dispatcher,
):
    await client.delete(f"/documents/{seeded_document}")

    # Nothing is in flight for a document that is already ready.
    assert task_dispatcher.revoked == []


async def test_a_revoked_document_is_not_resurrected_by_a_late_worker(
    client,
    blank_pdf_bytes,
    document_processor,
    session_factory,
    vector_store,
    storage,
):
    """The worker may already be past the revoke; the missing row must stop it."""
    from backend.worker.tasks import process_with

    accepted = (
        await client.post("/documents", files=upload(blank_pdf_bytes))
    ).json()
    document_id = uuid.UUID(accepted["document_id"])

    await client.delete(f"/documents/{document_id}")

    # The task runs anyway, as it would if it had been dequeued already.
    result = await process_with(document_processor, session_factory, document_id)

    assert result["status"] == "deleted"
    assert vector_store.points == {}
    assert not storage.exists(document_id)

    async with session_factory() as session:
        assert (await session.execute(select(Document))).scalars().all() == []

    assert (await client.get(f"/documents/{document_id}")).status_code == 404

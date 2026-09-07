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


@pytest.fixture
def text_pdf(monkeypatch, blank_pdf_bytes):
    """Bypass real PDF text extraction so ingestion can be tested offline."""

    def fake_extract(_file_bytes):
        return 2, [
            {"page": 1, "text": " ".join(f"альфа{index}" for index in range(60))},
            {"page": 2, "text": " ".join(f"бета{index}" for index in range(60))},
        ]

    monkeypatch.setattr(document_service_module, "extract_pdf_pages", fake_extract)

    return blank_pdf_bytes


def upload(content: bytes, filename="doc.pdf", content_type="application/pdf"):
    return {"file": (filename, content, content_type)}


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


async def test_uploaded_cyrillic_filename_is_preserved(client, text_pdf, vector_store):
    response = await client.post(
        "/documents",
        files=upload(text_pdf, filename="квартальный отчёт.pdf"),
    )
    body = response.json()

    assert response.status_code == 201
    assert body["original_filename"] == "квартальный отчёт.pdf"
    assert body["filename"] == "квартальный отчёт.pdf"

    payload = next(iter(vector_store.points.values()))
    assert payload["filename"] == "квартальный отчёт.pdf"


async def test_upload_indexes_the_document(client, text_pdf, vector_store, session_factory):
    response = await client.post("/documents", files=upload(text_pdf))
    body = response.json()

    assert response.status_code == 201
    assert body["status"] == "ready"
    assert body["total_pages"] == 2
    assert body["extracted_pages"] == 2
    assert body["chunks_count"] > 0
    assert body["error_message"] is None
    assert body["original_filename"] == "doc.pdf"

    # Chunk metadata in PostgreSQL and vectors in Qdrant must agree.
    async with session_factory() as session:
        chunks = (await session.execute(select(DocumentChunk))).scalars().all()

    assert len(chunks) == body["chunks_count"]
    assert len(vector_store.points) == body["chunks_count"]

    payload = next(iter(vector_store.points.values()))
    assert set(payload) >= {
        "document_id",
        "filename",
        "page",
        "chunk_index",
        "text",
    }
    assert payload["document_id"] == body["id"]


async def test_legacy_upload_endpoint_keeps_its_response_shape(client, text_pdf):
    response = await client.post("/documents/upload", files=upload(text_pdf))
    body = response.json()

    assert response.status_code == 200
    assert set(body) == {
        "document_id",
        "filename",
        "total_pages",
        "extracted_pages",
        "chunks",
    }


async def test_upload_rejects_non_pdf_extension(client):
    response = await client.post(
        "/documents",
        files=upload(b"%PDF-1.4 fake", filename="notes.txt", content_type="text/plain"),
    )

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


async def test_upload_rejects_wrong_content_type(client):
    response = await client.post(
        "/documents",
        files=upload(b"%PDF-1.4 fake", filename="doc.pdf", content_type="image/png"),
    )

    assert response.status_code == 400


async def test_upload_rejects_a_file_that_is_not_really_a_pdf(client):
    response = await client.post("/documents", files=upload(b"just some plain text"))

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


async def test_upload_rejects_empty_file(client):
    response = await client.post("/documents", files=upload(b""))

    assert response.status_code == 400


async def test_upload_rejects_a_corrupt_pdf(client):
    response = await client.post("/documents", files=upload(b"%PDF-1.4\nbroken"))

    assert response.status_code == 400


async def test_upload_of_a_pdf_without_text_fails_and_is_recorded(
    client,
    blank_pdf_bytes,
    session_factory,
):
    response = await client.post("/documents", files=upload(blank_pdf_bytes))

    assert response.status_code == 400

    # The attempt stays visible in the document list with its failure reason.
    async with session_factory() as session:
        document = (await session.execute(select(Document))).scalar_one()

    assert document.status == "failed"
    assert document.error_message
    assert "traceback" not in response.text.lower()


async def test_upload_larger_than_the_limit_returns_413(client, settings):
    oversized = b"%PDF-1.4" + b"0" * (settings.max_upload_size_bytes + 1024)

    response = await client.post("/documents", files=upload(oversized))

    assert response.status_code == 413


async def test_list_and_get_documents(client, seeded_document):
    listing = await client.get("/documents")

    assert listing.status_code == 200
    assert listing.json()["total"] == 1

    detail = await client.get(f"/documents/{seeded_document}")

    assert detail.status_code == 200
    assert detail.json()["chunks"] == []

    with_chunks = await client.get(f"/documents/{seeded_document}?include_chunks=true")
    chunks = with_chunks.json()["chunks"]

    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1, 2]


async def test_get_unknown_document_returns_404(client):
    response = await client.get(f"/documents/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


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

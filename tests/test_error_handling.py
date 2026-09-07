"""Regression tests for the HTTP status code of every domain error.

The rule under test: a client gets a precise status code and a plain
``{"detail": ...}`` body, never a Python traceback.
"""

import io
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from pypdf import PdfWriter

RANDOM_ID = "11111111-1111-1111-1111-111111111111"
INVALID_ID = "not-a-uuid"


def assert_clean_error(response, status_code: int):
    assert response.status_code == status_code

    body = response.text.lower()
    for leak in ("traceback", "file \"/", "sqlalchemy.exc", "self._adapt_connection"):
        assert leak not in body, f"response leaked internals: {response.text[:400]}"


# ---------------------------------------------------------------------------
# 404 — unknown conversation
# ---------------------------------------------------------------------------


async def test_get_unknown_conversation_is_404(client):
    response = await client.get(f"/conversations/{RANDOM_ID}")

    assert_clean_error(response, 404)
    assert response.json() == {"detail": "Conversation not found"}


async def test_delete_unknown_conversation_is_404(client):
    response = await client.delete(f"/conversations/{RANDOM_ID}")

    assert_clean_error(response, 404)
    assert response.json() == {"detail": "Conversation not found"}


async def test_post_message_to_unknown_conversation_is_404(client):
    response = await client.post(
        f"/conversations/{RANDOM_ID}/messages",
        json={"content": "привет"},
    )

    assert_clean_error(response, 404)
    assert response.json() == {"detail": "Conversation not found"}


async def test_list_messages_of_unknown_conversation_is_404(client):
    response = await client.get(f"/conversations/{RANDOM_ID}/messages")

    assert_clean_error(response, 404)


# ---------------------------------------------------------------------------
# 404 — unknown document
# ---------------------------------------------------------------------------


async def test_get_unknown_document_is_404(client):
    response = await client.get(f"/documents/{RANDOM_ID}")

    assert_clean_error(response, 404)
    assert response.json() == {"detail": "Document not found"}


async def test_delete_unknown_document_is_404(client):
    response = await client.delete(f"/documents/{RANDOM_ID}")

    assert_clean_error(response, 404)
    assert response.json() == {"detail": "Document not found"}


async def test_rag_with_unknown_document_id_is_404(client):
    conversation = (
        await client.post("/conversations", json={"title": "t"})
    ).json()

    response = await client.post(
        f"/conversations/{conversation['id']}/messages",
        json={"content": "q", "use_rag": True, "document_ids": [str(uuid.uuid4())]},
    )

    assert_clean_error(response, 404)


# ---------------------------------------------------------------------------
# 422 — malformed UUID must be a validation error, never a 500
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", f"/conversations/{INVALID_ID}"),
        ("DELETE", f"/conversations/{INVALID_ID}"),
        ("GET", f"/conversations/{INVALID_ID}/messages"),
        ("GET", f"/documents/{INVALID_ID}"),
        ("DELETE", f"/documents/{INVALID_ID}"),
    ],
)
async def test_malformed_uuid_in_path_is_422(client, method, path):
    response = await client.request(method, path)

    assert_clean_error(response, 422)
    assert response.json()["detail"][0]["type"] == "uuid_parsing"


async def test_malformed_uuid_when_posting_a_message_is_422(client):
    response = await client.post(
        f"/conversations/{INVALID_ID}/messages",
        json={"content": "привет"},
    )

    assert_clean_error(response, 422)


async def test_malformed_uuid_in_document_ids_is_422(client):
    conversation = (
        await client.post("/conversations", json={"title": "t"})
    ).json()

    response = await client.post(
        f"/conversations/{conversation['id']}/messages",
        json={"content": "q", "use_rag": True, "document_ids": ["nope"]},
    )

    assert_clean_error(response, 422)


# ---------------------------------------------------------------------------
# 400 / 413 — uploads
# ---------------------------------------------------------------------------


async def test_non_pdf_upload_is_400(client):
    response = await client.post(
        "/documents",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )

    assert_clean_error(response, 400)


async def test_upload_without_pdf_magic_is_400(client):
    response = await client.post(
        "/documents",
        files={"file": ("fake.pdf", b"not a pdf at all", "application/pdf")},
    )

    assert_clean_error(response, 400)


async def test_structurally_valid_but_unparseable_pdf_is_accepted_then_failed(client):
    """Deep validation moved to the worker; the request only does cheap checks.

    A file that starts with %PDF cannot be rejected without parsing it, so it
    is accepted and the failure surfaces as ``status: failed`` on the document.
    """
    response = await client.post(
        "/documents",
        files={"file": ("broken.pdf", b"%PDF-1.4\nnot really", "application/pdf")},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "processing"


async def test_deprecated_sync_upload_still_reports_a_bad_pdf_as_400(client):
    """The compatibility endpoint parses inline, so it can still answer 400."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)

    response = await client.post(
        "/documents/upload",
        files={"file": ("blank.pdf", buffer.getvalue(), "application/pdf")},
    )

    assert_clean_error(response, 400)


async def test_oversized_upload_is_413(client, settings):
    payload = b"%PDF-1.4" + b"0" * (settings.max_upload_size_bytes + 1024)

    response = await client.post(
        "/documents",
        files={"file": ("big.pdf", payload, "application/pdf")},
    )

    assert_clean_error(response, 413)


# ---------------------------------------------------------------------------
# 502 / 503 — downstream failures
# ---------------------------------------------------------------------------


async def test_unavailable_inference_is_502(client, inference):
    conversation = (
        await client.post("/conversations", json={"title": "t"})
    ).json()
    inference.available = False

    response = await client.post(
        f"/conversations/{conversation['id']}/messages",
        json={"content": "вопрос"},
    )

    assert_clean_error(response, 502)
    assert response.json() == {"detail": "Inference service is unavailable"}


async def test_unavailable_inference_on_rag_ask_is_502(client, seeded_document, inference):
    inference.available = False

    response = await client.post("/rag/ask", json={"question": "вопрос"})

    assert_clean_error(response, 502)


async def test_unavailable_vector_store_on_delete_is_503(
    client,
    seeded_document,
    vector_store,
):
    vector_store.fail_on_delete = True

    response = await client.delete(f"/documents/{seeded_document}")

    assert_clean_error(response, 503)


# ---------------------------------------------------------------------------
# 500 — an unexpected failure is still a clean JSON body
# ---------------------------------------------------------------------------


async def test_unexpected_error_returns_500_without_a_traceback(app, vector_store):
    async def explode(*args, **kwargs):
        raise RuntimeError("simulated internal failure with secret details")

    vector_store.search = explode

    # raise_app_exceptions=False mirrors what a real ASGI server does: the
    # response is delivered to the client and the exception goes to the logs.
    transport = ASGITransport(app=app, raise_app_exceptions=False)

    async with AsyncClient(transport=transport, base_url="http://test") as raw_client:
        response = await raw_client.post("/rag/retrieve", json={"question": "q"})

    assert_clean_error(response, 500)
    assert response.json() == {"detail": "Internal server error"}
    assert "simulated internal failure" not in response.text

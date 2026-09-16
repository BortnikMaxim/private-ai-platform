"""Multi-tenancy: one user must never observe another user's data.

A foreign resource is reported as 404 rather than 403 throughout, so the API
never confirms that somebody else's UUID exists.
"""

import io
import uuid

import pytest
from pypdf import PdfWriter
from sqlalchemy import select

from backend.models import Conversation, Document


@pytest.fixture
def pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


async def make_conversation(http, title="private") -> str:
    response = await http.post("/conversations", json={"title": title})
    assert response.status_code == 201
    return response.json()["id"]


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


async def test_a_conversation_is_invisible_to_another_user(client, other_client):
    mine = await make_conversation(client, "alice private")

    response = await other_client.get(f"/conversations/{mine}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found"


async def test_a_foreign_conversation_cannot_be_deleted(
    client,
    other_client,
    session_factory,
):
    mine = await make_conversation(client)

    assert (await other_client.delete(f"/conversations/{mine}")).status_code == 404

    # Still there for its owner.
    assert (await client.get(f"/conversations/{mine}")).status_code == 200

    async with session_factory() as session:
        assert (await session.execute(select(Conversation))).scalars().all()


async def test_a_foreign_conversation_cannot_receive_messages(
    client,
    other_client,
    inference,
    session_factory,
):
    mine = await make_conversation(client)

    response = await other_client.post(
        f"/conversations/{mine}/messages",
        json={"content": "инъекция в чужой диалог"},
    )

    assert response.status_code == 404

    # Nothing was written into somebody else's transcript.
    detail = (await client.get(f"/conversations/{mine}")).json()
    assert detail["messages"] == []


async def test_a_foreign_conversation_cannot_be_used_by_the_agent(
    client,
    other_client,
    inference,
):
    mine = await make_conversation(client)

    response = await other_client.post(
        f"/conversations/{mine}/agent",
        json={"content": "чужой агентский запрос"},
    )

    assert response.status_code == 404


async def test_a_foreign_conversation_messages_are_not_listable(client, other_client):
    mine = await make_conversation(client)

    assert (
        await other_client.get(f"/conversations/{mine}/messages")
    ).status_code == 404


async def test_listing_shows_only_your_own_conversations(client, other_client):
    await make_conversation(client, "alice one")
    await make_conversation(client, "alice two")
    await make_conversation(other_client, "bob one")

    mine = (await client.get("/conversations")).json()
    theirs = (await other_client.get("/conversations")).json()

    assert mine["total"] == 2
    assert {item["title"] for item in mine["items"]} == {"alice one", "alice two"}

    assert theirs["total"] == 1
    assert {item["title"] for item in theirs["items"]} == {"bob one"}


async def test_deleting_your_own_conversation_still_works(client):
    mine = await make_conversation(client)

    assert (await client.delete(f"/conversations/{mine}")).status_code == 200
    assert (await client.get(f"/conversations/{mine}")).status_code == 404


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


async def test_a_document_is_invisible_to_another_user(
    client,
    other_client,
    seeded_document,
):
    response = await other_client.get(f"/documents/{seeded_document}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"

    # Its owner sees it.
    assert (await client.get(f"/documents/{seeded_document}")).status_code == 200


async def test_a_foreign_document_cannot_be_deleted(
    client,
    other_client,
    seeded_document,
    vector_store,
    session_factory,
):
    assert (await other_client.delete(f"/documents/{seeded_document}")).status_code == 404

    # Neither the row nor the vectors were touched.
    async with session_factory() as session:
        assert (await session.execute(select(Document))).scalar_one()

    assert vector_store.points


async def test_listing_shows_only_your_own_documents(
    client,
    other_client,
    make_document,
    user,
    other_user,
):
    await make_document(user, filename="alice.pdf")
    await make_document(other_user, filename="bob.pdf")

    mine = (await client.get("/documents")).json()
    theirs = (await other_client.get("/documents")).json()

    assert {item["filename"] for item in mine["items"]} == {"alice.pdf"}
    assert mine["total"] == 1

    assert {item["filename"] for item in theirs["items"]} == {"bob.pdf"}
    assert theirs["total"] == 1


async def test_an_upload_is_owned_by_the_uploader(
    client,
    pdf_bytes,
    user,
    session_factory,
):
    response = await client.post(
        "/documents",
        files={"file": ("doc.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 202

    async with session_factory() as session:
        document = (await session.execute(select(Document))).scalar_one()

    assert document.user_id == user.id


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


async def test_retrieval_never_crosses_the_tenant_boundary(
    client,
    other_client,
    make_document,
    user,
    other_user,
):
    await make_document(user, filename="alice.pdf", texts=["альфа секрет альфа"])
    await make_document(other_user, filename="bob.pdf", texts=["бета секрет бета"])

    mine = (
        await client.post("/rag/retrieve", json={"question": "секрет"})
    ).json()
    theirs = (
        await other_client.post("/rag/retrieve", json={"question": "секрет"})
    ).json()

    assert {hit["filename"] for hit in mine["vector_results"]} == {"alice.pdf"}
    assert {hit["filename"] for hit in theirs["vector_results"]} == {"bob.pdf"}

    assert not any("бета" in (hit["text"] or "") for hit in mine["vector_results"])
    assert not any("альфа" in (hit["text"] or "") for hit in theirs["vector_results"])


async def test_the_tenant_filter_is_sent_to_the_vector_store(
    client,
    seeded_document,
    vector_store,
    user,
):
    await client.post("/rag/retrieve", json={"question": "проекты"})

    # The filter travels with the query rather than being applied afterwards.
    assert vector_store.searches[-1]["user_id"] == str(user.id)


async def test_a_foreign_document_id_in_the_body_yields_nothing(
    other_client,
    seeded_document,
    make_document,
    other_user,
):
    """A caller may name any UUID; the tenant filter still applies on top."""
    await make_document(other_user, filename="bob.pdf")

    response = await other_client.post(
        "/rag/retrieve",
        json={"question": "проекты", "document_ids": [str(seeded_document)]},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["vector_results"] == []
    assert body["reranked_results"] == []


async def test_rag_ask_is_scoped(client, other_client, seeded_document, inference):
    assert (
        await client.post("/rag/ask", json={"question": "Какие проекты?"})
    ).status_code == 200

    # Bob has no documents at all, so there is nothing to ground on.
    response = await other_client.post("/rag/ask", json={"question": "Какие проекты?"})

    assert response.status_code == 404
    assert response.json()["detail"] == "No relevant documents found"


async def test_conversation_rag_is_scoped(client, other_client, seeded_document, inference):
    theirs = await make_conversation(other_client)

    response = await other_client.post(
        f"/conversations/{theirs}/messages",
        json={"content": "Какие проекты?", "use_rag": True},
    )

    assert response.status_code == 201
    assert response.json()["sources"] == []


async def test_naming_a_foreign_document_in_a_message_is_404(
    other_client,
    seeded_document,
):
    theirs = await make_conversation(other_client)

    response = await other_client.post(
        f"/conversations/{theirs}/messages",
        json={
            "content": "вопрос",
            "use_rag": True,
            "document_ids": [str(seeded_document)],
        },
    )

    # Same answer as for a UUID that does not exist anywhere.
    assert response.status_code == 404
    assert "Unknown document ids" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------


async def test_the_metadata_tool_cannot_read_a_foreign_document(
    tool_registry,
    session_factory,
    rag_service,
    document_service,
    settings,
    seeded_document,
    other_user,
):
    from backend.agent.tools.base import ToolContext

    async with session_factory() as session:
        context = ToolContext(
            session=session,
            rag_service=rag_service,
            document_service=document_service,
            settings=settings,
            user_id=str(other_user.id),
        )

        result = await tool_registry.execute(
            "get_document_metadata",
            {"document_id": str(seeded_document)},
            context,
        )

    assert result["success"] is False
    assert "not found" in result["error"]
    # No metadata leaked through the error message.
    assert "projects.pdf" not in result["error"]


async def test_the_search_tool_is_scoped_to_the_current_tenant(
    tool_registry,
    rag_service,
    document_service,
    settings,
    seeded_document,
    other_user,
    vector_store,
):
    from backend.agent.tools.base import ToolContext

    context = ToolContext(
        rag_service=rag_service,
        document_service=document_service,
        settings=settings,
        user_id=str(other_user.id),
    )

    result = await tool_registry.execute(
        "search_documents",
        {"query": "проекты"},
        context,
    )

    assert result["success"] is True
    assert result["result"]["matches"] == 0
    assert vector_store.searches[-1]["user_id"] == str(other_user.id)


async def test_the_search_tool_refuses_to_run_without_a_tenant(
    tool_registry,
    rag_service,
    settings,
    seeded_document,
):
    from backend.agent.tools.base import ToolContext

    context = ToolContext(rag_service=rag_service, settings=settings, user_id=None)

    result = await tool_registry.execute(
        "search_documents", {"query": "проекты"}, context
    )

    # Failing closed: an unscoped search would read every tenant's chunks.
    assert result["success"] is False
    assert "authenticated user" in result["error"]


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


async def test_the_vector_store_refuses_an_untagged_chunk(vector_store):
    with pytest.raises(ValueError, match="user_id"):
        await vector_store.upsert_chunks(
            [
                {
                    "point_id": str(uuid.uuid4()),
                    "vector": [0.0] * 8,
                    "document_id": str(uuid.uuid4()),
                    "filename": "x.pdf",
                    "page": 1,
                    "chunk_index": 0,
                    "text": "тело",
                }
            ]
        )


async def test_a_search_without_a_tenant_is_refused(vector_store):
    with pytest.raises(ValueError, match="user_id"):
        await vector_store.search(vector=[0.0] * 8, limit=5, user_id="")

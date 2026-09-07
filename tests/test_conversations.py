import uuid

from sqlalchemy import select

from backend.models import Message


async def create_conversation(client, title="Тестовый диалог"):
    response = await client.post("/conversations", json={"title": title})
    assert response.status_code == 201
    return response.json()


async def test_create_and_get_conversation(client):
    created = await create_conversation(client)

    assert created["title"] == "Тестовый диалог"
    assert created["user_id"] is None

    response = await client.get(f"/conversations/{created['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]
    assert response.json()["messages"] == []


async def test_list_conversations(client):
    await create_conversation(client, "первый")
    await create_conversation(client, "второй")

    response = await client.get("/conversations")
    body = response.json()

    assert response.status_code == 200
    assert body["total"] == 2
    assert {item["title"] for item in body["items"]} == {"первый", "второй"}


async def test_get_unknown_conversation_returns_404(client):
    response = await client.get(f"/conversations/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found"


async def test_delete_conversation_removes_it_and_its_messages(
    client,
    inference,
    session_factory,
):
    conversation = await create_conversation(client)

    await client.post(
        f"/conversations/{conversation['id']}/messages",
        json={"content": "привет"},
    )

    response = await client.delete(f"/conversations/{conversation['id']}")
    assert response.status_code == 200
    assert response.json()["deleted"] is True

    assert (await client.get(f"/conversations/{conversation['id']}")).status_code == 404

    async with session_factory() as session:
        remaining = (await session.execute(select(Message))).scalars().all()

    assert remaining == []


async def test_delete_unknown_conversation_returns_404(client):
    response = await client.delete(f"/conversations/{uuid.uuid4()}")

    assert response.status_code == 404


async def test_post_message_persists_both_turns(client, inference):
    conversation = await create_conversation(client)

    response = await client.post(
        f"/conversations/{conversation['id']}/messages",
        json={"content": "Как дела?"},
    )
    body = response.json()

    assert response.status_code == 201
    assert body["message"]["role"] == "assistant"
    assert body["message"]["content"] == inference.answer
    assert body["used_rag"] is False
    assert body["sources"] == []

    detail = (await client.get(f"/conversations/{conversation['id']}")).json()

    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][0]["content"] == "Как дела?"


async def test_post_message_sends_system_prompt_and_history(client, inference):
    conversation = await create_conversation(client)

    for text in ("первый вопрос", "второй вопрос"):
        await client.post(
            f"/conversations/{conversation['id']}/messages",
            json={"content": text},
        )

    last_call = inference.calls[-1]

    assert last_call[0]["role"] == "system"
    assert [message["role"] for message in last_call[1:]] == [
        "user",
        "assistant",
        "user",
    ]
    assert last_call[-1]["content"] == "второй вопрос"


async def test_history_sent_to_the_model_is_capped(client, inference, settings):
    conversation = await create_conversation(client)

    for index in range(5):
        await client.post(
            f"/conversations/{conversation['id']}/messages",
            json={"content": f"вопрос {index}"},
        )

    history = inference.calls[-1][1:]  # drop the system prompt

    assert len(history) <= settings.chat_history_limit

    # Everything persisted, only the prompt window is trimmed.
    detail = (await client.get(f"/conversations/{conversation['id']}")).json()
    assert len(detail["messages"]) == 10


async def test_post_message_to_unknown_conversation_returns_404(client):
    response = await client.post(
        f"/conversations/{uuid.uuid4()}/messages",
        json={"content": "привет"},
    )

    assert response.status_code == 404


async def test_post_message_rejects_empty_content(client):
    conversation = await create_conversation(client)

    response = await client.post(
        f"/conversations/{conversation['id']}/messages",
        json={"content": ""},
    )

    assert response.status_code == 422


async def test_inference_outage_returns_502_but_keeps_the_user_turn(
    client,
    inference,
    session_factory,
):
    conversation = await create_conversation(client)
    inference.available = False

    response = await client.post(
        f"/conversations/{conversation['id']}/messages",
        json={"content": "вопрос в офлайне"},
    )

    assert response.status_code == 502
    assert "traceback" not in response.text.lower()

    async with session_factory() as session:
        messages = (await session.execute(select(Message))).scalars().all()

    assert [message.role for message in messages] == ["user"]


async def test_rag_turn_returns_sources(client, inference, seeded_document):
    conversation = await create_conversation(client)

    response = await client.post(
        f"/conversations/{conversation['id']}/messages",
        json={"content": "Какие проекты описаны?", "use_rag": True},
    )
    body = response.json()

    assert response.status_code == 201
    assert body["used_rag"] is True
    assert body["sources"]

    source = body["sources"][0]
    assert source["document_id"] == str(seeded_document)
    assert source["filename"] == "projects.pdf"
    assert source["page"] == 1
    assert source["chunk_index"] is not None
    assert source["vector_score"] is not None
    assert source["rerank_score"] is not None

    # The grounded context is injected into the final user turn.
    assert "КОНТЕКСТ" in inference.calls[-1][-1]["content"]
    assert "[SOURCE 1]" in inference.calls[-1][-1]["content"]


async def test_rag_turn_scopes_retrieval_to_requested_documents(
    client,
    seeded_document,
    vector_store,
):
    conversation = await create_conversation(client)

    response = await client.post(
        f"/conversations/{conversation['id']}/messages",
        json={
            "content": "Какие проекты описаны?",
            "use_rag": True,
            "document_ids": [str(seeded_document)],
        },
    )

    assert response.status_code == 201
    assert vector_store.searches[-1]["document_ids"] == [str(seeded_document)]


async def test_rag_turn_with_unknown_document_returns_404(client):
    conversation = await create_conversation(client)

    response = await client.post(
        f"/conversations/{conversation['id']}/messages",
        json={
            "content": "вопрос",
            "use_rag": True,
            "document_ids": [str(uuid.uuid4())],
        },
    )

    assert response.status_code == 404
    assert "Unknown document ids" in response.json()["detail"]


async def test_rag_turn_without_matches_still_answers(client, inference):
    conversation = await create_conversation(client)

    response = await client.post(
        f"/conversations/{conversation['id']}/messages",
        json={"content": "вопрос без документов", "use_rag": True},
    )

    assert response.status_code == 201
    assert response.json()["sources"] == []
    # The model is told there is nothing to ground on rather than being left
    # to invent an answer.
    assert "не найдено" in inference.calls[-1][-1]["content"]

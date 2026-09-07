"""POST /conversations/{id}/agent — persistence, contract and what must not leak."""

import json
import uuid

from sqlalchemy import select

from backend.models import Message
from backend.prompts.agent import (
    ROUTER_SYSTEM_PROMPT,
    TOOL_SELECTION_SYSTEM_PROMPT,
)


def route_reply(
    route: str,
    tool_name: str | None = None,
    reason: str = "секретное рассуждение",
) -> str:
    payload: dict = {"route": route, "reason": reason}
    if tool_name:
        payload["tool_name"] = tool_name
    return json.dumps(payload)


def tool_reply(tool: str, **arguments) -> str:
    return json.dumps({"tool": tool, "arguments": arguments})


async def make_conversation(client) -> str:
    response = await client.post("/conversations", json={"title": "Агент"})
    assert response.status_code == 201
    return response.json()["id"]


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


async def test_direct_answer_turn(client, inference):
    conversation = await make_conversation(client)
    inference.script(route_reply("direct_answer"), "Эмбеддинги — это векторы.")

    response = await client.post(
        f"/conversations/{conversation}/agent",
        json={"content": "Что такое embeddings?", "use_rag": False},
    )
    body = response.json()

    assert response.status_code == 201
    assert body["route"] == "direct_answer"
    assert body["message"]["role"] == "assistant"
    assert body["message"]["content"] == "Эмбеддинги — это векторы."
    assert body["tools_used"] == []
    assert body["sources"] == []


async def test_rag_turn_returns_sources(client, inference, seeded_document):
    conversation = await make_conversation(client)
    inference.script(route_reply("rag_search"), "Проект Борей — логистика.")

    response = await client.post(
        f"/conversations/{conversation}/agent",
        json={"content": "Какие проекты описаны?", "use_rag": True},
    )
    body = response.json()

    assert response.status_code == 201
    assert body["route"] == "rag_search"
    assert body["sources"]

    source = body["sources"][0]
    assert source["document_id"] == str(seeded_document)
    assert source["filename"] == "projects.pdf"
    assert source["page"] == 1
    assert source["chunk_index"] is not None
    assert source["vector_score"] is not None
    assert source["rerank_score"] is not None


async def test_tool_turn_reports_the_tool(client, inference):
    conversation = await make_conversation(client)
    inference.script(
        route_reply("tool", "calculator"),
        tool_reply("calculator", expression="17 * 23"),
        "Будет 391.",
    )

    response = await client.post(
        f"/conversations/{conversation}/agent",
        json={"content": "Сколько будет 17 * 23?"},
    )
    body = response.json()

    assert response.status_code == 201
    assert body["route"] == "tool"
    assert body["tools_used"] == [
        {"name": "calculator", "success": True, "error": None}
    ]
    assert "391" in body["message"]["content"]


async def test_tool_failure_is_visible_in_the_contract(client, inference):
    conversation = await make_conversation(client)
    inference.script(
        route_reply("tool", "calculator"),
        tool_reply("calculator", expression="1/0"),
        "Не удалось посчитать.",
    )

    response = await client.post(
        f"/conversations/{conversation}/agent",
        json={"content": "1/0?"},
    )
    body = response.json()

    assert response.status_code == 201
    assert body["tools_used"][0]["success"] is False
    assert body["tools_used"][0]["error"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


async def test_both_turns_are_persisted_as_plain_messages(
    client,
    inference,
    session_factory,
):
    conversation = await make_conversation(client)
    inference.script(route_reply("direct_answer"), "Ответ агента.")

    await client.post(
        f"/conversations/{conversation}/agent",
        json={"content": "Вопрос пользователя", "use_rag": False},
    )

    async with session_factory() as session:
        messages = (
            await session.execute(select(Message).order_by(Message.created_at))
        ).scalars().all()

    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "Вопрос пользователя"
    assert messages[1].content == "Ответ агента."


async def test_agent_history_is_visible_to_the_plain_conversation_api(
    client,
    inference,
):
    conversation = await make_conversation(client)
    inference.script(route_reply("direct_answer"), "Ответ агента.")

    await client.post(
        f"/conversations/{conversation}/agent",
        json={"content": "Вопрос", "use_rag": False},
    )

    detail = (await client.get(f"/conversations/{conversation}")).json()

    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]


async def test_agent_reuses_earlier_conversation_history(client, inference):
    conversation = await make_conversation(client)

    inference.script(route_reply("direct_answer"), "Первый ответ.")
    await client.post(
        f"/conversations/{conversation}/agent",
        json={"content": "первый вопрос", "use_rag": False},
    )

    inference.script(route_reply("direct_answer"), "Второй ответ.")
    await client.post(
        f"/conversations/{conversation}/agent",
        json={"content": "второй вопрос", "use_rag": False},
    )

    answer_call = inference.calls[-1]
    assert [m["role"] for m in answer_call[1:]] == ["user", "assistant", "user"]


async def test_history_sent_to_the_agent_is_capped(client, inference, settings):
    conversation = await make_conversation(client)

    for index in range(4):
        inference.script(route_reply("direct_answer"), f"Ответ {index}")
        await client.post(
            f"/conversations/{conversation}/agent",
            json={"content": f"вопрос {index}", "use_rag": False},
        )

    history = inference.calls[-1][1:]
    assert len(history) <= settings.chat_history_limit


# ---------------------------------------------------------------------------
# Nothing internal may leak
# ---------------------------------------------------------------------------


async def test_response_carries_no_prompts_or_reasoning(client, inference):
    conversation = await make_conversation(client)
    inference.script(
        route_reply("tool", "calculator", reason="СЕКРЕТНОЕ РАССУЖДЕНИЕ"),
        tool_reply("calculator", expression="2+2"),
        "Четыре.",
    )

    response = await client.post(
        f"/conversations/{conversation}/agent",
        json={"content": "2+2?"},
    )
    raw = response.text

    assert "СЕКРЕТНОЕ РАССУЖДЕНИЕ" not in raw
    assert ROUTER_SYSTEM_PROMPT[:40] not in raw
    assert TOOL_SELECTION_SYSTEM_PROMPT[:40] not in raw

    body = response.json()
    assert set(body) == {"message", "route", "tools_used", "sources"}
    for forbidden in ("reason", "step_count", "errors", "chat_history", "tool_calls"):
        assert forbidden not in body


async def test_internal_state_is_not_persisted_as_a_message(
    client,
    inference,
    session_factory,
):
    conversation = await make_conversation(client)
    inference.script(
        route_reply("tool", "calculator", reason="скрытая цепочка рассуждений"),
        tool_reply("calculator", expression="2+2"),
        "Четыре.",
    )

    await client.post(f"/conversations/{conversation}/agent", json={"content": "2+2?"})

    async with session_factory() as session:
        messages = (await session.execute(select(Message))).scalars().all()

    assert len(messages) == 2
    for message in messages:
        assert message.role in ("user", "assistant")
        assert "скрытая цепочка" not in message.content
        assert "calculator" not in message.content


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


async def test_unknown_conversation_is_404(client, inference):
    response = await client.post(
        f"/conversations/{uuid.uuid4()}/agent",
        json={"content": "вопрос"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found"


async def test_unknown_document_is_404(client, inference):
    conversation = await make_conversation(client)

    response = await client.post(
        f"/conversations/{conversation}/agent",
        json={"content": "вопрос", "document_ids": [str(uuid.uuid4())]},
    )

    assert response.status_code == 404


async def test_malformed_uuid_is_422(client):
    response = await client.post(
        "/conversations/not-a-uuid/agent",
        json={"content": "вопрос"},
    )

    assert response.status_code == 422


async def test_empty_content_is_422(client):
    conversation = await make_conversation(client)

    response = await client.post(
        f"/conversations/{conversation}/agent",
        json={"content": ""},
    )

    assert response.status_code == 422


async def test_inference_outage_is_502_and_keeps_the_user_turn(
    client,
    inference,
    session_factory,
):
    conversation = await make_conversation(client)
    inference.available = False

    response = await client.post(
        f"/conversations/{conversation}/agent",
        json={"content": "вопрос в офлайне"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Inference service is unavailable"}
    assert "traceback" not in response.text.lower()

    async with session_factory() as session:
        messages = (await session.execute(select(Message))).scalars().all()

    assert [message.role for message in messages] == ["user"]


async def test_agent_unavailable_is_503(client, conversation_service):
    conversation = await make_conversation(client)
    conversation_service.agent = None

    response = await client.post(
        f"/conversations/{conversation}/agent",
        json={"content": "вопрос"},
    )

    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


async def test_the_plain_message_endpoint_is_untouched(client, inference, seeded_document):
    conversation = await make_conversation(client)

    response = await client.post(
        f"/conversations/{conversation}/messages",
        json={"content": "Какие проекты?", "use_rag": True},
    )
    body = response.json()

    assert response.status_code == 201
    # Same contract as before the agent existed: no route, no tools_used.
    assert set(body) == {"message", "used_rag", "sources"}
    assert body["used_rag"] is True
    assert body["sources"]


async def test_rag_ask_is_untouched(client, inference, seeded_document):
    response = await client.post(
        "/rag/ask",
        json={"question": "Какие проекты описаны?", "top_k": 2},
    )
    body = response.json()

    assert response.status_code == 200
    assert set(body) == {"answer", "sources"}

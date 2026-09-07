"""The agent state machine.

Every LLM reply is scripted through the fake inference client, so the graph is
exercised end to end without a model, a broker or a network.
"""

import json
import uuid

import pytest

from backend.agent.graph import select_branch
from backend.agent.state import initial_state, tools_used
from backend.errors import InferenceUnavailableError
from backend.prompts.agent import FALLBACK_ANSWER, STEP_LIMIT_ANSWER


def route_reply(route: str, tool_name: str | None = None) -> str:
    payload: dict = {"route": route, "reason": "test"}
    if tool_name:
        payload["tool_name"] = tool_name
    return json.dumps(payload)


def tool_reply(tool: str, **arguments) -> str:
    return json.dumps({"tool": tool, "arguments": arguments})


async def run(agent_service, message: str, **kwargs):
    return await agent_service.run(
        conversation_id=kwargs.pop("conversation_id", uuid.uuid4()),
        user_message=message,
        chat_history=kwargs.pop("chat_history", [{"role": "user", "content": message}]),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Graph shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "route,expected",
    [
        ("direct_answer", "direct_answer"),
        ("rag_search", "rag_search"),
        ("tool", "tool_execution"),
        ("fallback", "fallback"),
        ("", "fallback"),
        ("something_invented", "fallback"),
    ],
)
def test_branch_selection_is_total(route, expected):
    assert select_branch({"route": route}) == expected


def test_graph_has_no_cycles(agent_service):
    """The state machine is a fixed DAG — no node can be revisited."""
    graph = agent_service.graph.get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert ("__start__", "classify") in edges
    for branch in ("direct_answer", "rag_search", "tool_execution", "fallback"):
        assert ("classify", branch) in edges
        assert (branch, "compose_answer") in edges
    assert ("compose_answer", "__end__") in edges

    # Nothing ever points back at the router.
    assert not [edge for edge in edges if edge[1] == "classify" and edge[0] != "__start__"]


# ---------------------------------------------------------------------------
# direct_answer
# ---------------------------------------------------------------------------


async def test_direct_route_answers_without_retrieval(
    agent_service,
    inference,
    vector_store,
):
    inference.script(route_reply("direct_answer"), "Эмбеддинги — это векторы.")

    state = await run(agent_service, "Что такое embeddings?")

    assert state["route"] == "direct_answer"
    assert state["final_answer"] == "Эмбеддинги — это векторы."
    assert state["retrieved_sources"] == []
    assert state["tool_results"] == []
    # No RAG context was built for a direct answer.
    assert vector_store.searches == []
    assert "КОНТЕКСТ" not in inference.calls[-1][-1]["content"]


async def test_direct_route_receives_the_conversation_history(
    agent_service,
    inference,
):
    inference.script(route_reply("direct_answer"), "Ответ")

    history = [
        {"role": "user", "content": "первый"},
        {"role": "assistant", "content": "ответ"},
        {"role": "user", "content": "второй"},
    ]
    await run(agent_service, "второй", chat_history=history)

    answer_call = inference.calls[-1]
    assert answer_call[0]["role"] == "system"
    assert [m["role"] for m in answer_call[1:]] == ["user", "assistant", "user"]


# ---------------------------------------------------------------------------
# rag_search
# ---------------------------------------------------------------------------


async def test_rag_route_returns_sources_and_grounded_answer(
    agent_service,
    inference,
    seeded_document,
):
    inference.script(route_reply("rag_search"), "Проект Борей — логистика.")

    state = await run(
        agent_service,
        "Какие проекты описаны?",
        use_rag=True,
    )

    assert state["route"] == "rag_search"
    assert state["retrieved_sources"]
    assert state["final_answer"] == "Проект Борей — логистика."

    grounded = inference.calls[-1][-1]["content"]
    assert "КОНТЕКСТ" in grounded
    assert "[SOURCE 1]" in grounded


async def test_rag_route_respects_document_scope(
    agent_service,
    inference,
    seeded_document,
    vector_store,
):
    inference.script(route_reply("rag_search"), "Ответ")

    await run(
        agent_service,
        "вопрос",
        use_rag=True,
        document_ids=[str(seeded_document)],
    )

    assert vector_store.searches[-1]["document_ids"] == [str(seeded_document)]


async def test_empty_retrieval_refuses_to_invent_an_answer(
    agent_service,
    inference,
):
    inference.script(route_reply("rag_search"))

    state = await run(agent_service, "вопрос без документов", use_rag=True)

    assert state["route"] == "rag_search"
    assert state["retrieved_sources"] == []
    assert "недостаточно информации" in state["final_answer"]
    # The model was never asked to compose from an empty context.
    assert len(inference.calls) == 1


# ---------------------------------------------------------------------------
# tool route
# ---------------------------------------------------------------------------


async def test_calculator_tool_route(agent_service, inference):
    inference.script(
        route_reply("tool", "calculator"),
        tool_reply("calculator", expression="125 * 8"),
        "Будет 1000.",
    )

    state = await run(agent_service, "Сколько будет 125 * 8?")

    assert state["route"] == "tool"
    assert tools_used(state) == [
        {"name": "calculator", "success": True, "error": None}
    ]
    assert state["tool_results"][0]["result"]["result"] == 1000
    assert state["final_answer"] == "Будет 1000."

    # The tool result, not the raw expression, was handed to the model.
    assert "1000" in inference.calls[-1][-1]["content"]


async def test_document_metadata_tool_route(
    agent_service,
    inference,
    seeded_document,
    session_factory,
):
    inference.script(
        route_reply("tool", "get_document_metadata"),
        tool_reply("get_document_metadata", document_id=str(seeded_document)),
        "Документ projects.pdf готов.",
    )

    async with session_factory() as session:
        state = await agent_service.run(
            conversation_id=uuid.uuid4(),
            user_message="Что за документ?",
            chat_history=[],
            session=session,
        )

    assert state["route"] == "tool"
    assert state["tool_results"][0]["success"] is True
    assert state["tool_results"][0]["result"]["filename"] == "projects.pdf"


async def test_search_documents_tool_surfaces_sources(
    agent_service,
    inference,
    seeded_document,
):
    inference.script(
        route_reply("tool", "search_documents"),
        tool_reply("search_documents", query="проекты"),
        "Нашёл проекты.",
    )

    state = await run(agent_service, "Найди проекты")

    assert state["route"] == "tool"
    assert state["retrieved_sources"], "tool-produced chunks must be citable"


async def test_tool_failure_is_reported_and_still_answered(agent_service, inference):
    inference.script(
        route_reply("tool", "calculator"),
        tool_reply("calculator", expression="1/0"),
        "Не удалось посчитать: деление на ноль.",
    )

    state = await run(agent_service, "Сколько будет 1/0?")

    assert state["route"] == "tool"
    assert tools_used(state)[0]["success"] is False
    assert "zero" in tools_used(state)[0]["error"]
    assert state["final_answer"]
    assert state["errors"]


async def test_model_cannot_call_a_tool_outside_the_allowlist(
    agent_service,
    inference,
):
    inference.script(
        route_reply("tool", "calculator"),
        tool_reply("run_shell", command="rm -rf /"),
        "Не могу выполнить это действие.",
    )

    state = await run(agent_service, "удали всё")

    assert tools_used(state) == [
        {"name": "run_shell", "success": False, "error": tools_used(state)[0]["error"]}
    ]
    assert "unknown tool" in tools_used(state)[0]["error"]


async def test_router_naming_an_unknown_tool_falls_back_to_a_safe_route(
    agent_service,
    inference,
):
    inference.script(route_reply("tool", "definitely_not_a_tool"), "Обычный ответ.")

    state = await run(agent_service, "вопрос", use_rag=False)

    # An untrustworthy routing decision degrades instead of executing.
    assert state["route"] == "direct_answer"
    assert any("unknown tool" in error for error in state["errors"])


async def test_invalid_tool_arguments_are_a_controlled_failure(
    agent_service,
    inference,
):
    inference.script(
        route_reply("tool", "calculator"),
        tool_reply("calculator", wrong_field="oops"),
        "Не хватило данных.",
    )

    state = await run(agent_service, "посчитай")

    assert tools_used(state)[0]["success"] is False
    assert "invalid arguments" in tools_used(state)[0]["error"]


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


async def test_unparseable_routing_falls_back_to_the_use_rag_default(
    agent_service,
    inference,
    seeded_document,
):
    # Two junk replies: the initial call and the single repair.
    inference.script("мусор", "тоже мусор", "Ответ по документам.")

    state = await run(agent_service, "вопрос", use_rag=True)

    assert state["route"] == "rag_search"
    assert any("routing failed" in error for error in state["errors"])


async def test_unparseable_routing_without_rag_falls_back_to_direct(
    agent_service,
    inference,
):
    inference.script("мусор", "тоже мусор", "Прямой ответ.")

    state = await run(agent_service, "вопрос", use_rag=False)

    assert state["route"] == "direct_answer"
    assert state["final_answer"] == "Прямой ответ."


async def test_tool_selection_failure_routes_to_fallback(agent_service, inference):
    inference.script(
        route_reply("tool", "calculator"),
        "не json",
        "тоже не json",
    )

    state = await run(agent_service, "посчитай что-нибудь")

    assert state["route"] == "fallback"
    assert state["final_answer"] == FALLBACK_ANSWER
    assert state["tool_results"] == []


async def test_inference_outage_propagates_as_a_domain_error(
    agent_service,
    inference,
):
    inference.available = False

    with pytest.raises(InferenceUnavailableError):
        await run(agent_service, "вопрос")


# ---------------------------------------------------------------------------
# Step limit
# ---------------------------------------------------------------------------


async def test_step_limit_stops_the_graph_with_a_controlled_answer(
    agent_service,
    inference,
    settings,
):
    settings.agent_max_steps = 1  # only classify fits
    inference.script(route_reply("direct_answer"), "не должно быть использовано")

    state = await run(agent_service, "вопрос")

    assert state["final_answer"] == STEP_LIMIT_ANSWER
    assert state["route"] == "fallback"
    assert any("step limit" in error for error in state["errors"])


async def test_a_normal_run_stays_well_inside_the_step_budget(
    agent_service,
    inference,
    settings,
):
    inference.script(route_reply("direct_answer"), "Ответ")

    state = await run(agent_service, "вопрос")

    # classify + direct_answer; compose_answer passes through.
    assert state["step_count"] == 2
    assert state["step_count"] < settings.agent_max_steps


async def test_step_count_covers_the_longest_branch(
    agent_service,
    inference,
    seeded_document,
):
    inference.script(
        route_reply("tool", "calculator"),
        tool_reply("calculator", expression="2+2"),
        "Четыре.",
    )

    state = await run(agent_service, "2+2?")

    # classify + tool_execution + compose_answer
    assert state["step_count"] == 3


# ---------------------------------------------------------------------------
# State hygiene
# ---------------------------------------------------------------------------


def test_initial_state_is_json_serialisable():
    state = initial_state(
        conversation_id=str(uuid.uuid4()),
        user_message="привет",
        chat_history=[{"role": "user", "content": "привет"}],
        use_rag=True,
        document_ids=[str(uuid.uuid4())],
    )

    assert json.loads(json.dumps(state)) == state


async def test_final_state_is_json_serialisable(agent_service, inference):
    inference.script(route_reply("direct_answer"), "Ответ")

    state = await run(agent_service, "вопрос")

    # No sessions, services or model objects leaked into the state.
    assert json.loads(json.dumps(state)) == state

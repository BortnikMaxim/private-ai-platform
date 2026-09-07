"""Structured LLM output: JSON extraction, validation, one repair, then give up."""

import pytest
from pydantic import BaseModel

from backend.agent.schemas import RouteDecision, ToolInvocation
from backend.agent.structured import StructuredCaller, extract_json_object
from backend.errors import InferenceUnavailableError


class Sample(BaseModel):
    name: str
    count: int


@pytest.fixture
def caller(inference, settings) -> StructuredCaller:
    return StructuredCaller(inference, settings)


# ---------------------------------------------------------------------------
# JSON location
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('Вот ответ:\n{"a": 1}\nНадеюсь, помог.', '{"a": 1}'),
        ('{"a": {"b": 2}}', '{"a": {"b": 2}}'),
        ('{"a": "}"}', '{"a": "}"}'),
        ('{"a": "text with \\" quote"}', '{"a": "text with \\" quote"}'),
    ],
)
def test_json_object_is_located_inside_noise(raw, expected):
    assert extract_json_object(raw) == expected


@pytest.mark.parametrize("raw", ["", "no json here", "{unbalanced", "}{"])
def test_missing_or_broken_json_is_not_located(raw):
    assert extract_json_object(raw) in (None, "{}")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_valid_json_needs_no_repair(caller, inference):
    inference.script('{"name": "ok", "count": 3}')

    result = await caller.call(Sample, "system", "user")

    assert result.ok
    assert result.value.name == "ok"
    assert result.value.count == 3
    assert result.repaired is False
    assert len(inference.calls) == 1


async def test_schema_is_included_in_the_system_prompt(caller, inference):
    inference.script('{"name": "ok", "count": 1}')

    await caller.call(Sample, "ROUTER RULES", "the question")

    system = inference.calls[0][0]["content"]
    assert "ROUTER RULES" in system
    assert '"count"' in system  # the JSON Schema is handed to the model


async def test_json_wrapped_in_prose_is_accepted(caller, inference):
    inference.script('Конечно! ```json\n{"name": "x", "count": 2}\n``` Готово.')

    result = await caller.call(Sample, "system", "user")

    assert result.ok
    assert result.value.count == 2


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------


async def test_invalid_json_triggers_exactly_one_repair(caller, inference):
    inference.script("совсем не json", '{"name": "fixed", "count": 1}')

    result = await caller.call(Sample, "system", "user")

    assert result.ok
    assert result.repaired is True
    assert result.value.name == "fixed"
    assert len(inference.calls) == 2


async def test_schema_violation_triggers_repair(caller, inference):
    # Valid JSON, wrong shape.
    inference.script('{"name": "x"}', '{"name": "x", "count": 7}')

    result = await caller.call(Sample, "system", "user")

    assert result.ok
    assert result.repaired is True
    assert result.value.count == 7


async def test_repair_prompt_shows_the_model_its_own_output_and_the_error(
    caller,
    inference,
):
    inference.script('{"name": "x"}', '{"name": "x", "count": 7}')

    await caller.call(Sample, "system", "user")

    repair_prompt = inference.calls[1][1]["content"]
    assert '{"name": "x"}' in repair_prompt
    assert "count" in repair_prompt


async def test_a_failed_repair_gives_up_instead_of_looping(caller, inference):
    inference.script("still not json", "also not json")

    result = await caller.call(Sample, "system", "user")

    assert result.ok is False
    assert result.value is None
    assert result.repaired is True
    assert result.error
    # Hard ceiling: one attempt plus one repair. Never more.
    assert len(inference.calls) == 2


async def test_repair_can_be_disabled(caller, inference, settings):
    settings.agent_structured_repair_attempts = 0
    inference.script("not json")

    result = await caller.call(Sample, "system", "user")

    assert result.ok is False
    assert len(inference.calls) == 1


async def test_inference_outage_propagates_rather_than_repairing(caller, inference):
    inference.available = False

    with pytest.raises(InferenceUnavailableError):
        await caller.call(Sample, "system", "user")

    # Repairing an unreachable model would be pointless.
    assert len(inference.calls) == 1


# ---------------------------------------------------------------------------
# The real agent schemas
# ---------------------------------------------------------------------------


async def test_route_decision_parses(caller, inference):
    inference.script('{"route": "rag_search", "reason": "про документы"}')

    result = await caller.call(RouteDecision, "s", "u")

    assert result.value.route == "rag_search"
    assert result.value.tool_name is None


async def test_route_decision_rejects_an_unknown_route(caller, inference):
    inference.script('{"route": "delete_everything"}', '{"route": "direct_answer"}')

    result = await caller.call(RouteDecision, "s", "u")

    assert result.value.route == "direct_answer"
    assert result.repaired is True


async def test_tool_invocation_parses(caller, inference):
    inference.script('{"tool": "calculator", "arguments": {"expression": "2+2"}}')

    result = await caller.call(ToolInvocation, "s", "u")

    assert result.value.tool == "calculator"
    assert result.value.arguments == {"expression": "2+2"}


async def test_extra_fields_are_ignored_not_fatal(caller, inference):
    inference.script('{"route": "direct_answer", "confidence": 0.9, "junk": [1,2]}')

    result = await caller.call(RouteDecision, "s", "u")

    assert result.ok
    assert result.repaired is False

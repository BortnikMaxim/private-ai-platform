"""Tool registry and the document / datetime tools."""

import uuid
from datetime import UTC, datetime

import pytest

from backend.agent.tools.base import (
    Tool,
    ToolContext,
    ToolError,
    UnknownToolError,
    parse_uuid,
)
from backend.agent.tools.registry import ToolRegistry, default_registry


@pytest.fixture
def context(session_factory, rag_service, document_service, settings):
    return ToolContext(
        rag_service=rag_service,
        document_service=document_service,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_default_registry_is_an_explicit_allowlist(tool_registry):
    assert tool_registry.names() == [
        "calculator",
        "get_current_datetime",
        "get_document_metadata",
        "search_documents",
    ]


def test_known_tool_is_resolvable(tool_registry):
    assert tool_registry.get("calculator").name == "calculator"
    assert "calculator" in tool_registry


def test_unknown_tool_raises(tool_registry):
    assert "shell" not in tool_registry

    with pytest.raises(UnknownToolError, match="unknown tool"):
        tool_registry.get("shell")


def test_duplicate_registration_is_rejected():
    from backend.agent.tools.calculator import CalculatorTool

    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry([CalculatorTool(), CalculatorTool()])


def test_catalogue_lists_every_tool_with_arguments(tool_registry):
    described = tool_registry.describe()

    for name in tool_registry.names():
        assert name in described
    assert "expression" in described


async def test_executing_an_unknown_tool_is_a_controlled_failure(tool_registry, context):
    result = await tool_registry.execute("rm_rf", {}, context)

    assert result == {
        "name": "rm_rf",
        "success": False,
        "error": result["error"],
    }
    assert "unknown tool" in result["error"]


async def test_invalid_arguments_are_a_controlled_failure(tool_registry, context):
    result = await tool_registry.execute("calculator", {"wrong": "field"}, context)

    assert result["success"] is False
    assert "invalid arguments" in result["error"]


async def test_a_tool_error_is_reported_not_raised(tool_registry, context):
    result = await tool_registry.execute(
        "calculator", {"expression": "1/0"}, context
    )

    assert result["success"] is False
    assert "zero" in result["error"]


async def test_a_crashing_tool_never_leaks_a_traceback(tool_registry, context):
    class Exploding(Tool):
        name = "exploding"
        description = "always fails"
        input_schema = type(
            "Empty", (__import__("pydantic").BaseModel,), {"__annotations__": {}}
        )

        async def execute(self, arguments, ctx):
            raise RuntimeError("secret internal detail")

    registry = ToolRegistry([Exploding()])
    result = await registry.execute("exploding", {}, context)

    assert result["success"] is False
    assert "secret internal detail" not in result["error"]
    assert result["error"] == "tool execution failed"


async def test_successful_execution_shape(tool_registry, context):
    result = await tool_registry.execute(
        "calculator", {"expression": "125 * 8"}, context
    )

    assert result == {
        "name": "calculator",
        "success": True,
        "result": {"expression": "125 * 8", "result": 1000},
    }


# ---------------------------------------------------------------------------
# get_current_datetime
# ---------------------------------------------------------------------------


async def test_current_datetime_returns_utc(tool_registry, context):
    result = await tool_registry.execute("get_current_datetime", {}, context)

    assert result["success"] is True
    payload = result["result"]
    assert payload["timezone"] == "UTC"

    parsed = datetime.fromisoformat(payload["utc_iso"])
    assert parsed.tzinfo is not None
    assert abs((datetime.now(UTC) - parsed).total_seconds()) < 60


# ---------------------------------------------------------------------------
# search_documents
# ---------------------------------------------------------------------------


async def test_search_documents_uses_the_existing_rag_pipeline(
    tool_registry,
    context,
    seeded_document,
    vector_store,
):
    result = await tool_registry.execute(
        "search_documents",
        {"query": "складскую логистику"},
        context,
    )

    assert result["success"] is True
    assert result["result"]["matches"] > 0
    # The retrieval went through RagService -> the shared vector store.
    assert vector_store.searches
    # Citations are handed back for the response contract.
    assert context.sources


async def test_search_documents_honours_an_explicit_scope(
    tool_registry,
    context,
    seeded_document,
    vector_store,
):
    other = str(uuid.uuid4())

    await tool_registry.execute(
        "search_documents",
        {"query": "проект", "document_ids": [other]},
        context,
    )

    assert vector_store.searches[-1]["document_ids"] == [other]


async def test_search_documents_inherits_the_request_scope(
    tool_registry,
    context,
    seeded_document,
    vector_store,
):
    context.document_ids = [str(seeded_document)]

    await tool_registry.execute("search_documents", {"query": "проект"}, context)

    assert vector_store.searches[-1]["document_ids"] == [str(seeded_document)]


async def test_search_documents_rejects_a_malformed_document_id(
    tool_registry,
    context,
):
    result = await tool_registry.execute(
        "search_documents",
        {"query": "проект", "document_ids": ["not-a-uuid"]},
        context,
    )

    assert result["success"] is False
    assert "UUID" in result["error"]


# ---------------------------------------------------------------------------
# get_document_metadata
# ---------------------------------------------------------------------------


async def test_document_metadata_reads_from_postgres(
    tool_registry,
    context,
    session_factory,
    seeded_document,
):
    async with session_factory() as session:
        context.session = session

        result = await tool_registry.execute(
            "get_document_metadata",
            {"document_id": str(seeded_document)},
            context,
        )

    assert result["success"] is True
    payload = result["result"]
    assert payload["filename"] == "projects.pdf"
    assert payload["status"] == "ready"
    assert payload["chunks_count"] == 3


async def test_document_metadata_for_an_unknown_id_is_a_controlled_failure(
    tool_registry,
    context,
    session_factory,
):
    async with session_factory() as session:
        context.session = session

        result = await tool_registry.execute(
            "get_document_metadata",
            {"document_id": str(uuid.uuid4())},
            context,
        )

    assert result["success"] is False
    assert "not found" in result["error"]


async def test_document_metadata_rejects_a_malformed_id(tool_registry, context):
    result = await tool_registry.execute(
        "get_document_metadata",
        {"document_id": "../../etc/passwd"},
        context,
    )

    assert result["success"] is False


def test_parse_uuid_rejects_junk():
    with pytest.raises(ToolError):
        parse_uuid("not-a-uuid")


def test_default_registry_is_a_fresh_instance_each_time():
    assert default_registry() is not default_registry()

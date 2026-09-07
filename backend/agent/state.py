"""Typed state carried through the agent graph.

Every field is a plain JSON type so the whole state can be serialised, logged
selectively and asserted on in tests without any framework machinery.
"""

from typing import Any, Literal, TypedDict

Route = Literal["direct_answer", "rag_search", "tool", "fallback"]

ROUTES: tuple[str, ...] = ("direct_answer", "rag_search", "tool", "fallback")


class ToolCall(TypedDict):
    name: str
    arguments: dict[str, Any]


class ToolResult(TypedDict, total=False):
    name: str
    success: bool
    result: Any
    error: str


class AgentState(TypedDict, total=False):
    # -- input ------------------------------------------------------------
    conversation_id: str
    user_message: str
    chat_history: list[dict[str, str]]
    use_rag: bool
    document_ids: list[str]

    # -- routing ----------------------------------------------------------
    route: str

    # -- evidence ---------------------------------------------------------
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    retrieved_sources: list[dict[str, Any]]

    # -- output -----------------------------------------------------------
    final_answer: str

    # -- bookkeeping ------------------------------------------------------
    step_count: int
    errors: list[str]


def initial_state(
    conversation_id: str,
    user_message: str,
    chat_history: list[dict[str, str]] | None = None,
    use_rag: bool = False,
    document_ids: list[str] | None = None,
) -> AgentState:
    return AgentState(
        conversation_id=conversation_id,
        user_message=user_message,
        chat_history=list(chat_history or []),
        use_rag=use_rag,
        document_ids=list(document_ids or []),
        route="",
        tool_calls=[],
        tool_results=[],
        retrieved_sources=[],
        final_answer="",
        step_count=0,
        errors=[],
    )


def tools_used(state: AgentState) -> list[dict[str, Any]]:
    """Public, non-revealing summary of the tools this run touched."""
    return [
        {
            "name": result.get("name", "unknown"),
            "success": bool(result.get("success")),
            "error": result.get("error"),
        }
        for result in state.get("tool_results", [])
    ]

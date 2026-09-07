"""Explicit tool allowlist.

Tools are registered by passing instances in — there is no dynamic import, no
entry-point scanning and no lookup by attribute name. A tool the model asks for
that is not in this mapping is rejected outright.
"""

import logging
from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from backend.agent.tools.base import (
    InvalidToolArgumentsError,
    Tool,
    ToolContext,
    ToolError,
    UnknownToolError,
)
from backend.agent.tools.calculator import CalculatorTool
from backend.agent.tools.datetime_tool import CurrentDatetimeTool
from backend.agent.tools.documents import DocumentMetadataTool, SearchDocumentsTool
from backend.observability import AGENT_TOOL_CALLS_TOTAL

logger = logging.getLogger(__name__)

MAX_ERROR_LENGTH = 300


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {}

        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)

        if tool is None:
            raise UnknownToolError(
                f"unknown tool '{name}'; available: {', '.join(self.names())}"
            )

        return tool

    def describe(self) -> str:
        """Tool catalogue for the selection prompt."""
        return "\n".join(self._tools[name].describe() for name in self.names())

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> dict[str, Any]:
        """Validate and run a tool. Always returns a result dict, never raises.

        Failures come back as ``{"success": False, "error": ...}`` with a short
        message that is safe to hand to the model.
        """
        try:
            tool = self.get(name)
        except UnknownToolError as exc:
            return self._failure(name, str(exc), "unknown_tool")

        try:
            parsed = tool.input_schema.model_validate(arguments or {})
        except ValidationError as exc:
            first = exc.errors()[0]
            location = ".".join(str(part) for part in first.get("loc", ())) or "<root>"
            message = f"invalid arguments — {location}: {first.get('msg', 'invalid')}"
            return self._failure(name, message, "invalid_arguments")

        try:
            result = await tool.execute(parsed, context)
        except ToolError as exc:
            return self._failure(name, str(exc), "tool_error")
        except InvalidToolArgumentsError as exc:
            return self._failure(name, str(exc), "invalid_arguments")
        except Exception:
            # Unexpected: the traceback goes to the log, never to the model.
            logger.exception("agent_tool_crashed tool=%s", name)
            return self._failure(name, "tool execution failed", "crash")

        AGENT_TOOL_CALLS_TOTAL.labels(tool=name, status="success").inc()

        return {"name": name, "success": True, "result": result}

    @staticmethod
    def _failure(name: str, message: str, status: str) -> dict[str, Any]:
        AGENT_TOOL_CALLS_TOTAL.labels(tool=name, status=status).inc()
        logger.info("agent_tool_failed tool=%s status=%s", name, status)

        return {"name": name, "success": False, "error": message[:MAX_ERROR_LENGTH]}


def default_registry() -> ToolRegistry:
    """The allowlist the application ships with."""
    return ToolRegistry(
        [
            SearchDocumentsTool(),
            DocumentMetadataTool(),
            CalculatorTool(),
            CurrentDatetimeTool(),
        ]
    )

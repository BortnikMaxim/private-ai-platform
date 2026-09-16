"""Explicit tool allowlist.

Tools are registered by passing instances in — there is no dynamic import, no
entry-point scanning and no lookup by attribute name. A tool the model asks for
that is not in this mapping is rejected outright.
"""

import logging
import time
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
from backend.observability import AGENT_TOOL_CALLS_TOTAL, AGENT_TOOL_DURATION_SECONDS
from backend.tracing import NULL_TRACER, Tracer

logger = logging.getLogger(__name__)

MAX_ERROR_LENGTH = 300


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool], tracer: Tracer | None = None) -> None:
        self.tracer = tracer or NULL_TRACER
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
        started = time.perf_counter()

        # Tool arguments can carry a user's question, so they are content and
        # only reach the span when capture is enabled.
        with self.tracer.span("agent.tool", as_type="tool", tool=name) as span:
            span.set_content(input=arguments)

            try:
                tool = self.get(name)
            except UnknownToolError as exc:
                span.update(status="unknown_tool")
                return self._failure(name, str(exc), "unknown_tool", started)

            try:
                parsed = tool.input_schema.model_validate(arguments or {})
            except ValidationError as exc:
                first = exc.errors()[0]
                location = (
                    ".".join(str(part) for part in first.get("loc", ())) or "<root>"
                )
                message = (
                    f"invalid arguments — {location}: {first.get('msg', 'invalid')}"
                )
                span.update(status="invalid_arguments")
                return self._failure(name, message, "invalid_arguments", started)

            try:
                result = await tool.execute(parsed, context)
            except ToolError as exc:
                span.update(status="tool_error")
                return self._failure(name, str(exc), "tool_error", started)
            except InvalidToolArgumentsError as exc:
                span.update(status="invalid_arguments")
                return self._failure(name, str(exc), "invalid_arguments", started)
            except Exception:
                # Unexpected: the traceback goes to the log, never to the model.
                logger.exception("agent_tool_crashed tool=%s", name)
                span.update(status="crash")
                return self._failure(name, "tool execution failed", "crash", started)

            duration = time.perf_counter() - started
            AGENT_TOOL_CALLS_TOTAL.labels(tool=name, status="success").inc()
            AGENT_TOOL_DURATION_SECONDS.labels(tool=name).observe(duration)

            span.update(status="success", duration_ms=round(duration * 1000, 1))
            span.set_content(output=result)

            return {"name": name, "success": True, "result": result}

    @staticmethod
    def _failure(
        name: str,
        message: str,
        status: str,
        started: float | None = None,
    ) -> dict[str, Any]:
        AGENT_TOOL_CALLS_TOTAL.labels(tool=name, status=status).inc()

        if started is not None:
            AGENT_TOOL_DURATION_SECONDS.labels(tool=name).observe(
                time.perf_counter() - started
            )

        logger.info("agent_tool_failed tool=%s status=%s", name, status)

        return {"name": name, "success": False, "error": message[:MAX_ERROR_LENGTH]}


def default_registry(tracer: Tracer | None = None) -> ToolRegistry:
    """The allowlist the application ships with."""
    return ToolRegistry(
        [
            SearchDocumentsTool(),
            DocumentMetadataTool(),
            CalculatorTool(),
            CurrentDatetimeTool(),
        ],
        tracer=tracer,
    )

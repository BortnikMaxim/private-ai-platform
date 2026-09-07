"""Graph node implementations.

Each node takes the current :class:`~backend.agent.state.AgentState` and returns
only the keys it changes. Nodes never raise for expected failures — they record
an error and let the graph converge on ``compose_answer``, which is what keeps
the response contract stable.
"""

import json
import logging
import time
from typing import Any

from langchain_core.runnables import RunnableConfig

from backend.agent.schemas import RouteDecision, ToolInvocation
from backend.agent.state import AgentState
from backend.agent.structured import StructuredCaller
from backend.agent.tools.base import ToolContext
from backend.agent.tools.registry import ToolRegistry
from backend.config import Settings
from backend.observability import agent_event
from backend.prompts import CHAT_SYSTEM_PROMPT, GROUNDED_SYSTEM_PROMPT
from backend.prompts.agent import (
    FALLBACK_ANSWER,
    ROUTER_SYSTEM_PROMPT,
    ROUTER_USER_TEMPLATE,
    STEP_LIMIT_ANSWER,
    TOOL_ANSWER_SYSTEM_PROMPT,
    TOOL_ANSWER_USER_TEMPLATE,
    TOOL_SELECTION_SYSTEM_PROMPT,
    TOOL_SELECTION_USER_TEMPLATE,
)
from backend.services.inference_client import InferenceClient
from backend.services.rag_service import RagService

logger = logging.getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 4000
NO_CONTEXT_ANSWER = (
    "В загруженных документах недостаточно информации, чтобы ответить на этот "
    "вопрос."
)


class StepLimitExceeded(Exception):
    """Raised internally when a node would exceed AGENT_MAX_STEPS."""


class AgentNodes:
    """The node bodies, bound to the services they need.

    Holding these on an instance (rather than module functions reaching for
    globals) is what lets the tests build a graph over fakes.
    """

    def __init__(
        self,
        inference: InferenceClient,
        rag: RagService,
        registry: ToolRegistry,
        structured: StructuredCaller,
        settings: Settings,
    ) -> None:
        self.inference = inference
        self.rag = rag
        self.registry = registry
        self.structured = structured
        self.settings = settings

    # -- helpers ----------------------------------------------------------

    def _step(self, state: AgentState, node: str) -> int:
        """Consume one step, or signal that the budget is gone."""
        used = state.get("step_count", 0) + 1

        if used > self.settings.agent_max_steps:
            agent_event(
                "agent_step_limit_exceeded",
                state.get("conversation_id", ""),
                node=node,
                step=used,
                limit=self.settings.agent_max_steps,
            )
            raise StepLimitExceeded(node)

        return used

    @staticmethod
    def _limit_exceeded(state: AgentState, node: str) -> dict[str, Any]:
        return {
            "step_count": state.get("step_count", 0) + 1,
            "route": "fallback",
            "final_answer": STEP_LIMIT_ANSWER,
            "errors": [*state.get("errors", []), f"step limit exceeded at {node}"],
        }

    def _history_messages(
        self,
        state: AgentState,
        system_prompt: str,
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]

        for entry in state.get("chat_history", []):
            role = entry.get("role")
            content = (entry.get("content") or "").strip()

            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        return messages

    # -- nodes ------------------------------------------------------------

    async def classify(self, state: AgentState) -> dict[str, Any]:
        """Pick a route via structured output, with a safe default."""
        node = "classify"
        conversation_id = state.get("conversation_id", "")

        try:
            step = self._step(state, node)
        except StepLimitExceeded:
            return self._limit_exceeded(state, node)

        started = time.perf_counter()
        use_rag = bool(state.get("use_rag"))
        default_route = "rag_search" if use_rag else "direct_answer"

        user_prompt = ROUTER_USER_TEMPLATE.format(
            tools=self.registry.describe(),
            use_rag="да" if use_rag else "нет",
            scoped="да" if state.get("document_ids") else "нет",
            message=state.get("user_message", ""),
        )

        errors = list(state.get("errors", []))

        # An unreachable inference service is deliberately NOT caught here. No
        # branch can produce a real answer without the model, and an empty
        # retrieval would otherwise be reported as "nothing in the documents",
        # which is a lie about why the request failed. It propagates to a 502.
        result = await self.structured.call(
            RouteDecision,
            system_prompt=ROUTER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        if not result.ok:
            errors.append(f"routing failed: {result.error}")
            route = default_route
            tool_name = None
        else:
            decision: RouteDecision = result.value  # type: ignore[assignment]
            route = decision.route
            tool_name = decision.tool_name

            # A tool route naming an unknown tool is not trustworthy.
            if route == "tool" and tool_name and tool_name not in self.registry:
                errors.append(f"router requested unknown tool '{tool_name}'")
                route = default_route
                tool_name = None

        update: dict[str, Any] = {
            "step_count": step,
            "route": route,
            "errors": errors,
        }

        # Carry the suggested tool forward without exposing the reasoning.
        if route == "tool" and tool_name:
            update["tool_calls"] = [{"name": tool_name, "arguments": {}}]

        agent_event(
            "agent_routed",
            conversation_id,
            route=route,
            tool_name=tool_name,
            step=step,
            repaired=result.repaired if result.ok else None,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )

        return update

    async def direct_answer(self, state: AgentState) -> dict[str, Any]:
        """Answer from the model alone — no retrieval, no tools."""
        node = "direct_answer"

        try:
            step = self._step(state, node)
        except StepLimitExceeded:
            return self._limit_exceeded(state, node)

        started = time.perf_counter()
        messages = self._history_messages(state, CHAT_SYSTEM_PROMPT)

        answer = await self.inference.chat(
            messages,
            max_tokens=self.settings.agent_answer_max_tokens,
        )

        agent_event(
            "agent_direct_completed",
            state.get("conversation_id", ""),
            step=step,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )

        return {"step_count": step, "final_answer": answer}

    async def rag_search(self, state: AgentState) -> dict[str, Any]:
        """Retrieve grounding chunks through the existing RagService."""
        node = "rag_search"

        try:
            step = self._step(state, node)
        except StepLimitExceeded:
            return self._limit_exceeded(state, node)

        started = time.perf_counter()
        document_ids = state.get("document_ids") or None

        sources = await self.rag.retrieve(
            question=state.get("user_message", ""),
            document_ids=document_ids,
        )

        agent_event(
            "agent_rag_completed",
            state.get("conversation_id", ""),
            step=step,
            sources=len(sources),
            scoped=bool(document_ids),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )

        return {"step_count": step, "retrieved_sources": sources}

    async def tool_execution(
        self,
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Choose a tool via structured output, validate it and run it.

        Per-request handles (the database session, the document service) arrive
        through LangGraph's ``config`` rather than the state, which keeps the
        state a plain serialisable dict.
        """
        node = "tool_execution"
        conversation_id = state.get("conversation_id", "")

        try:
            step = self._step(state, node)
        except StepLimitExceeded:
            return self._limit_exceeded(state, node)

        started = time.perf_counter()
        errors = list(state.get("errors", []))
        suggested = (state.get("tool_calls") or [{}])[0].get("name")

        user_prompt = TOOL_SELECTION_USER_TEMPLATE.format(
            tools=self.registry.describe(),
            message=state.get("user_message", ""),
        )

        if suggested:
            user_prompt = (
                f"{user_prompt}\n\nМаршрутизатор предложил инструмент: {suggested}"
            )

        # As in classify, an outage propagates rather than degrading silently.
        result = await self.structured.call(
            ToolInvocation,
            system_prompt=TOOL_SELECTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        if not result.ok:
            errors.append(f"tool selection failed: {result.error}")
            agent_event(
                "agent_tool_selection_failed",
                conversation_id,
                step=step,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            return {"step_count": step, "route": "fallback", "errors": errors}

        invocation: ToolInvocation = result.value  # type: ignore[assignment]

        agent_event(
            "agent_tool_called",
            conversation_id,
            tool_name=invocation.tool,
            step=step,
        )

        context = self._tool_context(state, config)
        tool_result = await self.registry.execute(
            invocation.tool,
            invocation.arguments,
            context,
        )

        if not tool_result.get("success"):
            errors.append(f"tool '{invocation.tool}' failed")

        agent_event(
            "agent_tool_completed",
            conversation_id,
            tool_name=invocation.tool,
            success=tool_result.get("success"),
            step=step,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )

        update: dict[str, Any] = {
            "step_count": step,
            "tool_calls": [
                {"name": invocation.tool, "arguments": invocation.arguments}
            ],
            "tool_results": [tool_result],
            "errors": errors,
        }

        # search_documents produces citable chunks; surface them like RAG does.
        if context.sources:
            update["retrieved_sources"] = context.sources

        return update

    def _tool_context(
        self,
        state: AgentState,
        config: RunnableConfig | None,
    ) -> ToolContext:
        runtime = (config or {}).get("configurable", {}) or {}

        return ToolContext(
            session=runtime.get("session"),
            rag_service=self.rag,
            document_service=runtime.get("document_service"),
            settings=self.settings,
            document_ids=state.get("document_ids") or None,
            sources=[],
        )

    async def fallback(self, state: AgentState) -> dict[str, Any]:
        """Safe terminal branch when routing could not be trusted."""
        node = "fallback"

        try:
            step = self._step(state, node)
        except StepLimitExceeded:
            return self._limit_exceeded(state, node)

        agent_event(
            "agent_fallback",
            state.get("conversation_id", ""),
            step=step,
            errors=len(state.get("errors", [])),
        )

        return {"step_count": step, "final_answer": FALLBACK_ANSWER}

    async def compose_answer(self, state: AgentState) -> dict[str, Any]:
        """Turn whatever the branch produced into the user facing answer."""
        node = "compose_answer"

        if state.get("final_answer"):
            # direct_answer and fallback already produced the text.
            return {}

        try:
            step = self._step(state, node)
        except StepLimitExceeded:
            return self._limit_exceeded(state, node)

        started = time.perf_counter()
        route = state.get("route")

        if route == "tool":
            answer = await self._compose_from_tool(state)
        elif route == "fallback":
            # A branch degraded mid-flight (e.g. tool selection failed). Saying
            # "not enough information in the documents" here would be wrong —
            # the documents were never the problem.
            answer = FALLBACK_ANSWER
        else:
            answer = await self._compose_from_sources(state)

        agent_event(
            "agent_composed",
            state.get("conversation_id", ""),
            route=route,
            step=step,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )

        return {"step_count": step, "final_answer": answer}

    async def _compose_from_sources(self, state: AgentState) -> str:
        sources = state.get("retrieved_sources") or []

        if not sources:
            # Never let the model invent an answer out of an empty context.
            return NO_CONTEXT_ANSWER

        context = self.rag.build_context(sources)
        messages = self._history_messages(state, GROUNDED_SYSTEM_PROMPT)

        if messages and messages[-1]["role"] == "user":
            messages[-1] = {
                "role": "user",
                "content": (
                    f"КОНТЕКСТ:\n\n{context}\n\n"
                    f"ВОПРОС:\n{state.get('user_message', '')}"
                ),
            }
        else:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"КОНТЕКСТ:\n\n{context}\n\n"
                        f"ВОПРОС:\n{state.get('user_message', '')}"
                    ),
                }
            )

        return await self.inference.chat(
            messages,
            max_tokens=self.settings.agent_answer_max_tokens,
        )

    async def _compose_from_tool(self, state: AgentState) -> str:
        results = state.get("tool_results") or []

        if not results:
            return FALLBACK_ANSWER

        result = results[-1]
        payload = result.get("result") if result.get("success") else result.get("error")
        rendered = _render(payload)

        messages = [
            {"role": "system", "content": TOOL_ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": TOOL_ANSWER_USER_TEMPLATE.format(
                    message=state.get("user_message", ""),
                    tool_name=result.get("name", "unknown"),
                    result=rendered,
                ),
            },
        ]

        return await self.inference.chat(
            messages,
            max_tokens=self.settings.agent_answer_max_tokens,
        )


def _render(payload: Any) -> str:
    if isinstance(payload, str):
        return payload[:MAX_TOOL_RESULT_CHARS]

    try:
        return json.dumps(payload, ensure_ascii=False)[:MAX_TOOL_RESULT_CHARS]
    except (TypeError, ValueError):
        return str(payload)[:MAX_TOOL_RESULT_CHARS]

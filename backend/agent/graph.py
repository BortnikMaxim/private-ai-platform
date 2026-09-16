"""The agent state machine and the service that runs it.

The graph is a fixed, finite DAG — there is no autonomous loop and no edge that
returns to an earlier node:

    START -> classify -> { direct_answer | rag_search | tool_execution
                           | fallback } -> compose_answer -> END

A normal run therefore executes exactly three nodes. ``AGENT_MAX_STEPS`` is a
belt-and-braces ceiling that every node checks, so extending the graph later can
never turn it into a runaway.
"""

import logging
import time
import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agent.nodes import AgentNodes
from backend.agent.state import AgentState, initial_state
from backend.agent.structured import StructuredCaller
from backend.agent.tools.registry import ToolRegistry
from backend.config import Settings
from backend.errors import InferenceUnavailableError
from backend.observability import (
    AGENT_DURATION_SECONDS,
    AGENT_REQUESTS_TOTAL,
    agent_event,
)
from backend.prompts.agent import FALLBACK_ANSWER
from backend.services.document_service import DocumentService
from backend.services.inference_client import InferenceClient
from backend.services.rag_service import RagService
from backend.tracing import NULL_TRACER, Tracer, current_request_id

logger = logging.getLogger(__name__)

BRANCHES = {
    "direct_answer": "direct_answer",
    "rag_search": "rag_search",
    "tool": "tool_execution",
    "fallback": "fallback",
}


def select_branch(state: AgentState) -> str:
    """Conditional edge out of ``classify``. Unknown routes go to fallback."""
    return BRANCHES.get(state.get("route", ""), "fallback")


def build_graph(nodes: AgentNodes):
    graph = StateGraph(AgentState)

    graph.add_node("classify", nodes.classify)
    graph.add_node("direct_answer", nodes.direct_answer)
    graph.add_node("rag_search", nodes.rag_search)
    graph.add_node("tool_execution", nodes.tool_execution)
    graph.add_node("fallback", nodes.fallback)
    graph.add_node("compose_answer", nodes.compose_answer)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        select_branch,
        {
            "direct_answer": "direct_answer",
            "rag_search": "rag_search",
            "tool_execution": "tool_execution",
            "fallback": "fallback",
        },
    )

    for branch in ("direct_answer", "rag_search", "tool_execution", "fallback"):
        graph.add_edge(branch, "compose_answer")

    graph.add_edge("compose_answer", END)

    return graph.compile()


class AgentService:
    """Runs the graph for one conversation turn."""

    def __init__(
        self,
        inference: InferenceClient,
        rag: RagService,
        documents: DocumentService,
        registry: ToolRegistry,
        settings: Settings,
        tracer: Tracer | None = None,
    ) -> None:
        self.inference = inference
        self.rag = rag
        self.documents = documents
        self.registry = registry
        self.settings = settings
        self.tracer = tracer or NULL_TRACER

        self.nodes = AgentNodes(
            inference=inference,
            rag=rag,
            registry=registry,
            structured=StructuredCaller(inference, settings),
            settings=settings,
            tracer=self.tracer,
        )
        # Compiled once; per-request data travels in the state and the config.
        self.graph = build_graph(self.nodes)

    async def run(
        self,
        conversation_id: uuid.UUID | str,
        user_id: uuid.UUID | str,
        user_message: str,
        chat_history: list[dict[str, str]] | None = None,
        use_rag: bool = False,
        document_ids: list[str] | None = None,
        session: AsyncSession | None = None,
    ) -> AgentState:
        key = str(conversation_id)
        state = initial_state(
            conversation_id=key,
            user_id=str(user_id),
            user_message=user_message,
            chat_history=chat_history,
            use_rag=use_rag,
            document_ids=document_ids,
        )

        agent_event(
            "agent_started",
            key,
            use_rag=use_rag,
            scoped=bool(document_ids),
            history=len(state["chat_history"]),
        )

        started = time.perf_counter()
        config: dict[str, Any] = {
            "configurable": {
                "session": session,
                "document_service": self.documents,
            }
        }

        # Root span for the turn. The user message is content and is attached
        # only when capture is switched on.
        with self.tracer.span(
            "agent.request",
            as_type="agent",
            request_id=current_request_id(),
            use_rag=use_rag,
            scoped=bool(document_ids),
            history=len(state["chat_history"]),
        ) as turn_span:
            turn_span.set_content(input=user_message)

            try:
                final = await self._invoke(state, config)
            except Exception:
                raise
            else:
                turn_span.update(
                    route=final.get("route"),
                    steps=final.get("step_count"),
                    tools=len(final.get("tool_results", [])),
                    sources=len(final.get("retrieved_sources", [])),
                    errors=len(final.get("errors", [])),
                )
                turn_span.set_content(output=final.get("final_answer"))

        duration = time.perf_counter() - started
        AGENT_DURATION_SECONDS.observe(duration)

        route = final.get("route") or "fallback"
        answer = (final.get("final_answer") or "").strip()

        if not answer:
            # A branch that produced nothing must still return something safe.
            answer = FALLBACK_ANSWER
            final["final_answer"] = answer
            final.setdefault("errors", []).append("empty answer")

        status = "error" if final.get("errors") else "success"
        AGENT_REQUESTS_TOTAL.labels(route=route, status=status).inc()

        agent_event(
            "agent_completed",
            key,
            route=route,
            step=final.get("step_count", 0),
            tools=len(final.get("tool_results", [])),
            sources=len(final.get("retrieved_sources", [])),
            errors=len(final.get("errors", [])),
            duration_ms=round(duration * 1000, 1),
        )

        return final

    async def _invoke(self, state: AgentState, config: dict[str, Any]) -> AgentState:
        started = time.perf_counter()
        key = state.get("conversation_id", "")

        try:
            return await self.graph.ainvoke(state, config=config)

        except InferenceUnavailableError:
            # Propagates to the router as a 502; the endpoint has already
            # persisted the user turn.
            AGENT_REQUESTS_TOTAL.labels(route="unknown", status="inference_error").inc()
            AGENT_DURATION_SECONDS.observe(time.perf_counter() - started)
            agent_event("agent_failed", key, reason="inference_unavailable")
            raise

        except Exception:
            AGENT_REQUESTS_TOTAL.labels(route="unknown", status="error").inc()
            AGENT_DURATION_SECONDS.observe(time.perf_counter() - started)
            # Traceback to the log, never to the client.
            logger.exception("agent_crashed conversation_id=%s", key)
            agent_event("agent_failed", key, reason="internal_error")
            raise

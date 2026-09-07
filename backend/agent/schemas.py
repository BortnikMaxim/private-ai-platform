"""Pydantic models the LLM must fill in.

These double as the JSON Schema handed to the model in the prompt and as the
validator for whatever comes back.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RouteDecision(BaseModel):
    """Router output."""

    model_config = ConfigDict(extra="ignore")

    route: Literal["direct_answer", "rag_search", "tool"]
    tool_name: str | None = Field(
        default=None,
        description="Name of the tool to use; only when route is 'tool'",
        max_length=64,
    )
    reason: str = Field(
        default="",
        description="Very short justification, at most one sentence",
        max_length=300,
    )


class ToolInvocation(BaseModel):
    """Tool selection output."""

    model_config = ConfigDict(extra="ignore")

    tool: str = Field(max_length=64)
    arguments: dict[str, Any] = Field(default_factory=dict)

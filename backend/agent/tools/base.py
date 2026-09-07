"""Uniform tool interface.

Every tool declares a name, a description and a Pydantic input schema. The
registry validates arguments against that schema before anything executes, so a
tool body never sees unvalidated model output.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession


class ToolError(Exception):
    """A controlled tool failure whose message is safe to show the model."""


class UnknownToolError(ToolError):
    """The model asked for a tool that is not in the allowlist."""


class InvalidToolArgumentsError(ToolError):
    """Arguments did not match the tool's schema."""


@dataclass(slots=True)
class ToolContext:
    """Per-request handles a tool may use. Never a global."""

    session: AsyncSession | None = None
    rag_service: Any = None
    document_service: Any = None
    settings: Any = None
    document_ids: list[str] | None = None
    # Populated by tools that produce citable chunks.
    sources: list[dict[str, Any]] | None = None

    def collect_sources(self, chunks: list[dict[str, Any]]) -> None:
        if self.sources is None:
            self.sources = []
        self.sources.extend(chunks)


class Tool(ABC):
    name: str
    description: str
    input_schema: type[BaseModel]

    @abstractmethod
    async def execute(self, arguments: BaseModel, context: ToolContext) -> dict[str, Any]:
        """Run the tool. Raise ToolError for controlled, reportable failures."""

    def describe(self) -> str:
        """One-line summary plus argument names, for the selection prompt."""
        fields = ", ".join(
            f"{name}: {_type_name(info.annotation)}"
            for name, info in self.input_schema.model_fields.items()
        )
        return f"- {self.name}({fields}) — {self.description}"


def _type_name(annotation: Any) -> str:
    return getattr(annotation, "__name__", None) or str(annotation)


def parse_uuid(value: str, field: str = "document_id") -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise InvalidToolArgumentsError(f"{field} is not a valid UUID") from exc

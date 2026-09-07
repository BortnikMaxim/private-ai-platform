"""Current UTC time. No network, no external API."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from backend.agent.tools.base import Tool, ToolContext


class CurrentDatetimeInput(BaseModel):
    """Takes no arguments."""


class CurrentDatetimeTool(Tool):
    name = "get_current_datetime"
    description = "Возвращает текущие дату и время в UTC. Без аргументов."
    input_schema = CurrentDatetimeInput

    async def execute(
        self,
        arguments: CurrentDatetimeInput,
        context: ToolContext,
    ) -> dict[str, Any]:
        now = datetime.now(UTC)

        return {
            "utc_iso": now.isoformat(),
            "date": now.date().isoformat(),
            "time": now.strftime("%H:%M:%S"),
            "timezone": "UTC",
        }

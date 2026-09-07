"""Structured LLM output on top of a plain chat API.

The local Gemma build has no native JSON mode and no function calling, so the
contract is enforced from this side:

1. a strict system prompt plus the model's JSON Schema,
2. locate the JSON object in whatever the model wrapped it in,
3. validate it with Pydantic,
4. on failure, **one** repair round trip that shows the model its own output
   and the validation error,
5. on a second failure, give up and report it — the caller falls back.

Two LLM calls is the hard ceiling. There is no repair loop.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from backend.config import Settings
from backend.errors import InferenceUnavailableError
from backend.prompts.agent import (
    SCHEMA_INSTRUCTION_TEMPLATE,
    STRUCTURED_REPAIR_SYSTEM_PROMPT,
    STRUCTURED_REPAIR_USER_TEMPLATE,
)
from backend.services.inference_client import InferenceClient

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MAX_ECHOED_OUTPUT = 600


@dataclass(slots=True)
class StructuredResult:
    """Outcome of one structured call. ``value`` is None when it failed."""

    value: BaseModel | None
    repaired: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None


def extract_json_object(text: str) -> str | None:
    """Pull the first balanced ``{...}`` object out of a model response.

    Small local models like to wrap JSON in prose or ``` fences. This walks the
    string tracking brace depth and string state — it is a locator, not a
    parser, and the result is still validated by Pydantic.
    """
    if not text:
        return None

    depth = 0
    start = -1
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : index + 1]
            if depth < 0:
                return None

    return None


class StructuredCaller:
    def __init__(self, inference: InferenceClient, settings: Settings) -> None:
        self.inference = inference
        self.settings = settings

    async def call(
        self,
        model: type[T],
        system_prompt: str,
        user_prompt: str,
    ) -> StructuredResult:
        schema = json.dumps(model.model_json_schema(), ensure_ascii=False)
        instruction = SCHEMA_INSTRUCTION_TEMPLATE.format(schema=schema)

        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n{instruction}"},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw = await self._chat(messages)
        except InferenceUnavailableError as exc:
            # Propagate: an unreachable model is not something a repair fixes.
            logger.warning("structured_call_inference_unavailable model=%s", model.__name__)
            raise exc

        parsed, error = self._parse(model, raw)

        if parsed is not None:
            return StructuredResult(value=parsed)

        if self.settings.agent_structured_repair_attempts < 1:
            return StructuredResult(value=None, error=error)

        logger.info(
            "structured_call_repairing model=%s error=%s",
            model.__name__,
            error,
        )

        repair_messages = [
            {"role": "system", "content": STRUCTURED_REPAIR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": STRUCTURED_REPAIR_USER_TEMPLATE.format(
                    schema=schema,
                    previous=(raw or "")[:MAX_ECHOED_OUTPUT],
                    error=error,
                ),
            },
        ]

        repaired_raw = await self._chat(repair_messages)
        repaired, repair_error = self._parse(model, repaired_raw)

        if repaired is not None:
            logger.info("structured_call_repaired model=%s", model.__name__)
            return StructuredResult(value=repaired, repaired=True)

        logger.warning(
            "structured_call_failed model=%s error=%s",
            model.__name__,
            repair_error,
        )
        return StructuredResult(value=None, repaired=True, error=repair_error)

    async def _chat(self, messages: list[dict[str, str]]) -> str:
        return await self.inference.chat(
            messages,
            max_tokens=self.settings.agent_structured_max_tokens,
            temperature=self.settings.agent_router_temperature,
        )

    @staticmethod
    def _parse(model: type[T], raw: str) -> tuple[T | None, str | None]:
        candidate = extract_json_object(raw or "")

        if candidate is None:
            return None, "response contained no JSON object"

        try:
            payload: Any = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc.msg}"

        if not isinstance(payload, dict):
            return None, "top level JSON value is not an object"

        try:
            return model.model_validate(payload), None
        except ValidationError as exc:
            first = exc.errors()[0]
            location = ".".join(str(part) for part in first.get("loc", ())) or "<root>"
            return None, f"{location}: {first.get('msg', 'validation failed')}"

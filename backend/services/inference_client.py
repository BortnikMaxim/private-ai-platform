"""HTTP client for the local MLX/Gemma inference service.

One :class:`httpx.AsyncClient` is created during application startup and reused
for every request; it is closed again on shutdown.
"""

import logging
import time
from typing import Any

import httpx

from backend.errors import InferenceUnavailableError
from backend.observability import (
    LLM_REQUEST_DURATION_SECONDS,
    LLM_REQUESTS_TOTAL,
    LLM_TOKENS_TOTAL,
)
from backend.tracing import NULL_TRACER, Tracer

logger = logging.getLogger(__name__)

UNKNOWN_MODEL = "unknown"


class InferenceClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        health_timeout: float = 3.0,
        default_max_tokens: int = 500,
        default_temperature: float = 0.2,
        client: httpx.AsyncClient | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.tracer = tracer or NULL_TRACER
        self.health_timeout = health_timeout
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature

        # The key travels in a header only; it is never logged.
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={"X-API-Key": api_key},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        try:
            response = await self._client.get("/health", timeout=self.health_timeout)
            response.raise_for_status()
            return True
        except Exception:  # noqa: BLE001 - health checks never raise
            return False

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Call ``POST /v1/chat`` and return the assistant text.

        This is the single place every LLM call passes through, so it owns the
        generation span and the LLM metrics. The return type stays a plain
        string: instrumentation belongs here, not in the callers' signatures.
        """
        effective_max_tokens = max_tokens or self.default_max_tokens
        effective_temperature = (
            self.default_temperature if temperature is None else temperature
        )

        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": effective_temperature,
        }

        started = time.perf_counter()

        with self.tracer.generation(
            "llm.generate",
            messages=len(messages),
            max_tokens=effective_max_tokens,
            temperature=effective_temperature,
        ) as span:
            # Prompts are content: only attached when capture is enabled.
            span.set_content(input=messages)

            try:
                response = await self._client.post("/v1/chat", json=payload)
            except httpx.HTTPError as exc:
                # Log the failure class, never the prompt contents.
                logger.warning(
                    "inference_request_failed error=%s messages=%d",
                    type(exc).__name__,
                    len(messages),
                )
                self._record(UNKNOWN_MODEL, "unavailable", started)
                raise InferenceUnavailableError() from exc

            if response.status_code >= 400:
                logger.error(
                    "inference_bad_status status=%d messages=%d",
                    response.status_code,
                    len(messages),
                )
                self._record(UNKNOWN_MODEL, "http_error", started)
                raise InferenceUnavailableError(
                    f"Inference service returned HTTP {response.status_code}"
                )

            try:
                body = response.json()
                content = body["message"]["content"]
            except (ValueError, KeyError, TypeError) as exc:
                logger.error(
                    "inference_malformed_response status=%d", response.status_code
                )
                self._record(UNKNOWN_MODEL, "malformed", started)
                raise InferenceUnavailableError(
                    "Inference service returned an unexpected payload"
                ) from exc

            model = body.get("model") or UNKNOWN_MODEL
            usage = _usage_from(body)
            duration = self._record(model, "success", started, usage)

            span.update_generation(
                model=model,
                usage=usage,
                # Naming the origin keeps measured counts distinguishable from
                # anything estimated later.
                usage_source=(usage or {}).get("source"),
                finish_reason=body.get("finish_reason"),
                duration_ms=round(duration * 1000, 1),
            )
            span.set_content(output=content)

            return content

    def _record(
        self,
        model: str,
        status: str,
        started: float,
        usage: dict[str, int] | None = None,
    ) -> float:
        duration = time.perf_counter() - started

        LLM_REQUESTS_TOTAL.labels(model=model, status=status).inc()
        LLM_REQUEST_DURATION_SECONDS.labels(model=model).observe(duration)

        if usage:
            LLM_TOKENS_TOTAL.labels(model=model, kind="prompt").inc(
                usage["prompt_tokens"]
            )
            LLM_TOKENS_TOTAL.labels(model=model, kind="completion").inc(
                usage["completion_tokens"]
            )

        return duration


def _usage_from(body: dict[str, Any]) -> dict[str, int] | None:
    """Token usage from the response, or None when the service reported none.

    Absent counts are reported as absent. Estimating them from message lengths
    and presenting the guess as usage would make the token metrics quietly
    wrong, which is worse than having no metric.
    """
    usage = body.get("usage")

    if not isinstance(usage, dict):
        return None

    try:
        prompt = int(usage["prompt_tokens"])
        completion = int(usage["completion_tokens"])
    except (KeyError, TypeError, ValueError):
        return None

    total = usage.get("total_tokens")

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(total) if total is not None else prompt + completion,
        "source": str(usage.get("source") or "unknown"),
    }

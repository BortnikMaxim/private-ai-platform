"""HTTP client for the local MLX/Gemma inference service.

One :class:`httpx.AsyncClient` is created during application startup and reused
for every request; it is closed again on shutdown.
"""

import logging
from typing import Any

import httpx

from backend.errors import InferenceUnavailableError

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.base_url = base_url.rstrip("/")
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
        """Call ``POST /v1/chat`` and return the assistant text."""
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens or self.default_max_tokens,
            "temperature": (
                self.default_temperature if temperature is None else temperature
            ),
        }

        try:
            response = await self._client.post("/v1/chat", json=payload)
        except httpx.HTTPError as exc:
            # Log the failure class, never the prompt contents.
            logger.warning(
                "inference_request_failed error=%s messages=%d",
                type(exc).__name__,
                len(messages),
            )
            raise InferenceUnavailableError() from exc

        if response.status_code >= 400:
            logger.error(
                "inference_bad_status status=%d messages=%d",
                response.status_code,
                len(messages),
            )
            raise InferenceUnavailableError(
                f"Inference service returned HTTP {response.status_code}"
            )

        try:
            content = response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            logger.error("inference_malformed_response status=%d", response.status_code)
            raise InferenceUnavailableError(
                "Inference service returned an unexpected payload"
            ) from exc

        return content

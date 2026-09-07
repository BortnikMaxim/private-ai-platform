import httpx
import pytest

from backend.errors import InferenceUnavailableError
from backend.services.inference_client import InferenceClient


def build_client(handler) -> InferenceClient:
    transport = httpx.MockTransport(handler)

    return InferenceClient(
        base_url="http://inference.test",
        api_key="secret-key",
        client=httpx.AsyncClient(
            transport=transport,
            base_url="http://inference.test",
            headers={"X-API-Key": "secret-key"},
        ),
    )


async def test_chat_returns_the_assistant_content_and_sends_the_api_key():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("X-API-Key", "")
        seen["path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "model": "gemma",
                "message": {"role": "assistant", "content": "ответ"},
                "generation_time_seconds": 0.1,
            },
        )

    client = build_client(handler)

    try:
        answer = await client.chat([{"role": "user", "content": "вопрос"}])
    finally:
        await client.aclose()

    assert answer == "ответ"
    assert seen["key"] == "secret-key"
    assert seen["path"] == "/v1/chat"


async def test_chat_raises_on_connection_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = build_client(handler)

    try:
        with pytest.raises(InferenceUnavailableError):
            await client.chat([{"role": "user", "content": "вопрос"}])
    finally:
        await client.aclose()


@pytest.mark.parametrize("status_code", [401, 422, 500])
async def test_chat_raises_on_non_2xx_responses(status_code):
    client = build_client(lambda request: httpx.Response(status_code, json={"detail": "no"}))

    try:
        with pytest.raises(InferenceUnavailableError) as error:
            await client.chat([{"role": "user", "content": "вопрос"}])
    finally:
        await client.aclose()

    assert str(status_code) in error.value.detail


async def test_chat_raises_on_a_malformed_payload():
    client = build_client(lambda request: httpx.Response(200, json={"unexpected": True}))

    try:
        with pytest.raises(InferenceUnavailableError):
            await client.chat([{"role": "user", "content": "вопрос"}])
    finally:
        await client.aclose()


async def test_health_is_false_when_the_service_is_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = build_client(handler)

    try:
        assert await client.health() is False
    finally:
        await client.aclose()


async def test_health_is_true_on_200():
    client = build_client(lambda request: httpx.Response(200, json={"status": "ok"}))

    try:
        assert await client.health() is True
    finally:
        await client.aclose()

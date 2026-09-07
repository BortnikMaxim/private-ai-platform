"""Agent tests against the live stack.

These drive the running HTTP API, so they exercise the real router prompt with
the real local Gemma, real retrieval and the real tool registry. Nothing here
touches the public internet — every dependency is on localhost.

    docker compose up -d
    alembic upgrade head
    uvicorn inference.app:app --port 8001
    uvicorn backend.app:app --port 8000
    celery -A backend.worker.celery_app worker --pool=solo --queues=documents
    pytest -m integration
"""

import os

import httpx
import pytest

pytestmark = pytest.mark.integration

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
# Routing plus composition on a local 4B model is not fast.
TIMEOUT = httpx.Timeout(180.0)


def _skip_unless_ready() -> dict:
    try:
        response = httpx.get(f"{BACKEND_URL}/health", timeout=5.0)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"backend is not running at {BACKEND_URL}: {exc}")

    health = response.json()

    if health.get("inference") != "ok":
        pytest.skip("inference service is not available")

    return health


@pytest.fixture(scope="module")
def api() -> httpx.Client:
    _skip_unless_ready()

    with httpx.Client(base_url=BACKEND_URL, timeout=TIMEOUT) as client:
        yield client


@pytest.fixture
def conversation(api: httpx.Client) -> str:
    response = api.post("/conversations", json={"title": "agent integration"})
    response.raise_for_status()
    conversation_id = response.json()["id"]

    yield conversation_id

    api.delete(f"/conversations/{conversation_id}")


def _ready_document(api: httpx.Client) -> dict:
    listing = api.get("/documents", params={"limit": 50}).json()
    ready = [item for item in listing["items"] if item["status"] == "ready"]

    if not ready:
        pytest.skip("no ready document to query; upload one first")

    return ready[0]


def test_calculator_question_routes_to_a_tool(api, conversation):
    response = api.post(
        f"/conversations/{conversation}/agent",
        json={"content": "Сколько будет 125 * 8?", "use_rag": False},
    )
    body = response.json()

    assert response.status_code == 201
    assert body["route"] == "tool"
    assert [tool["name"] for tool in body["tools_used"]] == ["calculator"]
    assert body["tools_used"][0]["success"] is True
    assert "1000" in body["message"]["content"]


def test_document_question_routes_to_rag_and_cites_sources(api, conversation):
    document = _ready_document(api)

    response = api.post(
        f"/conversations/{conversation}/agent",
        json={
            "content": "Какие проекты описаны в документе и кто ими руководит?",
            "use_rag": True,
            "document_ids": [document["id"]],
        },
    )
    body = response.json()

    assert response.status_code == 201
    assert body["route"] == "rag_search"
    assert body["sources"], "a grounded answer must cite its chunks"
    assert body["sources"][0]["document_id"] == document["id"]
    assert body["sources"][0]["rerank_score"] is not None


def test_agent_turns_are_persisted_in_the_conversation(api, conversation):
    api.post(
        f"/conversations/{conversation}/agent",
        json={"content": "Сколько будет 2 + 2?", "use_rag": False},
    )

    detail = api.get(f"/conversations/{conversation}").json()

    assert [message["role"] for message in detail["messages"]] == [
        "user",
        "assistant",
    ]


def test_response_exposes_no_internal_reasoning(api, conversation):
    response = api.post(
        f"/conversations/{conversation}/agent",
        json={"content": "Сколько будет 3 * 3?", "use_rag": False},
    )

    assert set(response.json()) == {"message", "route", "tools_used", "sources"}


def test_plain_rag_endpoints_still_work(api):
    _ready_document(api)

    response = api.post(
        "/rag/retrieve",
        json={"question": "проекты", "top_k": 2, "candidate_k": 5},
    )

    assert response.status_code == 200
    assert "vector_results" in response.json()

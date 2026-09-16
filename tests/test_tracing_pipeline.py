"""Tracing wired into the real pipeline: retrieval, tools, LLM and correlation.

The Langfuse client is the in-memory fake from ``test_tracing``; everything
else — BM25, fusion, the tool registry, the HTTP client — is the real code.
"""

import httpx
import pytest

from backend.agent.tools.base import ToolContext
from backend.agent.tools.registry import default_registry
from backend.middleware import REQUEST_ID_HEADER
from backend.services.inference_client import InferenceClient
from backend.services.rag_service import MODE_DENSE, MODE_HYBRID, RagService
from backend.tracing import Tracer
from tests.test_tracing import FakeLangfuse

RU_TEXTS = [
    "Проект Атлас описывает миграцию биллинга на новую платформу",
    "Проект Борей отвечает за складскую логистику и автоматизацию",
]


@pytest.fixture
def fake_langfuse() -> FakeLangfuse:
    return FakeLangfuse()


@pytest.fixture
def tracer(settings, fake_langfuse) -> Tracer:
    settings.langfuse_enabled = True
    settings.langfuse_public_key = "pk-test"
    settings.langfuse_secret_key = "sk-test"
    return Tracer(settings, client=fake_langfuse)


@pytest.fixture
def traced_rag(embeddings, vector_store, settings, lexical_index, tracer) -> RagService:
    return RagService(
        embeddings=embeddings,
        vector_store=vector_store,
        settings=settings,
        lexical=lexical_index,
        tracer=tracer,
    )


# ---------------------------------------------------------------------------
# Retrieval spans
# ---------------------------------------------------------------------------


async def test_hybrid_retrieval_emits_a_span_per_stage(
    traced_rag,
    fake_langfuse,
    session_factory,
    make_document,
    user,
):
    await make_document(user, filename="ru.pdf", texts=RU_TEXTS)

    async with session_factory() as session:
        await traced_rag.retrieve(
            question="складская логистика",
            user_id=str(user.id),
            session=session,
            mode=MODE_HYBRID,
        )

    assert fake_langfuse.names() == [
        "retrieval",
        "retrieval.dense",
        "retrieval.lexical",
        "retrieval.fusion",
        "retrieval.rerank",
    ]


async def test_dense_mode_emits_no_lexical_or_fusion_span(
    traced_rag,
    fake_langfuse,
    session_factory,
    make_document,
    user,
):
    """Spans describe what actually ran, not a fixed template."""
    await make_document(user, filename="ru.pdf", texts=RU_TEXTS)

    async with session_factory() as session:
        await traced_rag.retrieve(
            question="складская логистика",
            user_id=str(user.id),
            session=session,
            mode=MODE_DENSE,
        )

    names = fake_langfuse.names()
    assert "retrieval.lexical" not in names
    assert "retrieval.fusion" not in names
    assert "retrieval.dense" in names


async def test_skipping_the_reranker_skips_its_span(
    traced_rag,
    fake_langfuse,
    session_factory,
    make_document,
    user,
):
    await make_document(user, filename="ru.pdf", texts=RU_TEXTS)

    async with session_factory() as session:
        await traced_rag.retrieve(
            question="складская логистика",
            user_id=str(user.id),
            session=session,
            mode=MODE_HYBRID,
            rerank=False,
        )

    assert "retrieval.rerank" not in fake_langfuse.names()
    assert fake_langfuse.named("retrieval")["metadata"]["reranked"] is False


async def test_retrieval_spans_carry_stage_counts(
    traced_rag,
    fake_langfuse,
    session_factory,
    make_document,
    user,
):
    await make_document(user, filename="ru.pdf", texts=RU_TEXTS)

    async with session_factory() as session:
        results = await traced_rag.retrieve(
            question="складская логистика",
            user_id=str(user.id),
            session=session,
            mode=MODE_HYBRID,
        )

    dense = fake_langfuse.named("retrieval.dense")["metadata"]
    fusion = fake_langfuse.named("retrieval.fusion")["metadata"]
    rerank = fake_langfuse.named("retrieval.rerank")["metadata"]
    root = fake_langfuse.named("retrieval")["metadata"]

    assert dense["candidates"] == 2
    assert fusion["dense_candidates"] == 2
    assert fusion["fused_candidates"] >= 2
    assert fusion["rrf_k"] == 60
    assert rerank["model"] == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    assert root["mode"] == "hybrid"
    assert root["results"] == len(results)


async def test_retrieval_spans_never_carry_chunk_text_by_default(
    traced_rag,
    fake_langfuse,
    session_factory,
    make_document,
    user,
):
    await make_document(user, filename="ru.pdf", texts=RU_TEXTS)

    async with session_factory() as session:
        await traced_rag.retrieve(
            question="складская логистика",
            user_id=str(user.id),
            session=session,
            mode=MODE_HYBRID,
        )

    blob = str(fake_langfuse.spans)
    assert "Проект Атлас" not in blob
    assert "складская логистика" not in blob


async def test_the_question_is_captured_only_when_opted_in(
    embeddings,
    vector_store,
    settings,
    lexical_index,
    fake_langfuse,
    session_factory,
    make_document,
    user,
):
    settings.langfuse_enabled = True
    settings.langfuse_public_key = "pk"
    settings.langfuse_secret_key = "sk"
    settings.langfuse_capture_content = True

    service = RagService(
        embeddings=embeddings,
        vector_store=vector_store,
        settings=settings,
        lexical=lexical_index,
        tracer=Tracer(settings, client=fake_langfuse),
    )
    await make_document(user, filename="ru.pdf", texts=RU_TEXTS)

    async with session_factory() as session:
        await service.retrieve(
            question="складская логистика",
            user_id=str(user.id),
            session=session,
            mode=MODE_HYBRID,
        )

    assert fake_langfuse.named("retrieval")["input"] == "складская логистика"


async def test_a_broken_tracer_does_not_break_retrieval(
    embeddings,
    vector_store,
    settings,
    lexical_index,
    session_factory,
    make_document,
    user,
):
    settings.langfuse_enabled = True
    settings.langfuse_public_key = "pk"
    settings.langfuse_secret_key = "sk"

    service = RagService(
        embeddings=embeddings,
        vector_store=vector_store,
        settings=settings,
        lexical=lexical_index,
        tracer=Tracer(settings, client=FakeLangfuse(fail_on_start=True)),
    )
    await make_document(user, filename="ru.pdf", texts=RU_TEXTS)

    async with session_factory() as session:
        results = await service.retrieve(
            question="складская логистика",
            user_id=str(user.id),
            session=session,
            mode=MODE_HYBRID,
        )

    # The whole point of fail-open: retrieval is unaffected.
    assert results
    assert results[0]["filename"] == "ru.pdf"


# ---------------------------------------------------------------------------
# Tool spans
# ---------------------------------------------------------------------------


async def test_a_tool_call_emits_a_span(tracer, fake_langfuse, settings):
    registry = default_registry(tracer=tracer)

    await registry.execute("calculator", {"expression": "125 * 8"}, ToolContext())

    span = fake_langfuse.named("agent.tool")
    assert span["as_type"] == "tool"
    assert span["metadata"]["tool"] == "calculator"
    assert span["metadata"]["status"] == "success"
    assert span["metadata"]["duration_ms"] >= 0


async def test_a_failing_tool_records_its_status(tracer, fake_langfuse):
    registry = default_registry(tracer=tracer)

    await registry.execute("calculator", {"expression": "1/0"}, ToolContext())

    assert fake_langfuse.named("agent.tool")["metadata"]["status"] == "tool_error"


async def test_an_unknown_tool_records_its_status(tracer, fake_langfuse):
    registry = default_registry(tracer=tracer)

    await registry.execute("run_shell", {"cmd": "rm -rf /"}, ToolContext())

    assert fake_langfuse.named("agent.tool")["metadata"]["status"] == "unknown_tool"


async def test_tool_arguments_are_not_captured_by_default(tracer, fake_langfuse):
    registry = default_registry(tracer=tracer)

    await registry.execute(
        "calculator", {"expression": "123456 * 7"}, ToolContext()
    )

    assert "123456" not in str(fake_langfuse.spans)


# ---------------------------------------------------------------------------
# LLM span and token usage
# ---------------------------------------------------------------------------


def build_client(handler, tracer: Tracer) -> InferenceClient:
    return InferenceClient(
        base_url="http://inference.test",
        api_key="secret-key",
        tracer=tracer,
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://inference.test",
        ),
    )


def reply(**extra):
    body = {
        "model": "mlx-community/gemma-3-4b-it-qat-4bit",
        "message": {"role": "assistant", "content": "ответ"},
        "generation_time_seconds": 1.5,
    }
    body.update(extra)
    return httpx.Response(200, json=body)


async def test_the_llm_span_records_provider_reported_usage(tracer, fake_langfuse):
    client = build_client(
        lambda request: reply(
            usage={
                "prompt_tokens": 120,
                "completion_tokens": 42,
                "total_tokens": 162,
                "source": "local_tokenizer",
            },
            finish_reason="stop",
        ),
        tracer,
    )

    try:
        await client.chat([{"role": "user", "content": "вопрос"}])
    finally:
        await client.aclose()

    span = fake_langfuse.named("llm.generate")
    assert span["as_type"] == "generation"
    assert span["model"] == "mlx-community/gemma-3-4b-it-qat-4bit"
    assert span["usage_details"] == {"input": 120, "output": 42, "total": 162}
    # The origin travels with the numbers so measured counts stay
    # distinguishable from anything estimated.
    assert span["metadata"]["usage_source"] == "local_tokenizer"
    assert span["metadata"]["finish_reason"] == "stop"


async def test_usage_is_omitted_when_the_service_does_not_report_it(
    tracer,
    fake_langfuse,
):
    """No usage field means no usage recorded — never an invented zero."""
    client = build_client(lambda request: reply(), tracer)

    try:
        await client.chat([{"role": "user", "content": "вопрос"}])
    finally:
        await client.aclose()

    assert "usage_details" not in fake_langfuse.named("llm.generate")


async def test_partial_usage_is_rejected_rather_than_guessed(tracer, fake_langfuse):
    client = build_client(
        lambda request: reply(usage={"prompt_tokens": 10}), tracer
    )

    try:
        await client.chat([{"role": "user", "content": "вопрос"}])
    finally:
        await client.aclose()

    assert "usage_details" not in fake_langfuse.named("llm.generate")


async def test_the_llm_span_does_not_carry_the_prompt_by_default(
    tracer,
    fake_langfuse,
):
    client = build_client(lambda request: reply(), tracer)

    try:
        await client.chat([{"role": "user", "content": "мой секретный вопрос"}])
    finally:
        await client.aclose()

    blob = str(fake_langfuse.spans)
    assert "секретный" not in blob
    assert "ответ" not in blob
    # The API key must never reach a span either.
    assert "secret-key" not in blob


async def test_an_inference_outage_is_recorded_and_still_raises(tracer, fake_langfuse):
    from backend.errors import InferenceUnavailableError

    def unreachable(request):
        raise httpx.ConnectError("connection refused", request=request)

    client = build_client(unreachable, tracer)

    try:
        with pytest.raises(InferenceUnavailableError):
            await client.chat([{"role": "user", "content": "вопрос"}])
    finally:
        await client.aclose()

    span = fake_langfuse.named("llm.generate")
    assert span["level"] == "ERROR"
    assert span["status_message"] == "InferenceUnavailableError"


async def test_a_broken_tracer_does_not_break_the_llm_call(settings, fake_langfuse):
    settings.langfuse_enabled = True
    settings.langfuse_public_key = "pk"
    settings.langfuse_secret_key = "sk"

    client = build_client(
        lambda request: reply(),
        Tracer(settings, client=FakeLangfuse(fail_on_start=True)),
    )

    try:
        answer = await client.chat([{"role": "user", "content": "вопрос"}])
    finally:
        await client.aclose()

    assert answer == "ответ"


# ---------------------------------------------------------------------------
# Request correlation through the API
# ---------------------------------------------------------------------------


async def test_the_response_carries_a_request_id(client):
    response = await client.get("/health")

    assert response.headers.get(REQUEST_ID_HEADER)


async def test_a_caller_supplied_request_id_is_reused(client):
    response = await client.get("/health", headers={REQUEST_ID_HEADER: "trace-me-123"})

    assert response.headers[REQUEST_ID_HEADER] == "trace-me-123"


async def test_a_hostile_request_id_is_replaced(client):
    """A client-controlled value ends up in logs and in a trace id seed."""
    response = await client.get(
        "/health",
        headers={REQUEST_ID_HEADER: "bad\nid injected=field"},
    )

    returned = response.headers[REQUEST_ID_HEADER]
    assert "\n" not in returned
    assert returned != "bad\nid injected=field"


async def test_each_request_gets_a_distinct_id(client):
    first = (await client.get("/health")).headers[REQUEST_ID_HEADER]
    second = (await client.get("/health")).headers[REQUEST_ID_HEADER]

    assert first != second


async def test_the_request_id_reaches_the_log_line(client, caplog):
    with caplog.at_level("INFO", logger="backend.request"):
        response = await client.get("/health")

    request_id = response.headers[REQUEST_ID_HEADER]
    assert any(request_id in record.message for record in caplog.records)


async def test_the_request_id_seeds_the_trace(app, make_client, user, fake_langfuse, settings):
    """Correlating a log line with a trace is the whole point of the id."""
    settings.langfuse_enabled = True
    settings.langfuse_public_key = "pk"
    settings.langfuse_secret_key = "sk"
    app.state.tracer = Tracer(settings, client=fake_langfuse)

    async with make_client(user) as authed:
        await authed.post(
            "/rag/ask",
            json={"question": "вопрос"},
            headers={REQUEST_ID_HEADER: "corr-1"},
        )

    trace = fake_langfuse.named("rag.request")
    assert trace is not None
    assert trace["metadata"]["request_id"] == "corr-1"

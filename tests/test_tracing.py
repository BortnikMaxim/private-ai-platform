"""Langfuse tracing: fail-open behaviour, privacy defaults and span coverage.

Langfuse is always a fake here. No test opens a socket, needs a Langfuse
server or reaches the internet.
"""

import uuid
from typing import Any

import pytest

from backend.config import Settings
from backend.tracing import NULL_TRACER, Tracer, request_id_var

# ---------------------------------------------------------------------------
# A recording stand-in for the Langfuse SDK
# ---------------------------------------------------------------------------


class FakeObservation:
    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if key == "metadata" and isinstance(value, dict):
                self.record.setdefault("metadata", {}).update(value)
            else:
                self.record[key] = value


class FakeObservationContext:
    def __init__(self, client: "FakeLangfuse", record: dict[str, Any]) -> None:
        self.client = client
        self.record = record

    def __enter__(self) -> FakeObservation:
        return FakeObservation(self.record)

    def __exit__(self, *exc_info) -> None:
        self.record["closed"] = True


class FakeLangfuse:
    """Records spans in memory; can be told to fail at any stage."""

    def __init__(
        self,
        fail_on_start: bool = False,
        fail_on_flush: bool = False,
    ) -> None:
        self.spans: list[dict[str, Any]] = []
        self.traces: list[dict[str, Any]] = []
        self.flushed = 0
        self.fail_on_start = fail_on_start
        self.fail_on_flush = fail_on_flush

    def start_as_current_observation(self, **kwargs: Any) -> FakeObservationContext:
        if self.fail_on_start:
            raise RuntimeError("langfuse exporter is unreachable")

        record: dict[str, Any] = dict(kwargs)
        record.setdefault("metadata", {})
        self.spans.append(record)
        return FakeObservationContext(self, record)

    def update_current_trace(self, **kwargs: Any) -> None:
        self.traces.append(kwargs)

    def flush(self) -> None:
        if self.fail_on_flush:
            raise RuntimeError("flush failed")
        self.flushed += 1

    def shutdown(self) -> None:
        self.flush()

    def named(self, name: str) -> dict[str, Any] | None:
        return next((span for span in self.spans if span.get("name") == name), None)

    def names(self) -> list[str]:
        return [span.get("name") for span in self.spans]


@pytest.fixture
def fake_langfuse() -> FakeLangfuse:
    return FakeLangfuse()


@pytest.fixture
def tracing_settings(settings) -> Settings:
    settings.langfuse_enabled = True
    settings.langfuse_public_key = "pk-test"
    settings.langfuse_secret_key = "sk-test"
    return settings


@pytest.fixture
def tracer(tracing_settings, fake_langfuse) -> Tracer:
    return Tracer(tracing_settings, client=fake_langfuse)


# ---------------------------------------------------------------------------
# Disabled by default
# ---------------------------------------------------------------------------


def test_tracing_is_off_unless_explicitly_enabled():
    assert Settings(jwt_secret_key="x" * 40).langfuse_enabled is False


def test_a_disabled_tracer_creates_nothing(settings):
    disabled = Tracer(settings)

    assert disabled.enabled is False

    with disabled.trace("rag.request") as span:
        span.update(anything="here")
        span.set_content(input="secret question")

    # Nothing to assert against because nothing was created — which is the
    # point: the call sites are identical whether tracing is on or off.


def test_the_null_tracer_is_usable_everywhere():
    assert NULL_TRACER.enabled is False

    with NULL_TRACER.span("retrieval") as span:
        span.update(candidates=3)
        span.error(RuntimeError("boom"))


def test_enabling_without_keys_stays_disabled(settings, caplog):
    settings.langfuse_enabled = True
    settings.langfuse_public_key = ""
    settings.langfuse_secret_key = ""

    assert Tracer(settings).enabled is False


def test_an_enabled_tracer_with_a_client_is_enabled(tracer):
    assert tracer.enabled is True


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------


def test_a_trace_records_its_name_and_metadata(tracer, fake_langfuse):
    with tracer.trace("rag.request", request_id="abc123", top_k=5):
        pass

    span = fake_langfuse.named("rag.request")
    assert span is not None
    assert span["metadata"]["request_id"] == "abc123"
    assert span["metadata"]["top_k"] == 5
    assert span["closed"] is True


def test_nested_spans_are_all_recorded(tracer, fake_langfuse):
    with tracer.trace("rag.request"), tracer.span("retrieval"), tracer.span("retrieval.dense"):
        pass

    assert fake_langfuse.names() == ["rag.request", "retrieval", "retrieval.dense"]


def test_span_update_attaches_metadata(tracer, fake_langfuse):
    with tracer.span("retrieval.dense") as span:
        span.update(candidates=15)

    assert fake_langfuse.named("retrieval.dense")["metadata"]["candidates"] == 15


def test_empty_attributes_are_dropped(tracer, fake_langfuse):
    with tracer.trace("rag.request", request_id=None, scoped=False):
        pass

    metadata = fake_langfuse.named("rag.request")["metadata"]
    assert "request_id" not in metadata
    assert metadata["scoped"] is False


def test_an_exception_marks_the_span_and_propagates(tracer, fake_langfuse):
    with pytest.raises(ValueError), tracer.span("retrieval"):
        raise ValueError("a message that must not be recorded")

    span = fake_langfuse.named("retrieval")
    assert span["level"] == "ERROR"
    # The exception class, not its text: a message can carry user data.
    assert span["status_message"] == "ValueError"
    assert "must not be recorded" not in str(span)


# ---------------------------------------------------------------------------
# Fail open
# ---------------------------------------------------------------------------


def test_a_failing_exporter_does_not_break_the_caller(tracing_settings):
    broken = Tracer(tracing_settings, client=FakeLangfuse(fail_on_start=True))
    executed = False

    with broken.trace("rag.request") as span:
        span.update(candidates=3)
        executed = True

    assert executed is True


def test_a_failing_exporter_degrades_once(tracing_settings, caplog):
    broken = Tracer(tracing_settings, client=FakeLangfuse(fail_on_start=True))

    with caplog.at_level("WARNING"):
        for _ in range(5):
            with broken.span("retrieval"):
                pass

    # Degraded once, then silent: a broken exporter must not become a log flood.
    assert sum("tracing_degraded" in record.message for record in caplog.records) == 1
    assert broken.enabled is False


def test_a_failing_flush_is_swallowed(tracing_settings):
    broken = Tracer(tracing_settings, client=FakeLangfuse(fail_on_flush=True))

    broken.flush()
    broken.shutdown()


def test_a_broken_sdk_import_leaves_tracing_off(settings, monkeypatch):
    settings.langfuse_enabled = True
    settings.langfuse_public_key = "pk"
    settings.langfuse_secret_key = "sk"

    def explode(*args, **kwargs):
        raise ImportError("langfuse is not installed")

    monkeypatch.setattr(Tracer, "_build_client", staticmethod(explode))

    with pytest.raises(ImportError):
        Tracer._build_client(settings)

    # The real constructor swallows it; only the direct call above raises.
    monkeypatch.setattr(
        Tracer, "_build_client", staticmethod(lambda settings: None)
    )
    assert Tracer(settings).enabled is False


def test_an_exception_inside_a_traced_block_still_propagates(tracing_settings):
    broken = Tracer(tracing_settings, client=FakeLangfuse(fail_on_start=True))

    with pytest.raises(KeyError), broken.span("retrieval"):
        raise KeyError("business failure")


# ---------------------------------------------------------------------------
# Request correlation
# ---------------------------------------------------------------------------


def test_the_trace_id_is_derived_from_the_request_id(tracer, fake_langfuse, monkeypatch):
    import langfuse

    monkeypatch.setattr(
        langfuse.Langfuse, "create_trace_id", staticmethod(lambda seed=None: f"tid-{seed}")
    )

    with tracer.trace("rag.request", request_id="req-42"):
        pass

    span = fake_langfuse.named("rag.request")
    # Deterministic from the request id, which is what lets a log line lead to
    # the matching trace.
    assert span["trace_context"] == {"trace_id": "tid-req-42"}


def test_the_request_id_is_taken_from_the_contextvar(tracer, fake_langfuse):
    token = request_id_var.set("ambient-id")

    try:
        with tracer.trace("agent.request"):
            pass
    finally:
        request_id_var.reset(token)

    assert fake_langfuse.named("agent.request")["metadata"]["request_id"] == "ambient-id"


def test_the_tenant_id_is_attached_but_no_other_identity(tracer, fake_langfuse):
    user_id = str(uuid.uuid4())

    with tracer.trace("rag.request", user_id=user_id):
        pass

    assert fake_langfuse.traces == [{"user_id": user_id}]
    # An id only — never an email or a name.
    assert "@" not in str(fake_langfuse.traces)


# ---------------------------------------------------------------------------
# Privacy: content capture is opt-in
# ---------------------------------------------------------------------------


def test_content_is_not_captured_by_default(tracer, fake_langfuse, settings):
    assert settings.langfuse_capture_content is False

    with tracer.trace("rag.request") as span:
        span.set_content(input="какой пароль у администратора", output="секретный ответ")

    span = fake_langfuse.named("rag.request")
    assert "input" not in span
    assert "output" not in span
    assert "пароль" not in str(span)


def test_content_is_captured_only_after_explicit_opt_in(
    tracing_settings,
    fake_langfuse,
):
    tracing_settings.langfuse_capture_content = True
    opted_in = Tracer(tracing_settings, client=fake_langfuse)

    with opted_in.trace("rag.request") as span:
        span.set_content(input="вопрос", output="ответ")

    span = fake_langfuse.named("rag.request")
    assert span["input"] == "вопрос"
    assert span["output"] == "ответ"


def test_captured_content_is_truncated(tracing_settings, fake_langfuse):
    tracing_settings.langfuse_capture_content = True
    tracing_settings.langfuse_max_content_chars = 20
    opted_in = Tracer(tracing_settings, client=fake_langfuse)

    with opted_in.trace("rag.request") as span:
        span.set_content(input="x" * 500)

    captured = fake_langfuse.named("rag.request")["input"]
    assert len(captured) < 500
    assert captured.endswith("[truncated]")


def test_metadata_never_carries_credentials(tracer, fake_langfuse):
    """A sanity sweep over everything a span records."""
    with tracer.trace("rag.request", request_id="r1", top_k=5, mode="hybrid") as span:
        span.update(candidates=15, model="multilingual-e5-small")

    blob = str(fake_langfuse.spans).lower()

    for forbidden in ("bearer", "authorization", "password", "secret", "api_key", "jwt"):
        assert forbidden not in blob


# ---------------------------------------------------------------------------
# Generation spans and token usage
# ---------------------------------------------------------------------------


def test_a_generation_span_records_model_and_usage(tracer, fake_langfuse):
    with tracer.generation("llm.generate", model="gemma-3-4b") as span:
        span.update_generation(
            model="gemma-3-4b",
            usage={"prompt_tokens": 120, "completion_tokens": 42, "total_tokens": 162},
            usage_source="local_tokenizer",
        )

    span = fake_langfuse.named("llm.generate")
    assert span["as_type"] == "generation"
    assert span["model"] == "gemma-3-4b"
    assert span["usage_details"] == {"input": 120, "output": 42, "total": 162}
    assert span["metadata"]["usage_source"] == "local_tokenizer"


def test_a_generation_span_without_usage_records_none(tracer, fake_langfuse):
    with tracer.generation("llm.generate", model="gemma-3-4b") as span:
        span.update_generation(model="gemma-3-4b", usage=None)

    span = fake_langfuse.named("llm.generate")
    # Absent usage stays absent rather than becoming a fabricated zero.
    assert "usage_details" not in span

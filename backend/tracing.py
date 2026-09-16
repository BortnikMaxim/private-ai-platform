"""Langfuse tracing, wrapped so it can never take a request down with it.

This complements the existing observability rather than replacing it:

* Prometheus answers "is the service healthy" — aggregates, alerting.
* Structured logs answer "what happened to request X" — correlation, debugging.
* Langfuse answers "where did this answer come from" — per-request stage
  timings, routing decisions, retrieval counts, token usage.

Two rules govern everything here.

**Fail open.** Tracing is diagnostic. A broken exporter, an expired key, a
network partition or a bug in this module must never surface to the caller, so
every Langfuse interaction is wrapped and failures are logged and swallowed.
The no-op path is also what runs when tracing is disabled, which keeps the
instrumented call sites free of ``if enabled`` branches.

**Capture nothing sensitive by default.** Spans carry counts, modes, model
names, durations and statuses. Prompts, questions, retrieved chunks and answers
are content, and content is only attached when ``LANGFUSE_CAPTURE_CONTENT`` is
explicitly switched on.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from backend.config import Settings

logger = logging.getLogger("backend.tracing")

# The id that ties a structured log line, a Prometheus scrape window and a
# Langfuse trace together. Set by RequestContextMiddleware.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def current_request_id() -> str | None:
    return request_id_var.get()


class SpanHandle:
    """What an instrumented call site talks to.

    Always usable: when tracing is off, or after a failure, the methods simply
    do nothing. Call sites never check whether tracing is alive.
    """

    __slots__ = ("_observation", "_tracer")

    def __init__(self, tracer: "Tracer", observation: Any = None) -> None:
        self._tracer = tracer
        self._observation = observation

    def update(self, **attributes: Any) -> None:
        """Attach metadata to this span. Content keys are filtered upstream."""
        if self._observation is None:
            return

        try:
            self._observation.update(metadata=_clean(attributes))
        except Exception as exc:  # noqa: BLE001 - tracing never raises
            self._tracer._degrade("span_update", exc)

    def update_generation(
        self,
        model: str | None = None,
        usage: dict[str, int] | None = None,
        **attributes: Any,
    ) -> None:
        """Attach model and token usage to an LLM generation span."""
        if self._observation is None:
            return

        payload: dict[str, Any] = {"metadata": _clean(attributes)}

        if model:
            payload["model"] = model

        if usage:
            # Langfuse's own field names, so token counts land in its usage UI
            # rather than in free-form metadata.
            payload["usage_details"] = {
                "input": usage.get("prompt_tokens"),
                "output": usage.get("completion_tokens"),
                "total": usage.get("total_tokens"),
            }

        try:
            self._observation.update(**payload)
        except Exception as exc:  # noqa: BLE001
            self._tracer._degrade("generation_update", exc)

    def set_content(self, input: Any = None, output: Any = None) -> None:
        """Attach prompt/response content — only if capture is switched on."""
        if self._observation is None or not self._tracer.capture_content:
            return

        payload: dict[str, Any] = {}

        if input is not None:
            payload["input"] = self._tracer.truncate(input)

        if output is not None:
            payload["output"] = self._tracer.truncate(output)

        if not payload:
            return

        try:
            self._observation.update(**payload)
        except Exception as exc:  # noqa: BLE001
            self._tracer._degrade("content_update", exc)

    def error(self, exc: BaseException) -> None:
        """Mark the span failed, recording the exception type but not its text.

        An exception message can carry a file path, a query or a row of data;
        the class name is enough to find the failure in the trace and the full
        traceback is already in the logs.
        """
        if self._observation is None:
            return

        try:
            self._observation.update(
                level="ERROR",
                status_message=type(exc).__name__,
            )
        except Exception as inner:  # noqa: BLE001
            self._tracer._degrade("span_error", inner)


class Tracer:
    """Creates spans if Langfuse is configured and reachable; otherwise not."""

    def __init__(self, settings: Settings | None = None, client: Any = None) -> None:
        self.settings = settings
        self.capture_content = bool(
            settings and settings.langfuse_capture_content
        )
        self.max_content_chars = (
            settings.langfuse_max_content_chars if settings else 0
        )
        self._client = client
        self._degraded = False

        if client is None and settings is not None and settings.langfuse_enabled:
            self._client = self._build_client(settings)

    # -- lifecycle -------------------------------------------------------

    @staticmethod
    def _build_client(settings: Settings) -> Any:
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            logger.warning(
                "LANGFUSE_ENABLED is set but the key pair is incomplete; "
                "tracing stays off"
            )
            return None

        try:
            from langfuse import Langfuse

            client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
                timeout=settings.langfuse_timeout_seconds,
                environment=settings.langfuse_environment,
            )
            logger.info("langfuse_tracing_enabled host=%s", settings.langfuse_host)
            return client
        except Exception:
            # A missing package or a bad configuration disables tracing; it
            # does not stop the service from starting.
            logger.exception("langfuse_init_failed; continuing without tracing")
            return None

    @property
    def enabled(self) -> bool:
        return self._client is not None and not self._degraded

    def _degrade(self, operation: str, exc: BaseException) -> None:
        """Record a tracing failure once, then stay quiet.

        A failing exporter would otherwise log on every span of every request
        and turn an observability problem into a log-volume problem.
        """
        if not self._degraded:
            self._degraded = True
            logger.warning(
                "tracing_degraded operation=%s error=%s; "
                "traces are dropped, requests are unaffected",
                operation,
                type(exc).__name__,
            )

    def truncate(self, value: Any) -> Any:
        if not isinstance(value, str) or self.max_content_chars <= 0:
            return value

        if len(value) <= self.max_content_chars:
            return value

        return value[: self.max_content_chars] + "…[truncated]"

    def flush(self) -> None:
        if self._client is None:
            return

        try:
            self._client.flush()
        except Exception as exc:  # noqa: BLE001
            self._degrade("flush", exc)

    def shutdown(self) -> None:
        self.flush()

        client = self._client

        if client is None:
            return

        try:
            shutdown = getattr(client, "shutdown", None)
            if callable(shutdown):
                shutdown()
        except Exception as exc:  # noqa: BLE001
            self._degrade("shutdown", exc)

    # -- spans -----------------------------------------------------------

    @contextmanager
    def trace(
        self,
        name: str,
        request_id: str | None = None,
        user_id: str | None = None,
        **attributes: Any,
    ) -> Iterator[SpanHandle]:
        """Root span for one user request.

        The trace id is derived deterministically from ``request_id``, which is
        what lets a developer jump from a log line to the matching trace.
        """
        identifier = request_id or current_request_id()

        if not self.enabled:
            yield SpanHandle(self)
            return

        trace_context = None

        if identifier:
            try:
                from langfuse import Langfuse

                trace_context = {"trace_id": Langfuse.create_trace_id(seed=identifier)}
            except Exception as exc:  # noqa: BLE001
                self._degrade("trace_id", exc)

        metadata = _clean({"request_id": identifier, **attributes})

        with self._observation(
            name=name,
            as_type="span",
            metadata=metadata,
            trace_context=trace_context,
            user_id=user_id,
        ) as handle:
            yield handle

    @contextmanager
    def span(self, name: str, as_type: str = "span", **attributes: Any) -> Iterator[SpanHandle]:
        """Nested span. Parenting comes from the surrounding OTEL context."""
        if not self.enabled:
            yield SpanHandle(self)
            return

        with self._observation(
            name=name,
            as_type=as_type,
            metadata=_clean(attributes),
        ) as handle:
            yield handle

    @contextmanager
    def generation(self, name: str, model: str | None = None, **attributes: Any):
        """Span for an LLM call, so token usage lands in Langfuse's usage view."""
        if not self.enabled:
            yield SpanHandle(self)
            return

        with self._observation(
            name=name,
            as_type="generation",
            metadata=_clean(attributes),
            model=model,
        ) as handle:
            yield handle

    @contextmanager
    def _observation(self, **kwargs: Any) -> Iterator[SpanHandle]:
        """Open one Langfuse observation, swallowing anything that goes wrong.

        If the SDK raises on entry the call site still gets a usable handle and
        the surrounding code runs untouched.
        """
        user_id = kwargs.pop("user_id", None)
        trace_context = kwargs.pop("trace_context", None)

        if trace_context is not None:
            kwargs["trace_context"] = trace_context

        manager = None
        observation = None

        try:
            manager = self._client.start_as_current_observation(**kwargs)
            observation = manager.__enter__()

            if user_id and hasattr(self._client, "update_current_trace"):
                # Tenant attribution, useful for filtering traces. The id only;
                # never an email or a name.
                self._client.update_current_trace(user_id=user_id)
        except Exception as exc:  # noqa: BLE001
            self._degrade("span_start", exc)
            manager = None
            observation = None

        handle = SpanHandle(self, observation)

        try:
            yield handle
        except Exception as exc:
            handle.error(exc)
            _close(manager, self)
            raise

        _close(manager, self)


def _close(manager: Any, tracer: Tracer) -> None:
    if manager is None:
        return

    try:
        manager.__exit__(None, None, None)
    except Exception as exc:  # noqa: BLE001
        tracer._degrade("span_end", exc)


def _clean(attributes: dict[str, Any]) -> dict[str, Any]:
    """Drop empty values so a span does not carry a wall of nulls."""
    return {key: value for key, value in attributes.items() if value is not None}


class NullTracer(Tracer):
    """Explicitly disabled tracer, used as the default everywhere."""

    def __init__(self) -> None:
        super().__init__(settings=None, client=None)


NULL_TRACER = NullTracer()

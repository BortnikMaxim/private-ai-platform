"""Request-scoped middleware.

``RequestContextMiddleware`` gives every request an id and publishes it in a
contextvar. That id is what makes the three observability layers line up: it
goes into the structured log lines, it seeds the Langfuse trace id, and it comes
back to the caller in ``X-Request-ID`` so a user-reported problem can be located
without guessing at timestamps.

The inference service has had this for a while; the backend had not.
"""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from backend.tracing import request_id_var

logger = logging.getLogger("backend.request")

REQUEST_ID_HEADER = "X-Request-ID"
MAX_INBOUND_ID_LENGTH = 128


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Conservative headers for an API-only service.

    No CSP: this backend serves JSON plus the Swagger page, and a strict policy
    would break the docs UI without protecting anything real.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")

        return response


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, log the outcome, echo the id back."""

    async def dispatch(self, request: Request, call_next):
        request_id = _inbound_id(request) or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        request.state.request_id = request_id

        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - started
            # The path is logged, never the query string or the body: both can
            # carry user content.
            logger.exception(
                "request_failed request_id=%s method=%s path=%s duration_ms=%.1f",
                request_id,
                request.method,
                request.url.path,
                duration * 1000,
            )
            raise
        finally:
            request_id_var.reset(token)

        duration = time.perf_counter() - started
        response.headers[REQUEST_ID_HEADER] = request_id

        logger.info(
            "request_completed request_id=%s method=%s path=%s status=%d "
            "duration_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration * 1000,
        )

        return response


def _inbound_id(request: Request) -> str | None:
    """Reuse a caller-supplied id, once it has been made safe to log.

    A client-controlled header ends up in log lines and in a trace id seed, so
    it is length-capped and restricted to characters that cannot forge a new
    log field or a line break.
    """
    candidate = request.headers.get(REQUEST_ID_HEADER)

    if not candidate:
        return None

    candidate = candidate.strip()[:MAX_INBOUND_ID_LENGTH]

    if not candidate or not all(
        char.isalnum() or char in "-_." for char in candidate
    ):
        return None

    return candidate

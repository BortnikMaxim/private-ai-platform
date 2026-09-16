"""Fixed-window rate limiting on Redis.

Redis is already part of the stack, and the whole policy is one INCR plus one
EXPIRE — a dedicated library would add a dependency for less control. The pair
runs inside a Lua script so the counter can never be left without a TTL if the
process dies between the two commands, which would otherwise wedge a caller out
permanently.

Keys hold a counter and nothing else: no email, no password, no token. A client
address is hashed before it becomes part of a key.
"""

import hashlib
import logging
import uuid

from redis.asyncio import Redis

from backend.errors import RateLimitExceededError
from backend.observability import RATE_LIMIT_REJECTIONS_TOTAL, auth_event

logger = logging.getLogger(__name__)

KEY_PREFIX = "ratelimit"

# INCR then, only on the first hit of a window, EXPIRE. Atomic as one script.
_INCREMENT_SCRIPT = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return current
"""


def hash_identifier(value: str) -> str:
    """Short, stable, non-reversible id for a client address."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


class RateLimiter:
    def __init__(
        self,
        redis: Redis | None,
        window_seconds: int = 60,
        enabled: bool = True,
    ) -> None:
        self.redis = redis
        self.window_seconds = window_seconds
        self.enabled = enabled

    def _key(self, route: str, identity: str) -> str:
        return f"{KEY_PREFIX}:{route}:{identity}"

    async def hit(self, route: str, identity: str, limit: int) -> int:
        """Count one request. Returns the new counter value.

        A Redis outage must not take the API down with it, so a failure here is
        logged and allowed through — availability wins over throttling for a
        self-hosted single-tenant deployment.
        """
        if not self.enabled or self.redis is None:
            return 0

        try:
            return int(
                await self.redis.eval(
                    _INCREMENT_SCRIPT,
                    1,
                    self._key(route, identity),
                    self.window_seconds,
                )
            )
        except Exception as exc:  # noqa: BLE001 - never fail closed on Redis
            logger.warning(
                "rate_limit_backend_unavailable route=%s error=%s",
                route,
                type(exc).__name__,
            )
            return 0

    async def enforce(self, route: str, identity: str, limit: int) -> None:
        """Raise :class:`RateLimitExceededError` once the window is exhausted."""
        current = await self.hit(route, identity, limit)

        if current > limit:
            RATE_LIMIT_REJECTIONS_TOTAL.labels(route=route).inc()
            auth_event(
                "rate_limit_exceeded",
                route=route,
                identity=identity,
                limit=limit,
            )
            raise RateLimitExceededError(
                f"Rate limit of {limit} requests per "
                f"{self.window_seconds} seconds exceeded",
                retry_after=self.window_seconds,
            )

    async def reset(self, route: str, identity: str) -> None:
        """Drop a counter. Used by tests and by administrative unblocking."""
        if self.redis is None:
            return

        try:
            await self.redis.delete(self._key(route, identity))
        except Exception:  # noqa: BLE001
            logger.warning("rate_limit_reset_failed route=%s", route)


def client_identity(user_id: uuid.UUID | None, client_host: str | None) -> str:
    """Prefer the authenticated user; fall back to a hashed address."""
    if user_id is not None:
        return f"user:{user_id}"

    return f"ip:{hash_identifier(client_host or 'unknown')}"

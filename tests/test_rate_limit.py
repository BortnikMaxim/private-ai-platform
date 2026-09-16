"""Redis-backed fixed-window rate limiting."""

import pytest

from backend.errors import RateLimitExceededError
from backend.services.rate_limiter import RateLimiter, client_identity, hash_identifier
from tests.conftest import DEFAULT_PASSWORD

GOOD_PASSWORD = "a-sufficiently-long-password"


# ---------------------------------------------------------------------------
# The limiter itself
# ---------------------------------------------------------------------------


async def test_requests_under_the_limit_pass(rate_limiter):
    for _ in range(5):
        await rate_limiter.enforce("chat", "user:alice", limit=5)


async def test_the_limit_is_enforced(rate_limiter):
    for _ in range(3):
        await rate_limiter.enforce("chat", "user:alice", limit=3)

    with pytest.raises(RateLimitExceededError) as error:
        await rate_limiter.enforce("chat", "user:alice", limit=3)

    assert error.value.status_code == 429
    assert error.value.retry_after == rate_limiter.window_seconds


async def test_counters_are_per_identity(rate_limiter):
    for _ in range(3):
        await rate_limiter.enforce("chat", "user:alice", limit=3)

    # Bob starts from zero even though Alice is exhausted.
    await rate_limiter.enforce("chat", "user:bob", limit=3)


async def test_counters_are_per_route(rate_limiter):
    for _ in range(3):
        await rate_limiter.enforce("chat", "user:alice", limit=3)

    await rate_limiter.enforce("upload", "user:alice", limit=3)


async def test_the_window_resets(rate_limiter, redis_client):
    for _ in range(3):
        await rate_limiter.enforce("chat", "user:alice", limit=3)

    with pytest.raises(RateLimitExceededError):
        await rate_limiter.enforce("chat", "user:alice", limit=3)

    # Simulate the TTL elapsing rather than sleeping through it.
    redis_client.expire_window()

    await rate_limiter.enforce("chat", "user:alice", limit=3)


async def test_the_first_hit_sets_a_ttl(rate_limiter, redis_client, settings):
    await rate_limiter.enforce("chat", "user:alice", limit=5)

    key = next(iter(redis_client.expiries))
    assert redis_client.expiries[key] == settings.rate_limit_window_seconds


async def test_a_redis_outage_does_not_block_the_api(rate_limiter, redis_client):
    redis_client.available = False

    # Availability wins over throttling for a self-hosted deployment.
    await rate_limiter.enforce("chat", "user:alice", limit=1)
    await rate_limiter.enforce("chat", "user:alice", limit=1)


async def test_disabling_the_limiter_is_a_no_op(redis_client):
    limiter = RateLimiter(redis=redis_client, enabled=False)

    for _ in range(50):
        await limiter.enforce("chat", "user:alice", limit=1)


async def test_reset_clears_a_counter(rate_limiter):
    for _ in range(3):
        await rate_limiter.enforce("chat", "user:alice", limit=3)

    await rate_limiter.reset("chat", "user:alice")

    await rate_limiter.enforce("chat", "user:alice", limit=3)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_an_authenticated_identity_uses_the_user_id():
    import uuid

    user_id = uuid.uuid4()

    assert client_identity(user_id, "203.0.113.4") == f"user:{user_id}"


def test_an_anonymous_identity_hashes_the_address():
    identity = client_identity(None, "203.0.113.4")

    assert identity.startswith("ip:")
    # The raw address never becomes part of a Redis key.
    assert "203.0.113.4" not in identity
    assert identity == f"ip:{hash_identifier('203.0.113.4')}"


def test_hashing_is_stable_and_not_reversible():
    assert hash_identifier("203.0.113.4") == hash_identifier("203.0.113.4")
    assert hash_identifier("203.0.113.4") != hash_identifier("203.0.113.5")
    assert len(hash_identifier("203.0.113.4")) == 16


async def test_no_key_ever_contains_an_email_or_password(rate_limiter, redis_client):
    await rate_limiter.enforce("auth_login", client_identity(None, "203.0.113.4"), 5)

    for key in redis_client.counters:
        assert "@" not in key
        assert GOOD_PASSWORD not in key


# ---------------------------------------------------------------------------
# Through the API
# ---------------------------------------------------------------------------


async def test_login_is_throttled(anonymous_client, settings, user):
    settings.rate_limit_auth_per_minute = 3

    payload = {"email": user.email, "password": DEFAULT_PASSWORD}

    for _ in range(3):
        assert (
            await anonymous_client.post("/auth/login", json=payload)
        ).status_code == 200

    response = await anonymous_client.post("/auth/login", json=payload)

    assert response.status_code == 429
    assert response.headers["Retry-After"] == str(settings.rate_limit_window_seconds)
    assert "traceback" not in response.text.lower()


async def test_register_is_throttled(anonymous_client, settings):
    settings.rate_limit_auth_per_minute = 2

    for index in range(2):
        await anonymous_client.post(
            "/auth/register",
            json={"email": f"user{index}@example.com", "password": GOOD_PASSWORD},
        )

    response = await anonymous_client.post(
        "/auth/register",
        json={"email": "blocked@example.com", "password": GOOD_PASSWORD},
    )

    assert response.status_code == 429


async def test_failed_logins_also_consume_the_budget(anonymous_client, settings, user):
    """Otherwise the limiter would not slow a password guessing loop down."""
    settings.rate_limit_auth_per_minute = 3

    for _ in range(3):
        assert (
            await anonymous_client.post(
                "/auth/login",
                json={"email": user.email, "password": "wrong-password-here"},
            )
        ).status_code == 401

    response = await anonymous_client.post(
        "/auth/login",
        json={"email": user.email, "password": DEFAULT_PASSWORD},
    )

    assert response.status_code == 429


async def test_chat_is_throttled_per_user(client, other_client, settings, inference):
    settings.rate_limit_chat_per_minute = 2

    conversation = (
        await client.post("/conversations", json={"title": "t"})
    ).json()["id"]
    other_conversation = (
        await other_client.post("/conversations", json={"title": "t"})
    ).json()["id"]

    for _ in range(2):
        assert (
            await client.post(
                f"/conversations/{conversation}/messages", json={"content": "привет"}
            )
        ).status_code == 201

    blocked = await client.post(
        f"/conversations/{conversation}/messages", json={"content": "привет"}
    )
    assert blocked.status_code == 429

    # Bob has his own budget.
    assert (
        await other_client.post(
            f"/conversations/{other_conversation}/messages",
            json={"content": "привет"},
        )
    ).status_code == 201


async def test_upload_is_throttled(client, settings):
    import io

    from pypdf import PdfWriter

    settings.rate_limit_upload_per_minute = 1

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    payload = buffer.getvalue()

    files = {"file": ("doc.pdf", payload, "application/pdf")}

    assert (await client.post("/documents", files=files)).status_code == 202
    assert (await client.post("/documents", files=files)).status_code == 429

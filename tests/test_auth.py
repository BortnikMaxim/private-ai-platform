"""Registration, login, tokens and the password primitives."""

import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from backend.errors import InvalidTokenError, WeakPasswordError
from backend.security.passwords import (
    hash_password,
    validate_password_policy,
    verify_password,
)
from backend.security.tokens import create_access_token, decode_access_token
from tests.conftest import DEFAULT_PASSWORD

GOOD_PASSWORD = "a-sufficiently-long-password"


# ---------------------------------------------------------------------------
# Password primitives
# ---------------------------------------------------------------------------


def test_hashing_is_salted_and_verifiable():
    first = hash_password(GOOD_PASSWORD)
    second = hash_password(GOOD_PASSWORD)

    assert first != second, "each hash must carry its own salt"
    assert first.startswith("$argon2id$")
    assert verify_password(GOOD_PASSWORD, first)
    assert verify_password(GOOD_PASSWORD, second)


def test_hash_never_contains_the_password():
    digest = hash_password(GOOD_PASSWORD)

    assert GOOD_PASSWORD not in digest


def test_wrong_password_does_not_verify():
    assert verify_password("wrong-password-entirely", hash_password(GOOD_PASSWORD)) is False


@pytest.mark.parametrize("bad_hash", ["", "not-a-hash", "!locked-no-login", "$argon2id$x"])
def test_a_malformed_or_locked_hash_never_verifies(bad_hash):
    """Migration 0003 writes a locked hash; nothing may authenticate against it."""
    assert verify_password(GOOD_PASSWORD, bad_hash) is False
    assert verify_password("", bad_hash) is False


def test_password_policy_rejects_short_and_long():
    with pytest.raises(WeakPasswordError, match="at least"):
        validate_password_policy("short", min_length=10, max_length=128)

    with pytest.raises(WeakPasswordError, match="at most"):
        validate_password_policy("x" * 200, min_length=10, max_length=128)

    validate_password_policy(GOOD_PASSWORD, min_length=10, max_length=128)


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def test_token_round_trip(settings):
    user_id = uuid.uuid4()
    token, expires_in = create_access_token(settings, user_id, "admin")

    claims = decode_access_token(settings, token)

    assert claims.user_id == user_id
    assert claims.role == "admin"
    assert expires_in == settings.jwt_access_token_expire_minutes * 60


def test_token_payload_carries_no_pii(settings):
    token, _ = create_access_token(settings, uuid.uuid4(), "user")
    payload = jwt.decode(token, options={"verify_signature": False})

    assert set(payload) == {"sub", "role", "iat", "exp", "iss"}
    assert "email" not in payload


@pytest.mark.parametrize(
    "token",
    ["", "not.a.token", "a.b.c", "Bearer something"],
)
def test_malformed_tokens_are_rejected(settings, token):
    with pytest.raises(InvalidTokenError):
        decode_access_token(settings, token)


def test_a_token_signed_with_another_key_is_rejected(settings):
    forged = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "role": "admin",
            "iat": int(datetime.now(UTC).timestamp()),
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            "iss": settings.jwt_issuer,
        },
        "a-completely-different-signing-key-value",
        algorithm="HS256",
    )

    with pytest.raises(InvalidTokenError):
        decode_access_token(settings, forged)


def test_an_expired_token_is_rejected(settings):
    past = datetime.now(UTC) - timedelta(hours=2)
    expired = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "role": "user",
            "iat": int(past.timestamp()),
            "exp": int((past + timedelta(minutes=1)).timestamp()),
            "iss": settings.jwt_issuer,
        },
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(InvalidTokenError, match="expired"):
        decode_access_token(settings, expired)


def test_an_alg_none_token_is_rejected(settings):
    """The classic JWT downgrade attack must not work."""
    unsigned = jwt.encode(
        {"sub": str(uuid.uuid4()), "role": "admin", "iat": 0, "exp": 9999999999},
        key="",
        algorithm="none",
    )

    with pytest.raises(InvalidTokenError):
        decode_access_token(settings, unsigned)


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------


async def test_register_returns_a_safe_user(anonymous_client):
    response = await anonymous_client.post(
        "/auth/register",
        json={"email": "new@example.com", "password": GOOD_PASSWORD},
    )
    body = response.json()

    assert response.status_code == 201
    assert body["email"] == "new@example.com"
    assert body["role"] == "user"
    assert body["is_active"] is True
    # Nothing about the credential may leave the service.
    assert "password" not in body
    assert "password_hash" not in body
    assert GOOD_PASSWORD not in response.text


async def test_register_normalises_the_email(anonymous_client):
    response = await anonymous_client.post(
        "/auth/register",
        json={"email": "MiXeD@Example.COM", "password": GOOD_PASSWORD},
    )

    assert response.json()["email"] == "mixed@example.com"


async def test_duplicate_email_is_409(anonymous_client):
    payload = {"email": "dup@example.com", "password": GOOD_PASSWORD}

    assert (await anonymous_client.post("/auth/register", json=payload)).status_code == 201

    response = await anonymous_client.post("/auth/register", json=payload)

    assert response.status_code == 409
    assert response.json()["detail"] == "Email is already registered"


async def test_duplicate_is_case_insensitive(anonymous_client):
    await anonymous_client.post(
        "/auth/register",
        json={"email": "case@example.com", "password": GOOD_PASSWORD},
    )

    response = await anonymous_client.post(
        "/auth/register",
        json={"email": "CASE@example.com", "password": GOOD_PASSWORD},
    )

    assert response.status_code == 409


async def test_a_weak_password_is_rejected(anonymous_client, settings):
    response = await anonymous_client.post(
        "/auth/register",
        json={"email": "weak@example.com", "password": "short"},
    )

    assert response.status_code == 422
    assert str(settings.password_min_length) in response.json()["detail"]


async def test_a_malformed_email_is_422(anonymous_client):
    response = await anonymous_client.post(
        "/auth/register",
        json={"email": "not-an-email", "password": GOOD_PASSWORD},
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


async def test_login_returns_a_usable_bearer_token(anonymous_client, user, settings):
    response = await anonymous_client.post(
        "/auth/login",
        json={"email": user.email, "password": DEFAULT_PASSWORD},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0

    claims = decode_access_token(settings, body["access_token"])
    assert claims.user_id == user.id


async def test_login_is_case_insensitive(anonymous_client, user):
    response = await anonymous_client.post(
        "/auth/login",
        json={"email": user.email.upper(), "password": DEFAULT_PASSWORD},
    )

    assert response.status_code == 200


async def test_wrong_password_is_401(anonymous_client, user):
    response = await anonymous_client.post(
        "/auth/login",
        json={"email": user.email, "password": "definitely-not-the-password"},
    )

    assert response.status_code == 401
    assert response.headers.get("WWW-Authenticate") == "Bearer"


async def test_login_does_not_reveal_whether_the_account_exists(anonymous_client, user):
    """The two failures must be indistinguishable to a client."""
    wrong_password = await anonymous_client.post(
        "/auth/login",
        json={"email": user.email, "password": "definitely-not-the-password"},
    )
    unknown_account = await anonymous_client.post(
        "/auth/login",
        json={"email": "nobody@example.com", "password": "definitely-not-the-password"},
    )

    assert wrong_password.status_code == unknown_account.status_code == 401
    assert wrong_password.json() == unknown_account.json()


async def test_inactive_user_cannot_log_in(anonymous_client, make_user):
    disabled = await make_user(email="disabled@example.com", is_active=False)

    response = await anonymous_client.post(
        "/auth/login",
        json={"email": disabled.email, "password": DEFAULT_PASSWORD},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "User account is inactive"


# ---------------------------------------------------------------------------
# /auth/me and the bearer dependency
# ---------------------------------------------------------------------------


async def test_me_returns_the_current_user(client, user):
    response = await client.get("/auth/me")
    body = response.json()

    assert response.status_code == 200
    assert body["id"] == str(user.id)
    assert body["email"] == user.email
    assert "password_hash" not in body


async def test_me_without_a_token_is_401(anonymous_client):
    response = await anonymous_client.get("/auth/me")

    assert response.status_code == 401
    assert response.headers.get("WWW-Authenticate") == "Bearer"


async def test_me_with_a_malformed_token_is_401(make_client, app):
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer not-a-real-token"},
    ) as broken:
        response = await broken.get("/auth/me")

    assert response.status_code == 401


async def test_me_with_an_expired_token_is_401(app, settings, user):
    from httpx import ASGITransport, AsyncClient

    past = datetime.now(UTC) - timedelta(hours=2)
    expired = jwt.encode(
        {
            "sub": str(user.id),
            "role": user.role,
            "iat": int(past.timestamp()),
            "exp": int((past + timedelta(minutes=1)).timestamp()),
            "iss": settings.jwt_issuer,
        },
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {expired}"},
    ) as stale:
        response = await stale.get("/auth/me")

    assert response.status_code == 401


async def test_a_token_for_a_deleted_account_is_401(app, session_factory, make_user, token_for):
    from httpx import ASGITransport, AsyncClient

    from backend.models import User

    ghost = await make_user(email="ghost@example.com")
    token = token_for(ghost)

    async with session_factory() as session:
        await session.delete(await session.get(User, ghost.id))
        await session.commit()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    ) as orphaned:
        response = await orphaned.get("/auth/me")

    assert response.status_code == 401


async def test_a_deactivated_user_is_403_on_protected_routes(
    app,
    session_factory,
    make_user,
    token_for,
):
    from httpx import ASGITransport, AsyncClient

    from backend.models import User

    victim = await make_user(email="revoked@example.com")
    token = token_for(victim)

    async with session_factory() as session:
        row = await session.get(User, victim.id)
        row.is_active = False
        await session.commit()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    ) as revoked:
        response = await revoked.get("/auth/me")

    # The token is still cryptographically valid; the account is not.
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Everything else needs a token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/conversations"),
        ("POST", "/conversations"),
        ("GET", f"/conversations/{uuid.uuid4()}"),
        ("DELETE", f"/conversations/{uuid.uuid4()}"),
        ("POST", f"/conversations/{uuid.uuid4()}/messages"),
        ("POST", f"/conversations/{uuid.uuid4()}/agent"),
        ("GET", "/documents"),
        ("POST", "/documents"),
        ("GET", f"/documents/{uuid.uuid4()}"),
        ("DELETE", f"/documents/{uuid.uuid4()}"),
        ("POST", "/rag/ask"),
        ("POST", "/rag/retrieve"),
        ("GET", "/admin/users"),
    ],
)
async def test_protected_routes_require_a_token(anonymous_client, method, path):
    response = await anonymous_client.request(method, path, json={})

    assert response.status_code == 401, f"{method} {path} was reachable anonymously"


@pytest.mark.parametrize("path", ["/health", "/health/workers", "/metrics"])
async def test_operational_routes_stay_public(anonymous_client, path):
    response = await anonymous_client.get(path)

    assert response.status_code == 200

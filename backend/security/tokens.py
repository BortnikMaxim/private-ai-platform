"""JWT issuing and verification.

PyJWT does the crypto and the validation, including expiry. Nothing here
implements a primitive by hand.

The payload carries an identifier and a role and nothing else — no email, no
name. A leaked token should reveal as little as possible about its owner.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt

from backend.config import Settings
from backend.errors import InvalidTokenError

logger = logging.getLogger(__name__)

TOKEN_TYPE = "bearer"


@dataclass(slots=True)
class TokenClaims:
    user_id: uuid.UUID
    role: str
    expires_at: datetime


def create_access_token(
    settings: Settings,
    user_id: uuid.UUID,
    role: str,
) -> tuple[str, int]:
    """Return ``(token, expires_in_seconds)``."""
    now = datetime.now(UTC)
    expires_delta = timedelta(minutes=settings.jwt_access_token_expire_minutes)
    expires_at = now + expires_delta

    payload = {
        "sub": str(user_id),
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "iss": settings.jwt_issuer,
    }

    token = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    return token, int(expires_delta.total_seconds())


def decode_access_token(settings: Settings, token: str) -> TokenClaims:
    """Verify a token and return its claims, or raise :class:`InvalidTokenError`.

    ``algorithms`` is pinned to the configured algorithm so a token cannot talk
    the verifier into ``none`` or into an asymmetric/symmetric confusion.
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise InvalidTokenError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        # Covers bad signature, wrong issuer, malformed payload, missing claims.
        # The reason is deliberately not echoed back to the client.
        logger.info("token_rejected reason=%s", type(exc).__name__)
        raise InvalidTokenError() from exc

    try:
        user_id = uuid.UUID(str(payload["sub"]))
    except (KeyError, ValueError, TypeError) as exc:
        raise InvalidTokenError() from exc

    role = payload.get("role") or "user"
    expires_at = datetime.fromtimestamp(payload["exp"], tz=UTC)

    return TokenClaims(user_id=user_id, role=str(role), expires_at=expires_at)

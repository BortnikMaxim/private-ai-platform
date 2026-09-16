"""Registration, login and self-inspection."""

import logging

from fastapi import APIRouter, Request, status

from backend.dependencies import (
    AuthServiceDep,
    CurrentUser,
    DbSession,
    RateLimiterDep,
    SettingsDep,
)
from backend.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserRead,
)
from backend.security.tokens import TOKEN_TYPE, create_access_token
from backend.services.rate_limiter import client_identity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


async def _throttle(
    request: Request,
    limiter: RateLimiterDep,
    settings: SettingsDep,
    route: str,
) -> None:
    """Unauthenticated routes are limited per client address, never per email.

    Keying on the submitted email would let an attacker lock a victim out of
    their own account, and would put an address into Redis.
    """
    identity = client_identity(None, request.client.host if request.client else None)
    await limiter.enforce(route, identity, settings.rate_limit_auth_per_minute)


@router.post(
    "/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    request: Request,
    payload: RegisterRequest,
    session: DbSession,
    auth: AuthServiceDep,
    limiter: RateLimiterDep,
    settings: SettingsDep,
) -> UserRead:
    """Create an account. Returns the user, never a password hash."""
    await _throttle(request, limiter, settings, "auth_register")

    user = await auth.register(session, email=payload.email, password=payload.password)

    return UserRead.model_validate(user)


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    payload: LoginRequest,
    session: DbSession,
    auth: AuthServiceDep,
    limiter: RateLimiterDep,
    settings: SettingsDep,
) -> TokenResponse:
    """Exchange credentials for a bearer token.

    A wrong password and an unknown address produce the identical 401, so the
    endpoint cannot be used to enumerate registered addresses.
    """
    await _throttle(request, limiter, settings, "auth_login")

    user = await auth.authenticate(
        session,
        email=payload.email,
        password=payload.password,
    )

    token, expires_in = create_access_token(settings, user.id, user.role)

    return TokenResponse(
        access_token=token,
        token_type=TOKEN_TYPE,
        expires_in=expires_in,
    )


@router.get("/me", response_model=UserRead)
async def me(user: CurrentUser) -> UserRead:
    return UserRead.model_validate(user)

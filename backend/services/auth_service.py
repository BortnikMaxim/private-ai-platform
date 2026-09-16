"""Registration, authentication and user lookup."""

import logging
import uuid

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Settings
from backend.errors import (
    EmailAlreadyRegisteredError,
    InactiveUserError,
    InvalidCredentialsError,
)
from backend.models import ROLE_USER, User
from backend.observability import AUTH_LOGINS_TOTAL, AUTH_REGISTRATIONS_TOTAL, auth_event
from backend.security.passwords import (
    hash_password,
    validate_password_policy,
    verify_password,
)

logger = logging.getLogger(__name__)


def normalize_email(email: str) -> str:
    return email.strip().lower()


class AuthService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # -- registration ------------------------------------------------------

    async def register(
        self,
        session: AsyncSession,
        email: str,
        password: str,
        role: str = ROLE_USER,
    ) -> User:
        address = normalize_email(email)

        # Policy first: a rejected password must never reach the hasher.
        validate_password_policy(
            password,
            min_length=self.settings.password_min_length,
            max_length=self.settings.password_max_length,
        )

        existing = await session.scalar(select(User.id).where(User.email == address))

        if existing is not None:
            AUTH_REGISTRATIONS_TOTAL.labels(status="duplicate").inc()
            raise EmailAlreadyRegisteredError()

        user = User(
            email=address,
            password_hash=hash_password(password),
            is_active=True,
            role=role,
        )
        session.add(user)

        try:
            await session.commit()
        except IntegrityError as exc:
            # Two concurrent registrations for the same address; the unique
            # index is the real arbiter.
            await session.rollback()
            AUTH_REGISTRATIONS_TOTAL.labels(status="duplicate").inc()
            raise EmailAlreadyRegisteredError() from exc

        await session.refresh(user)

        AUTH_REGISTRATIONS_TOTAL.labels(status="success").inc()
        auth_event("auth_register_success", user_id=user.id, role=user.role)

        return user

    # -- authentication ----------------------------------------------------

    async def authenticate(
        self,
        session: AsyncSession,
        email: str,
        password: str,
    ) -> User:
        """Return the user, or raise. Failure never says which half was wrong."""
        address = normalize_email(email)
        user = await session.scalar(select(User).where(User.email == address))

        if user is None:
            # Hash anyway so a missing account and a wrong password take
            # comparable time; otherwise login becomes an account oracle.
            verify_password(password, _DUMMY_HASH)
            AUTH_LOGINS_TOTAL.labels(status="invalid_credentials").inc()
            auth_event("auth_login_failed", reason="unknown_account")
            raise InvalidCredentialsError()

        if not verify_password(password, user.password_hash):
            AUTH_LOGINS_TOTAL.labels(status="invalid_credentials").inc()
            auth_event("auth_login_failed", user_id=user.id, reason="bad_password")
            raise InvalidCredentialsError()

        if not user.is_active:
            AUTH_LOGINS_TOTAL.labels(status="inactive").inc()
            auth_event("auth_login_failed", user_id=user.id, reason="inactive")
            raise InactiveUserError()

        AUTH_LOGINS_TOTAL.labels(status="success").inc()
        auth_event("auth_login_success", user_id=user.id, role=user.role)

        return user

    # -- lookup ------------------------------------------------------------

    async def get_by_id(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
    ) -> User | None:
        return await session.get(User, user_id)

    async def list_users(
        self,
        session: AsyncSession,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[User], int]:
        total = await session.scalar(select(func.count()).select_from(User)) or 0

        result = await session.execute(
            select(User).order_by(User.created_at.desc()).limit(limit).offset(offset)
        )

        return list(result.scalars().all()), total

    async def set_active(
        self,
        session: AsyncSession,
        user: User,
        is_active: bool,
    ) -> User:
        user.is_active = is_active
        session.add(user)
        await session.commit()
        await session.refresh(user)

        auth_event("auth_user_active_changed", user_id=user.id, is_active=is_active)

        return user


# A real Argon2 digest of a random string, used only to burn comparable CPU on
# the unknown-account path. It can never match a user-supplied password.
_DUMMY_HASH = hash_password(uuid.uuid4().hex + uuid.uuid4().hex)

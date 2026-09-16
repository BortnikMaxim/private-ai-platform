"""Create (or promote) an administrator account.

    python -m backend.scripts.create_admin --email admin@example.com

The password is read with :func:`getpass.getpass`, never from an argument, so
it does not land in the shell history, the process list or a CI log. Nothing
about the password — not its length, not a hash, not a token — is printed.
"""

import argparse
import asyncio
import getpass
import os
import sys

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from backend.config import get_settings
from backend.db import create_engine
from backend.errors import WeakPasswordError
from backend.models import ROLE_ADMIN, User
from backend.security.passwords import hash_password, validate_password_policy


def read_password(settings) -> str:
    """Prompt twice and confirm. Falls back to ADMIN_PASSWORD for automation."""
    from_env = os.getenv("ADMIN_PASSWORD")

    if from_env:
        print("Using ADMIN_PASSWORD from the environment.")
        return from_env

    if not sys.stdin.isatty():
        raise SystemExit(
            "No TTY available for a password prompt. "
            "Set ADMIN_PASSWORD in the environment instead."
        )

    password = getpass.getpass("Password: ")
    confirmation = getpass.getpass("Confirm password: ")

    if password != confirmation:
        raise SystemExit("Passwords do not match.")

    try:
        validate_password_policy(
            password,
            min_length=settings.password_min_length,
            max_length=settings.password_max_length,
        )
    except WeakPasswordError as exc:
        raise SystemExit(str(exc)) from exc

    return password


async def run(email: str, promote: bool) -> int:
    settings = get_settings()
    address = email.strip().lower()

    engine = create_engine(settings.database_url)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    try:
        async with session_factory() as session:
            existing = await session.scalar(
                select(User).where(User.email == address)
            )

            if existing is not None:
                if existing.role == ROLE_ADMIN and existing.is_active:
                    print(f"{address} is already an active administrator.")
                    return 0

                if not promote:
                    print(
                        f"{address} already exists "
                        f"(role={existing.role}, active={existing.is_active}). "
                        "Re-run with --promote to grant admin and reactivate."
                    )
                    return 1

                existing.role = ROLE_ADMIN
                existing.is_active = True
                session.add(existing)
                await session.commit()

                print(f"Promoted {address} to administrator.")
                return 0

            password = read_password(settings)

            user = User(
                email=address,
                password_hash=hash_password(password),
                is_active=True,
                role=ROLE_ADMIN,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)

            # The id is safe to print; nothing else about the account is.
            print(f"Created administrator {address} (id={user.id}).")
            return 0

    except Exception as exc:  # noqa: BLE001 - a CLI reports, it does not dump
        print(f"Failed to create the administrator: {type(exc).__name__}: {exc}")
        return 2
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.scripts.create_admin",
        description="Create an administrator account.",
    )
    parser.add_argument(
        "--email",
        default=os.getenv("ADMIN_EMAIL"),
        help="administrator email (or set ADMIN_EMAIL)",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="if the account already exists, grant admin and reactivate it",
    )

    args = parser.parse_args()

    if not args.email:
        parser.error("--email is required (or set ADMIN_EMAIL)")

    return asyncio.run(run(args.email, args.promote))


if __name__ == "__main__":
    sys.exit(main())

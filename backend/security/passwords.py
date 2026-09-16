"""Password hashing and policy.

Argon2id via ``argon2-cffi`` — the current recommendation, memory-hard, and
with per-hash salts and parameters embedded in the digest, so rotating cost
parameters later does not invalidate existing hashes.

Nothing in this module logs, returns or raises with the password in it.
"""

import logging

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from backend.errors import WeakPasswordError

logger = logging.getLogger(__name__)

# argon2-cffi's defaults track the RFC 9106 recommendations; they are not
# restated here so that a library upgrade improves the cost automatically.
_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time-ish verification that never raises on bad input.

    A malformed or deliberately locked hash (see migration 0003) simply fails,
    which is what an account with no usable password should do.
    """
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False
    except Exception:
        logger.exception("password_verification_error")
        return False


def needs_rehash(password_hash: str) -> bool:
    try:
        return _hasher.check_needs_rehash(password_hash)
    except (InvalidHashError, ValueError):
        return False


def validate_password_policy(
    password: str,
    min_length: int,
    max_length: int,
) -> None:
    """Raise :class:`WeakPasswordError` if the password is unacceptable.

    Length only: complexity rules push people towards predictable
    substitutions, and NIST has recommended against them for years.
    """
    if len(password) < min_length:
        raise WeakPasswordError(
            f"Password must be at least {min_length} characters long"
        )

    if len(password) > max_length:
        raise WeakPasswordError(
            f"Password must be at most {max_length} characters long"
        )

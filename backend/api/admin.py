"""Administrative endpoints.

Deliberately a separate surface rather than a privileged mode of the normal
routes: an admin browsing ``/documents`` still sees only their own documents,
so an accidental bug in a user endpoint cannot silently become a tenant
bypass. Anything cross-tenant has to be an explicit ``/admin`` call.
"""

import uuid

from fastapi import APIRouter, Query

from backend.dependencies import AdminUser, AuthServiceDep, DbSession
from backend.errors import NotFoundError
from backend.schemas import SetActiveRequest, UserListResponse, UserRead

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users", response_model=UserListResponse)
async def list_users(
    admin: AdminUser,
    session: DbSession,
    auth: AuthServiceDep,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> UserListResponse:
    items, total = await auth.list_users(session, limit=limit, offset=offset)

    return UserListResponse(
        items=[UserRead.model_validate(item) for item in items],
        total=total,
    )


@router.patch("/users/{user_id}/active", response_model=UserRead)
async def set_user_active(
    user_id: uuid.UUID,
    payload: SetActiveRequest,
    admin: AdminUser,
    session: DbSession,
    auth: AuthServiceDep,
) -> UserRead:
    """Enable or disable an account. A disabled user's tokens stop working."""
    user = await auth.get_by_id(session, user_id)

    if user is None:
        raise NotFoundError("User not found")

    updated = await auth.set_active(session, user, payload.is_active)

    return UserRead.model_validate(updated)

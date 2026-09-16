"""Admin surface and the role check behind it."""

import uuid

from sqlalchemy import select

from backend.models import ROLE_ADMIN, User


async def test_a_regular_user_cannot_list_users(client):
    response = await client.get("/admin/users")

    assert response.status_code == 403
    assert response.json()["detail"] == "Administrator privileges are required"


async def test_an_admin_can_list_users(admin_client, user, other_user, admin_user):
    response = await admin_client.get("/admin/users")
    body = response.json()

    assert response.status_code == 200
    assert body["total"] >= 3

    emails = {item["email"] for item in body["items"]}
    assert {user.email, other_user.email, admin_user.email} <= emails

    # Even here, no hash is ever serialised.
    assert "password_hash" not in response.text


async def test_an_anonymous_caller_gets_401_not_403(anonymous_client):
    response = await anonymous_client.get("/admin/users")

    assert response.status_code == 401


async def test_an_admin_can_deactivate_a_user(admin_client, other_user, session_factory):
    response = await admin_client.patch(
        f"/admin/users/{other_user.id}/active",
        json={"is_active": False},
    )

    assert response.status_code == 200
    assert response.json()["is_active"] is False

    async with session_factory() as session:
        row = await session.get(User, other_user.id)
        assert row.is_active is False


async def test_a_deactivated_user_loses_access(
    admin_client,
    other_client,
    other_user,
):
    assert (await other_client.get("/auth/me")).status_code == 200

    await admin_client.patch(
        f"/admin/users/{other_user.id}/active", json={"is_active": False}
    )

    # The token is unchanged and still valid; the account is not.
    assert (await other_client.get("/auth/me")).status_code == 403


async def test_reactivation_restores_access(admin_client, other_client, other_user):
    await admin_client.patch(
        f"/admin/users/{other_user.id}/active", json={"is_active": False}
    )
    await admin_client.patch(
        f"/admin/users/{other_user.id}/active", json={"is_active": True}
    )

    assert (await other_client.get("/auth/me")).status_code == 200


async def test_a_regular_user_cannot_deactivate_anybody(client, other_user):
    response = await client.patch(
        f"/admin/users/{other_user.id}/active",
        json={"is_active": False},
    )

    assert response.status_code == 403


async def test_patching_an_unknown_user_is_404(admin_client):
    response = await admin_client.patch(
        f"/admin/users/{uuid.uuid4()}/active",
        json={"is_active": False},
    )

    assert response.status_code == 404


async def test_admin_does_not_bypass_ownership_on_normal_routes(
    admin_client,
    client,
    seeded_document,
    make_document,
    admin_user,
):
    """Admin is a separate surface, not a master key on the user endpoints."""
    await make_document(admin_user, filename="admin-own.pdf")

    listing = (await admin_client.get("/documents")).json()

    assert {item["filename"] for item in listing["items"]} == {"admin-own.pdf"}
    assert (await admin_client.get(f"/documents/{seeded_document}")).status_code == 404


async def test_registration_never_yields_an_admin(anonymous_client, session_factory):
    await anonymous_client.post(
        "/auth/register",
        json={"email": "wannabe@example.com", "password": "a-long-enough-password"},
    )

    async with session_factory() as session:
        created = await session.scalar(
            select(User).where(User.email == "wannabe@example.com")
        )

    assert created.role != ROLE_ADMIN
    assert created.role == "user"


async def test_listing_users_survives_a_reserved_address(
    admin_client,
    session_factory,
):
    """Migration 0003 creates system@local.invalid, which EmailStr rejects.

    UserRead must not re-validate stored addresses, or listing users 500s on a
    database that has been through the ownership backfill.
    """
    async with session_factory() as session:
        session.add(
            User(
                email="system@local.invalid",
                password_hash="!locked-no-login",
                is_active=False,
                role="user",
            )
        )
        await session.commit()

    response = await admin_client.get("/admin/users")

    assert response.status_code == 200
    assert "system@local.invalid" in {
        item["email"] for item in response.json()["items"]
    }

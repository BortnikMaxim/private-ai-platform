"""authentication fields and non-nullable ownership

Revision ID: 0003_auth
Revises: 0002_task_id
Create Date: 2026-09-07

Backfill strategy
-----------------
Rows created before authentication existed have no owner, and the target
schema forbids that. Rather than delete them (silent data loss) or leave the
column nullable (a row nobody can ever reach), the migration adopts them:

1. every new column is added nullable,
2. a single locked "system" account is inserted — ``is_active = false`` and a
   password hash that is not a valid Argon2 digest, so no password can ever
   verify against it and nobody can log in as it,
3. orphaned conversations and documents are assigned to that account,
4. only then are the columns tightened to NOT NULL.

The data survives, stays inspectable by an admin, and is unreachable by any
authenticated user. Reassign it with a plain UPDATE if you want it back:

    UPDATE conversations SET user_id = '<real-user-id>'
     WHERE user_id = (SELECT id FROM users WHERE email = 'system@local.invalid');
"""

import sqlalchemy as sa
from alembic import op

revision: str = "0003_auth"
down_revision: str | None = "0002_task_id"
branch_labels: str | None = None
depends_on: str | None = None

SYSTEM_EMAIL = "system@local.invalid"
# Deliberately not a parseable Argon2 hash: verification always fails.
LOCKED_PASSWORD_HASH = "!locked-no-login"


def upgrade() -> None:
    # -- users: authentication columns ----------------------------------
    op.add_column("users", sa.Column("password_hash", sa.String(255), nullable=True))
    op.add_column("users", sa.Column("is_active", sa.Boolean(), nullable=True))
    op.add_column("users", sa.Column("role", sa.String(20), nullable=True))
    op.add_column(
        "users",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Any pre-existing user row is locked out until an admin resets it.
    op.execute(
        sa.text(
            "UPDATE users SET password_hash = :hash, is_active = false, "
            "role = 'user', updated_at = now() "
            "WHERE password_hash IS NULL"
        ).bindparams(hash=LOCKED_PASSWORD_HASH)
    )

    op.alter_column("users", "password_hash", nullable=False)
    op.alter_column(
        "users", "is_active", nullable=False, server_default=sa.true()
    )
    op.alter_column("users", "role", nullable=False, server_default="user")
    op.alter_column("users", "updated_at", nullable=False, server_default=sa.func.now())

    op.create_index("ix_users_role", "users", ["role"], unique=False)
    op.create_check_constraint(
        "ck_users_role", "users", "role IN ('user', 'admin')"
    )

    # -- the adoption account -------------------------------------------
    op.execute(
        sa.text(
            "INSERT INTO users (id, email, password_hash, is_active, role, "
            "created_at, updated_at) "
            "SELECT gen_random_uuid(), :email, :hash, false, 'user', now(), now() "
            "WHERE NOT EXISTS (SELECT 1 FROM users WHERE email = :email)"
        ).bindparams(email=SYSTEM_EMAIL, hash=LOCKED_PASSWORD_HASH)
    )

    # -- documents: add ownership ---------------------------------------
    op.add_column("documents", sa.Column("user_id", sa.Uuid(), nullable=True))
    op.execute(
        sa.text(
            "UPDATE documents SET user_id = "
            "(SELECT id FROM users WHERE email = :email) "
            "WHERE user_id IS NULL"
        ).bindparams(email=SYSTEM_EMAIL)
    )
    op.alter_column("documents", "user_id", nullable=False)
    op.create_foreign_key(
        "fk_documents_user_id_users",
        "documents",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_documents_user_id", "documents", ["user_id"], unique=False)
    op.create_index(
        "ix_documents_user_created", "documents", ["user_id", "created_at"]
    )

    # -- conversations: tighten ownership --------------------------------
    op.execute(
        sa.text(
            "UPDATE conversations SET user_id = "
            "(SELECT id FROM users WHERE email = :email) "
            "WHERE user_id IS NULL"
        ).bindparams(email=SYSTEM_EMAIL)
    )
    op.alter_column("conversations", "user_id", nullable=False)

    # ON DELETE SET NULL is no longer valid for a NOT NULL column.
    op.drop_constraint(
        "conversations_user_id_fkey", "conversations", type_="foreignkey"
    )
    op.create_foreign_key(
        "fk_conversations_user_id_users",
        "conversations",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_conversations_user_updated", "conversations", ["user_id", "updated_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_conversations_user_updated", table_name="conversations")
    op.drop_constraint(
        "fk_conversations_user_id_users", "conversations", type_="foreignkey"
    )
    op.create_foreign_key(
        "conversations_user_id_fkey",
        "conversations",
        "users",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.alter_column("conversations", "user_id", nullable=True)

    op.drop_index("ix_documents_user_created", table_name="documents")
    op.drop_index("ix_documents_user_id", table_name="documents")
    op.drop_constraint("fk_documents_user_id_users", "documents", type_="foreignkey")
    op.drop_column("documents", "user_id")

    op.drop_constraint("ck_users_role", "users", type_="check")
    op.drop_index("ix_users_role", table_name="users")
    op.drop_column("users", "updated_at")
    op.drop_column("users", "role")
    op.drop_column("users", "is_active")
    op.drop_column("users", "password_hash")

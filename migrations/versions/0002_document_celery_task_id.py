"""add documents.celery_task_id for async ingestion

Revision ID: 0002_task_id
Revises: 0001_initial
Create Date: 2026-09-07

"""

import sqlalchemy as sa
from alembic import op

revision: str = "0002_task_id"
down_revision: str | None = "0001_initial"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("celery_task_id", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_documents_celery_task_id",
        "documents",
        ["celery_task_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_documents_celery_task_id", table_name="documents")
    op.drop_column("documents", "celery_task_id")

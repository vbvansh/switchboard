"""Initial schema: users and the request ledger.

Revision ID: 0001
Revises:
Created: 2026-08-13

This is the starting point. Databases created before migrations existed have
the same shape, so `switchboard db stamp-baseline` marks them as already at
this revision rather than trying to recreate their tables.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("api_key_hash", sa.String(length=64), nullable=False),
        sa.Column("monthly_budget_usd", sa.Float(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_name", "users", ["name"], unique=True)
    op.create_index(
        "ix_users_api_key_hash", "users", ["api_key_hash"], unique=True
    )

    op.create_table(
        "requests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("requested_model", sa.String(length=128), nullable=False),
        sa.Column("served_model", sa.String(length=128), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("tokens_estimated", sa.Boolean(), nullable=False),
        sa.Column("simulated_cost_usd", sa.Float(), nullable=False),
        sa.Column("baseline_cost_usd", sa.Float(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("caused_model_switch", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("prompt_json", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_requests_user_id", "requests", ["user_id"])
    op.create_index("ix_requests_created_at", "requests", ["created_at"])
    op.create_index("ix_requests_status", "requests", ["status"])


def downgrade() -> None:
    op.drop_index("ix_requests_status", table_name="requests")
    op.drop_index("ix_requests_created_at", table_name="requests")
    op.drop_index("ix_requests_user_id", table_name="requests")
    op.drop_table("requests")

    op.drop_index("ix_users_api_key_hash", table_name="users")
    op.drop_index("ix_users_name", table_name="users")
    op.drop_table("users")

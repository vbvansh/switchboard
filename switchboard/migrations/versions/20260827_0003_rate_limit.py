"""Per-user rate limit.

Revision ID: 0003
Revises: 0002
Created: 2026-08-27

Nullable on purpose: NULL means "use the server default", so raising the
default lifts everyone who has not been given a specific limit. Backfilling
the current default into every row would freeze them at today's value.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users", sa.Column("requests_per_minute", sa.Integer(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("users", "requests_per_minute")

"""Record why the router chose a model.

Revision ID: 0002
Revises: 0001
Created: 2026-08-24

Nullable on purpose: every row written before routing existed has no reason,
and backfilling a guess would be worse than an honest blank.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("requests", sa.Column("routing_reason", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("requests", "routing_reason")

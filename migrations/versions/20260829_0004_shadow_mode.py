"""Record what routing would have done.

Revision ID: 0004
Revises: 0003
Created: 2026-08-29

Both columns nullable. A request served before shadow mode existed carries no
opinion, and inventing one - even "same as served" - would make every report
that reads these columns wrong in a way nobody could see.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "requests", sa.Column("shadow_model", sa.String(length=128), nullable=True)
    )
    op.add_column("requests", sa.Column("shadow_cost_usd", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("requests", "shadow_cost_usd")
    op.drop_column("requests", "shadow_model")

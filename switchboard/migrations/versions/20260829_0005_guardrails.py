"""Record what the usage policy thought of each request.

Revision ID: 0005
Revises: 0004
Created: 2026-08-29

Three nullable columns, and nullable is the meaningful part: a request served
while the policy was off has no opinion attached, and writing "allowed" for it
would make every report count requests that were never actually examined.

Note what is NOT here: the prompt text. The policy reads the prompt in memory
and stores only its verdict and the names of the rules that matched. A feature
built to police what people type must not become the reason a company starts
recording what people type - that stays behind SWITCHBOARD_STORE_PROMPTS.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "requests", sa.Column("guardrail_label", sa.String(length=32), nullable=True)
    )
    op.add_column(
        "requests", sa.Column("guardrail_action", sa.String(length=16), nullable=True)
    )
    op.add_column("requests", sa.Column("guardrail_rules", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("requests", "guardrail_rules")
    op.drop_column("requests", "guardrail_action")
    op.drop_column("requests", "guardrail_label")

"""Record when an answer failed a check and was retried on a stronger model.

Revision ID: 0007
Revises: 0006
Created: 2026-09-03

Escalation is the first thing in Switchboard that can make TWO provider calls
for one request. That makes the accounting rule from the cascade experiments
load-bearing rather than academic:

    a request that escalated has paid for BOTH calls, and the ledger must say so

Charging only for the model that produced the final answer is the easiest way
to make this feature look better than it is - the same self-flattering error the
cascade scoring was built to avoid. `simulated_cost_usd` therefore holds the
sum, and `attempts` says how many calls it took, so the two can never be
silently conflated.

Three columns, all nullable or defaulted, because every request written before
this existed had exactly one attempt and no verification.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Which checks fired, comma separated. NULL means verification was off -
    # deliberately different from "" which means it ran and found nothing.
    op.add_column("requests", sa.Column("verification", sa.Text(), nullable=True))
    # The model that answered first, when the answer was rejected and retried.
    op.add_column(
        "requests", sa.Column("escalated_from", sa.String(length=128), nullable=True)
    )
    # Provider calls made. 1 for everything that did not escalate, including
    # every row written before this migration.
    op.add_column(
        "requests",
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="1"),
    )


def downgrade() -> None:
    op.drop_column("requests", "attempts")
    op.drop_column("requests", "escalated_from")
    op.drop_column("requests", "verification")

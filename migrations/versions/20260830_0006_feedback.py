"""Let an application say whether an answer was any good.

Revision ID: 0006
Revises: 0005
Created: 2026-08-30

This is the column the router has been missing since routing existed.

Benchmarks come with an answer key, so training on them was free. Real traffic
has no answer key - nobody wrote down the correct response to "why is this test
flaky". Without a label there is nothing to learn from, which is why a
benchmark-trained router never improves no matter how long you run it.

So: `public_id` goes out in a response header, and the application sends it
back with a verdict.

WHY A PUBLIC ID RATHER THAN THE ROW NUMBER. Two reasons, and the second is the
real one. A sequential id tells anyone who sees one roughly how many requests
this instance has ever served. And feedback has to be checked against the
caller who made the request, so an id that can be guessed by counting is an
invitation to try. A random token is a few bytes and removes both.

Nullable throughout: every row written before this existed has no verdict, and
`unrated` is a different fact from `not asked`. Reports must be able to tell
them apart, so the blank stays a blank.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "requests", sa.Column("public_id", sa.String(length=32), nullable=True)
    )
    op.add_column(
        "requests", sa.Column("feedback", sa.String(length=16), nullable=True)
    )
    op.add_column("requests", sa.Column("feedback_at", sa.DateTime(), nullable=True))
    op.add_column("requests", sa.Column("feedback_note", sa.Text(), nullable=True))

    # Unique so a token cannot collide; indexed because every feedback call
    # looks a request up by it. Rows written before this migration keep NULL,
    # which both SQLite and PostgreSQL exclude from a unique index.
    op.create_index(
        "ix_requests_public_id", "requests", ["public_id"], unique=True
    )
    # Training reads "every rated request", so the filter deserves its own
    # index rather than a full scan of the ledger.
    op.create_index("ix_requests_feedback", "requests", ["feedback"])


def downgrade() -> None:
    op.drop_index("ix_requests_feedback", table_name="requests")
    op.drop_index("ix_requests_public_id", table_name="requests")
    op.drop_column("requests", "feedback_note")
    op.drop_column("requests", "feedback_at")
    op.drop_column("requests", "feedback")
    op.drop_column("requests", "public_id")

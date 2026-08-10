"""Database tables: who may spend, and what was spent.

Timestamps are stored as NAIVE datetimes that are always UTC. SQLite does not
preserve timezone information, so carrying tz-aware values through it invites
subtle comparison bugs. One rule, applied everywhere: `utcnow()` below is the
only source of "now".
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """Naive UTC. The single source of time for the ledger."""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class User(Base):
    """A developer allowed to spend through the router."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    # Only the hash is stored - never the key itself. See ledger/keys.py.
    api_key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    monthly_budget_usd: Mapped[float] = mapped_column(Float, default=50.0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    requests: Mapped[list[RequestLog]] = relationship(back_populates="user")

    def __repr__(self) -> str:
        return f"<User {self.name} budget=${self.monthly_budget_usd:.2f}>"


class RequestLog(Base):
    """One row per request. This table is the ledger.

    It is also the training data for milestone 4: the router learns what a hard
    request looks like from `prompt_json` paired with outcomes.
    """

    __tablename__ = "requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    # --- Routing decision --------------------------------------------------
    # What the client asked for ("auto" hands the choice to Switchboard) versus
    # what actually ran. From milestone 4 these will differ meaningfully.
    requested_model: Mapped[str] = mapped_column(String(128))
    served_model: Mapped[str] = mapped_column(String(128))

    # --- Size --------------------------------------------------------------
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # True when the provider did not report usage and we approximated from
    # character count. Keeps estimated rows distinguishable from measured ones.
    tokens_estimated: Mapped[bool] = mapped_column(Boolean, default=False)

    # --- Simulated money ---------------------------------------------------
    simulated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    # What this would have cost on the baseline (top-tier) model. The savings
    # figure is `baseline_cost_usd - simulated_cost_usd`, per row. Storing it
    # here means savings is a query, not a reconstruction.
    baseline_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    # --- Real measured cost ------------------------------------------------
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)

    # Proxy for a cold VRAM load: this request used a different model than the
    # previous one, so Ollama probably had to swap weights. A proxy, not a
    # measurement - named accordingly.
    caused_model_switch: Mapped[bool] = mapped_column(Boolean, default=False)

    # --- Outcome -----------------------------------------------------------
    # ok | blocked_budget | provider_error | client_error
    status: Mapped[str] = mapped_column(String(32), default="ok", index=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- Content (optional) ------------------------------------------------
    # The full `messages` array as JSON, stored only when settings.store_prompts
    # is enabled. Holds whatever the user typed - treat as sensitive. The
    # database file is gitignored.
    prompt_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped[User] = relationship(back_populates="requests")

    @property
    def saved_usd(self) -> float:
        return self.baseline_cost_usd - self.simulated_cost_usd

    def __repr__(self) -> str:
        return (
            f"<RequestLog {self.served_model} "
            f"${self.simulated_cost_usd:.6f} {self.status}>"
        )

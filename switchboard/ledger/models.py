"""Database tables: who may spend, and what was spent.

Timestamps are stored as NAIVE datetimes that are always UTC. SQLite does not
preserve timezone information, so carrying tz-aware values through it invites
subtle comparison bugs. One rule, applied everywhere: `utcnow()` below is the
only source of "now".
"""

from __future__ import annotations

import secrets
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


def new_public_id() -> str:
    """A random, unguessable handle for one request.

    Handed to the client in a response header so it can send feedback back
    later. Random rather than the row number for two reasons: a sequential id
    would tell anyone who saw one how many requests this instance has served,
    and feedback is checked against the caller who made the request, so an id
    that can be guessed by counting invites people to try.
    """
    return secrets.token_urlsafe(16)[:22]


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
    # Requests per minute. NULL means use the server-wide default, so an
    # operator can raise the default without editing every user row.
    requests_per_minute: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
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
    # The handle the outside world uses. See new_public_id().
    public_id: Mapped[str | None] = mapped_column(
        String(32), default=new_public_id, unique=True, index=True, nullable=True
    )
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

    # Why the router chose this model. Free text, written by the strategy.
    # Without it, a routing decision that looks wrong cannot be debugged after
    # the fact - and shadow mode in Phase E reads it directly.
    routing_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- Shadow mode -------------------------------------------------------
    # What the router WOULD have chosen, when it was not allowed to choose.
    # Null on every request made with shadow mode off - which is why reports
    # skip those rows rather than counting them as "routing agreed".
    shadow_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Estimated, not measured: the shadow model was never called, so this
    # prices the REAL request's tokens at the shadow model's rates.
    shadow_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- Usage policy ------------------------------------------------------
    # What the guardrails made of this request. NULL means the policy was off,
    # which is deliberately different from "allowed": a report must be able to
    # tell "examined and fine" from "never examined".
    #
    # Never the prompt text - only the category and the names of the rules that
    # matched, which is enough to explain a flag or argue with it.
    guardrail_label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # allowed | flagged | blocked | overridden
    guardrail_action: Mapped[str | None] = mapped_column(String(16), nullable=True)
    guardrail_rules: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- Was the answer any good? ------------------------------------------
    # "good" | "bad", sent by the application through POST /v1/feedback.
    #
    # THE COLUMN THE ROUTER HAS BEEN MISSING. A benchmark ships with an answer
    # key; real traffic does not, so without this there is nothing to learn
    # from and a router can never improve on your own workload.
    #
    # NULL means nobody rated it, which is a different fact from "it was bad".
    # Training counts only rated rows, so the blank stays a blank.
    feedback: Mapped[str | None] = mapped_column(
        String(16), nullable=True, index=True
    )
    feedback_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    feedback_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- Outcome -----------------------------------------------------------
    # ok | blocked_budget | blocked_policy | provider_error | client_error
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

"""Ledger business logic: who is this, can they afford it, what did it cost.

Everything that touches money or identity lives here rather than in api.py, so
the HTTP layer stays a thin translation of these rules into status codes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select

from switchboard.catalog import ModelCatalog
from switchboard.ledger.db import Database
from switchboard.ledger.keys import generate_api_key, hash_api_key
from switchboard.ledger.models import RequestLog, User, utcnow

# Status values written to RequestLog.status.
STATUS_OK = "ok"
STATUS_BLOCKED_BUDGET = "blocked_budget"
STATUS_PROVIDER_ERROR = "provider_error"
STATUS_CLIENT_ERROR = "client_error"
# Served from the response cache. Recorded so hits are visible in usage
# reports, and priced at zero because nothing was actually bought.
STATUS_CACHED = "cached"

#: Statuses that represent a request the user actually made and got an
#: answer for. A cache hit belongs here: it happened, it counts, and it
#: cost nothing - which is precisely the saving the cache produced.
SERVED_STATUSES = (STATUS_OK, STATUS_CACHED)

# Rough characters-per-token, used only when the provider fails to report
# usage. Deliberately crude - rows relying on it are flagged
# `tokens_estimated=True` so they never masquerade as measurements.
CHARS_PER_TOKEN = 4


class LedgerError(Exception):
    """Base class for ledger failures the API translates into responses."""


class AuthenticationError(LedgerError):
    """No key, unknown key, or a deactivated user. -> HTTP 401."""


class BudgetExceededError(LedgerError):
    """User has spent their monthly allowance. -> HTTP 402."""

    def __init__(self, user: str, spent: float, budget: float) -> None:
        self.user = user
        self.spent = spent
        self.budget = budget
        super().__init__(
            f"{user} has used ${spent:.4f} of a ${budget:.2f} monthly budget "
            "(simulated)."
        )


@dataclass(frozen=True)
class NewUser:
    """Returned once at creation - the only time the raw key exists."""

    name: str
    api_key: str
    monthly_budget_usd: float


@dataclass(frozen=True)
class UsageRow:
    name: str
    requests: int
    spent_usd: float
    baseline_usd: float
    budget_usd: float

    @property
    def saved_usd(self) -> float:
        return self.baseline_usd - self.spent_usd

    @property
    def saved_pct(self) -> float:
        if self.baseline_usd <= 0:
            return 0.0
        return 100.0 * self.saved_usd / self.baseline_usd

    @property
    def remaining_usd(self) -> float:
        return self.budget_usd - self.spent_usd


def month_start(now: datetime) -> datetime:
    """First instant of the calendar month containing `now` (UTC).

    Calendar month rather than a rolling 30-day window: it matches how people
    already think about budgets and is far easier to explain in a dashboard.
    """
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN) if text else 0


class LedgerService:
    def __init__(
        self, db: Database, catalog: ModelCatalog, store_prompts: bool
    ) -> None:
        self._db = db
        self._prices = catalog
        self._store_prompts = store_prompts

        # In-memory proxy for VRAM cold loads. Not persisted: it describes the
        # live process, and a restart genuinely does clear Ollama's warm set.
        self._last_served_model: str | None = None

    # --- Users -------------------------------------------------------------

    def create_user(self, name: str, monthly_budget_usd: float) -> NewUser:
        raw_key = generate_api_key()
        with self._db.session() as session:
            if session.scalar(select(User).where(User.name == name)):
                raise LedgerError(f"User {name!r} already exists.")
            session.add(
                User(
                    name=name,
                    api_key_hash=hash_api_key(raw_key),
                    monthly_budget_usd=monthly_budget_usd,
                )
            )
        return NewUser(name, raw_key, monthly_budget_usd)

    def set_active(self, name: str, active: bool) -> None:
        with self._db.session() as session:
            user = session.scalar(select(User).where(User.name == name))
            if user is None:
                raise LedgerError(f"No such user: {name!r}")
            user.is_active = active

    def set_budget(self, name: str, monthly_budget_usd: float) -> None:
        with self._db.session() as session:
            user = session.scalar(select(User).where(User.name == name))
            if user is None:
                raise LedgerError(f"No such user: {name!r}")
            user.monthly_budget_usd = monthly_budget_usd

    def authenticate(self, raw_key: str | None) -> User:
        if not raw_key:
            raise AuthenticationError(
                "Missing API key. Send it as 'Authorization: Bearer <key>'."
            )

        with self._db.session() as session:
            user = session.scalar(
                select(User).where(User.api_key_hash == hash_api_key(raw_key))
            )

        if user is None:
            raise AuthenticationError("Unknown API key.")
        if not user.is_active:
            raise AuthenticationError(f"User {user.name!r} is deactivated.")
        return user

    # --- Money -------------------------------------------------------------

    def month_to_date_spend(self, user_id: int, now: datetime | None = None) -> float:
        """Simulated spend this calendar month.

        Only successful requests count. A request blocked for lack of budget
        cost nothing to serve, so charging for it would let a user dig
        themselves deeper by retrying.
        """
        now = now or utcnow()
        with self._db.session() as session:
            total = session.scalar(
                select(func.coalesce(func.sum(RequestLog.simulated_cost_usd), 0.0))
                .where(RequestLog.user_id == user_id)
                .where(RequestLog.created_at >= month_start(now))
                .where(RequestLog.status.in_(SERVED_STATUSES))
            )
        return float(total or 0.0)

    def assert_within_budget(self, user: User) -> float:
        """Raise if the user is already over budget. Returns spend so far.

        Checked BEFORE serving, when the final cost is not yet knowable - the
        answer's length decides it. So this gates on spend-to-date only, which
        means a user's last request can carry them slightly over. That is the
        same trade-off commercial providers make; pre-authorising an unknown
        amount is not worth the complexity here.
        """
        spent = self.month_to_date_spend(user.id)
        if spent >= user.monthly_budget_usd:
            raise BudgetExceededError(user.name, spent, user.monthly_budget_usd)
        return spent

    # --- Recording ---------------------------------------------------------

    def record(
        self,
        *,
        user_id: int,
        requested_model: str,
        served_model: str,
        prompt_tokens: int,
        completion_tokens: int,
        tokens_estimated: bool,
        latency_ms: int,
        status: str = STATUS_OK,
        error_detail: str | None = None,
        messages: list | None = None,
        routing_reason: str | None = None,
    ) -> RequestLog:
        # A cache hit bought nothing, so it costs nothing. The BASELINE is
        # still what those tokens would have cost on the top-tier model, which
        # is exactly the saving the cache produced.
        cost = (
            0.0
            if status == STATUS_CACHED
            else self._prices.cost(served_model, prompt_tokens, completion_tokens)
        )
        baseline = self._prices.baseline_cost(prompt_tokens, completion_tokens)

        switched = (
            self._last_served_model is not None
            and self._last_served_model != served_model
        )
        if status == STATUS_OK:
            self._last_served_model = served_model

        entry = RequestLog(
            user_id=user_id,
            requested_model=requested_model,
            served_model=served_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_estimated=tokens_estimated,
            simulated_cost_usd=cost,
            baseline_cost_usd=baseline,
            latency_ms=latency_ms,
            caused_model_switch=switched,
            status=status,
            error_detail=error_detail,
            routing_reason=routing_reason,
            prompt_json=(
                json.dumps(messages, ensure_ascii=False)
                if self._store_prompts and messages is not None
                else None
            ),
        )
        with self._db.session() as session:
            session.add(entry)
        return entry

    # --- Reporting ---------------------------------------------------------

    def usage(self, now: datetime | None = None) -> list[UsageRow]:
        """Month-to-date usage per user, for the `usage` CLI command."""
        now = now or utcnow()
        start = month_start(now)

        with self._db.session() as session:
            rows = session.execute(
                select(
                    User.name,
                    func.count(RequestLog.id),
                    func.coalesce(func.sum(RequestLog.simulated_cost_usd), 0.0),
                    func.coalesce(func.sum(RequestLog.baseline_cost_usd), 0.0),
                    User.monthly_budget_usd,
                )
                .select_from(User)
                .outerjoin(
                    RequestLog,
                    (RequestLog.user_id == User.id)
                    & (RequestLog.created_at >= start)
                    & (RequestLog.status.in_(SERVED_STATUSES)),
                )
                .group_by(User.id)
                .order_by(User.name)
            ).all()

        return [
            UsageRow(
                name=name,
                requests=count,
                spent_usd=float(spent),
                baseline_usd=float(baseline),
                budget_usd=float(budget),
            )
            for name, count, spent, baseline, budget in rows
        ]

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
# Refused by the usage policy in `block` mode. Recorded so a refusal is
# visible and arguable, and priced at zero because nothing was bought.
STATUS_BLOCKED_POLICY = "blocked_policy"

#: Verdicts POST /v1/feedback accepts. Two values on purpose: a five-star
#: scale sounds richer and is not - people use the ends, the middle means
#: different things to different raters, and the router needs a yes or no.
FEEDBACK_GOOD = "good"
FEEDBACK_BAD = "bad"
FEEDBACK_VALUES = (FEEDBACK_GOOD, FEEDBACK_BAD)

#: Statuses that represent a request the user actually made and got an
#: answer for. A cache hit belongs here: it happened, it counts, and it
#: cost nothing - which is precisely the saving the cache produced.
SERVED_STATUSES = (STATUS_OK, STATUS_CACHED)

# Rough characters-per-token, used only when the provider fails to report
# usage. Deliberately crude - rows relying on it are flagged
# `tokens_estimated=True` so they never masquerade as measurements.
CHARS_PER_TOKEN = 4


class UnknownRequest(Exception):
    """No such request for this user. -> HTTP 404.

    Deliberately the same answer whether the id does not exist or belongs
    to somebody else. Telling them apart would turn the feedback endpoint
    into a way to discover other people's request ids.
    """


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
        shadow_model: str | None = None,
        shadow_cost_usd: float | None = None,
        guardrail_label: str | None = None,
        guardrail_action: str | None = None,
        guardrail_rules: str | None = None,
        public_id: str | None = None,
        verification: str | None = None,
        escalated_from: str | None = None,
        attempts: int = 1,
        # Cost of any earlier attempt whose answer was rejected. Added to
        # this row's cost, because the request paid for both calls.
        extra_cost_usd: float = 0.0,
    ) -> RequestLog:
        # A cache hit bought nothing, so it costs nothing. The BASELINE is
        # still what those tokens would have cost on the top-tier model, which
        # is exactly the saving the cache produced.
        cost = (
            0.0
            if status in (STATUS_CACHED, STATUS_BLOCKED_POLICY)
            else self._prices.cost(served_model, prompt_tokens, completion_tokens)
            + extra_cost_usd
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
            shadow_model=shadow_model,
            shadow_cost_usd=shadow_cost_usd,
            guardrail_label=guardrail_label,
            guardrail_action=guardrail_action,
            guardrail_rules=guardrail_rules,
            verification=verification,
            escalated_from=escalated_from,
            attempts=attempts,
            # Generated by the caller before the request is served, because
            # a streamed response sends its headers long before the ledger
            # row is written at the end of the stream.
            **({"public_id": public_id} if public_id else {}),
            prompt_json=(
                json.dumps(messages, ensure_ascii=False)
                if self._store_prompts and messages is not None
                else None
            ),
        )
        with self._db.session() as session:
            session.add(entry)
        return entry


    def shadow_rows(self, now: datetime | None = None) -> list:
        """Requests this month that carry a routing opinion.

        Rows without one are excluded at the query, not filtered later: a
        request served before shadow mode was switched on has no opinion, and
        counting it as "routing agreed" would dilute every projection.
        """
        now = now or utcnow()
        with self._db.session() as session:
            return list(
                session.scalars(
                    select(RequestLog)
                    .where(RequestLog.created_at >= month_start(now))
                    .where(RequestLog.shadow_model.is_not(None))
                )
            )

    # --- Feedback ----------------------------------------------------------

    def record_feedback(
        self,
        user_id: int,
        public_id: str,
        rating: str,
        note: str | None = None,
    ) -> RequestLog:
        """Attach a verdict to one request.

        Scoped to the caller's own requests. Without that check, anyone with a
        valid API key could rate anyone else's traffic, and since the ratings
        become training data, that is a way to steer another team's router.
        """
        if rating not in FEEDBACK_VALUES:
            raise LedgerError(
                f"rating must be one of {', '.join(FEEDBACK_VALUES)}, "
                f"got {rating!r}"
            )

        with self._db.session() as session:
            entry = session.scalar(
                select(RequestLog)
                .where(RequestLog.public_id == public_id)
                .where(RequestLog.user_id == user_id)
            )
            if entry is None:
                raise UnknownRequest(f"No request {public_id!r} for this user.")

            # Re-rating is allowed and overwrites. Someone who reads an answer
            # again and changes their mind is giving better information than
            # their first reaction, not worse.
            entry.feedback = rating
            entry.feedback_at = utcnow()
            entry.feedback_note = (note or None) and str(note)[:1000]
            session.flush()
            session.expunge(entry)
            return entry

    def rated_requests(self, since: datetime | None = None) -> list[RequestLog]:
        """Every request carrying a verdict. The router's training set.

        Only rows with a rating come back. An unrated request is not a bad one,
        and counting it as either would teach the router something nobody said.
        """
        with self._db.session() as session:
            query = (
                select(RequestLog)
                .where(RequestLog.feedback.is_not(None))
                .where(RequestLog.status.in_(SERVED_STATUSES))
                .order_by(RequestLog.created_at)
            )
            if since is not None:
                query = query.where(RequestLog.created_at >= since)
            rows = list(session.scalars(query))
            for row in rows:
                session.expunge(row)
            return rows

    def feedback_counts(self) -> list[tuple[str, str, int]]:
        """(model, rating, count) over all time, for the readiness report."""
        with self._db.session() as session:
            rows = session.execute(
                select(
                    RequestLog.served_model,
                    RequestLog.feedback,
                    func.count(RequestLog.id),
                )
                .where(RequestLog.feedback.is_not(None))
                .where(RequestLog.status.in_(SERVED_STATUSES))
                .group_by(RequestLog.served_model, RequestLog.feedback)
            ).all()
        return [(str(m), str(f), int(c)) for m, f, c in rows]

    def served_counts(self) -> list[tuple[str, int]]:
        """(model, requests served) over all time, rated or not."""
        with self._db.session() as session:
            rows = session.execute(
                select(RequestLog.served_model, func.count(RequestLog.id))
                .where(RequestLog.status.in_(SERVED_STATUSES))
                .group_by(RequestLog.served_model)
                .order_by(func.count(RequestLog.id).desc())
            ).all()
        return [(str(m), int(c)) for m, c in rows]

    def guardrail_counts(
        self, now: datetime | None = None
    ) -> list[tuple[str, str, int, float]]:
        """(label, action, requests, cost) this month for examined requests.

        Rows where the policy was off are excluded at the query. Counting them
        as "allowed" would let someone report a clean month that was never
        actually examined.
        """
        now = now or utcnow()
        with self._db.session() as session:
            rows = session.execute(
                select(
                    RequestLog.guardrail_label,
                    RequestLog.guardrail_action,
                    func.count(RequestLog.id),
                    func.coalesce(func.sum(RequestLog.simulated_cost_usd), 0.0),
                )
                .where(RequestLog.created_at >= month_start(now))
                .where(RequestLog.guardrail_action.is_not(None))
                .group_by(RequestLog.guardrail_label, RequestLog.guardrail_action)
                .order_by(func.count(RequestLog.id).desc())
            ).all()
        return [
            (str(label or "clean"), str(action), int(count), float(cost))
            for label, action, count, cost in rows
        ]

    def flagged_rules(self, now: datetime | None = None) -> list[tuple[str, int]]:
        """Which rules are doing the flagging, busiest first.

        This is the list an operator uses to find a rule that keeps tripping on
        their team's real work, so they can delete it from a custom rule file.
        """
        now = now or utcnow()
        counts: dict[str, int] = {}
        with self._db.session() as session:
            rows = session.scalars(
                select(RequestLog.guardrail_rules)
                .where(RequestLog.created_at >= month_start(now))
                .where(RequestLog.guardrail_rules.is_not(None))
            )
            for value in rows:
                for name in str(value).split(","):
                    if name := name.strip():
                        counts[name] = counts.get(name, 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])

    def by_model(self, now: datetime | None = None) -> list[tuple[str, int, float]]:
        """(model, requests, cost) this month, busiest first."""
        now = now or utcnow()
        with self._db.session() as session:
            rows = session.execute(
                select(
                    RequestLog.served_model,
                    func.count(RequestLog.id),
                    func.coalesce(func.sum(RequestLog.simulated_cost_usd), 0.0),
                )
                .where(RequestLog.created_at >= month_start(now))
                .where(RequestLog.status.in_(SERVED_STATUSES))
                .group_by(RequestLog.served_model)
                .order_by(func.count(RequestLog.id).desc())
            ).all()
        return [(str(m), int(c), float(cost)) for m, c, cost in rows]

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

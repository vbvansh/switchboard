"""Ledger rules: identity, budgets, month boundaries, accounting."""

from __future__ import annotations

from datetime import datetime

import pytest

from switchboard.ledger import (
    STATUS_BLOCKED_BUDGET,
    STATUS_OK,
    AuthenticationError,
    BudgetExceededError,
    LedgerError,
    LedgerService,
)
from switchboard.ledger.keys import (
    extract_bearer_token,
    generate_api_key,
    hash_api_key,
    looks_like_api_key,
)
from switchboard.ledger.service import month_start

# --- Keys ------------------------------------------------------------------


def test_generated_keys_are_unique_and_prefixed() -> None:
    keys = {generate_api_key() for _ in range(100)}
    assert len(keys) == 100
    assert all(looks_like_api_key(k) for k in keys)


def test_hash_is_deterministic_and_hides_the_key() -> None:
    key = generate_api_key()
    assert hash_api_key(key) == hash_api_key(key)
    assert key not in hash_api_key(key)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Bearer sk-swbd-abc", "sk-swbd-abc"),
        ("bearer sk-swbd-abc", "sk-swbd-abc"),
        ("sk-swbd-abc", "sk-swbd-abc"),
        ("  Bearer   sk-swbd-abc  ", "sk-swbd-abc"),
        (None, None),
        ("", None),
        ("Bearer ", None),
    ],
)
def test_bearer_token_extraction(header: str | None, expected: str | None) -> None:
    assert extract_bearer_token(header) == expected


# --- Users -----------------------------------------------------------------


def test_created_user_can_authenticate(ledger: LedgerService) -> None:
    created = ledger.create_user("alice", 50.0)
    assert ledger.authenticate(created.api_key).name == "alice"


def test_duplicate_user_is_rejected(ledger: LedgerService) -> None:
    ledger.create_user("alice", 50.0)
    with pytest.raises(LedgerError, match="already exists"):
        ledger.create_user("alice", 10.0)


def test_unknown_key_is_rejected(ledger: LedgerService) -> None:
    ledger.create_user("alice", 50.0)
    with pytest.raises(AuthenticationError, match="Unknown API key"):
        ledger.authenticate(generate_api_key())


def test_missing_key_is_rejected(ledger: LedgerService) -> None:
    with pytest.raises(AuthenticationError, match="Missing API key"):
        ledger.authenticate(None)


def test_deactivated_user_is_rejected(ledger: LedgerService) -> None:
    created = ledger.create_user("alice", 50.0)
    ledger.set_active("alice", False)
    with pytest.raises(AuthenticationError, match="deactivated"):
        ledger.authenticate(created.api_key)


def test_reactivated_user_works_again(ledger: LedgerService) -> None:
    created = ledger.create_user("alice", 50.0)
    ledger.set_active("alice", False)
    ledger.set_active("alice", True)
    assert ledger.authenticate(created.api_key).name == "alice"


# --- Month boundary --------------------------------------------------------


def test_month_start_truncates_to_first_instant() -> None:
    assert month_start(datetime(2026, 8, 17, 13, 45, 30)) == datetime(2026, 8, 1)


def test_spend_excludes_previous_months(ledger: LedgerService) -> None:
    """Last month's spend must not eat this month's budget."""
    user = ledger.authenticate(ledger.create_user("alice", 50.0).api_key)

    entry = ledger.record(
        user_id=user.id,
        requested_model="auto",
        served_model="qwen2.5:7b",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        tokens_estimated=False,
        latency_ms=10,
    )

    # Backdate it into the previous month.
    with ledger._db.session() as session:
        from switchboard.ledger.models import RequestLog

        row = session.get(RequestLog, entry.id)
        row.created_at = datetime(2020, 1, 15)

    assert ledger.month_to_date_spend(user.id) == 0.0


# --- Budgets ---------------------------------------------------------------


def test_spend_accumulates(ledger: LedgerService) -> None:
    user = ledger.authenticate(ledger.create_user("alice", 50.0).api_key)
    for _ in range(3):
        ledger.record(
            user_id=user.id,
            requested_model="auto",
            served_model="qwen2.5:7b",  # $3.00 per Mtok input
            prompt_tokens=1_000_000,
            completion_tokens=0,
            tokens_estimated=False,
            latency_ms=10,
        )
    assert ledger.month_to_date_spend(user.id) == pytest.approx(9.0)


def test_budget_blocks_once_exhausted(ledger: LedgerService) -> None:
    user = ledger.authenticate(ledger.create_user("alice", 5.0).api_key)
    ledger.assert_within_budget(user)  # fine at zero spend

    ledger.record(
        user_id=user.id,
        requested_model="auto",
        served_model="qwen2.5:7b",
        prompt_tokens=2_000_000,  # $6.00 - over the $5 budget
        completion_tokens=0,
        tokens_estimated=False,
        latency_ms=10,
    )

    with pytest.raises(BudgetExceededError):
        ledger.assert_within_budget(user)


def test_blocked_requests_do_not_accumulate_spend(ledger: LedgerService) -> None:
    """A blocked attempt costs nothing, so retrying cannot dig a deeper hole."""
    user = ledger.authenticate(ledger.create_user("alice", 50.0).api_key)
    ledger.record(
        user_id=user.id,
        requested_model="auto",
        served_model="qwen2.5:7b",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        tokens_estimated=False,
        latency_ms=0,
        status=STATUS_BLOCKED_BUDGET,
    )
    assert ledger.month_to_date_spend(user.id) == 0.0


def test_raising_the_budget_unblocks(ledger: LedgerService) -> None:
    created = ledger.create_user("alice", 1.0)
    user = ledger.authenticate(created.api_key)
    ledger.record(
        user_id=user.id,
        requested_model="auto",
        served_model="qwen2.5:7b",
        prompt_tokens=1_000_000,  # $3.00 against a $1 budget
        completion_tokens=0,
        tokens_estimated=False,
        latency_ms=10,
    )
    with pytest.raises(BudgetExceededError):
        ledger.assert_within_budget(user)

    ledger.set_budget("alice", 100.0)
    # Re-authenticate so the budget change is reflected on the User object.
    assert ledger.assert_within_budget(ledger.authenticate(created.api_key)) > 0


# --- Recording -------------------------------------------------------------


def test_record_stores_baseline_and_savings(ledger: LedgerService) -> None:
    user = ledger.authenticate(ledger.create_user("alice", 50.0).api_key)
    entry = ledger.record(
        user_id=user.id,
        requested_model="auto",
        served_model="qwen2.5:1.5b",  # $0.10 in / $0.40 out
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        tokens_estimated=False,
        latency_ms=10,
    )
    assert entry.simulated_cost_usd == pytest.approx(0.50)
    # Baseline qwen2.5:7b = $3.00 in + $15.00 out
    assert entry.baseline_cost_usd == pytest.approx(18.0)
    assert entry.saved_usd == pytest.approx(17.5)


def test_prompts_stored_when_enabled(ledger: LedgerService) -> None:
    user = ledger.authenticate(ledger.create_user("alice", 50.0).api_key)
    entry = ledger.record(
        user_id=user.id,
        requested_model="auto",
        served_model="qwen2.5:3b",
        prompt_tokens=10,
        completion_tokens=10,
        tokens_estimated=False,
        latency_ms=1,
        messages=[{"role": "user", "content": "secret question"}],
    )
    assert "secret question" in entry.prompt_json


def test_prompts_withheld_when_disabled(database) -> None:
    from switchboard.catalog import ModelCatalog

    private = LedgerService(database, ModelCatalog.load(), store_prompts=False)
    user = private.authenticate(private.create_user("bob", 50.0).api_key)
    entry = private.record(
        user_id=user.id,
        requested_model="auto",
        served_model="qwen2.5:3b",
        prompt_tokens=10,
        completion_tokens=10,
        tokens_estimated=False,
        latency_ms=1,
        messages=[{"role": "user", "content": "secret question"}],
    )
    assert entry.prompt_json is None


def test_model_switch_is_flagged(ledger: LedgerService) -> None:
    """Proxy for a VRAM cold load: the served model changed."""
    user = ledger.authenticate(ledger.create_user("alice", 50.0).api_key)

    def serve(model: str):
        return ledger.record(
            user_id=user.id,
            requested_model="auto",
            served_model=model,
            prompt_tokens=10,
            completion_tokens=10,
            tokens_estimated=False,
            latency_ms=1,
        )

    assert serve("qwen2.5:3b").caused_model_switch is False  # nothing was warm
    assert serve("qwen2.5:3b").caused_model_switch is False  # same model
    assert serve("qwen2.5:7b").caused_model_switch is True  # swapped


# --- Reporting -------------------------------------------------------------


def test_usage_reports_zero_for_a_new_user(ledger: LedgerService) -> None:
    ledger.create_user("alice", 50.0)
    (row,) = ledger.usage()
    assert (row.name, row.requests, row.spent_usd) == ("alice", 0, 0.0)
    assert row.remaining_usd == 50.0


def test_usage_summarises_spend_and_savings(ledger: LedgerService) -> None:
    user = ledger.authenticate(ledger.create_user("alice", 50.0).api_key)
    ledger.record(
        user_id=user.id,
        requested_model="auto",
        served_model="qwen2.5:1.5b",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        tokens_estimated=False,
        latency_ms=10,
        status=STATUS_OK,
    )
    (row,) = ledger.usage()
    assert row.requests == 1
    assert row.spent_usd == pytest.approx(0.10)
    assert row.baseline_usd == pytest.approx(3.00)
    assert row.saved_pct == pytest.approx(96.667, rel=1e-3)

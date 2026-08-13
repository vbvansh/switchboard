"""Privacy defaults.

Installing software and leaving the defaults alone must not cause it to start
recording everything users type. These tests pin that behaviour so it cannot be
undone by accident.
"""

from __future__ import annotations

from switchboard.catalog import ModelCatalog
from switchboard.config import Settings
from switchboard.ledger import Database, LedgerService


def test_prompt_storage_is_off_by_default() -> None:
    """The load-bearing test for the whole privacy posture.

    If this ever flips, every Switchboard install starts collecting whatever
    its users type - customer data, credentials, personal information - with
    the legal exposure that carries, and nobody chose it.
    """
    assert Settings().store_prompts is False


def test_prompt_storage_can_be_switched_on_deliberately() -> None:
    assert Settings(store_prompts=True).store_prompts is True


def test_default_settings_record_no_prompt_text(database: Database) -> None:
    """End to end: with stock settings, prompt text never reaches the database."""
    service = LedgerService(
        database, ModelCatalog.load(), store_prompts=Settings().store_prompts
    )
    user = service.authenticate(service.create_user("alice", 50.0).api_key)

    entry = service.record(
        user_id=user.id,
        requested_model="auto",
        served_model="qwen2.5:3b",
        prompt_tokens=10,
        completion_tokens=10,
        tokens_estimated=False,
        latency_ms=1,
        messages=[{"role": "user", "content": "my private customer data"}],
    )

    assert entry.prompt_json is None


def test_costs_are_still_recorded_without_prompt_text(database: Database) -> None:
    """Privacy must not cost you the accounting - that is the whole product."""
    service = LedgerService(database, ModelCatalog.load(), store_prompts=False)
    user = service.authenticate(service.create_user("alice", 50.0).api_key)

    entry = service.record(
        user_id=user.id,
        requested_model="auto",
        served_model="qwen2.5:1.5b",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        tokens_estimated=False,
        latency_ms=1,
        messages=[{"role": "user", "content": "secret"}],
    )

    assert entry.prompt_json is None
    assert entry.simulated_cost_usd > 0
    assert entry.baseline_cost_usd > 0
    assert entry.prompt_tokens == 1_000_000

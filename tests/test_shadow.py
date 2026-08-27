"""Shadow mode: measure what routing would do, without letting it do it.

The load-bearing property in this file is that shadow mode NEVER changes what
gets served. If it ever did, someone trialling routing safely would find it
had been live on their production traffic all along.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from sqlalchemy import select

from switchboard.catalog import ModelCatalog
from switchboard.dashboard import DashboardData, render
from switchboard.ledger.models import RequestLog
from switchboard.routing.base import RoutingDecision
from switchboard.shadow import ShadowSummary, estimate_cost, summarise


@dataclass
class Row:
    """A ledger row, reduced to what the summary reads."""

    served_model: str
    simulated_cost_usd: float
    shadow_model: str | None = None
    shadow_cost_usd: float | None = None


# --- Cost estimation --------------------------------------------------------


def test_shadow_cost_prices_the_real_tokens_at_the_other_model(
    prices: ModelCatalog,
) -> None:
    """The only honest option: the shadow model was never called, so its own
    token count does not exist."""
    cheap = estimate_cost(prices, "qwen2.5:1.5b", 1_000_000, 0)
    dear = estimate_cost(prices, "qwen2.5:7b", 1_000_000, 0)
    assert cheap < dear
    assert cheap == pytest.approx(0.10)


# --- Summarising ------------------------------------------------------------


def test_rows_without_an_opinion_are_skipped() -> None:
    """A request served before shadow mode was on has no opinion attached.

    Counting it as "routing agreed" would quietly dilute every projection
    towards zero.
    """
    summary = summarise([Row("big", 1.0), Row("big", 1.0)])
    assert summary.requests == 0


def test_a_projected_saving_is_computed() -> None:
    summary = summarise(
        [
            Row("big", 1.00, shadow_model="small", shadow_cost_usd=0.10),
            Row("big", 1.00, shadow_model="small", shadow_cost_usd=0.10),
        ]
    )
    assert summary.requests == 2
    assert summary.actual_cost_usd == pytest.approx(2.0)
    assert summary.shadow_cost_usd == pytest.approx(0.2)
    assert summary.projected_saving_pct == pytest.approx(90.0)


def test_routing_costing_more_is_reported_honestly() -> None:
    """A negative saving must show as a negative saving, not be hidden."""
    summary = summarise(
        [Row("small", 0.10, shadow_model="big", shadow_cost_usd=1.00)]
    )
    assert summary.projected_saving_usd < 0
    assert "COST" in summary.describe()


def test_changes_are_split_into_cheaper_and_dearer() -> None:
    summary = summarise(
        [
            Row("big", 1.00, shadow_model="small", shadow_cost_usd=0.10),
            Row("small", 0.10, shadow_model="big", shadow_cost_usd=1.00),
            Row("mid", 0.50, shadow_model="mid", shadow_cost_usd=0.50),
        ]
    )
    assert summary.requests == 3
    assert summary.changed == 2
    assert summary.downgraded == 1
    assert summary.upgraded == 1


def test_an_empty_summary_says_so() -> None:
    assert "No shadowed requests" in ShadowSummary().describe()


def test_zero_spend_does_not_divide_by_zero() -> None:
    summary = summarise([Row("m", 0.0, shadow_model="other", shadow_cost_usd=0.0)])
    assert summary.projected_saving_pct == 0.0


# --- The behaviour that matters ---------------------------------------------


class StubRouter:
    """A router that always wants the expensive model."""

    enabled = True
    metadata = type("M", (), {"models": ["a", "b"], "describe": lambda s: "stub"})()
    routable_models = ["qwen2.5:7b"]

    def __init__(self, pick: str = "qwen2.5:7b") -> None:
        self.pick = pick
        self.calls = 0

    def choose(self, context, limits=None):
        self.calls += 1
        return RoutingDecision(
            model=self.pick, strategy="stub", reason="stub always picks this"
        )


def _chat(model: str = "auto") -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
    }


@pytest.fixture
def shadow_on(monkeypatch):
    from switchboard.config import settings

    monkeypatch.setattr(settings, "shadow_mode", True)
    return settings


def test_shadow_mode_does_not_change_what_is_served(
    client, auth, provider, shadow_on
) -> None:
    """THE test. If this fails, shadow mode is live routing in disguise."""
    from switchboard.config import settings

    client.app.state.router = StubRouter("qwen2.5:7b")
    client.post("/v1/chat/completions", json=_chat("auto"), headers=auth)

    assert provider.last_payload["model"] == settings.default_model
    assert provider.last_payload["model"] != "qwen2.5:7b"


def test_shadow_mode_still_records_the_opinion(
    client, auth, database, shadow_on
) -> None:
    client.app.state.router = StubRouter("qwen2.5:7b")
    client.post("/v1/chat/completions", json=_chat("auto"), headers=auth)

    with database.session() as session:
        row = session.scalar(select(RequestLog))

    assert row.shadow_model == "qwen2.5:7b"
    assert row.shadow_cost_usd > 0
    assert "shadow" in row.routing_reason


def test_the_router_is_still_consulted(client, auth, shadow_on) -> None:
    """Recording an opinion means actually forming one."""
    router = StubRouter()
    client.app.state.router = router
    client.post("/v1/chat/completions", json=_chat("auto"), headers=auth)
    assert router.calls == 1


def test_with_shadow_off_the_router_decides(client, auth, provider) -> None:
    client.app.state.router = StubRouter("qwen2.5:7b")
    client.post("/v1/chat/completions", json=_chat("auto"), headers=auth)
    assert provider.last_payload["model"] == "qwen2.5:7b"


def test_an_explicit_model_is_still_honoured_and_shadowed(
    client, auth, provider, database
) -> None:
    """Naming a model must always win. The router's opinion is still recorded,
    which is how you measure routing against traffic that never used it."""
    client.app.state.router = StubRouter("qwen2.5:7b")
    client.post("/v1/chat/completions", json=_chat("qwen2.5:1.5b"), headers=auth)

    assert provider.last_payload["model"] == "qwen2.5:1.5b"
    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert row.shadow_model == "qwen2.5:7b"


def test_no_shadow_is_recorded_when_the_choice_agrees(
    client, auth, database, shadow_on
) -> None:
    """Recording "routing would have done what we did" adds a row of noise to
    every report for no information."""
    from switchboard.config import settings

    client.app.state.router = StubRouter(settings.default_model)
    client.post("/v1/chat/completions", json=_chat("auto"), headers=auth)

    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert row.shadow_model is None


def test_no_router_means_no_shadow(client, auth, database, shadow_on) -> None:
    client.app.state.router = None
    client.post("/v1/chat/completions", json=_chat("auto"), headers=auth)

    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert row.shadow_model is None


def test_health_reports_shadow_mode(client, shadow_on) -> None:
    assert client.get("/health").json()["shadow_mode"] is True


# --- The dashboard ----------------------------------------------------------


@dataclass
class UsageRow:
    name: str
    requests: int
    spent_usd: float
    baseline_usd: float
    budget_usd: float

    @property
    def saved_pct(self) -> float:
        return 100.0 * (self.baseline_usd - self.spent_usd) / self.baseline_usd

    @property
    def remaining_usd(self) -> float:
        return self.budget_usd - self.spent_usd


def sample_data(**overrides) -> DashboardData:
    defaults = {
        "usage_rows": [UsageRow("alice", 10, 1.0, 10.0, 50.0)],
        "model_rows": [("qwen2.5:3b", 8, 0.8), ("qwen2.5:7b", 2, 0.2)],
        "shadow": ShadowSummary(),
        "cache": {"hit_rate": 0.25, "hits": 4},
        "routing": {"enabled": True, "models": ["qwen2.5:3b"]},
        "simulated": True,
        "shadow_mode": False,
    }
    return DashboardData(**{**defaults, **overrides})


def test_the_page_renders() -> None:
    page = render(sample_data())
    assert page.startswith("<!doctype html>")
    assert "alice" in page


def test_the_page_has_no_external_requests() -> None:
    """It must work on a machine with no internet - the deployment
    SWITCHBOARD_LOCAL_ONLY exists to support - and must not tell a CDN when
    your engineers look at their AI spend."""
    page = render(sample_data())
    for forbidden in ("http://", "https://", "<script", "cdn."):
        assert forbidden not in page, forbidden


def test_simulated_pricing_is_declared_on_the_page() -> None:
    assert "Simulated pricing" in render(sample_data(simulated=True))
    assert "Simulated pricing" not in render(sample_data(simulated=False))


def test_user_content_is_escaped() -> None:
    """A user called <script> must not become one."""
    evil = UsageRow("<script>alert(1)</script>", 1, 1.0, 2.0, 5.0)
    page = render(sample_data(usage_rows=[evil]))
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page


def test_shadow_projections_are_labelled_as_projections() -> None:
    summary = summarise(
        [Row("big", 1.0, shadow_model="small", shadow_cost_usd=0.1)]
    )
    page = render(sample_data(shadow=summary, shadow_mode=True))
    assert "projections, not measurements" in page
    assert "estimated" in page


def test_the_page_explains_how_to_turn_shadow_mode_on() -> None:
    page = render(sample_data(shadow_mode=False))
    assert "SWITCHBOARD_SHADOW_MODE" in page


def test_an_empty_ledger_renders_without_crashing() -> None:
    page = render(sample_data(usage_rows=[], model_rows=[]))
    assert "No usage recorded yet" in page


def test_the_dashboard_endpoint_serves_html(client) -> None:
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Switchboard" in response.text

"""Saying "I don't know", and the order routers are tried in.

THE FAILURE THIS PREVENTS. In Phase C.4 the router was shown prompts unlike its
training data and returned 0.67 to 0.87 for every model — no discrimination at
all. Everything went to the cheapest model, and each ledger row carried a reason
implying a judgement had been made. The router was not wrong; it was silent
about knowing nothing, which is worse, because nothing in the logs said so.

Experiment 3 made that a permanent condition rather than an edge case. A router
trained across 40 suites scores 0.756 across suites and 0.600 within them: it
has learned which model suits which KIND of question, not which questions are
hard. On the kinds it has no signal for, it must say so.

Hence: when every model scores about the same, the router abstains and the
ladder decides. Same model chosen in most cases — but an honest reason, and a
number an operator can look at.
"""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import select

from switchboard.ledger.models import RequestLog
from switchboard.routing.artifact import RouterMetadata
from switchboard.routing.base import RoutingContext, RoutingDecision
from switchboard.routing.live import LiveRouter, shipped_router_path


class StubPredictor:
    """Returns whatever probabilities a test asks for."""

    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities
        self.models = list(probabilities)

    def predict_one(self, text: str) -> dict[str, float]:
        return dict(self.probabilities)


def router(probabilities: dict[str, float], min_spread: float = 0.08) -> LiveRouter:
    catalog_models = {name: name for name in probabilities}
    return LiveRouter(
        predictor=StubPredictor(probabilities),
        metadata=RouterMetadata(models=list(probabilities)),
        model_map=catalog_models,
        costs={name: index for index, name in enumerate(probabilities)},
        min_spread=min_spread,
    )


def ask(text: str = "fix this bug") -> RoutingContext:
    return RoutingContext(messages=[{"role": "user", "content": text}])


# --- The spread measurement -------------------------------------------------


def test_spread_is_the_gap_between_best_and_worst() -> None:
    assert LiveRouter._spread({"a": 0.9, "b": 0.2, "c": 0.5}) == pytest.approx(0.7)


def test_one_model_has_no_spread() -> None:
    """Nothing to compare, so nothing to be confident about."""
    assert LiveRouter._spread({"a": 0.9}) == 0.0


def test_no_models_does_not_divide_by_zero() -> None:
    assert LiveRouter._spread({}) == 0.0


# --- Abstaining -------------------------------------------------------------


def test_clustered_predictions_abstain() -> None:
    """THE C.4 case, reproduced exactly: 0.67-0.87 for everything."""
    decision = router({"cheap": 0.71, "mid": 0.74, "dear": 0.73}).choose(ask())
    assert decision.abstained
    assert "no usable discrimination" in decision.reason


def test_a_clear_difference_does_not_abstain() -> None:
    decision = router({"cheap": 0.20, "mid": 0.55, "dear": 0.88}).choose(ask())
    assert not decision.abstained


def test_abstaining_still_returns_a_usable_model() -> None:
    """A caller that ignores the flag must get a sane answer, not a crash."""
    decision = router({"cheap": 0.71, "mid": 0.72}).choose(ask())
    assert decision.model == "cheap"


def test_abstaining_picks_the_cheapest() -> None:
    """With no information, price is the only thing left to decide on."""
    decision = router({"dear": 0.71, "cheap": 0.72}).choose(ask())
    # `costs` follows insertion order in the helper, so "dear" is cheapest here.
    assert decision.model == "dear"


def test_the_reason_gives_the_number() -> None:
    """An operator has to be able to see how close the predictions were, not
    just be told they were close."""
    decision = router({"a": 0.70, "b": 0.72}).choose(ask())
    assert "0.02" in decision.reason


def test_the_probabilities_are_kept_for_debugging() -> None:
    decision = router({"a": 0.70, "b": 0.72}).choose(ask())
    assert decision.features["probabilities"] == {"a": 0.70, "b": 0.72}
    assert decision.features["spread"] == pytest.approx(0.02)


def test_abstention_can_be_switched_off() -> None:
    decision = router({"a": 0.70, "b": 0.72}, min_spread=0.0).choose(ask())
    assert not decision.abstained


def test_a_normal_decision_is_not_marked_as_abstained() -> None:
    """The flag defaults to False, so every existing strategy is unaffected."""
    assert not RoutingDecision(model="m", strategy="s").abstained


# --- Through the API --------------------------------------------------------


def _chat() -> dict:
    return {
        "model": "auto",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
    }


@pytest.fixture
def clustered(client, prices):
    """A router that has no opinion, wired to the real catalog."""
    ladder = client.app.state.ladder
    probabilities = dict.fromkeys(ladder.models, 0.72)
    client.app.state.router = LiveRouter(
        predictor=StubPredictor(probabilities),
        metadata=RouterMetadata(models=list(probabilities)),
        model_map={name: name for name in probabilities},
        costs={m: prices.models[m].output_per_mtok for m in probabilities},
        min_spread=0.08,
    )
    return client


def test_an_abstaining_router_hands_over_to_the_ladder(
    clustered, auth, provider, prices
) -> None:
    clustered.post("/v1/chat/completions", json=_chat(), headers=auth)
    assert provider.last_payload["model"] == prices.ladder[0]


def test_the_ledger_records_that_nothing_was_decided(
    clustered, auth, database
) -> None:
    """The whole point. C.4's failure was invisible because every row implied a
    judgement had been made."""
    clustered.post("/v1/chat/completions", json=_chat(), headers=auth)
    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert "no usable discrimination" in row.routing_reason
    assert "ladder chose" in row.routing_reason


def test_a_confident_router_still_decides(client, auth, provider, prices) -> None:
    """Abstention must not swallow a router that does have an opinion."""
    dear = prices.ladder[-1]
    cheap = prices.ladder[0]
    client.app.state.router = LiveRouter(
        predictor=StubPredictor({cheap: 0.10, dear: 0.95}),
        metadata=RouterMetadata(models=[cheap, dear]),
        model_map={cheap: cheap, dear: dear},
        costs={cheap: 1.0, dear: 20.0},
        min_spread=0.08,
    )
    client.post("/v1/chat/completions", json=_chat(), headers=auth)
    assert provider.last_payload["model"] == dear


def test_an_explicit_model_still_wins_over_abstention(
    clustered, auth, provider, prices
) -> None:
    """Naming a model always wins, whatever the router thinks."""
    named = prices.ladder[-1]
    clustered.post(
        "/v1/chat/completions", json=_chat() | {"model": named}, headers=auth
    )
    assert provider.last_payload["model"] == named


# --- Which router gets loaded -----------------------------------------------


def test_the_shipped_router_lives_inside_the_package() -> None:
    """It has to. A router bundled outside the package is not installed by pip,
    and a fresh install would silently have no router at all - which is the
    class of bug that hid routing being off inside Docker."""
    from switchboard import paths

    assert shipped_router_path().parent == paths.PACKAGE_ROOT


def test_a_missing_shipped_router_is_not_an_error(tmp_path, monkeypatch) -> None:
    """The repository does not commit one; it is built by
    `bench train-broad --save`. Absent, routing falls to the ladder."""
    from switchboard.catalog import ModelCatalog
    from switchboard.routing import live

    monkeypatch.setattr(live, "shipped_router_path", lambda: tmp_path / "nope.joblib")
    catalog = ModelCatalog.load()
    built = live.build_router(catalog, catalog.known_models(), path=None)
    assert built is None or built.enabled in (True, False)


def test_your_own_router_is_preferred_over_the_shipped_one(
    tmp_path, monkeypatch
) -> None:
    """A router trained on your traffic knows your prompts and your model
    names. A shipped one knows neither, and must never override it."""
    from switchboard.catalog import ModelCatalog
    from switchboard.routing import artifact as artifact_mod
    from switchboard.routing import live
    from switchboard.routing.features import FeatureExtractor
    from switchboard.routing.predictor import SuccessPredictor

    catalog = ModelCatalog.load()
    mine, shipped = tmp_path / "mine.joblib", tmp_path / "shipped.joblib"

    texts = [f"question number {i} about python lists" for i in range(40)]
    labels = np.array([i % 2 for i in range(40)])
    for path, name in ((mine, "yours"), (shipped, "theirs")):
        predictor = SuccessPredictor.fit(
            texts,
            {m: labels for m in catalog.ladder[:2]},
            FeatureExtractor(mode="surface"),
        )
        artifact_mod.save(path, predictor, RouterMetadata(source=name))

    monkeypatch.setattr(live, "shipped_router_path", lambda: shipped)
    built = live.build_router(catalog, catalog.known_models(), path=str(mine))
    assert built is not None
    assert built.metadata.source == "yours"


def test_the_coverage_table_survives_a_save_and_load(tmp_path) -> None:
    """It is the table that tells an operator which questions to trust the
    router on. Losing it in the sidecar would leave them guessing."""
    from switchboard.routing import artifact as artifact_mod

    metadata = RouterMetadata(
        source="public benchmarks",
        models=["a", "b"],
        coverage={"code": 0.76, "commonsense": 0.49},
        within_suite_auc=0.60,
    )
    path = tmp_path / "r.joblib"
    artifact_mod.save(path, object(), metadata)

    read = artifact_mod.read_metadata(path)
    assert read.coverage["code"] == pytest.approx(0.76)
    assert read.within_suite_auc == pytest.approx(0.60)


def test_older_metadata_without_coverage_still_loads(tmp_path) -> None:
    """Defaults, not required fields - an artifact from before Experiment 3
    must not become unreadable."""
    metadata = RouterMetadata(source="old")
    assert metadata.coverage == {}
    assert metadata.within_suite_auc != metadata.within_suite_auc  # NaN

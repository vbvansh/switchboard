"""The router that serves real traffic: artifacts, name mapping, limits."""

from __future__ import annotations

from pathlib import Path

import pytest

from switchboard.catalog import ModelCatalog
from switchboard.routing.artifact import (
    ARTIFACT_VERSION,
    ArtifactError,
    RouterMetadata,
    load,
    read_metadata,
    save,
)
from switchboard.routing.base import RoutingContext
from switchboard.routing.live import (
    HEADER_MAX_COST,
    HEADER_MAX_LATENCY,
    HEADER_MIN_QUALITY,
    LiveRouter,
    RequestLimits,
    build_model_map,
)


def ctx(text: str = "a question") -> RoutingContext:
    return RoutingContext(messages=[{"role": "user", "content": text}])


class FakePredictor:
    """Returns fixed probabilities, so routing logic is tested in isolation."""

    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities
        self.models = list(probabilities)

    def predict_one(self, text: str) -> dict[str, float]:
        return dict(self.probabilities)


# --- Artifacts --------------------------------------------------------------


def test_saving_and_loading_round_trips(tmp_path: Path) -> None:
    predictor = FakePredictor({"a": 0.9})
    metadata = RouterMetadata(source="test", models=["a"], n_train_questions=10)
    path = save(tmp_path / "router.joblib", predictor, metadata)

    loaded, loaded_metadata = load(path)
    assert loaded.models == ["a"]
    assert loaded_metadata.source == "test"
    assert loaded_metadata.n_train_questions == 10


def test_a_readable_sidecar_is_written(tmp_path: Path) -> None:
    """So a human - and `router info` - can see what a file holds without
    unpickling it."""
    path = save(
        tmp_path / "router.joblib",
        FakePredictor({"a": 0.5}),
        RouterMetadata(source="test", models=["a"]),
    )
    sidecar = path.with_suffix(".json")
    assert sidecar.exists()
    assert read_metadata(path).source == "test"


def test_a_missing_artifact_says_how_to_make_one(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="router train"):
        load(tmp_path / "absent.joblib")


def test_a_wrong_version_is_refused(tmp_path: Path) -> None:
    """Loading an artifact this build cannot interpret would route on nonsense."""
    path = tmp_path / "old.joblib"
    save(path, FakePredictor({"a": 0.5}), RouterMetadata(models=["a"]))

    import joblib

    blob = joblib.load(path)
    blob["metadata"]["version"] = ARTIFACT_VERSION + 99
    joblib.dump(blob, path)

    with pytest.raises(ArtifactError, match="different version"):
        load(path)


def test_a_file_that_is_not_an_artifact_is_refused(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "junk.joblib"
    joblib.dump({"something": "else"}, path)
    with pytest.raises(ArtifactError, match="not a Switchboard router"):
        load(path)


def test_metadata_describes_itself() -> None:
    described = RouterMetadata(
        source="xroutebench", benchmark="", models=["a", "b"], n_train_questions=5751
    ).describe()
    assert "xroutebench" in described
    assert "5,751" in described


# --- Name mapping -----------------------------------------------------------


CATALOG = {
    "baseline_model": "big",
    "ladder": ["small", "big"],
    "providers": [
        {
            "id": "local",
            "type": "openai-compatible",
            "base_url": "http://localhost:11434/v1",
            "enabled": True,
            "models": [
                {
                    "id": "small",
                    "tier": "T0",
                    "input_per_mtok": 1,
                    "output_per_mtok": 2,
                    "benchmark_alias": "tiny-hosted-model",
                    "typical_latency_s": 1.0,
                },
                {
                    "id": "big",
                    "tier": "T1",
                    "input_per_mtok": 10,
                    "output_per_mtok": 20,
                    "benchmark_alias": "large-hosted-model",
                    "typical_latency_s": 8.0,
                },
            ],
        }
    ],
}


@pytest.fixture
def catalog() -> ModelCatalog:
    return ModelCatalog.from_dict(CATALOG)


def test_aliases_map_benchmark_names_to_catalog_models(catalog: ModelCatalog) -> None:
    """A router trained on hosted models must be able to drive local ones."""
    mapping = build_model_map(
        catalog, ["tiny-hosted-model", "large-hosted-model"], ["small", "big"]
    )
    assert mapping == {"tiny-hosted-model": "small", "large-hosted-model": "big"}


def test_an_exact_id_match_also_works(catalog: ModelCatalog) -> None:
    assert build_model_map(catalog, ["small"], ["small", "big"]) == {"small": "small"}


def test_models_the_pool_cannot_serve_are_excluded(catalog: ModelCatalog) -> None:
    """Routing to a disabled provider would produce a confident 503."""
    mapping = build_model_map(
        catalog, ["tiny-hosted-model", "large-hosted-model"], ["small"]
    )
    assert mapping == {"tiny-hosted-model": "small"}


def test_unknown_benchmark_names_map_to_nothing(catalog: ModelCatalog) -> None:
    assert build_model_map(catalog, ["never-heard-of-it"], ["small", "big"]) == {}


# --- Request limits ---------------------------------------------------------


def test_limits_parse_from_headers() -> None:
    limits = RequestLimits.from_headers(
        {
            HEADER_MAX_LATENCY: "2.5",
            HEADER_MIN_QUALITY: "0.8",
            HEADER_MAX_COST: "0.01",
        }
    )
    assert limits.max_latency_s == 2.5
    assert limits.min_quality == 0.8
    assert limits.max_cost_usd == 0.01


def test_absent_headers_mean_no_limits() -> None:
    assert RequestLimits.from_headers({}).any is False


def test_a_malformed_limit_is_ignored_not_fatal() -> None:
    """These are hints on an otherwise valid request. Failing someone's chat
    completion over a bad header would be a poor trade."""
    limits = RequestLimits.from_headers({HEADER_MAX_LATENCY: "quickly please"})
    assert limits.max_latency_s is None


def test_a_nonsensical_limit_is_ignored() -> None:
    assert RequestLimits.from_headers({HEADER_MAX_LATENCY: "-5"}).max_latency_s is None


# --- The live router --------------------------------------------------------


def router(probabilities: dict[str, float], **kwargs) -> LiveRouter:
    return LiveRouter(
        predictor=FakePredictor(probabilities),
        metadata=RouterMetadata(models=list(probabilities)),
        model_map={"tiny-hosted-model": "small", "large-hosted-model": "big"},
        costs={"small": 0.001, "big": 0.100},
        latencies={"small": 1.0, "big": 8.0},
        **kwargs,
    )


def test_a_confident_cheap_model_wins() -> None:
    picked = router({"tiny-hosted-model": 0.9, "large-hosted-model": 0.95})
    assert picked.choose(ctx()).model == "small"


def test_an_unconfident_cheap_model_escalates() -> None:
    picked = router({"tiny-hosted-model": 0.1, "large-hosted-model": 0.95})
    assert picked.choose(ctx()).model == "big"


def test_a_latency_limit_blocks_the_slow_model() -> None:
    """The point of the SLA header: accuracy does not override the promise."""
    decision = router({"tiny-hosted-model": 0.1, "large-hosted-model": 0.99}).choose(
        ctx(), RequestLimits(max_latency_s=2.0)
    )
    assert decision.model == "small"


def test_a_cost_cap_blocks_the_expensive_model() -> None:
    decision = router({"tiny-hosted-model": 0.1, "large-hosted-model": 0.99}).choose(
        ctx(), RequestLimits(max_cost_usd=0.01)
    )
    assert decision.model == "small"


def test_a_per_request_quality_floor_overrides_the_default() -> None:
    strict = router({"tiny-hosted-model": 0.8, "large-hosted-model": 0.95})
    assert strict.choose(ctx()).model == "small"
    assert strict.choose(ctx(), RequestLimits(min_quality=0.9)).model == "big"


def test_an_impossible_limit_still_answers_and_says_so() -> None:
    """Dropping the request would be worse than serving it late."""
    decision = router({"tiny-hosted-model": 0.9, "large-hosted-model": 0.9}).choose(
        ctx(), RequestLimits(max_latency_s=0.001)
    )
    assert decision.model in {"small", "big"}
    assert "no model satisfies" in decision.reason


def test_when_nothing_clears_quality_the_best_chance_wins() -> None:
    """Falling back to the cheapest would abandon the hard requests."""
    picked = router({"tiny-hosted-model": 0.1, "large-hosted-model": 0.4})
    decision = picked.choose(ctx())
    assert decision.model == "big"
    assert "no model reached" in decision.reason


def test_every_decision_explains_itself() -> None:
    picked = router({"tiny-hosted-model": 0.9, "large-hosted-model": 0.9})
    assert picked.choose(ctx()).reason


def test_routing_needs_at_least_two_models() -> None:
    """One model is not a choice."""
    single = LiveRouter(
        predictor=FakePredictor({"a": 0.9}),
        metadata=RouterMetadata(models=["a"]),
        model_map={"a": "small"},
        costs={"small": 0.001},
    )
    assert single.enabled is False


def test_a_model_without_latency_history_fails_a_latency_limit() -> None:
    """Absence of evidence is not evidence of speed."""
    unknown = LiveRouter(
        predictor=FakePredictor({"tiny-hosted-model": 0.9, "large-hosted-model": 0.9}),
        metadata=RouterMetadata(models=["tiny-hosted-model", "large-hosted-model"]),
        model_map={"tiny-hosted-model": "small", "large-hosted-model": "big"},
        costs={"small": 0.001, "big": 0.100},
        latencies={},
    )
    decision = unknown.choose(ctx(), RequestLimits(max_latency_s=5.0))
    assert "no model satisfies" in decision.reason

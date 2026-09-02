"""The learned router: features, per-model success prediction, routing rule.

Embeddings are never used here. They need a 130 MB download and were measured
at roughly 0.5 texts per second on the development machine - unusable in a test
suite. None of the logic under test depends on them: surface and TF-IDF
features exercise splitting, training, the decision rule and the fallbacks.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.benchmarks import BenchmarkFrame, validate
from eval.benchmarks.features import (
    SURFACE_NAMES,
    FeatureExtractor,
    surface_features,
)
from eval.benchmarks.learned import (
    ConstantPredictor,
    LearnedRouter,
    routers_for_thresholds,
    split_questions,
    train_from_grid,
    training_report,
)
from eval.benchmarks.schema import Grid
from switchboard.routing import RoutingContext
from tests.test_benchmarks import rows


def ctx(text: str) -> RoutingContext:
    return RoutingContext(messages=[{"role": "user", "content": text}])


# --- Surface features ------------------------------------------------------


def test_surface_features_shape() -> None:
    assert surface_features("hello world").shape == (len(SURFACE_NAMES),)


def test_length_is_log_scaled() -> None:
    """A single enormous prompt must not dominate the feature scale.

    The property is compression: the SAME absolute increase in length moves
    the feature far less once the text is already long. 100 extra characters
    matter a lot at 100 characters and almost nothing at 10,000.
    """
    small_step = surface_features("a" * 200)[0] - surface_features("a" * 100)[0]
    large_step = surface_features("a" * 10_100)[0] - surface_features("a" * 10_000)[0]
    assert small_step > 10 * large_step


def test_code_is_detected() -> None:
    assert surface_features("def solve(x): return x")[SURFACE_NAMES.index("has_code")]
    assert not surface_features("what is the capital of Japan")[
        SURFACE_NAMES.index("has_code")
    ]


def test_maths_is_detected() -> None:
    index = SURFACE_NAMES.index("has_maths")
    assert surface_features("solve 3x + 2 = 11")[index]
    assert not surface_features("who wrote Hamlet")[index]


def test_empty_text_does_not_divide_by_zero() -> None:
    assert np.isfinite(surface_features("")).all()


# --- Extractor -------------------------------------------------------------

CORPUS = [
    "easy simple question about capitals",
    "hard tricky problem requiring proof",
    "what is the capital of Japan",
    "prove that the algorithm terminates",
]


@pytest.fixture
def extractor() -> FeatureExtractor:
    return FeatureExtractor(mode="surface").fit(CORPUS)


def test_surface_mode_width(extractor: FeatureExtractor) -> None:
    assert extractor.transform(["a", "bb"]).shape == (2, len(SURFACE_NAMES))


def test_transform_returns_one_row_per_text(extractor: FeatureExtractor) -> None:
    assert extractor.transform(["a", "bb", "ccc"]).shape[0] == 3


def test_transform_one_returns_a_single_row(extractor: FeatureExtractor) -> None:
    """A router must never fail on a prompt it has not seen before."""
    assert extractor.transform_one("never seen before").shape[0] == 1


def test_transform_before_fit_is_refused() -> None:
    """Transforming with an unfitted vocabulary would silently produce zeros."""
    with pytest.raises(RuntimeError, match="fit must be called"):
        FeatureExtractor(mode="surface").transform(["x"])


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        FeatureExtractor(mode="telepathy")


def test_tfidf_adds_vocabulary_columns() -> None:
    """TF-IDF is the default because embeddings are unusably slow on CPU."""
    tfidf = FeatureExtractor(mode="tfidf").fit(CORPUS)
    matrix = tfidf.transform(CORPUS)
    assert matrix.shape[0] == len(CORPUS)
    assert matrix.shape[1] > len(SURFACE_NAMES)


def test_tfidf_separates_different_topics() -> None:
    """The point of TF-IDF over surface stats: it sees the words."""
    tfidf = FeatureExtractor(mode="tfidf").fit(CORPUS)
    matrix = tfidf.transform(
        ["prove that the algorithm terminates", "what is the capital of Japan"]
    ).toarray()
    assert not np.allclose(matrix[0], matrix[1])


def test_tfidf_vocabulary_comes_only_from_fit() -> None:
    """Fitting on test text too would leak held-out information into training."""
    tfidf = FeatureExtractor(mode="tfidf").fit(CORPUS)
    before = len(tfidf._vectorizer.vocabulary_)
    tfidf.transform(["an entirely unrelated sentence about zebras"])
    assert len(tfidf._vectorizer.vocabulary_) == before


def test_describe_names_the_representation() -> None:
    assert "surface" in FeatureExtractor(mode="surface").fit(CORPUS).describe()
    assert "tfidf" in FeatureExtractor(mode="tfidf").fit(CORPUS).describe()


# --- Splitting -------------------------------------------------------------


def make_grid(n_questions: int = 100) -> Grid:
    """Cheap model right on even-numbered questions, big model right always."""
    records = []
    for i in range(n_questions):
        cheap_correct = 1.0 if i % 2 == 0 else 0.0
        records.append(("b", f"q{i:03d}", "cheap", cheap_correct, 0.01, 1.0))
        records.append(("b", f"q{i:03d}", "big", 1.0, 1.00, 2.0))
    return BenchmarkFrame(validate(rows(*records)), "test").grid()


def test_split_is_by_question_not_by_row() -> None:
    """The same question in both halves would leak the answer into training."""
    grid = make_grid(100)
    train, test = split_questions(grid, test_size=0.3, seed=0)
    assert len(train) + len(test) == grid.n_queries
    assert not set(train) & set(test)


def test_split_is_reproducible() -> None:
    grid = make_grid(100)
    first = split_questions(grid, seed=7)[1]
    second = split_questions(grid, seed=7)[1]
    assert list(first) == list(second)


def test_split_respects_test_size() -> None:
    train, test = split_questions(make_grid(100), test_size=0.25, seed=0)
    assert len(test) == 25


def test_subset_grids_are_disjoint_and_complete() -> None:
    grid = make_grid(50)
    train, test = split_questions(grid, test_size=0.4, seed=0)
    assert grid.subset(train).n_queries == 30
    assert grid.subset(test).n_queries == 20


# --- Constant predictor ----------------------------------------------------


def test_constant_predictor_returns_its_probability() -> None:
    """Stands in when a model was right, or wrong, on every training question."""
    probabilities = ConstantPredictor(0.8).predict_proba(np.zeros((3, 5)))
    assert probabilities.shape == (3, 2)
    assert probabilities[:, 1].tolist() == [0.8, 0.8, 0.8]


# --- Training --------------------------------------------------------------


@pytest.fixture(scope="module")
def trained():
    """Train where the label is perfectly predictable from the text.

    Questions containing "hard" are missed by the cheap model. A working
    pipeline must learn that; if it cannot, nothing downstream is meaningful.
    """
    records = []
    texts = {}
    for i in range(120):
        hard = i % 2 == 1
        key = ("b", f"q{i:03d}")
        texts[key] = "hard tricky problem" if hard else "easy simple question"
        records.append(("b", f"q{i:03d}", "cheap", 0.0 if hard else 1.0, 0.01, 1.0))
        records.append(("b", f"q{i:03d}", "big", 1.0, 1.00, 2.0))

    grid = BenchmarkFrame(validate(rows(*records)), "test").grid()
    predictor = train_from_grid(
        grid, texts, FeatureExtractor(mode="surface")
    )
    return predictor, grid, texts


def test_predictor_covers_every_model(trained) -> None:
    predictor, grid, _ = trained
    assert set(predictor.models) == set(grid.models)


def test_predictor_returns_probabilities(trained) -> None:
    predictor, _, _ = trained
    probabilities = predictor.predict(["easy simple question"])
    assert set(probabilities.columns) == set(predictor.models)
    assert ((probabilities >= 0) & (probabilities <= 1)).all().all()


def test_predictor_learns_a_real_signal(trained) -> None:
    """The load-bearing test: it must distinguish easy from hard."""
    predictor, _, _ = trained
    easy = predictor.predict_one("easy simple question")["cheap"]
    hard = predictor.predict_one("hard tricky problem")["cheap"]
    assert easy > hard


def test_a_model_that_never_fails_gets_a_constant(trained) -> None:
    """`big` is right on everything, so logistic regression cannot fit it."""
    predictor, _, _ = trained
    assert isinstance(predictor.classifiers["big"], ConstantPredictor)


def test_training_report_scores_each_model(trained) -> None:
    predictor, grid, texts = trained
    report = training_report(predictor, grid, texts)
    assert set(report.index) == set(grid.models)
    assert report.loc["cheap", "auc"] > 0.9  # perfectly predictable here
    assert np.isnan(report.loc["big", "auc"])  # single-class, no AUC defined


# --- The routing rule ------------------------------------------------------


def test_router_prefers_the_cheapest_model_that_clears_the_bar(trained) -> None:
    predictor, grid, _ = trained
    router = LearnedRouter(predictor, grid.mean_cost_per_model(), threshold=0.5)
    assert router.choose(ctx("easy simple question")).model == "cheap"


def test_router_escalates_when_the_cheap_model_is_unlikely(trained) -> None:
    predictor, grid, _ = trained
    router = LearnedRouter(predictor, grid.mean_cost_per_model(), threshold=0.5)
    assert router.choose(ctx("hard tricky problem")).model == "big"


def test_a_high_threshold_escalates_everything(trained) -> None:
    predictor, grid, _ = trained
    router = LearnedRouter(predictor, grid.mean_cost_per_model(), threshold=0.99)
    assert router.choose(ctx("easy simple question")).model == "big"


def test_a_zero_threshold_always_takes_the_cheapest(trained) -> None:
    predictor, grid, _ = trained
    router = LearnedRouter(predictor, grid.mean_cost_per_model(), threshold=0.0)
    assert router.choose(ctx("hard tricky problem")).model == "cheap"


def test_router_falls_back_to_the_most_likely_model(trained) -> None:
    """When nothing clears the bar, pick the best chance - not the cheapest.

    Giving up and going cheap would throw away exactly the hard questions
    routing exists to handle.
    """
    predictor, grid, _ = trained
    router = LearnedRouter(predictor, grid.mean_cost_per_model(), threshold=1.01)
    decision = router.choose(ctx("hard tricky problem"))
    assert decision.model in grid.models
    assert "no model reached" in decision.reason


def test_decisions_explain_themselves(trained) -> None:
    predictor, grid, _ = trained
    router = LearnedRouter(predictor, grid.mean_cost_per_model(), threshold=0.5)
    decision = router.choose(ctx("easy simple question"))
    assert decision.reason
    assert "probabilities" in decision.features


def test_router_only_sees_the_prompt(trained) -> None:
    """No access to recorded outcomes - that is what keeps scoring honest."""
    predictor, grid, _ = trained
    router = LearnedRouter(predictor, grid.mean_cost_per_model(), threshold=0.5)
    decision = router.choose(ctx("easy simple question"))
    assert set(decision.features["probabilities"]) == set(grid.models)


# --- Threshold sweep -------------------------------------------------------


def test_sweep_reuses_one_trained_predictor(trained) -> None:
    """The payoff of predicting probabilities: a curve without retraining."""
    predictor, grid, _ = trained
    routers = routers_for_thresholds(
        predictor, grid.mean_cost_per_model(), [0.2, 0.5, 0.8]
    )
    assert [r.threshold for r in routers] == [0.2, 0.5, 0.8]
    assert all(r.predictor is predictor for r in routers)


def test_sweep_names_are_distinct(trained) -> None:
    predictor, grid, _ = trained
    routers = routers_for_thresholds(
        predictor, grid.mean_cost_per_model(), [0.2, 0.5, 0.8]
    )
    assert len({r.name for r in routers}) == 3


def test_raising_the_threshold_never_makes_routing_cheaper(trained) -> None:
    """Monotonicity: more confidence demanded means more escalation."""
    predictor, grid, _ = trained
    costs = grid.mean_cost_per_model()
    prompts = ["easy simple question", "hard tricky problem"] * 10

    spend = []
    for threshold in (0.1, 0.5, 0.9):
        router = LearnedRouter(predictor, costs, threshold)
        spend.append(sum(costs[router.choose(ctx(p)).model] for p in prompts))

    assert spend == sorted(spend)

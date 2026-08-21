"""A router that learns which model to trust, instead of being told.

The framing that makes this work is not "predict the best model". It is:

    For each model, predict the probability it answers THIS question correctly.

That gives a probability per model rather than a single choice, and the routing
rule sits on top as a separate, tunable decision:

    Send it to the CHEAPEST model whose predicted success clears a threshold.

Two things fall out of that separation. Raising the threshold buys accuracy and
spends money; lowering it does the reverse - so one trained model produces an
entire cost/quality curve rather than a single point, and an operator can move
along it without retraining. And a quality threshold, which the project needs
anyway, is simply that number made explicit.

Everything here is trained on one set of questions and scored on a different
set. A model that has seen a question can memorise its answer, and a router
scored on its training data measures memory rather than judgement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from eval.benchmarks.features import FeatureExtractor
from eval.benchmarks.schema import Grid
from switchboard.routing.base import RoutingContext, RoutingDecision, RoutingStrategy

logger = logging.getLogger(__name__)

#: A model is treated as having answered correctly above this score. Most
#: benchmarks here are already 0/1; the few graded ones (F1 on squad) need a
#: line drawn somewhere, and half credit is the natural place.
CORRECT_THRESHOLD = 0.5

DEFAULT_TEST_SIZE = 0.3
DEFAULT_SEED = 0


def split_questions(
    grid: Grid, test_size: float = DEFAULT_TEST_SIZE, seed: int = DEFAULT_SEED
) -> tuple[pd.Index, pd.Index]:
    """Split by QUESTION, never by row.

    Splitting rows would put the same question in both halves - answered by
    model A in training and model B in test - and the router would be scored on
    questions it had already been shown. The result would look excellent and
    mean nothing.
    """
    from sklearn.model_selection import train_test_split

    questions = grid.correct.index
    train, test = train_test_split(
        np.arange(len(questions)), test_size=test_size, random_state=seed
    )
    return questions[np.sort(train)], questions[np.sort(test)]


class _ConstantPredictor:
    """Stands in when a model was right (or wrong) on every training question.

    Logistic regression cannot fit a single-class target. Rather than dropping
    the model - which would quietly remove it from routing - its probability is
    fixed at what the training data showed.
    """

    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        column = np.full((len(features), 1), self.probability)
        return np.hstack([1.0 - column, column])


@dataclass
class SuccessPredictor:
    """One calibrated classifier per model: will it get this question right?"""

    models: list[str]
    classifiers: dict[str, object]
    extractor: FeatureExtractor

    @classmethod
    def train(
        cls,
        grid: Grid,
        texts: dict[tuple[str, str], str],
        extractor: FeatureExtractor | None = None,
        seed: int = DEFAULT_SEED,
    ) -> SuccessPredictor:
        from sklearn.linear_model import LogisticRegression

        extractor = extractor or FeatureExtractor()
        question_texts = [texts.get(key, "") for key in grid.correct.index]

        # Fit on the TRAINING questions only. Fitting the vocabulary on the
        # test set too would leak information about held-out questions.
        scaled = extractor.fit(question_texts).transform(question_texts)

        classifiers: dict[str, object] = {}
        for model in grid.models:
            labels = (grid.correct[model] > CORRECT_THRESHOLD).to_numpy().astype(int)

            if labels.min() == labels.max():
                classifiers[model] = _ConstantPredictor(labels.mean())
                logger.info(
                    "%s was %s on every training question; using a constant.",
                    model,
                    "correct" if labels[0] else "wrong",
                )
                continue

            # `balanced` matters: a model right 90% of the time would otherwise
            # be best served by a classifier that always predicts "correct",
            # which carries no information about which 10% it fails.
            classifiers[model] = LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed
            ).fit(scaled, labels)

        return cls(list(grid.models), classifiers, extractor)

    def predict(self, texts: list[str]) -> pd.DataFrame:
        """P(correct) for every model, one row per text."""
        scaled = self.extractor.transform(texts)
        return pd.DataFrame(
            {
                model: self.classifiers[model].predict_proba(scaled)[:, 1]
                for model in self.models
            },
            index=range(len(texts)),
        )

    def predict_one(self, text: str) -> dict[str, float]:
        scaled = self.extractor.transform_one(text)
        return {
            model: float(self.classifiers[model].predict_proba(scaled)[0, 1])
            for model in self.models
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Probabilities for many texts in one pass.

        Routing one question at a time means one tiny sklearn call per model
        per question - tens of thousands of calls whose Python overhead dwarfs
        the arithmetic. Batching is what makes a threshold sweep quick.
        """
        frame = self.predict(texts)
        return frame.to_dict("records")


class LearnedRouter(RoutingStrategy):
    """Cheapest model whose predicted success clears the threshold.

    Sees only the question text, exactly as it would in production. It has no
    access to recorded outcomes - that is what keeps the evaluation honest.
    """

    def __init__(
        self,
        predictor: SuccessPredictor,
        costs: pd.Series,
        threshold: float = 0.5,
        name: str | None = None,
    ) -> None:
        self.predictor = predictor
        # Cheapest first, so the first model clearing the bar is also the
        # cheapest one that does.
        self.costs = costs.sort_values()
        self.threshold = threshold
        self.name = name or f"learned@{threshold:.2f}"
        self._probabilities: dict[str, dict[str, float]] = {}

    def warm(self, texts: list[str]) -> None:
        """Precompute probabilities for a batch of prompts.

        `choose` is called once per request, but predicting one row at a time
        costs far more in call overhead than in arithmetic. A replay warms the
        whole set first and each decision becomes a dictionary lookup.
        """
        unseen = [t for t in dict.fromkeys(texts) if t not in self._probabilities]
        if not unseen:
            return
        for text, row in zip(
            unseen, self.predictor.predict_batch(unseen), strict=True
        ):
            self._probabilities[text] = row

    def choose(self, context: RoutingContext) -> RoutingDecision:
        text = context.prompt_text
        probabilities = self._probabilities.get(text)
        if probabilities is None:
            probabilities = self.predictor.predict_one(text)

        for model in self.costs.index:
            if probabilities.get(model, 0.0) >= self.threshold:
                return RoutingDecision(
                    model=model,
                    strategy=self.name,
                    reason=(
                        f"cheapest model clearing p>={self.threshold:.2f} "
                        f"(predicted {probabilities[model]:.2f})"
                    ),
                    features={"probabilities": probabilities},
                )

        # Nothing cleared the bar. Fall back to whichever model is most likely
        # to succeed - giving up and sending it to the cheapest would throw
        # away exactly the hard questions routing exists to handle.
        best = max(probabilities, key=probabilities.get)
        return RoutingDecision(
            model=best,
            strategy=self.name,
            reason=(
                f"no model reached p>={self.threshold:.2f}; "
                f"best available was {probabilities[best]:.2f}"
            ),
            features={"probabilities": probabilities},
        )


def build_router(
    train_grid: Grid,
    texts: dict[tuple[str, str], str],
    threshold: float = 0.5,
    extractor: FeatureExtractor | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[LearnedRouter, SuccessPredictor]:
    predictor = SuccessPredictor.train(train_grid, texts, extractor, seed=seed)
    router = LearnedRouter(predictor, train_grid.mean_cost_per_model(), threshold)
    return router, predictor


def routers_for_thresholds(
    predictor: SuccessPredictor, costs: pd.Series, thresholds: list[float]
) -> list[LearnedRouter]:
    """One trained predictor, many operating points.

    Sweeping the threshold traces the cost/quality curve without retraining -
    which is the practical payoff of predicting probabilities rather than
    picking a model outright.
    """
    return [LearnedRouter(predictor, costs, t) for t in thresholds]


def training_report(
    predictor: SuccessPredictor, grid: Grid, texts: dict
) -> pd.DataFrame:
    """How well each per-model classifier actually predicts success.

    A router can only be as good as these. If every model's AUC sits near 0.5,
    the features carry no signal about difficulty and no decision rule on top
    will help.
    """
    from sklearn.metrics import roc_auc_score

    question_texts = [texts.get(key, "") for key in grid.correct.index]
    probabilities = predictor.predict(question_texts)

    rows = []
    for model in grid.models:
        labels = (grid.correct[model] > CORRECT_THRESHOLD).to_numpy().astype(int)
        auc = (
            roc_auc_score(labels, probabilities[model])
            if labels.min() != labels.max()
            else float("nan")
        )
        rows.append(
            {
                "model": model,
                "base_rate": labels.mean(),
                "auc": auc,
                "mean_predicted": probabilities[model].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("auc", ascending=False)

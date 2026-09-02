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

import numpy as np
import pandas as pd

from eval.benchmarks.schema import Grid
from switchboard.routing.base import RoutingContext, RoutingDecision, RoutingStrategy
from switchboard.routing.features import FeatureExtractor
from switchboard.routing.predictor import (
    CORRECT_THRESHOLD,
    ConstantPredictor,
    SuccessPredictor,
)

logger = logging.getLogger(__name__)

DEFAULT_TEST_SIZE = 0.3
DEFAULT_SEED = 0

__all__ = [
    "CORRECT_THRESHOLD",
    "ConstantPredictor",
    "LearnedRouter",
    "SuccessPredictor",
    "build_router",
    "routers_for_thresholds",
    "split_questions",
    "train_from_grid",
    "training_report",
]


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


def train_from_grid(
    grid: Grid,
    texts: dict[tuple[str, str], str],
    extractor: FeatureExtractor | None = None,
    seed: int = DEFAULT_SEED,
) -> SuccessPredictor:
    """Turn a benchmark grid into the shape `SuccessPredictor.fit` expects.

    The grid is dense - every question answered by every model - so each model
    gets a label for every question. Live traffic is the opposite: one model
    per request. Both end up as {model: 0/1 array} over a shared list of texts,
    which is why the trainer itself does not need to know where they came from.
    """
    question_texts = [texts.get(key, "") for key in grid.correct.index]
    labels = {
        model: (grid.correct[model] > CORRECT_THRESHOLD).to_numpy().astype(int)
        for model in grid.models
    }
    return SuccessPredictor.fit(question_texts, labels, extractor, seed=seed)


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
    predictor = train_from_grid(train_grid, texts, extractor, seed=seed)
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

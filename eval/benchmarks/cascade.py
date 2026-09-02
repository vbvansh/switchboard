"""Cascades: try a cheap model first, check the answer, escalate if unconvinced.

The learned router in `learned.py` decides BEFORE calling anything - it guesses
from the question alone. A cascade decides AFTER: it pays for a cheap call,
looks at what came back, and only then chooses whether to pay for more.

That extra information is the whole point, and it is not free. A cascade that
escalates has paid for both calls, so the accounting here charges for every
call made, not just the last one. Getting that wrong would make cascades look
better than they are.

Two ways to judge the cheap answer, neither of which may look at whether it was
actually correct - that would be the oracle cheating:

* **Agreement** - ask two cheap models and compare their answers. Two models
  independently reaching the same answer is evidence it is right; disagreement
  is evidence something is hard. No training required.

* **Learned verifier** - a classifier trained to predict "was the cheap model
  right?" from the question plus what the cheap model actually did: how long
  its answer was, and whether a second model agreed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from eval.benchmarks.features import FeatureExtractor
from eval.benchmarks.learned import CORRECT_THRESHOLD, DEFAULT_SEED
from eval.benchmarks.schema import Grid
from switchboard.routing.predictor import ConstantPredictor

logger = logging.getLogger(__name__)


def has_answers(grid: Grid) -> bool:
    """Does this grid record parsed answers to compare between models?"""
    if grid.prediction is None:
        return False
    return bool((grid.prediction.fillna("") != "").any().any())


def cheapest_models(grid: Grid, n: int = 2) -> list[str]:
    return list(grid.cost.sum().sort_values().index[:n])


def strongest_model(grid: Grid) -> str:
    return grid.model_accuracy().index[0]


# --- Agreement cascade ------------------------------------------------------


def agreement_paths(
    grid: Grid,
    first: str | None = None,
    second: str | None = None,
    escalate_to: str | None = None,
) -> pd.Series:
    """Two cheap models vote; disagreement escalates.

    Returns one path per question - the ordered list of models actually called.

    When the two agree, their answers are identical, so their correctness is
    identical too and either can be reported as the result.
    """
    if not has_answers(grid):
        raise ValueError(
            "This source records no parsed answers, so models cannot be "
            "compared. Agreement cascades need `prediction`."
        )

    cheap = cheapest_models(grid, 2)
    first = first or cheap[0]
    second = second or (cheap[1] if len(cheap) > 1 else cheap[0])
    escalate_to = escalate_to or strongest_model(grid)

    answers = grid.prediction.fillna("")
    # An empty answer is unknown, not agreement - two blanks must not count as
    # two models confirming each other.
    agree = (answers[first] == answers[second]) & (answers[first] != "")

    paths = []
    for agreed in agree:
        if agreed:
            paths.append((first, second))
        else:
            paths.append((first, second, escalate_to))
    return pd.Series(paths, index=grid.correct.index)


# --- Learned verifier cascade -----------------------------------------------


def observation_features(grid: Grid, model: str, peer: str | None = None) -> np.ndarray:
    """What we can see after paying for `model`, before grading it.

    Deliberately excludes the correctness label. These are things a live
    deployment genuinely observes: how long the answer was, how that compares
    to this model's usual length, and whether a second model agreed.
    """
    tokens = (
        grid.output_tokens[model]
        if grid.output_tokens is not None
        else pd.Series(0.0, index=grid.correct.index)
    )
    tokens = tokens.fillna(0.0)
    typical = tokens.median() or 1.0

    columns = [
        np.log1p(tokens.to_numpy()),
        (tokens / typical).to_numpy(),
    ]

    if peer is not None and has_answers(grid):
        answers = grid.prediction.fillna("")
        agrees = ((answers[model] == answers[peer]) & (answers[model] != "")).to_numpy()
        columns.append(agrees.astype(float))
    else:
        columns.append(np.zeros(len(tokens)))

    return np.column_stack(columns).astype(np.float32)


def _stack(question_features, observations: np.ndarray):
    """Join question features to post-call observations, sparse or dense."""
    from scipy.sparse import csr_matrix, hstack, issparse

    if issparse(question_features):
        return hstack([question_features, csr_matrix(observations)]).tocsr()
    return np.hstack([question_features, observations])


class VerifierCascade:
    """Call the cheap model, predict whether it was right, escalate if not.

    Strictly better informed than predicting from the question alone - it has
    seen the answer's shape and a second opinion. Whether that is worth the
    extra call is exactly what the replay measures.
    """

    def __init__(
        self,
        first: str,
        peer: str | None,
        escalate_to: str,
        classifier,
        extractor: FeatureExtractor,
        threshold: float = 0.5,
        name: str | None = None,
    ) -> None:
        self.first = first
        self.peer = peer
        self.escalate_to = escalate_to
        self.classifier = classifier
        self.extractor = extractor
        self.threshold = threshold
        self.name = name or f"cascade@{threshold:.2f}"

    @classmethod
    def train(
        cls,
        grid: Grid,
        texts: dict[tuple[str, str], str],
        extractor: FeatureExtractor,
        threshold: float = 0.5,
        seed: int = DEFAULT_SEED,
    ) -> VerifierCascade:
        from sklearn.linear_model import LogisticRegression

        cheap = cheapest_models(grid, 2)
        first = cheap[0]
        peer = cheap[1] if len(cheap) > 1 and has_answers(grid) else None
        escalate_to = strongest_model(grid)

        question_texts = [texts.get(key, "") for key in grid.correct.index]
        # The extractor is fitted here, on training questions only.
        features = extractor.fit(question_texts).transform(question_texts)
        combined = _stack(features, observation_features(grid, first, peer))

        labels = (grid.correct[first] > CORRECT_THRESHOLD).to_numpy().astype(int)
        if labels.min() == labels.max():
            classifier = ConstantPredictor(labels.mean())
            logger.info(
                "Cheap model %s was uniform on training; using a constant.", first
            )
        else:
            classifier = LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed
            ).fit(combined, labels)

        return cls(first, peer, escalate_to, classifier, extractor, threshold)

    def confidence(self, grid: Grid, texts: dict) -> np.ndarray:
        question_texts = [texts.get(key, "") for key in grid.correct.index]
        features = self.extractor.transform(question_texts)
        combined = _stack(features, observation_features(grid, self.first, self.peer))
        return self.classifier.predict_proba(combined)[:, 1]

    def paths(self, grid: Grid, texts: dict) -> pd.Series:
        confident = self.confidence(grid, texts) >= self.threshold
        paths = [
            (self.first,) if ok else (self.first, self.escalate_to) for ok in confident
        ]
        return pd.Series(paths, index=grid.correct.index)


def cascades_for_thresholds(
    template: VerifierCascade, thresholds: list[float]
) -> list[VerifierCascade]:
    """One trained verifier, many escalation points - same trick as the router."""
    return [
        VerifierCascade(
            template.first,
            template.peer,
            template.escalate_to,
            template.classifier,
            template.extractor,
            threshold,
        )
        for threshold in thresholds
    ]

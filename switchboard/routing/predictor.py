"""One classifier per model: will THIS model get THIS question right?

The framing matters more than the algorithm. It is not "predict the best
model" - that would be a single choice with no dial on it. It is a probability
per model, with the routing rule sitting on top as a separate decision:

    send it to the CHEAPEST model whose predicted success clears a threshold

Two things fall out of that separation. Raising the threshold buys accuracy and
spends money, lowering it does the reverse - so one trained predictor produces
a whole cost/quality curve rather than a single point. And the "minimum
quality" limit the product needs anyway is simply that number made explicit.

WHY THIS LIVES IN `switchboard/` AND NOT IN `eval/`.

It used to live in `eval/`, with the rest of the research code. That was a bug,
and an invisible one.

A trained router is a joblib pickle, and a pickle records the module each class
came from. While these classes lived in `eval.benchmarks.learned`, every
artifact pointed there - and the Docker image deliberately does not copy
`eval/`, because it drags in 500 MB of research tooling a server never runs.

So inside a container the artifact could not unpickle. `build_router` caught
the failure exactly as designed, routing switched itself off, and `/health`
reported "no router artifact loaded" with nothing pointing at the real cause.
Anyone who deployed with a trained router was running without routing and had
no way to find out.

Anything a trained artifact refers to has to ship with the server. That is the
rule this module exists to keep.

`pandas` is imported lazily for the same reason: it is a research dependency,
not a runtime one, and the serving path must not need it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from switchboard.routing.features import FeatureExtractor

logger = logging.getLogger(__name__)

#: A model counts as having answered correctly above this score. Most
#: benchmarks are already 0/1; the graded ones (F1 on squad) need a line drawn
#: somewhere, and half credit is the natural place. Live feedback is always
#: 0 or 1, so this does not apply to it.
CORRECT_THRESHOLD = 0.5

DEFAULT_SEED = 0


class ConstantPredictor:
    """Stands in when a model was right (or wrong) on every training example.

    Logistic regression cannot fit a single-class target. Dropping the model
    instead would quietly remove it from routing, so its probability is fixed
    at whatever the training data showed.

    Deliberately NOT used by the live trainer. On benchmark data a single-class
    model is a real finding over thousands of graded questions. On thirty
    pieces of user feedback that all happen to be positive, a constant 1.0
    would make that model win every routing decision forever on the strength of
    thirty examples. `switchboard/training.py` refuses those models instead.
    """

    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        column = np.full((len(features), 1), self.probability)
        return np.hstack([1.0 - column, column])


@dataclass
class SuccessPredictor:
    """A calibrated classifier per model, plus the extractor they share."""

    models: list[str]
    classifiers: dict[str, Any]
    extractor: FeatureExtractor

    @classmethod
    def fit(
        cls,
        texts: list[str],
        labels: dict[str, np.ndarray],
        extractor: FeatureExtractor | None = None,
        seed: int = DEFAULT_SEED,
        allow_constant: bool = True,
    ) -> SuccessPredictor:
        """Train from texts and a 0/1 label array per model.

        `texts` is the shared corpus the extractor learns its vocabulary from.
        `labels` maps a model name to an array the same length as `texts`,
        holding 1 where that model answered correctly.

        Deliberately knows nothing about benchmarks or ledgers. Benchmark
        training builds these inputs from a grid; live training builds them
        from recorded feedback. Both arrive here in the same shape.
        """
        from sklearn.linear_model import LogisticRegression

        extractor = extractor or FeatureExtractor()

        # Fit on the TRAINING texts only. Fitting the vocabulary on the test
        # set as well would leak information about held-out questions and
        # inflate every score that follows.
        scaled = extractor.fit(texts).transform(texts)

        classifiers: dict[str, Any] = {}
        trained: list[str] = []
        for model, raw in labels.items():
            values = np.asarray(raw).astype(int)

            if values.min() == values.max():
                if not allow_constant:
                    logger.info(
                        "%s has only one outcome in its training data; skipped.",
                        model,
                    )
                    continue
                classifiers[model] = ConstantPredictor(values.mean())
                trained.append(model)
                continue

            # `balanced` matters: a model right 90% of the time would otherwise
            # be best served by a classifier that always answers "correct",
            # which carries no information about which 10% it fails.
            classifiers[model] = LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed
            ).fit(scaled, values)
            trained.append(model)

        return cls(trained, classifiers, extractor)

    @classmethod
    def fit_per_model(
        cls,
        corpus: list[str],
        per_model: dict[str, tuple[list[str], np.ndarray]],
        extractor: FeatureExtractor | None = None,
        seed: int = DEFAULT_SEED,
    ) -> SuccessPredictor:
        """Train where each model has its OWN examples rather than shared ones.

        This is the shape live traffic arrives in. A benchmark is dense - every
        question answered by every model - so one label array per model over
        one shared list of questions works. Real traffic is sparse: each
        request was answered by exactly one model, and what the others would
        have said is unknown.

        The distinction matters more than it looks. Flattening sparse data into
        the dense shape means writing a 0 wherever a model was not asked, which
        would teach every classifier that every question another model handled
        was one it personally got wrong.

        `corpus` is every prompt, used only to fit the shared vocabulary - the
        text representation should know the whole workload even where labels
        are thin. Each classifier then sees only its own rows.

        No constant fallback here: a model with one-sided data is dropped, and
        the caller is expected to have refused it earlier. See
        `switchboard/training.py` for why thirty positive ratings must not
        become a classifier that answers "yes" to everything.
        """
        from sklearn.linear_model import LogisticRegression

        extractor = extractor or FeatureExtractor()
        extractor.fit(corpus)

        classifiers: dict[str, Any] = {}
        trained: list[str] = []
        for model, (texts, raw) in per_model.items():
            values = np.asarray(raw).astype(int)
            if len(texts) == 0 or values.min() == values.max():
                logger.info("%s has one-sided or empty training data; skipped.", model)
                continue
            classifiers[model] = LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed
            ).fit(extractor.transform(texts), values)
            trained.append(model)

        return cls(trained, classifiers, extractor)

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
        if not texts:
            return []
        scaled = self.extractor.transform(texts)
        columns = {
            model: self.classifiers[model].predict_proba(scaled)[:, 1]
            for model in self.models
        }
        return [
            {model: float(values[index]) for model, values in columns.items()}
            for index in range(len(texts))
        ]

    def predict(self, texts: list[str]):
        """The same thing as a DataFrame, for the research code.

        pandas is imported here rather than at the top of the module because it
        is a research dependency. The serving path uses `predict_one` and must
        not require it.
        """
        import pandas as pd

        scaled = self.extractor.transform(texts)
        return pd.DataFrame(
            {
                model: self.classifiers[model].predict_proba(scaled)[:, 1]
                for model in self.models
            },
            index=range(len(texts)),
        )

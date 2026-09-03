"""How hard is this question, and can that be predicted from its text alone?

THE IDEA BEING TESTED HERE.

The router today asks a question that is welded to one model: "will qwen2.5:3b
get this right?" Answering it needs examples of that model, on your traffic,
graded. No traffic, no router - which is the cold start.

Split the question in two and one half comes free:

    how hard is this question?      <- a property of the QUESTION
    how capable is this model?      <- a property of the MODEL

Difficulty does not mention any model. A question most models fail is hard,
whether you own Claude or a laptop Llama. So it can be measured once, here, on
public data, and shipped to everyone - and a router can work on request number
one instead of after a month of collecting.

This module answers whether that is actually true, before anything is built on
it.

HOW DIFFICULTY IS MEASURED. It is the same benchmark table read the other way:

                     gpt-5   claude  gemini  llama
    question 1       right   right   right   right    -> 0 of 4 failed -> 0.00
    question 2       right   wrong   right   wrong    -> 2 of 4 failed -> 0.50
    question 3       wrong   wrong   wrong   wrong    -> 4 of 4 failed -> 1.00

Read across a row for difficulty; read down a column for capability. No
modelling, just arithmetic over data already on disk.

WHY THIS WORKS WHERE THE GRID DOES NOT. `Grid` needs a COMPLETE rectangle -
every model must have answered every question, and Phase C found that combining
suites often leaves an empty one, because different models ran different
splits. Difficulty needs no such thing. "Of the models that tried this
question, what fraction failed?" is well defined at three models, and it does
not care that the next question had a different three. That is what makes it
possible to use all forty suites instead of one.

THE TEST THAT MATTERS. Not "can we predict difficulty" - that would be easy and
meaningless, because a model that has seen GPQA questions can pattern-match
GPQA. The real question is whether it transfers:

    train on some suites, then predict difficulty on suites never seen at all

That is the closest offline stand-in for "a user sends us a prompt unlike
anything in our training data", which is exactly where the current router
fails.

AND IT IS SCORED AGAINST DOING NOTHING. Always predicting the average
difficulty is the baseline. Anything that cannot beat that constant carries no
signal, however good its correlation looks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: A question needs at least this many models to have attempted it before its
#: difficulty means anything. One model failing tells you nothing about the
#: question; it might just be that model.
MIN_MODELS = 3

#: Prompt-length buckets, in characters. Chosen around the shapes that actually
#: occur: a chat message, a normal question, an exam item with options, and a
#: pasted document or stack trace.
LENGTH_BUCKETS = (
    ("under 100", 0, 100),
    ("100-500", 100, 500),
    ("500-2000", 500, 2000),
    ("2000-8000", 2000, 8000),
    ("over 8000", 8000, 10**9),
)


def per_question(frame: pd.DataFrame, min_models: int = MIN_MODELS) -> pd.DataFrame:
    """Difficulty for every question: the fraction of models that got it wrong.

    Returns one row per (benchmark, query_id) with `difficulty`, `n_models` and
    `mean_cost`. Questions attempted by fewer than `min_models` models are
    dropped - their difficulty would be noise dressed up as a number.
    """
    grouped = (
        frame.groupby(["benchmark", "query_id"], observed=True)
        .agg(
            n_models=("model", "nunique"),
            mean_correct=("correct", "mean"),
            mean_cost=("cost_usd", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["n_models"] >= min_models].copy()
    grouped["difficulty"] = 1.0 - grouped["mean_correct"]
    return grouped


def attach_text(questions: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """Join the question text on, dropping anything with no text to learn from."""
    merged = questions.merge(
        queries[["benchmark", "query_id", "query"]],
        on=["benchmark", "query_id"],
        how="left",
    )
    merged["query"] = merged["query"].fillna("")
    merged = merged[merged["query"].str.len() > 0].copy()
    merged["length"] = merged["query"].str.len()
    return merged


def split_by_suite(
    questions: pd.DataFrame, holdout: int = 5, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Hold back WHOLE SUITES, not a random sample of questions.

    This is the point of the whole experiment. Splitting questions at random
    would leave GPQA items in both halves, and a model that has seen GPQA can
    pattern-match GPQA - a score that says nothing about a prompt from outside
    the benchmark world.

    Holding back entire suites asks the question that matters: does this
    transfer to a KIND of question it has never seen?
    """
    suites = sorted(questions["benchmark"].unique())
    if len(suites) <= holdout:
        raise ValueError(
            f"Only {len(suites)} suite(s) available; need more than the "
            f"{holdout} being held out. Build more sources, or lower --holdout."
        )

    rng = np.random.default_rng(seed)
    held = sorted(rng.permutation(suites)[:holdout].tolist())

    train = questions[~questions["benchmark"].isin(held)]
    test = questions[questions["benchmark"].isin(held)]
    return train, test, held


@dataclass
class Score:
    """How well predicted difficulty matched real difficulty."""

    n: int = 0
    spearman: float = float("nan")
    mae: float = float("nan")
    #: Error of always predicting the training mean. The bar to beat.
    baseline_mae: float = float("nan")

    @property
    def closer_on_average(self) -> bool:
        """Lower absolute error than always guessing the mean.

        On its own this proves almost nothing, and the first version of this
        report treated it as the headline, which was wrong. A model can be
        closer on average purely because a suite's mean difficulty happens to
        sit nearer its prediction than the global mean does - while ranking the
        questions inside that suite completely backwards.
        """
        return self.mae == self.mae and self.mae < self.baseline_mae

    @property
    def ranks_correctly(self) -> bool:
        """Does it actually order questions by hardness?

        THE property routing needs. A router does not care about a question's
        absolute difficulty score; it cares whether THIS question is harder
        than THAT one, so it can send the hard ones somewhere better.
        """
        return self.spearman == self.spearman and self.spearman > 0.1

    @property
    def useful(self) -> bool:
        return self.ranks_correctly and self.closer_on_average

    @property
    def improvement_pct(self) -> float:
        if not (self.baseline_mae and self.baseline_mae == self.baseline_mae):
            return float("nan")
        return 100.0 * (self.baseline_mae - self.mae) / self.baseline_mae


def evaluate(
    predicted: np.ndarray, actual: np.ndarray, constant: float
) -> Score:
    """Score predictions against a constant-guess baseline.

    Spearman rather than Pearson because routing only needs the ORDER to be
    right - is this question harder than that one - not the absolute value.
    """
    from scipy.stats import spearmanr

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) < 3:
        return Score(n=len(actual))

    # Undefined when everything has identical difficulty, which happens in a
    # suite where every model scored the same on every question.
    correlation = float("nan")
    if actual.std() > 0 and predicted.std() > 0:
        correlation = float(spearmanr(predicted, actual).statistic)

    return Score(
        n=len(actual),
        spearman=correlation,
        mae=float(np.abs(predicted - actual).mean()),
        baseline_mae=float(np.abs(constant - actual).mean()),
    )


@dataclass
class Report:
    """Everything the experiment measured."""

    n_questions: int = 0
    n_suites: int = 0
    mean_difficulty: float = float("nan")
    held_out_suites: list[str] = field(default_factory=list)
    overall: Score = field(default_factory=Score)
    by_length: dict[str, Score] = field(default_factory=dict)
    by_suite: dict[str, Score] = field(default_factory=dict)
    features: str = ""

    @property
    def within_suite_spearman(self) -> float:
        """Median correlation INSIDE each held-out suite.

        This is the number that decides whether a cold-start router is
        possible, and the first version of this report failed to compute it.

        The overall figure mixes every held-out suite together, so a model that
        has merely learned "AIME questions are hard, ARC questions are easy"
        scores well on it - by recognising which suite a question came from,
        which is a completely different skill from telling one AIME question
        from another.

        A real user's traffic IS one suite. Telling their workload apart from
        somebody else's is worth nothing; telling their easy requests from
        their hard ones is the entire job.
        """
        values = [
            score.spearman
            for score in self.by_suite.values()
            if score.spearman == score.spearman
        ]
        if not values:
            return float("nan")
        return float(np.median(values))

    def verdict(self) -> str:
        """Should a cold-start router be built on this?

        Judged on WITHIN-suite ranking, not on the overall figure. The overall
        figure rewards recognising which suite a question came from, and a real
        user only ever has one.
        """
        within = self.within_suite_spearman
        if within != within:
            return "NOT SCORABLE. No held-out suite had enough variety to score."

        if within < 0.10:
            return (
                f"NO USABLE SIGNAL. Within a suite the ranking correlation is "
                f"{within:.3f} - effectively zero. The overall figure of "
                f"{self.overall.spearman:.3f} comes from telling SUITES apart, "
                "not questions. A real user's traffic is one suite, so that "
                "skill is worth nothing to them. Do not build a shipped prior "
                "on this."
            )
        if within >= 0.35:
            return (
                f"TRANSFERS. Within-suite correlation {within:.3f} on suites "
                "never seen. A shipped prior is worth building, and this is the "
                "number it must beat."
            )
        return (
            f"WEAK. Within-suite correlation {within:.3f}. Real but small - "
            "worth building only WITH a confidence estimate, so the router can "
            "fall back to a tier policy on prompts it cannot read."
        )


def run(
    questions: pd.DataFrame,
    holdout: int = 5,
    features: str = "tfidf",
    seed: int = 0,
    max_train: int = 60_000,
) -> Report:
    """Train on some suites, predict difficulty on suites never seen.

    `max_train` caps the training set. Fitting TF-IDF over half a million
    questions on a 7 GB machine is not the experiment - and the answer does not
    change much past a few tens of thousands of examples.
    """
    from sklearn.linear_model import Ridge

    from switchboard.routing.features import FeatureExtractor

    train, test, held = split_by_suite(questions, holdout, seed)

    if len(train) > max_train:
        train = train.sample(max_train, random_state=seed)

    extractor = FeatureExtractor(mode=features)
    logger.info("Fitting features over %d training questions ...", len(train))
    matrix = extractor.fit(train["query"].tolist()).transform(train["query"].tolist())

    # Ridge rather than plain least squares: TF-IDF gives thousands of mostly
    # empty columns, and unregularised regression memorises rare words.
    regressor = Ridge(alpha=1.0).fit(matrix, train["difficulty"].to_numpy())

    constant = float(train["difficulty"].mean())
    test_matrix = extractor.transform(test["query"].tolist())
    predicted = np.clip(regressor.predict(test_matrix), 0.0, 1.0)
    actual = test["difficulty"].to_numpy()

    report = Report(
        n_questions=len(questions),
        n_suites=questions["benchmark"].nunique(),
        mean_difficulty=float(questions["difficulty"].mean()),
        held_out_suites=held,
        overall=evaluate(predicted, actual, constant),
        features=extractor.describe(),
    )

    lengths = test["length"].to_numpy()
    for name, low, high in LENGTH_BUCKETS:
        mask = (lengths >= low) & (lengths < high)
        if mask.sum() >= 3:
            report.by_length[name] = evaluate(
                predicted[mask], actual[mask], constant
            )

    suites = test["benchmark"].to_numpy()
    for suite in held:
        mask = suites == suite
        if mask.sum() >= 3:
            report.by_suite[suite] = evaluate(predicted[mask], actual[mask], constant)

    return report

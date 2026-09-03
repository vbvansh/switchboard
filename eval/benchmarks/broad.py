"""Training one router across every suite, and reporting where it works.

WHAT THIS TESTS, AND WHY IT IS DIFFERENT FROM WHAT CAME BEFORE.

Every router this project has shipped was trained on ONE suite:

    switchboard bench train llmrouterbench --suite mmlupro

MMLU-Pro. Academic multiple choice. Then it was surprising that it had no
opinion about a chat message.

Forty suites are on disk - maths, code generation, real GitHub issues, factual
lookup, reading comprehension, open-ended chat, graduate exams. Nobody has
trained on them, because `bench train` needs a COMPLETE grid: every model must
have answered every question. Combine suites and the grid empties out, which is
the bug Phase C hit.

That constraint is gone. `SuccessPredictor.fit_per_model` was written for live
traffic, where each request was answered by exactly one model - the same sparse
shape benchmark data has once you stop demanding a rectangle. Each classifier
learns from the questions its own model actually answered, across every suite.

TWO MEASUREMENTS, AND ONLY ONE OF THEM IS IN DOUBT.

  IN-DOMAIN   hold out QUESTIONS from suites the router has seen.
              This is what a user gets whose traffic resembles a covered
              domain, and it is the number that decides whether shipping a
              broad router is worth doing.

  TRANSFER    hold out whole SUITES.
              This is expected to fail. Two experiments already showed nothing
              transfers to an unseen domain. It is measured anyway, because a
              claim repeated without measurement becomes folklore.

THE OUTPUT THAT MATTERS is neither of those averages. It is the PER-DOMAIN
table: which kinds of question this router can actually judge, and which it
cannot. An average across forty suites hides every weak row, and a weak row is
exactly what an operator needs to know about before trusting a decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Answers a model needs before a classifier is fitted for it. Below this it is
#: fitting noise and then routing real traffic on the result.
MIN_PER_MODEL = 200

#: And at least this many of each outcome. A model that was right on every
#: training question produces a classifier that answers "yes" to everything.
MIN_PER_CLASS = 30

#: Rows kept per model. Some models answered tens of thousands of questions and
#: the answer stops moving long before that; the cap keeps the run inside a
#: modest machine's memory.
MAX_ROWS_PER_MODEL = 12_000

#: A score above this counts as correct. Most suites are already 0/1; the
#: graded ones need a line drawn somewhere and half credit is the natural one.
CORRECT_THRESHOLD = 0.5


@dataclass
class ModelScore:
    """How well one model's classifier predicts its own successes."""

    model: str
    n_train: int = 0
    n_test: int = 0
    auc: float = float("nan")
    #: How often this model was simply right. A classifier for a model that is
    #: right 95% of the time has little left to predict, and its AUC should be
    #: read next to this rather than on its own.
    base_rate: float = float("nan")

    @property
    def usable(self) -> bool:
        return self.auc == self.auc and self.auc >= 0.60


@dataclass
class SuiteScore:
    """How well the router judges one KIND of question."""

    suite: str
    n_questions: int = 0
    mean_auc: float = float("nan")
    n_models: int = 0

    @property
    def verdict(self) -> str:
        if self.mean_auc != self.mean_auc:
            return "not scorable"
        if self.mean_auc >= 0.65:
            return "works"
        if self.mean_auc >= 0.57:
            return "weak"
        return "no signal"


@dataclass
class BroadReport:
    n_rows: int = 0
    n_questions: int = 0
    n_models: int = 0
    n_suites: int = 0
    held_out_suites: list[str] = field(default_factory=list)
    features: str = ""
    #: Trained on the TRAINING split only. Kept so the CLI can report against
    #: it, but never the thing that gets shipped - see `retrain_on_everything`.
    predictor: object | None = None
    in_domain: dict[str, ModelScore] = field(default_factory=dict)
    transfer: dict[str, ModelScore] = field(default_factory=dict)
    by_suite: dict[str, SuiteScore] = field(default_factory=dict)

    @staticmethod
    def _mean(scores) -> float:
        values = [s.auc for s in scores if s.auc == s.auc]
        return float(np.mean(values)) if values else float("nan")

    @property
    def in_domain_auc(self) -> float:
        return self._mean(self.in_domain.values())

    @property
    def transfer_auc(self) -> float:
        return self._mean(self.transfer.values())

    @property
    def within_suite_auc(self) -> float:
        """Mean AUC measured INSIDE each suite, one at a time.

        Read this next to `in_domain_auc`, never instead of it, because the two
        answer different questions and the gap between them is informative.

        `in_domain_auc` mixes every suite together, so a classifier scores well
        for learning "this model is good at code and bad at maths" - topic
        recognition. `within_suite_auc` asks the harder thing: among questions
        of the SAME kind, can it tell which ones this model will get wrong?

        For the difficulty experiment that gap was fatal, because knowing a
        suite is hard shifts every model's estimate equally and changes no
        routing decision. Here it is not fatal: knowing one model suits code
        and another suits maths genuinely changes which model gets picked. It
        is real routing value - just topic routing rather than difficulty
        routing, and the docs must say which one is being sold.
        """
        values = [
            s.mean_auc for s in self.by_suite.values() if s.mean_auc == s.mean_auc
        ]
        return float(np.mean(values)) if values else float("nan")

    @property
    def working_suites(self) -> list[SuiteScore]:
        return [s for s in self.by_suite.values() if s.verdict == "works"]

    def verdict(self) -> str:
        in_domain = self.in_domain_auc
        transfer = self.transfer_auc
        working = len(self.working_suites)
        total = len(self.by_suite)

        if in_domain != in_domain or in_domain < 0.55:
            return (
                f"NO USABLE SIGNAL. Even on questions from suites it trained "
                f"on, mean AUC is {in_domain:.3f}. Breadth of training data is "
                "not the problem, and a shipped router is not worth building "
                "from this."
            )

        within = self.within_suite_auc
        transferred = (
            f"Transfer to unseen suites {transfer:.3f}."
            if transfer == transfer
            else "Transfer not scorable."
        )

        if within >= 0.62:
            return (
                f"WORKS BOTH WAYS. {in_domain:.3f} across suites and "
                f"{within:.3f} within them, so it can pick a model by topic "
                f"AND tell hard questions from easy ones. {transferred} Ship "
                f"it for the {working} of {total} suites that clear the bar."
            )
        if in_domain >= 0.65:
            return (
                f"TOPIC ROUTING ONLY. {in_domain:.3f} across suites but "
                f"{within:.3f} within them - it has learned which model suits "
                "which KIND of question, not which questions are hard. That is "
                "still worth shipping, and it is a different claim from the "
                f"one the benchmarks make. {transferred} "
                f"{working} of {total} suites clear the bar outright."
            )
        return (
            f"NO USABLE SIGNAL. {in_domain:.3f} across suites, {within:.3f} "
            "within them. Breadth of training data is not the problem, and a "
            "shipped router is not worth building from this."
        )


def rows_with_text(frame: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """Join question text onto every (question, model, correct) row."""
    merged = frame.merge(
        queries[["benchmark", "query_id", "query"]],
        on=["benchmark", "query_id"],
        how="left",
    )
    merged["query"] = merged["query"].fillna("")
    merged = merged[merged["query"].str.len() > 0].copy()
    merged["label"] = (merged["correct"] > CORRECT_THRESHOLD).astype(int)
    return merged[["benchmark", "query_id", "model", "query", "label"]]


def split(
    rows: pd.DataFrame,
    holdout_suites: int = 6,
    test_size: float = 0.25,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Three sets: train, in-domain test, transfer test.

    The in-domain split is by QUESTION, not by row. Splitting rows would put
    the same question in both halves - answered by model A in training and
    model B in test - and the score would measure memory of the question rather
    than judgement about it.
    """
    rng = np.random.default_rng(seed)

    suites = sorted(rows["benchmark"].unique())
    if len(suites) <= holdout_suites:
        raise ValueError(
            f"Only {len(suites)} suites; cannot hold out {holdout_suites}."
        )
    held = sorted(rng.permutation(suites)[:holdout_suites].tolist())

    seen = rows[~rows["benchmark"].isin(held)]
    transfer = rows[rows["benchmark"].isin(held)]

    questions = seen[["benchmark", "query_id"]].drop_duplicates()
    shuffled = questions.sample(frac=1.0, random_state=seed)
    cut = max(1, int(len(shuffled) * test_size))
    test_keys = set(map(tuple, shuffled.iloc[:cut].to_numpy()))

    is_test = [
        (benchmark, query_id) in test_keys
        for benchmark, query_id in zip(
            seen["benchmark"], seen["query_id"], strict=True
        )
    ]
    is_test = np.array(is_test)
    return seen[~is_test], seen[is_test], transfer, held


def _cap(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Keep at most MAX_ROWS_PER_MODEL per model."""
    # Written with plain indexing rather than groupby.apply: pandas has
    # changed the meaning of `include_groups` twice, and a sampling helper is
    # not worth tracking that across versions.
    keep = []
    for _, group in rows.groupby("model", observed=True):
        if len(group) > MAX_ROWS_PER_MODEL:
            group = group.sample(MAX_ROWS_PER_MODEL, random_state=seed)
        keep.append(group)
    return pd.concat(keep, ignore_index=True)


def eligible_models(train: pd.DataFrame) -> list[str]:
    """Models with enough balanced examples to fit a classifier for."""
    keep = []
    for model, group in train.groupby("model", observed=True):
        positives = int(group["label"].sum())
        negatives = len(group) - positives
        if (
            len(group) >= MIN_PER_MODEL
            and positives >= MIN_PER_CLASS
            and negatives >= MIN_PER_CLASS
        ):
            keep.append(str(model))
    return sorted(keep)


def fit(train: pd.DataFrame, models: list[str], features: str, seed: int):
    """Train one classifier per model over its own rows, across all suites."""
    from switchboard.routing.features import FeatureExtractor
    from switchboard.routing.predictor import SuccessPredictor

    per_model = {}
    for model in models:
        subset = train[train["model"] == model]
        per_model[model] = (
            subset["query"].tolist(),
            subset["label"].to_numpy(),
        )

    corpus = train["query"].drop_duplicates().tolist()
    logger.info(
        "Fitting %d classifiers over a %d-question vocabulary ...",
        len(per_model),
        len(corpus),
    )
    return SuccessPredictor.fit_per_model(
        corpus=corpus,
        per_model=per_model,
        extractor=FeatureExtractor(mode=features),
        seed=seed,
    )


def score(predictor, test: pd.DataFrame, train_counts: dict) -> dict[str, ModelScore]:
    """Held-out AUC per model. NaN where it cannot be computed, never 0.5."""
    from sklearn.metrics import roc_auc_score

    results: dict[str, ModelScore] = {}
    for model in predictor.models:
        subset = test[test["model"] == model]
        entry = ModelScore(
            model=model,
            n_train=train_counts.get(model, 0),
            n_test=len(subset),
        )
        if len(subset) >= 20 and subset["label"].nunique() == 2:
            labels = subset["label"].to_numpy()
            probabilities = np.array(
                [
                    row[model]
                    for row in predictor.predict_batch(subset["query"].tolist())
                ]
            )
            entry.auc = float(roc_auc_score(labels, probabilities))
            entry.base_rate = float(labels.mean())
        results[model] = entry
    return results


def score_by_suite(predictor, test: pd.DataFrame) -> dict[str, SuiteScore]:
    """The table that matters: which KINDS of question this router can judge.

    Averaged across models within each suite, because a suite where every
    model's classifier is useless is a suite the router should not be trusted
    on - regardless of how well it does elsewhere.
    """
    from sklearn.metrics import roc_auc_score

    results: dict[str, SuiteScore] = {}
    for suite, group in test.groupby("benchmark", observed=True):
        aucs = []
        for model in predictor.models:
            subset = group[group["model"] == model]
            if len(subset) < 20 or subset["label"].nunique() != 2:
                continue
            probabilities = np.array(
                [
                    row[model]
                    for row in predictor.predict_batch(subset["query"].tolist())
                ]
            )
            aucs.append(
                float(roc_auc_score(subset["label"].to_numpy(), probabilities))
            )

        results[str(suite)] = SuiteScore(
            suite=str(suite),
            n_questions=int(group["query_id"].nunique()),
            mean_auc=float(np.mean(aucs)) if aucs else float("nan"),
            n_models=len(aucs),
        )
    return results


def run(
    rows: pd.DataFrame,
    holdout_suites: int = 6,
    test_size: float = 0.25,
    features: str = "tfidf",
    seed: int = 0,
) -> BroadReport:
    train, in_domain_test, transfer_test, held = split(
        rows, holdout_suites, test_size, seed
    )
    train = _cap(train, seed)

    models = eligible_models(train)
    if len(models) < 2:
        raise ValueError(
            f"Only {len(models)} model(s) have enough balanced training rows "
            f"(need {MIN_PER_MODEL} rows with {MIN_PER_CLASS} of each outcome)."
        )

    predictor = fit(train, models, features, seed)
    counts = train.groupby("model", observed=True).size().to_dict()

    return BroadReport(
        n_rows=len(rows),
        n_questions=int(rows["query_id"].nunique()),
        n_models=len(predictor.models),
        n_suites=int(rows["benchmark"].nunique()),
        held_out_suites=held,
        features=predictor.extractor.describe(),
        predictor=predictor,
        in_domain=score(predictor, in_domain_test, counts),
        transfer=score(predictor, transfer_test, counts),
        by_suite=score_by_suite(predictor, in_domain_test),
    )


def retrain_on_everything(
    rows: pd.DataFrame, features: str = "tfidf", seed: int = 0
):
    """Refit on ALL the data, for the artifact that actually ships.

    Two different jobs, and conflating them is a classic way to publish a
    number nobody can reproduce:

        MEASURING   train on part, score on the rest. The held-out score is
                    the only honest description of how well it works.
        SHIPPING    train on everything. More data is strictly better for the
                    artifact, and there is nothing left to score it against -
                    which is fine, because the score already came from the
                    measuring run.

    So the numbers in the coverage table come from the split run, and the file
    users load comes from this one. The metadata records both facts.
    """
    capped = _cap(rows, seed)
    models = eligible_models(capped)
    if len(models) < 2:
        raise ValueError("Not enough models with balanced data to ship.")
    return fit(capped, models, features, seed)

"""Turning your own served traffic into a router that understands it.

THE PROBLEM THIS CLOSES.

A router trained on public benchmarks learns from ~700-character exam
questions. Shown a 34-character chat message it returns roughly the same
confidence for every model, has no opinion, and everything falls through to the
cheapest one. That is distribution shift, and it is documented rather than
hidden - see `switchboard/routing/live.py` and docs/RESULTS.md section 6.

The fix has always been obvious: train on the traffic you actually serve.
Shadow mode was built to collect it. This module is the step that was missing
between the two.

WHY IT NEEDS FEEDBACK, AND WILL NOT PROCEED WITHOUT IT.

Training needs pairs of (question, did this model get it right). A benchmark
ships with an answer key, so the second half is free. Real traffic has no
answer key - nobody wrote down the correct response to "why is this test
flaky".

The only honest source is a person saying so, which is what POST /v1/feedback
collects.

There is a tempting alternative: guess. "The user asked again thirty seconds
later, so the first answer was probably bad." It is clever, it is cheap, and it
is wrong often enough to matter. A wrong label does not raise an error - it
quietly teaches the router something false, and every decision afterwards is
built on it. Same rule as refusing to invent a model's price in
`switchboard/discovery.py`: a made-up label is worse than no label.

WHAT IS DIFFERENT ABOUT LIVE DATA.

Benchmark data is DENSE: every question was answered by every model, so a
router can compare them directly on the same question. Live data is SPARSE:
each request was answered by exactly one model, and what the others would have
said is unknown.

That is fine for this design, because there was never one model comparing all
options - there is one classifier per model, each learning "will I get this
kind of question right?" from the requests it personally handled. Sparse data
suits that exactly. It just means each model needs enough of its own examples,
which is what the gates below enforce.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

#: Rated requests a model needs before it may be trained on. Below this the
#: classifier is fitting noise: it will produce confident probabilities from
#: almost nothing and route real traffic on them.
MIN_PER_MODEL = 30

#: And at least this many of EACH verdict. Forty ratings that all say "good"
#: describe a model nobody has seen fail, not a model that does not fail. A
#: classifier fitted on them answers "yes" to everything, which would hand that
#: model every routing decision on the strength of forty examples.
MIN_PER_CLASS = 5

#: A router with one model is not a router.
MIN_MODELS = 2

#: Fraction of requests held back for scoring.
DEFAULT_TEST_SIZE = 0.3
DEFAULT_SEED = 0


@dataclass(frozen=True)
class LabelledRequest:
    """One served request that somebody rated."""

    prompt: str
    model: str
    correct: bool
    created_at: datetime | None = None


@dataclass
class ModelReadiness:
    """Whether one model has enough rated traffic to train on."""

    model: str
    served: int = 0
    good: int = 0
    bad: int = 0

    @property
    def rated(self) -> int:
        return self.good + self.bad

    @property
    def usable(self) -> bool:
        return (
            self.rated >= MIN_PER_MODEL
            and self.good >= MIN_PER_CLASS
            and self.bad >= MIN_PER_CLASS
        )

    def blocker(self) -> str:
        """Why this model cannot be trained on yet, in one line."""
        if self.usable:
            return ""
        if self.rated < MIN_PER_MODEL:
            return f"needs {MIN_PER_MODEL - self.rated} more rated requests"
        missing = []
        if self.good < MIN_PER_CLASS:
            missing.append(f"{MIN_PER_CLASS - self.good} more rated good")
        if self.bad < MIN_PER_CLASS:
            missing.append(f"{MIN_PER_CLASS - self.bad} more rated bad")
        return "needs " + " and ".join(missing)


@dataclass
class Readiness:
    """Can a router be trained from this ledger yet, and if not, why not."""

    models: list[ModelReadiness] = field(default_factory=list)
    total_served: int = 0
    total_rated: int = 0
    store_prompts: bool = True
    with_prompt_text: int = 0
    first_rated: datetime | None = None
    last_rated: datetime | None = None

    @property
    def usable_models(self) -> list[ModelReadiness]:
        return [m for m in self.models if m.usable]

    @property
    def can_train(self) -> bool:
        return len(self.usable_models) >= MIN_MODELS and self.with_prompt_text > 0

    @property
    def coverage_pct(self) -> float:
        """Share of served requests that carry a rating."""
        if not self.total_served:
            return 0.0
        return 100.0 * self.total_rated / self.total_served

    @property
    def period(self) -> str:
        if not (self.first_rated and self.last_rated):
            return ""
        return f"{self.first_rated:%Y-%m-%d} to {self.last_rated:%Y-%m-%d}"

    def blockers(self) -> list[str]:
        """Everything standing between here and a trained router.

        Written to be actionable. "Not enough data" is not a useful thing to
        tell somebody who has been collecting for a fortnight.
        """
        problems: list[str] = []

        if not self.store_prompts:
            problems.append(
                "Prompt text is not being stored, so there is nothing to learn "
                "from. Set SWITCHBOARD_STORE_PROMPTS=true - and tell your "
                "users, because it means recording what they type."
            )
        elif self.total_rated and not self.with_prompt_text:
            problems.append(
                f"{self.total_rated:,} requests are rated but none has prompt "
                "text. They were served before prompt storage was switched on; "
                "only traffic from now on can be trained on."
            )

        if not self.total_rated:
            problems.append(
                "No request has been rated. Send the X-Switchboard-Request-Id "
                "from a response back to POST /v1/feedback with a verdict - "
                "that is what a thumbs up/down in your application does."
            )

        usable = self.usable_models
        if self.total_rated and len(usable) < MIN_MODELS:
            short = [m for m in self.models if not m.usable and m.rated]
            detail = "; ".join(f"{m.model} {m.blocker()}" for m in short[:4])
            problems.append(
                f"Only {len(usable)} model(s) have enough rated traffic; "
                f"{MIN_MODELS} are needed. {detail}"
                if detail
                else f"Only {len(usable)} model(s) have enough rated traffic."
            )

        return problems


def prompt_from_json(prompt_json: str | None) -> str:
    """Recover the user's text from a stored `messages` array.

    System messages are skipped, exactly as the usage policy skips them: they
    are written by the application, not the person, so a fixed system prompt
    would otherwise be identical across every training example and teach the
    classifier nothing while dominating the vocabulary.
    """
    if not prompt_json:
        return ""
    try:
        messages = json.loads(prompt_json)
    except (ValueError, TypeError):
        return ""
    if not isinstance(messages, list):
        return ""

    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") == "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
    return "\n".join(parts).strip()


def collect(rows) -> list[LabelledRequest]:
    """Turn rated ledger rows into training examples.

    Rows with no recoverable prompt text are dropped rather than trained on as
    empty strings, which would teach every classifier that a blank question is
    a normal question.
    """
    examples: list[LabelledRequest] = []
    for row in rows:
        prompt = prompt_from_json(getattr(row, "prompt_json", None))
        if not prompt:
            continue
        rating = getattr(row, "feedback", None)
        if rating not in ("good", "bad"):
            continue
        examples.append(
            LabelledRequest(
                prompt=prompt,
                model=str(row.served_model),
                correct=rating == "good",
                created_at=getattr(row, "created_at", None),
            )
        )
    return examples


def assess(
    rated_rows, served_counts: list[tuple[str, int]], store_prompts: bool
) -> Readiness:
    """Build the readiness report, without training anything."""
    examples = collect(rated_rows)
    served = dict(served_counts)

    per_model: dict[str, ModelReadiness] = {
        model: ModelReadiness(model=model, served=count)
        for model, count in served.items()
    }

    # Ratings are counted from the RAW rows, not from `examples`. A rating on a
    # request with no stored prompt still happened, and hiding it would make
    # the report say "nobody has rated anything" to somebody who has.
    rated_total = 0
    stamps: list[datetime] = []
    for row in rated_rows:
        rating = getattr(row, "feedback", None)
        if rating not in ("good", "bad"):
            continue
        rated_total += 1
        model = str(row.served_model)
        entry = per_model.setdefault(model, ModelReadiness(model=model))
        if rating == "good":
            entry.good += 1
        else:
            entry.bad += 1
        if (stamp := getattr(row, "created_at", None)) is not None:
            stamps.append(stamp)

    # Only requests that can actually be trained on count towards usability.
    trainable: dict[str, ModelReadiness] = {
        model: ModelReadiness(model=model, served=served.get(model, 0))
        for model in {e.model for e in examples}
    }
    for example in examples:
        entry = trainable[example.model]
        if example.correct:
            entry.good += 1
        else:
            entry.bad += 1

    for model, entry in trainable.items():
        per_model[model].good = entry.good
        per_model[model].bad = entry.bad

    return Readiness(
        models=sorted(per_model.values(), key=lambda m: -m.rated),
        total_served=sum(served.values()),
        total_rated=rated_total,
        store_prompts=store_prompts,
        with_prompt_text=len(examples),
        first_rated=min(stamps) if stamps else None,
        last_rated=max(stamps) if stamps else None,
    )


class NotEnoughData(RuntimeError):
    """The ledger cannot support a router yet. Carries the reasons."""

    def __init__(self, readiness: Readiness) -> None:
        self.readiness = readiness
        super().__init__(" ".join(readiness.blockers()) or "not enough rated data")


def split(
    examples: list[LabelledRequest],
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[list[LabelledRequest], list[LabelledRequest]]:
    """Hold some requests back for scoring.

    Split by DISTINCT PROMPT, not by row. The same question asked twice - by a
    retrying script, or two people hitting the same problem - must not land in
    both halves, or the router is scored on text it was trained on and the AUC
    is a measure of memory.
    """
    prompts = sorted({e.prompt for e in examples})
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(prompts))
    cut = max(1, int(len(shuffled) * test_size))
    held_out = set(shuffled[:cut])

    train = [e for e in examples if e.prompt not in held_out]
    test = [e for e in examples if e.prompt in held_out]
    return train, test


def training_sets(
    examples: list[LabelledRequest], models: list[str]
) -> dict[str, tuple[list[str], np.ndarray]]:
    """Group examples by the model that handled them.

    Each classifier learns only from requests ITS model actually served. That
    is what `SuccessPredictor.fit_per_model` expects, and it is why live data
    must not be flattened into the dense benchmark shape: writing a 0 wherever
    a model was not asked would teach every classifier that every question
    somebody else answered was one it got wrong.
    """
    grouped: dict[str, tuple[list[str], np.ndarray]] = {}
    for model in models:
        subset = [e for e in examples if e.model == model]
        if not subset:
            continue
        grouped[model] = (
            [e.prompt for e in subset],
            np.array([int(e.correct) for e in subset]),
        )
    return grouped


def train(
    examples: list[LabelledRequest],
    models: list[str],
    features: str = "tfidf",
    seed: int = DEFAULT_SEED,
):
    """Fit a predictor over live examples. Returns a SuccessPredictor."""
    from switchboard.routing.features import FeatureExtractor
    from switchboard.routing.predictor import SuccessPredictor

    return SuccessPredictor.fit_per_model(
        corpus=[e.prompt for e in examples],
        per_model=training_sets(examples, models),
        extractor=FeatureExtractor(mode=features),
        seed=seed,
    )


def score(predictor, examples: list[LabelledRequest]) -> dict[str, dict]:
    """How well each classifier predicts success on HELD-OUT requests.

    AUC of 0.5 means the classifier is guessing - the features carry no signal
    about which questions this model gets wrong, and no routing rule built on
    top can help. Reported per model rather than averaged away, because one
    strong classifier can hide two useless ones behind a decent mean.
    """
    from sklearn.metrics import roc_auc_score

    results: dict[str, dict] = {}
    for model in predictor.models:
        subset = [e for e in examples if e.model == model]
        labels = np.array([int(e.correct) for e in subset])
        entry = {
            "n": len(subset),
            "base_rate": float(labels.mean()) if len(subset) else float("nan"),
        }

        if len(subset) < 2 or labels.min() == labels.max():
            # Not enough held-out variety to score. Reported as unscored rather
            # than as a number, because an invented AUC is worse than a blank.
            entry["auc"] = float("nan")
            results[model] = entry
            continue

        probabilities = np.array(
            [row[model] for row in predictor.predict_batch([e.prompt for e in subset])]
        )
        entry["auc"] = float(roc_auc_score(labels, probabilities))
        results[model] = entry
    return results

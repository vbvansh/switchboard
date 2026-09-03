"""Saving a trained router, and loading it back into the running server.

Training happens offline against benchmark data; serving happens in a process
that must start in under a second. So the trained thing is written to a file
and loaded at startup.

A note on the file format. It is a joblib pickle, which executes code on load -
the same objection I raised against the RouterBench dataset in Phase B. The
difference is provenance: that was a 1.2 GB file from the internet, this is an
artifact the operator produced themselves on their own machine. Loading your
own build output is not the same risk as loading a stranger's. The path is
configuration, never a URL, and the metadata is checked before use.

The harder problem this file solves is NAMES. A router trained on public
benchmark data knows models called `qwen2.5-7b-instruct`. The operator's
catalog has `qwen2.5:7b` from their own Ollama. Those are the same family and
different strings, so `providers.yaml` lets a model declare which benchmark
model it stands in for, and the artifact is matched against that.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Bumped when the artifact layout changes in a way older code cannot read.
#
# v2: the predictor and feature extractor moved from `eval.benchmarks.*`
# into `switchboard.routing.*`. A pickle records the module a class came
# from, and `eval/` is not in the Docker image - so every v1 artifact was
# unloadable in a container, silently, with routing switching itself off.
# Bumping the version turns that into 'retrain it' instead of a mystery.
ARTIFACT_VERSION = 2


class ArtifactError(RuntimeError):
    """The router artifact is missing, unreadable, or incompatible."""


@dataclass
class RouterMetadata:
    """What this router was trained on, so a decision can be traced later."""

    version: int = ARTIFACT_VERSION
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    source: str = ""
    benchmark: str = ""
    features: str = ""
    #: "benchmark" or "live traffic". A router trained on recorded feedback
    #: from real users is a different thing from one trained on exam questions,
    #: and anyone reading a routing decision needs to know which they have.
    label_source: str = "benchmark"
    #: For live-trained routers: the span of traffic it learned from.
    period: str = ""
    #: suite -> held-out AUC, for a router trained across many benchmark
    #: suites. THE table an operator needs: a broad router is genuinely useful
    #: on some kinds of question and no better than guessing on others, and an
    #: average over the two hides exactly the rows that matter.
    coverage: dict[str, float] = field(default_factory=dict)
    #: Mean AUC measured WITHIN each suite, as opposed to across them mixed
    #: together. The gap between the two says whether the router is picking a
    #: model by topic or genuinely telling hard questions from easy ones.
    within_suite_auc: float = float("nan")
    #: Benchmark model names the router can choose between.
    models: list[str] = field(default_factory=list)
    n_train_questions: int = 0
    mean_auc: float = float("nan")

    def describe(self) -> str:
        where = f"{self.source}/{self.benchmark}" if self.benchmark else self.source
        noun = "requests" if self.label_source == "live traffic" else "questions"
        span = f" {self.period}" if self.period else ""
        return (
            f"trained {self.trained_at[:10]} on {where}{span} "
            f"({self.n_train_questions:,} {noun}, {len(self.models)} models, "
            f"{self.features} features)"
        )


def save(path: Path | str, predictor: Any, metadata: RouterMetadata) -> Path:
    """Write the trained router and a readable sidecar describing it."""
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"metadata": asdict(metadata), "predictor": predictor}, path)

    # A plain-text sidecar so `switchboard router info` - and a human with a
    # text editor - can see what a file contains without unpickling it.
    path.with_suffix(".json").write_text(
        json.dumps(asdict(metadata), indent=2), encoding="utf-8"
    )
    return path


def load(path: Path | str) -> tuple[Any, RouterMetadata]:
    import joblib

    path = Path(path)
    if not path.exists():
        raise ArtifactError(
            f"No router artifact at {path}.\n"
            "  Train one with: switchboard router train <source>"
        )

    try:
        blob = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - any failure means unusable
        raise ArtifactError(f"Could not read the router at {path}: {exc}") from exc

    if not isinstance(blob, dict) or "predictor" not in blob:
        raise ArtifactError(f"{path} is not a Switchboard router artifact.")

    metadata = RouterMetadata(**blob.get("metadata", {}))
    if metadata.version != ARTIFACT_VERSION:
        raise ArtifactError(
            f"{path} was written by a different version of Switchboard "
            f"(artifact v{metadata.version}, this build expects "
            f"v{ARTIFACT_VERSION}). Retrain it."
        )

    return blob["predictor"], metadata


def read_metadata(path: Path | str) -> RouterMetadata | None:
    """Read the sidecar without loading - and without unpickling - the model."""
    sidecar = Path(path).with_suffix(".json")
    if not sidecar.exists():
        return None
    try:
        return RouterMetadata(**json.loads(sidecar.read_text(encoding="utf-8")))
    except (ValueError, TypeError):
        return None

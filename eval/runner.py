"""Runs task sets through routing strategies and records what happened.

Design notes worth knowing:

* Requests go straight to Ollama, not through the HTTP proxy. The proxy adds
  authentication and budgets, which are irrelevant here and would only make runs
  harder to start. The strategy objects being measured are the same ones the
  proxy will use in milestone 4.

* Tasks run sequentially, in a fixed order. Parallel requests would thrash a
  4GB GPU and make latency numbers meaningless, and reordering per strategy
  would change the model-switch counts we want to compare.

* Results stream to a JSONL file as they complete. A full run on this hardware
  takes a long time; losing it to a crash at task 39 of 40 is not acceptable.
"""

from __future__ import annotations

import asyncio
import json
import platform
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from switchboard.config import Settings
from switchboard.pricing import PriceTable
from switchboard.providers.ollama import OllamaProvider, ProviderUnavailable
from switchboard.routing import RoutingContext, RoutingStrategy

from eval.datasets import Task, TaskSet
from eval.grading import SYSTEM_PROMPT, extract_answer

DEFAULT_MAX_TOKENS = 600


@dataclass
class TaskResult:
    """One task, run once, by one strategy."""

    strategy: str
    task_id: str
    difficulty: str
    category: str

    model: str
    routing_reason: str

    correct: bool
    followed_format: bool
    answer: str

    prompt_tokens: int
    completion_tokens: int
    simulated_cost_usd: float
    baseline_cost_usd: float

    latency_ms: int
    caused_model_switch: bool
    truncated: bool

    error: str | None = None
    raw_response: str = ""


@dataclass
class RunMetadata:
    """Provenance. A benchmark number without this is not comparable."""

    started_at: str
    task_set: str
    task_count: int
    strategies: list[str]
    ladder: list[str]
    baseline_model: str
    max_tokens: int
    temperature: float
    ollama_url: str
    # Not named `platform`: a field of that name would shadow the module inside
    # the class body and break the default_factory below.
    os_platform: str = field(default_factory=platform.platform)
    python_version: str = field(default_factory=platform.python_version)
    finished_at: str | None = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class EvalRunner:
    def __init__(
        self,
        settings: Settings,
        prices: PriceTable,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> None:
        self._settings = settings
        self._prices = prices
        self._max_tokens = max_tokens
        # Zero temperature so a re-run reproduces the same answers. Comparing
        # strategies against a moving target would be meaningless.
        self._temperature = temperature
        self._last_model: str | None = None

    async def run(
        self,
        taskset: TaskSet,
        strategies: Iterable[RoutingStrategy],
        output_path: Path,
        progress=None,
    ) -> tuple[list[TaskResult], RunMetadata]:
        strategies = list(strategies)
        metadata = RunMetadata(
            started_at=_now_iso(),
            task_set=taskset.name,
            task_count=len(taskset),
            strategies=[s.name for s in strategies],
            ladder=list(self._prices.ladder),
            baseline_model=self._prices.baseline_model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            ollama_url=self._settings.ollama_base_url,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        results: list[TaskResult] = []
        provider = OllamaProvider(self._settings)

        try:
            with output_path.open("w", encoding="utf-8") as stream:
                stream.write(
                    json.dumps({"_metadata": asdict(metadata)}, ensure_ascii=False)
                    + "\n"
                )
                for strategy in strategies:
                    # Each strategy starts with no warm model assumed, so switch
                    # counts are comparable between them.
                    self._last_model = None
                    for task in taskset:
                        result = await self._run_one(provider, strategy, task)
                        results.append(result)
                        stream.write(
                            json.dumps(asdict(result), ensure_ascii=False) + "\n"
                        )
                        stream.flush()
                        if progress is not None:
                            progress(result)
        finally:
            await provider.aclose()

        metadata.finished_at = _now_iso()
        return results, metadata

    async def _run_one(
        self, provider: OllamaProvider, strategy: RoutingStrategy, task: Task
    ) -> TaskResult:
        context = RoutingContext(
            messages=[{"role": "user", "content": task.prompt}]
        )
        decision = strategy.choose(context)

        switched = self._last_model is not None and self._last_model != decision.model
        self._last_model = decision.model

        payload: dict[str, Any] = {
            "model": decision.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task.prompt},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        started = asyncio.get_running_loop().time()
        try:
            response = await provider.chat_completion(payload)
        except ProviderUnavailable as exc:
            return self._failed(strategy, task, decision, switched, str(exc))

        latency_ms = int((asyncio.get_running_loop().time() - started) * 1000)

        if response.status_code >= 400:
            return self._failed(
                strategy, task, decision, switched, response.text[:300], latency_ms
            )

        body = response.json()
        choice = (body.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        truncated = choice.get("finish_reason") == "length"

        usage = body.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)

        answer, followed_format = extract_answer(text)
        correct = task.check.grade(answer)

        return TaskResult(
            strategy=strategy.name,
            task_id=task.id,
            difficulty=task.difficulty,
            category=task.category,
            model=decision.model,
            routing_reason=decision.reason,
            correct=correct,
            followed_format=followed_format,
            answer=answer[:200],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            simulated_cost_usd=self._prices.cost(
                decision.model, prompt_tokens, completion_tokens
            ),
            baseline_cost_usd=self._prices.baseline_cost(
                prompt_tokens, completion_tokens
            ),
            latency_ms=latency_ms,
            caused_model_switch=switched,
            truncated=truncated,
            raw_response=text[:1000],
        )

    def _failed(
        self,
        strategy: RoutingStrategy,
        task: Task,
        decision,
        switched: bool,
        error: str,
        latency_ms: int = 0,
    ) -> TaskResult:
        """A failed call counts as incorrect, not as a missing data point.

        Dropping failures would quietly inflate the accuracy of whichever model
        fails most - usually the slowest one, which is exactly the model whose
        weaknesses matter.
        """
        return TaskResult(
            strategy=strategy.name,
            task_id=task.id,
            difficulty=task.difficulty,
            category=task.category,
            model=decision.model,
            routing_reason=decision.reason,
            correct=False,
            followed_format=False,
            answer="",
            prompt_tokens=0,
            completion_tokens=0,
            simulated_cost_usd=0.0,
            baseline_cost_usd=0.0,
            latency_ms=latency_ms,
            caused_model_switch=switched,
            truncated=False,
            error=error,
        )


def load_results(path: Path) -> tuple[list[TaskResult], dict]:
    """Read a JSONL run back, including a run that crashed part-way."""
    metadata: dict = {}
    results: list[TaskResult] = []

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "_metadata" in record:
                metadata = record["_metadata"]
            else:
                results.append(TaskResult(**record))

    return results, metadata

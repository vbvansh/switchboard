"""Task sets, the runner, and reporting - all without touching Ollama."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from eval.datasets import load_taskset
from eval.report import summarise, to_markdown
from eval.runner import EvalRunner, TaskResult, load_results
from switchboard.catalog import ModelCatalog
from switchboard.config import Settings
from switchboard.routing import build_strategy

# --- Task set integrity ----------------------------------------------------


def test_builtin_taskset_loads() -> None:
    tasks = load_taskset("builtin")
    assert len(tasks) == 40


def test_every_difficulty_is_represented() -> None:
    counts = load_taskset("builtin").counts_by_difficulty()
    assert all(count > 0 for count in counts.values()), counts


def test_task_ids_are_unique() -> None:
    ids = [task.id for task in load_taskset("builtin")]
    assert len(set(ids)) == len(ids)


def test_every_task_has_a_working_check() -> None:
    """A task whose checker throws would silently mark everything wrong."""
    for task in load_taskset("builtin"):
        assert task.check.grade("definitely not the answer") is False


def test_reference_answers_grade_as_correct() -> None:
    """The expected answer must actually pass its own check."""
    for task in load_taskset("builtin"):
        if task.check.type == "numeric" or task.check.type == "exact":
            assert task.check.grade(str(task.check.value)), task.id
        else:
            assert task.check.grade(" ".join(task.check.values)), task.id


def test_missing_taskset_names_available_ones() -> None:
    with pytest.raises(FileNotFoundError, match="builtin"):
        load_taskset("does-not-exist")


def test_filtering(taskset_name: str = "builtin") -> None:
    tasks = load_taskset(taskset_name)
    assert len(tasks.filtered(limit=5)) == 5
    assert all(t.difficulty == "hard" for t in tasks.filtered(difficulty="hard"))
    assert all(t.category == "code" for t in tasks.filtered(category="code"))


# --- Runner ----------------------------------------------------------------


class ScriptedProvider:
    """Answers correctly only for models at or above a chosen tier.

    Lets the harness be tested end to end without Ollama, and mimics the real
    effect the whole project depends on: cheap models fail harder tasks.
    """

    def __init__(self, competent_models: set[str]) -> None:
        self.competent_models = competent_models
        self.calls: list[dict] = []

    async def aclose(self) -> None:
        pass

    async def chat_completion(self, payload: dict) -> httpx.Response:
        self.calls.append(payload)
        question = payload["messages"][-1]["content"]
        good = payload["model"] in self.competent_models
        answer = "45" if good else "wrong"
        return httpx.Response(
            200,
            content=json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": f"Working on: {question}\n"
                                f"ANSWER: {answer}"
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                }
            ).encode(),
        )


class ScriptedPool:
    """Minimal ProviderPool that hands the scripted provider to every model."""

    def __init__(self, provider: ScriptedProvider) -> None:
        self.provider = provider

    def for_model(self, model: str) -> ScriptedProvider:
        return self.provider

    async def aclose(self) -> None:
        await self.provider.aclose()


def _pool_factory(provider: ScriptedProvider):
    """Replacement for ProviderPool's constructor inside the runner."""
    return lambda catalog, local_only=False: ScriptedPool(provider)


@pytest.fixture
def runner(prices: ModelCatalog) -> EvalRunner:
    return EvalRunner(Settings(), prices)


@pytest.fixture
def one_task():
    return load_taskset("builtin").filtered(limit=1)  # arith-01, answer 45


def test_runner_grades_and_prices_a_task(
    runner: EvalRunner, one_task, prices: ModelCatalog, tmp_path: Path, monkeypatch
) -> None:
    provider = ScriptedProvider(competent_models=set(prices.ladder))
    monkeypatch.setattr(
        "eval.runner.ProviderPool", _pool_factory(provider)
    )

    import asyncio

    results, metadata = asyncio.run(
        runner.run(
            one_task,
            [build_strategy("always-cheap", prices)],
            tmp_path / "run.jsonl",
        )
    )

    (result,) = results
    assert result.correct is True
    assert result.model == prices.cheapest
    assert result.simulated_cost_usd > 0
    assert result.baseline_cost_usd > result.simulated_cost_usd
    assert metadata.finished_at is not None


def test_runner_records_incorrect_answers(
    runner: EvalRunner, one_task, prices: ModelCatalog, tmp_path: Path, monkeypatch
) -> None:
    provider = ScriptedProvider(competent_models={prices.most_expensive})
    monkeypatch.setattr("eval.runner.ProviderPool", _pool_factory(provider))

    import asyncio

    results, _ = asyncio.run(
        runner.run(
            one_task,
            [build_strategy("always-cheap", prices)],
            tmp_path / "run.jsonl",
        )
    )
    assert results[0].correct is False


def test_runner_applies_the_same_system_prompt_to_every_model(
    runner: EvalRunner, one_task, prices: ModelCatalog, tmp_path: Path, monkeypatch
) -> None:
    """Fairness: no strategy may get a different prompt."""
    provider = ScriptedProvider(competent_models=set(prices.ladder))
    monkeypatch.setattr("eval.runner.ProviderPool", _pool_factory(provider))

    import asyncio

    asyncio.run(
        runner.run(
            one_task,
            [
                build_strategy("always-cheap", prices),
                build_strategy("always-expensive", prices),
            ],
            tmp_path / "run.jsonl",
        )
    )

    systems = {call["messages"][0]["content"] for call in provider.calls}
    temperatures = {call["temperature"] for call in provider.calls}
    assert len(systems) == 1
    assert temperatures == {0.0}


def test_results_stream_to_disk_and_reload(
    runner: EvalRunner, prices: ModelCatalog, tmp_path: Path, monkeypatch
) -> None:
    """A long run must survive being interrupted."""
    provider = ScriptedProvider(competent_models=set(prices.ladder))
    monkeypatch.setattr("eval.runner.ProviderPool", _pool_factory(provider))

    import asyncio

    path = tmp_path / "run.jsonl"
    tasks = load_taskset("builtin").filtered(limit=3)
    asyncio.run(runner.run(tasks, [build_strategy("always-cheap", prices)], path))

    reloaded, metadata = load_results(path)
    assert len(reloaded) == 3
    assert metadata["task_set"] == "builtin"
    assert metadata["ladder"] == list(prices.ladder)


def test_model_switches_are_counted(
    runner: EvalRunner, prices: ModelCatalog, tmp_path: Path, monkeypatch
) -> None:
    provider = ScriptedProvider(competent_models=set(prices.ladder))
    monkeypatch.setattr("eval.runner.ProviderPool", _pool_factory(provider))

    import asyncio

    tasks = load_taskset("builtin").filtered(limit=6)
    results, _ = asyncio.run(
        runner.run(tasks, [build_strategy("random", prices)], tmp_path / "r.jsonl")
    )
    # A fixed model can never switch; random almost certainly does.
    assert any(r.caused_model_switch for r in results)


def test_each_strategy_starts_with_no_warm_model(
    runner: EvalRunner, prices: ModelCatalog, tmp_path: Path, monkeypatch
) -> None:
    """Otherwise the second strategy inherits the first one's warm state."""
    provider = ScriptedProvider(competent_models=set(prices.ladder))
    monkeypatch.setattr("eval.runner.ProviderPool", _pool_factory(provider))

    import asyncio

    tasks = load_taskset("builtin").filtered(limit=2)
    results, _ = asyncio.run(
        runner.run(
            tasks,
            [
                build_strategy("always-cheap", prices),
                build_strategy("always-expensive", prices),
            ],
            tmp_path / "r.jsonl",
        )
    )
    assert not any(r.caused_model_switch for r in results)


# --- Reporting -------------------------------------------------------------


def _result(strategy: str, model: str, correct: bool, cost: float) -> TaskResult:
    return TaskResult(
        strategy=strategy,
        task_id="t",
        difficulty="easy",
        category="arithmetic",
        model=model,
        routing_reason="",
        correct=correct,
        answer_format="marker",
        answer="",
        prompt_tokens=100,
        completion_tokens=50,
        simulated_cost_usd=cost,
        baseline_cost_usd=1.0,
        latency_ms=100,
        caused_model_switch=False,
        truncated=False,
    )


def test_summary_computes_accuracy_and_savings() -> None:
    results = [
        _result("cheap", "small", True, 0.1),
        _result("cheap", "small", False, 0.1),
    ]
    (summary,) = summarise(results)
    assert summary.accuracy == 50.0
    assert summary.cost_usd == pytest.approx(0.2)
    assert summary.saved_vs_baseline_pct == pytest.approx(90.0)


def test_summaries_are_ordered_cheapest_first() -> None:
    results = [
        _result("expensive", "big", True, 1.0),
        _result("cheap", "small", True, 0.1),
    ]
    assert [s.strategy for s in summarise(results)] == ["cheap", "expensive"]


def test_markdown_table_includes_every_strategy() -> None:
    markdown = to_markdown(
        summarise([_result("a", "m", True, 0.1), _result("b", "m", True, 0.2)])
    )
    assert "| a |" in markdown
    assert "| b |" in markdown


def test_zero_baseline_does_not_divide_by_zero() -> None:
    """Every provider call failing would otherwise crash the report."""
    result = _result("x", "m", False, 0.0)
    result.baseline_cost_usd = 0.0
    (summary,) = summarise([result])
    assert summary.saved_vs_baseline_pct == 0.0

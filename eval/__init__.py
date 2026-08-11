"""Evaluation harness: task sets, grading, running, reporting."""

from eval.datasets import Task, TaskSet, load_taskset
from eval.grading import Check, extract_answer, strip_thinking
from eval.report import StrategySummary, pareto_plot, summarise, to_markdown
from eval.runner import EvalRunner, TaskResult, load_results

__all__ = [
    "Check",
    "EvalRunner",
    "StrategySummary",
    "Task",
    "TaskResult",
    "TaskSet",
    "extract_answer",
    "load_results",
    "load_taskset",
    "pareto_plot",
    "strip_thinking",
    "summarise",
    "to_markdown",
]

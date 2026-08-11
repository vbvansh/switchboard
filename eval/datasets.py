"""Loading evaluation tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from eval.grading import Check

TASKS_DIR = Path(__file__).parent / "tasks"
DIFFICULTIES = ("easy", "medium", "hard")


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    check: Check
    difficulty: str
    category: str

    @classmethod
    def from_dict(cls, raw: dict) -> Task:
        return cls(
            id=raw["id"],
            prompt=raw["prompt"],
            check=Check.from_dict(raw["check"]),
            difficulty=raw.get("difficulty", "unknown"),
            category=raw.get("category", "general"),
        )


@dataclass(frozen=True)
class TaskSet:
    name: str
    tasks: tuple[Task, ...]

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def filtered(
        self,
        difficulty: str | None = None,
        category: str | None = None,
        limit: int | None = None,
    ) -> TaskSet:
        tasks = self.tasks
        if difficulty:
            tasks = tuple(t for t in tasks if t.difficulty == difficulty)
        if category:
            tasks = tuple(t for t in tasks if t.category == category)
        if limit is not None:
            tasks = tasks[:limit]
        return TaskSet(name=self.name, tasks=tasks)

    def counts_by_difficulty(self) -> dict[str, int]:
        return {
            level: sum(1 for t in self.tasks if t.difficulty == level)
            for level in DIFFICULTIES
        }


def load_taskset(name: str = "builtin") -> TaskSet:
    path = TASKS_DIR / f"{name}.json"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in TASKS_DIR.glob("*.json")))
        raise FileNotFoundError(
            f"No task set named {name!r}. Available: {available or 'none'}"
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks = tuple(Task.from_dict(entry) for entry in raw["tasks"])

    ids = [task.id for task in tasks]
    if len(set(ids)) != len(ids):
        duplicates = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"Duplicate task ids in {name}: {sorted(duplicates)}")

    return TaskSet(name=raw.get("name", name), tasks=tasks)

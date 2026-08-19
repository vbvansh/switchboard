"""Loader for LLMRouterBench.

    LLMRouterBench: A Massive Benchmark and Unified Framework for LLM Routing
    Findings of ACL 2026 - arXiv:2601.07206

Two things about this archive shape the loader.

First, the directory layout is inconsistent: some benchmarks are
`<benchmark>/test/<model>/*.json`, others put models directly under the
benchmark, and split directories are variously named test, hybrid, v1,
subset_500, verified. So the loader walks for JSON files and reads the
benchmark and model names out of the file's own metadata rather than trying to
parse them from the path.

Second, it is 6.5 GB of JSON. Records are streamed one benchmark at a time and
written straight to Parquet, so peak memory stays bounded - loading it all at
once would not fit in a typical laptop's RAM.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from eval.benchmarks.schema import COLUMNS, QUERY_COLUMNS

logger = logging.getLogger(__name__)

SOURCE = "llmrouterbench"
DEFAULT_ROOT = Path("data/benchmarks/llmrouterbench/bench-release")

#: `index` identifies the same question across models - but only WITHIN a
#: split. Several benchmarks ship more than one question set (hle has both
#: `subset_500` and `test`, with different data_fingerprints) and the index
#: restarts at 1 in each. So the key must be split-qualified; using the raw
#: index would silently merge two different questions into one row.
QUERY_ID_FIELD = "index"


def benchmark_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir())


def _records_in(path: Path) -> Iterator[tuple[dict, dict]]:
    """Yield (file_metadata, record) pairs from one result file."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Skipping unreadable %s: %s", path.name, exc)
        return

    if not isinstance(blob, dict) or "records" not in blob:
        return

    for record in blob["records"]:
        yield blob, record


def load_benchmark_dir(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise every result file under one benchmark directory."""
    rows: list[tuple] = []
    queries: dict[str, str] = {}
    query_conflicts = 0

    for path in sorted(directory.rglob("*.json")):
        for blob, record in _records_in(path):
            benchmark = blob.get("dataset_name") or directory.name
            model = blob.get("model_name")
            raw_id = record.get(QUERY_ID_FIELD)
            if model is None or raw_id is None:
                continue

            # Split-qualified: see QUERY_ID_FIELD above.
            split = blob.get("split") or path.parent.parent.name
            query_id = f"{split}:{raw_id}"
            text = record.get("origin_query") or record.get("prompt") or ""

            # Guard the join key. If two models disagree about which question
            # an index refers to, the whole grid is meaningless.
            existing = queries.get(query_id)
            if existing is None:
                queries[query_id] = text
            elif existing != text:
                query_conflicts += 1

            score = record.get("score")
            rows.append(
                (
                    SOURCE,
                    benchmark,
                    query_id,
                    model,
                    float(score) if score is not None else 0.0,
                    float(record.get("cost") or 0.0),
                    int(record.get("prompt_tokens") or 0),
                    int(record.get("completion_tokens") or 0),
                    float("nan"),  # this dataset records no per-request latency
                )
            )

    if query_conflicts:
        logger.warning(
            "%s: %d records where the same index maps to a different question. "
            "Model comparisons on this benchmark may be unreliable.",
            directory.name,
            query_conflicts,
        )

    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    query_frame = pd.DataFrame(
        [
            (SOURCE, directory.name, query_id, text)
            for query_id, text in queries.items()
        ],
        columns=list(QUERY_COLUMNS),
    )
    return frame, query_frame


def iter_benchmarks(
    root: Path = DEFAULT_ROOT,
) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Stream one benchmark at a time so memory stays bounded."""
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Download it first:\n"
            "  python scripts/fetch_llmrouterbench.py --extract"
        )

    for directory in benchmark_dirs(root):
        frame, queries = load_benchmark_dir(directory)
        if frame.empty:
            logger.info("Skipping %s: no usable records", directory.name)
            continue
        yield directory.name, frame, queries

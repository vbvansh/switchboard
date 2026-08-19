"""Loader for xRouteBench.

    LLMRouter: Unified Infrastructure for Developing, Evaluating, and
    Deploying LLM Routers - ulab-ai/xRouteBench on HuggingFace

Smaller and tidier than LLMRouterBench: Parquet, ~47 MB, downloaded on demand.
Its models are open-weight rather than the commercial flagships, but it is the
only one of the two that records PER-REQUEST LATENCY, which is what the latency
SLA work depends on.

Only the text configurations are loaded. The dataset also covers video, visual
maths and time-series routing, none of which a chat-completions proxy serves.
"""

from __future__ import annotations

import logging

import pandas as pd

from eval.benchmarks.schema import COLUMNS, QUERY_COLUMNS

logger = logging.getLogger(__name__)

SOURCE = "xroutebench"
REPO = "ulab-ai/xRouteBench"
CANDIDATES_FILE = "llm_candidates/train.parquet"

#: Text-only configurations. Each contributes a train and a test split.
CONFIGS = ("llmrouter_generic",)
SPLITS = ("train", "test")


def _download(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO, filename, repo_type="dataset")


def load_prices() -> pd.DataFrame:
    """Per-model pricing, in USD per million tokens."""
    prices = pd.read_parquet(_download(CANDIDATES_FILE))
    return prices.set_index("model_name")[
        ["input_price_per_1m", "output_price_per_1m", "size", "service"]
    ]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (outcomes, queries) in the common schema."""
    prices = load_prices()
    parts = []

    for config in CONFIGS:
        for split in SPLITS:
            try:
                path = _download(f"{config}/{split}.parquet")
            except Exception as exc:  # noqa: BLE001 - split may not exist
                logger.info("Skipping %s/%s: %s", config, split, exc)
                continue
            frame = pd.read_parquet(path)
            frame["split"] = split
            parts.append(frame)

    if not parts:
        raise RuntimeError(f"No usable splits found in {REPO}")

    raw = pd.concat(parts, ignore_index=True)

    # The dataset has no question id, and its task_id column is not reliable
    # across benchmarks. The query text is the only thing that identifies a
    # question consistently, so it becomes the key - hashed to keep the
    # outcome table free of text.
    query_hash = (
        pd.util.hash_pandas_object(raw["query"], index=False)
        .astype("uint64")
        .astype(str)
    )
    raw["query_id"] = raw["task_name"].astype(str) + ":" + query_hash

    joined = raw.join(prices, on="model_name")
    unpriced = joined["input_price_per_1m"].isna()
    if unpriced.any():
        missing = sorted(joined.loc[unpriced, "model_name"].unique())
        logger.warning("No price for %s; those rows are dropped.", missing)
        joined = joined[~unpriced]

    cost = (
        joined["input_tokens"] * joined["input_price_per_1m"]
        + joined["output_tokens"] * joined["output_price_per_1m"]
    ) / 1_000_000

    outcomes = pd.DataFrame(
        {
            "source": SOURCE,
            "benchmark": joined["task_name"],
            "query_id": joined["query_id"],
            "model": joined["model_name"],
            # `performance` is already 0..1 for every metric in this dataset,
            # clipped defensively so a stray value cannot break validation.
            "correct": joined["performance"].astype(float).clip(0.0, 1.0),
            "cost_usd": cost,
            "prompt_tokens": joined["input_tokens"].astype("int64"),
            "output_tokens": joined["output_tokens"].astype("int64"),
            "latency_s": joined["response_time"].astype(float),
        },
        columns=list(COLUMNS),
    )

    # One model answering the same question in both splits would be a genuine
    # duplicate; keep the first and let validation catch anything unexpected.
    outcomes = outcomes.drop_duplicates(
        subset=["source", "benchmark", "query_id", "model"], keep="first"
    )

    queries = (
        joined[["task_name", "query_id", "query"]]
        .drop_duplicates(subset=["query_id"])
        .rename(columns={"task_name": "benchmark"})
        .assign(source=SOURCE)[list(QUERY_COLUMNS)]
    )

    return outcomes, queries

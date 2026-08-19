"""Download the LLMRouterBench archive.

The dataset is NOT redistributed in this repository - it carries no explicit
licence of its own, and its contents derive from 21+ upstream benchmarks with
mixed terms plus outputs from commercial models. This script fetches it from
the original source instead, so anyone reproducing our results gets the data
under whatever terms its authors intend.

    LLMRouterBench: A Massive Benchmark and Unified Framework for LLM Routing
    Findings of ACL 2026 - arXiv:2601.07206
    https://github.com/ynulihao/LLMRouterBench

Usage:
    python scripts/fetch_llmrouterbench.py [--extract]
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

REPO_ID = "NPULH/LLMRouterBench"
ARCHIVE = "bench-release.tar.gz"
DEST = Path("data/benchmarks/llmrouterbench")


def download() -> Path:
    from huggingface_hub import hf_hub_download

    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID}/{ARCHIVE} (~1.28 GB) into {DEST}")
    print("Resumable: re-run this script if it is interrupted.\n")

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=ARCHIVE,
        repo_type="dataset",
        local_dir=str(DEST),
    )
    print(f"\nDownloaded to {path}")
    return Path(path)


def extract(archive: Path) -> None:
    """Extract with path validation.

    A tar archive can contain entries like ../../etc/passwd, so a naive
    extractall can write outside the target directory. Python 3.12 added a
    filter for exactly this; it is passed explicitly rather than relying on the
    default, which differs by version.
    """
    target = archive.parent
    print(f"Extracting into {target} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(target, filter="data")
    print("Done.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extract", action="store_true", help="Unpack the archive after download."
    )
    args = parser.parse_args()

    archive = download()
    if args.extract:
        extract(archive)
    else:
        print(f"\nExtract it with:\n  tar xzf {archive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

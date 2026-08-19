"""Public routing benchmarks, normalised into one shape.

The datasets themselves are NOT redistributed here - neither declares a licence
and both derive from many upstream benchmarks with mixed terms. They are
downloaded from their original sources on demand. See scripts/ and the README.
"""

# NOTE: the `replay` FUNCTION is deliberately not re-exported here. This
# package contains a module called `replay`, and exporting a function of the
# same name shadows it - so `from eval.benchmarks import replay` would hand
# back the function and every module-level access would fail. Import it as
# `from eval.benchmarks.replay import replay` instead.
from eval.benchmarks.replay import (
    BASELINE_STRATEGIES,
    REFERENCE_STRATEGIES,
    ReplayResult,
    compare,
    oracle_choices,
)
from eval.benchmarks.schema import (
    COLUMNS,
    QUERY_COLUMNS,
    BenchmarkError,
    Grid,
    validate,
)
from eval.benchmarks.store import (
    SOURCES,
    BenchmarkFrame,
    build,
    is_cached,
    load,
    load_queries,
)

__all__ = [
    "BASELINE_STRATEGIES",
    "COLUMNS",
    "REFERENCE_STRATEGIES",
    "QUERY_COLUMNS",
    "SOURCES",
    "BenchmarkError",
    "BenchmarkFrame",
    "Grid",
    "ReplayResult",
    "build",
    "compare",
    "is_cached",
    "load",
    "load_queries",
    "oracle_choices",
    "validate",
]

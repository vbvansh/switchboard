"""Moved to `switchboard.routing.features`.

The extractor is part of a trained router artifact, and joblib records the
module a class came from. While it lived here, every artifact pickled a
reference to `eval.benchmarks.features` - a package the Docker image
deliberately does not contain, because it carries 500 MB of research tooling a
server never runs.

The result was invisible: inside a container the artifact failed to unpickle,
the failure was caught safely, routing switched itself off, and /health said
"no router artifact loaded" with no hint as to why.

So it now lives in the shipped package, and this module re-exports it so
nothing in eval/ had to change.
"""

from switchboard.routing.features import (  # noqa: F401
    CODE_MARKERS,
    DEFAULT_MODE,
    EMBEDDING_DIMS,
    EMBEDDING_MAX_CHARS,
    EMBEDDING_MODEL,
    MATH_MARKERS,
    MODES,
    SURFACE_NAMES,
    FeatureExtractor,
    embed,
    surface_features,
    surface_matrix,
)

__all__ = [
    "CODE_MARKERS",
    "DEFAULT_MODE",
    "EMBEDDING_DIMS",
    "EMBEDDING_MAX_CHARS",
    "EMBEDDING_MODEL",
    "MATH_MARKERS",
    "MODES",
    "SURFACE_NAMES",
    "FeatureExtractor",
    "embed",
    "surface_features",
    "surface_matrix",
]

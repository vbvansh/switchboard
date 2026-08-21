"""Turning a question into numbers a classifier can learn from.

Three representations, selected by `mode`:

* ``surface``   - length, digit density, code markers. Crude but genuinely
                  predictive, and free.
* ``tfidf``     - the default. Word and phrase counts, weighted so that rare,
                  informative words matter more than common ones. Captures what
                  the question is actually *about*, and fits thousands of
                  documents in under a second.
* ``embedding`` - a neural model reads the question and produces 384 numbers.
                  Richer than TF-IDF in principle, and OPT-IN because it is not
                  viable on modest hardware: measured at roughly 0.5 texts per
                  second on the development machine, versus thousands per
                  second for TF-IDF. On a machine with a real GPU, use it.

Surface features are always included and always scaled; the text
representation is appended to them.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODES = ("surface", "tfidf", "embedding")
DEFAULT_MODE = "tfidf"

#: Small, fast, CPU-only. 384 dimensions. Downloaded once (~130 MB).
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMS = 384

#: Beyond this the embedding model truncates anyway, so passing more only costs
#: tokenizer time.
EMBEDDING_MAX_CHARS = 2000

CACHE_DIR = Path("data/benchmarks/cache/embeddings")

CODE_MARKERS = re.compile(r"```|def |class |import |;|\{|\}|<\w+>|=>|::")
MATH_MARKERS = re.compile(r"[=+\-*/^]|\\\w+|\d+\.\d+|\$")

#: Names of the surface features, in the order `surface_features` returns them.
SURFACE_NAMES = (
    "log_chars",
    "log_words",
    "mean_word_len",
    "digit_ratio",
    "upper_ratio",
    "punct_ratio",
    "n_questions",
    "n_newlines",
    "has_code",
    "has_maths",
)


def surface_features(text: str) -> np.ndarray:
    """Cheap statistics about the raw text.

    Lengths are log-scaled: the difference between a 50 and a 500 character
    question matters far more than between 5,000 and 5,450, and a raw count
    would let one enormous prompt dominate the scale.
    """
    chars = len(text)
    words = text.split()
    n_words = len(words)
    alpha = sum(c.isalpha() for c in text) or 1

    return np.array(
        [
            np.log1p(chars),
            np.log1p(n_words),
            (sum(len(w) for w in words) / n_words) if n_words else 0.0,
            sum(c.isdigit() for c in text) / (chars or 1),
            sum(c.isupper() for c in text) / alpha,
            sum(not c.isalnum() and not c.isspace() for c in text) / (chars or 1),
            float(text.count("?")),
            np.log1p(text.count("\n")),
            float(bool(CODE_MARKERS.search(text))),
            float(bool(MATH_MARKERS.search(text))),
        ],
        dtype=np.float32,
    )


def surface_matrix(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, len(SURFACE_NAMES)), dtype=np.float32)
    return np.vstack([surface_features(t) for t in texts])


# --- Embeddings (opt-in) ----------------------------------------------------


def _cache_key(texts: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(EMBEDDING_MODEL.encode())
    for text in texts:
        digest.update(text.encode("utf-8", "replace"))
        digest.update(b"\x00")
    return digest.hexdigest()[:20]


def embed(
    texts: list[str], use_cache: bool = True, batch_size: int = 128
) -> np.ndarray:
    """Embed a batch of texts, caching the result to disk.

    Slow on CPU. Cached aggressively because re-running an experiment must not
    mean re-paying that cost.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIMS), dtype=np.float32)

    path = CACHE_DIR / f"{_cache_key(texts)}.npy"
    if use_cache and path.exists():
        return np.load(path)

    from fastembed import TextEmbedding

    logger.info("Embedding %d texts with %s ...", len(texts), EMBEDDING_MODEL)
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    clipped = [t[:EMBEDDING_MAX_CHARS] for t in texts]
    vectors = np.asarray(
        list(model.embed(clipped, batch_size=batch_size)), dtype=np.float32
    )

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
    return vectors


# --- The extractor ----------------------------------------------------------


class FeatureExtractor:
    """Fit on training questions, then transform any question.

    Follows the sklearn fit/transform contract deliberately: the TF-IDF
    vocabulary is learned from the TRAINING questions only. Fitting it on the
    test set too would leak information about the held-out questions into the
    model and inflate the score.
    """

    def __init__(self, mode: str = DEFAULT_MODE, max_features: int = 3000) -> None:
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}. Available: {', '.join(MODES)}")
        self.mode = mode
        self.max_features = max_features
        self._vectorizer = None
        self._scaler = None
        self._embedding_cache: dict[str, np.ndarray] = {}

    def fit(self, texts: list[str]) -> FeatureExtractor:
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler().fit(surface_matrix(texts))

        if self.mode == "tfidf":
            self._vectorizer = self._fit_vectorizer(texts)

        elif self.mode == "embedding":
            self._warm_embeddings(texts)

        return self

    def _fit_vectorizer(self, texts: list[str]):
        """Build the TF-IDF vocabulary, adapting to how much text there is.

        `min_df=2` drops terms appearing in only one question - usually
        identifiers the model would otherwise memorise. On a small corpus that
        can prune everything, so the threshold relaxes rather than crashing:
        someone training on a 60-question suite should get a working router,
        not a stack trace.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        def build(min_df: int, stop_words):
            return TfidfVectorizer(
                max_features=self.max_features,
                # Unigrams and bigrams: "deadlock" is a signal on its own, and
                # "prove that" carries more than either word alone.
                ngram_range=(1, 2),
                min_df=min_df,
                sublinear_tf=True,
                stop_words=stop_words,
            ).fit(texts)

        attempts = [(2, "english"), (1, "english"), (1, None)]
        for min_df, stop_words in attempts:
            if min_df == 2 and len(texts) < 50:
                continue  # too little text for a document-frequency cut-off
            try:
                return build(min_df, stop_words)
            except ValueError as exc:
                logger.debug("TF-IDF fit failed (min_df=%s): %s", min_df, exc)

        raise ValueError(
            "Could not build a TF-IDF vocabulary from these questions - they "
            "may be empty. Try --features surface."
        )

    def _warm_embeddings(self, texts: list[str]) -> None:
        missing = [t for t in dict.fromkeys(texts) if t not in self._embedding_cache]
        if not missing:
            return
        for text, vector in zip(missing, embed(missing), strict=True):
            self._embedding_cache[text] = vector

    def transform(self, texts: list[str]):
        """Returns a dense array, or a sparse matrix in tfidf mode.

        Sparse is kept sparse on purpose: 3,000 TF-IDF columns across thousands
        of questions would be a large dense array for almost no information,
        and logistic regression accepts sparse input directly.
        """
        if self._scaler is None:
            raise RuntimeError("FeatureExtractor.fit must be called first.")

        surface = self._scaler.transform(surface_matrix(texts))

        if self.mode == "surface":
            return surface

        if self.mode == "tfidf":
            from scipy.sparse import csr_matrix, hstack

            parts = [csr_matrix(surface), self._vectorizer.transform(texts)]
            return hstack(parts).tocsr()

        self._warm_embeddings(texts)
        vectors = np.vstack([self._embedding_cache[t] for t in texts])
        return np.hstack([surface, vectors])

    def transform_one(self, text: str):
        return self.transform([text])

    def describe(self) -> str:
        if self.mode == "tfidf" and self._vectorizer is not None:
            vocabulary = len(self._vectorizer.vocabulary_)
            return f"tfidf ({vocabulary} terms) + {len(SURFACE_NAMES)} surface"
        if self.mode == "embedding":
            return f"embeddings ({EMBEDDING_DIMS}) + {len(SURFACE_NAMES)} surface"
        return f"{len(SURFACE_NAMES)} surface features"

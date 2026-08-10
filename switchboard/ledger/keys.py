"""API key generation and verification.

Keys are hashed with a plain SHA-256, deliberately - not bcrypt or argon2.

Those algorithms are slow *on purpose*, to make brute-forcing human-chosen
passwords expensive. An API key here is 32 bytes of cryptographic randomness,
so there is nothing to brute-force: guessing one is infeasible regardless of
hash speed. A slow hash would only add latency to every request.

The hash is unsalted so an incoming key can be looked up directly by its hash.
That is safe for the same reason - high-entropy secrets are not vulnerable to
the rainbow-table attacks that salting defends against.
"""

from __future__ import annotations

import hashlib
import secrets

KEY_PREFIX = "sk-swbd-"
KEY_BYTES = 32


def generate_api_key() -> str:
    """Create a new key. Shown to the operator once, then never recoverable."""
    return KEY_PREFIX + secrets.token_urlsafe(KEY_BYTES)


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def looks_like_api_key(value: str) -> bool:
    return value.startswith(KEY_PREFIX)


def redact(raw_key: str) -> str:
    """Safe-to-log fragment, e.g. 'sk-swbd-...a8f3'."""
    return f"{KEY_PREFIX}...{raw_key[-4:]}" if len(raw_key) > 4 else "sk-swbd-..."


def extract_bearer_token(authorization_header: str | None) -> str | None:
    """Pull the key out of an `Authorization` header.

    Accepts both "Bearer <key>" and a bare key, because some OpenAI-compatible
    clients send the raw value.
    """
    if not authorization_header:
        return None

    header = authorization_header.strip()
    if not header:
        return None

    # Split on the scheme rather than a "bearer " prefix: a header of exactly
    # "Bearer " collapses to "Bearer" after stripping, which a prefix check
    # misses - and it would then be treated as the key itself.
    scheme, _, rest = header.partition(" ")
    if scheme.lower() == "bearer":
        return rest.strip() or None
    return header

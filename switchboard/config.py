"""Configuration for Switchboard.

The one rule enforced here that matters: the inference provider must be local.
See `_require_local_host` - it is the mechanical guarantee that no request ever
leaves this machine and that no paid API can ever be contacted, even by
accident. There is deliberately no escape hatch. If remote Ollama is ever
needed, add an explicit allowlist rather than loosening this check.
"""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Hostnames that unambiguously resolve to this machine.
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

# Model name a client can send to hand model choice to Switchboard. Milestone 1
# resolves it to `default_model`; from milestone 3 it triggers real routing.
AUTO_MODEL = "auto"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SWITCHBOARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Provider -----------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"

    # Milestone 1 sends everything here. The tier ladder replaces it later.
    default_model: str = "qwen2.5:3b"

    # --- Server -------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000

    # --- Timeouts -----------------------------------------------------------
    # Read timeout is deliberately large: a cold 7b load on a 4GB GPU spills to
    # system RAM and can take well over a minute before the first token.
    connect_timeout: float = 10.0
    read_timeout: float = 600.0

    @field_validator("ollama_base_url")
    @classmethod
    def _require_local_host(cls, value: str) -> str:
        parsed = urlparse(value)

        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                f"ollama_base_url must be an http(s) URL, got {value!r}"
            )

        host = parsed.hostname
        if host is None:
            raise ValueError(f"ollama_base_url has no host: {value!r}")

        if host.lower() not in LOCAL_HOSTS:
            raise ValueError(
                f"Refusing non-local provider host {host!r}. Switchboard runs "
                "100% locally against Ollama; it holds no API keys and must "
                "never contact a remote inference endpoint."
            )

        return value.rstrip("/")

    @property
    def openai_compat_url(self) -> str:
        """Ollama's OpenAI-compatible API root."""
        return f"{self.ollama_base_url}/v1"


settings = Settings()

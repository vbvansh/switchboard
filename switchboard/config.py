"""Runtime configuration.

Providers and models are NOT configured here - they live in providers.yaml so
users can add them without touching Python. This file holds only settings that
change how the process behaves.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model name a client can send to hand model choice to Switchboard.
AUTO_MODEL = "auto"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SWITCHBOARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Providers ----------------------------------------------------------
    providers_file: str = str(PROJECT_ROOT / "providers.yaml")

    # Model used when a client sends "auto" or omits the field. Until the
    # router lands, this serves every request.
    default_model: str = "qwen2.5:3b"

    # A trained router artifact. When set and loadable, `model: "auto"`
    # routes; otherwise it falls back to `default_model` and says so in
    # /health. A stale artifact must never take the service down.
    router_path: str = "data/router.joblib"

    # Minimum predicted chance of success before a model is accepted. Raising
    # it escalates more often: more accuracy, more cost. Callers can override
    # per request with the X-Switchboard-Min-Quality header.
    router_min_quality: float = 0.5

    # Refuse to start if any enabled provider is not on this machine.
    #
    # Off by default, because talking to providers is the point of the product.
    # Switch it on and Switchboard becomes physically incapable of sending
    # prompts outside the host - which is what some organisations need before
    # they will let a gateway near their data. Enforced at startup, not per
    # request, so a violation is caught immediately rather than at 3am.
    local_only: bool = False

    # --- Server -------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000

    # --- Ledger -------------------------------------------------------------
    database_url: str = "sqlite:///data/switchboard.db"

    # Store the full `messages` array of every request.
    #
    # OFF by default, and it must stay that way. Turning it on means the
    # database records everything users type - which in a real deployment
    # includes customer data, credentials, and personal information, with the
    # legal exposure that carries. Nobody should acquire that liability by
    # installing software and leaving the defaults alone.
    #
    # It is genuinely useful: the routing classifier learns from real examples.
    # So it stays available, as a deliberate opt-in the operator has to read
    # about and choose. Safe by default, useful on request.
    store_prompts: bool = False


settings = Settings()

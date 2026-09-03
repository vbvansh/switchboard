"""Runtime configuration.

Providers and models are NOT configured here - they live in providers.yaml so
users can add them without touching Python. This file holds only settings that
change how the process behaves.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from switchboard import paths

#: Kept as a name because other modules import it. Prefer `switchboard.paths`
#: for anything new: this points at the package's parent, which is the repo
#: root in a checkout and a site-packages directory once installed.
PROJECT_ROOT = paths.BUNDLE_ROOT

# Model name a client can send to hand model choice to Switchboard.
AUTO_MODEL = "auto"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SWITCHBOARD_",
        # A .env beside the working directory still wins, which is what a
        # developer in a checkout expects. The second path is where an
        # installed copy keeps it, so `switchboard init` has somewhere to write
        # that survives an upgrade.
        env_file=(".env", str(paths.config_dir() / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Providers ----------------------------------------------------------
    providers_file: str = str(paths.providers_file())

    # Model used when a client sends "auto" or omits the field. Until the
    # router lands, this serves every request.
    default_model: str = "qwen2.5:3b"

    # A trained router artifact. When set and loadable, `model: "auto"`
    # routes; otherwise it falls back to `default_model` and says so in
    # /health. A stale artifact must never take the service down.
    router_path: str = str(paths.router_path())

    # Minimum predicted chance of success before a model is accepted. Raising
    # it escalates more often: more accuracy, more cost. Callers can override
    # per request with the X-Switchboard-Min-Quality header.
    router_min_quality: float = 0.5

    # Minimum spread between the best and worst predicted model before the
    # router is allowed to act on its own prediction.
    #
    # Measured in C.4: shown a prompt unlike its training data, the router
    # returned 0.67 to 0.87 for EVERY model - no discrimination at all, and it
    # said nothing about that. Everything went to the cheapest model and the
    # logs implied a decision had been made.
    #
    # Below this spread the router abstains and the ladder policy decides,
    # which is the same outcome with an honest reason attached - and it makes
    # "the router abstained on 80% of your traffic" a visible fact rather than
    # a silent one. Set to 0 to disable abstention.
    router_min_spread: float = 0.08

    # --- Response cache -----------------------------------------------------
    # Identical requests are answered from memory for nothing. Set entries to 0
    # to switch it off. Only deterministic, non-streaming requests qualify.
    cache_max_entries: int = 1000
    cache_ttl_s: float = 3600.0

    # --- Retries ------------------------------------------------------------
    # Applies only to transient failures - timeouts, 429, 5xx. A malformed
    # request is never retried; it would fail identically and cost twice.
    retry_attempts: int = 3
    retry_base_delay_s: float = 0.5

    # Watch what routing WOULD do, without letting it do it. Requests are
    # served exactly as they would be with no router, and the router's opinion
    # is recorded alongside. This is how a team trials routing on their own
    # traffic before trusting it. See switchboard/shadow.py.
    shadow_mode: bool = False

    # --- Usage policy (guardrails) -----------------------------------------
    # "off", "flag" or "block". See switchboard/guardrails.py for why the
    # default is flag and not block:
    #
    #   missing a personal request costs a fraction of a cent;
    #   wrongly blocking a real one stops an engineer working.
    #
    # In flag mode nothing is refused. A label and the names of the rules that
    # matched are written to the ledger - never the prompt text, which stays
    # behind store_prompts as before.
    guardrails_mode: str = "flag"

    # Optional YAML rule file. It REPLACES the built-in rules rather than
    # adding to them, so a shipped rule that keeps flagging your team's real
    # work can actually be removed. A file that fails to load stops startup:
    # a policy an operator thinks is running must never be silently off.
    guardrails_file: str | None = None

    # --- Verification and escalation ---------------------------------------
    # Look at each answer and decide whether it obviously failed.
    #
    #   "off"       do nothing
    #   "flag"      check every answer, record what fired, change nothing
    #   "escalate"  additionally retry on a stronger model - but ONLY for the
    #               checks where a retry would actually help
    #
    # DEFAULT IS "flag", and the reason is the same one that keeps blocking off
    # in the usage policy: escalation makes a second provider call, which
    # doubles the cost of the requests it touches. Nobody should acquire a
    # larger bill by installing software and leaving the defaults alone. Run in
    # flag mode first, look at how often checks fire on YOUR traffic, then
    # decide. See switchboard/verification.py.
    verify_mode: str = "flag"

    # How many times one request may be retried on a stronger model. One is
    # almost always right: if the cheap model returned nothing and the next one
    # up also returned nothing, a third call is unlikely to help and certain to
    # cost.
    max_escalations: int = 1

    # --- Rate limiting ------------------------------------------------------
    # Requests per minute per user. A monthly budget does not stop someone
    # spending it in ninety seconds; this does. Set to 0 to disable.
    # Individual users can be given a different limit in the database.
    rate_limit_per_minute: int = 60

    # --- Failover -----------------------------------------------------------
    # Consecutive failures before a provider is skipped, and for how long.
    # Without this, every request to a dead provider waits for its full
    # timeout before failing over. Set the threshold to 0 to disable.
    breaker_failure_threshold: int = 5
    breaker_cooldown_s: float = 30.0

    # Refuse to start if any enabled provider is not on this machine.
    #
    # Off by default, because talking to providers is the point of the product.
    # Switch it on and Switchboard becomes physically incapable of sending
    # prompts outside the host - which is what some organisations need before
    # they will let a gateway near their data. Enforced at startup, not per
    # request, so a violation is caught immediately rather than at 3am.
    local_only: bool = False

    # --- Dashboard access ---------------------------------------------------
    # Password for /dashboard. Empty means open, which is the historical
    # behaviour and right on a laptop.
    #
    # SET THIS BEFORE PUTTING SWITCHBOARD ON A PUBLIC URL. The page shows spend
    # per developer by name. It shows no prompt text and no API keys - that is
    # deliberate - but "who is spending what" is not something to hand to
    # anyone who finds the link.
    #
    # HTTP Basic, because a browser prompts for it natively and it needs no
    # login page, no cookies and no session store. Any username is accepted;
    # only the password is checked. Put it behind HTTPS - Basic sends the
    # password on every request, and every deployment guide here terminates TLS
    # in front of the app.
    #
    # /metrics is deliberately NOT covered: it carries no prompt text, no keys
    # and no user names, and a scrape endpoint that needs credentials is one
    # nobody gets round to configuring.
    dashboard_password: str = ""

    # --- Server -------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000

    # --- Ledger -------------------------------------------------------------
    database_url: str = paths.default_database_url()

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

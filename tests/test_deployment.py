"""Deployment concerns: health probes, database backends, config portability.

These cover the things that only bite once someone else runs the software - on
a different machine, in a container, behind an orchestrator.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from switchboard.catalog import ModelCatalog, expand_env
from switchboard.ledger import Database

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Port 1 refuses immediately, and connect_timeout caps the wait if a platform
# decides to hang instead. Without it these tests add minutes to the suite.
UNREACHABLE_POSTGRES = (
    "postgresql+psycopg://u:p@127.0.0.1:1/nope?connect_timeout=1"
)


# --- Liveness vs readiness -------------------------------------------------


def test_liveness_ignores_dependencies(client: TestClient, provider) -> None:
    """The important one.

    A failing liveness probe makes an orchestrator kill the container. If this
    checked providers, an outage at a provider would put Switchboard into a
    restart loop - fixing nothing and destroying its own logs.
    """
    provider.healthy = False
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_liveness_needs_no_credentials(client: TestClient) -> None:
    assert client.get("/health/live").status_code == 200


def test_readiness_passes_when_everything_works(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_readiness_fails_when_no_provider_answers(
    client: TestClient, provider
) -> None:
    """503 tells a load balancer to stop sending traffic, without a restart."""
    provider.healthy = False
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_readiness_fails_when_the_database_is_gone(client: TestClient) -> None:
    """Serving requests we cannot bill for would lose the accounting."""
    client.app.state.database = Database(UNREACHABLE_POSTGRES)
    assert client.get("/health/ready").status_code == 503


def test_readiness_never_raises(client: TestClient) -> None:
    """A probe that throws turns "dependency down" into "app broken".

    503 tells an orchestrator to route around this instance; 500 reads as a bug
    in Switchboard and sends people debugging the wrong thing.
    """
    client.app.state.database = Database(UNREACHABLE_POSTGRES)
    assert client.get("/health/ready").status_code == 503  # not 500


def test_health_reports_database_reachability(client: TestClient) -> None:
    assert client.get("/health").json()["database_reachable"] is True


def test_health_endpoints_report_a_version(client: TestClient) -> None:
    """An operator debugging a deployment needs to know what is running."""
    assert client.get("/health/live").json()["version"]
    assert client.get("/health").json()["version"]


# --- Database backends -----------------------------------------------------


def test_sqlite_gets_thread_safety_settings(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{tmp_path.as_posix()}/t.db")
    try:
        assert db.engine.dialect.name == "sqlite"
        assert db.is_reachable()
    finally:
        db.dispose()


def test_postgres_url_configures_a_pool_without_connecting() -> None:
    """Engine creation is lazy, so this validates configuration only.

    pool_pre_ping matters: pooled connections get closed underneath you by idle
    timeouts and server restarts, and without it the first request after any
    such event fails.
    """
    db = Database("postgresql+psycopg://u:p@localhost:5432/switchboard")
    try:
        assert db.engine.dialect.name == "postgresql"
        assert db.engine.pool._pre_ping is True
        # SQLite-only arguments must not leak into a server-backed driver.
        assert "check_same_thread" not in str(db.engine.url.query)
    finally:
        db.dispose()


def test_unreachable_database_reports_itself() -> None:
    db = Database(UNREACHABLE_POSTGRES)
    try:
        assert db.is_reachable() is False
    finally:
        db.dispose()


# --- Config portability ----------------------------------------------------


@pytest.mark.parametrize(
    ("template", "env", "expected"),
    [
        ("${A}", {"A": "x"}, "x"),
        ("${A:-fallback}", {}, "fallback"),
        ("${A:-fallback}", {"A": "set"}, "set"),
        ("http://${HOST:-localhost}:11434/v1", {}, "http://localhost:11434/v1"),
        (
            "http://${HOST:-localhost}:11434/v1",
            {"HOST": "host.docker.internal"},
            "http://host.docker.internal:11434/v1",
        ),
        ("no placeholders here", {}, "no placeholders here"),
        ("${MISSING}", {}, ""),
    ],
)
def test_env_expansion(template, env, expected, monkeypatch) -> None:
    for key in ("A", "HOST", "MISSING"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert expand_env(template) == expected


def test_the_shipped_catalog_relocates_with_an_env_var(monkeypatch) -> None:
    """One providers.yaml must work on a laptop and inside a container."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")
    catalog = ModelCatalog.load()
    assert (
        catalog.providers["ollama-local"].base_url
        == "http://host.docker.internal:11434/v1"
    )


def test_the_shipped_catalog_defaults_to_localhost(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    catalog = ModelCatalog.load()
    assert catalog.providers["ollama-local"].base_url == "http://localhost:11434/v1"


# --- Packaging -------------------------------------------------------------


def test_docker_files_exist() -> None:
    for name in ("Dockerfile", "docker-compose.yml", ".dockerignore"):
        assert (PROJECT_ROOT / name).exists(), name
    assert (PROJECT_ROOT / "docker" / "entrypoint.sh").exists()


def test_entrypoint_uses_exec_so_signals_reach_the_server() -> None:
    """Without exec, the shell keeps PID 1 and swallows SIGTERM - in-flight
    requests get killed instead of finishing."""
    entrypoint = (PROJECT_ROOT / "docker" / "entrypoint.sh").read_text()
    assert "exec python -m switchboard serve" in entrypoint


def test_dockerignore_excludes_secrets_and_local_data() -> None:
    """A committed image must never carry keys or someone's ledger."""
    ignored = (PROJECT_ROOT / ".dockerignore").read_text()
    for pattern in (".env", "data/", "*.db", ".git/"):
        assert pattern in ignored, pattern


def test_runtime_requirements_exclude_the_research_stack() -> None:
    """The server image should not carry tools it never runs."""
    runtime = (PROJECT_ROOT / "requirements.txt").read_text().lower()
    for heavy in ("matplotlib", "scikit-learn", "datasets", "pytest", "fastembed"):
        assert heavy not in runtime, f"{heavy} belongs in requirements-dev.txt"


def test_runtime_requirements_cover_what_the_server_imports() -> None:
    runtime = (PROJECT_ROOT / "requirements.txt").read_text().lower()
    for needed in ("fastapi", "uvicorn", "httpx", "sqlalchemy", "alembic", "pyyaml"):
        assert needed in runtime, needed

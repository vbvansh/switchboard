"""Being installable, and putting files where they belong.

The bug behind this file was invisible and expensive: paths were computed as
"two folders up from this source file". In a checkout that is the repository
root. Installed with pip it is inside `site-packages` - unwritable, and
replaced on the next upgrade.

The database was worse. `sqlite:///data/switchboard.db` is RELATIVE, so it
resolved against whatever directory the user was standing in. Start the server
from a different folder and every user, budget and spending record appears to
be gone - with no error, because SQLite cheerfully creates a new empty file.

Two properties are locked down here:

1. **A checkout keeps working exactly as before.** Nobody's development setup
   moves because we made the thing installable.
2. **Everything the running server needs ships inside the package.** Migrations
   especially: the server refuses to start against a schema it does not
   recognise, so a copy installed without them could never start at all.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from switchboard import paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text("utf-8"))


# --- The manifest -----------------------------------------------------------


def test_the_project_can_actually_be_built() -> None:
    """Without [build-system], `pip install` does not fail helpfully - it does
    not know how to build anything at all."""
    assert MANIFEST["build-system"]["build-backend"]
    assert MANIFEST["build-system"]["requires"]


def test_the_version_has_exactly_one_source() -> None:
    """Three files once held three different answers - 0.1.0, 0.3.0 and 0.4.0 -
    and nothing noticed until a release check compared them. A wrong version in
    a published package cannot be fixed: PyPI never reuses a number."""
    import switchboard
    from switchboard.api import app

    assert "version" in MANIFEST["project"].get("dynamic", [])
    assert "version" not in MANIFEST["project"], "a literal version has crept back"
    assert MANIFEST["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "switchboard.__version__"
    }
    assert app.version == switchboard.__version__


def test_the_switchboard_command_is_declared() -> None:
    """This single line is what puts `switchboard` on a user's PATH."""
    assert MANIFEST["project"]["scripts"]["switchboard"] == (
        "switchboard.__main__:cli"
    )


def test_the_entry_point_actually_resolves() -> None:
    """A typo here installs cleanly and then fails the first time anybody runs
    the command."""
    module_path, _, attribute = MANIFEST["project"]["scripts"][
        "switchboard"
    ].partition(":")
    module = __import__(module_path, fromlist=[attribute])
    assert callable(getattr(module, attribute))


def test_runtime_dependencies_are_declared() -> None:
    """pip reads this list, not requirements.txt. An empty one installs a
    package that cannot import anything it needs."""
    declared = " ".join(MANIFEST["project"]["dependencies"]).lower()
    for needed in ("fastapi", "uvicorn", "sqlalchemy", "alembic", "httpx", "typer"):
        assert needed in declared, needed


def test_the_two_dependency_lists_agree() -> None:
    """requirements.txt is for humans, pyproject is for pip. They drift apart
    silently, and the failure only shows up on somebody else's machine."""
    declared = " ".join(MANIFEST["project"]["dependencies"]).lower()
    text = (PROJECT_ROOT / "requirements.txt").read_text("utf-8").lower()

    named = {
        line.split(">=")[0].split("[")[0].strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    for package in named:
        assert package in declared, (
            f"{package} is in requirements.txt but not in pyproject"
        )


def test_the_research_stack_is_optional_not_required() -> None:
    """pandas, matplotlib and datasets are for measuring routing, never for
    serving it. In `dependencies` they would land on every user's machine."""
    required = " ".join(MANIFEST["project"]["dependencies"]).lower()
    for heavy in ("pandas", "matplotlib", "datasets", "fastembed"):
        assert heavy not in required, f"{heavy} should be an optional extra"
    research = " ".join(
        MANIFEST["project"]["optional-dependencies"]["research"]
    ).lower()
    assert "pandas" in research


def test_only_the_application_is_shipped() -> None:
    """`eval/` and `tests/` do not belong in a wheel a server installs."""
    include = MANIFEST["tool"]["setuptools"]["packages"]["find"]["include"]
    assert include == ["switchboard*"]


# --- What has to be inside the package --------------------------------------


def test_migrations_live_inside_the_package() -> None:
    """THE one that would break an install completely.

    The server refuses to start against a schema it does not recognise, and
    `db upgrade` is what fixes that. Both need the migration scripts. Outside
    the package, pip does not install them and the server can never start.
    """
    migrations = PROJECT_ROOT / "switchboard" / "migrations" / "versions"
    assert migrations.is_dir()
    assert list(migrations.glob("*.py")), "no migration scripts found"


def test_alembic_config_lives_inside_the_package() -> None:
    assert (PROJECT_ROOT / "switchboard" / "alembic.ini").exists()


def test_schema_module_points_inside_the_package() -> None:
    from switchboard import schema

    package = PROJECT_ROOT / "switchboard"
    assert schema.MIGRATIONS_DIR.is_relative_to(package)
    assert schema.ALEMBIC_INI.is_relative_to(package)


@pytest.mark.parametrize(
    "pattern",
    ["alembic.ini", "*.jsonl", "migrations/versions/*.py"],
)
def test_non_python_files_are_declared_as_package_data(pattern: str) -> None:
    """setuptools ships .py files automatically and nothing else. Anything not
    listed here is simply absent after `pip install`, with no warning."""
    declared = MANIFEST["tool"]["setuptools"]["package-data"]["switchboard"]
    assert pattern in declared


def test_every_shipped_data_file_is_covered_by_a_pattern() -> None:
    """Catches the next non-Python file somebody adds to the package and
    forgets to declare."""
    import fnmatch

    package = PROJECT_ROOT / "switchboard"
    patterns = MANIFEST["tool"]["setuptools"]["package-data"]["switchboard"]

    for path in package.rglob("*"):
        if path.is_dir() or path.suffix in {".py", ".pyc"}:
            continue
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(package).as_posix()
        assert any(
            fnmatch.fnmatch(relative, pattern) for pattern in patterns
        ), f"{relative} is in the package but not declared as package-data"


# --- Where files go ---------------------------------------------------------


def test_a_checkout_keeps_everything_where_it_was() -> None:
    """Making it installable must not move a developer's setup."""
    assert paths.is_bundled_layout()
    assert paths.config_dir() == PROJECT_ROOT
    assert paths.providers_file() == PROJECT_ROOT / "providers.yaml"


def test_the_database_url_is_absolute() -> None:
    """THE bug. A relative sqlite path resolves against the working directory,
    so running from two folders silently uses two databases - and creates the
    second one rather than complaining."""
    url = paths.default_database_url()
    assert url.startswith("sqlite:///")
    assert Path(url.removeprefix("sqlite:///")).is_absolute()


def test_settings_use_an_absolute_database_path() -> None:
    from switchboard.config import settings

    assert not settings.database_url.startswith("sqlite:///data/")


def test_switchboard_home_overrides_everything(monkeypatch, tmp_path) -> None:
    """One switch to run two instances side by side, and what tests use so they
    can never touch a real ledger."""
    monkeypatch.setenv("SWITCHBOARD_HOME", str(tmp_path))
    assert paths.config_dir() == tmp_path
    assert paths.data_dir() == tmp_path / "data"
    assert str(tmp_path.as_posix()) in paths.default_database_url()


def test_an_installed_copy_uses_operating_system_directories(monkeypatch) -> None:
    """With no providers.yaml beside the package - which is what pip produces -
    files must go to the OS config and data directories rather than into
    site-packages, which is unwritable and replaced on upgrade."""
    monkeypatch.delenv("SWITCHBOARD_HOME", raising=False)
    monkeypatch.setattr(paths, "is_bundled_layout", lambda: False)

    config, data = paths.config_dir(), paths.data_dir()
    package = Path(paths.PACKAGE_ROOT)

    assert not config.is_relative_to(package)
    assert not data.is_relative_to(package)
    assert "switchboard" in str(config).lower()


def test_config_and_data_are_kept_apart(monkeypatch) -> None:
    """Different things to a user: config is edited and worth copying between
    machines, data is generated and worth backing up."""
    monkeypatch.delenv("SWITCHBOARD_HOME", raising=False)
    monkeypatch.setattr(paths, "is_bundled_layout", lambda: False)
    if paths.config_dir() == paths.data_dir():
        pytest.skip("this platform uses one directory for both")
    assert paths.config_dir() != paths.data_dir()


def test_the_data_directory_is_created_on_demand(monkeypatch, tmp_path) -> None:
    """SQLite will not create a missing parent directory. It reports "unable to
    open database file", which reads like a permissions problem and sends
    people looking in entirely the wrong place."""
    monkeypatch.setenv("SWITCHBOARD_HOME", str(tmp_path / "fresh"))
    created = paths.ensure_data_dir()
    assert created.is_dir()


def test_describe_reports_the_layout() -> None:
    described = paths.describe()
    assert described["layout"] in {"bundled", "installed"}
    assert described["providers_file"].endswith("providers.yaml")

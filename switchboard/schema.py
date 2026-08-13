"""Schema version management.

Wraps Alembic so the rest of the codebase never imports it directly, and so
`switchboard db upgrade` and the startup check share one implementation.

The rule this module exists to enforce: an application must never run against a
database whose shape it does not recognise. Doing so does not fail cleanly - it
fails later, halfway through a request, with an error that points at the wrong
place. Refusing to start, with an instruction for what to run, is far kinder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
MIGRATIONS_DIR = PROJECT_ROOT / "migrations"


class SchemaOutOfDate(RuntimeError):
    """The database exists but is not at the revision this code expects."""


@dataclass(frozen=True)
class SchemaStatus:
    current: str | None
    head: str
    #: True when the database has no version stamp at all - either brand new,
    #: or created before migrations existed.
    unmanaged: bool

    @property
    def up_to_date(self) -> bool:
        return self.current == self.head

    def describe(self) -> str:
        if self.up_to_date:
            return f"up to date (revision {self.head})"
        if self.unmanaged:
            return f"not initialised (expected revision {self.head})"
        return f"at revision {self.current}, expected {self.head}"


def alembic_config(database_url: str) -> Config:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    # env.py reads this back. Percent signs are escaped because ConfigParser
    # treats them as interpolation syntax, and they appear in URL-encoded
    # passwords for Postgres connection strings.
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def head_revision() -> str:
    """The newest migration in the repository."""
    script = ScriptDirectory(str(MIGRATIONS_DIR))
    head = script.get_current_head()
    if head is None:
        raise RuntimeError(f"No migrations found in {MIGRATIONS_DIR}")
    return head


def current_revision(database_url: str) -> str | None:
    """What the database says it is. None if it has never been migrated."""
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            return MigrationContext.configure(connection).get_current_revision()
    finally:
        engine.dispose()


def status(database_url: str) -> SchemaStatus:
    current = current_revision(database_url)
    return SchemaStatus(
        current=current, head=head_revision(), unmanaged=current is None
    )


def upgrade(database_url: str, revision: str = "head") -> None:
    """Apply every migration the database is missing."""
    command.upgrade(alembic_config(database_url), revision)


def downgrade(database_url: str, revision: str) -> None:
    """Roll the schema back to an earlier revision.

    Deliberately not exposed as a CLI command. Downgrading drops columns, and
    dropping a column destroys the data in it - not something to make one
    keystroke away in a tool people run on production databases. It exists so
    migrations can be tested as reversible, and for operators who know exactly
    what they are doing.
    """
    command.downgrade(alembic_config(database_url), revision)


def stamp(database_url: str, revision: str = "head") -> None:
    """Mark the database as being at a revision without running anything.

    For databases created before migrations existed: their tables are already
    correct, so running migration 0001 would fail on "table already exists".
    Stamping records the version and moves on.
    """
    command.stamp(alembic_config(database_url), revision)


def require_up_to_date(database_url: str) -> None:
    """Guard for application startup."""
    state = status(database_url)
    if state.up_to_date:
        return

    raise SchemaOutOfDate(
        f"Database schema is {state.describe()}.\n"
        f"  Database: {database_url}\n"
        "  Fix it with: switchboard db upgrade\n"
        "  If this database predates migrations and its tables already exist, "
        "run: switchboard db stamp-baseline"
    )

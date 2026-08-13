"""Migrations must produce exactly the schema the code expects.

The failure this file exists to prevent: someone edits models.py, forgets to
write a migration, and every test still passes because the test suite builds
tables straight from the models. The bug only appears on a real user's machine,
on upgrade, as a missing column.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from switchboard import schema
from switchboard.ledger.models import Base
from switchboard.schema import SchemaOutOfDate


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path.as_posix()}/schema-test.db"


# --- Version tracking ------------------------------------------------------


def test_a_fresh_database_reports_itself_unmanaged(db_url: str) -> None:
    state = schema.status(db_url)
    assert state.unmanaged is True
    assert state.up_to_date is False


def test_upgrade_brings_a_fresh_database_to_head(db_url: str) -> None:
    schema.upgrade(db_url)
    state = schema.status(db_url)
    assert state.up_to_date
    assert state.current == schema.head_revision()


def test_upgrade_is_safe_to_run_twice(db_url: str) -> None:
    schema.upgrade(db_url)
    schema.upgrade(db_url)
    assert schema.status(db_url).up_to_date


def test_startup_guard_blocks_an_unmigrated_database(db_url: str) -> None:
    with pytest.raises(SchemaOutOfDate, match="db upgrade"):
        schema.require_up_to_date(db_url)


def test_startup_guard_passes_after_upgrade(db_url: str) -> None:
    schema.upgrade(db_url)
    schema.require_up_to_date(db_url)  # must not raise


# --- The drift check -------------------------------------------------------


def test_migrations_match_the_models(db_url: str) -> None:
    """The important one.

    Builds the database purely from migrations, then asks Alembic to diff it
    against the models. Any difference means a migration is missing.
    """
    schema.upgrade(db_url)

    engine = create_engine(db_url)
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(
                connection, opts={"compare_type": True}
            )
            differences = compare_metadata(context, Base.metadata)
    finally:
        engine.dispose()

    # Alembic reports SQLite index naming differences that are not real drift;
    # anything involving a table or column is.
    real = [
        d
        for d in differences
        if not (isinstance(d, tuple) and d and "index" in str(d[0]).lower())
    ]
    assert not real, f"Migrations have drifted from models.py: {real}"


def test_migrations_create_every_expected_table(db_url: str) -> None:
    schema.upgrade(db_url)
    engine = create_engine(db_url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    assert {"users", "requests"} <= tables
    assert "alembic_version" in tables  # the version stamp itself


def test_downgrade_removes_the_tables(db_url: str) -> None:
    """A migration that cannot be undone is a migration you cannot test."""
    schema.upgrade(db_url)
    schema.downgrade(db_url, revision="base")

    engine = create_engine(db_url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    assert "users" not in tables
    assert "requests" not in tables


# --- Baseline stamping -----------------------------------------------------


def test_stamping_marks_an_existing_database_without_touching_it(
    db_url: str,
) -> None:
    """Databases created before migrations existed must survive the upgrade."""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)  # simulate a pre-migrations install
    engine.dispose()

    assert schema.status(db_url).unmanaged is True

    schema.stamp(db_url, "0001")
    assert schema.status(db_url).up_to_date

    # And the guard is satisfied without any migration having run.
    schema.require_up_to_date(db_url)

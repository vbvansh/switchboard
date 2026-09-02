"""Alembic environment.

Two things here are not the Alembic defaults and both matter:

1. The database URL comes from Switchboard's settings, not alembic.ini, so
   there is one source of truth about which database is in use.

2. `render_as_batch=True`. SQLite cannot ALTER an existing column - it simply
   has no such statement. Batch mode makes Alembic emulate it by creating a new
   table, copying the rows across, and swapping it in. Without this, the first
   migration that changes a column fails on SQLite, which is the default
   database for most Switchboard installs.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from switchboard.config import settings
from switchboard.ledger.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# What the tables *should* look like. Autogenerate compares the live database
# against this to work out what changed.
target_metadata = Base.metadata


def _database_url() -> str:
    """Which database to migrate, most specific source first.

    The order matters. When Switchboard drives Alembic in-process it sets the
    URL on the config object; if that were not honoured here, a caller asking
    to migrate one database would silently migrate a different one. That is not
    a theoretical risk - it is exactly what happened the first time this ran.
    """
    x_args = context.get_x_argument(as_dictionary=True)
    if url := x_args.get("url"):
        return url

    if configured := config.get_main_option("sqlalchemy.url", None):
        return configured

    return settings.database_url


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it.

    Useful when a DBA has to review and apply changes by hand, which is common
    in companies that do not let applications alter their own schema.
    """
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect and apply migrations directly."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(
        section, prefix="sqlalchemy.", poolclass=pool.NullPool
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

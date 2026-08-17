"""Database connection and session management.

SQLite is the right choice here: one file, no server, no setup, and Switchboard
runs on a single machine. Going through SQLAlchemy rather than raw SQL keeps the
door open to Postgres later without rewriting every query.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from switchboard.ledger.models import Base

SQLITE_MEMORY_URL = "sqlite://"


def _sqlite_file_path(url: str) -> Path | None:
    """Filesystem path behind a sqlite URL, or None for in-memory."""
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return None
    raw = url[len(prefix) :]
    return Path(raw) if raw else None


class Database:
    def __init__(
        self,
        url: str,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
    ) -> None:
        self.url = url
        is_sqlite = url.startswith("sqlite")
        is_memory = is_sqlite and _sqlite_file_path(url) is None

        if (path := _sqlite_file_path(url)) is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

        kwargs: dict = {"echo": echo}
        if is_sqlite:
            # FastAPI serves requests across threads; SQLite objects are
            # otherwise pinned to their creating thread.
            kwargs["connect_args"] = {"check_same_thread": False}
        else:
            # Server-backed databases (Postgres) hand out pooled connections
            # that can be closed underneath us - by an idle timeout, a network
            # blip, or the server restarting. pool_pre_ping tests a connection
            # before handing it over and transparently replaces a dead one.
            # Without it, the first request after any such event fails.
            kwargs["pool_pre_ping"] = True
            kwargs["pool_size"] = pool_size
            kwargs["max_overflow"] = max_overflow
            # Recycle below the typical 5-minute idle cut-off used by managed
            # Postgres and connection proxies.
            kwargs["pool_recycle"] = 280

        if is_memory:
            # Without StaticPool every connection gets its own blank in-memory
            # database - which silently breaks tests.
            kwargs["poolclass"] = StaticPool

        self.engine: Engine = create_engine(url, **kwargs)

        if is_sqlite and not is_memory:
            _enable_wal(self.engine)

        self._session_factory = sessionmaker(
            bind=self.engine, expire_on_commit=False
        )

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Transactional scope: commit on success, roll back on failure."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()

    def is_reachable(self) -> bool:
        """Can we actually talk to the database right now?

        Used by the readiness probe. A configured URL proves nothing - the
        server may be down, the credentials wrong, or the network gone.
        """
        # Broad except is deliberate. This backs a readiness probe, and a probe
        # that raises turns a clean "not ready" 503 into a 500, which reads as
        # "the app is broken" rather than "a dependency is down". Any failure
        # to reach the database means the same thing: not ready.
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except (SQLAlchemyError, AttributeError, OSError):
            return False
        return True


def _enable_wal(engine: Engine) -> None:
    """Write-Ahead Logging: lets reads proceed while a write is in progress.

    Default SQLite locks the whole file during a write, so a slow ledger insert
    would block a concurrent budget check. WAL removes that at our scale.
    """

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, _connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

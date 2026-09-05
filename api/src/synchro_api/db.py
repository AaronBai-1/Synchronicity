"""Database setup for the API service (docs/plan.md "Platform").

Postgres holds all analytics in production (trajectories stay in object-storage Parquet);
SQLite is the dev/test default so the full API + ingest path runs offline with zero
infrastructure. Schema creation here is `create_all` only — Alembic migrations arrive
once the schema stops churning (before the Phase 1 launch); until then, dropping the dev
DB *is* the migration story.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

DEFAULT_DB_URL = "sqlite:///./synchro_dev.db"

_engine: Engine | None = None


class Base(DeclarativeBase):
    """Typed declarative base shared by all ORM models."""


def get_engine() -> Engine:
    """Process-wide engine from SYNCHRO_DB_URL (default: local SQLite).

    Cached so every request shares one connection pool. Tests point SYNCHRO_DB_URL at a
    tmp path and call reset_engine() to pick it up. SQLite needs check_same_thread=False
    because the FastAPI TestClient (and any threaded server) hands pooled connections
    across threads.
    """
    global _engine
    if _engine is None:
        url = os.environ.get("SYNCHRO_DB_URL", DEFAULT_DB_URL)
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args)
    return _engine


def reset_engine() -> None:
    """Dispose and forget the cached engine (tests re-point SYNCHRO_DB_URL between runs)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def init_db(engine: Engine | None = None) -> None:
    """Create all tables (dev/test convenience — Alembic replaces this in production)."""
    from synchro_api import models  # noqa: F401  # import registers the tables on Base

    Base.metadata.create_all(engine or get_engine())


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one Session per request. Routers commit explicitly on writes;
    read-only requests never commit, so incidental state is discarded with the session."""
    with Session(get_engine()) as session:
        yield session

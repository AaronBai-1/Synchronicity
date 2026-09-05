"""FastAPI app assembly (docs/plan.md "Platform").

Run locally with:  uvicorn synchro_api.main:app --reload

Startup calls init_db() (create_all) — fine for dev/test; production migrates with
Alembic before deploy and skips DDL at boot. Auth today is the X-Workspace-Id header
(see deps.py); Clerk replaces it before multi-user launch.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from synchro_api.db import init_db
from synchro_api.routers import matches, players, rallies


def create_app() -> FastAPI:
    """App factory — tests build a fresh app after re-pointing SYNCHRO_DB_URL."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        init_db()
        yield

    app = FastAPI(
        title="Synchronicity API",
        description="Badminton analytics: matches / games / rallies / shots",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(matches.router)
    app.include_router(rallies.router)
    app.include_router(players.router)

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()

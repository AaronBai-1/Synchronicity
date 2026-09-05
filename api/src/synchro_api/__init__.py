"""synchro_api — FastAPI service over the matches/games/rallies/shots data model.

Layout:
    db.py        engine/session setup (SYNCHRO_DB_URL; SQLite dev, Postgres prod)
    models.py    ORM tables mirroring docs/plan.md's "Platform" data model
    ingest.py    the pipeline→API contract: pydantic records → rows
    storage.py   upload-target abstraction (local file:// dev, R2 presigned later)
    routers/     the API surface: matches, rallies, players
"""

__version__ = "0.1.0"

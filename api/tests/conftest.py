"""API test fixtures: a TestClient over tmp SQLite + a tmp data dir.

Everything runs offline with zero infrastructure — the same property the dev setup has.
SYNCHRO_DB_URL / SYNCHRO_DATA_DIR are pointed at pytest's tmp_path before the app (and
its cached engine) is built, and the engine cache is reset around each test so tests
never share state.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("SYNCHRO_DB_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("SYNCHRO_DATA_DIR", str(tmp_path / "data"))

    from synchro_api import db
    from synchro_api.main import create_app

    db.reset_engine()
    with TestClient(create_app()) as test_client:  # context manager runs lifespan → init_db
        yield test_client
    db.reset_engine()

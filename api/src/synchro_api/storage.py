"""Video storage abstraction (docs/plan.md "Platform": Cloudflare R2, zero-egress).

The API never proxies video bytes — it hands the client an upload target and remembers
the canonical `video_uri` the pipeline will read (plan S0 ingests from that URI). In
production this is an R2/S3 presigned PUT with a short expiry; the LocalStorage dev
implementation returns a file:// target under a data directory so the whole flow works
offline. The R2 implementation arrives with real infra (Phase 0 "walking skeleton"
deploy) and only needs to satisfy `StorageBackend`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DATA_DIR = "./data"
_SAFE_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".ts", ".m4v", ".webm"}


@dataclass(frozen=True)
class UploadTarget:
    """Where the client should send the video, and where the pipeline will find it."""

    video_uri: str  # canonical location, stored on the match row
    upload_uri: str  # where the client sends bytes (== video_uri for local dev)
    method: str  # "file" for local dev, "PUT" for presigned uploads
    headers: dict[str, str] = field(default_factory=dict)  # required upload headers
    expires_in_s: int | None = None  # presign expiry; None for local


class StorageBackend(ABC):
    """Interface every storage implementation (local dev, R2/S3 presigned) satisfies."""

    @abstractmethod
    def create_upload_target(self, match_id: str, filename: str | None = None) -> UploadTarget:
        """Allocate an upload location for a match's source video."""


class LocalStorage(StorageBackend):
    """Dev/test backend: file:// targets under SYNCHRO_DATA_DIR (default ./data).

    The client (or a test) writes the file directly; there is no presigning to do.
    Directory layout mirrors the object-store key scheme (matches/{id}/source.<ext>) so
    swapping in R2 changes URIs, not structure.
    """

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root or os.environ.get("SYNCHRO_DATA_DIR", DEFAULT_DATA_DIR)).resolve()

    def create_upload_target(self, match_id: str, filename: str | None = None) -> UploadTarget:
        suffix = Path(filename).suffix.lower() if filename else ".mp4"
        if suffix not in _SAFE_SUFFIXES:
            suffix = ".mp4"  # never trust a client-supplied extension into a path
        target_dir = self.root / "matches" / match_id
        target_dir.mkdir(parents=True, exist_ok=True)
        uri = (target_dir / f"source{suffix}").as_uri()
        return UploadTarget(video_uri=uri, upload_uri=uri, method="file")


def get_storage() -> StorageBackend:
    """Storage backend from SYNCHRO_STORAGE (only "local" exists until R2 lands)."""
    backend = os.environ.get("SYNCHRO_STORAGE", "local")
    if backend == "local":
        return LocalStorage()
    raise RuntimeError(
        f"unknown storage backend {backend!r}: only 'local' is implemented; "
        "R2/S3 presigned uploads arrive with real infra"
    )

"""Offline pipeline stage framework — docs/plan.md "Pipeline architecture" (S0–S7).

Why this shape:
    * Stages run cheap→expensive (the plan's staged ordering). Each stage declares the
      artifact keys it consumes/produces, so the runner can order any subset of the DAG
      and modal_app.py can ship stage *groups* to differently-sized machines.
    * Caching is content-addressed: a stage's cache key = SHA256 over (stage name,
      stage version, sorted input artifact hashes) — the plan's "artifact caching keyed
      on input hash + model hash". `Stage.version` stands in for the model hash: bumping
      it invalidates that stage, and changed output hashes cascade the invalidation
      downstream. Reprocess-on-model-upgrade falls out for free.
    * Artifacts live on the local filesystem under workdir/artifacts/<stage>/ with a
      per-stage manifest.json, so a workdir on a shared volume (Modal Volume in
      production) resumes across processes and lets stage groups run in separate
      containers against the same store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

ArtifactKind = Literal["json", "file"]

SOURCE_VIDEO = "source_video"
"""Artifact key of the uploaded match video — registered as an external input at
context creation, and the sole input of S0 (docs/plan.md stage S0 Ingest)."""

_EXTERNAL_STAGE = "_external"
_MANIFEST_NAME = "manifest.json"


class PipelineError(RuntimeError):
    """Framework-level failure: unsatisfiable DAG, contract violation, missing artifact."""


# --- hashing -----------------------------------------------------------------------


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


# --- artifact store ----------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactEntry:
    """One stored artifact: where it lives, its content hash, and who produced it."""

    key: str
    kind: ArtifactKind
    path: Path
    sha256: str
    stage: str


class ArtifactStore:
    """Local-filesystem artifact store under ``root`` (= ``workdir/artifacts``).

    Layout: ``root/<stage>/`` holds that stage's output files plus a ``manifest.json``
    recording the cache key and the per-artifact content hashes. On construction the
    store scans existing manifests so partial pipelines (e.g. only S1 on a workdir
    where S0 already ran — or ran in another container) find their inputs; the cache
    key check in `run_pipeline` is what protects against consuming stale outputs of a
    stage that is actually part of the current run.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, ArtifactEntry] = {}
        self._pending: dict[str, dict[str, ArtifactEntry]] = {}
        self._load_existing_manifests()

    # -- layout ---------------------------------------------------------------

    def stage_dir(self, stage_name: str) -> Path:
        return self.root / stage_name

    def _load_existing_manifests(self) -> None:
        for manifest_path in sorted(self.root.glob(f"*/{_MANIFEST_NAME}")):
            try:
                data = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                logger.warning("ignoring unreadable manifest %s", manifest_path)
                continue
            entries = self._entries_from_manifest(data, manifest_path.parent)
            if entries is None:
                logger.warning("manifest %s references missing files; ignoring", manifest_path)
                continue
            for entry in entries:
                self._index[entry.key] = entry

    @staticmethod
    def _entries_from_manifest(data: dict, stage_dir: Path) -> list[ArtifactEntry] | None:
        """Materialise manifest artifact rows; None if any referenced file is gone."""
        entries: list[ArtifactEntry] = []
        for row in data.get("artifacts", []):
            path = stage_dir / row["filename"]
            if not path.exists():
                return None
            entries.append(
                ArtifactEntry(
                    key=row["key"],
                    kind=row["kind"],
                    path=path,
                    sha256=row["sha256"],
                    stage=data.get("stage", stage_dir.name),
                )
            )
        return entries

    # -- reads ----------------------------------------------------------------

    def keys(self) -> frozenset[str]:
        return frozenset(self._index)

    def has(self, key: str) -> bool:
        return key in self._index

    def _require(self, key: str) -> ArtifactEntry:
        try:
            return self._index[key]
        except KeyError:
            raise PipelineError(
                f"artifact '{key}' is not in the store — did its producing stage run?"
            ) from None

    def hash_of(self, key: str) -> str:
        return self._require(key).sha256

    def load_path(self, key: str) -> Path:
        """Path of a raw file artifact (also works for JSON artifacts)."""
        return self._require(key).path

    def load_model(self, key: str, model_type: type[M]) -> M:
        entry = self._require(key)
        return model_type.model_validate_json(entry.path.read_text())

    # -- writes (stage lifecycle, driven by run_pipeline) ----------------------

    def begin_stage(self, stage: Stage) -> None:
        """Wipe the stage dir (stale outputs from an older version must not mix in)."""
        stage_dir = self.stage_dir(stage.name)
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True)
        self._pending[stage.name] = {}

    def save_model(self, stage: Stage, key: str, model: BaseModel) -> Path:
        """Persist a pydantic model as pretty JSON under the stage's artifact dir."""
        payload = model.model_dump_json(indent=2).encode()
        path = self.stage_dir(stage.name) / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        self._record(stage, ArtifactEntry(key, "json", path, _sha256_bytes(payload), stage.name))
        return path

    def prepare_file(self, stage: Stage, filename: str) -> Path:
        """Where a stage should write a raw file artifact (e.g. an ffmpeg output).

        Write the file, then call `register_file` — hashing happens once, at
        registration, and the hash is reused from the manifest ever after.
        """
        stage_dir = self.stage_dir(stage.name)
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir / filename

    def register_file(self, stage: Stage, key: str, path: Path) -> Path:
        if not path.exists():
            raise PipelineError(f"stage '{stage.name}' registered missing file {path} as '{key}'")
        self._record(stage, ArtifactEntry(key, "file", path, _sha256_file(path), stage.name))
        return path

    def register_external(self, key: str, path: Path) -> None:
        """Register an input that exists outside the store (the uploaded source video)."""
        if not path.exists():
            raise PipelineError(f"external artifact '{key}' not found at {path}")
        self._index[key] = ArtifactEntry(key, "file", path, _sha256_file(path), _EXTERNAL_STAGE)

    def _record(self, stage: Stage, entry: ArtifactEntry) -> None:
        self._index[entry.key] = entry
        self._pending.setdefault(stage.name, {})[entry.key] = entry

    def commit_stage(self, stage: Stage, cache_key: str) -> None:
        """Validate outputs against the stage's declaration and write the manifest.

        The declared-vs-produced check keeps the DAG honest: a stage that silently
        drops (or sneaks in) an artifact would corrupt downstream cache keys.
        """
        pending = self._pending.pop(stage.name, {})
        declared, produced = set(stage.outputs), set(pending)
        if missing := declared - produced:
            raise PipelineError(
                f"stage '{stage.name}' did not produce declared outputs: {sorted(missing)}"
            )
        if undeclared := produced - declared:
            raise PipelineError(
                f"stage '{stage.name}' produced undeclared outputs: {sorted(undeclared)}"
            )
        manifest = {
            "stage": stage.name,
            "version": stage.version,
            "cache_key": cache_key,
            "artifacts": [
                {"key": e.key, "kind": e.kind, "filename": e.path.name, "sha256": e.sha256}
                for e in pending.values()
            ],
        }
        manifest_path = self.stage_dir(stage.name) / _MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def restore_cached_stage(self, stage: Stage, cache_key: str) -> bool:
        """If the stage dir holds a committed run for this exact cache key, adopt it."""
        manifest_path = self.stage_dir(stage.name) / _MANIFEST_NAME
        if not manifest_path.exists():
            return False
        try:
            data = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if data.get("cache_key") != cache_key:
            return False
        entries = self._entries_from_manifest(data, manifest_path.parent)
        if entries is None:
            return False
        for entry in entries:
            self._index[entry.key] = entry
        return True


# --- stage + context ---------------------------------------------------------------


class Stage(ABC):
    """One pipeline stage (a row of the plan's stage DAG table).

    Subclasses declare, as class attributes:
        name:    stable identifier (also the artifact subdirectory name)
        version: bump whenever the stage's logic, parameters, or model weights change —
                 this is the "model hash" component of the cache key
        inputs:  artifact keys consumed (must be produced upstream or registered external)
        outputs: artifact keys produced (enforced by ArtifactStore.commit_stage)

    `run` must materialise every declared output through ctx.artifacts (save_model /
    prepare_file + register_file) and nothing else. Anything torch-based must lazy-import
    inside `run` — torch is an optional extra, and the framework has to import clean.
    """

    name: ClassVar[str]
    version: ClassVar[str]
    inputs: ClassVar[tuple[str, ...]]
    outputs: ClassVar[tuple[str, ...]]

    @abstractmethod
    def run(self, ctx: PipelineContext) -> None:
        """Execute the stage. Called by run_pipeline only on cache miss."""


@dataclass
class PipelineContext:
    """Everything a stage needs: identity, scratch space, source video, artifact store."""

    match_id: str
    workdir: Path
    source_video: Path
    artifacts: ArtifactStore

    @classmethod
    def create(cls, match_id: str, workdir: Path, source_video: Path) -> PipelineContext:
        """Build a context, registering the source video as the DAG's external root input."""
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        store = ArtifactStore(workdir / "artifacts")
        store.register_external(SOURCE_VIDEO, Path(source_video))
        return cls(
            match_id=match_id,
            workdir=workdir,
            source_video=Path(source_video),
            artifacts=store,
        )


# --- runner ------------------------------------------------------------------------


class StageReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    version: str
    cache_key: str
    cached: bool
    duration_s: float


class PipelineReport(BaseModel):
    """Per-stage timing/caching summary of one run_pipeline call."""

    model_config = ConfigDict(extra="forbid")

    match_id: str
    stages: list[StageReport] = Field(default_factory=list)

    @property
    def total_duration_s(self) -> float:
        return sum(s.duration_s for s in self.stages)


def stage_cache_key(stage: Stage, store: ArtifactStore) -> str:
    """SHA256 over (stage identity+version, sorted input artifact hashes) — plan's
    "artifact caching keyed on input hash + model hash"."""
    digest = hashlib.sha256()
    digest.update(f"{stage.name}@{stage.version}".encode())
    for key in sorted(stage.inputs):
        digest.update(b"\x00" + key.encode() + b"=" + store.hash_of(key).encode())
    return digest.hexdigest()


def order_stages(stages: list[Stage], preavailable: frozenset[str]) -> list[Stage]:
    """Order stages by declared dependencies (stable Kahn passes, given order preserved
    among ready stages).

    Keys produced by any stage in this run are deliberately *not* treated as available
    up front, even if a previous run left them in the store — an included producer must
    always run (or cache-restore) before its consumers, so consumers never read a stale
    generation of an artifact that this run is about to replace.
    """
    producer_of: dict[str, str] = {}
    for stage in stages:
        for out in stage.outputs:
            if out in producer_of:
                raise PipelineError(
                    f"artifact '{out}' is declared by both '{producer_of[out]}' and '{stage.name}'"
                )
            producer_of[out] = stage.name

    available = set(preavailable) - set(producer_of)
    ordered: list[Stage] = []
    remaining = list(stages)
    while remaining:
        ready = [s for s in remaining if all(k in available for k in s.inputs)]
        if not ready:
            blocked = {
                s.name: sorted(set(s.inputs) - available) for s in remaining
            }
            raise PipelineError(f"cannot order stages — unsatisfied inputs: {blocked}")
        for stage in ready:
            ordered.append(stage)
            available.update(stage.outputs)
            remaining.remove(stage)
    return ordered


def run_pipeline(stages: list[Stage], ctx: PipelineContext) -> PipelineReport:
    """Execute stages in dependency order, honoring the content-addressed cache.

    Cache keys are computed just-in-time (after upstream stages have committed), so a
    version bump upstream cascades through changed output hashes and re-runs exactly
    the affected suffix of the DAG.
    """
    report = PipelineReport(match_id=ctx.match_id)
    for stage in order_stages(stages, ctx.artifacts.keys()):
        started = time.perf_counter()
        cache_key = stage_cache_key(stage, ctx.artifacts)
        if ctx.artifacts.restore_cached_stage(stage, cache_key):
            logger.info("stage %s@%s: cache hit (%.12s…) — skipping", stage.name, stage.version, cache_key)
            cached = True
        else:
            logger.info("stage %s@%s: running", stage.name, stage.version)
            ctx.artifacts.begin_stage(stage)
            stage.run(ctx)
            ctx.artifacts.commit_stage(stage, cache_key)
            cached = False
        report.stages.append(
            StageReport(
                stage=stage.name,
                version=stage.version,
                cache_key=cache_key,
                cached=cached,
                duration_s=time.perf_counter() - started,
            )
        )
    return report

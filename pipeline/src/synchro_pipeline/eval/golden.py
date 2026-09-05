"""Golden-set annotation schemas + loader (docs/plan.md Phase 0: hand-label 5 full
matches — rally boundaries, hit frames, court corners, scores — and wire them into CI).

One JSON file per match. These files are HAND-EDITED (and written by the labeling tools
in pipeline/tools/), which drives every design choice here:

    * stable, spelled-out field names — no abbreviations that rot
    * extra="forbid" so a typo'd key is a loud validation error, never silently ignored
    * normalization on load (rallies/hits/labels sorted) so hand-appended entries don't
      need manual ordering, while contradictions (overlapping rallies, duplicate court
      labels for a frame) are rejected with messages naming the offending frames
    * load_golden_match wraps every failure mode (missing file, broken JSON, schema
      violation) in GoldenLoadError with the path and a human-locatable message

Frame indices refer to the canonical mezzanine timeline (plan S0: the frame index is the
time authority). Court keypoints are image pixels in domain.court.COURT_KEYPOINT_NAMES
order; a per-point null means "not labelled / not visible in this frame".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from synchro_pipeline.domain.court import N_COURT_KEYPOINTS
from synchro_pipeline.schemas.records import Side


class GoldenLoadError(ValueError):
    """A golden-set file could not be loaded; the message says which file and why."""


class GoldenVideoMismatch(GoldenLoadError):
    """The video on disk is not the encode the labels were made against.

    Frame-exact labels are only meaningful against one specific encode — a different
    download or re-encode shifts frames and silently poisons every benchmark. Refusing
    to proceed is the plan's abstain-over-wrong rule applied to ground truth itself.
    """


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 of a file (hex digest) — the identity of a specific encode."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_video_hash(match: GoldenMatch, video_path: str | Path) -> str:
    """Check `video_path` against the match's recorded hash; returns the computed hash.

    Raises GoldenVideoMismatch on disagreement. A match with no recorded hash passes
    (callers decide whether to backfill it — the labeling tools do).
    """
    actual = sha256_file(video_path)
    if match.video_sha256 is not None and actual != match.video_sha256:
        raise GoldenVideoMismatch(
            f"{video_path} is not the encode the labels in match {match.match_id!r} were "
            f"made against (sha256 {actual[:12]}… != recorded {match.video_sha256[:12]}…). "
            "Frame-exact labels do not transfer between encodes; find the original file, "
            "or relabel against this one."
        )
    return actual


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoldenHit(_Model):
    """A hand-labelled racket-shuttle contact (plan S5 ground truth, frame-exact)."""

    frame: int = Field(ge=0)
    side: Side  # which physical end hit the shuttle: "near" | "far"


class GoldenShuttlePoint(_Model):
    """One frame of hand-labelled shuttle position (optional S4 ground truth).

    visible=false means a human confirmed the shuttle is not findable in the frame
    (occluded/out of frame) — that is a real label, distinct from "not labelled",
    which is simply the frame's absence from the list.
    """

    frame: int = Field(ge=0)
    x: float | None = None
    y: float | None = None
    visible: bool

    @model_validator(mode="after")
    def _xy_present_iff_visible(self) -> GoldenShuttlePoint:
        if self.visible and (self.x is None or self.y is None):
            raise ValueError(f"shuttle point at frame {self.frame}: visible=true requires x and y")
        return self


class GoldenRally(_Model):
    """One rally: inclusive frame span, its hits, and (optionally) a shuttle track.

    `shuttle` is None when the rally has no shuttle labels (the common case — plan only
    requires shuttle labels on a benchmark subset); an empty list would instead mean
    "labelled: the shuttle was never visible", so the two are deliberately distinct.
    """

    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    hits: list[GoldenHit] = Field(default_factory=list)
    shuttle: list[GoldenShuttlePoint] | None = None

    @model_validator(mode="after")
    def _normalize_and_check(self) -> GoldenRally:
        if self.end_frame < self.start_frame:
            raise ValueError(
                f"rally end_frame {self.end_frame} is before start_frame {self.start_frame}"
            )
        for hit in self.hits:
            if not (self.start_frame <= hit.frame <= self.end_frame):
                raise ValueError(
                    f"hit at frame {hit.frame} lies outside its rally "
                    f"[{self.start_frame}, {self.end_frame}]"
                )
        self.hits = sorted(self.hits, key=lambda h: h.frame)
        if self.shuttle is not None:
            self.shuttle = sorted(self.shuttle, key=lambda p: p.frame)
        return self


class GoldenCourtLabel(_Model):
    """Hand-labelled court keypoints for one frame (plan S2 ground truth, PCK@5px).

    keypoints has exactly 16 entries in domain.court.COURT_KEYPOINT_NAMES order; each is
    [x, y] in image pixels or null for an unlabelled/occluded point.
    """

    frame: int = Field(ge=0)
    keypoints: list[tuple[float, float] | None]

    @field_validator("keypoints")
    @classmethod
    def _exactly_sixteen(
        cls, v: list[tuple[float, float] | None]
    ) -> list[tuple[float, float] | None]:
        if len(v) != N_COURT_KEYPOINTS:
            raise ValueError(
                f"expected {N_COURT_KEYPOINTS} keypoints "
                f"(order = domain.court.COURT_KEYPOINT_NAMES), got {len(v)}"
            )
        return v


class GoldenScoreEntry(_Model):
    """The scoreboard value first shown at `frame` (plan S1c OCR ground truth)."""

    frame: int = Field(ge=0)
    a: int = Field(ge=0)
    b: int = Field(ge=0)


class GoldenMatch(_Model):
    """The golden annotation file for one match — the unit the benchmark CLI consumes."""

    match_id: str = Field(min_length=1)
    video_uri: str = Field(min_length=1)
    # Identity of the exact encode the frame labels index into. Stamped automatically by
    # the labeling tools (labeling.load_or_create_golden) and verified by the benchmark
    # CLI before scoring — see sha256_file / verify_video_hash.
    video_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_description: str | None = None  # original filename/source before renaming
    broadcaster: str | None = None  # plan risk #2: track per-broadcaster performance
    rallies: list[GoldenRally] = Field(default_factory=list)
    court_labels: list[GoldenCourtLabel] = Field(default_factory=list)
    score_timeline: list[GoldenScoreEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_and_check(self) -> GoldenMatch:
        self.rallies = sorted(self.rallies, key=lambda r: r.start_frame)
        for prev, cur in zip(self.rallies, self.rallies[1:], strict=False):
            if prev.end_frame >= cur.start_frame:
                raise ValueError(
                    f"rallies overlap: [{prev.start_frame}, {prev.end_frame}] and "
                    f"[{cur.start_frame}, {cur.end_frame}] (frame spans are inclusive)"
                )
        self.court_labels = sorted(self.court_labels, key=lambda c: c.frame)
        for prev_c, cur_c in zip(self.court_labels, self.court_labels[1:], strict=False):
            if prev_c.frame == cur_c.frame:
                raise ValueError(f"duplicate court_labels entry for frame {cur_c.frame}")
        self.score_timeline = sorted(self.score_timeline, key=lambda s: s.frame)
        for prev_s, cur_s in zip(self.score_timeline, self.score_timeline[1:], strict=False):
            if prev_s.frame == cur_s.frame:
                raise ValueError(f"duplicate score_timeline entry for frame {cur_s.frame}")
        return self


# --- load / save --------------------------------------------------------------------


def _format_validation_error(err: ValidationError) -> str:
    """Flatten pydantic's error list into `path: message` lines a labeller can act on."""
    lines = []
    for e in err.errors():
        loc = ".".join(str(part) for part in e["loc"]) or "<root>"
        lines.append(f"  {loc}: {e['msg']}")
    return "\n".join(lines)


def load_golden_match(path: str | Path) -> GoldenMatch:
    """Load one golden annotation file, raising GoldenLoadError with a precise reason.

    Forgiving where it is safe (ordering is normalized), loud where it matters (typos,
    contradictions) — hand-edited data must fail fast and legibly, never load wrong.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise GoldenLoadError(f"golden file not found: {p}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GoldenLoadError(
            f"{p} is not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}"
        ) from exc
    try:
        return GoldenMatch.model_validate(data)
    except ValidationError as exc:
        raise GoldenLoadError(
            f"{p} failed golden-schema validation:\n{_format_validation_error(exc)}"
        ) from exc


def save_golden_match(match: GoldenMatch, path: str | Path) -> None:
    """Write a golden file as indented JSON (stable field order → clean hand-edit diffs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(match.model_dump_json(indent=2) + "\n", encoding="utf-8")

"""Shuttle-tracker interface + a deterministic fake (plan stage S4, shuttle tracking).

WHY a protocol: S4 must be swappable — TrackNetV3 is the production pick, WASB-SBDT is the
planned eval cross-check, and a TensorRT fork may replace both for cost (docs/plan.md S4).
The benchmark harness (eval/benchmark_shuttle.py) and downstream stages depend only on this
contract, so swapping trackers is a one-line change that the golden-set CI immediately
re-scores ("did the numbers move").

WHY a fake: the benchmark/CI plumbing must run offline with no GPU, no weights and no real
footage (torch is an optional extra that is often not installed). FakeShuttleTracker
produces a deterministic, seedable synthetic trajectory with all three visibility states so
every consumer of the protocol — including the `detected|inpainted|missing` handling the
plan requires carrying downstream — is exercised in tests.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from synchro_pipeline.schemas.records import ShuttleObservation, Visibility


class ShuttleTrackPoint(BaseModel):
    """One per-frame shuttle observation, tagged with its frame index.

    Semantics match schemas.records.ShuttleObservation (image pixels at the 720p mezzanine,
    conf in [0, 1], visibility detected|inpainted|missing). frame_idx is added because
    trackers run over sparse frame subsets (rally frames only — plan's cost ordering), so
    positional alignment with a frame list is not enough.

    For visibility="missing" the (x, y) values are meaningless placeholders (0, 0) and
    conf is 0.0; consumers must gate on visibility, never on coordinates.
    """

    model_config = ConfigDict(extra="forbid")

    frame_idx: int = Field(ge=0)
    x: float
    y: float
    conf: float = Field(ge=0.0, le=1.0)
    visibility: Visibility

    def to_observation(self) -> ShuttleObservation:
        """Strip the frame index to produce the FrameRecord-embeddable observation."""
        return ShuttleObservation(x=self.x, y=self.y, conf=self.conf, visibility=self.visibility)


@runtime_checkable
class ShuttleTracker(Protocol):
    """Anything that can produce per-frame shuttle observations for a video.

    Implementations must be constructible without heavy dependencies; all expensive
    imports (torch, vendored code) belong inside track() so that merely selecting a
    tracker never fails offline.
    """

    name: str

    def track(
        self, video_path: str | Path, frames: Sequence[int] | None = None
    ) -> list[ShuttleTrackPoint]:
        """Track the shuttle across `frames` (all frames when None).

        Returns one ShuttleTrackPoint per requested frame (duplicates dropped), sorted by
        frame_idx ascending. Frames where the tracker found nothing still appear, with
        visibility="missing" — absence of evidence is data the confidence system needs.
        """
        ...


class FakeShuttleTracker:
    """Deterministic synthetic shuttle trajectory — benchmark/CI plumbing, not a model.

    The trajectory is a chain of parabolic "strokes" alternating near↔far across a
    synthetic broadcast frame, with per-frame visibility drawn as detected/inpainted/
    missing. Every value is a pure function of (seed, frame_idx): tracking frames
    [5, 6, 7] yields exactly the same points as tracking the whole video and slicing,
    which keeps benchmark subset logic honest.

    The video file is only probed for its frame count, and only when `frames` is None;
    an unreadable/missing file falls back to `default_n_frames` so the end-to-end
    benchmark path (`--tracker fake`) runs with no footage at all.
    """

    name = "fake"

    def __init__(
        self,
        seed: int = 0,
        image_size: tuple[int, int] = (1280, 720),
        default_n_frames: int = 300,
        stroke_period_frames: int = 30,
        miss_rate: float = 0.04,
        inpaint_rate: float = 0.08,
    ) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if stroke_period_frames < 2:
            raise ValueError("stroke_period_frames must be >= 2")
        if not (0.0 <= miss_rate and 0.0 <= inpaint_rate and miss_rate + inpaint_rate <= 1.0):
            raise ValueError("miss_rate/inpaint_rate must be >= 0 and sum to <= 1")
        self._seed = seed
        self._image_size = image_size
        self._default_n_frames = default_n_frames
        self._stroke_period = stroke_period_frames
        self._miss_rate = miss_rate
        self._inpaint_rate = inpaint_rate

    def track(
        self, video_path: str | Path, frames: Sequence[int] | None = None
    ) -> list[ShuttleTrackPoint]:
        """See ShuttleTracker.track. Deterministic for a given (seed, frames) pair."""
        if frames is None:
            idxs: list[int] = list(range(self._probe_n_frames(video_path)))
        else:
            idxs = sorted({int(f) for f in frames})
        return [self._point_at(f) for f in idxs]

    # -- internals -------------------------------------------------------------

    def _probe_n_frames(self, video_path: str | Path) -> int:
        """Frame count via cv2, falling back to default_n_frames on any failure.

        Lazy import keeps module import light and mirrors the real trackers' style.
        """
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                return self._default_n_frames
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        return n if n > 0 else self._default_n_frames

    def _stroke_endpoint_x(self, stroke: int) -> float:
        """x position (px) at the boundary between stroke-1 and stroke; per-stroke random."""
        w, _ = self._image_size
        rng = np.random.default_rng([self._seed, 1_000_003, stroke])
        return w * (0.15 + 0.70 * rng.random())

    def _point_at(self, frame_idx: int) -> ShuttleTrackPoint:
        w, h = self._image_size
        period = self._stroke_period
        stroke = frame_idx // period
        s = (frame_idx % period) / period

        # x: linear between per-stroke endpoints (continuous across stroke boundaries)
        x0 = self._stroke_endpoint_x(stroke)
        x1 = self._stroke_endpoint_x(stroke + 1)
        x = x0 + (x1 - x0) * s
        # y: parabolic arc alternating near (bottom) ↔ far (top) of the frame
        y_near, y_far = 0.85 * h, 0.30 * h
        y0, y1 = (y_near, y_far) if stroke % 2 == 0 else (y_far, y_near)
        y = y0 + (y1 - y0) * s - 0.18 * h * math.sin(math.pi * s)

        # per-frame stream so the value depends only on (seed, frame_idx); draw order fixed
        rng = np.random.default_rng([self._seed, 7, frame_idx])
        u = rng.random()
        if u < self._miss_rate:
            return ShuttleTrackPoint(
                frame_idx=frame_idx, x=0.0, y=0.0, conf=0.0, visibility="missing"
            )
        if u < self._miss_rate + self._inpaint_rate:
            noise = rng.normal(0.0, 3.0, 2)  # inpainted points are less precise
            conf = 0.35 + 0.30 * rng.random()
            visibility: Visibility = "inpainted"
        else:
            noise = rng.normal(0.0, 1.0, 2)
            conf = 0.75 + 0.24 * rng.random()
            visibility = "detected"
        px = float(np.clip(x + noise[0], 0.0, w - 1.0))
        py = float(np.clip(y + noise[1], 0.0, h - 1.0))
        return ShuttleTrackPoint(frame_idx=frame_idx, x=px, y=py, conf=conf, visibility=visibility)

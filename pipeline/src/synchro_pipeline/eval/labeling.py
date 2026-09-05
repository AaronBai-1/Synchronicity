"""Pure logic behind the golden-set labeling tools (pipeline/tools/label_*.py).

WHY this module exists: the labeling CLIs are OpenCV GUI loops, which cannot run in CI.
Everything with behaviour worth trusting — corner→16-keypoint projection, nudge
bookkeeping, golden-JSON merging, rally/hit session state — lives here, importable
without a display and covered by pipeline/tests/test_labeling_helpers.py. The tools
themselves are thin event loops over these functions (docs/plan.md Phase 0: the golden
set is CI, so the code that *writes* it must be tested).

State-mutating helpers return new lists / new GoldenMatch instances (re-validated via
the golden schema, so its overlap/ordering invariants are re-checked on every edit)
rather than mutating in place; the GUI layer keeps whatever it received until an edit
succeeds, which makes "show the ValueError, keep the session" the natural failure mode.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from synchro_pipeline.domain.court import (
    COURT_KEYPOINT_NAMES,
    COURT_KEYPOINTS,
    N_COURT_KEYPOINTS,
    fit_homography,
    keypoint_array,
    project_points,
)
from synchro_pipeline.eval.golden import (
    GoldenCourtLabel,
    GoldenHit,
    GoldenMatch,
    GoldenRally,
    GoldenVideoMismatch,
    load_golden_match,
    sha256_file,
    verify_video_hash,
)
from synchro_pipeline.schemas.records import Side

# --- court labeling -----------------------------------------------------------------

# The 4 doubles corners the labeller clicks, in this exact order (stated in the tool's
# on-screen instructions). Chosen clockwise starting bottom-left as seen in a broadcast
# frame: near-left, near-right, far-right, far-left.
CORNER_CLICK_ORDER: tuple[str, ...] = (
    "near_corner_left",
    "near_corner_right",
    "far_corner_right",
    "far_corner_left",
)

# Court line segments as (name, name) pairs — drawn by the tool so the labeller can
# verify the projected keypoints against the painted lines before saving.
COURT_LINES: tuple[tuple[str, str], ...] = (
    ("near_corner_left", "near_corner_right"),  # near baseline
    ("far_corner_left", "far_corner_right"),  # far baseline
    ("near_corner_left", "far_corner_left"),  # left doubles sideline
    ("near_corner_right", "far_corner_right"),  # right doubles sideline
    ("near_singles_left", "far_singles_left"),  # left singles sideline
    ("near_singles_right", "far_singles_right"),  # right singles sideline
    ("near_short_service_left", "near_short_service_right"),  # near short service line
    ("far_short_service_left", "far_short_service_right"),  # far short service line
    ("near_baseline_centre", "near_t_point"),  # near centre line
    ("far_baseline_centre", "far_t_point"),  # far centre line
)

COURT_LINE_INDICES: tuple[tuple[int, int], ...] = tuple(
    (COURT_KEYPOINT_NAMES.index(a), COURT_KEYPOINT_NAMES.index(b)) for a, b in COURT_LINES
)


def corners_to_keypoints(corners_px: Sequence[Sequence[float]]) -> np.ndarray | None:
    """Project all 16 canonical keypoints from 4 clicked doubles corners.

    `corners_px` are image pixels in CORNER_CLICK_ORDER. Fits the image→court
    homography via domain.court.fit_homography against the corners' canonical
    positions, then projects the full keypoint set back into the image so the labeller
    verifies 16 points while clicking only 4.

    Returns a (16, 2) float array in COURT_KEYPOINT_NAMES order, or None when the
    corners are degenerate (collinear / repeated clicks) and no homography exists.
    """
    corners = np.asarray(corners_px, dtype=np.float64)
    if corners.shape != (4, 2):
        raise ValueError(f"expected 4 (x, y) corners, got array of shape {corners.shape}")
    court_corners = np.array([COURT_KEYPOINTS[name] for name in CORNER_CLICK_ORDER])
    h_img2court, _ = fit_homography(corners, court_corners)
    if h_img2court is None:
        return None
    h_court2img = np.linalg.inv(h_img2court)
    return project_points(h_court2img, keypoint_array())


def nudge_point(points: np.ndarray, index: int, dx: float, dy: float) -> np.ndarray:
    """Return a copy of `points` with points[index] moved by (dx, dy).

    Pure (input untouched) so the GUI can keep an undo trail by holding old arrays.
    """
    pts = np.asarray(points, dtype=np.float64)
    if not (0 <= index < pts.shape[0]):
        raise ValueError(f"point index {index} out of range [0, {pts.shape[0]})")
    out = pts.copy()
    out[index, 0] += dx
    out[index, 1] += dy
    return out


def keypoints_to_label(
    points: np.ndarray, dropped: Iterable[int] = ()
) -> list[tuple[float, float] | None]:
    """Convert a (16, 2) array to the golden court_labels keypoint list.

    `dropped` indices become null entries (point not visible/labelable in this frame).
    Coordinates are rounded to 2 decimals — sub-centipixel precision is noise and makes
    hand-edited diffs unreadable.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (N_COURT_KEYPOINTS, 2):
        raise ValueError(f"expected shape ({N_COURT_KEYPOINTS}, 2), got {pts.shape}")
    dropped_set = set(dropped)
    if not dropped_set <= set(range(N_COURT_KEYPOINTS)):
        raise ValueError(f"dropped indices out of range: {sorted(dropped_set)}")
    return [
        None if i in dropped_set else (round(float(x), 2), round(float(y), 2))
        for i, (x, y) in enumerate(pts)
    ]


def merge_court_label(
    match: GoldenMatch, frame: int, keypoints: list[tuple[float, float] | None]
) -> GoldenMatch:
    """Insert-or-replace the court label for `frame`; returns a re-validated match."""
    kept = [c.model_dump() for c in match.court_labels if c.frame != frame]
    kept.append(GoldenCourtLabel(frame=frame, keypoints=keypoints).model_dump())
    return GoldenMatch.model_validate({**match.model_dump(), "court_labels": kept})


def load_or_create_golden(
    path: str,
    match_id: str | None = None,
    video_uri: str | None = None,
    video_path: str | None = None,
    allow_hash_mismatch: bool = False,
) -> GoldenMatch:
    """Load an existing golden file, or build a fresh skeleton for a new match.

    Creating requires match_id and video_uri so a fat-fingered --golden path fails
    loudly instead of silently starting an empty second file for the same match.

    When `video_path` is given (the file about to be labeled), the video hash is
    handled automatically: stamped on creation, backfilled on a legacy file without
    one, and VERIFIED on an existing file — labeling frames against the wrong encode
    corrupts ground truth, so a mismatch raises GoldenVideoMismatch unless
    `allow_hash_mismatch` explicitly overrides (which also re-stamps the new hash).
    """
    from pathlib import Path

    if Path(path).exists():
        match = load_golden_match(path)
        if video_path is not None:
            try:
                actual = verify_video_hash(match, video_path)
            except GoldenVideoMismatch:
                if not allow_hash_mismatch:
                    raise
                actual = sha256_file(video_path)
            match = match.model_copy(update={"video_sha256": actual})
        return match
    if not match_id or not video_uri:
        raise ValueError(
            f"golden file {path} does not exist; pass --match-id and --video-uri to create it"
        )
    return GoldenMatch(
        match_id=match_id,
        video_uri=video_uri,
        video_sha256=sha256_file(video_path) if video_path is not None else None,
    )


# --- rally / hit labeling -----------------------------------------------------------


def find_rally(rallies: Sequence[GoldenRally], frame: int) -> GoldenRally | None:
    """The rally whose inclusive [start_frame, end_frame] span contains `frame`, if any."""
    for rally in rallies:
        if rally.start_frame <= frame <= rally.end_frame:
            return rally
    return None


def insert_rally(rallies: Sequence[GoldenRally], new: GoldenRally) -> list[GoldenRally]:
    """Add a rally, rejecting any overlap (inclusive spans) with a frame-naming error."""
    for rally in rallies:
        if new.start_frame <= rally.end_frame and rally.start_frame <= new.end_frame:
            raise ValueError(
                f"new rally [{new.start_frame}, {new.end_frame}] overlaps existing rally "
                f"[{rally.start_frame}, {rally.end_frame}]"
            )
    return sorted([*rallies, new], key=lambda r: r.start_frame)


def remove_rally(rallies: Sequence[GoldenRally], frame: int) -> list[GoldenRally]:
    """Drop the rally containing `frame`; error if no rally contains it."""
    target = find_rally(rallies, frame)
    if target is None:
        raise ValueError(f"no rally contains frame {frame}")
    return [r for r in rallies if r is not target]


def add_hit(rallies: Sequence[GoldenRally], frame: int, side: Side) -> list[GoldenRally]:
    """Add (or replace, if the frame already has one) a hit inside its enclosing rally."""
    target = find_rally(rallies, frame)
    if target is None:
        raise ValueError(
            f"frame {frame} is not inside any labeled rally — mark the rally span first"
        )
    hits = [h.model_dump() for h in target.hits if h.frame != frame]
    hits.append(GoldenHit(frame=frame, side=side).model_dump())
    updated = GoldenRally.model_validate({**target.model_dump(), "hits": hits})
    return sorted(
        [r for r in rallies if r is not target] + [updated], key=lambda r: r.start_frame
    )


def remove_hit(rallies: Sequence[GoldenRally], frame: int) -> list[GoldenRally]:
    """Remove the hit labelled exactly at `frame`; error if no rally has one there."""
    for target in rallies:
        if any(h.frame == frame for h in target.hits):
            hits = [h.model_dump() for h in target.hits if h.frame != frame]
            updated = GoldenRally.model_validate({**target.model_dump(), "hits": hits})
            return sorted(
                [r for r in rallies if r is not target] + [updated],
                key=lambda r: r.start_frame,
            )
    raise ValueError(f"no hit labelled at frame {frame}")


def merge_rallies(match: GoldenMatch, rallies: Sequence[GoldenRally]) -> GoldenMatch:
    """Replace the match's rally list; re-validation re-checks overlap/ordering."""
    return GoldenMatch.model_validate(
        {**match.model_dump(), "rallies": [r.model_dump() for r in rallies]}
    )


@dataclass
class RallyLabelSession:
    """Keyboard-labeling state machine for rally spans and hit frames.

    Drives tools/label_rallies.py but is GUI-free: every operation either succeeds or
    raises ValueError with a message the tool surfaces verbatim, leaving the session
    unchanged (edits build new lists; state is only assigned on success — so a failed
    mark_end keeps the rally open with its pending hits intact for correction).
    """

    rallies: list[GoldenRally] = field(default_factory=list)
    pending_start: int | None = None
    pending_hits: list[GoldenHit] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.pending_start is not None

    def mark_start(self, frame: int) -> None:
        if self.pending_start is not None:
            raise ValueError(
                f"a rally is already open at frame {self.pending_start}; "
                "mark its end or cancel it first"
            )
        if find_rally(self.rallies, frame) is not None:
            raise ValueError(f"frame {frame} is inside an already-labeled rally")
        self.pending_start = frame

    def mark_end(self, frame: int) -> None:
        if self.pending_start is None:
            raise ValueError("no open rally — mark a start first")
        if frame <= self.pending_start:
            raise ValueError(
                f"rally end (frame {frame}) must be after its start (frame {self.pending_start})"
            )
        new = GoldenRally(
            start_frame=self.pending_start, end_frame=frame, hits=list(self.pending_hits)
        )
        self.rallies = insert_rally(self.rallies, new)  # may raise; pending stays open
        self.pending_start = None
        self.pending_hits = []

    def mark_hit(self, frame: int, side: Side) -> None:
        """Label a hit: into the open rally if one is open, else into an existing rally."""
        if self.pending_start is not None:
            if frame < self.pending_start:
                raise ValueError(
                    f"hit at frame {frame} is before the open rally's start "
                    f"({self.pending_start})"
                )
            hits = [h for h in self.pending_hits if h.frame != frame]
            hits.append(GoldenHit(frame=frame, side=side))
            self.pending_hits = sorted(hits, key=lambda h: h.frame)
        else:
            self.rallies = add_hit(self.rallies, frame, side)

    def undo_pending_hit(self) -> GoldenHit:
        """Drop and return the latest (highest-frame) pending hit.

        Highest-frame, not insertion order: pending hits are kept frame-sorted, and when
        correcting mid-rally the labeller is almost always undoing the furthest mark.
        """
        if not self.pending_hits:
            raise ValueError("no pending hits to undo")
        last = max(self.pending_hits, key=lambda h: h.frame)
        self.pending_hits = [h for h in self.pending_hits if h is not last]
        return last

    def cancel(self) -> None:
        """Abandon the open rally and its pending hits."""
        self.pending_start = None
        self.pending_hits = []

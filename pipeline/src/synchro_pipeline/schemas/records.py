"""Inter-stage data contracts (Pydantic v2), mirroring docs/plan.md.

Granularities: FrameRecord (per rally frame) → ShotRecord (per hit — the product's core
contract, a ShuttleSet-compatible superset) → RallyRecord → MatchRecord.

Conventions:
    * image coordinates in pixels at the 720p mezzanine resolution
    * court coordinates in metres in the frame defined by domain.court
    * times derive from frame_idx / fps_effective — never from container timestamps
    * a missing/low-confidence value is None + a qa_flag, never a silent guess
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from synchro_pipeline.domain.taxonomy import CoarseShotType, MergedShotType, ShotType

Side = Literal["near", "far"]
PlayerRef = Literal["A", "B"]
Visibility = Literal["detected", "inpainted", "missing"]
LandingHeight = Literal["net", "mid", "high"]
ServeType = Literal["short", "long"]
Discipline = Literal["MS", "WS", "MD", "WD", "XD"]
SpeedConfTier = Literal["A", "B", "C"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class XY(_Model):
    x: float
    y: float


class Keypoint2D(_Model):
    x: float
    y: float
    conf: float = Field(ge=0.0, le=1.0)


class ScorePair(_Model):
    a: int = Field(ge=0)
    b: int = Field(ge=0)


# --- per-frame ---------------------------------------------------------------------


class CourtObservation(_Model):
    keypoints: list[Keypoint2D]  # image px, order = domain.court.COURT_KEYPOINT_NAMES
    homography: list[list[float]] | None = None  # 3x3, image px -> court metres
    reprojection_error_px: float | None = None
    valid: bool


class ShuttleObservation(_Model):
    x: float
    y: float
    conf: float = Field(ge=0.0, le=1.0)
    visibility: Visibility


class PlayerObservation(_Model):
    track_id: int
    side: Side
    bbox_xyxy: tuple[float, float, float, float]
    foot_court_xy: XY | None = None  # metres; None when homography invalid
    pose_keypoints: list[Keypoint2D] = Field(default_factory=list)  # COCO-17 order
    pose_conf_mean: float | None = None


class FrameRecord(_Model):
    frame_idx: int
    t_ms: float
    rally_id: str | None = None
    court: CourtObservation | None = None
    shuttle: ShuttleObservation | None = None
    players: list[PlayerObservation] = Field(default_factory=list)


# --- per-hit / per-stroke: the product's core contract -----------------------------


class ShotRecord(_Model):
    """One stroke. ShuttleSet-compatible superset (mirrors CoachAI Track-1's field spec).

    Fields the v1 UI ignores (backhand, aroundhead, landing_height, lose_reason) are
    carried anyway — they are cheap and unlock ShuttleSet-trained tactical models later.
    """

    match_id: str
    set_no: int = Field(ge=1)  # game number; "set" follows ShuttleSet naming
    rally_id: str
    ball_round: int = Field(ge=1)  # stroke sequence within the rally, 1 = serve
    frame_num: int
    t_ms: float
    player: PlayerRef | None = None
    side: Side

    type_canonical: ShotType | None = None
    type_merged: MergedShotType | None = None
    type_coarse: CoarseShotType | None = None
    type_probs: dict[MergedShotType, float] | None = None  # full distribution, not argmax
    type_conf: float | None = Field(default=None, ge=0.0, le=1.0)

    backhand: bool | None = None
    aroundhead: bool | None = None
    landing_height: LandingHeight | None = None

    hit_xy_court: XY | None = None
    landing_xy_court: XY | None = None
    landing_area: int | None = Field(default=None, ge=1, le=16)
    player_location: XY | None = None
    player_location_area: int | None = Field(default=None, ge=1, le=16)
    opponent_location: XY | None = None
    opponent_location_area: int | None = Field(default=None, ge=1, le=16)

    shot_speed_2d_kmh: float | None = None
    shot_speed_3d_kmh: float | None = None
    speed_conf_tier: SpeedConfTier | None = None

    is_winner: bool | None = None
    is_error: bool | None = None
    lose_reason: str | None = None

    hit_det_conf: float | None = Field(default=None, ge=0.0, le=1.0)
    traj_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    qa_flags: list[str] = Field(default_factory=list)


# --- per-rally ---------------------------------------------------------------------


class RallyConfidence(_Model):
    segmentation: float | None = Field(default=None, ge=0.0, le=1.0)
    ocr: float | None = Field(default=None, ge=0.0, le=1.0)
    hits: float | None = Field(default=None, ge=0.0, le=1.0)
    tracking: float | None = Field(default=None, ge=0.0, le=1.0)


class RallyRecord(_Model):
    rally_id: str
    match_id: str
    set_no: int = Field(ge=1)
    seq: int = Field(ge=1)  # rally number within the game
    start_frame: int
    end_frame: int
    duration_s: float
    video_shot_ids: list[int] = Field(default_factory=list)  # S1a shot segments spanned

    score_before: ScorePair | None = None
    score_after: ScorePair | None = None
    server: PlayerRef | None = None
    serve_type: ServeType | None = None
    winner: PlayerRef | None = None
    lose_reason: str | None = None

    n_strokes: int | None = None
    stroke_ids: list[str] = Field(default_factory=list)
    score_verified: bool = False  # passed the scoring-state-machine cross-check

    confidence: RallyConfidence = Field(default_factory=RallyConfidence)
    qa_flags: list[str] = Field(default_factory=list)
    clip_uri: str | None = None


# --- per-match ---------------------------------------------------------------------


class SourceMeta(_Model):
    width: int
    height: int
    fps_effective: float
    duration_s: float
    broadcaster_guess: str | None = None


class PlayerInfo(_Model):
    name: str | None = None
    handedness: Literal["left", "right"] | None = None


class GameSummary(_Model):
    set_no: int = Field(ge=1)
    final_score: ScorePair
    rally_ids: list[str] = Field(default_factory=list)


class QualityReport(_Model):
    pct_rallies_clean: float | None = Field(default=None, ge=0.0, le=100.0)
    pct_strokes_classified: float | None = Field(default=None, ge=0.0, le=100.0)
    flags_summary: dict[str, int] = Field(default_factory=dict)


class MatchRecord(_Model):
    match_id: str
    source_meta: SourceMeta
    discipline: Discipline = "MS"
    players: dict[PlayerRef, PlayerInfo] = Field(default_factory=dict)
    first_server: PlayerRef | None = None
    a_on_near_side_at_start: bool | None = None
    sets: list[GameSummary] = Field(default_factory=list)

    pipeline_version: str
    stage_model_hashes: dict[str, str] = Field(default_factory=dict)
    quality_report: QualityReport = Field(default_factory=QualityReport)

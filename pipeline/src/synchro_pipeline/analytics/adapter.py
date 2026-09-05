# Portions adapted from BadmintonAnalyzer (Apache-2.0), Copyright Dhyey Mavani.
"""Bridge from pipeline contracts (ShotRecord/RallyRecord) to analytics DataFrames.

Frame layout follows BadmintonAnalyzer's column naming where it eases the port
(match_id / rally_id / shot_number / origin_zone / target_zone / outcome / ...), but
every vocabulary is ours (docs/plan.md "S7 Metrics" + trust architecture):

    * ``shot_type`` — our 8-class CoarseShotType coach vocabulary, NOT their 11-type enum.
    * zones — a 6-zone vocabulary (front/mid/rear × left/right) derived from
      domain.court coordinates; see :func:`zone6_from_xy`.
    * QA gating is explicit: shots dropped by ``min_type_conf`` / ``exclude_qa_flags``
      are counted per reason in the companion :class:`AdapterReport`. Callers always
      see what was filtered — nothing is silently excluded.
    * unknown stays unknown: a missing type/zone/score becomes None (NaN in the frame),
      never a guess. Downstream aggregations bucket these as ``"unknown"``.

Zone convention (:func:`zone6_from_xy`):
    Depth is measured from the net on the half where the point lies:

        front:  |y| <  1.98   (up to the short service line, ``Y_SHORT_SERVICE``)
        mid:    1.98 <= |y| < 4.34
        rear:   4.34 <= |y| <= 6.70

    The rear boundary is the midpoint between the short service line and the baseline
    ((1.98 + 6.70) / 2 = 4.34): landmark-anchored "thirds" rather than exact 2.23 m
    slices, so "front" matches the painted service box a coach reasons about.

    Left/right is from the perspective of the player occupying ``side`` (facing the
    net): for the near player left means x < 0, for the far player left means x > 0;
    x == 0 counts as right. Player-perspective labels are side-invariant, so
    aggregating a player across the mid-match end change never mixes mirrored zones.

Outcome convention (:func:`outcome_from_flags`): ``winner`` / ``error_forced`` /
``error_unforced`` / ``error`` / ``in_play``. The forced/unforced split is best-effort
string matching on ``lose_reason``; when the reason does not distinguish, the honest
undifferentiated label ``"error"`` is emitted (never a guessed subclass).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from synchro_pipeline.domain.court import X_DOUBLES, Y_BASELINE, Y_SHORT_SERVICE
from synchro_pipeline.schemas.records import RallyRecord, ShotRecord

# Trust knob shared by every analytics module: aggregate rows with n below this are
# flagged low_sample (docs/plan.md — low-sample rows are flagged, never hidden).
DEFAULT_MIN_N = 15

UNKNOWN = "unknown"

TERMINAL_OUTCOMES: frozenset[str] = frozenset(
    {"winner", "error", "error_forced", "error_unforced"}
)
ERROR_OUTCOMES: frozenset[str] = frozenset({"error", "error_forced", "error_unforced"})

# --- 6-zone vocabulary --------------------------------------------------------------

ZONE_DEPTH_FRONT_MAX_M = Y_SHORT_SERVICE  # 1.98
ZONE_DEPTH_MID_MAX_M = (Y_SHORT_SERVICE + Y_BASELINE) / 2  # 4.34

ZONE6_LABELS: tuple[str, ...] = (
    "front_left",
    "front_right",
    "mid_left",
    "mid_right",
    "rear_left",
    "rear_right",
)

# Best-effort band per 4x4-grid row (rows run net -> baseline, each 1.675 m deep),
# assigned by the row-centre depth: 0.84 -> front, 2.51 -> mid, 4.19 -> mid, 5.86 -> rear.
_AREA_ROW_TO_BAND = ("front", "mid", "mid", "rear")


def zone6_from_xy(x: float, y: float, side: str) -> str | None:
    """Map a court-plane point (metres) to a 6-zone label, or None when out of bounds.

    ``side`` names the half whose occupant's perspective defines left/right ("near" or
    "far"); depth comes from |y| so the same call works for either half. See the module
    docstring for the exact band boundaries.
    """
    if abs(x) > X_DOUBLES or abs(y) > Y_BASELINE:
        return None
    depth = abs(y)
    if depth < ZONE_DEPTH_FRONT_MAX_M:
        band = "front"
    elif depth < ZONE_DEPTH_MID_MAX_M:
        band = "mid"
    else:
        band = "rear"
    x_player = x if side == "near" else -x
    return f"{band}_{'left' if x_player < 0 else 'right'}"


def zone6_from_area(area: int | None, side: str) -> str | None:
    """Best-effort 6-zone label from a 1-16 area id (fallback when court xy is missing).

    Mirrors domain.court.area_from_xy's grid: area = row * 4 + col + 1, columns running
    camera-left to camera-right. Depth band is assigned by row-centre depth
    (``_AREA_ROW_TO_BAND``); columns 0-1 are camera-left (x < 0) and are mirrored into
    the player perspective of ``side``. Returns None for a missing or invalid area.
    """
    if area is None or not 1 <= area <= 16:
        return None
    row, col = divmod(int(area) - 1, 4)
    band = _AREA_ROW_TO_BAND[row]
    camera_left = col <= 1
    left = camera_left if side == "near" else not camera_left
    return f"{band}_{'left' if left else 'right'}"


# --- outcome derivation -------------------------------------------------------------


def outcome_from_flags(
    is_winner: bool | None, is_error: bool | None, lose_reason: str | None
) -> str:
    """Per-shot outcome from ShotRecord flags.

    The forced/unforced split is best-effort substring matching on ``lose_reason``
    ("unforced" is checked before "forced" — it contains it); an error whose reason
    does not distinguish stays the undifferentiated ``"error"``.
    """
    if is_winner:
        return "winner"
    if is_error:
        reason = (lose_reason or "").lower()
        if "unforced" in reason:
            return "error_unforced"
        if "forced" in reason:
            return "error_forced"
        return "error"
    return "in_play"


def coalesce_unknown(value: object) -> object:
    """Missing (None/NaN) group keys become the honest 'unknown' bucket, never dropped."""
    return UNKNOWN if pd.isna(value) else value


# --- adapter report -----------------------------------------------------------------


@dataclass(frozen=True)
class AdapterReport:
    """What the adapter filtered — so callers can always see what was dropped.

    ``n_dropped_by_reason`` keys: ``"low_type_conf"`` (below ``min_type_conf``, or
    confidence missing while a gate is active) and ``"qa_flag:<flag>"``. Each dropped
    record is attributed to exactly one reason (confidence gate first, then the
    record's qa_flags in order), so the counts sum to ``n_shots_in - n_kept``.
    """

    n_shots_in: int
    n_kept: int
    n_dropped_by_reason: dict[str, int] = field(default_factory=dict)


# --- frame builders -----------------------------------------------------------------

SHOT_FRAME_COLUMNS: tuple[str, ...] = (
    "match_id",
    "game_number",
    "rally_id",
    "rally_seq",
    "shot_number",
    "t_ms",
    "player",
    "side",
    "shot_type",
    "origin_zone",
    "target_zone",
    "outcome",
    "score_before",
    "score_a",
    "score_b",
    "server",
    "serve_type",
    "winner",
    "rally_length",
    "duration",
)

RALLY_FRAME_COLUMNS: tuple[str, ...] = (
    "match_id",
    "game_number",
    "rally_id",
    "seq",
    "score_before",
    "score_a",
    "score_b",
    "server",
    "serve_type",
    "winner",
    "rally_length",
    "duration",
    "score_verified",
)


def _shot_drop_reason(
    shot: ShotRecord, min_type_conf: float | None, exclude_flags: frozenset[str]
) -> str | None:
    if min_type_conf is not None and (shot.type_conf is None or shot.type_conf < min_type_conf):
        # Unknown confidence under an active confidence gate cannot be certified -> drop
        # (counted, never silent).
        return "low_type_conf"
    for flag in shot.qa_flags:
        if flag in exclude_flags:
            return f"qa_flag:{flag}"
    return None


def _origin_zone(shot: ShotRecord) -> str | None:
    """Hitter-position zone: hit_xy_court -> player_location -> player_location_area."""
    if shot.hit_xy_court is not None:
        return zone6_from_xy(shot.hit_xy_court.x, shot.hit_xy_court.y, shot.side)
    if shot.player_location is not None:
        return zone6_from_xy(shot.player_location.x, shot.player_location.y, shot.side)
    return zone6_from_area(shot.player_location_area, shot.side)


def _target_zone(shot: ShotRecord) -> str | None:
    """Landing zone: landing_xy_court -> landing_area.

    With coordinates, the half is read off the landing's y sign and the zone is
    described from the perspective of that half's occupant (a net error landing on the
    hitter's own half is labelled in the hitter's frame). The area fallback cannot see
    the half, so it assumes the half opposite the hitter (documented best-effort).
    """
    if shot.landing_xy_court is not None:
        landing_side = "near" if shot.landing_xy_court.y < 0 else "far"
        return zone6_from_xy(shot.landing_xy_court.x, shot.landing_xy_court.y, landing_side)
    opposite = "far" if shot.side == "near" else "near"
    return zone6_from_area(shot.landing_area, opposite)


def build_shot_frame(
    shots: list[ShotRecord],
    rallies: list[RallyRecord],
    *,
    min_type_conf: float | None = None,
    exclude_qa_flags: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, AdapterReport]:
    """One row per kept stroke, rally context joined in; plus a filtering report.

    ``min_type_conf=None`` (default) keeps every shot regardless of classifier
    confidence. Shots whose qa_flags intersect ``exclude_qa_flags`` are dropped and
    counted in the report. Rows are sorted by (match_id, game_number, t_ms,
    shot_number); shots whose rally is not in ``rallies`` keep their own fields and get
    None for the joined rally context.
    """
    exclude = frozenset(exclude_qa_flags or ())
    rally_by_id = {(r.match_id, r.rally_id): r for r in rallies}
    dropped: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    for shot in shots:
        reason = _shot_drop_reason(shot, min_type_conf, exclude)
        if reason is not None:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        rally = rally_by_id.get((shot.match_id, shot.rally_id))
        score = rally.score_before if rally is not None else None
        rows.append(
            {
                "match_id": shot.match_id,
                "game_number": shot.set_no,
                "rally_id": shot.rally_id,
                "rally_seq": rally.seq if rally is not None else None,
                "shot_number": shot.ball_round,
                "t_ms": shot.t_ms,
                "player": shot.player,
                "side": shot.side,
                "shot_type": shot.type_coarse.value if shot.type_coarse is not None else None,
                "origin_zone": _origin_zone(shot),
                "target_zone": _target_zone(shot),
                "outcome": outcome_from_flags(shot.is_winner, shot.is_error, shot.lose_reason),
                "score_before": f"{score.a}-{score.b}" if score is not None else None,
                "score_a": score.a if score is not None else None,
                "score_b": score.b if score is not None else None,
                "server": rally.server if rally is not None else None,
                "serve_type": rally.serve_type if rally is not None else None,
                "winner": rally.winner if rally is not None else None,
                "rally_length": rally.n_strokes if rally is not None else None,
                "duration": rally.duration_s if rally is not None else None,
            }
        )
    report = AdapterReport(
        n_shots_in=len(shots), n_kept=len(rows), n_dropped_by_reason=dropped
    )
    if not rows:
        return pd.DataFrame(columns=list(SHOT_FRAME_COLUMNS)), report
    frame = (
        pd.DataFrame(rows, columns=list(SHOT_FRAME_COLUMNS))
        .sort_values(["match_id", "game_number", "t_ms", "shot_number"])
        .reset_index(drop=True)
    )
    return frame, report


def build_rally_frame(
    rallies: list[RallyRecord],
    *,
    exclude_qa_flags: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, AdapterReport]:
    """One row per kept rally plus a filtering report (``n_shots_in`` = rallies offered).

    Rallies whose qa_flags intersect ``exclude_qa_flags`` are dropped and counted under
    ``"qa_flag:<flag>"`` (first matching flag in record order). Rows are sorted by
    (match_id, game_number, seq).
    """
    exclude = frozenset(exclude_qa_flags or ())
    dropped: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    for rally in rallies:
        reason = next(
            (f"qa_flag:{flag}" for flag in rally.qa_flags if flag in exclude), None
        )
        if reason is not None:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        score = rally.score_before
        rows.append(
            {
                "match_id": rally.match_id,
                "game_number": rally.set_no,
                "rally_id": rally.rally_id,
                "seq": rally.seq,
                "score_before": f"{score.a}-{score.b}" if score is not None else None,
                "score_a": score.a if score is not None else None,
                "score_b": score.b if score is not None else None,
                "server": rally.server,
                "serve_type": rally.serve_type,
                "winner": rally.winner,
                "rally_length": rally.n_strokes,
                "duration": rally.duration_s,
                "score_verified": rally.score_verified,
            }
        )
    report = AdapterReport(
        n_shots_in=len(rallies), n_kept=len(rows), n_dropped_by_reason=dropped
    )
    if not rows:
        return pd.DataFrame(columns=list(RALLY_FRAME_COLUMNS)), report
    frame = (
        pd.DataFrame(rows, columns=list(RALLY_FRAME_COLUMNS))
        .sort_values(["match_id", "game_number", "seq"])
        .reset_index(drop=True)
    )
    return frame, report

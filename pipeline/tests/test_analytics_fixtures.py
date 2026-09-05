"""Shared synthetic-match fixture for the analytics test suite (no tests in here).

A hand-designed 2-game, 10-rally, 42-shot singles match built from OUR contracts
(schemas.records.ShotRecord / RallyRecord). Every expected number asserted across the
test_analytics_* files was computed by hand from the rally table below.

Layout (players A/B; A is near in game 1 and far in game 2):

    set seq score  server serve  sequence (last marker decides the rally)
    1   1   0-0    A short  A serve, B lift, A smash WINNER            -> A
    1   2   1-0    A short  A serve, B lift, A smash WINNER            -> A
    1   3   2-0    A short  A serve, B net, A lift, B smash WINNER     -> B
    1   4   2-1    B long   B serve, A clear, B drop, A net, B lift, A smash WINNER -> A
    1   5   3-1    A short  A serve, B lift, A smash, B drive, A net ERROR(unforced) -> B
    1   6   3-2    B short  B serve, A lift, B clear, A drop, B net, A lift, B smash WINNER -> B
    2   1   19-19  B short  B serve, A lift, B smash WINNER            -> B
    2   2   19-20  B short  B serve, A net, B lift, A smash WINNER     -> A
    2   3   20-20  A short  A serve, B lift, A smash WINNER            -> A
    2   4   21-20  A long   A serve, B lift, A smash, B drive ERROR    -> A

Hand-computed reference values (used by the exact-assertion tests):
    * A shots = 22 (serve 6, smash 7, lift 4, net 3, clear 1, drop 1); B shots = 20.
    * A transitions (A shot -> B response): serve->lift 5, serve->net 1, lift->smash 3,
      lift->clear 1, clear->drop 1, net->lift 2, smash->drive 2, drop->net 1 (16 total).
    * A 2-gram patterns over A's own shots: serve->smash n=5 (4 won), net->smash n=2
      (2 won); 12 windows total, 7 in won rallies (baseline 7/12).
    * Momentum (game boundaries respected): game 1 winners A,A,B,A,B,B; game 2 winners
      B,A,A,A -> longest A streak 3, longest A skid 2 (naive cross-game skid would be 3).
    * A serves by (parity of own score, type): even/short n=3 won 2; odd/short n=2
      won 1; odd/long n=1 won 1.
"""

from __future__ import annotations

from synchro_pipeline.domain.court import area_from_xy
from synchro_pipeline.domain.taxonomy import CoarseShotType
from synchro_pipeline.schemas.records import XY, RallyRecord, ScorePair, ShotRecord

MATCH_ID = "m1"

# Default positions for a NEAR-side hitter (metres, our court frame); a far-side hitter
# mirrors both coordinates. Hit position is on the hitter's half (y < 0 for near).
_HIT_XY: dict[str, tuple[float, float]] = {
    "serve": (0.5, -2.2),
    "clear": (0.5, -4.6),
    "smash": (0.5, -4.6),
    "lift": (0.5, -4.6),
    "drop": (0.5, -4.6),
    "net": (-0.6, -1.5),
    "drive": (-0.6, -1.5),
    "push_rush": (-0.6, -1.5),
}
# Landing position on the opponent half (y > 0 for a near hitter); serves depend on the
# rally's serve_type. Error shots get landing=None (unknown, never guessed).
_LANDING_XY: dict[str, tuple[float, float]] = {
    "serve_short": (-0.4, 1.6),
    "serve_long": (0.3, 6.0),
    "smash": (1.0, 4.9),
    "clear": (0.6, 6.1),
    "drop": (-0.5, 1.2),
    "net": (-0.5, 1.2),
    "lift": (0.8, 5.5),
    "drive": (0.9, 3.0),
    "push_rush": (0.7, 2.5),
}

_ERROR_MARKERS = ("error", "error_unforced")

# (set_no, seq, (score_a, score_b), server, serve_type, [(player, coarse_type, marker)])
_RALLIES = [
    (1, 1, (0, 0), "A", "short",
     [("A", "serve", None), ("B", "lift", None), ("A", "smash", "winner")]),
    (1, 2, (1, 0), "A", "short",
     [("A", "serve", None), ("B", "lift", None), ("A", "smash", "winner")]),
    (1, 3, (2, 0), "A", "short",
     [("A", "serve", None), ("B", "net", None), ("A", "lift", None),
      ("B", "smash", "winner")]),
    (1, 4, (2, 1), "B", "long",
     [("B", "serve", None), ("A", "clear", None), ("B", "drop", None),
      ("A", "net", None), ("B", "lift", None), ("A", "smash", "winner")]),
    (1, 5, (3, 1), "A", "short",
     [("A", "serve", None), ("B", "lift", None), ("A", "smash", None),
      ("B", "drive", None), ("A", "net", "error_unforced")]),
    (1, 6, (3, 2), "B", "short",
     [("B", "serve", None), ("A", "lift", None), ("B", "clear", None),
      ("A", "drop", None), ("B", "net", None), ("A", "lift", None),
      ("B", "smash", "winner")]),
    (2, 1, (19, 19), "B", "short",
     [("B", "serve", None), ("A", "lift", None), ("B", "smash", "winner")]),
    (2, 2, (19, 20), "B", "short",
     [("B", "serve", None), ("A", "net", None), ("B", "lift", None),
      ("A", "smash", "winner")]),
    (2, 3, (20, 20), "A", "short",
     [("A", "serve", None), ("B", "lift", None), ("A", "smash", "winner")]),
    (2, 4, (21, 20), "A", "long",
     [("A", "serve", None), ("B", "lift", None), ("A", "smash", None),
      ("B", "drive", "error")]),
]


def _side_of(player: str, set_no: int) -> str:
    """A is near in game 1 and far in game 2 (ends swap between games)."""
    a_near = set_no == 1
    on_near = a_near if player == "A" else not a_near
    return "near" if on_near else "far"


def _mirror(xy: tuple[float, float], side: str) -> tuple[float, float]:
    return xy if side == "near" else (-xy[0], -xy[1])


def build_synthetic_match() -> tuple[list[ShotRecord], list[RallyRecord]]:
    """Build the fixture match; returns (shots, rallies) in chronological order."""
    shots: list[ShotRecord] = []
    rallies: list[RallyRecord] = []
    for set_no, seq, (a, b), server, serve_type, shot_specs in _RALLIES:
        rally_id = f"{MATCH_ID}_g{set_no}_r{seq:02d}"
        base_frame = (set_no * 100 + seq) * 1000
        last_player, last_marker = shot_specs[-1][0], shot_specs[-1][2]
        if last_marker == "winner":
            winner = last_player
        else:
            winner = "B" if last_player == "A" else "A"
        n_strokes = len(shot_specs)
        for i, (player, coarse, marker) in enumerate(shot_specs):
            side = _side_of(player, set_no)
            hit = _mirror(_HIT_XY[coarse], side)
            landing_key = f"serve_{serve_type}" if coarse == "serve" else coarse
            landing = (
                None if marker in _ERROR_MARKERS else _mirror(_LANDING_XY[landing_key], side)
            )
            frame_num = base_frame + i * 30
            shots.append(
                ShotRecord(
                    match_id=MATCH_ID,
                    set_no=set_no,
                    rally_id=rally_id,
                    ball_round=i + 1,
                    frame_num=frame_num,
                    t_ms=frame_num / 30.0 * 1000.0,
                    player=player,  # type: ignore[arg-type]
                    side=side,  # type: ignore[arg-type]
                    type_coarse=CoarseShotType(coarse),
                    type_conf=0.9,
                    hit_xy_court=XY(x=hit[0], y=hit[1]),
                    landing_xy_court=XY(x=landing[0], y=landing[1]) if landing else None,
                    landing_area=area_from_xy(*landing) if landing else None,
                    player_location=XY(x=hit[0], y=hit[1]),
                    player_location_area=area_from_xy(*hit),
                    is_winner=True if marker == "winner" else None,
                    is_error=True if marker in _ERROR_MARKERS else None,
                    lose_reason="unforced" if marker == "error_unforced" else None,
                )
            )
        rallies.append(
            RallyRecord(
                rally_id=rally_id,
                match_id=MATCH_ID,
                set_no=set_no,
                seq=seq,
                start_frame=base_frame,
                end_frame=base_frame + n_strokes * 30,
                duration_s=float(n_strokes),
                score_before=ScorePair(a=a, b=b),
                server=server,  # type: ignore[arg-type]
                serve_type=serve_type,  # type: ignore[arg-type]
                winner=winner,  # type: ignore[arg-type]
                n_strokes=n_strokes,
                stroke_ids=[f"{rally_id}_s{i + 1}" for i in range(n_strokes)],
                score_verified=True,
            )
        )
    return shots, rallies

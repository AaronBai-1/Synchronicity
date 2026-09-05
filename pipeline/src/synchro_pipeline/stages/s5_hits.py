"""S5 Hit detection — STUB. docs/plan.md stage "S5 Hit detection" (the linchpin).

Planned approach: reimplement MonoTrack's HitNet formulation — GRU over the shuttle
track plus both players' poses, 3-class no-hit/near-hit/far-hit — fused with
pose-derived swing cues (Sensors-2024 recipe: 90.5% F1 vs 72.3% trajectory-only).
Training labels come free from ShuttleSet frame_num + BFMD hit events. Postprocessing:
near/far alternation prior + minimum-gap constraint. Target hit F1 ≥ 0.90 at ±3 frames.

MonoTrack is BLUEPRINT-ONLY (license unverified): reimplement the ideas, never copy
its code (plan license hard rules).

Declared contract:
    inputs:  shuttle_track (from S4), player_tracks (from S3), rally_segments (from S1b)
    outputs: hit_events — per-rally hit list (frame_num, side, hit_det_conf) that
             seeds schemas.records.ShotRecord rows; model defined with the
             implementation. Every event carries confidence — low-confidence rallies
             are excluded from aggregates and flagged, never silently included
             (plan "Core design principle").

Torch note: HitNet must lazy-import torch inside run() — torch is an optional extra
and this module must import clean without it.
"""

from __future__ import annotations

from synchro_pipeline.stages.base import PipelineContext, Stage
from synchro_pipeline.stages.s1b_rally_gate import RALLY_SEGMENTS
from synchro_pipeline.stages.s3_players import PLAYER_TRACKS
from synchro_pipeline.stages.s4_shuttle import SHUTTLE_TRACK

# Artifact key produced by S5.
HIT_EVENTS = "hit_events"


class S5HitDetection(Stage):
    """STUB — HitNet-style GRU over shuttle track + poses, alternation-constrained."""

    name = "s5_hits"
    version = "0.0.0"
    inputs = (SHUTTLE_TRACK, PLAYER_TRACKS, RALLY_SEGMENTS)
    outputs = (HIT_EVENTS,)

    def run(self, ctx: PipelineContext) -> None:
        raise NotImplementedError(
            "S5 hit detection is not implemented yet — see docs/plan.md 'S5 Hit detection' "
            "(HitNet reimplementation fused with pose swing cues; alternation prior + "
            "min-gap postprocessing; MonoTrack is blueprint-only)."
        )

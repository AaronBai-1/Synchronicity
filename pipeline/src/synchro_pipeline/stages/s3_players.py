"""S3 Players/pose — STUB. docs/plan.md stage "S3 Players".

Planned approach (all Apache/MIT — license hard rule: no Ultralytics YOLOv8/11 (AGPL),
no YOLOv7 (GPL)): YOLOX or RT-DETR detection → court-polygon filter (via S2 homography)
→ ByteTrack association → RTMPose top-down on crops (handles the small far-court
player) → near/far assignment by court half, anchored across side switches by the
domain.scoring side-switch schedule.

Declared contract:
    inputs:  mezzanine_video (file), rally_segments (from S1b),
             court_observations (from S2)
    outputs: player_tracks — per-frame schemas.records.PlayerObservation lists
             (bbox, track_id, side, foot_court_xy in metres, COCO-17 pose);
             container model defined with the implementation.

Torch note: detectors/pose models must lazy-import torch inside run() — optional extra.
"""

from __future__ import annotations

from synchro_pipeline.stages.base import PipelineContext, Stage
from synchro_pipeline.stages.s0_ingest import MEZZANINE_VIDEO
from synchro_pipeline.stages.s1b_rally_gate import RALLY_SEGMENTS
from synchro_pipeline.stages.s2_court import COURT_OBSERVATIONS

# Artifact key produced by S3.
PLAYER_TRACKS = "player_tracks"


class S3PlayerTracking(Stage):
    """STUB — detect, track, and pose both players on rally frames."""

    name = "s3_players"
    version = "0.0.0"
    inputs = (MEZZANINE_VIDEO, RALLY_SEGMENTS, COURT_OBSERVATIONS)
    outputs = (PLAYER_TRACKS,)

    def run(self, ctx: PipelineContext) -> None:
        raise NotImplementedError(
            "S3 player tracking is not implemented yet — see docs/plan.md 'S3 Players' "
            "(YOLOX/RT-DETR + ByteTrack + RTMPose; Apache/MIT only — no AGPL/GPL models)."
        )

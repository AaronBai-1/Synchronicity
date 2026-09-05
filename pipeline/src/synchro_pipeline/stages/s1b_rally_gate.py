"""S1b Rally-camera gating — STUB. docs/plan.md stage "S1b Rally-camera gating".

Planned approach (no standalone classifier): reuse the S2 court keypoint net — a frame
is "game camera" iff court detection succeeds with sane geometry
(domain.court.homography_is_sane), which solves replay rejection, zoom changes, and
calibration with one model; a ResNet-18 rally/non-rally vote trained on BFMD is the
second signal. Gating before the heavy S3–S6 models is what cuts 60–70% of frames and
keeps GPU cost in the plan's $0.50–1.50/match-hour budget.

Declared contract:
    inputs:  mezzanine_video (file), video_shots (s1_shot_splitting.VideoShotList)
    outputs: rally_segments — candidate rally windows on the mezzanine timeline
             (start/end frames per rally, pre score/state-machine verification);
             the concrete pydantic model lands with the implementation and will feed
             schemas.records.RallyRecord (start_frame/end_frame/video_shot_ids).

Torch note: the court net and ResNet vote must lazy-import torch inside run() —
torch is an optional extra and this module must import clean without it.
"""

from __future__ import annotations

from synchro_pipeline.stages.base import PipelineContext, Stage
from synchro_pipeline.stages.s0_ingest import MEZZANINE_VIDEO
from synchro_pipeline.stages.s1_shot_splitting import VIDEO_SHOTS

# Artifact key produced by S1b.
RALLY_SEGMENTS = "rally_segments"


class S1bRallyGate(Stage):
    """STUB — gate camera shots down to candidate rally windows."""

    name = "s1b_rally_gate"
    version = "0.0.0"
    inputs = (MEZZANINE_VIDEO, VIDEO_SHOTS)
    outputs = (RALLY_SEGMENTS,)

    def run(self, ctx: PipelineContext) -> None:
        raise NotImplementedError(
            "S1b rally-camera gating is not implemented yet — see docs/plan.md "
            "'S1b Rally-camera gating' (reuse the S2 court net as the game-camera test; "
            "ResNet-18 BFMD vote as second signal)."
        )

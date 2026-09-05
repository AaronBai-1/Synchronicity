"""S2 Court homography — STUB. docs/plan.md stage "S2 Court homography".

The main custom build: a heatmap keypoint net predicting the 16 court keypoints of
domain.court.COURT_KEYPOINT_NAMES (TennisCourtDetector architecture retrained on
2–5k self-labeled badminton frames — architecture only; we train our own weights),
then per-frame RANSAC via domain.court.fit_homography + temporal smoothing.
Target: <5px reprojection at 720p; homography_is_sane() doubles as S1b's camera gate.

Declared contract:
    inputs:  mezzanine_video (file), rally_segments (from S1b)
    outputs: court_observations — per-frame schemas.records.CourtObservation
             (keypoints in mezzanine px, 3x3 image→court-metres homography, validity),
             for rally frames only; container model defined with the implementation.

Torch note: the keypoint net must lazy-import torch inside run() — torch is an
optional extra and this module must import clean without it.
"""

from __future__ import annotations

from synchro_pipeline.stages.base import PipelineContext, Stage
from synchro_pipeline.stages.s0_ingest import MEZZANINE_VIDEO
from synchro_pipeline.stages.s1b_rally_gate import RALLY_SEGMENTS

# Artifact key produced by S2.
COURT_OBSERVATIONS = "court_observations"


class S2CourtHomography(Stage):
    """STUB — per-frame court keypoints → RANSAC homography → temporal smoothing."""

    name = "s2_court"
    version = "0.0.0"
    inputs = (MEZZANINE_VIDEO, RALLY_SEGMENTS)
    outputs = (COURT_OBSERVATIONS,)

    def run(self, ctx: PipelineContext) -> None:
        raise NotImplementedError(
            "S2 court homography is not implemented yet — see docs/plan.md "
            "'S2 Court homography' (16-keypoint heatmap net + domain.court.fit_homography; "
            "<5px reprojection target at 720p)."
        )

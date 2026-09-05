"""S4 Shuttle tracking — STUB. docs/plan.md stage "S4 Shuttle".

Planned approach: TrackNetV3 (MIT, pretrained) on rally frames only — rally gating
first is what keeps GPU cost inside the plan's budget. Every frame carries
``visibility: detected|inpainted|missing`` downstream (schemas.records.Visibility) so
S5/S7 can discount inpainted points instead of trusting them silently. WASB-SBDT (MIT)
serves as an eval cross-check, not a production dependency. Weight provenance must be
confirmed for commercial use before launch (plan "Top risks" #3).

Declared contract:
    inputs:  mezzanine_video (file), rally_segments (from S1b)
    outputs: shuttle_track — per-frame schemas.records.ShuttleObservation in mezzanine
             px with visibility labels; container model defined with the implementation.

Torch note: TrackNetV3 must lazy-import torch inside run() — torch is an optional
extra and this module must import clean without it.
"""

from __future__ import annotations

from synchro_pipeline.stages.base import PipelineContext, Stage
from synchro_pipeline.stages.s0_ingest import MEZZANINE_VIDEO
from synchro_pipeline.stages.s1b_rally_gate import RALLY_SEGMENTS

# Artifact key produced by S4.
SHUTTLE_TRACK = "shuttle_track"


class S4ShuttleTracking(Stage):
    """STUB — TrackNetV3 shuttle trajectory on rally frames."""

    name = "s4_shuttle"
    version = "0.0.0"
    inputs = (MEZZANINE_VIDEO, RALLY_SEGMENTS)
    outputs = (SHUTTLE_TRACK,)

    def run(self, ctx: PipelineContext) -> None:
        raise NotImplementedError(
            "S4 shuttle tracking is not implemented yet — see docs/plan.md 'S4 Shuttle' "
            "(TrackNetV3 on rally frames only; per-frame visibility carried downstream)."
        )

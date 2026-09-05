"""S7 Metrics/insights — STUB. docs/plan.md stage "S7 Metrics".

Planned approach: pure-CPU aggregation — positions/placement through the S2 homography
(domain.court.area_from_xy for ShuttleSet-style area ids), velocity as 2D court-plane
speed bands explicitly labeled "estimated" (3D lift deferred; never print a km/h we
can't defend), scenario/serve analytics as SQL over the shot table, ShuttleNet-style
player-profile embeddings in a later phase. The scoring state machine
(domain.scoring.MatchState) cross-checks score deltas vs rally boundaries vs serve
sides here; inconsistencies become qa_flags, and low-confidence rallies are excluded
from aggregates and flagged, never silently included.

Declared contract:
    inputs:  shot_records (from S6), rally_segments (from S1b),
             score_timeline (from S1c), source_meta (from S0)
    outputs: match_record — schemas.records.MatchRecord (incl. pipeline_version,
             stage_model_hashes from the stage manifests, quality_report)
             rally_records — verified schemas.records.RallyRecord list with
             per-rally confidence; container model defined with the implementation.
"""

from __future__ import annotations

from synchro_pipeline.stages.base import PipelineContext, Stage
from synchro_pipeline.stages.s0_ingest import SOURCE_META
from synchro_pipeline.stages.s1b_rally_gate import RALLY_SEGMENTS
from synchro_pipeline.stages.s1c_score_ocr import SCORE_TIMELINE
from synchro_pipeline.stages.s6_shot_classify import SHOT_RECORDS

# Artifact keys produced by S7.
MATCH_RECORD = "match_record"
RALLY_RECORDS = "rally_records"


class S7Metrics(Stage):
    """STUB — state-machine-verified match/rally records and aggregate metrics."""

    name = "s7_metrics"
    version = "0.0.0"
    inputs = (SHOT_RECORDS, RALLY_SEGMENTS, SCORE_TIMELINE, SOURCE_META)
    outputs = (MATCH_RECORD, RALLY_RECORDS)

    def run(self, ctx: PipelineContext) -> None:
        raise NotImplementedError(
            "S7 metrics is not implemented yet — see docs/plan.md 'S7 Metrics' "
            "(homography-based placement, estimated speed bands, state-machine "
            "cross-checks emitting qa_flags)."
        )

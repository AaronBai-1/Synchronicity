"""S6 Shot classification — STUB. docs/plan.md stage "S6 Shot classification".

Planned approach: BST-style transformer over pose + shuttle + positions (exactly what
S2–S5 emit), retrained on ShuttleSet clips regenerated through our own perception
stack so train and inference distributions match. Output granularity is the merged
~11-class taxonomy (domain.taxonomy.MergedShotType) — ≥80% accuracy only exists
merged; never surface finer labels than the model can defend. The full probability
distribution is stored (ShotRecord.type_probs), not just the argmax.

Declared contract:
    inputs:  hit_events (from S5), shuttle_track (from S4), player_tracks (from S3),
             court_observations (from S2)
    outputs: shot_records — the product's core contract: one
             schemas.records.ShotRecord per stroke (ShuttleSet-compatible superset);
             container model defined with the implementation.

Torch note: the BST transformer must lazy-import torch inside run() — optional extra.
"""

from __future__ import annotations

from synchro_pipeline.stages.base import PipelineContext, Stage
from synchro_pipeline.stages.s2_court import COURT_OBSERVATIONS
from synchro_pipeline.stages.s3_players import PLAYER_TRACKS
from synchro_pipeline.stages.s4_shuttle import SHUTTLE_TRACK
from synchro_pipeline.stages.s5_hits import HIT_EVENTS

# Artifact key produced by S6.
SHOT_RECORDS = "shot_records"


class S6ShotClassification(Stage):
    """STUB — BST-style stroke classifier at merged-class granularity."""

    name = "s6_shot_classify"
    version = "0.0.0"
    inputs = (HIT_EVENTS, SHUTTLE_TRACK, PLAYER_TRACKS, COURT_OBSERVATIONS)
    outputs = (SHOT_RECORDS,)

    def run(self, ctx: PipelineContext) -> None:
        raise NotImplementedError(
            "S6 shot classification is not implemented yet — see docs/plan.md "
            "'S6 Shot classification' (BST retrained on ShuttleSet regenerated through "
            "our own perception stack; merged classes only)."
        )

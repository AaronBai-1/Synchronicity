"""Pipeline stages S0–S7 (docs/plan.md "Pipeline architecture") and the run framework.

Honest status of the DAG (Phase 0 walking skeleton):
    implemented: S0 ingest, S1a shot splitting
    stubbed:     S1b rally gate, S1c score OCR, S2 court, S3 players, S4 shuttle,
                 S5 hits, S6 shot classify, S7 metrics (each raises NotImplementedError
                 pointing at its plan section)
"""

from __future__ import annotations

from synchro_pipeline.stages.base import (
    SOURCE_VIDEO,
    ArtifactStore,
    PipelineContext,
    PipelineError,
    PipelineReport,
    Stage,
    StageReport,
    run_pipeline,
)
from synchro_pipeline.stages.s0_ingest import S0Ingest
from synchro_pipeline.stages.s1_shot_splitting import S1aShotSplitting
from synchro_pipeline.stages.s1b_rally_gate import S1bRallyGate
from synchro_pipeline.stages.s1c_score_ocr import S1cScoreOcr
from synchro_pipeline.stages.s2_court import S2CourtHomography
from synchro_pipeline.stages.s3_players import S3PlayerTracking
from synchro_pipeline.stages.s4_shuttle import S4ShuttleTracking
from synchro_pipeline.stages.s5_hits import S5HitDetection
from synchro_pipeline.stages.s6_shot_classify import S6ShotClassification
from synchro_pipeline.stages.s7_metrics import S7Metrics


def default_pipeline() -> list[Stage]:
    """The full S0–S7 DAG in plan order — run_pipeline re-derives ordering from the
    declared artifact keys, so this list is a factory, not a hidden second source of
    dependency truth."""
    return [
        S0Ingest(),
        S1aShotSplitting(),
        S1bRallyGate(),
        S1cScoreOcr(),
        S2CourtHomography(),
        S3PlayerTracking(),
        S4ShuttleTracking(),
        S5HitDetection(),
        S6ShotClassification(),
        S7Metrics(),
    ]


__all__ = [
    "SOURCE_VIDEO",
    "ArtifactStore",
    "PipelineContext",
    "PipelineError",
    "PipelineReport",
    "S0Ingest",
    "S1aShotSplitting",
    "S1bRallyGate",
    "S1cScoreOcr",
    "S2CourtHomography",
    "S3PlayerTracking",
    "S4ShuttleTracking",
    "S5HitDetection",
    "S6ShotClassification",
    "S7Metrics",
    "Stage",
    "StageReport",
    "default_pipeline",
    "run_pipeline",
]

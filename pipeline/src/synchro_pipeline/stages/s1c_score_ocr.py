"""S1c Score OCR — STUB. docs/plan.md stage "S1c Score OCR".

Planned approach: auto-localize the static score bug via temporal pixel variance,
PaddleOCR digit reads, then constraint-decode through domain.scoring — only
`valid_next_scores` transitions are accepted, which is how ≥99% score-tuple accuracy
is reached from noisy per-frame reads. Score deltas also cross-validate S1b's rally
boundaries (plan "Core design principle": mutually verifying signals).

Declared contract:
    inputs:  mezzanine_video (file), rally_segments (from S1b)
    outputs: score_timeline — per-rally (score_before, score_after, server) tuples
             feeding schemas.records.RallyRecord.score_before/score_after/server and
             score_verified; model defined with the implementation.

Torch/Paddle note: OCR runtimes must lazy-import inside run() — optional extras only.
"""

from __future__ import annotations

from synchro_pipeline.stages.base import PipelineContext, Stage
from synchro_pipeline.stages.s0_ingest import MEZZANINE_VIDEO
from synchro_pipeline.stages.s1b_rally_gate import RALLY_SEGMENTS

# Artifact key produced by S1c.
SCORE_TIMELINE = "score_timeline"


class S1cScoreOcr(Stage):
    """STUB — scoreboard OCR wrapped in the scoring state machine."""

    name = "s1c_score_ocr"
    version = "0.0.0"
    inputs = (MEZZANINE_VIDEO, RALLY_SEGMENTS)
    outputs = (SCORE_TIMELINE,)

    def run(self, ctx: PipelineContext) -> None:
        raise NotImplementedError(
            "S1c score OCR is not implemented yet — see docs/plan.md 'S1c Score OCR' "
            "(temporal-variance bug localization, PaddleOCR digits, constraint decoding "
            "through domain.scoring.valid_next_scores)."
        )

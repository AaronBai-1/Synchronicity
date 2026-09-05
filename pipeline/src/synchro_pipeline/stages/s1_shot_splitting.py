"""S1a Shot splitting — docs/plan.md stage "S1a Shot splitting".

PySceneDetect's AdaptiveDetector segments the mezzanine into broadcast camera shots
("video shots" here — never to be confused with badminton strokes, which live in
schemas.records.ShotRecord). Rally boundaries (S1b) and clip export both consume these
segments; RallyRecord.video_shot_ids references them by ``shot_id``.

AdaptiveDetector (rolling-average content delta) rather than a fixed threshold because
broadcast badminton mixes static end-court views with fast pans/replays. The plan's
TransNetV2 refinement for fades/wipes is a later version of *this* stage — when it
lands, bump `version` and the cache framework reprocesses affected matches for free.

Frame conventions: frames index the 30 fps CFR mezzanine (the time authority);
``end_frame`` is exclusive, so consecutive segments tile the video exactly.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field
from scenedetect import AdaptiveDetector, detect

from synchro_pipeline.stages.base import PipelineContext, PipelineError, Stage
from synchro_pipeline.stages.s0_ingest import MEZZANINE_VIDEO

logger = logging.getLogger(__name__)

# Artifact key produced by S1a.
VIDEO_SHOTS = "video_shots"

# Broadcast camera shots shorter than ~0.5 s at 30 fps are cut-detection noise, not
# real direction changes.
_MIN_SHOT_LEN_FRAMES = 15


class VideoShot(BaseModel):
    """One broadcast camera shot on the mezzanine timeline (end_frame exclusive)."""

    model_config = ConfigDict(extra="forbid")

    shot_id: int = Field(ge=0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)


class VideoShotList(BaseModel):
    """S1a's output artifact: the mezzanine tiled into camera shots."""

    model_config = ConfigDict(extra="forbid")

    fps: float
    segments: list[VideoShot] = Field(default_factory=list)


class S1aShotSplitting(Stage):
    """Split the mezzanine into camera shots with PySceneDetect's AdaptiveDetector."""

    name = "s1a_shot_splitting"
    version = "0.1.0"
    inputs = (MEZZANINE_VIDEO,)
    outputs = (VIDEO_SHOTS,)

    def __init__(
        self,
        adaptive_threshold: float = 3.0,
        min_shot_len_frames: int = _MIN_SHOT_LEN_FRAMES,
    ) -> None:
        # Detector knobs are constructor params so eval sweeps (golden-set rally F1,
        # plan "Verification approach") can grid-search without editing the stage.
        # Non-default values should ship with a version bump.
        self.adaptive_threshold = adaptive_threshold
        self.min_shot_len_frames = min_shot_len_frames

    def run(self, ctx: PipelineContext) -> None:
        mezzanine = ctx.artifacts.load_path(MEZZANINE_VIDEO)
        # start_in_scene=True: a cut-free video is one shot, not zero — downstream
        # rally gating must always see full-timeline coverage.
        scenes = detect(
            str(mezzanine),
            AdaptiveDetector(
                adaptive_threshold=self.adaptive_threshold,
                min_scene_len=self.min_shot_len_frames,
            ),
            start_in_scene=True,
        )
        if not scenes:
            raise PipelineError(f"scene detection returned no frames for {mezzanine}")

        fps = float(scenes[0][0].framerate)
        segments = [
            VideoShot(
                shot_id=idx,
                start_frame=start.frame_num,
                end_frame=end.frame_num,
                start_s=start.seconds,
                end_s=end.seconds,
            )
            for idx, (start, end) in enumerate(scenes)
        ]
        logger.info("s1a: %d camera shot(s) over %.1fs", len(segments), segments[-1].end_s)
        ctx.artifacts.save_model(self, VIDEO_SHOTS, VideoShotList(fps=fps, segments=segments))

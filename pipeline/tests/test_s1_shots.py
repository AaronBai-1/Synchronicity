"""S1a shot splitting: a synthetic hard cut must yield two segments split at the cut."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from synchro_pipeline.stages.base import PipelineContext, run_pipeline
from synchro_pipeline.stages.s0_ingest import S0Ingest
from synchro_pipeline.stages.s1_shot_splitting import (
    VIDEO_SHOTS,
    S1aShotSplitting,
    VideoShotList,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)

FPS = 30
CUT_FRAME = 2 * FPS  # hard cut at t=2s → mezzanine frame 60
TOTAL_FRAMES = 4 * FPS


@pytest.fixture(scope="module")
def cut_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """4 s @ 30 fps: 2 s moving test pattern, hard cut to 2 s solid blue."""
    path = tmp_path_factory.mktemp("video") / "cut.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=128x72:rate={FPS}:duration=2",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:size=128x72:rate={FPS}:duration=2",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture(scope="module")
def shots(tmp_path_factory: pytest.TempPathFactory, cut_video: Path) -> VideoShotList:
    """Run S0 → S1a once for the module; assertions share the result."""
    ctx = PipelineContext.create(
        match_id="m-s1",
        workdir=tmp_path_factory.mktemp("work"),
        source_video=cut_video,
    )
    run_pipeline([S0Ingest(), S1aShotSplitting()], ctx)
    return ctx.artifacts.load_model(VIDEO_SHOTS, VideoShotList)


class TestS1aShotSplitting:
    def test_finds_exactly_the_one_hard_cut(self, shots: VideoShotList):
        assert len(shots.segments) == 2
        boundary = shots.segments[1].start_frame
        assert abs(boundary - CUT_FRAME) <= 3

    def test_segments_tile_the_mezzanine(self, shots: VideoShotList):
        first, second = shots.segments
        assert (first.shot_id, second.shot_id) == (0, 1)
        assert first.start_frame == 0
        assert first.end_frame == second.start_frame  # end exclusive, no gap/overlap
        assert abs(second.end_frame - TOTAL_FRAMES) <= 3

    def test_times_derive_from_frames(self, shots: VideoShotList):
        assert shots.fps == pytest.approx(30.0)
        for seg in shots.segments:
            assert seg.start_s == pytest.approx(seg.start_frame / shots.fps)
            assert seg.end_s == pytest.approx(seg.end_frame / shots.fps)

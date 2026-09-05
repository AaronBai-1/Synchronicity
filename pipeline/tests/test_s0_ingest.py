"""S0 ingest: synthetic video → validated SourceMeta + canonical 720p30 CFR mezzanine."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from synchro_pipeline.schemas.records import SourceMeta
from synchro_pipeline.stages.base import PipelineContext, run_pipeline
from synchro_pipeline.stages.s0_ingest import (
    INGEST_INFO,
    MEZZANINE_AUDIO,
    MEZZANINE_VIDEO,
    SOURCE_META,
    IngestError,
    IngestInfo,
    S0Ingest,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


def _probe(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """2 s, 128x72 (16:9, sub-720p), 30 fps, no audio — tiny so the suite stays fast."""
    path = tmp_path_factory.mktemp("video") / "src.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=128x72:rate=30:duration=2",
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


@pytest.fixture()
def ran_ctx(tmp_path: Path, synthetic_video: Path) -> PipelineContext:
    ctx = PipelineContext.create(
        match_id="m-s0", workdir=tmp_path / "work", source_video=synthetic_video
    )
    run_pipeline([S0Ingest()], ctx)
    return ctx


class TestS0Ingest:
    def test_source_meta_describes_the_mezzanine(self, ran_ctx: PipelineContext):
        meta = ran_ctx.artifacts.load_model(SOURCE_META, SourceMeta)
        assert meta.height == 720
        assert meta.width == 1280  # 128x72 upscaled at fixed aspect
        assert meta.fps_effective == pytest.approx(30.0)
        assert meta.duration_s == pytest.approx(2.0, abs=0.3)

    def test_mezzanine_is_720p30_cfr_h264(self, ran_ctx: PipelineContext):
        mezz = ran_ctx.artifacts.load_path(MEZZANINE_VIDEO)
        assert mezz.exists() and mezz.stat().st_size > 0
        stream = next(
            s for s in _probe(mezz)["streams"] if s["codec_type"] == "video"
        )
        assert stream["codec_name"] == "h264"
        assert (stream["width"], stream["height"]) == (1280, 720)
        assert stream["avg_frame_rate"] == "30/1"
        assert stream["r_frame_rate"] == "30/1"  # CFR: nominal == effective

    def test_silent_audio_synthesized_and_flagged(self, ran_ctx: PipelineContext):
        wav = ran_ctx.artifacts.load_path(MEZZANINE_AUDIO)
        stream = next(s for s in _probe(wav)["streams"] if s["codec_type"] == "audio")
        assert stream["codec_name"] == "pcm_s16le"
        assert int(stream["sample_rate"]) == 16000
        assert int(stream["channels"]) == 1

        info = ran_ctx.artifacts.load_model(INGEST_INFO, IngestInfo)
        assert info.has_source_audio is False
        assert "no_audio_stream" in info.qa_flags
        assert "upscaled_below_720p" in info.qa_flags
        assert (info.source_width, info.source_height) == (128, 72)

    def test_rerun_is_a_cache_hit(self, ran_ctx: PipelineContext):
        report = run_pipeline([S0Ingest()], ran_ctx)
        assert [s.cached for s in report.stages] == [True]

    def test_garbage_input_rejected(self, tmp_path: Path):
        garbage = tmp_path / "not_a_video.mp4"
        garbage.write_bytes(b"this is not media")
        ctx = PipelineContext.create(
            match_id="m-bad", workdir=tmp_path / "work", source_video=garbage
        )
        with pytest.raises(IngestError):
            run_pipeline([S0Ingest()], ctx)

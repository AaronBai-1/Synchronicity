"""S0 Ingest — docs/plan.md stage "S0 Ingest": FFmpeg → canonical 720p30 CFR mezzanine.

Everything downstream references the mezzanine, never the upload: records.py pins image
coordinates to "the 720p mezzanine resolution", and the plan makes the mezzanine's
*frame index* the time authority — so `SourceMeta.fps_effective` is probed from the
transcoded mezzanine, not trusted from the source container.

Scaling choice: scale to height 720 preserving aspect ratio (``scale=-2:720``, width
rounded to even for yuv420p). No pad/crop — a non-16:9 mezzanine is fine because all
geometry flows through the S2 homography, and stretching or padding would only distort
court keypoint training data. Sub-720p sources (e.g. tiny test videos) are upscaled and
flagged, never rejected.

Audio: mono 16 kHz WAV for the eventual S5 audio hit-cue experiments (plan Phase 5+).
Sources with no audio stream get a silent WAV of matching duration so the artifact
contract stays uniform — with an explicit ``no_audio_stream`` qa_flag in IngestInfo,
never a silent (pun intended) substitution.
"""

from __future__ import annotations

import json
import logging
import subprocess
from fractions import Fraction
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from synchro_pipeline.schemas.records import SourceMeta
from synchro_pipeline.stages.base import SOURCE_VIDEO, PipelineContext, Stage

logger = logging.getLogger(__name__)

# Artifact keys produced by S0.
SOURCE_META = "source_meta"
MEZZANINE_VIDEO = "mezzanine_video"
MEZZANINE_AUDIO = "mezzanine_audio"
INGEST_INFO = "ingest_info"

MEZZANINE_HEIGHT = 720
MEZZANINE_FPS = 30
AUDIO_SAMPLE_RATE = 16_000

# Sanity bounds for the *source* (rejects still images, corrupt headers, screen grabs
# with absurd rates). The mezzanine is always exactly 30 fps CFR afterwards.
_MIN_DURATION_S = 0.2
_MIN_FPS = 5.0
_MAX_FPS = 240.0


class IngestError(RuntimeError):
    """The upload failed validation or transcoding — surfaced to the user, not retried."""


class IngestInfo(BaseModel):
    """Facts about the original upload (the mezzanine's own facts live in SourceMeta).

    Kept as a separate artifact so SourceMeta stays exactly the schemas.records contract
    while ingest-side honesty flags (upscaling, missing audio) still land somewhere.
    """

    model_config = ConfigDict(extra="forbid")

    container_format: str | None = None
    video_codec: str | None = None
    source_width: int
    source_height: int
    source_fps: float
    source_duration_s: float
    has_source_audio: bool
    qa_flags: list[str] = Field(default_factory=list)


# --- ffprobe / ffmpeg plumbing -----------------------------------------------------


def mezzanine_transcode_cmd(source: Path, out: Path) -> list[str]:
    """The canonical mezzanine ffmpeg invocation — the single source of truth.

    Used by both the pipeline stage and the `python -m` CLI so hand-made mezzanines
    (docs/golden-file-guide.md §2) can never drift from what the pipeline produces.
    """
    return [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-vf",
        f"scale=-2:{MEZZANINE_HEIGHT},setsar=1",
        "-r",
        str(MEZZANINE_FPS),
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(out),
    ]


def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise IngestError(f"{what} failed (exit {proc.returncode}): {proc.stderr.strip()[:2000]}")
    return proc


def ffprobe(path: Path) -> dict:
    """Container + stream metadata as a dict (``ffprobe -print_format json``)."""
    proc = _run(
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
        what=f"ffprobe on {path.name}",
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise IngestError(f"ffprobe produced unparseable output for {path.name}") from exc


def _parse_rate(rate: str | None) -> float | None:
    """'30000/1001' → 29.97…; None for missing/degenerate ('0/0') rates."""
    if not rate:
        return None
    try:
        value = Fraction(rate)
    except (ValueError, ZeroDivisionError):
        return None
    return float(value) if value > 0 else None


def _stream_of(probe: dict, codec_type: str) -> dict | None:
    return next((s for s in probe.get("streams", []) if s.get("codec_type") == codec_type), None)


def _duration_s(probe: dict, video_stream: dict) -> float | None:
    for holder in (probe.get("format", {}), video_stream):
        try:
            return float(holder["duration"])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _fps_of(video_stream: dict) -> float | None:
    # avg_frame_rate = frames/duration (the honest number for VFR sources);
    # r_frame_rate is the container's nominal tick rate — fallback only.
    return _parse_rate(video_stream.get("avg_frame_rate")) or _parse_rate(
        video_stream.get("r_frame_rate")
    )


# --- the stage ---------------------------------------------------------------------


class S0Ingest(Stage):
    """Validate the upload, transcode the canonical mezzanine, extract audio."""

    name = "s0_ingest"
    version = "0.1.0"
    inputs = (SOURCE_VIDEO,)
    outputs = (SOURCE_META, MEZZANINE_VIDEO, MEZZANINE_AUDIO, INGEST_INFO)

    def run(self, ctx: PipelineContext) -> None:
        source = ctx.artifacts.load_path(SOURCE_VIDEO)
        probe = ffprobe(source)
        video_stream = _stream_of(probe, "video")
        if video_stream is None:
            raise IngestError(f"{source.name}: no video stream found")

        src_fps = _fps_of(video_stream)
        src_duration = _duration_s(probe, video_stream)
        src_width = int(video_stream.get("width") or 0)
        src_height = int(video_stream.get("height") or 0)
        if src_width <= 0 or src_height <= 0:
            raise IngestError(f"{source.name}: undecodable frame size {src_width}x{src_height}")
        if src_duration is None or src_duration < _MIN_DURATION_S:
            raise IngestError(f"{source.name}: missing/implausible duration ({src_duration})")
        if src_fps is None or not (_MIN_FPS <= src_fps <= _MAX_FPS):
            raise IngestError(f"{source.name}: implausible frame rate ({src_fps})")

        qa_flags: list[str] = []
        if src_height < MEZZANINE_HEIGHT:
            qa_flags.append("upscaled_below_720p")

        # -y + a fixed filename inside the (freshly wiped) stage dir keeps this idempotent.
        mezzanine = ctx.artifacts.prepare_file(self, "mezzanine.mp4")
        _run(mezzanine_transcode_cmd(source, mezzanine), what="mezzanine transcode")

        audio_stream = _stream_of(probe, "audio")
        wav = ctx.artifacts.prepare_file(self, "audio.wav")
        if audio_stream is not None:
            audio_cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(source)]
        else:
            qa_flags.append("no_audio_stream")
            logger.warning("%s has no audio stream; writing silent WAV", source.name)
            audio_cmd = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r={AUDIO_SAMPLE_RATE}:cl=mono",
                "-t",
                f"{src_duration:.3f}",
            ]
        audio_cmd += [
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(AUDIO_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(wav),
        ]
        _run(audio_cmd, what="audio extraction")

        # Probe the *mezzanine* — the plan makes its frame index the time authority,
        # so fps_effective/dimensions/duration must describe what we actually wrote.
        mezz_probe = ffprobe(mezzanine)
        mezz_stream = _stream_of(mezz_probe, "video")
        if mezz_stream is None:
            raise IngestError("mezzanine transcode produced no video stream")
        mezz_fps = _fps_of(mezz_stream)
        mezz_duration = _duration_s(mezz_probe, mezz_stream)
        if mezz_fps is None or mezz_duration is None:
            raise IngestError("mezzanine probe missing fps/duration")

        meta = SourceMeta(
            width=int(mezz_stream["width"]),
            height=int(mezz_stream["height"]),
            fps_effective=mezz_fps,
            duration_s=mezz_duration,
            broadcaster_guess=None,  # Phase 1: infer from score-bug layout (plan S1c)
        )
        info = IngestInfo(
            container_format=probe.get("format", {}).get("format_name"),
            video_codec=video_stream.get("codec_name"),
            source_width=src_width,
            source_height=src_height,
            source_fps=src_fps,
            source_duration_s=src_duration,
            has_source_audio=audio_stream is not None,
            qa_flags=qa_flags,
        )

        ctx.artifacts.save_model(self, SOURCE_META, meta)
        ctx.artifacts.save_model(self, INGEST_INFO, info)
        ctx.artifacts.register_file(self, MEZZANINE_VIDEO, mezzanine)
        ctx.artifacts.register_file(self, MEZZANINE_AUDIO, wav)


def main(argv: list[str] | None = None) -> int:
    """`python -m synchro_pipeline.stages.s0_ingest <source> <out.mp4>` — make a
    mezzanine outside the pipeline (golden-set prep) with the exact stage encode."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m synchro_pipeline.stages.s0_ingest",
        description="Transcode a source video to the canonical 720p30 CFR mezzanine "
        "using the pipeline's own ffmpeg parameters (drift-proof by construction).",
    )
    parser.add_argument("source", type=Path, help="source video (mp4/webm/mkv)")
    parser.add_argument("out", type=Path, help="output mezzanine path (.mp4)")
    args = parser.parse_args(argv)

    probe = ffprobe(args.source)
    if _stream_of(probe, "video") is None:
        print(f"error: {args.source}: no video stream found")
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _run(mezzanine_transcode_cmd(args.source, args.out), what="mezzanine transcode")

    out_probe = ffprobe(args.out)
    stream = _stream_of(out_probe, "video")
    fps = _fps_of(stream) if stream else None
    duration = _duration_s(out_probe, stream) if stream else None
    print(
        f"mezzanine written: {args.out} "
        f"({stream.get('width')}x{stream.get('height')} @ {fps} fps, {duration:.1f}s)"
        if stream
        else f"mezzanine written: {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

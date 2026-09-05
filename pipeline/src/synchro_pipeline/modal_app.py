"""Modal deployment skeleton — docs/plan.md "Platform": each pipeline stage group is a
Modal serverless function (queue/retries/scale-to-zero built in; no self-managed GPU
fleet). RunPod is the fallback provider.

Modal is an OPTIONAL extra and is not installed in the default environment, so this
module import-guards it: everything degrades to `app = None` plus locally-runnable
`run_stage_group`, and the Modal function definitions only exist when modal imports.

Eventual deploy path:
    uv sync --extra cloud
    modal deploy pipeline/src/synchro_pipeline/modal_app.py

Stage grouping mirrors the plan's stage DAG table (sized cheap→expensive so rally
gating cuts 60–70% of frames before anything touches an L4):
    s0_ingest                 → CPU        (FFmpeg only)
    s1_preprocess (S1a–S1c)   → CPU/T4     (PySceneDetect CPU; court-net gate + OCR
                                            want a small GPU once implemented)
    s2_s3_perception (S2–S3)  → T4         (court keypoint net, YOLOX/ByteTrack/RTMPose)
    s4_s6_shuttle_hits_shots  → L4         (TrackNetV3 + HitNet + BST on rally frames)
    s7_metrics                → CPU        (pure aggregation over the shot table)

All groups share one Modal Volume mounted at /matches: ArtifactStore scans the
per-stage manifests there, so each group finds upstream artifacts and the
content-addressed cache works identically across containers.
"""

from __future__ import annotations

from pathlib import Path

from synchro_pipeline.stages import default_pipeline
from synchro_pipeline.stages.base import PipelineContext, PipelineReport, run_pipeline

try:  # optional 'cloud' extra — keep this module importable without it
    import modal
except ModuleNotFoundError:  # pragma: no cover - exercised implicitly in CI (no modal)
    modal = None  # type: ignore[assignment]

APP_NAME = "synchro-pipeline"
VOLUME_NAME = "synchro-match-workdirs"
VOLUME_MOUNT = "/matches"

# Group name → stage names (must match Stage.name values in synchro_pipeline.stages).
STAGE_GROUPS: dict[str, tuple[str, ...]] = {
    "s0_ingest": ("s0_ingest",),
    "s1_preprocess": ("s1a_shot_splitting", "s1b_rally_gate", "s1c_score_ocr"),
    "s2_s3_perception": ("s2_court", "s3_players"),
    "s4_s6_shuttle_hits_shots": ("s4_shuttle", "s5_hits", "s6_shot_classify"),
    "s7_metrics": ("s7_metrics",),
}


def run_stage_group(
    group: str, match_id: str, workdir: Path, source_video: Path
) -> PipelineReport:
    """Run one stage group locally — the body every Modal function is a thin wrapper of.

    Keeping the real logic here (and Modal-free) means local dev, tests, and the cloud
    run the exact same code path; only the container/GPU placement differs.
    """
    if group not in STAGE_GROUPS:
        raise KeyError(f"unknown stage group '{group}' (expected one of {sorted(STAGE_GROUPS)})")
    names = STAGE_GROUPS[group]
    stages = [stage for stage in default_pipeline() if stage.name in names]
    ctx = PipelineContext.create(match_id=match_id, workdir=workdir, source_video=source_video)
    return run_pipeline(stages, ctx)


if modal is not None:
    app = modal.App(APP_NAME)

    _image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg")
        .pip_install(
            "pydantic>=2.7",
            "numpy>=1.26",
            "opencv-python-headless>=4.10",
            "scenedetect>=0.6.4",
        )
        # GPU groups additionally need the 'ml' extra (torch/torchvision) baked into
        # their image once S2+ are implemented; stages lazy-import torch so the CPU
        # groups never pay for it.
    )
    _volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    _volumes = {VOLUME_MOUNT: _volume}

    def _run_group_remote(group: str, match_id: str, source_video: str) -> dict:
        report = run_stage_group(
            group,
            match_id=match_id,
            workdir=Path(VOLUME_MOUNT) / match_id,
            source_video=Path(source_video),
        )
        _volume.commit()
        return report.model_dump(mode="json")

    @app.function(image=_image, volumes=_volumes, timeout=1800)
    def s0_ingest(match_id: str, source_video: str) -> dict:
        """S0 on CPU: ffprobe validation + mezzanine transcode + audio extraction."""
        return _run_group_remote("s0_ingest", match_id, source_video)

    @app.function(image=_image, volumes=_volumes, gpu="T4", timeout=1800)
    def s1_preprocess(match_id: str, source_video: str) -> dict:
        """S1a–S1c: shot splitting (CPU), rally gating + score OCR (small GPU)."""
        return _run_group_remote("s1_preprocess", match_id, source_video)

    @app.function(image=_image, volumes=_volumes, gpu="T4", timeout=3600)
    def s2_s3_perception(match_id: str, source_video: str) -> dict:
        """S2–S3: court homography + player tracking/pose on rally frames."""
        return _run_group_remote("s2_s3_perception", match_id, source_video)

    @app.function(image=_image, volumes=_volumes, gpu="L4", timeout=3600)
    def s4_s6_shuttle_hits_shots(match_id: str, source_video: str) -> dict:
        """S4–S6 on L4: shuttle tracking, hit detection, shot classification."""
        return _run_group_remote("s4_s6_shuttle_hits_shots", match_id, source_video)

    @app.function(image=_image, volumes=_volumes, timeout=900)
    def s7_metrics(match_id: str, source_video: str) -> dict:
        """S7 on CPU: state-machine-verified records + aggregate metrics."""
        return _run_group_remote("s7_metrics", match_id, source_video)

    @app.local_entrypoint()
    def process_match(match_id: str, source_video: str) -> None:
        """Sequential end-to-end run (Phase 0 walking skeleton; queueing comes later)."""
        for fn in (s0_ingest, s1_preprocess, s2_s3_perception, s4_s6_shuttle_hits_shots, s7_metrics):
            report = fn.remote(match_id, source_video)
            print(report)

else:  # modal not installed — cloud deployment unavailable, local runs unaffected
    app = None

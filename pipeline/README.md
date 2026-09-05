# synchro_pipeline

Badminton analytics CV/ML pipeline: stages S0–S7, schema contracts, the golden-set
evaluation harness, and the analytics/tactical suite. See `docs/plan.md` at the repo
root for the full architecture; this README covers day-to-day setup and the Phase 0
golden-set workflow.

## Setup

```sh
# from the workspace root — installs pipeline + api workspace members into .venv
uv sync --all-packages

# heavy ML deps (torch/torchvision) are an opt-in extra; NOT needed for tests or the
# fake-tracker benchmark path
uv sync --all-packages --extra ml
```

Run checks:

```sh
uv run pytest pipeline/tests -q
uv run ruff check pipeline
```

Everything under `pipeline/tests` passes offline: no network, no GPU, no torch. Anything
that needs torch lazy-imports it inside methods and fails with setup instructions, never
with an `ImportError` at import time.

## Vendoring TrackNetV3 (stage S4)

TrackNetV3 (MIT) is vendored research code, not a package dependency:

```sh
sh pipeline/scripts/vendor_tracknetv3.sh   # clones into pipeline/vendor/TrackNetV3 (git-ignored)
```

Pretrained checkpoints are **not** downloaded automatically — get `TrackNet_best.pt` and
`InpaintNet_best.pt` per the vendor README (`pipeline/vendor/TrackNetV3/README.md`), then:

```sh
export SYNCHRO_TRACKNET_WEIGHTS=/path/to/TrackNet_best.pt
# optional; defaults to a sibling InpaintNet_best.pt if present:
export SYNCHRO_TRACKNET_INPAINT_WEIGHTS=/path/to/InpaintNet_best.pt
```

The wrapper (`synchro_pipeline/perception/tracknet_v3.py`) shells out to the vendored
`predict.py` and maps its output onto the plan's `detected|inpainted|missing` visibility
contract: frames filled only by TrackNetV3's rectification (InpaintNet) pass become
`inpainted`.

## Shuttle benchmark

```sh
# plumbing check — runs end-to-end with no weights, GPU or footage:
uv run python -m synchro_pipeline.eval.benchmark_shuttle \
    --video match.mp4 --golden golden/match.json --tracker fake --out report.json

# the real thing (after vendoring + weights):
uv run python -m synchro_pipeline.eval.benchmark_shuttle \
    --video match.mp4 --golden golden/match.json --tracker tracknetv3 --out report.json
```

The report always contains per-rally and overall **coverage** (detected/inpainted/missing
split — the cheap early-warning signal on a new broadcaster). Where a golden rally
carries hand-labelled shuttle points, it also contains `shuttle_detection_f1` at 5 px
(plan target metric). Rallies without labels report `metrics: null` — never silently
folded into aggregates (plan: honest abstention).

## Hit-detection baseline benchmark

`perception/hits_baseline.py` is the plan's trajectory-only S5 baseline (risk #1
week-2 prototype): direction/speed-change events over a shuttle track, per-rally
adaptive thresholds, honest `saliency` (not a probability). Score it against golden
hit labels — `--tracker golden` replays the golden file's own shuttle points, so the
baseline logic is measurable before TrackNetV3 is even vendored:

```sh
uv run python -m synchro_pipeline.eval.benchmark_hits \
    --video match.mp4 --golden golden/match.json --tracker golden --out hits.json
```

## Analytics / tactical suite

`synchro_pipeline/analytics/` (adapted from BadmintonAnalyzer, Apache-2.0 — see
repo-root NOTICE): `adapter.py` turns `ShotRecord`/`RallyRecord` lists into DataFrames
with QA gating and drop reporting; `descriptive.py` (shot mix, serve patterns,
terminal conversion, pressure/momentum, head-to-head, trends), `tactical.py` (Markov
transition tables, n-gram pattern mining), `scouting.py` (template scouting reports).
Every aggregate row carries `n` + `low_sample`; unknowns become explicit buckets,
never guesses.

## Golden-set labeling workflow (Phase 0)

One JSON per match (`GoldenMatch` in `synchro_pipeline/eval/golden.py`): rally
boundaries, frame-exact hits, 16 court keypoints per labelled frame, score timeline,
optional per-rally shuttle tracks. Files are hand-editable; the loader rejects typos and
contradictions with messages naming the offending field/frames. The video's sha256 is
stamped/verified automatically by the tools and benchmark CLIs. Step-by-step:
`docs/golden-file-guide.md`.

0. **Mezzanine** — all frames index the canonical 720p30 CFR encode:

   ```sh
   uv run python -m synchro_pipeline.stages.s0_ingest source.webm mezzanine.mp4
   ```

1. **ShuttleSet shortcut** (when the match is in ShuttleSet): convert their per-stroke
   frame labels into golden hits from 2–4 manual anchors (`tools/align_shuttleset.py`;
   workflow in `docs/golden-set.md` §5.1), then refine boundaries + spot-check.

2. **Rally boundaries + hit frames** (S1/S5 ground truth — plan risk #1 says hit labels
   come first):

   ```sh
   uv run python pipeline/tools/label_rallies.py \
       --video mezzanine.mp4 --golden golden/match.json \
       --match-id 2024_ao_final --video-uri mezzanine.mp4 \
       --audio source.webm     # audio-onset proposals: o/O jump between candidates
   ```

   Keyboard-driven scrubber: `space` play/pause, `,`/`.` step, `[`/`]`/`{`/`}` jump,
   `o`/`O` next/prev audio-proposed onset, `r`/`e` rally start/end, `n`/`f` near/far
   hit, `s` save, `q` quit. Full key map in the module docstring. Onset proposals are
   jump targets only — nothing reaches the golden JSON without a keypress.

3. **Court corners** (S2 ground truth, PCK@5px) — pick a handful of frames per camera
   setup:

   ```sh
   uv run python pipeline/tools/label_court.py \
       --video match.mp4 --frame 1200 --golden golden/match.json
   ```

   Click the 4 doubles corners (near-left, near-right, far-right, far-left); the tool
   fits the homography, projects all 16 canonical keypoints back for verification,
   arrow keys nudge, `x` nulls an occluded point, `s` saves.

4. **Benchmark** against the file as shown above (`benchmark_shuttle`,
   `benchmark_hits`); wire the report numbers into CI (plan: "every model change
   answers 'did the numbers move'").

All labeling *logic* is in `synchro_pipeline/eval/labeling.py` (pure, unit-tested); the
tools are thin OpenCV event loops.

## License posture for perception assets (from docs/plan.md)

Hard rules: **no Ultralytics YOLOv8/11 (AGPL), no YOLOv7 (GPL), no GPL/AGPL dependencies
of any kind.** MonoTrack is blueprint-only (license unverified) — reimplement ideas,
never copy code.

| Asset | Role (stage) | License | Posture |
|---|---|---|---|
| FFmpeg / PyAV | S0 ingest | LGPL/BSD | ok |
| PySceneDetect | S1a shot splitting | BSD-3 | ok |
| TransNetV2 | S1a refine | MIT | ok |
| PaddleOCR | S1c score OCR | Apache-2.0 | ok |
| TennisCourtDetector | S2 architecture template | see repo | retrain on own labels |
| YOLOX / RT-DETR | S3 detection | Apache-2.0 | ok |
| ByteTrack | S3 tracking | MIT | ok |
| RTMPose | S3 pose | Apache-2.0 | ok |
| TrackNetV3 | S4 shuttle | MIT | vendored; **verify weight provenance before launch** |
| WASB-SBDT | S4 eval cross-check | MIT | ok |
| MonoTrack (HitNet) | S5 blueprint | unverified | **blueprint-only — no code reuse** |
| BST | S6 shot classification | official repo | retrain via our perception stack |
| CoachAI-Projects | S7 analytics models | MIT | ok |
| ShuttleSet / BFMD | training labels | dataset terms | author outreach week 1 (plan) |
| BadmintonAnalyzer | analytics suite origin | Apache-2.0 | code adapted (see NOTICE); its bundled YOLO weights are AGPL — **never copy them** |

Fallback if any asset turns out research-only: all architectures above are cleanly
licensed — retrain on our own labels (plan estimates 4–6 person-weeks, not a redesign).

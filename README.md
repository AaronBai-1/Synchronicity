# Synchronicity — Badminton Match Analytics

Offline analysis of broadcast badminton footage → stroke-level tactical insights for players and coaches: shuttle trajectory/velocity, shot classification, player positioning, serve/shot-selection by scenario, and cross-match player profiles.

**Core design principle:** the badminton scoring state machine + court homography are the trust anchors. Scoring rules cross-validate scoreboard OCR, rally segmentation, and player identity. Low-confidence rallies are excluded from aggregates and flagged — never silently included.

See [docs/plan.md](docs/plan.md) for the full project plan (pipeline stages, data model, roadmap, risks).

## Repository layout

```
pipeline/   Python CV/ML pipeline (synchro_pipeline) — stages S0–S7, schemas, eval harness, labeling tools
            └─ analytics/ — tactical/analytics suite adapted from BadmintonAnalyzer (Apache-2.0, see NOTICE)
api/        FastAPI service (synchro_api) — matches/rallies/shots API over Postgres (SQLite in dev)
web/        Next.js dashboard (scaffold; requires Node.js)
docs/       Plan, design notes (docs/design/), golden-set labeling spec, legal/license outreach drafts
NOTICE      Apache-2.0 attribution for code adapted from BadmintonAnalyzer
```

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg. Node.js ≥ 20 for `web/`.

```sh
uv sync --all-packages          # create .venv with pipeline + api (dev deps included)
make test                       # run all tests
make lint                       # ruff
```

ML extras (Torch, TrackNetV3 inference) are opt-in — see `pipeline/README.md`:

```sh
uv sync --all-packages --extra ml
pipeline/scripts/vendor_tracknetv3.sh   # fetch TrackNetV3 (MIT) into pipeline/vendor/
```

## Phase 0 status

- [x] Repo scaffold, schema contracts (ShuttleSet-compatible), scoring state machine, court geometry
- [x] Stage framework + S0 ingest (ffprobe/ffmpeg) + S1a shot splitting (PySceneDetect)
- [x] Eval metrics (rally F1, hit F1@±3f, PCK, reprojection error) + TrackNetV3 benchmark harness
- [x] Golden-set labeling tooling (court corners, rally boundaries, hit frames)
- [x] Analytics/tactical suite ported from predecessor (Apache-2.0, see NOTICE)
- [x] S5 trajectory-only hit baseline + benchmark
- [x] Labeling accelerators: ShuttleSet aligner, audio-onset hit proposals, S0 mezzanine CLI
- [ ] Golden set labeled (5 matches across broadcasters) — spec: [docs/golden-set.md](docs/golden-set.md), how-to: [docs/golden-file-guide.md](docs/golden-file-guide.md)
- [ ] TrackNetV3 pretrained benchmark on golden set + target GPU
- [ ] License outreach sent (drafts in [docs/legal/license-outreach-emails.md](docs/legal/license-outreach-emails.md))

**License hard rules:** no Ultralytics YOLOv8/11 (AGPL) and no YOLOv7 (GPL) anywhere in this codebase. Allowed model licenses: MIT / Apache-2.0 / BSD.

## Dev fixtures

`data/source/fixture-ausopen-prannoy-weng.mp4` — a 74s BWF broadcast clip used
as a private dev/test fixture (score OCR, court corners). It is git-ignored
(`data/*`, `*.mp4`) and must never be committed or redistributed; broadcast
footage stays private per the legal posture in [docs/plan.md](docs/plan.md).

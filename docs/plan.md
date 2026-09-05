# Badminton Analytics Platform — Project Plan

## Context

Greenfield project (the `Synchronicity` directory is empty — no code yet). Goal: a badminton analytics **product for players and coaches** ("SwingVision for badminton" — a verified market gap; no direct competitor exists as of Aug 2026). It ingests **broadcast/pro match footage** (user-uploaded), processes it **offline**, and produces stroke-level tactical insights: shuttle trajectory/velocity, shot classification, player positioning, serve/shot-selection by scenario, and cross-match player profiles.

Decisions confirmed with the user: broadcast footage input · offline batch processing · multi-user product · leverage existing open-source models/datasets heavily.

Research verified (Aug 2026) via parallel web-research and design passes.

## Core design principle

**The badminton scoring state machine + court homography are the trust anchors.** Scoring rules (rally point to 21, win-by-2, cap 30, server = last rally winner, side switches per game and at 11 in game 3) mutually verify three independent signals: scoreboard OCR, rally segmentation, and player identity. Every metric carries confidence; low-confidence rallies are **excluded from aggregates and flagged, never silently included**. Trust via honest abstention is the product's moat with coaches.

## Pipeline architecture (offline, staged cheap→expensive)

```
video → S0 ingest/normalize → S1 broadcast preprocessing → S2 court homography ─┐
                                                                                ├→ S3 players/pose ─┐
                                                                                ├→ S4 shuttle track ─┼→ S5 hit detection → S6 shot classification → S7 metrics/insights
```

| Stage | Approach | Key tool (license) |
|---|---|---|
| S0 Ingest | FFmpeg → canonical 720p30 CFR mezzanine; frame index is the time authority | FFmpeg/PyAV |
| S1a Shot splitting | PySceneDetect AdaptiveDetector; TransNetV2 refine for fades/wipes | BSD-3 / MIT |
| S1b Rally-camera gating | **No standalone classifier** — reuse S2 court net: frame is "game camera" iff court detection succeeds with sane geometry. Solves replay rejection + zoom + calibration with one model. ResNet-18 rally/non-rally vote (trained on BFMD) as second signal | custom |
| S1c Score OCR | Auto-localize static score bug (temporal variance), PaddleOCR digits, wrapped in the scoring state machine (≥99% after constraint decoding) | Apache-2.0 |
| S2 Court homography | **Main custom build.** Heatmap keypoint net (~16 court keypoints; retrain TennisCourtDetector architecture on 2–5k self-labeled frames) → per-frame RANSAC homography + temporal smoothing. Target: <5px reprojection at 720p | custom (template: yastrebksv/TennisCourtDetector) |
| S3 Players | YOLOX or RT-DETR (Apache) → court-polygon filter → ByteTrack (MIT) → RTMPose top-down on crops (handles small far-court player) → near/far by court half + side-switch schedule from state machine | all Apache/MIT |
| S4 Shuttle | **TrackNetV3** (MIT, pretrained, 97.5% acc on broadcast badminton, ~25 FPS) on rally frames only; carry per-frame `visibility: detected|inpainted|missing` downstream; WASB-SBDT (MIT) as eval cross-check; TensorRT INT8 fork for cost | MIT |
| S5 Hit detection | **The linchpin.** Reimplement MonoTrack's HitNet formulation (GRU over shuttle track + both poses, 3-class no-hit/near-hit/far-hit) fused with pose-derived swing cues (Sensors-2024 recipe: 90.5% F1 vs 72.3% trajectory-only). Train labels free from ShuttleSet frame_num + BFMD hit events. Alternation prior + min-gap postprocessing | custom (blueprint: MonoTrack paper) |
| S6 Shot classification | BST-style transformer (pose + shuttle + positions — exactly what S2–S5 emit), retrained on ShuttleSet clips **regenerated through our own perception stack**. ≥80% at merged classes | official BST repo, retrain |
| S7 Metrics | Positions/placement via homography; velocity as 2D court-plane speed **bands labeled "estimated"** (defer 3D lift); scenario/serve analytics as SQL over the shot table; ShuttleNet-style embeddings = player profiles (later phase) | CoachAI-Projects (MIT) |

**License hard rules:** no Ultralytics YOLOv8/11 (AGPL), no YOLOv7 (GPL). Verify before launch: TrackNetV3 weight provenance, ShuttleSet/BFMD annotation commercial terms, MonoTrack license (blueprint-only until confirmed). Fallback if research-only: all architectures are cleanly licensed — retrain on own labels (~4–6 person-weeks, not a redesign).

## Shot taxonomy (three layers, one source of truth)

1. **Canonical storage: ShuttleSet 18-type** — shared by ShuttleSet/ShuttleSet22/CoachAI (~70k labeled strokes); every reusable downstream model consumes it
2. **Classifier output: merged ~11–12 classes** (wrist smash→smash etc.) — 80%+ accuracy only exists merged (BST: 82.5% merged vs ~63% fine-grained)
3. **UI: 8-class coach vocabulary** (serve short/long, clear, drop, smash, drive, net, lift, push/rush)

Store the full ShuttleSet-compatible per-stroke schema (incl. `landing_height`, `backhand`, `aroundhead`, `lose_reason`) even where v1 UI ignores it — public datasets and CoachAI models drop in without migration. Court coords: origin at center, x∈[−3.05, 3.05], y∈[−6.7, 6.7] m, plus player-relative coords so side switches vanish from tendency analytics.

## Platform

- **No self-managed GPU fleet**: each pipeline stage = a Modal serverless function (queue/retries/scale-to-zero built in); RunPod as fallback. Rally-gating before heavy models cuts 60–70% of frames (broadcasts ≈ 30–40% live play) → **$0.50–1.50 GPU cost per match-hour** on T4/L4
- **Stack**: Next.js + Tailwind/shadcn + visx (one shared SVG court component) + hls.js · FastAPI + Pydantic v2 (shared schema models with the pipeline) · Postgres (all analytics; trajectories stay in S3 Parquet) · Cloudflare R2 (zero egress on video) · Clerk auth with workspace/org model
- **Data model**: `players / matches / games / rallies / shots` in Postgres; per-shot contract mirrors CoachAI Challenge Track-1 spec; every match gets a `confidence_report`
- **Dashboard (build order)**: ① Match report (score worm, headline tiles, shot distribution, rally table — every stat click-through to video clips; the stat→clip loop is the product) ② Rally player with synced video + animated top-down court ③ Serve & scenario views ④ Positioning heatmaps ⑤ Cross-match profiles / opponent scouting
- **Legal posture (designed in, not bolted on)**: uploads private-per-workspace, never curated/featured/redistributed; shareable outputs are derived data only (uncopyrightable facts); registered DMCA agent + takedown + repeat-infringer policy; self-recorded footage accepted as first-class clean-path input; counsel review (~$3–5k) before launch; BWF scouting-rights license (Wyscout template) if traction

## Roadmap

- **Phase 0 (3–4 wks): Walking skeleton + golden eval set.** Hand-label 5 full matches across broadcasters (rally boundaries, hit frames, court corners, scores). Upload → Modal → rally clips end-to-end with zero ML novelty. Benchmark TrackNetV3 pretrained on golden set + L4. **Email dataset/weight authors for commercial-use confirmation in week 1.**
- **Phase 1 (8–10 wks): "Match Report Lite" MVP** — deliberately **no hit detection or shot classification yet**: court net + rally segmentation + score OCR/state machine + player tracking → movement heatmaps, coverage, rally-length analysis, points timeline, rally clip player. First coach-trusted insight ("lost 7 of 9 rallies longer than 15 shots; coverage collapsed to backhand rear corner").
- **Phase 2 (8–12 wks): Shuttle + hit detection** — the riskiest layer gets its own phase. Shot counts, tempo, speed bands, confidence system.
- **Phase 3 (6–8 wks): Shot classification + placement analytics.**
- **Phase 4 (8–10 wks): Scenario/serve analytics + player profiles + team workspaces** — the differentiator nothing commercial offers.
- **Phase 5+ (non-promises)**: doubles (singles-only datasets make it structural), 3D velocity lift, audio cues, self-recorded domain tuning.

~3 months to paying-coach-worthy MVP; ~9–11 months to full v1 (2 engineers). Pricing target $15–25/user/mo in the verified gap between technique apps and Hudl-class manual tools.

## Top risks (each with early de-risk action)

1. **Hit-detection cascade** (every shot-level stat depends on it) → golden set with frame-exact hit labels first; prototype trajectory-only baseline in week 2; per-rally confidence + abstention architecture from day one
2. **Shuttle tracking on non-BWF broadcast styles** → eval pretrained weights across 3+ broadcasters in first two weeks; fine-tune on worst performers; upload-quality gating
3. **Dataset/weight licenses** → author outreach week 1; fallback = retrain clean architectures on own labels
4. **Broadcast-footage copyright** → private-per-workspace + DMCA hygiene + derived-data-only sharing; counsel review pre-launch
5. **Court net is the custom-build gating everything** → validate at 1k labels before scaling to 5k
6. **GPU cost** → rally-frames-only ordering; INT8; kill criterion >$4/match
7. **Velocity credibility** → bands + "estimated" labels; never print a km/h we can't defend

## Non-goals for v1

Doubles · real-time · precise km/h claims · 18-class labels in UI · technique/form analysis · auto player naming (one-click manual assignment instead) · public sharing/highlight reels · RL strategy simulation · unvetted 2025-26 releases (TrackNetV5, CourtKeyNet) as production dependencies.

## Verification approach

- **Golden eval set as CI**: per-stage metrics (rally F1, homography reprojection error, shuttle F1@5px, hit F1@±3 frames, merged-class accuracy) wired into the repo; every model change answers "did the numbers move"
- **State-machine self-checks in production**: score deltas vs rally boundaries vs serve sides — inconsistencies become `qa_flags`, visible per match
- **Stage targets**: rally seg ≥97% · OCR score-tuple ≥99% post-constraint · homography <5px/720p · identity ≥98% · hit F1 ≥0.90 · merged shot classes ≥80%, coarse ≥90%, serve short/long ≥95%
- End-to-end: upload a known match, diff extracted shot table against ShuttleSet's human annotations for that match

## Predecessor integration (BadmintonAnalyzer)

Audit of the Apache-2.0 predecessor ("Synchronicity Badminton Analytics", Dhyey Mavani; local reference copy at `BadmintonAnalyzer-main/`, git-ignored, never imported or executed).

**Adopted (with attribution headers; see NOTICE):**
- Analytics/tactical suite ported into `pipeline/src/synchro_pipeline/analytics/`, reworked onto our ShotRecord contract with the trust architecture (every aggregate carries `n`, low-sample flagging, no fabricated values) fixed in during the port
- Review-queue pattern (draft → review flag → enum-constrained editable grid → validated import) adopted as the label-flywheel design — [docs/design/review-queue.md](design/review-queue.md)
- Trajectory-only hit-detection heuristics as concept-level input to the S5 baseline skeleton (`perception/hits_baseline.py`) + benchmark — the week-2 baseline from risk #1
- Their BWF broadcast test clip retained as a private git-ignored dev fixture (`data/source/fixture-ausopen-prannoy-weng.mp4`)

**Rejected:**
- The CV layer (court/pose/shot geometry): defective geometry and confidence handling (constant-fallback confidence, speed proxies, fabricated outcomes) — reimplementing per this plan instead
- All bundled model weights, including the YOLO11n shuttle fine-tune: Ultralytics AGPL — violates the license hard rules
- Their 11-type shot taxonomy: incompatible with the ShuttleSet-18 canonical layer; our three-layer taxonomy stands

**Lead to verify:** the Roboflow "Shuttlecock" dataset their shuttle fine-tune trained on — license unverified; do not use until confirmed.


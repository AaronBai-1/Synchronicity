# Golden Eval Set — Phase 0 Labeling Spec

This document is the working instruction sheet for hand-labeling the five golden
matches (docs/plan.md, Phase 0: "Walking skeleton + golden eval set"). The golden set
is the project's CI: every per-stage metric (rally F1, homography reprojection error,
shuttle F1, hit F1@±3 frames, shot-class accuracy) is computed against these labels,
and every model change must answer "did the numbers move".

Budget: **~2–3 person-days per match, ~10–15 person-days total** (plan estimate).
Label quality beats label quantity — a wrong hit frame poisons the hit-detection
metric silently; an honest "uncertain" flag does not.

> Step-by-step mechanics (file locations, mezzanine command, the JSON field by field,
> commit checklist): see **[golden-file-guide.md](golden-file-guide.md)**.

## 1. Sourcing the five matches

You source the footage yourself. Legal posture (from docs/plan.md): footage stays
**private to this workspace**, is never curated, featured, or redistributed, and the
repository only ever contains **derived annotation data** (frame numbers, coordinates,
scores — uncopyrightable facts) plus a content hash to identify your local copy.
Do not commit video files or links that republish them.

Pick five singles matches that deliberately spread the broadcast conditions the
pipeline must survive (plan risk #2: shuttle tracking across broadcast styles):

| # | Criterion | Why |
|---|---|---|
| 1 | BWF World Tour broadcast (recent, HD) | The "easy" reference condition; matches what public models were trained on |
| 2 | National-federation feed (e.g. Badminton Europe / national association stream) | Different camera height, graphics package, colour grading |
| 3 | YouTube re-encode of visibly different quality (compression artifacts, lower bitrate) | Upload reality: users bring re-encodes, not masters |
| 4 | Older-era match (roughly pre-2015, lower resolution / interlacing era) | Stress-tests court detection and shuttle visibility assumptions |
| 5 | Non-BWF arena (club/university hall, unusual floor colour or lighting) | The self-recorded clean-path input the product promises to accept |

Prefer matches that also appear in ShuttleSet if possible — the plan's end-to-end
check diffs our extracted shot table against ShuttleSet's human annotations.

The content hash of the video is handled automatically: the labeling tools stamp
`video_sha256` (sha256 of the mezzanine you label against) into the golden JSON on
first use and refuse to open the file against a different encode (`--allow-hash-mismatch`
overrides and re-stamps); the benchmark CLI verifies it before scoring (exit code 4 on
mismatch, `--skip-hash-check` to override). Record manually, before labeling: the
original source description/filename (`source_description` field), players, event,
discipline (`MS`/`WS`).

## 2. Canonical timebase

All frame numbers refer to the **S0 mezzanine** (720p30 CFR — plan stage S0: "frame
index is the time authority"), not the original download. Run the match through S0
ingest first and label on its output. If S0 is not runnable yet, label on a local
constant-frame-rate 30 fps re-encode and record the exact ffmpeg command with the
labels so the mapping is reproducible.

## 3. What to label per match

### 3.1 Rally boundaries (every rally)

- `start_frame`: first frame of the server's forward service motion (racquet starts
  moving toward the shuttle). Tolerance ±5 frames — rally segmentation is evaluated
  with slack; the *serve hit frame* below is the frame-exact anchor.
- `end_frame`: frame where the rally is decided — shuttle touches the floor, hits the
  net cord and falls, or the umpire's call is evident. When the broadcast cuts away
  before the shuttle lands, use the last frame where the outcome is visible and add
  flag `end_occluded`.
- Include lets/replayed points as rallies with flag `let` and no score change.
- Do NOT include broadcast replays — a replayed rally is labeled once, at its live
  occurrence (this is exactly what the S1b rally-camera gate must learn to reject).

### 3.2 Hit frames (every stroke of every rally — the linchpin labels, plan risk #1)

- Frame-exact racquet–shuttle contact. At 30 fps contact often falls between frames:
  label the **first frame at or after contact** (the first frame where the shuttle's
  direction has visibly changed). Consistency matters more than the convention itself.
- Per hit: `frame`, `side` — `near` | `far` — which court half the striking player
  occupies, using the pipeline convention (domain/court.py: near = closest to the
  broadcast camera, court y < 0).
- The serve is hit #1 of each rally. Hits must alternate near/far within a rally.
- If contact is occluded (player body, net post, graphics), give your best frame and
  add flag `hit_occluded` — flagged hits are excluded from the frame-exact F1 and
  scored at ±3 frames only.

### 3.3 Court corners / keypoints (~20 sampled frames per match)

- Sample ~20 game-camera frames **spread across zoom and camera variations**: both
  ends of each game, tight vs wide framings, before and after side switches, and any
  segment where the broadcast visibly changes framing. Do not sample 20 near-identical
  frames — the point is to bracket the homography net's operating range (plan S2
  target: <5 px reprojection at 720p).
- Per frame: pixel coordinates (mezzanine 720p) of the four doubles corners at
  minimum, and every other visible keypoint of the 16 defined in
  `synchro_pipeline.domain.court.COURT_KEYPOINT_NAMES` (singles×baseline,
  short-service×sideline, centre-line T points, baseline centres). Mark occluded or
  out-of-frame keypoints as absent — never guess.
- Click the **line-intersection centre** (lines are 40 mm wide; aim for the middle of
  the painted line), matching the keypoint convention in domain/court.py.

### 3.4 Score timeline (every rally end)

- At each rally end: the score shown by the broadcast scorebug **after** the rally
  (`a`, `b` in a fixed player mapping you choose once per match and keep for all
  games), plus the game number.
- Record the first server of the match and which physical side player A starts on —
  the scoring state machine needs both to derive serve courts and the side-switch
  schedule.
- If the scorebug is absent/wrong at a rally end (graphics glitch), infer from
  context and flag `score_inferred`.

## 4. File format

One JSON file per match: `data/golden/<match_id>.json` (directory is gitignored
except for the JSON labels — labels are derived data and safe to commit).

The schema source of truth is **`pipeline/src/synchro_pipeline/eval/golden.py`**
(Pydantic models; load/validate labels through it, never by hand).

> NOTE (2026-08-02): `eval/golden.py` did not exist when this spec was written — it
> is being scaffolded in parallel. The intended structure is below; if the module
> diverges, **the module wins** — update this section to match it, not vice versa.

Intended `GoldenMatch` structure:

```jsonc
{
  "match_id": "2024_worldtour_xyz",
  "video_sha256": "…",
  "source_notes": "BWF World Tour 2024, 720p30 mezzanine from 1080p60 source",
  "discipline": "MS",
  "first_server": "A",
  "a_on_near_side_at_start": true,
  "rallies": [
    {
      "rally_id": "g1_r001",
      "set_no": 1,
      "start_frame": 1502,
      "end_frame": 1893,
      "hits": [
        { "frame": 1531, "side": "near" },
        { "frame": 1568, "side": "far", "flags": ["hit_occluded"] }
      ],
      "score_after": { "a": 1, "b": 0 },
      "winner": "A",
      "flags": []
    }
  ],
  "court_frames": [
    {
      "frame": 1520,
      "keypoints": {
        "near_corner_left": { "x": 213.5, "y": 641.0 },
        "far_t_point": null
      }
    }
  ]
}
```

## 5. Tooling

Labeling helpers live in `pipeline/tools/` (OpenCV-window based, offline):

```bash
cd /path/to/Synchronicity
uv run python pipeline/tools/label_court.py   data/mezzanine/<match>.mp4 data/golden/<match_id>.json
uv run python pipeline/tools/label_rallies.py data/mezzanine/<match>.mp4 data/golden/<match_id>.json
```

- `label_court.py` — steps through sampled frames; click the 16 keypoints in
  contract order, skip key for occluded ones.
- `label_rallies.py` — scrub/step the video; mark rally start/end, hit frames with
  near/far, and the score at each rally end.

> Same caveat as §4: the tools are being built in parallel with this spec. If flags
> or filenames differ when you run them, trust `--help` and update this section.

## 6. Acceptance checklist (run before a match counts as "golden")

The whole point of the golden set is that it is *internally verified* by the scoring
state machine (`synchro_pipeline.domain.scoring`) — the same trust anchor the
production pipeline uses (plan: score deltas × rally boundaries × serve sides
mutually verify).

- [ ] **Score timeline replays.** Starting from `MatchState` with the recorded first
      server and sides, applying each rally's winner via `apply_rally` reproduces
      every recorded `score_after`, every game's final score is terminal per
      `game_winner`, and every rally-to-rally delta passes `is_valid_transition`.
      Lets (`let` flag) must produce no score change.
- [ ] **Serve sides consistent.** For each rally, the labeled `side` of hit #1
      matches the state machine's derived server and the side-switch schedule
      (`MatchState.side_of`), including the deciding-game switch at 11.
- [ ] **Hit sanity.** Within each rally: frames strictly increasing, sides strictly
      alternating, hit #1 ≥ `start_frame`, last hit ≤ `end_frame`, minimum gap
      between consecutive hits ≥ 4 frames (at 30 fps).
- [ ] **Rally sanity.** Rallies non-overlapping, strictly ordered, every non-let
      rally has a winner and a score change of exactly one point.
- [ ] **Court frames verify.** For every court frame with ≥ 4 labeled keypoints,
      `synchro_pipeline.domain.court.fit_homography` succeeds and RMS reprojection
      error (`reprojection_error_px`) is < 5 px, and `homography_is_sane` passes.
- [ ] **Coverage.** ≥ 20 court frames spanning all games and the match's zoom range;
      100% of rallies have boundaries + score; 100% of hits labeled (occluded ones
      flagged, not skipped).
- [ ] **File validates** against the Pydantic schema in
      `pipeline/src/synchro_pipeline/eval/golden.py`.

A match that fails a checklist item is not "mostly done" — unverified labels are
worse than missing labels, because CI will trust them.

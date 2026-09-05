# Design note: Camera changes (zoom, pan, cuts, end flips)

Status: design (pins decisions for S1b/S2/S3 implementation)
Stubs referenced: `pipeline/src/synchro_pipeline/stages/s1b_rally_gate.py`,
`stages/s2_court.py`, `stages/s3_players.py`;
contract: `schemas/records.py` (`RallyRecord.video_shot_ids`).

## Baseline: per-frame court detection absorbs continuous camera motion

S2 runs the court keypoint net **per frame** (docs/plan.md "S2 Court
homography"), so zooms and pans within a game-camera shot are not special
cases — the homography simply changes frame to frame. No global camera model
is assumed.

## Rally-camera gate rejects non-game cameras

S1b's gate (see `s1b_rally_gate.py`): a frame is "game camera" iff court
detection succeeds with sane geometry (`domain.court.homography_is_sane`),
with a ResNet-18 rally/non-rally vote as second signal. Replays, crowd shots,
close-ups, and interviews fail the court test and are dropped before the
heavy S3–S6 stages.

## The END-FLIP hazard

A badminton court is symmetric under 180° rotation. A broadcast cut to a
camera at the **opposite end** of the arena produces a perfectly valid court
detection and a sane homography — with near and far ends silently swapped.
Every downstream near/far assignment (player identity, serve side, shot
direction, player-relative coords) would be inverted, and nothing in the
geometry alone can notice.

Three independent defenses, any one of which flags the flip:

1. **Scoring-state serve-side check.** The scoring state machine
   (`domain/scoring.py`) knows who serves and from which service court for
   every score. If the observed server stands at the end the state machine
   says is wrong, either the homography orientation flipped or identity did —
   raise a qa flag and try the 180°-rotated homography.
2. **Appearance continuity across rally boundaries.** S3 player tracks carry
   appearance (jersey color histogram / embedding). Between consecutive
   rallies, near-player appearance should persist except at scheduled side
   switches (known from the state machine). An unscheduled swap in appearance
   near↔far marks a probable end flip.
3. **Arena-background consistency.** The regions *outside* the court polygon
   (banners, umpire chair, backdrop) are not symmetric. A cheap per-shot
   background descriptor compared against the match's dominant game camera
   distinguishes the two ends even when the court itself cannot.

Disagreement among the defenses is never resolved by guessing: the affected
rally gets a qa flag and drops out of side-sensitive aggregates.

## Homography smoothing resets at video-shot boundaries

S2's temporal smoothing must **not** smooth across a cut — blending
homographies from two different cameras yields a transform valid for neither.
`RallyRecord.video_shot_ids` records which S1a shot segments a rally spans;
the smoother resets its state at every segment boundary, and any rally
spanning multiple video shots re-runs the end-flip defenses per segment.

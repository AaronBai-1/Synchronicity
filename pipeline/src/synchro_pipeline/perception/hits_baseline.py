# Portions adapted from BadmintonAnalyzer (Apache-2.0), Copyright Dhyey Mavani.
"""Trajectory-only hit-detection baseline (docs/plan.md risk #1, plan stage S5).

Plan risk #1 calls for a week-2 trajectory-only hit-detection prototype so the gap to
the fused S5 approach can be measured on the golden set (literature cited in plan S5:
~72.3% F1 trajectory-only vs 90.5% fused). This module is that prototype: it proposes
hit events from the shuttle track alone, working in IMAGE pixel space — the shuttle is
airborne, so projecting it through the court-plane homography would be meaningless.

The event-skeleton idea (velocity-change spikes along the shuttle track, an adaptive
threshold, a minimum gap between events) is adapted from BadmintonAnalyzer's
cv/shots.py, with that code's audited defects deliberately fixed, not replicated:

    * the adaptive threshold is computed PER INPUT SEQUENCE (one rally), never over a
      whole video — a flat-drive rally and a lift-heavy rally have different motion
      statistics;
    * the event signal is scale-free (direction-change angle + relative speed change),
      not pixel speeds run through a wrong metres-per-pixel constant;
    * `saliency` is documented as a heuristic strength score, never dressed up as a
      probability (the predecessor reported ``min(speed / 25, 1)`` as a confidence);
    * no event is ever forced at the first sample; an empty proposal list is a
      legitimate, common output;
    * abstention over guessing (plan trust architecture): too few usable points means
      no proposals at all, and an ambiguous side seed leaves side=None on every hit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from synchro_pipeline.perception.base import ShuttleTrackPoint
from synchro_pipeline.schemas.records import Side


class ProposedHit(BaseModel):
    """A candidate racket-shuttle contact proposed from the shuttle trajectory alone.

    `saliency` is a heuristic event-strength score — the down-weighted sum of the
    direction-change angle (normalized by pi) and the relative speed change at the
    proposed frame, so it lives in [0, 2]. It is NOT a calibrated probability and must
    never be presented, thresholded, or averaged as one; it is only meaningful for
    ranking proposals within the sequence that produced it. (The predecessor shipped
    ``speed / 25`` as "Detection_Confidence" — this field's name and this paragraph
    exist so that mistake is not repeated.)

    `side` is which physical end struck the shuttle ("near" | "far"), or None when the
    evidence was too ambiguous to say (see assign_sides — unknown stays None, never a
    coin flip).
    """

    model_config = ConfigDict(extra="forbid")

    frame: int = Field(ge=0)
    side: Side | None = None
    saliency: float = Field(ge=0.0)


def propose_hits(
    points: Sequence[ShuttleTrackPoint],
    fps: float,
    *,
    min_gap_ms: float = 250.0,
    k_sigma: float = 1.6,
    min_points: int = 8,
    min_saliency: float = 0.55,
    min_speed_px: float = 3.0,
    accel_snr: float = 4.0,
    max_gap_frames: int = 5,
    inpaint_weight: float = 0.6,
) -> list[ProposedHit]:
    """Propose hit frames from one rally's shuttle track (image pixel space).

    Core signal, computed at every interior usable point (detected or inpainted;
    "missing" points are placeholders and are ignored entirely):

        raw      = direction_change_angle / pi  +  |Δspeed| / max(speed_in, speed_out)
        saliency = min(weight of the three supporting points) * raw

    where velocities are finite differences between consecutive usable points,
    normalized by their frame gap (gaps in frame_idx are respected; a gap larger than
    `max_gap_frames` breaks the chain — no velocity is asserted across it). Inpainted
    points are down-weighted by `inpaint_weight`; both terms of `raw` are ratios, so
    the signal carries no resolution- or fps-dependent constants.

    A point is proposed as a hit when ALL of the following hold:

        * its supporting speeds reach `min_speed_px` px/frame (at the plan-S0 canonical
          720p mezzanine, slower motion is tracking jitter, not shuttle flight — the
          angle between near-zero velocity vectors is noise);
        * its velocity change |v_out - v_in| exceeds `accel_snr` times the sequence's
          median velocity change. The median over a rally is dominated by quiet
          in-flight samples, so it estimates THIS track's noise scale; a real hit
          reverses a real velocity and clears it by an order of magnitude, while pure
          jitter — however wild its angles — never stands far above its own median;
        * `raw >= min_saliency` — an absolute geometry floor. Required because the
          adaptive threshold below is relative: in ANY nondegenerate signal some sample
          exceeds mean + k*sigma, so without a floor a rally of pure drift would still
          "detect" its largest wobble;
        * `saliency > mean + k_sigma * std` of the saliency series of THIS input
          sequence — the adaptive threshold is per rally, never global.

    Surviving candidates then go through local-max suppression: greedily keep the most
    salient, dropping any candidate within `min_gap_ms` of an already-kept one (two
    real hits cannot be closer than a human swing cycle). No event is ever forced at
    the first sample.

    Fewer than `min_points` usable points (or none evaluable after gap-breaking) is too
    little evidence to estimate a per-rally threshold: the function abstains and
    returns [] (plan trust architecture: no output beats a fabricated one).
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if min_points < 3:
        raise ValueError("min_points must be >= 3 (an event needs two velocities)")

    usable = sorted(
        (p for p in points if p.visibility in ("detected", "inpainted")),
        key=lambda p: p.frame_idx,
    )
    n = len(usable)
    if n < min_points:
        return []

    frames = np.array([p.frame_idx for p in usable], dtype=np.int64)
    xy = np.array([(p.x, p.y) for p in usable], dtype=np.float64)
    weight = np.array(
        [1.0 if p.visibility == "detected" else inpaint_weight for p in usable],
        dtype=np.float64,
    )

    dt = np.diff(frames).astype(np.float64)  # >= 1 (frame_idx strictly increases)
    vel = np.diff(xy, axis=0) / dt[:, None]  # px/frame
    speed = np.hypot(vel[:, 0], vel[:, 1])
    vel_ok = dt <= max_gap_frames  # no velocity claim across long missing gaps

    raw = np.zeros(n, dtype=np.float64)
    saliency = np.zeros(n, dtype=np.float64)
    dv = np.zeros(n, dtype=np.float64)  # |v_out - v_in|, px/frame
    evaluable = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if not (vel_ok[i - 1] and vel_ok[i]):
            continue
        evaluable[i] = True
        dv[i] = float(np.hypot(*(vel[i] - vel[i - 1])))
        s_in, s_out = float(speed[i - 1]), float(speed[i])
        if min(s_in, s_out) < min_speed_px:
            continue  # jitter regime: signal stays 0 but the sample stays evaluable
        cos_angle = float(np.dot(vel[i - 1], vel[i])) / (s_in * s_out)
        theta = math.acos(max(-1.0, min(1.0, cos_angle)))
        rel_dspeed = abs(s_out - s_in) / max(s_in, s_out)
        raw[i] = theta / math.pi + rel_dspeed
        saliency[i] = min(weight[i - 1], weight[i], weight[i + 1]) * raw[i]

    signal = saliency[evaluable]
    if signal.size == 0:
        return []
    threshold = float(signal.mean() + k_sigma * signal.std())
    # Per-rally noise scale: quiet in-flight samples dominate the median. Strict `>`
    # matters for the noiseless-track edge case (scale 0: any real kink passes, a
    # perfectly straight track proposes nothing).
    dv_floor = accel_snr * float(np.median(dv[evaluable]))

    candidates = [
        i
        for i in np.nonzero(evaluable)[0]
        if raw[i] >= min_saliency and dv[i] > dv_floor and saliency[i] > threshold
    ]

    # Local-max suppression: strongest first, deterministic tie-break on frame.
    gap_frames = (min_gap_ms / 1000.0) * fps
    kept: list[int] = []
    for i in sorted(candidates, key=lambda i: (-saliency[i], frames[i])):
        if all(abs(float(frames[i] - frames[j])) > gap_frames for j in kept):
            kept.append(i)
    kept.sort(key=lambda i: frames[i])
    return [
        ProposedHit(frame=int(frames[i]), side=None, saliency=float(saliency[i])) for i in kept
    ]


def _outgoing_vertical_velocity(
    hit_frame: int,
    usable: Sequence[ShuttleTrackPoint],
    lookback_frames: int,
    lookahead_frames: int,
) -> float | None:
    """vy (px/frame, image axes: +y is down) just after `hit_frame`, or None.

    Uses the last usable point at/before the hit (the contact vertex) and the first
    usable point after it; either missing within its window means no vote.
    """
    before = [p for p in usable if 0 <= hit_frame - p.frame_idx <= lookback_frames]
    after = [p for p in usable if 0 < p.frame_idx - hit_frame <= lookahead_frames]
    if not before or not after:
        return None
    p0 = max(before, key=lambda p: p.frame_idx)
    p1 = min(after, key=lambda p: p.frame_idx)
    return (p1.y - p0.y) / float(p1.frame_idx - p0.frame_idx)


def assign_sides(
    hits: Sequence[ProposedHit],
    points: Sequence[ShuttleTrackPoint],
    *,
    lookback_frames: int = 3,
    lookahead_frames: int = 8,
) -> list[ProposedHit]:
    """Assign near/far sides to proposed hits by seeded alternation.

    Broadcast framing puts the far court at the top of the image, so a shuttle moving
    up-frame (y decreasing) just after a hit was struck by the NEAR player, and one
    moving down-frame by the FAR player. Each hit votes for a side via its outgoing
    vertical velocity, with |vy| as the vote's strength; hits with no usable points
    around them (or vy == 0) cast no vote.

    Within a rally players hit in strict turn, so there are only two possible
    labelings (near-first or far-first alternation). Both are scored against the votes
    (+strength for agreement, -strength for disagreement) and the better one is kept —
    a single bad vote cannot break the chain as long as the good votes outweigh it.

    If the two labelings score EQUALLY — no informative votes at all, or perfectly
    conflicting ones — the seed is ambiguous and every hit is returned with side=None
    rather than a 50/50 guess (plan trust architecture: unknown stays None; downstream
    consumers must handle side=None, not assume it). Sides are all-or-nothing by
    design: a parity chain with holes guessed in the middle would be worse than an
    honest abstention.

    Returns new ProposedHit objects; the inputs are not mutated.
    """
    if not hits:
        return []
    usable = sorted(
        (p for p in points if p.visibility in ("detected", "inpainted")),
        key=lambda p: p.frame_idx,
    )
    ordered = sorted(hits, key=lambda h: h.frame)

    votes: list[tuple[Side | None, float]] = []
    for hit in ordered:
        vy = _outgoing_vertical_velocity(hit.frame, usable, lookback_frames, lookahead_frames)
        if vy is None or vy == 0.0:
            votes.append((None, 0.0))
        elif vy < 0.0:  # up-frame -> toward the far court -> struck by the near player
            votes.append(("near", -vy))
        else:
            votes.append(("far", vy))

    def score(first: Side) -> float:
        total = 0.0
        for i, (vote, strength) in enumerate(votes):
            if vote is None:
                continue
            expected: Side = first if i % 2 == 0 else ("far" if first == "near" else "near")
            total += strength if vote == expected else -strength
        return total

    near_first, far_first = score("near"), score("far")
    if near_first == far_first:  # ambiguous seed: abstain on every hit, never guess
        return [h.model_copy(update={"side": None}) for h in ordered]
    first: Side = "near" if near_first > far_first else "far"
    other: Side = "far" if first == "near" else "near"
    return [
        h.model_copy(update={"side": first if i % 2 == 0 else other})
        for i, h in enumerate(ordered)
    ]

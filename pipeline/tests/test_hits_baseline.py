"""Trajectory-only hit-detection baseline tests (perception/hits_baseline.py).

Synthetic piecewise-linear "shuttle" paths with direction reversals at known frames
(plus gaussian pixel noise) stand in for real tracks. The tests pin the behaviours the
plan's trust architecture demands and the predecessor's audited defects it must not
replicate: per-sequence adaptive threshold, no forced first event, abstention on thin
or ambiguous evidence, inpainted evidence down-weighted, saliency never a probability.
"""

import numpy as np
import pytest
from pydantic import ValidationError
from synchro_pipeline.eval.metrics import event_f1
from synchro_pipeline.perception.base import ShuttleTrackPoint
from synchro_pipeline.perception.hits_baseline import ProposedHit, assign_sides, propose_hits

FPS = 30.0


def make_track(
    segments,
    *,
    start=(400.0, 600.0),
    start_frame=0,
    noise=0.0,
    seed=0,
    inpainted=frozenset(),
    missing=frozenset(),
):
    """Piecewise-linear track from (n_frames, vx, vy) segments.

    Returns (points, boundary_frames) where boundary_frames are the segment joins —
    the frames where the synthetic "shuttle" changes direction, i.e. the ground-truth
    hit frames.
    """
    rng = np.random.default_rng(seed)
    x, y = start
    frame = start_frame
    ideal = [(frame, x, y)]
    boundaries = []
    for n_frames, vx, vy in segments:
        for _ in range(n_frames):
            frame += 1
            x += vx
            y += vy
            ideal.append((frame, x, y))
        boundaries.append(frame)
    boundaries = boundaries[:-1]  # the last join is the track end, not a reversal

    points = []
    for f, ix, iy in ideal:
        if f in missing:
            points.append(
                ShuttleTrackPoint(frame_idx=f, x=0.0, y=0.0, conf=0.0, visibility="missing")
            )
            continue
        nx, ny = rng.normal(0.0, noise, 2) if noise else (0.0, 0.0)
        vis = "inpainted" if f in inpainted else "detected"
        points.append(
            ShuttleTrackPoint(
                frame_idx=f,
                x=ix + nx,
                y=iy + ny,
                conf=0.4 if vis == "inpainted" else 0.9,
                visibility=vis,
            )
        )
    return points, boundaries


class TestProposedHitModel:
    def test_defaults_and_fields(self):
        hit = ProposedHit(frame=5, saliency=1.2)
        assert hit.side is None
        assert hit.frame == 5

    def test_rejects_bad_values(self):
        with pytest.raises(ValidationError):
            ProposedHit(frame=-1, saliency=0.5)
        with pytest.raises(ValidationError):
            ProposedHit(frame=1, saliency=-0.1)
        with pytest.raises(ValidationError):
            ProposedHit(frame=1, saliency=0.5, side="left")


class TestProposeHits:
    def test_clean_reversals_scores_high_f1(self):
        points, gt = make_track(
            [(30, 2, -12), (30, -1.5, 12.5), (30, 2.5, -12), (30, -2, 12), (30, 1.5, -12.5),
             (30, -2, 12)],
            noise=1.0,
            seed=1,
        )
        hits = propose_hits(points, FPS)
        result = event_f1([h.frame for h in hits], gt, tolerance_frames=3)
        assert result.f1 >= 0.9, (result.as_dict(), [h.frame for h in hits], gt)

    def test_no_forced_first_event(self):
        # Predecessor defect: index 0 was unconditionally emitted as a hit.
        points, gt = make_track(
            [(30, 2, -12), (30, -1.5, 12.5), (30, 2.5, -12), (30, -2, 12)],
            noise=1.0,
            seed=1,
        )
        hits = propose_hits(points, FPS)
        assert gt[0] == 30
        assert all(h.frame >= 27 for h in hits)

    def test_gentle_drift_rally_produces_zero_events(self):
        # Per-sequence adaptive threshold + absolute floor: a rally that is nothing
        # but gentle drift must yield no events, even though some sample always
        # exceeds mean + k*sigma of its own distribution.
        points, _ = make_track([(120, 5, 1.5)], noise=0.5, seed=2)
        assert propose_hits(points, FPS) == []

    def test_slow_jitter_below_speed_gate_produces_zero_events(self):
        # Near-stationary noise has essentially random velocity directions; the
        # px/frame speed gate keeps it from masquerading as direction changes.
        points, _ = make_track([(100, 0.5, 0.3)], noise=1.2, seed=3, start=(400.0, 300.0))
        assert propose_hits(points, FPS) == []

    def test_modest_reversal_in_quiet_rally_is_detected(self):
        # The threshold adapts DOWN to a quiet rally: a modest (non-smash) reversal
        # still stands out against low per-sequence motion statistics.
        points, gt = make_track([(40, 3, 8), (40, 3, -7)], start=(300.0, 150.0), noise=0.3, seed=5)
        hits = propose_hits(points, FPS)
        assert len(hits) == 1
        assert abs(hits[0].frame - gt[0]) <= 1

    def test_min_gap_suppresses_nearby_events(self):
        points, gt = make_track(
            [(30, 2, -10), (4, -2, 10), (30, 2, -10), (30, -2, 10)],
            noise=0.3,
            seed=4,
        )
        assert gt == [30, 34, 64]
        # default 250ms @ 30fps = 7.5 frames: the 30/34 pair collapses to one event
        near_pair = [h for h in propose_hits(points, FPS) if 27 <= h.frame <= 37]
        assert len(near_pair) == 1
        # 100ms = 3 frames: 4 frames apart is now far enough — both events survive
        near_pair = [h for h in propose_hits(points, FPS, min_gap_ms=100) if 27 <= h.frame <= 37]
        assert len(near_pair) == 2

    def test_too_few_points_abstains(self):
        points, _ = make_track([(2, 3, -12), (2, 3, 12)])  # 5 points, huge reversal
        assert propose_hits(points, FPS) == []
        assert propose_hits([], FPS) == []

    def test_missing_gap_is_a_break_not_motion(self):
        # A straight segment with an occlusion gap longer than max_gap_frames: the
        # (0, 0) "missing" placeholders must be ignored and no velocity asserted
        # across the gap — otherwise the gap edges would spike as fake hits.
        points, _ = make_track(
            [(60, 4, -6)],
            start=(200.0, 500.0),
            noise=0.3,
            seed=6,
            missing=set(range(25, 33)),
        )
        assert propose_hits(points, FPS) == []

    def test_inpainted_support_lowers_saliency(self):
        segments = [(40, 2, -11), (40, -2, 11), (40, 2, -11)]
        clean, gt = make_track(segments, noise=0.2, seed=7)
        marked, _ = make_track(segments, noise=0.2, seed=7, inpainted={79, 80, 81})
        assert gt == [40, 80]

        hits = propose_hits(marked, FPS)
        assert [h.frame for h in hits] == gt  # both reversals still found
        by_frame = {h.frame: h.saliency for h in hits}
        clean_by_frame = {h.frame: h.saliency for h in propose_hits(clean, FPS)}
        # identical geometry, but the inpainted-supported event is down-weighted
        assert by_frame[80] < 0.8 * by_frame[40]
        assert by_frame[80] < clean_by_frame[80]
        assert by_frame[40] == pytest.approx(clean_by_frame[40])

    def test_parameter_validation(self):
        points, _ = make_track([(30, 2, -12), (30, -2, 12)])
        with pytest.raises(ValueError):
            propose_hits(points, 0.0)
        with pytest.raises(ValueError):
            propose_hits(points, FPS, min_points=2)


class TestAssignSides:
    def _rally_points(self, **kwargs):
        # down (toward near player), up (near hit), down (far hit), up (near hit)
        return make_track(
            [(30, 2, 12), (35, 2, -12), (35, -2, 13), (30, 2, -12)],
            start=(400.0, 240.0),
            **kwargs,
        )

    def test_alternation_seeded_by_vertical_direction(self):
        points, gt = self._rally_points(noise=0.5, seed=8)
        hits = propose_hits(points, FPS)
        assert [h.frame for h in hits] == gt == [30, 65, 100]
        sided = assign_sides(hits, points)
        assert [h.side for h in sided] == ["near", "far", "near"]
        # inputs are not mutated
        assert all(h.side is None for h in hits)

    def test_uninformative_hit_filled_by_alternation(self):
        # No usable points shortly after the second hit -> it casts no vote, but the
        # other hits seed the parity and alternation fills it in.
        points, _ = self._rally_points(noise=0.3, seed=9, missing=set(range(66, 76)))
        hits = [
            ProposedHit(frame=30, saliency=1.0),
            ProposedHit(frame=65, saliency=1.0),
            ProposedHit(frame=100, saliency=1.0),
        ]
        sided = assign_sides(hits, points)
        assert [h.side for h in sided] == ["near", "far", "near"]

    def test_horizontal_rally_is_ambiguous_all_none(self):
        # Pure horizontal reversals carry no vertical seed: abstain on every hit
        # rather than coin-flip the parity.
        points, gt = make_track(
            [(30, 12, 0), (30, -12, 0), (30, 12, 0)], start=(200.0, 400.0)
        )
        hits = [ProposedHit(frame=f, saliency=1.0) for f in gt]
        assert [h.side for h in assign_sides(hits, points)] == [None, None]

    def test_perfectly_conflicting_votes_all_none(self):
        # Two equally strong "near" votes cannot both be right under alternation and
        # neither labeling wins -> all None, never a guess.
        points, _ = make_track([(30, 0, 10), (60, 0, -10)], start=(400.0, 200.0))
        hits = [ProposedHit(frame=30, saliency=1.0), ProposedHit(frame=60, saliency=1.0)]
        assert [h.side for h in assign_sides(hits, points)] == [None, None]

    def test_empty_hits(self):
        points, _ = self._rally_points()
        assert assign_sides([], points) == []

"""Hand-computed cases for every golden-set metric (eval/metrics.py).

Every expected number here was worked out by hand — if a metric implementation changes
behaviour, these tests say exactly which convention broke (greedy matching, inclusive
thresholds, zero-division rules)."""

import pytest
from synchro_pipeline.eval.metrics import (
    event_f1,
    interval_f1,
    interval_iou,
    pck,
    shuttle_detection_f1,
)


class TestEventF1:
    def test_hand_computed_mixed_case(self):
        # gt 11: pred 10 and 12 both at distance 1 -> only one may match (10 wins the
        # tie via lower pred index after sorting). gt 41: pred 40. gt 90: pred 100 is
        # 10 frames away -> unmatched.
        r = event_f1([10, 12, 40, 100], [11, 41, 90], tolerance_frames=3)
        assert (r.tp, r.fp, r.fn) == (2, 2, 1)
        assert r.precision == pytest.approx(0.5)
        assert r.recall == pytest.approx(2 / 3)
        assert r.f1 == pytest.approx(4 / 7)
        assert r.matches == ((10, 11), (40, 41))

    def test_exact_match(self):
        r = event_f1([5, 10], [5, 10], tolerance_frames=0)
        assert (r.tp, r.fp, r.fn) == (2, 0, 0)
        assert r.f1 == 1.0

    def test_tolerance_boundary_inclusive(self):
        assert event_f1([7], [10], tolerance_frames=3).tp == 1  # |7-10| == 3 counts
        assert event_f1([7], [10], tolerance_frames=2).tp == 0

    def test_two_preds_cannot_share_one_gt(self):
        r = event_f1([10, 11], [10], tolerance_frames=3)
        assert (r.tp, r.fp, r.fn) == (1, 1, 0)

    def test_one_pred_cannot_satisfy_two_gts(self):
        r = event_f1([10], [9, 11], tolerance_frames=3)
        assert (r.tp, r.fp, r.fn) == (1, 0, 1)

    def test_greedy_prefers_nearest_pair(self):
        # pred 13 (distance 1) must claim gt 12 over pred 10 (distance 2)
        r = event_f1([10, 13], [12], tolerance_frames=3)
        assert r.matches == ((13, 12),)
        assert (r.tp, r.fp, r.fn) == (1, 1, 0)

    def test_unsorted_input_ok(self):
        r = event_f1([40, 10], [41, 9], tolerance_frames=3)
        assert r.tp == 2

    def test_empty_preds(self):
        r = event_f1([], [5], tolerance_frames=3)
        assert (r.tp, r.fp, r.fn) == (0, 0, 1)
        assert (r.precision, r.recall, r.f1) == (0.0, 0.0, 0.0)

    def test_empty_gt(self):
        r = event_f1([5], [], tolerance_frames=3)
        assert (r.tp, r.fp, r.fn) == (0, 1, 0)
        assert (r.precision, r.recall, r.f1) == (0.0, 0.0, 0.0)

    def test_both_empty_is_vacuous_perfect(self):
        r = event_f1([], [], tolerance_frames=3)
        assert (r.precision, r.recall, r.f1) == (1.0, 1.0, 1.0)

    def test_negative_tolerance_rejected(self):
        with pytest.raises(ValueError):
            event_f1([1], [1], tolerance_frames=-1)


class TestIntervalIou:
    def test_identical(self):
        assert interval_iou((0, 9), (0, 9)) == 1.0

    def test_disjoint(self):
        assert interval_iou((0, 9), (20, 29)) == 0.0

    def test_inclusive_frame_counting(self):
        # (0,4) has 5 frames; vs (0,9): inter 5, union 10
        assert interval_iou((0, 4), (0, 9)) == pytest.approx(0.5)
        # single shared frame: (0,0) vs (0,1) -> 1/2
        assert interval_iou((0, 0), (0, 1)) == pytest.approx(0.5)

    def test_invalid_interval_rejected(self):
        with pytest.raises(ValueError):
            interval_iou((5, 3), (0, 9))


class TestIntervalF1:
    def test_hand_computed_mixed_case(self):
        # (0,9) perfect; (20,29) vs (21,32): inter 9, union 13 -> 9/13 ~ 0.69 matched;
        # (50,59) vs (70,79): disjoint.
        r = interval_f1(
            [(0, 9), (20, 29), (50, 59)], [(0, 9), (21, 32), (70, 79)], iou_threshold=0.5
        )
        assert (r.tp, r.fp, r.fn) == (2, 1, 1)
        assert r.precision == pytest.approx(2 / 3)
        assert r.recall == pytest.approx(2 / 3)
        assert r.f1 == pytest.approx(2 / 3)
        assert ((0, 9), (0, 9)) in r.matches
        assert ((20, 29), (21, 32)) in r.matches

    def test_iou_threshold_boundary_inclusive(self):
        # IoU exactly 0.5 matches at threshold 0.5
        assert interval_f1([(0, 4)], [(0, 9)], iou_threshold=0.5).tp == 1
        # IoU 5/11 < 0.5 does not
        assert interval_f1([(0, 4)], [(0, 10)], iou_threshold=0.5).tp == 0

    def test_one_pred_spanning_two_gts_matches_only_one(self):
        r = interval_f1([(0, 19)], [(0, 9), (10, 19)], iou_threshold=0.5)
        assert (r.tp, r.fp, r.fn) == (1, 0, 1)

    def test_empty_cases(self):
        assert interval_f1([], [], iou_threshold=0.5).f1 == 1.0
        r = interval_f1([], [(0, 9)])
        assert (r.precision, r.recall, r.f1) == (0.0, 0.0, 0.0)
        r = interval_f1([(0, 9)], [])
        assert (r.tp, r.fp, r.fn) == (0, 1, 0)

    def test_bad_threshold_rejected(self):
        with pytest.raises(ValueError):
            interval_f1([], [], iou_threshold=0.0)
        with pytest.raises(ValueError):
            interval_f1([], [], iou_threshold=1.5)

    def test_invalid_interval_rejected(self):
        with pytest.raises(ValueError):
            interval_f1([(9, 0)], [(0, 9)])


class TestPCK:
    def test_hand_computed(self):
        gt = [(0.0, 0.0), (10.0, 10.0), None, (20.0, 20.0)]
        pred = [(3.0, 4.0), (10.0, 18.0), (1.0, 1.0), None]
        # point 0: dist 5.0 == threshold -> correct (inclusive)
        # point 1: dist 8 -> wrong; point 2: gt None -> excluded; point 3: pred None -> wrong
        r = pck(pred, gt, threshold_px=5.0)
        assert (r.n_valid, r.n_correct) == (3, 1)
        assert r.pck == pytest.approx(1 / 3)

    def test_all_correct(self):
        pts = [(1.0, 2.0), (3.0, 4.0)]
        assert pck(pts, pts, threshold_px=0.0).pck == 1.0

    def test_all_gt_none_is_vacuous_perfect(self):
        r = pck([None, None], [None, None], threshold_px=5.0)
        assert (r.n_valid, r.pck) == (0, 1.0)

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="matched by index"):
            pck([(0.0, 0.0)], [(0.0, 0.0), (1.0, 1.0)], threshold_px=5.0)

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValueError):
            pck([], [], threshold_px=-1.0)


class TestShuttleDetectionF1:
    def test_hand_computed(self):
        gt = {0: (100.0, 100.0), 1: (200.0, 200.0), 2: None, 3: (50.0, 50.0), 4: None}
        pred = {
            0: (103.0, 104.0),  # dist 5.0 -> TP (inclusive threshold)
            1: (210.0, 200.0),  # dist 10 -> FN + FP (missed gt AND wrong assertion)
            2: (5.0, 5.0),  # gt invisible -> FP
            # 3: absent -> FN
            4: None,  # explicit no-claim on invisible frame -> nothing
            7: (1.0, 1.0),  # frame not in gt -> ignored (unlabelled)
        }
        r = shuttle_detection_f1(pred, gt, dist_threshold_px=5.0)
        assert (r.tp, r.fp, r.fn) == (1, 2, 2)
        assert r.precision == pytest.approx(1 / 3)
        assert r.recall == pytest.approx(1 / 3)
        assert r.matches == ((0, 0),)

    def test_perfect(self):
        gt = {0: (1.0, 1.0), 1: None}
        pred = {0: (1.0, 1.0)}
        r = shuttle_detection_f1(pred, gt)
        assert (r.precision, r.recall, r.f1) == (1.0, 1.0, 1.0)

    def test_pred_none_counts_as_no_prediction(self):
        gt = {0: (1.0, 1.0)}
        r = shuttle_detection_f1({0: None}, gt)
        assert (r.tp, r.fp, r.fn) == (0, 0, 1)

    def test_empty_gt_ignores_all_preds(self):
        # no labelled frames at all -> nothing evaluable -> vacuous perfect
        r = shuttle_detection_f1({0: (1.0, 1.0)}, {})
        assert (r.precision, r.recall, r.f1) == (1.0, 1.0, 1.0)

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValueError):
            shuttle_detection_f1({}, {}, dist_threshold_px=-0.1)

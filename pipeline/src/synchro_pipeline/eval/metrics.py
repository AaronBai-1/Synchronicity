"""Pure golden-set metrics (docs/plan.md "Verification approach").

One function per stage target:

    * event_f1              — S5 hit detection, plan target F1@±3 frames >= 0.90
    * interval_f1           — S1 rally segmentation, plan target >= 97%
    * pck                   — S2 court keypoints, plan target PCK@5px >= 0.95
    * shuttle_detection_f1  — S4 shuttle tracking, plan target F1@5px

Shared conventions (deliberate, documented once):

    Greedy one-to-one matching. Candidate (pred, gt) pairs are ranked best-first
    (smallest frame distance for events, largest IoU for intervals; deterministic
    tie-breaks), then accepted greedily — each prediction and each ground-truth item
    matches at most once. This prevents both double-counting failure modes: two nearby
    predictions cannot both claim one gt hit, and one prediction cannot satisfy two gt
    hits. Greedy (not Hungarian) is intentional: simpler to reason about when eyeballing
    disagreements on the golden set, and the difference is negligible at eval scale.

    Zero-division convention. precision = tp/(tp+fp) and recall = tp/(tp+fn); an
    undefined ratio (0/0) is scored 0.0 — the safe side for CI — EXCEPT the fully
    vacuous case (no predictions AND no ground truth), which scores 1.0 everywhere:
    predicting nothing when there is nothing is correct behaviour, and a rally with no
    labels must not drag aggregate numbers down.

    Distance thresholds are inclusive: a hit exactly `tolerance_frames` away, or a
    keypoint at exactly `threshold_px`, counts as correct ("F1@±3" includes ±3).

Everything here is pure and dependency-light (stdlib + math only) so CI can never
break these numbers by way of an environment problem.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Interval = tuple[int, int]  # (start_frame, end_frame), inclusive on both ends


@dataclass(frozen=True)
class F1Result:
    """Precision/recall/F1 with counts and the accepted one-to-one matches.

    `matches` holds (pred, gt) pairs — frame numbers for event metrics, (start, end)
    tuples for interval metrics — so a failing CI run can print exactly which labels
    went unmatched.
    """

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    matches: tuple[tuple[object, object], ...] = ()

    def as_dict(self) -> dict[str, float | int]:
        """JSON-friendly summary (matches omitted — they are for debugging, not reports)."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass(frozen=True)
class PCKResult:
    """Percentage of Correct Keypoints. n_valid counts labelled (non-None) gt points."""

    n_valid: int
    n_correct: int
    pck: float

    def as_dict(self) -> dict[str, float | int]:
        return {"n_valid": self.n_valid, "n_correct": self.n_correct, "pck": self.pck}


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Apply the zero-division convention documented in the module docstring."""
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def event_f1(
    pred_frames: Sequence[int], gt_frames: Sequence[int], tolerance_frames: int
) -> F1Result:
    """Frame-event detection F1 with a ± frame tolerance (plan: hit detection F1@±3).

    Matching: candidate pairs with |pred - gt| <= tolerance_frames, ranked by distance
    ascending (ties: earlier gt, then earlier pred, both after ascending sort), accepted
    greedily one-to-one. Inputs may be unsorted and may contain duplicates; duplicates
    are real extra events (a duplicated prediction that finds no free gt is a FP).
    """
    if tolerance_frames < 0:
        raise ValueError("tolerance_frames must be >= 0")
    pred = sorted(int(f) for f in pred_frames)
    gt = sorted(int(f) for f in gt_frames)

    candidates: list[tuple[int, int, int]] = []  # (distance, gt_idx, pred_idx)
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            d = abs(p - g)
            if d <= tolerance_frames:
                candidates.append((d, gi, pi))
    candidates.sort()

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((pred[pi], gt[gi]))

    tp = len(matches)
    precision, recall, f1 = _prf(tp, len(pred) - tp, len(gt) - tp)
    matches.sort(key=lambda m: (m[1], m[0]))
    return F1Result(precision, recall, f1, tp, len(pred) - tp, len(gt) - tp, tuple(matches))


def interval_iou(a: Interval, b: Interval) -> float:
    """IoU of two inclusive frame intervals, counted in frames (a 1-frame interval has
    length 1 — frame indices are discrete, plan S0: the frame index is the time authority).
    """
    for iv in (a, b):
        if iv[1] < iv[0]:
            raise ValueError(f"invalid interval {iv}: end < start")
    inter = min(a[1], b[1]) - max(a[0], b[0]) + 1
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union


def interval_f1(
    pred_intervals: Sequence[Interval],
    gt_intervals: Sequence[Interval],
    iou_threshold: float = 0.5,
) -> F1Result:
    """Interval detection F1 at an IoU threshold (plan: rally segmentation >= 97%).

    Matching: candidate pairs with IoU >= iou_threshold (inclusive), ranked by IoU
    descending (ties: earlier gt, then earlier pred, after sorting both lists by start
    frame), accepted greedily one-to-one — one long prediction spanning two gt rallies
    matches only one of them, surfacing under-segmentation as a FN.
    """
    if not (0.0 < iou_threshold <= 1.0):
        raise ValueError("iou_threshold must be in (0, 1]")
    pred = sorted((int(s), int(e)) for s, e in pred_intervals)
    gt = sorted((int(s), int(e)) for s, e in gt_intervals)

    candidates: list[tuple[float, int, int]] = []  # (-iou, gt_idx, pred_idx)
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            iou = interval_iou(p, g)
            if iou >= iou_threshold:
                candidates.append((-iou, gi, pi))
    candidates.sort()

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[Interval, Interval]] = []
    for _, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((pred[pi], gt[gi]))

    tp = len(matches)
    precision, recall, f1 = _prf(tp, len(pred) - tp, len(gt) - tp)
    matches.sort(key=lambda m: (m[1], m[0]))
    return F1Result(precision, recall, f1, tp, len(pred) - tp, len(gt) - tp, tuple(matches))


def pck(
    pred_pts: Sequence[tuple[float, float] | None],
    gt_pts: Sequence[tuple[float, float] | None],
    threshold_px: float,
) -> PCKResult:
    """Percentage of Correct Keypoints (plan: court keypoints PCK@5px >= 0.95).

    Points are index-aligned (order = domain.court.COURT_KEYPOINT_NAMES for court use).
    A None gt point is unlabelled/occluded and excluded from the denominator; a None
    pred against a labelled gt is simply incorrect. Distance is Euclidean, inclusive at
    the threshold. With zero labelled gt points pck is 1.0 (vacuous — see module
    docstring); the golden-set loader is responsible for ensuring labels exist.
    """
    if len(pred_pts) != len(gt_pts):
        raise ValueError(
            f"pred/gt keypoint counts differ: {len(pred_pts)} vs {len(gt_pts)} "
            "(points are matched by index, not proximity)"
        )
    if threshold_px < 0:
        raise ValueError("threshold_px must be >= 0")
    n_valid = 0
    n_correct = 0
    for pred, gt in zip(pred_pts, gt_pts, strict=True):
        if gt is None:
            continue
        n_valid += 1
        if pred is None:
            continue
        if math.hypot(pred[0] - gt[0], pred[1] - gt[1]) <= threshold_px:
            n_correct += 1
    return PCKResult(n_valid, n_correct, n_correct / n_valid if n_valid > 0 else 1.0)


def shuttle_detection_f1(
    pred_xy_by_frame: Mapping[int, tuple[float, float] | None],
    gt_xy_by_frame: Mapping[int, tuple[float, float] | None],
    dist_threshold_px: float = 5.0,
) -> F1Result:
    """Per-frame shuttle detection F1 honoring visibility (plan: shuttle F1@5px).

    Inputs map frame -> (x, y) when the shuttle is claimed/labelled visible there, or
    -> None when explicitly invisible. Only frames present in `gt_xy_by_frame` are
    evaluated — predictions on unlabelled frames are ignored, so partial golden shuttle
    labels never manufacture false positives.

    Per-frame accounting (no cross-frame matching — the correspondence is the frame):
        * gt visible, pred within threshold (inclusive)  -> TP
        * gt visible, no pred                            -> FN
        * gt visible, pred farther than threshold        -> FN *and* FP — the gt frame
          was missed AND a wrong position was asserted; counting both keeps precision
          honest about position quality, not just presence
        * gt invisible (None), pred present              -> FP (hallucinated shuttle)

    `matches` holds (frame, frame) pairs for the TP frames.
    """
    if dist_threshold_px < 0:
        raise ValueError("dist_threshold_px must be >= 0")
    tp = fp = fn = 0
    matches: list[tuple[int, int]] = []
    for frame in sorted(gt_xy_by_frame):
        gt = gt_xy_by_frame[frame]
        pred = pred_xy_by_frame.get(frame)
        if gt is not None:
            if pred is None:
                fn += 1
            elif math.hypot(pred[0] - gt[0], pred[1] - gt[1]) <= dist_threshold_px:
                tp += 1
                matches.append((frame, frame))
            else:
                fn += 1
                fp += 1
        elif pred is not None:
            fp += 1
    precision, recall, f1 = _prf(tp, fp, fn)
    return F1Result(precision, recall, f1, tp, fp, fn, tuple(matches))

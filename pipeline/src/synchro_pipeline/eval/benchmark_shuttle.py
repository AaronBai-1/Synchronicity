"""Shuttle-tracking benchmark against a golden match file (docs/plan.md Phase 0:
"Benchmark TrackNetV3 pretrained on golden set"; risk #2: eval across broadcasters).

Runs a ShuttleTracker over the golden match's rally frames (plan cost ordering: shuttle
tracking is rally-frames-only), then:

    * always reports coverage — the detected/inpainted/missing split per rally and
      overall. Coverage is the cheap early-warning signal on new broadcasters even
      before anyone labels shuttle positions.
    * where a rally carries hand-labelled shuttle points, scores shuttle_detection_f1
      (plan target metric: shuttle F1@5px). Rallies without labels are never silently
      folded in — per plan's honest-abstention principle they appear with null metrics.

Usage:
    uv run python -m synchro_pipeline.eval.benchmark_shuttle \\
        --video match.mp4 --golden golden/match.json --tracker fake --out report.json

`--tracker fake` runs end-to-end with no weights/GPU/footage — it exists so this
harness (and CI) can prove the plumbing before TrackNetV3 is vendored.

Before scoring, the video's sha256 is checked against the hash stamped in the golden
file (frame labels only transfer within one exact encode); mismatch aborts with exit
code 4 unless --skip-hash-check. Exit codes: 0 ok, 2 golden load failure, 3 tracker
could not run, 4 video/label encode mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from synchro_pipeline.eval.golden import (
    GoldenLoadError,
    GoldenMatch,
    GoldenRally,
    GoldenVideoMismatch,
    load_golden_match,
    verify_video_hash,
)
from synchro_pipeline.eval.metrics import shuttle_detection_f1
from synchro_pipeline.perception.base import (
    FakeShuttleTracker,
    ShuttleTracker,
    ShuttleTrackPoint,
)

TRACKER_CHOICES = ("fake", "tracknetv3")


def build_tracker(name: str, seed: int = 0, weights: str | None = None) -> ShuttleTracker:
    """Construct a tracker by CLI name. Construction never needs torch/weights (lazy)."""
    if name == "fake":
        return FakeShuttleTracker(seed=seed)
    if name == "tracknetv3":
        # Imported here so `--tracker fake` works even if the wrapper module grows deps.
        from synchro_pipeline.perception.tracknet_v3 import TrackNetV3Tracker

        return TrackNetV3Tracker(weights_path=weights)
    raise ValueError(f"unknown tracker {name!r}; choices: {TRACKER_CHOICES}")


def rally_frames(match: GoldenMatch) -> list[int]:
    """All frames inside any golden rally span, sorted (spans are inclusive)."""
    frames: set[int] = set()
    for rally in match.rallies:
        frames.update(range(rally.start_frame, rally.end_frame + 1))
    return sorted(frames)


def coverage_stats(points: Sequence[ShuttleTrackPoint]) -> dict[str, int | float]:
    """Visibility histogram + percentages for a set of tracked points."""
    counts = {"detected": 0, "inpainted": 0, "missing": 0}
    for p in points:
        counts[p.visibility] += 1
    total = len(points)
    stats: dict[str, int | float] = {"n_frames": total, **counts}
    for state, n in counts.items():
        stats[f"{state}_pct"] = round(100.0 * n / total, 2) if total else 0.0
    return stats


def gt_shuttle_dict(rally: GoldenRally) -> dict[int, tuple[float, float] | None]:
    """Golden shuttle labels as the frame->xy|None mapping shuttle_detection_f1 wants."""
    assert rally.shuttle is not None
    return {
        p.frame: ((p.x, p.y) if p.visible else None)  # type: ignore[misc]
        for p in rally.shuttle
    }


def pred_shuttle_dict(
    points: Sequence[ShuttleTrackPoint],
) -> dict[int, tuple[float, float] | None]:
    """Tracker output as a frame->xy mapping; 'missing' frames carry no position claim.

    Inpainted points DO count as predictions: plan S4 carries them downstream, so the
    benchmark must score them as the position assertions they are.
    """
    return {
        p.frame_idx: (p.x, p.y)
        for p in points
        if p.visibility in ("detected", "inpainted")
    }


def _benchmark(match: GoldenMatch, points: list[ShuttleTrackPoint]) -> dict:
    """Assemble the JSON-ready report from tracked points and golden labels."""
    by_frame = {p.frame_idx: p for p in points}
    pred = pred_shuttle_dict(points)

    per_rally: list[dict] = []
    combined_gt: dict[int, tuple[float, float] | None] = {}
    for rally in match.rallies:
        span = range(rally.start_frame, rally.end_frame + 1)
        rally_points = [by_frame[f] for f in span if f in by_frame]
        entry: dict = {
            "start_frame": rally.start_frame,
            "end_frame": rally.end_frame,
            "coverage": coverage_stats(rally_points),
            "has_shuttle_labels": rally.shuttle is not None,
            "metrics": None,
        }
        if rally.shuttle is not None:
            gt = gt_shuttle_dict(rally)
            entry["metrics"] = {
                **shuttle_detection_f1(pred, gt).as_dict(),
                "n_labeled_frames": len(gt),
            }
            combined_gt.update(gt)  # rally spans never overlap (golden schema invariant)
        per_rally.append(entry)

    overall = None
    if combined_gt:
        overall = {
            **shuttle_detection_f1(pred, combined_gt).as_dict(),
            "n_labeled_frames": len(combined_gt),
        }
    return {
        "per_rally": per_rally,
        "coverage": coverage_stats(points),
        "shuttle_metrics": overall,
        "n_rallies": len(match.rallies),
        "n_rallies_with_shuttle_labels": sum(1 for r in match.rallies if r.shuttle is not None),
        "n_frames_tracked": len(points),
    }


def _print_table(report: dict) -> None:
    """Small human-readable summary; the JSON file is the machine artifact."""
    header = (
        f"{'rally':>13}  {'frames':>6}  {'det%':>6}  {'inp%':>6}  {'miss%':>6}  "
        f"{'labeled':>7}  {'P':>6}  {'R':>6}  {'F1':>6}"
    )
    print(header)
    print("-" * len(header))
    rows = [*report["per_rally"], None]
    for row in rows:
        if row is None:
            cov = report["coverage"]
            name, labeled, metrics = "overall", "-", report["shuttle_metrics"]
        else:
            cov = row["coverage"]
            name = f"{row['start_frame']}-{row['end_frame']}"
            labeled = "yes" if row["has_shuttle_labels"] else "no"
            metrics = row["metrics"]
        prf = (
            (f"{metrics['precision']:6.3f}", f"{metrics['recall']:6.3f}", f"{metrics['f1']:6.3f}")
            if metrics
            else ("     -", "     -", "     -")
        )
        print(
            f"{name:>13}  {cov['n_frames']:>6}  {cov['detected_pct']:>6.1f}  "
            f"{cov['inpainted_pct']:>6.1f}  {cov['missing_pct']:>6.1f}  "
            f"{labeled:>7}  {prf[0]}  {prf[1]}  {prf[2]}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m synchro_pipeline.eval.benchmark_shuttle",
        description="Benchmark a shuttle tracker against a golden match file.",
    )
    parser.add_argument("--video", required=True, help="path to the match video")
    parser.add_argument("--golden", required=True, help="path to the golden match JSON")
    parser.add_argument("--tracker", required=True, choices=TRACKER_CHOICES)
    parser.add_argument("--out", required=True, help="path to write the JSON report")
    parser.add_argument("--seed", type=int, default=0, help="fake tracker seed (default 0)")
    parser.add_argument(
        "--weights",
        default=None,
        help="TrackNetV3 checkpoint (default: $SYNCHRO_TRACKNET_WEIGHTS)",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="score even if --video is not the encode the labels were made against",
    )
    args = parser.parse_args(argv)

    try:
        match = load_golden_match(args.golden)
    except GoldenLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Frame-exact labels only mean anything against the exact encode they were made on;
    # scoring against a different file would report precise-looking nonsense.
    if args.skip_hash_check:
        pass
    elif match.video_sha256 is None:
        print(
            "warning: golden file has no video_sha256 — footage identity unverified "
            "(label with the current tools to stamp it)",
            file=sys.stderr,
        )
    else:
        try:
            verify_video_hash(match, args.video)
        except (GoldenVideoMismatch, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 4

    tracker = build_tracker(args.tracker, seed=args.seed, weights=args.weights)
    frames = rally_frames(match)
    try:
        # No rallies labeled yet -> track the whole video rather than nothing, so the
        # coverage numbers still say something useful about a fresh match.
        points = tracker.track(args.video, frames if frames else None)
    except RuntimeError as exc:
        print(f"error: tracker {tracker.name!r} could not run:\n{exc}", file=sys.stderr)
        return 3

    report = {
        "match_id": match.match_id,
        "video": str(args.video),
        "golden": str(args.golden),
        "tracker": tracker.name,
        "seed": args.seed if args.tracker == "fake" else None,
        **_benchmark(match, points),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"match {match.match_id}  tracker={tracker.name}  frames={len(points)}")
    _print_table(report)
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

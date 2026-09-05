"""Trajectory-only hit-detection benchmark against a golden match file (docs/plan.md
risk #1: prototype a trajectory-only baseline early and measure the gap to the fused
S5 approach; plan target for full S5 is hit F1@±3 frames >= 0.90).

For every golden rally this CLI builds a shuttle trajectory, runs
perception.hits_baseline.propose_hits + assign_sides over it, and — where the rally
carries hand-labelled hit frames — scores eval.metrics.event_f1 at a ± frame tolerance
plus side accuracy over the matched pairs. Rallies without hit labels (or without a
trajectory) are reported with null metrics, never silently folded in (plan's honest
abstention), and every aggregate carries its n.

Trajectory sources (--tracker):

    * fake       — FakeShuttleTracker; proves the plumbing offline (no weights/footage).
    * tracknetv3 — the real S4 tracker (needs weights).
    * golden     — the golden file's OWN hand-labelled shuttle points are used as the
      trajectory. This benchmarks the baseline's event logic under perfect tracking,
      isolating it from tracker error — the number to quote for "trajectory-only
      ceiling" before TrackNetV3 lands.

Usage:
    uv run python -m synchro_pipeline.eval.benchmark_hits \\
        --video match.mp4 --golden golden/match.json --tracker golden --out report.json

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

from synchro_pipeline.eval.benchmark_shuttle import build_tracker, rally_frames
from synchro_pipeline.eval.golden import (
    GoldenLoadError,
    GoldenMatch,
    GoldenRally,
    GoldenVideoMismatch,
    load_golden_match,
    verify_video_hash,
)
from synchro_pipeline.eval.metrics import event_f1
from synchro_pipeline.perception.base import ShuttleTrackPoint
from synchro_pipeline.perception.hits_baseline import ProposedHit, assign_sides, propose_hits

TRACKER_CHOICES = ("fake", "tracknetv3", "golden")

# Plan S0: the canonical mezzanine is 720p30 CFR, so 30 fps is the honest default for
# converting min_gap_ms to frames without probing footage (which may be absent for the
# fake/golden sources).
DEFAULT_FPS = 30.0


def golden_rally_points(rally: GoldenRally) -> list[ShuttleTrackPoint] | None:
    """A rally's hand-labelled shuttle points as tracker-shaped output (or None).

    visible=true labels become visibility="detected" with conf=1.0 (human ground
    truth); visible=false becomes "missing" with the (0, 0) placeholder — same
    contract as ShuttleTracker.track. Unlabelled frames are simply absent;
    propose_hits treats large frame gaps as breaks, not as motion.
    """
    if rally.shuttle is None:
        return None
    return [
        ShuttleTrackPoint(frame_idx=p.frame, x=p.x, y=p.y, conf=1.0, visibility="detected")
        if p.visible
        else ShuttleTrackPoint(frame_idx=p.frame, x=0.0, y=0.0, conf=0.0, visibility="missing")
        for p in rally.shuttle
    ]


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Micro precision/recall/F1 under eval.metrics' documented zero-division
    convention: fully vacuous (no predictions, no ground truth) scores 1.0; any other
    0/0 scores 0.0."""
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _side_stats(
    proposals: Sequence[ProposedHit],
    rally: GoldenRally,
    matches: Sequence[tuple[object, object]],
) -> dict:
    """Side accuracy over the F1-matched (pred, gt) pairs, with its n's.

    Only proposals that actually asserted a side count toward accuracy — abstentions
    (side=None) are reported via n_sided vs n_matched, never scored as wrong guesses
    (nor as right ones).
    """
    pred_side = {h.frame: h.side for h in proposals}
    gt_side = {h.frame: h.side for h in rally.hits}
    n_sided = 0
    n_correct = 0
    for pred_frame, gt_frame in matches:
        side = pred_side.get(int(pred_frame))  # type: ignore[call-overload]
        if side is None:
            continue
        n_sided += 1
        if side == gt_side.get(int(gt_frame)):  # type: ignore[call-overload]
            n_correct += 1
    return {
        "n_matched": len(matches),
        "n_sided": n_sided,
        "n_correct": n_correct,
        "accuracy": (n_correct / n_sided) if n_sided else None,
    }


def _benchmark(
    match: GoldenMatch,
    points_for_rally: dict[int, list[ShuttleTrackPoint] | None],
    fps: float,
    tolerance_frames: int,
) -> dict:
    """Assemble the JSON-ready report. `points_for_rally` is keyed by rally index."""
    per_rally: list[dict] = []
    tp = fp = fn = 0
    n_matched = n_sided = n_correct = 0
    n_scored = 0
    for idx, rally in enumerate(match.rallies):
        points = points_for_rally[idx]
        # Golden hits default to [] with no labelled/unlabelled distinction in the
        # schema, so an empty list is treated as "no hit labels" and never scored.
        has_hit_labels = len(rally.hits) > 0
        entry: dict = {
            "start_frame": rally.start_frame,
            "end_frame": rally.end_frame,
            "has_trajectory": points is not None,
            "has_hit_labels": has_hit_labels,
            "n_gt_hits": len(rally.hits),
            "n_proposed": None,
            "metrics": None,
            "side": None,
        }
        if points is not None:
            proposals = assign_sides(propose_hits(points, fps), points)
            entry["n_proposed"] = len(proposals)
            if has_hit_labels:
                result = event_f1(
                    [h.frame for h in proposals],
                    [h.frame for h in rally.hits],
                    tolerance_frames,
                )
                side = _side_stats(proposals, rally, result.matches)
                entry["metrics"] = result.as_dict()
                entry["side"] = side
                tp += result.tp
                fp += result.fp
                fn += result.fn
                n_matched += side["n_matched"]
                n_sided += side["n_sided"]
                n_correct += side["n_correct"]
                n_scored += 1
        per_rally.append(entry)

    overall = None
    if n_scored:
        precision, recall, f1 = _prf(tp, fp, fn)
        overall = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "side": {
                "n_matched": n_matched,
                "n_sided": n_sided,
                "n_correct": n_correct,
                "accuracy": (n_correct / n_sided) if n_sided else None,
            },
        }
    return {
        "per_rally": per_rally,
        "hit_metrics": overall,
        "n_rallies": len(match.rallies),
        "n_rallies_with_hit_labels": sum(1 for r in match.rallies if r.hits),
        "n_rallies_scored": n_scored,
    }


def _print_table(report: dict) -> None:
    """Small human-readable summary; the JSON file is the machine artifact."""
    header = (
        f"{'rally':>13}  {'traj':>4}  {'gt':>4}  {'pred':>4}  "
        f"{'P':>6}  {'R':>6}  {'F1':>6}  {'side%':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in [*report["per_rally"], None]:
        if row is None:
            name, traj, gt, pred = "overall", "-", "-", "-"
            metrics, side = report["hit_metrics"], None
            if metrics is not None:
                side = metrics["side"]
        else:
            name = f"{row['start_frame']}-{row['end_frame']}"
            traj = "yes" if row["has_trajectory"] else "no"
            gt = str(row["n_gt_hits"]) if row["has_hit_labels"] else "-"
            pred = str(row["n_proposed"]) if row["n_proposed"] is not None else "-"
            metrics, side = row["metrics"], row["side"]
        prf = (
            (f"{metrics['precision']:6.3f}", f"{metrics['recall']:6.3f}", f"{metrics['f1']:6.3f}")
            if metrics
            else ("     -", "     -", "     -")
        )
        side_pct = (
            f"{100.0 * side['accuracy']:5.1f}%"
            if side is not None and side["accuracy"] is not None
            else "     -"
        )
        print(f"{name:>13}  {traj:>4}  {gt:>4}  {pred:>4}  {prf[0]}  {prf[1]}  {prf[2]}  {side_pct}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m synchro_pipeline.eval.benchmark_hits",
        description="Benchmark the trajectory-only hit-detection baseline on a golden match.",
    )
    parser.add_argument("--video", required=True, help="path to the match video")
    parser.add_argument("--golden", required=True, help="path to the golden match JSON")
    parser.add_argument(
        "--tracker",
        required=True,
        choices=TRACKER_CHOICES,
        help="trajectory source; 'golden' replays the golden file's own shuttle labels",
    )
    parser.add_argument("--out", required=True, help="path to write the JSON report")
    parser.add_argument("--seed", type=int, default=0, help="fake tracker seed (default 0)")
    parser.add_argument(
        "--weights",
        default=None,
        help="TrackNetV3 checkpoint (default: $SYNCHRO_TRACKNET_WEIGHTS)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="frames per second of the label timeline (default 30: plan-S0 720p30 mezzanine)",
    )
    parser.add_argument(
        "--tolerance-frames",
        type=int,
        default=3,
        help="event-matching tolerance in frames (default 3: plan target is F1@±3)",
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

    # Frame-exact labels only mean anything against the exact encode they were made on
    # (same gate as benchmark_shuttle; it applies to --tracker golden too, because the
    # labels themselves are only trustworthy relative to that encode).
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

    points_for_rally: dict[int, list[ShuttleTrackPoint] | None]
    if args.tracker == "golden":
        source_name = "golden"
        points_for_rally = {
            i: golden_rally_points(rally) for i, rally in enumerate(match.rallies)
        }
    else:
        tracker = build_tracker(args.tracker, seed=args.seed, weights=args.weights)
        source_name = tracker.name
        frames = rally_frames(match)
        try:
            points = tracker.track(args.video, frames if frames else None)
        except RuntimeError as exc:
            print(f"error: tracker {tracker.name!r} could not run:\n{exc}", file=sys.stderr)
            return 3
        by_frame = {p.frame_idx: p for p in points}
        points_for_rally = {
            i: [
                by_frame[f]
                for f in range(rally.start_frame, rally.end_frame + 1)
                if f in by_frame
            ]
            for i, rally in enumerate(match.rallies)
        }

    report = {
        "match_id": match.match_id,
        "video": str(args.video),
        "golden": str(args.golden),
        "tracker": source_name,
        "seed": args.seed if args.tracker == "fake" else None,
        "fps": args.fps,
        "tolerance_frames": args.tolerance_frames,
        **_benchmark(match, points_for_rally, args.fps, args.tolerance_frames),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"match {match.match_id}  tracker={source_name}  "
        f"rallies_scored={report['n_rallies_scored']}/{report['n_rallies']}"
    )
    _print_table(report)
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

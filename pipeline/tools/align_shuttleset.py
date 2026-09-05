"""Import a ShuttleSet-labelled match into a golden file via anchor alignment
(docs/plan.md Phase 0; docs/golden-set.md §7 has the full workflow).

ShuttleSet (wywyWang/CoachAI-Projects) already carries human per-stroke labels — a
frame number, order and score for every hit of 44 pro matches — but its frame numbers
index THEIR video encode. This tool fits a linear frame map from 2–4 human-provided
anchor correspondences, transforms every stroke, and merges PROPOSED rally spans,
hit frames and an approximate score timeline into the golden JSON, cutting days of
scrub-and-mark labeling down to minutes of anchoring plus a review pass.

Anchoring: pick 2–4 hits spread across the match (first + last rally is ideal — the
lever arm tightens the fit), find each in BOTH timelines (ShuttleSet's frame_num
column ↔ the mezzanine frame in label_rallies), and pass them as
--anchors "ss_frame:mz_frame,ss_frame:mz_frame,...". The fit refuses when any anchor
sits more than --tolerance-frames off the fitted line — that means a mis-identified
anchor (or non-linear timelines), never something to override by raising tolerance.

Near/far sides: by default derived per-rally from ShuttleSet's player-location pixels
(camera-frame y: lower = nearer the broadcast camera — verified convention, see
eval/shuttleset.py). When locations are missing/ambiguous the affected rallies are
skipped and reported; pass --near-serves-first or --far-serves-first (which physical
end the first provided set's opening server occupies — read it off the video once)
to derive ALL sides from serve order + BWF end-change rules instead. Each mode's
exact assumptions are documented on the resolver functions.

What remains human (the emitted labels are PROPOSALS, per the plan's review rule):
refine every rally's start/end in label_rallies, verify the score timeline against
the scorebug, and spot-check ~10 hits against the video before committing.

Existing hand-labelled rallies are never clobbered: a proposed rally whose span
overlaps one already in the golden file is skipped and reported, and the proposed
score timeline is dropped entirely if the file already has one.

Example:
    uv run python pipeline/tools/align_shuttleset.py \\
        --csv shuttleset/<match>/set1.csv --csv shuttleset/<match>/set2.csv \\
        --golden data/golden/<match_id>.json \\
        --video data/mezzanine/<match_id>.mp4 \\
        --match-id <match_id> --video-uri data/mezzanine/<match_id>.mp4 \\
        --anchors "10418:5030,68544:63180" --dry-run

All decision logic lives in synchro_pipeline.eval.shuttleset (unit-tested, GUI- and
network-free); this file is only argument plumbing and printing.
"""

from __future__ import annotations

import argparse
import sys

import cv2
from synchro_pipeline.eval.golden import GoldenMatch, GoldenRally, save_golden_match
from synchro_pipeline.eval.labeling import insert_rally, load_or_create_golden
from synchro_pipeline.eval.shuttleset import (
    Alignment,
    AlignmentError,
    ConversionReport,
    ShuttleSetError,
    load_shuttleset_match,
    make_serve_side_resolver,
    strokes_to_golden,
)


def parse_anchors(text: str) -> list[tuple[int, int]]:
    """Parse "ss:mz,ss:mz,..." into (shuttleset_frame, mezzanine_frame) pairs."""
    anchors: list[tuple[int, int]] = []
    for chunk in text.split(","):
        parts = chunk.strip().split(":")
        try:
            ss, mz = (int(p) for p in parts)
        except ValueError:
            raise ValueError(
                f"bad anchor {chunk.strip()!r} — expected shuttleset_frame:mezzanine_frame, "
                'e.g. --anchors "10418:5030,68544:63180"'
            ) from None
        if ss < 0 or mz < 0:
            raise ValueError(f"bad anchor {chunk.strip()!r} — frame numbers must be >= 0")
        anchors.append((ss, mz))
    return anchors


def merge_proposals(
    match: GoldenMatch, report: ConversionReport, rallies: list[GoldenRally]
) -> tuple[GoldenMatch, list[str]]:
    """Merge proposed rallies + score timeline into the match, never clobbering.

    Hand labels win every conflict: a proposed rally overlapping an existing span is
    skipped (message returned for the report), and the proposed score timeline is
    dropped wholesale when the file already has one — mixing approximate entries into
    a hand-verified timeline would hide which is which.
    """
    skipped: list[str] = []
    merged = list(match.rallies)
    for rally in rallies:
        try:
            merged = insert_rally(merged, rally)
        except ValueError as exc:
            skipped.append(f"kept existing hand labels: {exc}")
    if match.score_timeline and report.score_timeline:
        skipped.append(
            f"golden file already has {len(match.score_timeline)} score entries — "
            f"dropped all {len(report.score_timeline)} proposed (approximate) entries"
        )
        score_timeline = match.score_timeline
    else:
        score_timeline = report.score_timeline
    updated = GoldenMatch.model_validate(
        {
            **match.model_dump(),
            "rallies": [r.model_dump() for r in merged],
            "score_timeline": [s.model_dump() for s in score_timeline],
        }
    )
    return updated, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv",
        action="append",
        required=True,
        help="ShuttleSet per-set csv (repeat per set; set number comes from the "
        "set<N>.csv filename)",
    )
    parser.add_argument("--golden", required=True, help="golden match JSON to write/update")
    parser.add_argument(
        "--video",
        required=True,
        help="the mezzanine the golden labels index into — stamps/verifies its sha256 "
        "and clamps proposed frames to its length",
    )
    parser.add_argument(
        "--anchors",
        required=True,
        help='2-4 correspondences "ss_frame:mz_frame,..." — the same visually-identified '
        "hits in ShuttleSet's frame_num column and the mezzanine",
    )
    parser.add_argument("--match-id", default=None, help="required when creating a new golden file")
    parser.add_argument("--video-uri", default=None, help="required when creating a new golden file")
    parser.add_argument(
        "--tolerance-frames",
        type=float,
        default=5.0,
        help="refuse the fit when any anchor is further than this off the fitted line "
        "(default 5; a violation means a wrong anchor, not a knob to loosen)",
    )
    side = parser.add_mutually_exclusive_group()
    side.add_argument(
        "--near-serves-first",
        action="store_true",
        help="derive ALL near/far sides from serve order, seeded by: the first provided "
        "set's opening server stands on the NEAR end (closest to camera); use when "
        "location-based resolution abstains",
    )
    side.add_argument(
        "--far-serves-first",
        action="store_true",
        help="as --near-serves-first, but the opening server stands on the FAR end",
    )
    parser.add_argument(
        "--pad-s",
        type=float,
        default=1.5,
        help="seconds of padding around each rally's hit span for the PROPOSED "
        "boundaries (default 1.5; refine in label_rallies)",
    )
    parser.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help="proceed (and re-stamp) when --video is not the encode the golden file's "
        "labels were made against",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the fit and the full report without writing the golden file",
    )
    args = parser.parse_args(argv)

    try:
        anchors = parse_anchors(args.anchors)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    try:
        strokes = load_shuttleset_match(args.csv)
    except ShuttleSetError as exc:
        sys.exit(f"error: {exc}")
    if not strokes:
        sys.exit(f"error: no strokes parsed from {args.csv}")

    try:
        alignment = Alignment.fit(anchors, tolerance_frames=args.tolerance_frames)
    except AlignmentError as exc:
        sys.exit(f"error: {exc}")
    print(
        f"alignment: mezzanine = {alignment.scale:.6f} * shuttleset + {alignment.offset:.2f}"
    )
    for (ss, mz), residual in zip(alignment.anchors, alignment.residuals, strict=True):
        print(f"  anchor {ss}:{mz}  residual {residual:+.2f} frames")
    print(
        f"  max residual {alignment.max_residual_frames:.2f} frames "
        f"(tolerance {args.tolerance_frames:g})"
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"error: could not open video {args.video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0  # mezzanine contract is 30 CFR
    cap.release()
    if n_frames <= 0:
        sys.exit(f"error: video reports no frames: {args.video}")

    try:
        match = load_or_create_golden(
            args.golden,
            args.match_id,
            args.video_uri,
            video_path=args.video,
            allow_hash_mismatch=args.allow_hash_mismatch,
        )
    except (ValueError, OSError) as exc:
        sys.exit(f"error: {exc}")

    side_resolver = None
    if args.near_serves_first or args.far_serves_first:
        try:
            side_resolver = make_serve_side_resolver(
                strokes, "near" if args.near_serves_first else "far"
            )
        except ShuttleSetError as exc:
            sys.exit(f"error: {exc}")

    rallies, report = strokes_to_golden(
        strokes,
        alignment,
        side_resolver=side_resolver,
        pad_s=args.pad_s,
        fps=fps,
        max_frame=n_frames - 1,
    )
    if side_resolver is not None:
        report.side_mode = (
            "serve order, seeded: first provided set's opening server is on the "
            f"{'near' if args.near_serves_first else 'far'} end"
        )
    for line in report.summary_lines():
        print(line)

    if args.dry_run:
        print(f"dry run — {args.golden} not written")
        return 0

    updated, merge_skipped = merge_proposals(match, report, rallies)
    for line in merge_skipped:
        print(f"SKIPPED  {line}")
    save_golden_match(updated, args.golden)
    added = len(updated.rallies) - len(match.rallies)
    print(
        f"saved {args.golden}: +{added} rallies ({len(updated.rallies)} total), "
        f"{len(updated.score_timeline)} score entries — now refine boundaries in "
        "label_rallies and verify per docs/golden-set.md §7"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

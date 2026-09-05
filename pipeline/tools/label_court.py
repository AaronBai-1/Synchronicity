"""Golden-set court-corner labeling tool (docs/plan.md Phase 0: hand-label court corners;
plan S2 ground truth for PCK@5px / homography reprojection).

Click the 4 DOUBLES corners in this order:

    1. near-left   2. near-right   3. far-right   4. far-left
    (near = bottom of the broadcast frame; left = screen-left)

The tool fits the image→court homography from those 4 clicks
(domain.court.fit_homography), projects all 16 canonical keypoints back onto the frame,
and draws the full court grid so you can verify against the painted lines. Nudge any
point that is off, mark occluded points null, then save into the golden JSON's
court_labels (insert-or-replace for that frame).

Keys:
    [ / ]        select previous / next keypoint
    arrows       nudge the selected point by 1 px   (h/j/k/l also work)
    H/J/K/L      nudge by 5 px
    x            toggle the selected point null (occluded / not labelable)
    r            reset — re-click the 4 corners
    s            save this frame's label into the golden JSON
    q or ESC     quit

Example:
    uv run python pipeline/tools/label_court.py \\
        --video match.mp4 --frame 1200 --golden golden/match.json \\
        --match-id 2024_ao_final --video-uri r2://matches/2024_ao_final.mp4

All decision logic lives in synchro_pipeline.eval.labeling (unit-tested, GUI-free);
this file is only the OpenCV event loop, which is deliberately untested.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np
from synchro_pipeline.domain.court import COURT_KEYPOINT_NAMES, N_COURT_KEYPOINTS
from synchro_pipeline.eval.golden import save_golden_match
from synchro_pipeline.eval.labeling import (
    CORNER_CLICK_ORDER,
    COURT_LINE_INDICES,
    corners_to_keypoints,
    keypoints_to_label,
    load_or_create_golden,
    merge_court_label,
    nudge_point,
)

WINDOW = "label_court"

# waitKeyEx arrow codes across platforms (macOS AppKit, Linux GTK/Qt, Windows).
ARROWS_LEFT = {63234, 65361, 2424832}
ARROWS_UP = {63232, 65362, 2490368}
ARROWS_RIGHT = {63235, 65363, 2555904}
ARROWS_DOWN = {63233, 65364, 2621440}


def read_frame_image(args: argparse.Namespace) -> np.ndarray:
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            sys.exit(f"error: could not read image {args.image}")
        return image
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"error: could not open video {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, image = cap.read()
    cap.release()
    if not ok or image is None:
        sys.exit(f"error: could not read frame {args.frame} from {args.video}")
    return image


def render(
    base: np.ndarray,
    clicks: list[tuple[int, int]],
    points: np.ndarray | None,
    selected: int,
    dropped: set[int],
    message: str,
) -> np.ndarray:
    img = base.copy()
    if points is None:
        next_name = CORNER_CLICK_ORDER[len(clicks)] if len(clicks) < 4 else "?"
        hud = f"click corner {len(clicks) + 1}/4: {next_name}"
        for i, (cx, cy) in enumerate(clicks):
            cv2.circle(img, (cx, cy), 6, (0, 255, 255), 2)
            cv2.putText(
                img, str(i + 1), (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2,
            )
    else:
        for ia, ib in COURT_LINE_INDICES:
            if ia in dropped or ib in dropped:
                continue
            pa = tuple(int(round(v)) for v in points[ia])
            pb = tuple(int(round(v)) for v in points[ib])
            cv2.line(img, pa, pb, (80, 220, 80), 1, cv2.LINE_AA)
        for i, (px, py) in enumerate(points):
            centre = (int(round(px)), int(round(py)))
            if i == selected:
                colour = (0, 255, 255)
            elif i in dropped:
                colour = (0, 0, 255)
            else:
                colour = (80, 220, 80)
            cv2.circle(img, centre, 4, colour, -1 if i == selected else 1)
        state = "NULL" if selected in dropped else "ok"
        hud = (
            f"[{selected:02d}] {COURT_KEYPOINT_NAMES[selected]} ({state})  "
            "[/]=select arrows=nudge x=null r=reset s=save q=quit"
        )
    cv2.putText(img, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if message:
        cv2.putText(img, message, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return img


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="video to grab the frame from")
    source.add_argument("--image", help="alternatively, a still image file")
    parser.add_argument("--frame", type=int, default=0, help="frame index within --video")
    parser.add_argument("--golden", required=True, help="golden match JSON to write/update")
    parser.add_argument("--match-id", default=None, help="required when creating a new golden file")
    parser.add_argument("--video-uri", default=None, help="required when creating a new golden file")
    parser.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help="proceed (and re-stamp) when --video is not the encode the labels were made against",
    )
    args = parser.parse_args(argv)

    try:
        # Hash handling only applies against the real video; a still --image can't
        # vouch for the mezzanine, so it neither stamps nor verifies.
        match = load_or_create_golden(
            args.golden,
            args.match_id,
            args.video_uri,
            video_path=args.video,
            allow_hash_mismatch=args.allow_hash_mismatch,
        )
    except (ValueError, OSError) as exc:
        sys.exit(f"error: {exc}")

    base = read_frame_image(args)
    frame_idx = args.frame if args.video else 0

    clicks: list[tuple[int, int]] = []
    points: np.ndarray | None = None
    selected = 0
    dropped: set[int] = set()
    message = ""

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and points is None and len(clicks) < 4:
            clicks.append((x, y))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        if points is None and len(clicks) == 4:
            points = corners_to_keypoints(clicks)
            if points is None:
                message = "degenerate corners (collinear?) — re-click"
                clicks.clear()

        cv2.imshow(WINDOW, render(base, clicks, points, selected, dropped, message))
        key = cv2.waitKeyEx(30)
        if key == -1:
            continue

        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            clicks.clear()
            points = None
            dropped.clear()
            selected = 0
            message = ""
            continue
        if points is None:
            continue

        dx = dy = 0.0
        step_small, step_big = 1.0, 5.0
        if key == ord("]"):
            selected = (selected + 1) % N_COURT_KEYPOINTS
        elif key == ord("["):
            selected = (selected - 1) % N_COURT_KEYPOINTS
        elif key in ARROWS_LEFT or key == ord("h"):
            dx = -step_small
        elif key in ARROWS_RIGHT or key == ord("l"):
            dx = step_small
        elif key in ARROWS_UP or key == ord("k"):
            dy = -step_small
        elif key in ARROWS_DOWN or key == ord("j"):
            dy = step_small
        elif key == ord("H"):
            dx = -step_big
        elif key == ord("L"):
            dx = step_big
        elif key == ord("K"):
            dy = -step_big
        elif key == ord("J"):
            dy = step_big
        elif key == ord("x"):
            dropped.symmetric_difference_update({selected})
        elif key == ord("s"):
            try:
                label = keypoints_to_label(points, dropped)
                match = merge_court_label(match, frame_idx, label)
                save_golden_match(match, args.golden)
                message = f"saved court label for frame {frame_idx} -> {args.golden}"
            except ValueError as exc:
                message = f"save failed: {exc}"
        if dx or dy:
            points = nudge_point(points, selected, dx, dy)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

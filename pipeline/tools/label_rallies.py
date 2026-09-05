"""Golden-set rally-boundary + hit-frame labeling tool (docs/plan.md Phase 0: hand-label
rally boundaries and frame-exact hit frames — the S1 rally-segmentation and S5
hit-detection ground truth; risk #1 says hit labels come first).

Keyboard-driven video scrubber: mark rally start/end frames and per-hit frames (near or
far side) while playing/stepping the mezzanine video, then save into the golden JSON.

Keys:
    space        play / pause
    , / .        step 1 frame back / forward
    [ / ]        jump 30 frames back / forward
    { / }        jump 300 frames back / forward
    r            mark rally START at the current frame
    e            mark rally END at the current frame (closes the rally)
    c            cancel the open rally (drops its pending hits)
    n            mark a HIT by the NEAR player at the current frame
    f            mark a HIT by the FAR player at the current frame
    u            undo the latest pending hit (open rally only)
    d            delete the rally containing the current frame
    x            delete the hit labelled exactly at the current frame
    s            save all rallies into the golden JSON
    q or ESC     quit (press twice if there are unsaved edits)

Example:
    uv run python pipeline/tools/label_rallies.py \\
        --video match.mp4 --golden golden/match.json \\
        --match-id 2024_ao_final --video-uri r2://matches/2024_ao_final.mp4

All decision logic lives in synchro_pipeline.eval.labeling (unit-tested, GUI-free);
this file is only the OpenCV event loop, which is deliberately untested.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np
from synchro_pipeline.eval.golden import save_golden_match
from synchro_pipeline.eval.labeling import (
    RallyLabelSession,
    find_rally,
    load_or_create_golden,
    merge_rallies,
    remove_hit,
    remove_rally,
)

WINDOW = "label_rallies"


def overlay(
    frame: np.ndarray,
    idx: int,
    n_frames: int,
    session: RallyLabelSession,
    playing: bool,
    dirty: bool,
    message: str,
) -> np.ndarray:
    img = frame.copy()
    current = find_rally(session.rallies, idx)
    lines = [
        f"frame {idx}/{n_frames - 1}  {'PLAY' if playing else 'PAUSE'}"
        f"{'  *unsaved*' if dirty else ''}",
        f"rallies: {len(session.rallies)}"
        + (
            f"  open rally since {session.pending_start}"
            f" ({len(session.pending_hits)} hits pending)"
            if session.is_open
            else ""
        ),
        (
            f"inside rally [{current.start_frame}, {current.end_frame}]"
            f" ({len(current.hits)} hits)"
            if current
            else "outside any rally"
        ),
        "space=play ,/.=step [/]=30 {/}=300 r=start e=end c=cancel "
        "n/f=hit u=undo d/x=del s=save q=quit",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            img, text, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )
    if message:
        cv2.putText(
            img, message, (10, 25 + 25 * len(lines)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2,
        )
    return img


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True, help="the mezzanine video to label")
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
        match = load_or_create_golden(
            args.golden,
            args.match_id,
            args.video_uri,
            video_path=args.video,
            allow_hash_mismatch=args.allow_hash_mismatch,
        )
    except (ValueError, OSError) as exc:
        sys.exit(f"error: {exc}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"error: could not open video {args.video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        sys.exit(f"error: video reports no frames: {args.video}")

    session = RallyLabelSession(rallies=list(match.rallies))
    idx = 0
    shown = -1  # frame index currently decoded into `frame`
    frame: np.ndarray | None = None
    playing = False
    dirty = False
    quit_armed = False
    message = ""

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    while True:
        if playing and shown == idx:
            idx = min(idx + 1, n_frames - 1)
            if idx == n_frames - 1:
                playing = False
        if idx != shown:
            if idx != shown + 1:  # random access; sequential reads just continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, decoded = cap.read()
            if ok and decoded is not None:
                frame, shown = decoded, idx
            else:
                message = f"could not decode frame {idx}"
                idx = shown if shown >= 0 else 0
        if frame is None:
            sys.exit("error: could not decode any frame")

        cv2.imshow(WINDOW, overlay(frame, idx, n_frames, session, playing, dirty, message))
        key = cv2.waitKeyEx(15 if playing else 40)
        if key == -1:
            continue
        if key not in (ord("q"), 27):
            quit_armed = False

        try:
            if key in (ord("q"), 27):
                if dirty and not quit_armed:
                    quit_armed = True
                    message = "unsaved edits — press q again to quit anyway, or s to save"
                    continue
                break
            elif key == ord(" "):
                playing = not playing
            elif key == ord("."):
                idx = min(idx + 1, n_frames - 1)
            elif key == ord(","):
                idx = max(idx - 1, 0)
            elif key == ord("]"):
                idx = min(idx + 30, n_frames - 1)
            elif key == ord("["):
                idx = max(idx - 30, 0)
            elif key == ord("}"):
                idx = min(idx + 300, n_frames - 1)
            elif key == ord("{"):
                idx = max(idx - 300, 0)
            elif key == ord("r"):
                session.mark_start(idx)
                dirty = True
                message = f"rally opened at frame {idx}"
            elif key == ord("e"):
                session.mark_end(idx)
                dirty = True
                message = f"rally closed at frame {idx}"
            elif key == ord("c"):
                session.cancel()
                message = "open rally cancelled"
            elif key == ord("n"):
                session.mark_hit(idx, "near")
                dirty = True
                message = f"near-side hit at frame {idx}"
            elif key == ord("f"):
                session.mark_hit(idx, "far")
                dirty = True
                message = f"far-side hit at frame {idx}"
            elif key == ord("u"):
                undone = session.undo_pending_hit()
                message = f"removed pending hit at frame {undone.frame}"
            elif key == ord("d"):
                session.rallies = remove_rally(session.rallies, idx)
                dirty = True
                message = f"deleted rally containing frame {idx}"
            elif key == ord("x"):
                session.rallies = remove_hit(session.rallies, idx)
                dirty = True
                message = f"deleted hit at frame {idx}"
            elif key == ord("s"):
                if session.is_open:
                    raise ValueError(
                        f"rally open since frame {session.pending_start} — "
                        "mark its end ('e') or cancel ('c') before saving"
                    )
                match = merge_rallies(match, session.rallies)
                save_golden_match(match, args.golden)
                dirty = False
                message = f"saved {len(session.rallies)} rallies -> {args.golden}"
        except ValueError as exc:
            message = str(exc)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

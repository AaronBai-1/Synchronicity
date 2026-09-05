"""Golden-set rally-boundary + hit-frame labeling tool (docs/plan.md Phase 0: hand-label
rally boundaries and frame-exact hit frames — the S1 rally-segmentation and S5
hit-detection ground truth; risk #1 says hit labels come first).

Keyboard-driven video scrubber: mark rally start/end frames and per-hit frames (near or
far side) while playing/stepping the mezzanine video, then save into the golden JSON.

Audio-proposed hit candidates (--onsets, default auto): racket impacts are sharp
broadband transients, so perception/audio_onsets.py proposes candidate hit frames from
the soundtrack and o/O jumps between them — hit labeling becomes confirm-or-reject
(jump, check the frame, press n/f) instead of scrub-and-hunt. Proposals are jump
targets ONLY; nothing is written to the golden JSON without a keypress (plan trust
rule: every automated output is human-reviewed before it becomes ground truth). The
canonical mezzanine is encoded without audio, so pass the S0 audio artifact (or any
WAV/video with the same timeline) via --audio; extraction failures just disable onsets.

Keys:
    space        play / pause
    , / .        step 1 frame back / forward
    [ / ]        jump 30 frames back / forward
    { / }        jump 300 frames back / forward
    o / O        jump to the next / previous audio-proposed onset
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
        --match-id 2024_ao_final --video-uri r2://matches/2024_ao_final.mp4 \\
        --audio data/runs/2024_ao_final/s0_ingest/audio.wav

All decision logic lives in synchro_pipeline.eval.labeling and
synchro_pipeline.perception.audio_onsets (unit-tested, GUI-free); this file is only
the OpenCV event loop, which is deliberately untested.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

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
from synchro_pipeline.perception.audio_onsets import (
    AudioExtractionError,
    OnsetEvent,
    detect_onsets,
    extract_audio,
    next_onset,
    onset_near,
    prev_onset,
)

WINDOW = "label_rallies"


def load_onsets(video: str, audio: str | None, fps: float) -> list[OnsetEvent] | None:
    """Best-effort onset proposals: from --audio if given, else extracted from --video.

    Returns None when onsets are unavailable (no/undecodable audio, no ffmpeg) — the
    tool prints why and labeling continues exactly as before; onsets are an
    accelerator, never a prerequisite.
    """
    try:
        if audio is not None:
            if audio.lower().endswith(".wav"):
                return detect_onsets(audio, fps)
            with tempfile.TemporaryDirectory() as tmp:  # --audio is itself a video
                return detect_onsets(extract_audio(audio, Path(tmp) / "audio.wav"), fps)
        with tempfile.TemporaryDirectory() as tmp:
            return detect_onsets(extract_audio(video, Path(tmp) / "audio.wav"), fps)
    except (AudioExtractionError, ValueError, OSError) as exc:
        print(f"notice: onset proposals disabled — {exc}")
        return None


def overlay(
    frame: np.ndarray,
    idx: int,
    n_frames: int,
    session: RallyLabelSession,
    playing: bool,
    dirty: bool,
    message: str,
    onsets: list[OnsetEvent] | None,
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
        )
        + (f"  onsets: {len(onsets)} proposed" if onsets is not None else ""),
        (
            f"inside rally [{current.start_frame}, {current.end_frame}]"
            f" ({len(current.hits)} hits)"
            if current
            else "outside any rally"
        ),
        "space=play ,/.=step [/]=30 {/}=300 "
        + ("o/O=onset " if onsets is not None else "")
        + "r=start e=end c=cancel n/f=hit u=undo d/x=del s=save q=quit",
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
    if onsets:
        hit = onset_near(onsets, idx)  # +-2 frames: audio->frame rounding is inexact
        if hit is not None:
            centre = (img.shape[1] - 30, 30)
            cv2.circle(img, centre, 10, (0, 200, 255), -1)
            cv2.putText(
                img, f"onset {hit.frame}", (img.shape[1] - 175, 36),
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
    parser.add_argument(
        "--onsets",
        choices=("auto", "off"),
        default="auto",
        help="propose candidate hit frames from audio onsets (o/O jump keys); "
        "'auto' extracts audio from --video (or uses --audio) and degrades to 'off' "
        "with a notice when no usable audio exists",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="audio source for --onsets auto instead of extracting from --video: the S0 "
        "audio.wav artifact or a video sharing the mezzanine timeline (the canonical "
        "mezzanine itself is encoded with -an, so it has no audio to extract)",
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

    onsets: list[OnsetEvent] | None = None
    if args.onsets == "auto":
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0  # mezzanine contract is 30 CFR
        onsets = load_onsets(args.video, args.audio, fps)
        if onsets is not None:
            print(f"onsets: {len(onsets)} proposed hit candidates (o/O to jump)")

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

        cv2.imshow(
            WINDOW, overlay(frame, idx, n_frames, session, playing, dirty, message, onsets)
        )
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
            elif key in (ord("o"), ord("O")):
                if not onsets:
                    message = "no onset proposals loaded (--onsets auto needs usable audio)"
                    continue
                target = next_onset(onsets, idx) if key == ord("o") else prev_onset(onsets, idx)
                if target is None:
                    message = f"no proposed onset {'after' if key == ord('o') else 'before'} here"
                else:
                    idx = min(max(target.frame, 0), n_frames - 1)
                    message = (
                        f"onset at frame {target.frame} (strength {target.strength:.1f}) "
                        "— n/f to confirm as a hit"
                    )
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

"""TrackNetV3 shuttle-tracker wrapper (plan stage S4 — MIT-licensed, pretrained on
broadcast badminton, ~97.5% reported accuracy).

Integration model
-----------------
The upstream repo (https://github.com/qaz812345/TrackNetV3) is *vendored*, not a package
dependency: run pipeline/scripts/vendor_tracknetv3.sh to clone it (git-ignored) into
pipeline/vendor/TrackNetV3, then download its pretrained checkpoints manually per the
vendor README and point SYNCHRO_TRACKNET_WEIGHTS at the TrackNet checkpoint.

WHY vendor + subprocess instead of importing its modules: the repo is research code with
no package layout or stable API; shelling out to its predict.py and consuming the
documented ball CSV (Frame,Visibility,X,Y columns) couples us only to a file format.
It also keeps torch entirely out of this process — torch is an optional `ml` extra
(docs/plan.md license/deps posture), so everything heavy is checked lazily inside
track() and reported as a RuntimeError with exact setup steps, never an ImportError
at module import time.

Visibility mapping (plan S4 contract: detected|inpainted|missing)
-----------------------------------------------------------------
TrackNetV3 = a TrackNet detection network + an InpaintNet "rectification" module that
fills gaps in the raw trajectory. We run predict.py twice — once with the TrackNet
checkpoint only (raw detections), once adding the InpaintNet checkpoint (rectified
trajectory) — and merge:

    * frame present in the raw detections          -> visibility="detected"  (raw x, y)
    * frame filled only by the rectified pass      -> visibility="inpainted" (rectified x, y)
    * frame absent from both                       -> visibility="missing"   (x=y=0, conf=0)

i.e. TrackNetV3's rectification output *is* our "inpainted" state: positions the model
asserts without having detected the shuttle in that frame. The two-pass cost is accepted
for Phase 0 benchmarking; a single-pass in-process integration can replace it later
without touching the ShuttleTracker contract.

The predict CSV carries no per-frame confidence, so conf is a documented heuristic:
detected=1.0, inpainted=0.5, missing=0.0. Downstream code must treat these as tiers,
not calibrated probabilities.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from synchro_pipeline.perception.base import ShuttleTrackPoint

_PIPELINE_DIR = Path(__file__).resolve().parents[3]  # .../pipeline
DEFAULT_VENDOR_DIR = _PIPELINE_DIR / "vendor" / "TrackNetV3"
WEIGHTS_ENV = "SYNCHRO_TRACKNET_WEIGHTS"
INPAINT_WEIGHTS_ENV = "SYNCHRO_TRACKNET_INPAINT_WEIGHTS"

_DETECTED_CONF = 1.0
_INPAINTED_CONF = 0.5


def _setup_help(vendor_dir: Path) -> str:
    """The one true setup message — every failure mode points at the same steps."""
    return (
        "TrackNetV3 setup steps:\n"
        f"  1. vendor the repo:   sh pipeline/scripts/vendor_tracknetv3.sh   (-> {vendor_dir})\n"
        "  2. install torch:     uv sync --all-packages --extra ml\n"
        "  3. download the pretrained checkpoints (TrackNet_best.pt, InpaintNet_best.pt)\n"
        f"     per {vendor_dir}/README.md — they are NOT auto-downloaded\n"
        f"  4. export {WEIGHTS_ENV}=/path/to/TrackNet_best.pt\n"
        f"     (optional: {INPAINT_WEIGHTS_ENV}=/path/to/InpaintNet_best.pt; defaults to a\n"
        "     sibling InpaintNet_best.pt next to the TrackNet checkpoint if present)"
    )


def _parse_ball_csv(csv_path: Path) -> dict[int, tuple[float, float] | None]:
    """Parse a TrackNetV3 ball CSV into {frame: (x, y) | None-when-invisible}.

    The vendor format is Frame,Visibility,X,Y (header names matched case-insensitively so
    minor upstream renames don't break us). Visibility 0 means "no shuttle this frame".
    """
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise RuntimeError(f"TrackNetV3 output CSV is empty: {csv_path}")
        cols = {name.strip().lower(): name for name in reader.fieldnames}
        missing = {"frame", "visibility", "x", "y"} - cols.keys()
        if missing:
            raise RuntimeError(
                f"TrackNetV3 output CSV {csv_path} lacks expected columns {sorted(missing)} "
                f"(found {reader.fieldnames}); the vendored predict.py output format may have "
                "changed — check the vendor README."
            )
        out: dict[int, tuple[float, float] | None] = {}
        for row in reader:
            frame = int(float(row[cols["frame"]]))
            visible = int(float(row[cols["visibility"]])) != 0
            out[frame] = (
                (float(row[cols["x"]]), float(row[cols["y"]])) if visible else None
            )
        return out


def _merge_tracks(
    raw: dict[int, tuple[float, float] | None],
    rectified: dict[int, tuple[float, float] | None] | None,
) -> list[ShuttleTrackPoint]:
    """Fuse raw-detection and rectified passes into the protocol's visibility states.

    See the module docstring for the mapping rationale. Pure so it stays unit-testable
    without torch or the vendored repo.
    """
    frames = sorted(set(raw) | set(rectified or {}))
    points: list[ShuttleTrackPoint] = []
    for frame in frames:
        raw_xy = raw.get(frame)
        rect_xy = (rectified or {}).get(frame)
        if raw_xy is not None:
            points.append(
                ShuttleTrackPoint(
                    frame_idx=frame,
                    x=raw_xy[0],
                    y=raw_xy[1],
                    conf=_DETECTED_CONF,
                    visibility="detected",
                )
            )
        elif rect_xy is not None:
            points.append(
                ShuttleTrackPoint(
                    frame_idx=frame,
                    x=rect_xy[0],
                    y=rect_xy[1],
                    conf=_INPAINTED_CONF,
                    visibility="inpainted",
                )
            )
        else:
            points.append(
                ShuttleTrackPoint(
                    frame_idx=frame, x=0.0, y=0.0, conf=0.0, visibility="missing"
                )
            )
    return points


class TrackNetV3Tracker:
    """ShuttleTracker backed by the vendored TrackNetV3 repo.

    Construction is always safe (no heavy imports, no filesystem checks) so the
    benchmark CLI can list/select it offline; every requirement is verified inside
    track() with a RuntimeError carrying precise setup instructions.
    """

    name = "tracknetv3"

    def __init__(
        self,
        weights_path: str | Path | None = None,
        inpaint_weights_path: str | Path | None = None,
        vendor_dir: str | Path | None = None,
    ) -> None:
        self._weights_arg = weights_path
        self._inpaint_arg = inpaint_weights_path
        self._vendor_dir = Path(vendor_dir) if vendor_dir is not None else DEFAULT_VENDOR_DIR

    def track(
        self, video_path: str | Path, frames: Sequence[int] | None = None
    ) -> list[ShuttleTrackPoint]:
        """See ShuttleTracker.track.

        Note: predict.py always processes the whole video; `frames` filters the result
        afterwards. Fine for Phase 0 benchmarking (golden matches are processed whole);
        revisit when rally-gated S4 lands.
        """
        vendor = self._require_vendor()
        self._require_torch()
        weights, inpaint = self._require_weights()
        video = Path(video_path)
        if not video.is_file():
            raise RuntimeError(f"video not found: {video}")

        with tempfile.TemporaryDirectory(prefix="tracknetv3-") as tmp:
            raw_csv = self._run_predict(vendor, video, weights, None, Path(tmp) / "raw")
            raw = _parse_ball_csv(raw_csv)
            rectified: dict[int, tuple[float, float] | None] | None = None
            if inpaint is not None:
                rect_csv = self._run_predict(
                    vendor, video, weights, inpaint, Path(tmp) / "rectified"
                )
                rectified = _parse_ball_csv(rect_csv)

        points = _merge_tracks(raw, rectified)
        if frames is not None:
            wanted = {int(f) for f in frames}
            points = [p for p in points if p.frame_idx in wanted]
        return points

    # -- requirement checks (lazy, with actionable errors) ----------------------

    def _require_vendor(self) -> Path:
        if not (self._vendor_dir / "predict.py").is_file():
            raise RuntimeError(
                f"TrackNetV3 is not vendored at {self._vendor_dir} (predict.py missing).\n"
                + _setup_help(self._vendor_dir)
            )
        return self._vendor_dir

    def _require_torch(self) -> None:
        # Lazy: torch is an optional extra and must never be imported at module level.
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            raise RuntimeError(
                "torch is not installed (optional `ml` extra) — TrackNetV3 needs it.\n"
                + _setup_help(self._vendor_dir)
            )

    def _require_weights(self) -> tuple[Path, Path | None]:
        raw = self._weights_arg or os.environ.get(WEIGHTS_ENV)
        if not raw:
            raise RuntimeError(
                f"no TrackNetV3 weights: pass weights_path or set {WEIGHTS_ENV}.\n"
                + _setup_help(self._vendor_dir)
            )
        weights = Path(raw)
        if not weights.is_file():
            raise RuntimeError(
                f"TrackNetV3 weights not found at {weights}.\n" + _setup_help(self._vendor_dir)
            )
        inpaint_raw = self._inpaint_arg or os.environ.get(INPAINT_WEIGHTS_ENV)
        if inpaint_raw:
            inpaint = Path(inpaint_raw)
            if not inpaint.is_file():
                raise RuntimeError(
                    f"InpaintNet weights not found at {inpaint}.\n"
                    + _setup_help(self._vendor_dir)
                )
            return weights, inpaint
        sibling = weights.with_name("InpaintNet_best.pt")
        # No inpaint checkpoint is allowed — the tracker then never emits "inpainted".
        return weights, sibling if sibling.is_file() else None

    # -- subprocess plumbing -----------------------------------------------------

    def _run_predict(
        self,
        vendor: Path,
        video: Path,
        weights: Path,
        inpaint: Path | None,
        save_dir: Path,
    ) -> Path:
        """Run vendored predict.py, return the ball CSV it wrote."""
        save_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(vendor / "predict.py"),
            "--video_file",
            str(video),
            "--tracknet_file",
            str(weights),
            "--save_dir",
            str(save_dir),
        ]
        if inpaint is not None:
            cmd += ["--inpaintnet_file", str(inpaint)]
        proc = subprocess.run(cmd, cwd=vendor, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"TrackNetV3 predict.py failed (exit {proc.returncode}).\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stderr tail:\n{proc.stderr[-2000:]}\n"
                f"If its CLI changed upstream, check {vendor}/README.md."
            )
        preferred = save_dir / f"{video.stem}_ball.csv"
        if preferred.is_file():
            return preferred
        candidates = sorted(save_dir.rglob("*.csv"))
        if len(candidates) == 1:
            return candidates[0]
        raise RuntimeError(
            f"could not identify TrackNetV3's ball CSV in {save_dir} "
            f"(expected {preferred.name}, found {[c.name for c in candidates]}); "
            f"the vendored predict.py output layout may have changed — check {vendor}/README.md."
        )

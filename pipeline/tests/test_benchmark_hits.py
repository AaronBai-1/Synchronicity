"""End-to-end tests for the trajectory-only hit-detection benchmark
(eval/benchmark_hits.py).

The `--tracker golden` source replays the golden file's own hand-labelled shuttle
points as the trajectory, so the baseline's event logic can be scored under perfect
tracking with no weights, GPU or footage — the plan-risk-#1 "gap to fused" number.
"""

import json
import subprocess
import sys

from synchro_pipeline.eval.benchmark_hits import main
from synchro_pipeline.eval.golden import (
    GoldenHit,
    GoldenMatch,
    GoldenRally,
    GoldenShuttlePoint,
    save_golden_match,
    sha256_file,
)


def shuttle_line(start_frame, start_xy, segments):
    """GoldenShuttlePoints along a piecewise-linear path of (n_frames, vx, vy)."""
    x, y = start_xy
    frame = start_frame
    points = [GoldenShuttlePoint(frame=frame, x=x, y=y, visible=True)]
    for n_frames, vx, vy in segments:
        for _ in range(n_frames):
            frame += 1
            x += vx
            y += vy
            points.append(GoldenShuttlePoint(frame=frame, x=x, y=y, visible=True))
    return points


def golden_for_hits() -> GoldenMatch:
    """Three rallies covering the honest-abstention matrix.

    rally 1: hit labels + a matching synthetic shuttle track (reversal at each hit,
             vertical direction consistent with the labelled side) -> fully scored.
    rally 2: hit labels but no shuttle labels -> no trajectory under --tracker golden.
    rally 3: shuttle labels (straight drift, no hits) but no hit labels -> never scored.
    """
    rally1_shuttle = shuttle_line(
        100, (400.0, 290.0), [(30, 2, 12), (35, 2, -13), (35, 2, 13), (30, 2, -12)]
    )
    rally3_shuttle = shuttle_line(400, (300.0, 350.0), [(60, 5, 1.5)])
    return GoldenMatch(
        match_id="hits-bench",
        video_uri="local/hits-bench.mp4",
        rallies=[
            GoldenRally(
                start_frame=100,
                end_frame=230,
                hits=[
                    GoldenHit(frame=130, side="near"),
                    GoldenHit(frame=165, side="far"),
                    GoldenHit(frame=200, side="near"),
                ],
                shuttle=rally1_shuttle,
            ),
            GoldenRally(
                start_frame=300,
                end_frame=380,
                hits=[GoldenHit(frame=320, side="near")],
            ),
            GoldenRally(start_frame=400, end_frame=460, shuttle=rally3_shuttle),
        ],
    )


def run_cli(tmp_path, match: GoldenMatch, tracker: str = "golden", extra=()) -> tuple[int, dict]:
    golden_path = tmp_path / "golden.json"
    save_golden_match(match, golden_path)
    out = tmp_path / "report.json"
    rc = main(
        [
            "--video", str(tmp_path / "missing.mp4"),
            "--golden", str(golden_path),
            "--tracker", tracker,
            "--out", str(out),
            *extra,
        ]
    )
    report = json.loads(out.read_text()) if out.is_file() else {}
    return rc, report


class TestGoldenTrackerEndToEnd:
    def test_scores_and_abstains_per_rally(self, tmp_path, capsys):
        rc, report = run_cli(tmp_path, golden_for_hits())
        assert rc == 0
        assert report["tracker"] == "golden"
        assert report["tolerance_frames"] == 3
        assert report["n_rallies"] == 3
        assert report["n_rallies_with_hit_labels"] == 2
        assert report["n_rallies_scored"] == 1

        scored, no_traj, no_labels = report["per_rally"]

        # rally 1: perfect track + clean reversals -> perfect event F1 and sides
        assert scored["has_trajectory"] and scored["has_hit_labels"]
        assert scored["n_gt_hits"] == 3
        metrics = scored["metrics"]
        assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (3, 0, 0)
        assert metrics["f1"] == 1.0
        side = scored["side"]
        assert side["n_matched"] == 3 and side["n_sided"] == 3
        assert side["accuracy"] == 1.0

        # rally 2: hits labelled but no shuttle labels -> null metrics, not zeros
        assert no_traj["has_trajectory"] is False and no_traj["has_hit_labels"] is True
        assert no_traj["metrics"] is None and no_traj["n_proposed"] is None

        # rally 3: trajectory but no hit labels -> proposals counted, never scored
        assert no_labels["has_trajectory"] is True and no_labels["has_hit_labels"] is False
        assert no_labels["n_proposed"] == 0  # straight drift proposes nothing
        assert no_labels["metrics"] is None

        overall = report["hit_metrics"]
        assert overall["f1"] == 1.0
        assert overall["side"]["accuracy"] == 1.0
        assert overall["side"]["n_sided"] == 3

        stdout = capsys.readouterr().out
        assert "overall" in stdout and "hits-bench" in stdout

    def test_no_scorable_rallies_reports_null_overall(self, tmp_path):
        match = GoldenMatch(
            match_id="empty",
            video_uri="v",
            rallies=[GoldenRally(start_frame=0, end_frame=50)],
        )
        rc, report = run_cli(tmp_path, match)
        assert rc == 0
        assert report["hit_metrics"] is None
        assert report["n_rallies_scored"] == 0

    def test_runnable_as_python_module(self, tmp_path):
        golden_path = tmp_path / "golden.json"
        save_golden_match(golden_for_hits(), golden_path)
        out = tmp_path / "report.json"
        proc = subprocess.run(
            [
                sys.executable, "-m", "synchro_pipeline.eval.benchmark_hits",
                "--video", str(tmp_path / "missing.mp4"),
                "--golden", str(golden_path),
                "--tracker", "golden",
                "--out", str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert json.loads(out.read_text())["tracker"] == "golden"


class TestOtherTrackers:
    def test_fake_tracker_plumbing(self, tmp_path):
        # The fake tracker's trajectory has nothing to do with the labels; the point
        # is that the whole path runs offline and still reports honest numbers.
        rc, report = run_cli(tmp_path, golden_for_hits(), tracker="fake")
        assert rc == 0
        assert report["tracker"] == "fake"
        scored = report["per_rally"][0]
        assert scored["has_trajectory"] is True
        assert scored["metrics"] is not None
        assert set(scored["metrics"]) >= {"precision", "recall", "f1", "tp", "fp", "fn"}
        # under a real (fake) tracker, rally 2 has a trajectory too and is scored
        assert report["n_rallies_scored"] == 2


class TestFailureModes:
    def test_bad_golden_file_exit_2(self, tmp_path, capsys):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        rc = main(
            [
                "--video", "x.mp4",
                "--golden", str(bad),
                "--tracker", "golden",
                "--out", str(tmp_path / "r.json"),
            ]
        )
        assert rc == 2
        assert "not valid JSON" in capsys.readouterr().err


class TestVideoHashGate:
    def _argv(self, tmp_path, stamp_correct: bool, extra=()):
        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake-mezzanine-bytes")
        match = golden_for_hits().model_copy(
            update={"video_sha256": sha256_file(video) if stamp_correct else "0" * 64}
        )
        golden_path = tmp_path / "golden.json"
        save_golden_match(match, golden_path)
        return [
            "--video", str(video),
            "--golden", str(golden_path),
            "--tracker", "golden",
            "--out", str(tmp_path / "r.json"),
            *extra,
        ]

    def test_matching_hash_scores(self, tmp_path):
        assert main(self._argv(tmp_path, stamp_correct=True)) == 0

    def test_mismatch_refuses_with_exit_4(self, tmp_path, capsys):
        rc = main(self._argv(tmp_path, stamp_correct=False))
        assert rc == 4
        assert "not the encode" in capsys.readouterr().err

    def test_skip_flag_overrides(self, tmp_path):
        assert main(self._argv(tmp_path, stamp_correct=False, extra=("--skip-hash-check",))) == 0

    def test_unstamped_golden_warns_but_scores(self, tmp_path, capsys):
        rc, _ = run_cli(tmp_path, golden_for_hits())
        assert rc == 0
        assert "no video_sha256" in capsys.readouterr().err

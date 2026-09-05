"""End-to-end benchmark harness test with the fake tracker (eval/benchmark_shuttle.py).

Proves the plan's Phase 0 plumbing claim: the shuttle benchmark runs offline with no
weights, no GPU and no real footage, and its JSON report has the shape CI will consume.
"""

import json
import subprocess
import sys

import pytest
from synchro_pipeline.eval.benchmark_shuttle import main, rally_frames
from synchro_pipeline.eval.golden import (
    GoldenMatch,
    GoldenRally,
    GoldenShuttlePoint,
    save_golden_match,
    sha256_file,
)
from synchro_pipeline.perception.base import FakeShuttleTracker, ShuttleTracker


def golden_with_labels() -> GoldenMatch:
    shuttle = [
        GoldenShuttlePoint(frame=f, x=100.0 + f, y=300.0, visible=True) for f in range(100, 115)
    ] + [GoldenShuttlePoint(frame=115, visible=False)]
    return GoldenMatch(
        match_id="bench",
        video_uri="local/bench.mp4",
        rallies=[
            GoldenRally(start_frame=100, end_frame=220, shuttle=shuttle),
            GoldenRally(start_frame=300, end_frame=380),
        ],
    )


class TestFakeTracker:
    def test_is_a_shuttle_tracker(self):
        assert isinstance(FakeShuttleTracker(), ShuttleTracker)

    def test_deterministic_and_subset_consistent(self):
        tracker = FakeShuttleTracker(seed=42)
        full = tracker.track("does-not-exist.mp4", frames=range(0, 50))
        again = tracker.track("does-not-exist.mp4", frames=range(0, 50))
        assert full == again
        subset = tracker.track("does-not-exist.mp4", frames=[5, 6, 7])
        assert subset == [p for p in full if p.frame_idx in (5, 6, 7)]

    def test_missing_video_falls_back_to_default_frame_count(self):
        tracker = FakeShuttleTracker(seed=0, default_n_frames=25)
        points = tracker.track("does-not-exist.mp4")
        assert [p.frame_idx for p in points] == list(range(25))

    def test_emits_all_three_visibility_states(self):
        points = FakeShuttleTracker(seed=1).track("x.mp4", frames=range(0, 500))
        states = {p.visibility for p in points}
        assert states == {"detected", "inpainted", "missing"}
        for p in points:
            if p.visibility == "missing":
                assert p.conf == 0.0


def test_rally_frames_union():
    match = golden_with_labels()
    frames = rally_frames(match)
    assert frames[0] == 100 and frames[-1] == 380
    assert len(frames) == 121 + 81
    assert 250 not in frames


class TestBenchmarkCli:
    def run_cli(self, tmp_path, match: GoldenMatch, seed: int = 1) -> dict:
        golden_path = tmp_path / "golden.json"
        save_golden_match(match, golden_path)
        out = tmp_path / "report.json"
        rc = main(
            [
                "--video", str(tmp_path / "missing.mp4"),
                "--golden", str(golden_path),
                "--tracker", "fake",
                "--seed", str(seed),
                "--out", str(out),
            ]
        )
        assert rc == 0
        return json.loads(out.read_text())

    def test_end_to_end_with_labels(self, tmp_path, capsys):
        report = self.run_cli(tmp_path, golden_with_labels())
        assert report["match_id"] == "bench"
        assert report["tracker"] == "fake"
        assert report["n_rallies"] == 2
        assert report["n_rallies_with_shuttle_labels"] == 1
        assert report["n_frames_tracked"] == 121 + 81

        cov = report["coverage"]
        assert cov["detected"] + cov["inpainted"] + cov["missing"] == cov["n_frames"]

        metrics = report["shuttle_metrics"]
        assert metrics is not None
        assert metrics["n_labeled_frames"] == 16
        assert set(metrics) >= {"precision", "recall", "f1", "tp", "fp", "fn"}

        labelled, unlabelled = report["per_rally"]
        assert labelled["has_shuttle_labels"] is True and labelled["metrics"] is not None
        assert unlabelled["has_shuttle_labels"] is False and unlabelled["metrics"] is None

        table = capsys.readouterr().out
        assert "overall" in table and "bench" in table

    def test_no_labels_reports_coverage_only(self, tmp_path):
        match = GoldenMatch(
            match_id="cov",
            video_uri="v",
            rallies=[GoldenRally(start_frame=0, end_frame=49)],
        )
        report = self.run_cli(tmp_path, match)
        assert report["shuttle_metrics"] is None
        assert report["coverage"]["n_frames"] == 50

    def test_same_seed_is_reproducible(self, tmp_path):
        a = self.run_cli(tmp_path / "a", golden_with_labels(), seed=7)
        b = self.run_cli(tmp_path / "b", golden_with_labels(), seed=7)
        for key in ("coverage", "shuttle_metrics", "per_rally"):
            assert a[key] == b[key]

    def test_bad_golden_file_exit_code(self, tmp_path, capsys):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        rc = main(
            [
                "--video", "x.mp4",
                "--golden", str(bad),
                "--tracker", "fake",
                "--out", str(tmp_path / "r.json"),
            ]
        )
        assert rc == 2
        assert "not valid JSON" in capsys.readouterr().err

    @pytest.mark.parametrize("module", ["synchro_pipeline.eval.benchmark_shuttle"])
    def test_runnable_as_python_module(self, tmp_path, module):
        golden_path = tmp_path / "golden.json"
        save_golden_match(golden_with_labels(), golden_path)
        out = tmp_path / "report.json"
        proc = subprocess.run(
            [
                sys.executable, "-m", module,
                "--video", str(tmp_path / "missing.mp4"),
                "--golden", str(golden_path),
                "--tracker", "fake",
                "--out", str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert out.is_file()
        assert json.loads(out.read_text())["tracker"] == "fake"


class TestVideoHashGate:
    """Exit-code-4 gate: never score labels against the wrong encode."""

    def _setup(self, tmp_path, stamp_correct: bool):
        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake-mezzanine-bytes")
        match = golden_with_labels()
        recorded = (
            sha256_file(video) if stamp_correct else "0" * 64
        )
        match = match.model_copy(update={"video_sha256": recorded})
        golden_path = tmp_path / "golden.json"
        save_golden_match(match, golden_path)
        out = tmp_path / "report.json"
        return [
            "--video", str(video),
            "--golden", str(golden_path),
            "--tracker", "fake",
            "--out", str(out),
        ]

    def test_matching_hash_scores(self, tmp_path):
        assert main(self._setup(tmp_path, stamp_correct=True)) == 0

    def test_mismatch_refuses_with_exit_4(self, tmp_path, capsys):
        rc = main(self._setup(tmp_path, stamp_correct=False))
        assert rc == 4
        assert "not the encode" in capsys.readouterr().err

    def test_skip_flag_overrides(self, tmp_path):
        argv = self._setup(tmp_path, stamp_correct=False) + ["--skip-hash-check"]
        assert main(argv) == 0

    def test_unstamped_golden_warns_but_scores(self, tmp_path, capsys):
        video = tmp_path / "v.mp4"
        video.write_bytes(b"whatever")
        golden_path = tmp_path / "golden.json"
        save_golden_match(golden_with_labels(), golden_path)
        rc = main(
            ["--video", str(video), "--golden", str(golden_path),
             "--tracker", "fake", "--out", str(tmp_path / "r.json")]
        )
        assert rc == 0
        assert "no video_sha256" in capsys.readouterr().err

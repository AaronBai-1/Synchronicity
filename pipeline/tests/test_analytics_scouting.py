"""Scouting-text tests: deterministic templates, embedded n, low-sample gating."""

from __future__ import annotations

import pandas as pd
import pytest
from synchro_pipeline.analytics import (
    build_rally_frame,
    build_recommendations,
    build_scouting_report,
    build_shot_frame,
)
from test_analytics_fixtures import build_synthetic_match


@pytest.fixture(scope="module")
def frames():
    shots, rallies = build_synthetic_match()
    shot_df, _ = build_shot_frame(shots, rallies)
    rally_df, _ = build_rally_frame(rallies)
    return shot_df, rally_df


class TestRecommendations:
    def test_gated_recommendations_at_min_n_4(self, frames):
        shot_df, rally_df = frames
        recs = build_recommendations(shot_df, rally_df, "A", min_n=4)
        # transition rec (row_total 6) and winning finish (n=5) qualify; the error
        # source (n=1) must be omitted rather than asserted from thin air.
        assert len(recs) == 2
        assert "'serve'" in recs[0]
        assert "'lift'" in recs[0]
        assert "n=6" in recs[0]  # probability denominator
        assert "n=5 decided" in recs[0]  # win-rate denominator
        assert "'smash'" in recs[1]
        assert "rear_left" in recs[1]
        assert "n=5" in recs[1]
        assert not any("Reduce risk" in rec for rec in recs)

    def test_error_recommendation_appears_at_min_n_1(self, frames):
        shot_df, rally_df = frames
        recs = build_recommendations(shot_df, rally_df, "A", min_n=1)
        assert len(recs) == 3
        error_rec = next(rec for rec in recs if "Reduce risk" in rec)
        assert "'net'" in error_rec
        assert "front_left" in error_rec
        assert "n=1" in error_rec

    def test_every_claim_embeds_its_n(self, frames):
        shot_df, rally_df = frames
        for min_n in (1, 4):
            for rec in build_recommendations(shot_df, rally_df, "A", min_n=min_n):
                assert "n=" in rec

    def test_nothing_clears_a_high_bar(self, frames):
        shot_df, rally_df = frames
        assert build_recommendations(shot_df, rally_df, "A", min_n=50) == []

    def test_deterministic(self, frames):
        shot_df, rally_df = frames
        first = build_recommendations(shot_df, rally_df, "A", min_n=4)
        second = build_recommendations(shot_df, rally_df, "A", min_n=4)
        assert first == second

    def test_empty_input(self):
        assert build_recommendations(pd.DataFrame(), pd.DataFrame(), "A") == []


class TestScoutingReport:
    def test_report_with_adequate_samples(self, frames):
        shot_df, rally_df = frames
        report = build_scouting_report(
            shot_df, rally_df, "A", min_n=4, pattern_length=2, min_support=2
        )
        assert "# Scouting report — player A" in report
        assert "n=22 shots" in report
        assert "n=10 rallies" in report
        assert "smash: 32% of shots (n=7)" in report
        assert "'serve -> smash' occurred n=5 times" in report
        assert "80% vs 58% baseline" in report
        assert "lift 1.37" in report
        assert "Lean into 'serve'" in report

    def test_report_pressure_sections_stay_honest(self, frames):
        shot_df, rally_df = frames
        report = build_scouting_report(
            shot_df, rally_df, "A", min_n=4, pattern_length=2, min_support=2
        )
        # n=1 pressure contexts must be marked insufficient, not asserted as findings
        assert "game_point_for: insufficient data (n=1)." in report

    def test_default_gates_produce_insufficient_data_not_claims(self, frames):
        shot_df, rally_df = frames
        report = build_scouting_report(shot_df, rally_df, "A")  # min_n=15, support=10
        assert "insufficient data" in report
        assert "serve -> smash" not in report
        assert "Lean into" not in report

    def test_every_numeric_claim_embeds_n(self, frames):
        shot_df, rally_df = frames
        report = build_scouting_report(
            shot_df, rally_df, "A", min_n=4, pattern_length=2, min_support=2
        )
        for line in report.splitlines():
            if "%" in line:
                assert "n=" in line, line

    def test_deterministic(self, frames):
        shot_df, rally_df = frames
        kwargs = {"min_n": 4, "pattern_length": 2, "min_support": 2}
        assert build_scouting_report(shot_df, rally_df, "A", **kwargs) == build_scouting_report(
            shot_df, rally_df, "A", **kwargs
        )

    def test_empty_inputs_still_produce_a_gated_report(self):
        report = build_scouting_report(pd.DataFrame(), pd.DataFrame(), "A")
        assert "# Scouting report — player A" in report
        assert "n=0 shots" in report
        assert "insufficient data" in report
        assert "%" not in report  # no fabricated rates anywhere

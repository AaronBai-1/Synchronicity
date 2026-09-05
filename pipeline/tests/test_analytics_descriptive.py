"""Descriptive-analytics tests: hand-computed aggregates over the fixture match.

Every expected number is derived by hand from the rally table documented in
test_analytics_fixtures.py (A shots = 22, B shots = 20, A wins 6/10 rallies, ...).
"""

from __future__ import annotations

import pandas as pd
import pytest
from synchro_pipeline.analytics import (
    PRESSURE_CONTEXTS,
    build_rally_frame,
    build_shot_frame,
    conditional_shot_mix,
    error_profile,
    head_to_head,
    match_trend,
    momentum_summary,
    pressure_summary,
    serve_patterns,
    shot_distribution,
    terminal_conversion,
)
from test_analytics_fixtures import build_synthetic_match


@pytest.fixture(scope="module")
def frames():
    shots, rallies = build_synthetic_match()
    shot_df, _ = build_shot_frame(shots, rallies)
    rally_df, _ = build_rally_frame(rallies)
    return shot_df, rally_df


class TestShotDistribution:
    def test_hand_computed_counts_and_shares(self, frames):
        shot_df, _ = frames
        dist = shot_distribution(shot_df, "A")
        by_type = dist.set_index("shot_type")
        assert int(dist["n"].sum()) == 22
        assert int(by_type.loc["smash", "n"]) == 7
        assert int(by_type.loc["serve", "n"]) == 6
        assert int(by_type.loc["lift", "n"]) == 4
        assert int(by_type.loc["net", "n"]) == 3
        assert by_type.loc["smash", "share"] == pytest.approx(7 / 22)
        assert dist["share"].sum() == pytest.approx(1.0)
        assert dist.iloc[0]["shot_type"] == "smash"  # sorted by n desc

    def test_low_sample_flags(self, frames):
        shot_df, _ = frames
        assert shot_distribution(shot_df, "A")["low_sample"].all()  # min_n=15 default
        dist5 = shot_distribution(shot_df, "A", min_n=5).set_index("shot_type")
        assert not dist5.loc["smash", "low_sample"]
        assert not dist5.loc["serve", "low_sample"]
        assert dist5.loc["lift", "low_sample"]

    def test_empty_input(self):
        result = shot_distribution(pd.DataFrame(), "A")
        assert result.empty
        assert list(result.columns) == ["shot_type", "n", "share", "low_sample"]


class TestConditionalShotMix:
    def test_response_to_opposing_serve(self, frames):
        shot_df, _ = frames
        mix = conditional_shot_mix(shot_df, "A")
        serve_group = mix[mix["prev_shot_type"] == "serve"].set_index("shot_type")
        assert int(serve_group.loc["lift", "n"]) == 2
        assert serve_group.loc["lift", "share"] == pytest.approx(0.5)
        assert serve_group.loc["clear", "share"] == pytest.approx(0.25)
        assert serve_group.loc["net", "share"] == pytest.approx(0.25)

    def test_lift_always_answered_with_smash(self, frames):
        shot_df, _ = frames
        mix = conditional_shot_mix(shot_df, "A")
        lift_group = mix[mix["prev_shot_type"] == "lift"]
        assert len(lift_group) == 1
        assert lift_group.iloc[0]["shot_type"] == "smash"
        assert int(lift_group.iloc[0]["n"]) == 7
        assert lift_group.iloc[0]["share"] == pytest.approx(1.0)

    def test_total_pairs(self, frames):
        # opponent->A consecutive pairs per rally: 1+1+1+3+2+3+1+2+1+1 = 16
        shot_df, _ = frames
        assert int(conditional_shot_mix(shot_df, "A")["n"].sum()) == 16

    def test_empty_input(self):
        assert conditional_shot_mix(pd.DataFrame(), "A").empty


class TestServePatterns:
    def test_hand_computed_parity_groups(self, frames):
        _, rally_df = frames
        serves = serve_patterns(rally_df, "A").set_index(["parity", "serve_type"])
        assert int(serves["n"].sum()) == 6
        even_short = serves.loc[("even", "short")]
        assert int(even_short["n"]) == 3
        assert even_short["share"] == pytest.approx(1.0)
        assert even_short["win_rate"] == pytest.approx(2 / 3)
        odd_short = serves.loc[("odd", "short")]
        assert int(odd_short["n"]) == 2
        assert odd_short["share"] == pytest.approx(2 / 3)
        assert odd_short["win_rate"] == pytest.approx(0.5)
        odd_long = serves.loc[("odd", "long")]
        assert int(odd_long["n"]) == 1
        assert odd_long["win_rate"] == pytest.approx(1.0)

    def test_undecided_rallies_yield_none_not_zero(self):
        shots, rallies = build_synthetic_match()
        undecided = [r.model_copy(update={"winner": None}) for r in rallies]
        rally_df, _ = build_rally_frame(undecided)
        serves = serve_patterns(rally_df, "A")
        assert (serves["n_decided"] == 0).all()
        assert serves["win_rate"].isna().all()

    def test_empty_input(self):
        assert serve_patterns(pd.DataFrame(), "A").empty


class TestTerminalConversion:
    def test_hand_computed(self, frames):
        shot_df, _ = frames
        conv = terminal_conversion(shot_df, "A").set_index("shot_type")
        assert len(conv) == 2  # smash winners + one net error
        assert int(conv.loc["smash", "n"]) == 5
        assert conv.loc["smash", "winner_rate"] == pytest.approx(1.0)
        assert conv.loc["smash", "error_rate"] == pytest.approx(0.0)
        assert int(conv.loc["net", "n"]) == 1
        assert conv.loc["net", "error_rate"] == pytest.approx(1.0)

    def test_empty_input(self):
        assert terminal_conversion(pd.DataFrame(), "A").empty


class TestErrorProfile:
    def test_unforced_attribution_for_a(self, frames):
        shot_df, _ = frames
        errors = error_profile(shot_df, "A")
        assert len(errors) == 1
        row = errors.iloc[0]
        assert row["shot_type"] == "net"
        assert row["origin_zone"] == "front_left"
        assert row["target_zone"] == "unknown"  # error landing is unknown, not guessed
        assert (int(row["n"]), int(row["n_unforced"]), int(row["n_unattributed"])) == (1, 1, 0)
        assert bool(row["low_sample"])

    def test_undistinguished_error_for_b(self, frames):
        shot_df, _ = frames
        errors = error_profile(shot_df, "B")
        assert len(errors) == 1
        row = errors.iloc[0]
        assert row["shot_type"] == "drive"
        assert (int(row["n_forced"]), int(row["n_unforced"]), int(row["n_unattributed"])) == (
            0,
            0,
            1,
        )

    def test_empty_input(self):
        assert error_profile(pd.DataFrame(), "A").empty


class TestPressureSummary:
    def test_all_contexts_always_present(self, frames):
        _, rally_df = frames
        summary = pressure_summary(rally_df, "A")
        assert list(summary["context"]) == list(PRESSURE_CONTEXTS)
        assert int(summary["n"].sum()) == 10

    def test_hand_computed_contexts_for_a(self, frames):
        _, rally_df = frames
        rows = pressure_summary(rally_df, "A").set_index("context")
        # 21-20 with A serving: game point for A, won
        assert int(rows.loc["game_point_for", "n"]) == 1
        assert rows.loc["game_point_for", "win_rate"] == pytest.approx(1.0)
        # 19-20: B holds game point, A won the rally anyway
        assert int(rows.loc["game_point_against", "n"]) == 1
        assert rows.loc["game_point_against", "win_rate"] == pytest.approx(1.0)
        # 20-20 is deuce (tied >= 20); 19-19 is neutral
        assert int(rows.loc["deuce", "n"]) == 1
        assert int(rows.loc["neutral", "n"]) == 7
        assert rows.loc["neutral", "win_rate"] == pytest.approx(3 / 7)

    def test_empty_context_rate_is_none(self, frames):
        _, rally_df = frames
        rows = pressure_summary(rally_df, "A").set_index("context")
        assert int(rows.loc["game_point_both", "n"]) == 0
        assert pd.isna(rows.loc["game_point_both", "win_rate"])

    def test_perspective_swaps_for_b(self, frames):
        _, rally_df = frames
        rows = pressure_summary(rally_df, "B").set_index("context")
        assert int(rows.loc["game_point_for", "n"]) == 1  # 19-20 from B's side
        assert rows.loc["game_point_for", "win_rate"] == pytest.approx(0.0)

    def test_empty_input(self):
        assert pressure_summary(pd.DataFrame(), "A").empty


class TestMomentumSummary:
    def test_streaks_respect_game_boundaries(self, frames):
        _, rally_df = frames
        rows = momentum_summary(rally_df, "A").set_index("metric")
        # game 1 winners: A,A,B,A,B,B; game 2: B,A,A,A -> streak 3 (game 2).
        assert rows.loc["longest_point_streak", "value"] == pytest.approx(3.0)
        # skid must be 2 (end of game 1); a cross-game run would fabricate 3
        # (B,B | B across the game break).
        assert rows.loc["longest_skid", "value"] == pytest.approx(2.0)
        assert int(rows.loc["longest_point_streak", "n"]) == 10

    def test_close_score_win_rate(self, frames):
        _, rally_df = frames
        rows = momentum_summary(rally_df, "A").set_index("metric")
        assert rows.loc["close_score_win_rate", "value"] == pytest.approx(0.6)
        assert int(rows.loc["close_score_win_rate", "n"]) == 10

    def test_unknown_winner_breaks_runs(self):
        shots, rallies = build_synthetic_match()
        game1 = rallies[:3]  # winners A, A, B
        patched = [
            game1[0],
            game1[1].model_copy(update={"winner": None}),
            game1[2].model_copy(update={"winner": "A"}),
        ]
        rally_df, _ = build_rally_frame(patched)
        rows = momentum_summary(rally_df, "A").set_index("metric")
        assert rows.loc["longest_point_streak", "value"] == pytest.approx(1.0)

    def test_empty_input(self):
        assert momentum_summary(pd.DataFrame(), "A").empty


class TestHeadToHead:
    def test_hand_computed(self, frames):
        shot_df, rally_df = frames
        h2h = head_to_head(shot_df, rally_df).set_index(["player", "metric"])
        assert h2h.loc[("A", "rally_win_rate"), "value"] == pytest.approx(0.6)
        assert int(h2h.loc[("A", "rally_win_rate"), "n"]) == 10
        assert h2h.loc[("A", "winner_rate"), "value"] == pytest.approx(5 / 6)
        assert int(h2h.loc[("A", "winner_rate"), "n"]) == 6
        assert h2h.loc[("A", "error_share"), "value"] == pytest.approx(1 / 22)
        assert h2h.loc[("B", "rally_win_rate"), "value"] == pytest.approx(0.4)
        assert h2h.loc[("B", "winner_rate"), "value"] == pytest.approx(3 / 4)
        assert int(h2h.loc[("B", "winner_rate"), "n"]) == 4
        assert int(h2h.loc[("B", "error_share"), "n"]) == 20

    def test_every_row_carries_n_and_low_sample(self, frames):
        shot_df, rally_df = frames
        h2h = head_to_head(shot_df, rally_df)
        assert {"n", "low_sample"}.issubset(h2h.columns)
        assert h2h["n"].notna().all()

    def test_empty_input(self):
        assert head_to_head(pd.DataFrame(), pd.DataFrame()).empty


class TestMatchTrend:
    def test_single_match_row(self, frames):
        shot_df, rally_df = frames
        trend = match_trend(shot_df, rally_df, "A")
        assert len(trend) == 1
        row = trend.iloc[0]
        assert row["match_id"] == "m1"
        assert int(row["n_rallies"]) == 10
        assert int(row["n_decided"]) == 10
        assert int(row["rallies_won"]) == 6
        assert row["win_rate"] == pytest.approx(0.6)
        assert int(row["n_shots"]) == 22
        assert int(row["n_terminal"]) == 6
        assert row["winner_rate"] == pytest.approx(5 / 6)
        assert bool(row["low_sample"])  # 10 decided < default min_n 15

    def test_min_n_gate(self, frames):
        shot_df, rally_df = frames
        trend = match_trend(shot_df, rally_df, "A", min_n=10)
        assert not bool(trend.iloc[0]["low_sample"])

    def test_empty_input(self):
        assert match_trend(pd.DataFrame(), pd.DataFrame(), "A").empty

"""Tactical-mining tests: hand-computed transition probabilities and pattern lift.

Reference tallies (from the fixture rally table, player A):
    transitions (A shot -> B response): serve->lift 5, serve->net 1, lift->smash 3,
    lift->clear 1, clear->drop 1, net->lift 2, smash->drive 2, drop->net 1 = 16 total.
    2-grams over A's own shots: 12 windows, 7 in won rallies (baseline 7/12);
    serve->smash n=5 (4 won), net->smash n=2 (2 won).
"""

from __future__ import annotations

import pandas as pd
import pytest
from synchro_pipeline.analytics import (
    PATTERN_COLUMNS,
    TRANSITION_COLUMNS,
    build_rally_frame,
    build_shot_frame,
    build_transition_table,
    mine_sequence_patterns,
)
from test_analytics_fixtures import build_synthetic_match


@pytest.fixture(scope="module")
def shot_df():
    shots, rallies = build_synthetic_match()
    frame, _ = build_shot_frame(shots, rallies)
    return frame


class TestTransitionTable:
    def test_hand_computed_counts_and_probabilities(self, shot_df):
        table = build_transition_table(shot_df, "A")
        assert int(table["n"].sum()) == 16
        assert len(table) == 8
        indexed = table.set_index(["current_state", "response_state"])
        serve_lift = indexed.loc[("serve", "lift")]
        assert int(serve_lift["n"]) == 5
        assert int(serve_lift["row_total"]) == 6
        assert serve_lift["probability"] == pytest.approx(5 / 6)
        serve_net = indexed.loc[("serve", "net")]
        assert int(serve_net["n"]) == 1
        assert serve_net["probability"] == pytest.approx(1 / 6)
        lift_smash = indexed.loc[("lift", "smash")]
        assert int(lift_smash["n"]) == 3
        assert lift_smash["probability"] == pytest.approx(3 / 4)
        net_lift = indexed.loc[("net", "lift")]
        assert int(net_lift["n"]) == 2
        assert net_lift["probability"] == pytest.approx(1.0)

    def test_rows_normalize_to_one_per_state(self, shot_df):
        table = build_transition_table(shot_df, "A")
        sums = table.groupby("current_state")["probability"].sum()
        for value in sums:
            assert value == pytest.approx(1.0)

    def test_win_rates_over_decided_rallies(self, shot_df):
        indexed = build_transition_table(shot_df, "A").set_index(
            ["current_state", "response_state"]
        )
        serve_lift = indexed.loc[("serve", "lift")]
        assert serve_lift["win_rate"] == pytest.approx(0.8)  # R1,R2,R9,R10 won; R5 lost
        assert int(serve_lift["n_decided"]) == 5
        assert indexed.loc[("serve", "net"), "win_rate"] == pytest.approx(0.0)
        assert indexed.loc[("net", "lift"), "win_rate"] == pytest.approx(1.0)
        assert indexed.loc[("smash", "drive"), "win_rate"] == pytest.approx(0.5)

    def test_sorted_by_count_desc(self, shot_df):
        table = build_transition_table(shot_df, "A")
        assert table.iloc[0]["current_state"] == "serve"
        assert table.iloc[0]["response_state"] == "lift"
        assert table["n"].is_monotonic_decreasing

    def test_low_sample_flags_use_row_total(self, shot_df):
        table = build_transition_table(shot_df, "A", min_n=15)
        assert table["low_sample"].all()
        table6 = build_transition_table(shot_df, "A", min_n=6).set_index(
            ["current_state", "response_state"]
        )
        assert not table6.loc[("serve", "lift"), "low_sample"]  # row_total 6
        assert not table6.loc[("serve", "net"), "low_sample"]  # same row denominator
        assert table6.loc[("net", "lift"), "low_sample"]  # row_total 2

    def test_type_zone_tokens_use_our_zone_vocabulary(self, shot_df):
        table = build_transition_table(shot_df, "A", token="type_zone")
        indexed = table.set_index(["current_state", "response_state"])
        row = indexed.loc[("serve@front_right", "lift@rear_left")]
        assert int(row["n"]) == 4  # short-serve rallies R1,R2,R5,R9
        assert int(row["row_total"]) == 5
        assert int(indexed.loc[("serve@rear_left", "lift@rear_left"), "n"]) == 1  # long serve

    def test_unknown_token_mode_raises(self, shot_df):
        with pytest.raises(ValueError, match="token mode"):
            build_transition_table(shot_df, "A", token="bogus")

    def test_empty_input(self):
        table = build_transition_table(pd.DataFrame(), "A")
        assert table.empty
        assert list(table.columns) == list(TRANSITION_COLUMNS)


class TestSequencePatterns:
    def test_hand_computed_lift(self, shot_df):
        patterns = mine_sequence_patterns(shot_df, "A", n=2, min_support=2)
        assert len(patterns) == 2
        top = patterns.iloc[0]
        assert top["pattern"] == "serve -> smash"
        assert (int(top["n"]), int(top["n_won"]), int(top["n_lost"])) == (5, 4, 1)
        assert top["win_rate"] == pytest.approx(0.8)
        assert top["baseline_win_rate"] == pytest.approx(7 / 12)
        assert top["lift"] == pytest.approx(0.8 / (7 / 12))
        second = patterns.iloc[1]
        assert second["pattern"] == "net -> smash"
        assert int(second["n"]) == 2
        assert second["win_rate"] == pytest.approx(1.0)
        assert second["lift"] == pytest.approx(12 / 7)

    def test_default_min_support_guard_filters_thin_patterns(self, shot_df):
        # Largest 2-gram support in the fixture is 5 < default min_support 10.
        patterns = mine_sequence_patterns(shot_df, "A", n=2)
        assert patterns.empty
        assert list(patterns.columns) == list(PATTERN_COLUMNS)

    def test_min_support_one_keeps_all_windows(self, shot_df):
        patterns = mine_sequence_patterns(shot_df, "A", n=2, min_support=1)
        assert len(patterns) == 7
        assert int(patterns["n"].sum()) == 12

    def test_top_k_truncation(self, shot_df):
        patterns = mine_sequence_patterns(shot_df, "A", n=2, min_support=1, top_k=1)
        assert len(patterns) == 1
        assert patterns.iloc[0]["pattern"] == "serve -> smash"

    def test_low_sample_flag_independent_of_min_support(self, shot_df):
        patterns = mine_sequence_patterns(shot_df, "A", n=2, min_support=2, min_n=5)
        indexed = patterns.set_index("pattern")
        assert not indexed.loc["serve -> smash", "low_sample"]  # n=5 >= 5
        assert indexed.loc["net -> smash", "low_sample"]  # n=2 < 5

    def test_undecided_rallies_contribute_no_windows(self):
        shots, rallies = build_synthetic_match()
        undecided = [r.model_copy(update={"winner": None}) for r in rallies]
        rally_df, _ = build_rally_frame(undecided)
        shot_df, _ = build_shot_frame(shots, undecided)
        patterns = mine_sequence_patterns(shot_df, "A", n=2, min_support=1)
        assert patterns.empty
        assert rally_df["winner"].isna().all()

    def test_empty_input(self):
        patterns = mine_sequence_patterns(pd.DataFrame(), "A")
        assert patterns.empty
        assert list(patterns.columns) == list(PATTERN_COLUMNS)

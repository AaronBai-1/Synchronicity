"""Adapter tests: zone vocabulary, outcome derivation, QA-drop reporting, empty safety.

Expected values are hand-computed from the fixture match in test_analytics_fixtures.py
and from the zone conventions documented in analytics/adapter.py (front < 1.98 m,
mid < 4.34 m, rear otherwise; left/right in the perspective of the half's occupant).
"""

from __future__ import annotations

import pandas as pd
import pytest
from synchro_pipeline.analytics import (
    RALLY_FRAME_COLUMNS,
    SHOT_FRAME_COLUMNS,
    build_rally_frame,
    build_shot_frame,
    outcome_from_flags,
    zone6_from_area,
    zone6_from_xy,
)
from synchro_pipeline.schemas.records import XY
from test_analytics_fixtures import MATCH_ID, build_synthetic_match


def test_pandas_major_version_is_3():
    # The suite must pass against pandas 3.x (installed: 3.0.5); ported pandas-2 idioms
    # would surface here first.
    assert int(pd.__version__.split(".")[0]) >= 3


@pytest.fixture(scope="module")
def match():
    return build_synthetic_match()


@pytest.fixture(scope="module")
def frames(match):
    shots, rallies = match
    shot_df, shot_report = build_shot_frame(shots, rallies)
    rally_df, rally_report = build_rally_frame(rallies)
    return shot_df, rally_df, shot_report, rally_report


class TestZone6FromXY:
    def test_depth_bands_near(self):
        assert zone6_from_xy(0.5, -1.0, "near") == "front_right"
        assert zone6_from_xy(0.5, -1.98, "near") == "mid_right"  # boundary goes to mid
        assert zone6_from_xy(0.5, -4.33, "near") == "mid_right"
        assert zone6_from_xy(0.5, -4.34, "near") == "rear_right"  # boundary goes to rear
        assert zone6_from_xy(0.5, -6.70, "near") == "rear_right"

    def test_left_right_is_player_perspective(self):
        # near player faces +y: left means x < 0; far player faces -y: left means x > 0
        assert zone6_from_xy(-1.0, -3.0, "near") == "mid_left"
        assert zone6_from_xy(-1.0, 3.0, "far") == "mid_right"
        assert zone6_from_xy(1.0, 3.0, "far") == "mid_left"

    def test_x_zero_counts_right_on_both_sides(self):
        assert zone6_from_xy(0.0, -1.0, "near") == "front_right"
        assert zone6_from_xy(0.0, 1.0, "far") == "front_right"

    def test_out_of_bounds_is_none_not_a_guess(self):
        assert zone6_from_xy(3.1, 0.0, "near") is None
        assert zone6_from_xy(0.0, 6.8, "far") is None
        assert zone6_from_xy(-3.06, -2.0, "near") is None


class TestZone6FromArea:
    def test_corners_and_mirroring(self):
        # area 1 = net-adjacent, camera-left column
        assert zone6_from_area(1, "near") == "front_left"
        assert zone6_from_area(1, "far") == "front_right"
        # area 16 = baseline, camera-right column
        assert zone6_from_area(16, "near") == "rear_right"
        assert zone6_from_area(16, "far") == "rear_left"

    def test_row_centre_band_assignment(self):
        assert zone6_from_area(6, "near") == "mid_left"  # row 1 col 1
        assert zone6_from_area(9, "near") == "mid_left"  # row 2 centre 4.19 m -> mid
        assert zone6_from_area(15, "far") == "rear_left"  # row 3 col 2 camera-right

    def test_invalid_area_is_none(self):
        assert zone6_from_area(None, "near") is None
        assert zone6_from_area(0, "near") is None
        assert zone6_from_area(17, "far") is None


class TestOutcomeFromFlags:
    def test_winner(self):
        assert outcome_from_flags(True, None, None) == "winner"

    def test_error_split_is_best_effort(self):
        assert outcome_from_flags(None, True, "unforced error") == "error_unforced"
        assert outcome_from_flags(None, True, "forced error") == "error_forced"
        # "unforced" contains "forced": must be checked first
        assert outcome_from_flags(None, True, "unforced") == "error_unforced"

    def test_undistinguished_error_stays_generic(self):
        assert outcome_from_flags(None, True, None) == "error"
        assert outcome_from_flags(None, True, "net") == "error"

    def test_in_play(self):
        assert outcome_from_flags(None, None, None) == "in_play"
        assert outcome_from_flags(False, False, None) == "in_play"


class TestBuildShotFrame:
    def test_shape_columns_and_report(self, frames):
        shot_df, _, report, _ = frames
        assert list(shot_df.columns) == list(SHOT_FRAME_COLUMNS)
        assert len(shot_df) == 42
        assert report.n_shots_in == 42
        assert report.n_kept == 42
        assert report.n_dropped_by_reason == {}

    def test_chronological_order(self, frames):
        shot_df, _, _, _ = frames
        assert shot_df.iloc[0]["rally_id"] == f"{MATCH_ID}_g1_r01"
        assert shot_df.iloc[0]["shot_number"] == 1
        assert shot_df["t_ms"].is_monotonic_increasing

    def test_outcomes(self, frames):
        shot_df, _, _, _ = frames
        r01 = shot_df[shot_df["rally_id"] == f"{MATCH_ID}_g1_r01"]
        assert list(r01["outcome"]) == ["in_play", "in_play", "winner"]
        r05 = shot_df[shot_df["rally_id"] == f"{MATCH_ID}_g1_r05"]
        assert r05.iloc[-1]["outcome"] == "error_unforced"
        g2r04 = shot_df[shot_df["rally_id"] == f"{MATCH_ID}_g2_r04"]
        assert g2r04.iloc[-1]["outcome"] == "error"

    def test_rally_context_join(self, frames):
        shot_df, _, _, _ = frames
        first = shot_df.iloc[0]
        assert first["game_number"] == 1
        assert first["score_before"] == "0-0"
        assert (first["score_a"], first["score_b"]) == (0, 0)
        assert first["server"] == "A"
        assert first["serve_type"] == "short"
        assert first["winner"] == "A"
        assert first["rally_length"] == 3
        assert first["duration"] == pytest.approx(3.0)

    def test_zones_use_our_vocabulary(self, frames):
        shot_df, _, _, _ = frames
        first = shot_df.iloc[0]  # A short serve, near side, game 1
        assert first["shot_type"] == "serve"
        assert first["origin_zone"] == "mid_right"
        assert first["target_zone"] == "front_right"
        smashes = shot_df[(shot_df["player"] == "A") & (shot_df["shot_type"] == "smash")]
        landed = smashes[smashes["target_zone"].notna()]
        assert set(landed["target_zone"]) == {"rear_left"}
        assert set(landed["origin_zone"]) == {"rear_right"}

    def test_zone_labels_are_side_invariant(self, frames):
        # B smashes from the far side (game 1) and A from the far side (game 2) must
        # yield the same player-perspective labels as near-side smashes.
        shot_df, _, _, _ = frames
        b_smash = shot_df[(shot_df["player"] == "B") & (shot_df["shot_type"] == "smash")]
        assert set(b_smash["origin_zone"]) == {"rear_right"}
        assert set(b_smash["target_zone"].dropna()) == {"rear_left"}

    def test_error_shot_target_zone_unknown(self, frames):
        shot_df, _, _, _ = frames
        r05_last = shot_df[shot_df["rally_id"] == f"{MATCH_ID}_g1_r05"].iloc[-1]
        assert pd.isna(r05_last["target_zone"])

    def test_qa_gating_drops_are_reported(self, match):
        shots, rallies = match
        flagged = shots[0].model_copy(
            update={"ball_round": 90, "qa_flags": ["hit_uncertain"]}
        )
        low_conf = shots[0].model_copy(update={"ball_round": 91, "type_conf": 0.2})
        no_conf = shots[0].model_copy(update={"ball_round": 92, "type_conf": None})
        shot_df, report = build_shot_frame(
            [*shots, flagged, low_conf, no_conf],
            rallies,
            min_type_conf=0.5,
            exclude_qa_flags={"hit_uncertain"},
        )
        assert report.n_shots_in == 45
        assert report.n_kept == 42
        assert report.n_dropped_by_reason == {"low_type_conf": 2, "qa_flag:hit_uncertain": 1}
        assert len(shot_df) == 42
        assert not shot_df["shot_number"].isin([90, 91, 92]).any()

    def test_default_gating_keeps_everything(self, match):
        shots, rallies = match
        no_conf = shots[0].model_copy(update={"ball_round": 92, "type_conf": None})
        shot_df, report = build_shot_frame([*shots, no_conf], rallies)
        assert report.n_kept == 43
        assert len(shot_df) == 43

    def test_missing_rally_join_leaves_context_unknown(self, match):
        shots, rallies = match
        orphan = shots[0].model_copy(update={"rally_id": "nope", "ball_round": 50})
        shot_df, _ = build_shot_frame([orphan], rallies)
        row = shot_df.iloc[0]
        assert pd.isna(row["winner"])
        assert pd.isna(row["score_a"])
        assert pd.isna(row["serve_type"])

    def test_own_half_landing_described_in_landing_half_perspective(self, match):
        shots, rallies = match
        netted = shots[0].model_copy(
            update={"landing_xy_court": XY(x=-0.5, y=-0.3), "landing_area": None}
        )
        shot_df, _ = build_shot_frame([netted], rallies)
        assert shot_df.iloc[0]["target_zone"] == "front_left"

    def test_area_fallback_chain(self, match):
        shots, rallies = match
        blind = shots[0].model_copy(
            update={
                "hit_xy_court": None,
                "player_location": None,
                "player_location_area": 6,
                "landing_xy_court": None,
                "landing_area": 15,
            }
        )
        shot_df, _ = build_shot_frame([blind], rallies)
        row = shot_df.iloc[0]
        assert row["origin_zone"] == "mid_left"  # area 6, near perspective
        assert row["target_zone"] == "rear_left"  # area 15 on the far half

    def test_empty_inputs(self):
        shot_df, report = build_shot_frame([], [])
        assert shot_df.empty
        assert list(shot_df.columns) == list(SHOT_FRAME_COLUMNS)
        assert (report.n_shots_in, report.n_kept) == (0, 0)
        assert report.n_dropped_by_reason == {}


class TestBuildRallyFrame:
    def test_shape_and_order(self, frames):
        _, rally_df, _, report = frames
        assert list(rally_df.columns) == list(RALLY_FRAME_COLUMNS)
        assert len(rally_df) == 10
        assert (report.n_shots_in, report.n_kept) == (10, 10)
        assert list(rally_df["game_number"]) == [1] * 6 + [2] * 4
        assert list(rally_df[rally_df["game_number"] == 1]["seq"]) == [1, 2, 3, 4, 5, 6]

    def test_scores_and_winners(self, frames):
        _, rally_df, _, _ = frames
        r05 = rally_df[rally_df["rally_id"] == f"{MATCH_ID}_g1_r05"].iloc[0]
        assert r05["score_before"] == "3-1"
        assert (r05["score_a"], r05["score_b"]) == (3, 1)
        assert r05["winner"] == "B"
        assert list(rally_df["winner"]) == ["A", "A", "B", "A", "B", "B", "B", "A", "A", "A"]

    def test_qa_gating(self, match):
        _, rallies = match
        flagged = rallies[0].model_copy(update={"qa_flags": ["ocr_conflict"]})
        rally_df, report = build_rally_frame(
            [flagged, *rallies[1:]], exclude_qa_flags={"ocr_conflict"}
        )
        assert len(rally_df) == 9
        assert report.n_dropped_by_reason == {"qa_flag:ocr_conflict": 1}

    def test_empty_input(self):
        rally_df, report = build_rally_frame([])
        assert rally_df.empty
        assert list(rally_df.columns) == list(RALLY_FRAME_COLUMNS)
        assert (report.n_shots_in, report.n_kept) == (0, 0)

"""Unit tests for the pure labeling helpers (eval/labeling.py) that back the GUI tools
in pipeline/tools/. The GUI event loops themselves are deliberately untested; every
decision they make routes through these functions."""

import cv2
import numpy as np
import pytest
from synchro_pipeline.domain.court import (
    COURT_KEYPOINTS,
    N_COURT_KEYPOINTS,
    keypoint_array,
    project_points,
)
from synchro_pipeline.eval.golden import (
    GoldenMatch,
    GoldenRally,
    GoldenVideoMismatch,
    save_golden_match,
    sha256_file,
)
from synchro_pipeline.eval.labeling import (
    CORNER_CLICK_ORDER,
    COURT_LINE_INDICES,
    RallyLabelSession,
    add_hit,
    corners_to_keypoints,
    find_rally,
    insert_rally,
    keypoints_to_label,
    load_or_create_golden,
    merge_court_label,
    merge_rallies,
    nudge_point,
    remove_hit,
    remove_rally,
)


def synthetic_court_to_image_h() -> np.ndarray:
    """A plausible court->image projection for a 1280x720 behind-court camera
    (mirrors the fixture in test_court.py)."""
    court_corners = np.array([[-3.05, -6.7], [3.05, -6.7], [-3.05, 6.7], [3.05, 6.7]])
    image_corners = np.array([[140.0, 660.0], [1140.0, 660.0], [430.0, 260.0], [850.0, 260.0]])
    h, _ = cv2.findHomography(court_corners, image_corners)
    return h


class TestCornersToKeypoints:
    def test_recovers_all_16_from_4_corners(self):
        h_c2i = synthetic_court_to_image_h()
        corner_court = np.array([COURT_KEYPOINTS[n] for n in CORNER_CLICK_ORDER])
        corner_px = project_points(h_c2i, corner_court)
        expected = project_points(h_c2i, keypoint_array())
        result = corners_to_keypoints(corner_px)
        assert result is not None
        assert result.shape == (N_COURT_KEYPOINTS, 2)
        assert np.allclose(result, expected, atol=0.1)  # px

    def test_degenerate_corners_return_none(self):
        collinear = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]
        assert corners_to_keypoints(collinear) is None

    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match="4"):
            corners_to_keypoints([[0.0, 0.0], [1.0, 1.0]])

    def test_line_indices_are_valid(self):
        assert all(
            0 <= a < N_COURT_KEYPOINTS and 0 <= b < N_COURT_KEYPOINTS
            for a, b in COURT_LINE_INDICES
        )


class TestNudgeAndLabelConversion:
    def test_nudge_is_pure_and_targeted(self):
        pts = np.zeros((N_COURT_KEYPOINTS, 2))
        moved = nudge_point(pts, 3, 2.0, -1.0)
        assert moved[3, 0] == 2.0 and moved[3, 1] == -1.0
        assert np.all(pts == 0.0)  # input untouched
        assert np.all(moved[np.arange(N_COURT_KEYPOINTS) != 3] == 0.0)

    def test_nudge_index_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            nudge_point(np.zeros((16, 2)), 16, 1.0, 0.0)

    def test_keypoints_to_label_dropped_and_rounded(self):
        pts = np.full((N_COURT_KEYPOINTS, 2), 1.23456)
        label = keypoints_to_label(pts, dropped=[0, 15])
        assert label[0] is None and label[15] is None
        assert label[1] == (1.23, 1.23)
        assert len(label) == N_COURT_KEYPOINTS

    def test_keypoints_to_label_validates_inputs(self):
        with pytest.raises(ValueError, match="shape"):
            keypoints_to_label(np.zeros((4, 2)))
        with pytest.raises(ValueError, match="out of range"):
            keypoints_to_label(np.zeros((16, 2)), dropped=[16])


class TestGoldenMerging:
    def label(self, frame: int) -> list:
        return keypoints_to_label(np.zeros((N_COURT_KEYPOINTS, 2)) + frame)

    def test_merge_court_label_appends_and_sorts(self):
        match = GoldenMatch(match_id="m", video_uri="v")
        match = merge_court_label(match, 200, self.label(200))
        match = merge_court_label(match, 100, self.label(100))
        assert [c.frame for c in match.court_labels] == [100, 200]

    def test_merge_court_label_replaces_same_frame(self):
        match = GoldenMatch(match_id="m", video_uri="v")
        match = merge_court_label(match, 100, self.label(1))
        match = merge_court_label(match, 100, self.label(2))
        assert len(match.court_labels) == 1
        assert match.court_labels[0].keypoints[0] == (2.0, 2.0)

    def test_merge_rallies_replaces_and_revalidates(self):
        match = GoldenMatch(
            match_id="m", video_uri="v", rallies=[GoldenRally(start_frame=0, end_frame=10)]
        )
        merged = merge_rallies(
            match,
            [GoldenRally(start_frame=50, end_frame=60), GoldenRally(start_frame=20, end_frame=30)],
        )
        assert [r.start_frame for r in merged.rallies] == [20, 50]

    def test_load_or_create_creates_skeleton(self, tmp_path):
        match = load_or_create_golden(str(tmp_path / "new.json"), "m9", "uri://v")
        assert match.match_id == "m9" and match.rallies == []

    def test_load_or_create_requires_identity_for_new_files(self, tmp_path):
        with pytest.raises(ValueError, match="--match-id"):
            load_or_create_golden(str(tmp_path / "new.json"))

    def test_load_or_create_loads_existing(self, tmp_path):
        path = tmp_path / "m.json"
        save_golden_match(GoldenMatch(match_id="existing", video_uri="v"), path)
        assert load_or_create_golden(str(path)).match_id == "existing"


class TestRallyHelpers:
    def rallies(self) -> list[GoldenRally]:
        return [
            GoldenRally(start_frame=100, end_frame=200),
            GoldenRally(start_frame=300, end_frame=400),
        ]

    def test_find_rally_inclusive_bounds(self):
        rallies = self.rallies()
        assert find_rally(rallies, 100) is rallies[0]
        assert find_rally(rallies, 200) is rallies[0]
        assert find_rally(rallies, 250) is None

    def test_insert_rally_sorts(self):
        out = insert_rally(self.rallies(), GoldenRally(start_frame=210, end_frame=290))
        assert [r.start_frame for r in out] == [100, 210, 300]

    def test_insert_rally_rejects_overlap_including_shared_boundary(self):
        with pytest.raises(ValueError, match="overlaps"):
            insert_rally(self.rallies(), GoldenRally(start_frame=150, end_frame=250))
        with pytest.raises(ValueError, match="overlaps"):
            insert_rally(self.rallies(), GoldenRally(start_frame=200, end_frame=250))

    def test_add_hit_and_replace_same_frame(self):
        out = add_hit(self.rallies(), 150, "near")
        out = add_hit(out, 120, "far")
        assert [(h.frame, h.side) for h in out[0].hits] == [(120, "far"), (150, "near")]
        out = add_hit(out, 150, "far")  # replace, don't duplicate
        assert [(h.frame, h.side) for h in out[0].hits] == [(120, "far"), (150, "far")]

    def test_add_hit_outside_any_rally(self):
        with pytest.raises(ValueError, match="not inside any labeled rally"):
            add_hit(self.rallies(), 250, "near")

    def test_remove_hit(self):
        out = add_hit(self.rallies(), 150, "near")
        out = remove_hit(out, 150)
        assert out[0].hits == []
        with pytest.raises(ValueError, match="no hit"):
            remove_hit(out, 150)

    def test_remove_rally(self):
        out = remove_rally(self.rallies(), 350)
        assert [r.start_frame for r in out] == [100]
        with pytest.raises(ValueError, match="no rally"):
            remove_rally(out, 350)


class TestRallyLabelSession:
    def test_full_flow_start_hits_end(self):
        session = RallyLabelSession()
        session.mark_start(100)
        session.mark_hit(130, "far")
        session.mark_hit(110, "near")  # out of order on purpose
        session.mark_end(150)
        assert not session.is_open
        assert len(session.rallies) == 1
        rally = session.rallies[0]
        assert (rally.start_frame, rally.end_frame) == (100, 150)
        assert [(h.frame, h.side) for h in rally.hits] == [(110, "near"), (130, "far")]

    def test_hit_into_existing_rally_when_closed(self):
        session = RallyLabelSession(rallies=[GoldenRally(start_frame=0, end_frame=50)])
        session.mark_hit(20, "near")
        assert [h.frame for h in session.rallies[0].hits] == [20]

    def test_end_without_start(self):
        with pytest.raises(ValueError, match="no open rally"):
            RallyLabelSession().mark_end(10)

    def test_double_start(self):
        session = RallyLabelSession()
        session.mark_start(10)
        with pytest.raises(ValueError, match="already open"):
            session.mark_start(20)

    def test_start_inside_existing_rally(self):
        session = RallyLabelSession(rallies=[GoldenRally(start_frame=0, end_frame=50)])
        with pytest.raises(ValueError, match="already-labeled"):
            session.mark_start(25)

    def test_end_not_after_start(self):
        session = RallyLabelSession()
        session.mark_start(100)
        with pytest.raises(ValueError, match="after its start"):
            session.mark_end(100)

    def test_hit_before_open_start(self):
        session = RallyLabelSession()
        session.mark_start(100)
        with pytest.raises(ValueError, match="before the open rally"):
            session.mark_hit(90, "near")

    def test_failed_end_keeps_rally_open_with_hits(self):
        session = RallyLabelSession(rallies=[GoldenRally(start_frame=200, end_frame=300)])
        session.mark_start(100)
        session.mark_hit(150, "near")
        with pytest.raises(ValueError, match="overlaps"):
            session.mark_end(250)  # would overlap the existing rally
        assert session.is_open
        assert [h.frame for h in session.pending_hits] == [150]
        session.mark_end(199)  # correction succeeds
        assert len(session.rallies) == 2

    def test_undo_and_cancel(self):
        session = RallyLabelSession()
        session.mark_start(0)
        session.mark_hit(10, "near")
        session.mark_hit(20, "far")
        undone = session.undo_pending_hit()
        assert undone.frame == 20
        session.cancel()
        assert not session.is_open and session.pending_hits == []
        with pytest.raises(ValueError, match="no pending hits"):
            session.undo_pending_hit()


class TestLoadOrCreateGoldenVideoHash:
    """Automatic hash stamping/verification (the labeling tools' anti-wrong-encode gate)."""

    def _video(self, tmp_path, content=b"encode-one"):
        f = tmp_path / "mezz.mp4"
        f.write_bytes(content)
        return f

    def test_create_stamps_hash(self, tmp_path):
        video = self._video(tmp_path)
        match = load_or_create_golden(
            str(tmp_path / "new.json"), "m1", "uri://v", video_path=str(video)
        )
        assert match.video_sha256 == sha256_file(video)

    def test_create_without_video_leaves_hash_none(self, tmp_path):
        match = load_or_create_golden(str(tmp_path / "new.json"), "m1", "uri://v")
        assert match.video_sha256 is None

    def test_load_backfills_missing_hash(self, tmp_path):
        video = self._video(tmp_path)
        path = tmp_path / "legacy.json"
        save_golden_match(GoldenMatch(match_id="m1", video_uri="v"), path)
        match = load_or_create_golden(str(path), video_path=str(video))
        assert match.video_sha256 == sha256_file(video)

    def test_load_verifies_matching_hash(self, tmp_path):
        video = self._video(tmp_path)
        path = tmp_path / "ok.json"
        save_golden_match(
            GoldenMatch(match_id="m1", video_uri="v", video_sha256=sha256_file(video)), path
        )
        match = load_or_create_golden(str(path), video_path=str(video))
        assert match.video_sha256 == sha256_file(video)

    def test_load_rejects_wrong_encode(self, tmp_path):
        video = self._video(tmp_path)
        path = tmp_path / "labels.json"
        save_golden_match(
            GoldenMatch(match_id="m1", video_uri="v", video_sha256=sha256_file(video)), path
        )
        other = self._video(tmp_path, content=b"encode-two")
        with pytest.raises(GoldenVideoMismatch):
            load_or_create_golden(str(path), video_path=str(other))

    def test_explicit_override_restamps(self, tmp_path):
        video = self._video(tmp_path)
        path = tmp_path / "labels.json"
        save_golden_match(
            GoldenMatch(match_id="m1", video_uri="v", video_sha256=sha256_file(video)), path
        )
        other = self._video(tmp_path, content=b"encode-two")
        match = load_or_create_golden(
            str(path), video_path=str(other), allow_hash_mismatch=True
        )
        assert match.video_sha256 == sha256_file(other)

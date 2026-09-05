"""Golden-set schema round-trip + loader error behaviour (eval/golden.py)."""

import pytest
from pydantic import ValidationError
from synchro_pipeline.eval.golden import (
    GoldenCourtLabel,
    GoldenHit,
    GoldenLoadError,
    GoldenMatch,
    GoldenRally,
    GoldenScoreEntry,
    GoldenShuttlePoint,
    GoldenVideoMismatch,
    load_golden_match,
    save_golden_match,
    sha256_file,
    verify_video_hash,
)


def keypoints_16(missing_idx: int | None = None) -> list[tuple[float, float] | None]:
    pts: list[tuple[float, float] | None] = [(float(10 * i), float(5 * i)) for i in range(16)]
    if missing_idx is not None:
        pts[missing_idx] = None
    return pts


def make_match(**overrides) -> GoldenMatch:
    base = dict(
        match_id="m1",
        video_uri="r2://matches/m1.mp4",
        broadcaster="bwf",
        rallies=[
            GoldenRally(
                start_frame=100,
                end_frame=220,
                hits=[GoldenHit(frame=110, side="near"), GoldenHit(frame=150, side="far")],
                shuttle=[
                    GoldenShuttlePoint(frame=110, x=400.0, y=300.0, visible=True),
                    GoldenShuttlePoint(frame=111, visible=False),
                ],
            ),
            GoldenRally(start_frame=300, end_frame=380),
        ],
        court_labels=[GoldenCourtLabel(frame=100, keypoints=keypoints_16(missing_idx=3))],
        score_timeline=[GoldenScoreEntry(frame=90, a=0, b=0)],
    )
    base.update(overrides)
    return GoldenMatch(**base)


class TestSchema:
    def test_round_trip_through_file(self, tmp_path):
        match = make_match()
        path = tmp_path / "m1.json"
        save_golden_match(match, path)
        assert load_golden_match(path) == match

    def test_rallies_and_hits_are_sorted_on_validation(self):
        match = make_match(
            rallies=[
                GoldenRally(
                    start_frame=300,
                    end_frame=380,
                    hits=[GoldenHit(frame=350, side="far"), GoldenHit(frame=310, side="near")],
                ),
                GoldenRally(start_frame=100, end_frame=220),
            ]
        )
        assert [r.start_frame for r in match.rallies] == [100, 300]
        assert [h.frame for h in match.rallies[1].hits] == [310, 350]

    def test_shuttle_none_vs_empty_are_distinct(self):
        unlabelled = GoldenRally(start_frame=0, end_frame=10)
        labelled_empty = GoldenRally(start_frame=20, end_frame=30, shuttle=[])
        assert unlabelled.shuttle is None
        assert labelled_empty.shuttle == []

    def test_overlapping_rallies_rejected(self):
        with pytest.raises(ValidationError, match="overlap"):
            make_match(
                rallies=[
                    GoldenRally(start_frame=100, end_frame=220),
                    GoldenRally(start_frame=220, end_frame=300),  # shares frame 220
                ]
            )

    def test_rally_end_before_start_rejected(self):
        with pytest.raises(ValidationError, match="before start_frame"):
            GoldenRally(start_frame=100, end_frame=99)

    def test_hit_outside_rally_rejected(self):
        with pytest.raises(ValidationError, match="outside its rally"):
            GoldenRally(start_frame=100, end_frame=200, hits=[GoldenHit(frame=50, side="near")])

    def test_visible_shuttle_point_requires_xy(self):
        with pytest.raises(ValidationError, match="requires x and y"):
            GoldenShuttlePoint(frame=1, visible=True)

    def test_court_label_needs_exactly_16_keypoints(self):
        with pytest.raises(ValidationError, match="expected 16 keypoints"):
            GoldenCourtLabel(frame=0, keypoints=keypoints_16()[:15])

    def test_duplicate_court_label_frame_rejected(self):
        with pytest.raises(ValidationError, match="duplicate court_labels"):
            make_match(
                court_labels=[
                    GoldenCourtLabel(frame=100, keypoints=keypoints_16()),
                    GoldenCourtLabel(frame=100, keypoints=keypoints_16()),
                ]
            )

    def test_unknown_field_rejected(self):
        # extra="forbid" is what catches hand-editing typos
        with pytest.raises(ValidationError):
            GoldenMatch(match_id="m1", video_uri="v", ralies=[])


class TestLoader:
    def test_missing_file(self, tmp_path):
        with pytest.raises(GoldenLoadError, match="not found"):
            load_golden_match(tmp_path / "nope.json")

    def test_invalid_json_reports_location(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text('{"match_id": "m1",\n  "video_uri": }\n')
        with pytest.raises(GoldenLoadError, match=r"not valid JSON \(line 2"):
            load_golden_match(path)

    def test_schema_violation_reports_field_path(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(
            '{"match_id": "m1", "video_uri": "v", '
            '"rallies": [{"start_frame": 10, "end_frame": 5}]}'
        )
        with pytest.raises(GoldenLoadError, match=r"rallies\.0"):
            load_golden_match(path)

    def test_typo_key_is_a_clear_error(self, tmp_path):
        path = tmp_path / "typo.json"
        path.write_text('{"match_id": "m1", "video_uri": "v", "ralies": []}')
        with pytest.raises(GoldenLoadError, match="ralies"):
            load_golden_match(path)

    def test_hand_written_minimal_file_loads(self, tmp_path):
        # the smallest file a labeller could start from by hand
        path = tmp_path / "minimal.json"
        path.write_text('{"match_id": "m1", "video_uri": "local/m1.mp4"}')
        match = load_golden_match(path)
        assert match.match_id == "m1"
        assert match.rallies == []
        assert match.broadcaster is None


class TestVideoHash:
    # sha256("hello") — a fixed vector so a broken streaming read can't self-validate
    HELLO = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_sha256_file_known_vector(self, tmp_path):
        f = tmp_path / "v.bin"
        f.write_bytes(b"hello")
        assert sha256_file(f) == self.HELLO

    def test_malformed_hash_rejected_by_schema(self):
        with pytest.raises(ValidationError):
            GoldenMatch(match_id="m", video_uri="v", video_sha256="not-a-hash")
        with pytest.raises(ValidationError):
            GoldenMatch(match_id="m", video_uri="v", video_sha256=self.HELLO.upper())

    def test_verify_passes_and_returns_hash(self, tmp_path):
        f = tmp_path / "v.bin"
        f.write_bytes(b"hello")
        match = GoldenMatch(match_id="m", video_uri="v", video_sha256=self.HELLO)
        assert verify_video_hash(match, f) == self.HELLO

    def test_verify_without_recorded_hash_is_permissive(self, tmp_path):
        f = tmp_path / "v.bin"
        f.write_bytes(b"hello")
        match = GoldenMatch(match_id="m", video_uri="v")
        assert verify_video_hash(match, f) == self.HELLO  # caller may backfill

    def test_verify_mismatch_names_both_hashes(self, tmp_path):
        f = tmp_path / "v.bin"
        f.write_bytes(b"different bytes")
        match = GoldenMatch(match_id="m", video_uri="v", video_sha256=self.HELLO)
        with pytest.raises(GoldenVideoMismatch, match="2cf24dba"):
            verify_video_hash(match, f)

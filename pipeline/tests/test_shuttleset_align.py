"""ShuttleSet → golden alignment (eval/shuttleset.py + tools/align_shuttleset.py).

All fixtures are synthetic but use the VERIFIED real column layout (fetched from the
CoachAI-Projects repo — see the module docstring of eval/shuttleset.py); tests run
offline with no network and no GUI. The CLI test drives main(argv) end-to-end against
a tiny cv2-written mp4.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from synchro_pipeline.domain.taxonomy import ShotType
from synchro_pipeline.eval.golden import (
    GoldenMatch,
    GoldenRally,
    GoldenScoreEntry,
    load_golden_match,
    save_golden_match,
    sha256_file,
)
from synchro_pipeline.eval.shuttleset import (
    Alignment,
    AlignmentError,
    ShuttleSetError,
    ShuttleStroke,
    SideResolutionError,
    fit_alignment,
    load_shuttleset_match,
    make_serve_side_resolver,
    resolve_sides_by_location,
    strokes_to_golden,
)

# The exact header of the real ShuttleSet set CSVs (verified 2026-09-04; see
# eval/shuttleset.py module docstring for provenance).
HEADER = (
    "rally,ball_round,time,frame_num,roundscore_A,roundscore_B,player,server,type,"
    "aroundhead,backhand,hit_height,hit_area,hit_x,hit_y,landing_height,landing_area,"
    "landing_x,landing_y,lose_reason,win_reason,getpoint_player,flaw,"
    "player_location_area,player_location_x,player_location_y,opponent_location_area,"
    "opponent_location_x,opponent_location_y,db"
)


def csv_row(
    rally: int,
    ball_round: int,
    frame_num: str,
    score_a: int,
    score_b: int,
    player: str,
    shot_type: str,
    loc_y: str = "",
    loc_x: str = "600.0",
    getpoint: str = "",
) -> str:
    """One data row in the verified 30-column layout (unused columns left empty)."""
    return (
        f"{rally},{ball_round}.0,00:00:00,{frame_num},{score_a},{score_b},{player},2,"
        f"{shot_type},,,1.0,,,,1.0,3,700.0,600.0,,,{getpoint},,8,{loc_x},{loc_y},8,,,0"
    )


def write_csv(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def stroke(
    set_no: int = 1,
    rally: int = 1,
    ball_round: int = 1,
    frame_num: int | None = 1000,
    player: str | None = "A",
    score_a: int | None = 1,
    score_b: int | None = 0,
    loc_y: float | None = None,
    shot_type: ShotType | None = ShotType.CLEAR,
) -> ShuttleStroke:
    return ShuttleStroke(
        set_no=set_no,
        rally=rally,
        ball_round=ball_round,
        frame_num=frame_num,
        type=shot_type,
        type_raw=shot_type.value if shot_type else "未知球種",
        player=player,  # type: ignore[arg-type]
        roundscore_a=score_a,
        roundscore_b=score_b,
        player_location_y=loc_y,
    )


def make_rally(
    set_no: int,
    rally: int,
    frames: list[int],
    score: tuple[int, int],
    first_player: str = "A",
    near_first: bool = True,
) -> list[ShuttleStroke]:
    """Alternating-player rally with location-y values putting the first hitter near."""
    strokes = []
    for i, f in enumerate(frames):
        player = first_player if i % 2 == 0 else ("B" if first_player == "A" else "A")
        near = near_first == (i % 2 == 0)
        strokes.append(
            stroke(
                set_no=set_no,
                rally=rally,
                ball_round=i + 1,
                frame_num=f,
                player=player,
                score_a=score[0],
                score_b=score[1],
                loc_y=620.0 if near else 350.0,  # verified convention: larger y = near
            )
        )
    return strokes


class TestFitAlignment:
    def test_two_anchors_recover_exact_scale_and_offset(self):
        # 25fps source → 30fps mezzanine trimmed by 500 frames: mz = 1.2*ss - 500
        a = Alignment.fit([(10000, 11500), (60000, 71500)])
        assert a.scale == pytest.approx(1.2)
        assert a.offset == pytest.approx(-500.0)
        assert a.max_residual_frames == pytest.approx(0.0, abs=1e-6)
        assert a.map_frame(20000) == 23500

    def test_least_squares_over_noisy_anchors(self):
        true = lambda x: 0.5 * x + 100  # noqa: E731
        anchors = [(x, int(true(x)) + d) for x, d in ((1000, 1), (5000, -1), (9000, 1), (13000, -1))]
        a = fit_alignment(anchors)
        assert a.scale == pytest.approx(0.5, abs=1e-3)
        assert a.offset == pytest.approx(100.0, abs=5.0)
        assert len(a.residuals) == 4
        assert a.max_residual_frames <= 1.5

    def test_residuals_are_predicted_minus_actual(self):
        a = fit_alignment([(0, 0), (100, 100), (200, 206)])
        predicted = [a.scale * ss + a.offset for ss, _ in a.anchors]
        assert list(a.residuals) == pytest.approx(
            [p - mz for p, (_, mz) in zip(predicted, a.anchors, strict=True)]
        )

    def test_refuses_when_an_anchor_is_off_the_line(self):
        # The middle anchor fights the other two; the worst residual (~16.7 frames at
        # anchor 100:50) is named in the refusal.
        with pytest.raises(AlignmentError, match="anchor 100:50"):
            Alignment.fit([(0, 0), (100, 50), (200, 150)], tolerance_frames=5.0)

    def test_tolerance_none_skips_the_gate(self):
        a = Alignment.fit([(0, 0), (100, 50), (200, 150)], tolerance_frames=None)
        assert a.max_residual_frames > 5.0
        with pytest.raises(AlignmentError):
            a.require_within(5.0)

    def test_within_tolerance_passes_the_gate(self):
        a = Alignment.fit([(0, 2), (100, 100), (200, 202)], tolerance_frames=5.0)
        assert a.max_residual_frames <= 5.0

    def test_fewer_than_two_anchors_refused(self):
        with pytest.raises(AlignmentError, match="at least 2"):
            fit_alignment([(100, 200)])

    def test_duplicate_shuttleset_frames_refused(self):
        with pytest.raises(AlignmentError, match="distinct"):
            fit_alignment([(100, 200), (100, 300)])

    def test_negative_scale_refused(self):
        with pytest.raises(AlignmentError, match="not positive"):
            fit_alignment([(100, 900), (900, 100)])


class TestLoadShuttleSetMatch:
    def test_parses_verified_layout_and_translates_chinese_types(self, tmp_path):
        write_csv(
            tmp_path / "set1.csv",
            [
                csv_row(1, 1, "10418.0", 1, 0, "B", "發長球", loc_y="356.0"),
                csv_row(1, 2, "10481.0", 1, 0, "A", "切球", loc_y="641.0", getpoint="A"),
            ],
        )
        strokes = load_shuttleset_match([tmp_path / "set1.csv"])
        assert len(strokes) == 2
        first = strokes[0]
        assert first.set_no == 1
        assert first.rally == 1
        assert first.ball_round == 1
        assert first.frame_num == 10418
        assert first.type is ShotType.LONG_SERVICE
        assert first.player == "B"
        assert (first.roundscore_a, first.roundscore_b) == (1, 0)
        assert first.player_location_y == 356.0
        assert strokes[1].type is ShotType.DROP
        assert strokes[1].getpoint_player == "A"

    def test_set_number_from_filename_and_multi_csv_sort(self, tmp_path):
        write_csv(tmp_path / "set2.csv", [csv_row(1, 1, "50000.0", 1, 0, "A", "長球")])
        write_csv(tmp_path / "set1.csv", [csv_row(1, 1, "10000.0", 0, 1, "B", "長球")])
        strokes = load_shuttleset_match([tmp_path / "set2.csv", tmp_path / "set1.csv"])
        assert [s.set_no for s in strokes] == [1, 2]

    def test_unknown_type_becomes_none_with_raw_preserved(self, tmp_path):
        write_csv(tmp_path / "set1.csv", [csv_row(1, 1, "100.0", 1, 0, "A", "未知球種")])
        (s,) = load_shuttleset_match([tmp_path / "set1.csv"])
        assert s.type is None
        assert s.type_raw == "未知球種"

    def test_english_type_strings_load_directly(self, tmp_path):
        write_csv(tmp_path / "set1.csv", [csv_row(1, 1, "100.0", 1, 0, "A", "wrist smash")])
        (s,) = load_shuttleset_match([tmp_path / "set1.csv"])
        assert s.type is ShotType.WRIST_SMASH

    def test_missing_frame_num_becomes_none(self, tmp_path):
        write_csv(tmp_path / "set1.csv", [csv_row(1, 1, "", 1, 0, "A", "長球")])
        (s,) = load_shuttleset_match([tmp_path / "set1.csv"])
        assert s.frame_num is None

    def test_unparseable_filename_requires_explicit_set_numbers(self, tmp_path):
        p = write_csv(tmp_path / "labels.csv", [csv_row(1, 1, "100.0", 1, 0, "A", "長球")])
        with pytest.raises(ShuttleSetError, match="set number"):
            load_shuttleset_match([p])
        (s,) = load_shuttleset_match([p], set_numbers=[3])
        assert s.set_no == 3

    def test_missing_columns_named(self, tmp_path):
        p = tmp_path / "set1.csv"
        p.write_text("rally,frame_num\n1,100.0\n", encoding="utf-8")
        with pytest.raises(ShuttleSetError, match="ball_round"):
            load_shuttleset_match([p])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ShuttleSetError, match="not found"):
            load_shuttleset_match([tmp_path / "set1.csv"])


class TestResolveSidesByLocation:
    def test_alternating_locations_resolve_sides(self):
        rally = make_rally(1, 1, [100, 130, 160], score=(1, 0), near_first=True)
        assert resolve_sides_by_location(rally) == ["near", "far", "near"]

    def test_far_first_hitter(self):
        rally = make_rally(1, 1, [100, 130], score=(1, 0), near_first=False)
        assert resolve_sides_by_location(rally) == ["far", "near"]

    def test_missing_locations_abstain(self):
        rally = [
            stroke(ball_round=1, player="A", loc_y=None),
            stroke(ball_round=2, player="B", frame_num=1030, loc_y=350.0),
        ]
        with pytest.raises(SideResolutionError, match="player A"):
            resolve_sides_by_location(rally)

    def test_ambiguous_separation_abstains(self):
        rally = [
            stroke(ball_round=1, player="A", loc_y=500.0),
            stroke(ball_round=2, player="B", frame_num=1030, loc_y=510.0),
        ]
        with pytest.raises(SideResolutionError, match="ambiguous"):
            resolve_sides_by_location(rally)

    def test_non_alternating_players_abstain(self):
        rally = [
            stroke(ball_round=1, player="A", loc_y=620.0),
            stroke(ball_round=2, player="A", frame_num=1030, loc_y=620.0),
        ]
        with pytest.raises(SideResolutionError, match="alternate"):
            resolve_sides_by_location(rally)


class TestServeSideResolver:
    def test_seeds_from_first_provided_sets_opening_server(self):
        # B serves rally 1 → B near; strokes alternate B,A,B.
        rally = make_rally(1, 1, [100, 130, 160], score=(0, 1), first_player="B")
        resolver = make_serve_side_resolver(rally, "near")
        assert resolver(rally) == ["near", "far", "near"]

    def test_far_serves_first(self):
        rally = make_rally(1, 1, [100, 130], score=(1, 0), first_player="A")
        resolver = make_serve_side_resolver(rally, "far")
        assert resolver(rally) == ["far", "near"]

    def test_ends_swap_between_sets(self):
        s1 = make_rally(1, 1, [100, 130], score=(1, 0), first_player="A")
        s2 = make_rally(2, 1, [5000, 5030], score=(1, 0), first_player="A")
        resolver = make_serve_side_resolver(s1 + s2, "near")
        assert resolver(s1) == ["near", "far"]
        assert resolver(s2) == ["far", "near"]  # players changed ends after game 1

    def test_deciding_set_switches_ends_at_eleven(self):
        before = make_rally(3, 20, [100, 130], score=(10, 9), first_player="A")
        at_eleven = make_rally(3, 21, [500, 530], score=(11, 9), first_player="A")
        after = make_rally(3, 22, [900, 930], score=(11, 10), first_player="A")
        resolver = make_serve_side_resolver(before + at_eleven + after, "near")
        assert resolver(before) == ["near", "far"]
        assert resolver(at_eleven) == ["near", "far"]  # the 11th point is played first
        assert resolver(after) == ["far", "near"]  # then ends change

    def test_unknown_rally_abstains(self):
        known = make_rally(1, 1, [100, 130], score=(1, 0))
        resolver = make_serve_side_resolver(known, "near")
        with pytest.raises(SideResolutionError, match="not part of"):
            resolver(make_rally(1, 9, [999, 1030], score=(2, 0)))


class TestStrokesToGolden:
    def identity(self) -> Alignment:
        return fit_alignment([(0, 0), (10000, 10000)])

    def test_groups_pads_and_orders(self):
        strokes = make_rally(1, 1, [1000, 1030, 1060], score=(1, 0)) + make_rally(
            1, 2, [2000, 2030], score=(1, 1), first_player="B", near_first=False
        )
        rallies, report = strokes_to_golden(strokes, self.identity(), pad_s=1.5, fps=30.0)
        assert report.n_rallies == 2
        assert report.n_emitted == 2
        assert report.n_hits == 5
        r1, r2 = rallies
        assert (r1.start_frame, r1.end_frame) == (1000 - 45, 1060 + 45)
        assert [h.frame for h in r1.hits] == [1000, 1030, 1060]
        assert [h.side for h in r1.hits] == ["near", "far", "near"]
        assert [h.side for h in r2.hits] == ["far", "near"]

    def test_alignment_applied_to_hit_frames(self):
        alignment = fit_alignment([(0, 100), (1000, 1300)])  # mz = 1.2*ss + 100
        strokes = make_rally(1, 1, [1000, 1100], score=(1, 0))
        rallies, _ = strokes_to_golden(strokes, alignment)
        assert [h.frame for h in rallies[0].hits] == [1300, 1420]

    def test_padding_clamped_to_video(self):
        strokes = make_rally(1, 1, [10, 40], score=(1, 0))
        rallies, _ = strokes_to_golden(strokes, self.identity(), max_frame=60)
        assert (rallies[0].start_frame, rallies[0].end_frame) == (0, 60)

    def test_close_rallies_trim_padding_not_hits(self):
        strokes = make_rally(1, 1, [1000, 1030], score=(1, 0)) + make_rally(
            1, 2, [1060, 1090], score=(1, 1)
        )
        rallies, _ = strokes_to_golden(strokes, self.identity(), pad_s=1.5, fps=30.0)
        r1, r2 = rallies
        assert r1.end_frame < r2.start_frame  # golden schema forbids overlap
        assert r1.end_frame >= 1030 and r2.start_frame <= 1060  # hits stay inside
        GoldenMatch(match_id="m", video_uri="u", rallies=[r.model_dump() for r in rallies])

    def test_score_timeline_from_roundscores(self):
        strokes = make_rally(1, 1, [1000, 1030], score=(1, 0)) + make_rally(
            1, 2, [3000, 3030], score=(1, 1)
        )
        _, report = strokes_to_golden(strokes, self.identity(), fps=30.0, score_delta_s=1.0)
        assert report.score_timeline == [
            GoldenScoreEntry(frame=1060, a=1, b=0),
            GoldenScoreEntry(frame=3060, a=1, b=1),
        ]
        assert any("approximations" in n or "NOT the observed" in n for n in report.notes)

    def test_missing_frame_num_skips_rally_with_reason(self):
        broken = make_rally(1, 1, [1000, 1030], score=(1, 0))
        broken[1] = broken[1].model_copy(update={"frame_num": None})
        ok = make_rally(1, 2, [3000, 3030], score=(1, 1))
        rallies, report = strokes_to_golden(broken + ok, self.identity())
        assert len(rallies) == 1
        assert rallies[0].hits[0].frame == 3000
        assert any("set 1 rally 1" in s and "frame_num" in s for s in report.skipped)

    def test_out_of_range_frames_skip_rally(self):
        strokes = make_rally(1, 1, [1000, 1030], score=(1, 0))
        rallies, report = strokes_to_golden(strokes, self.identity(), max_frame=500)
        assert rallies == []
        assert any("outside the" in s for s in report.skipped)

    def test_side_resolution_abstention_skips_rally(self):
        no_locations = [
            stroke(ball_round=1, player="A", frame_num=1000),
            stroke(ball_round=2, player="B", frame_num=1030),
        ]
        rallies, report = strokes_to_golden(no_locations, self.identity())
        assert rallies == []
        assert report.n_emitted == 0
        assert any("cannot resolve near/far" in s for s in report.skipped)

    def test_serve_resolver_plugs_in(self):
        strokes = [
            stroke(ball_round=1, player="A", frame_num=1000),
            stroke(ball_round=2, player="B", frame_num=1030),
        ]
        resolver = make_serve_side_resolver(strokes, "far")
        rallies, report = strokes_to_golden(strokes, self.identity(), side_resolver=resolver)
        assert [h.side for h in rallies[0].hits] == ["far", "near"]

    def test_out_of_order_rally_skipped_not_forced(self):
        first = make_rally(1, 1, [2000, 2030], score=(1, 0))
        backwards = make_rally(1, 2, [500, 530], score=(1, 1))
        rallies, report = strokes_to_golden(first + backwards, self.identity())
        assert len(rallies) == 1
        assert any("out-of-order" in s for s in report.skipped)

    def test_illegal_score_transition_warned(self):
        strokes = make_rally(1, 1, [1000, 1030], score=(1, 0)) + make_rally(
            1, 2, [3000, 3030], score=(3, 0)  # two points in one rally: impossible
        )
        _, report = strokes_to_golden(strokes, self.identity())
        assert any("not a legal" in w for w in report.warnings)


def load_tool():
    path = Path(__file__).resolve().parent.parent / "tools" / "align_shuttleset.py"
    spec = importlib.util.spec_from_file_location("align_shuttleset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return load_tool()


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real 200-frame 30fps mp4 so the CLI's frame clamp and hash stamp are exercised."""
    path = tmp_path_factory.mktemp("video") / "mezz.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    assert writer.isOpened()
    for i in range(200):
        writer.write(np.full((48, 64, 3), i % 255, np.uint8))
    writer.release()
    return path


class TestParseAnchors:
    def test_parses_pairs(self, tool):
        assert tool.parse_anchors("10418:5030, 68544:63180") == [(10418, 5030), (68544, 63180)]

    def test_rejects_malformed(self, tool):
        with pytest.raises(ValueError, match="bad anchor"):
            tool.parse_anchors("10418-5030")

    def test_rejects_negative(self, tool):
        with pytest.raises(ValueError, match=">= 0"):
            tool.parse_anchors("-1:100,200:300")


class TestCLI:
    """End-to-end via main(argv): synthetic csv + real tiny mp4, no GUI, no network."""

    def _csv(self, tmp_path: Path) -> Path:
        # Two rallies whose frame_nums map onto the 200-frame video via mz = 0.5*ss.
        return write_csv(
            tmp_path / "set1.csv",
            [
                csv_row(1, 1, "100.0", 1, 0, "B", "發長球", loc_y="356.0"),
                csv_row(1, 2, "160.0", 1, 0, "A", "長球", loc_y="641.0", getpoint="A"),
                csv_row(2, 1, "260.0", 1, 1, "A", "發短球", loc_y="620.0"),
                csv_row(2, 2, "320.0", 1, 1, "B", "挑球", loc_y="350.0", getpoint="B"),
            ],
        )

    def _argv(self, csv: Path, golden: Path, video: Path, *extra: str) -> list[str]:
        return [
            "--csv", str(csv),
            "--golden", str(golden),
            "--video", str(video),
            "--anchors", "100:50,320:160",  # mz = 0.5*ss exactly
            "--match-id", "ss-align-test",
            "--video-uri", "local/mezz.mp4",
            *extra,
        ]

    def test_dry_run_prints_fit_and_writes_nothing(self, tool, tmp_path, tiny_video, capsys):
        golden = tmp_path / "golden.json"
        rc = tool.main(self._argv(self._csv(tmp_path), golden, tiny_video, "--dry-run"))
        assert rc == 0
        out = capsys.readouterr().out
        assert "0.500000" in out  # scale
        assert "residual" in out
        assert "2 rallies / 4 hits" in out
        assert "dry run" in out
        assert not golden.exists()

    def test_writes_golden_with_aligned_labels_and_hash(self, tool, tmp_path, tiny_video):
        golden = tmp_path / "golden.json"
        rc = tool.main(self._argv(self._csv(tmp_path), golden, tiny_video))
        assert rc == 0
        match = load_golden_match(golden)
        assert match.match_id == "ss-align-test"
        assert match.video_sha256 == sha256_file(tiny_video)
        assert len(match.rallies) == 2
        assert [h.frame for h in match.rallies[0].hits] == [50, 80]
        assert [h.side for h in match.rallies[0].hits] == ["far", "near"]
        assert [h.frame for h in match.rallies[1].hits] == [130, 160]
        assert match.rallies[0].start_frame == max(0, 50 - 45)
        assert match.rallies[1].end_frame == 199  # clamped to the video's last frame
        assert [(s.a, s.b) for s in match.score_timeline] == [(1, 0), (1, 1)]
        # Round-trips through the strict loader (already proven by load_golden_match).
        assert json.loads(golden.read_text())["video_uri"] == "local/mezz.mp4"

    def test_never_clobbers_existing_hand_labels(self, tool, tmp_path, tiny_video, capsys):
        golden = tmp_path / "golden.json"
        hand = GoldenMatch(
            match_id="ss-align-test",
            video_uri="local/mezz.mp4",
            video_sha256=sha256_file(tiny_video),
            rallies=[GoldenRally(start_frame=40, end_frame=90)],  # overlaps proposed rally 1
        )
        save_golden_match(hand, golden)
        rc = tool.main(self._argv(self._csv(tmp_path), golden, tiny_video))
        assert rc == 0
        assert "kept existing hand labels" in capsys.readouterr().out
        match = load_golden_match(golden)
        assert len(match.rallies) == 2  # hand rally + the non-overlapping proposal
        assert match.rallies[0].hits == []  # the hand-labelled rally is untouched
        assert [h.frame for h in match.rallies[1].hits] == [130, 160]

    def test_existing_score_timeline_wins(self, tool, tmp_path, tiny_video, capsys):
        golden = tmp_path / "golden.json"
        hand = GoldenMatch(
            match_id="ss-align-test",
            video_uri="local/mezz.mp4",
            video_sha256=sha256_file(tiny_video),
            score_timeline=[GoldenScoreEntry(frame=7, a=0, b=1)],
        )
        save_golden_match(hand, golden)
        tool.main(self._argv(self._csv(tmp_path), golden, tiny_video))
        assert "dropped all" in capsys.readouterr().out
        match = load_golden_match(golden)
        assert match.score_timeline == [GoldenScoreEntry(frame=7, a=0, b=1)]

    def test_serve_side_mode_flag(self, tool, tmp_path, tiny_video):
        golden = tmp_path / "golden.json"
        rc = tool.main(
            self._argv(self._csv(tmp_path), golden, tiny_video, "--near-serves-first")
        )
        assert rc == 0
        match = load_golden_match(golden)
        # B opens set 1 near; rally 2's server is A (winner of rally 1) → far end… no:
        # ends don't change within a set — A stays on the far end all set.
        assert [h.side for h in match.rallies[0].hits] == ["near", "far"]
        assert [h.side for h in match.rallies[1].hits] == ["far", "near"]

    def test_bad_anchor_refused_with_exit(self, tool, tmp_path, tiny_video):
        golden = tmp_path / "golden.json"
        argv = self._argv(self._csv(tmp_path), golden, tiny_video)
        argv[argv.index("100:50,320:160")] = "100:50,320:160,200:140"  # 40 frames off
        with pytest.raises(SystemExit, match="alignment rejected"):
            tool.main(argv)
        assert not golden.exists()

    def test_wrong_encode_refused(self, tool, tmp_path, tiny_video):
        golden = tmp_path / "golden.json"
        save_golden_match(
            GoldenMatch(match_id="ss-align-test", video_uri="u", video_sha256="0" * 64), golden
        )
        with pytest.raises(SystemExit, match="not the encode"):
            tool.main(self._argv(self._csv(tmp_path), golden, tiny_video))

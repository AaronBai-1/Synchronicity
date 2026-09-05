"""End-to-end API flow: create match → deliver pipeline results → read it all back.

The results payload is built from the real pipeline schemas (synchro_pipeline.schemas),
so these tests exercise the actual pipeline→API contract, not a hand-rolled imitation:
a 2-game match, 3 rallies, 10 shots, including a shot the pipeline could not attribute
(player=None) that ingest must resolve from the side-switch schedule.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from synchro_pipeline.domain.taxonomy import CANONICAL_TO_COARSE, CANONICAL_TO_MERGED, ShotType
from synchro_pipeline.schemas.records import (
    XY,
    GameSummary,
    MatchRecord,
    PlayerInfo,
    QualityReport,
    RallyConfidence,
    RallyRecord,
    ScorePair,
    ShotRecord,
    SourceMeta,
)

PIPELINE_MATCH_ID = "pm-001"
FPS = 30.0


# --- payload builders --------------------------------------------------------------


def _shot(
    rally_id: str,
    set_no: int,
    ball_round: int,
    player: str | None,
    side: str,
    canonical: ShotType,
    frame_num: int,
    **overrides: Any,
) -> ShotRecord:
    return ShotRecord(
        match_id=PIPELINE_MATCH_ID,
        set_no=set_no,
        rally_id=rally_id,
        ball_round=ball_round,
        frame_num=frame_num,
        t_ms=frame_num / FPS * 1000.0,
        player=player,  # type: ignore[arg-type]
        side=side,  # type: ignore[arg-type]
        type_canonical=canonical,
        type_merged=CANONICAL_TO_MERGED[canonical],
        type_coarse=CANONICAL_TO_COARSE[canonical],
        type_conf=0.9,
        hit_det_conf=0.95,
        **overrides,
    )


def _rally(
    rally_id: str,
    set_no: int,
    seq: int,
    start_frame: int,
    score_before: tuple[int, int],
    score_after: tuple[int, int],
    server: str,
    winner: str,
    n_strokes: int,
    **overrides: Any,
) -> RallyRecord:
    return RallyRecord(
        rally_id=rally_id,
        match_id=PIPELINE_MATCH_ID,
        set_no=set_no,
        seq=seq,
        start_frame=start_frame,
        end_frame=start_frame + n_strokes * 30 + 30,
        duration_s=(n_strokes * 30 + 30) / FPS,
        score_before=ScorePair(a=score_before[0], b=score_before[1]),
        score_after=ScorePair(a=score_after[0], b=score_after[1]),
        server=server,  # type: ignore[arg-type]
        winner=winner,  # type: ignore[arg-type]
        n_strokes=n_strokes,
        confidence=RallyConfidence(segmentation=0.98, ocr=0.99, hits=0.9, tracking=0.95),
        **overrides,
    )


def build_results_payload(*, with_names: bool = True) -> dict[str, Any]:
    """A valid 2-game match: 3 rallies, 10 shots. Game 1: A near; game 2: A far."""
    players: dict[str, PlayerInfo] = {}
    if with_names:
        players = {
            "A": PlayerInfo(name="Viktor Axelsen", handedness="right"),
            "B": PlayerInfo(name="Kento Momota", handedness="left"),
        }
    match = MatchRecord(
        match_id=PIPELINE_MATCH_ID,
        source_meta=SourceMeta(
            width=1280, height=720, fps_effective=FPS, duration_s=1800.0,
            broadcaster_guess="bwf",
        ),
        discipline="MS",
        players=players,  # type: ignore[arg-type]
        first_server="A",
        a_on_near_side_at_start=True,
        sets=[
            GameSummary(set_no=1, final_score=ScorePair(a=21, b=15), rally_ids=["r1", "r2"]),
            GameSummary(set_no=2, final_score=ScorePair(a=21, b=18), rally_ids=["r3"]),
        ],
        pipeline_version="0.1.0",
        stage_model_hashes={"s2_court": "abc123"},
        quality_report=QualityReport(
            pct_rallies_clean=66.7,
            pct_strokes_classified=100.0,
            flags_summary={"ocr_low_conf": 1},
        ),
    )
    rallies = [
        _rally("r1", 1, 1, 1000, (0, 0), (1, 0), "A", "A", 4,
               serve_type="short", lose_reason="out", score_verified=True,
               clip_uri="file:///clips/r1.mp4"),
        _rally("r2", 1, 2, 2000, (1, 0), (1, 1), "A", "B", 3,
               serve_type="short", lose_reason="net", score_verified=True),
        _rally("r3", 2, 1, 3000, (0, 0), (1, 0), "A", "A", 3,
               serve_type="long", score_verified=False, qa_flags=["ocr_low_conf"]),
    ]
    shots = [
        # r1 (game 1: A near, B far) — B's 4th stroke goes out, A wins the point
        _shot("r1", 1, 1, "A", "near", ShotType.SHORT_SERVICE, 1010,
              hit_xy_court=XY(x=0.5, y=-2.0), landing_xy_court=XY(x=-1.0, y=2.5),
              landing_area=6, player_location=XY(x=0.5, y=-2.0),
              opponent_location=XY(x=0.0, y=3.0)),
        _shot("r1", 1, 2, "B", "far", ShotType.RETURN_NET, 1040,
              hit_xy_court=XY(x=-1.0, y=2.2)),
        _shot("r1", 1, 3, "A", "near", ShotType.LOB, 1070, backhand=True),
        _shot("r1", 1, 4, "B", "far", ShotType.LOB, 1100,
              is_error=True, lose_reason="out"),
        # r2 (game 1) — A nets a drive, B wins the point
        _shot("r2", 1, 1, "A", "near", ShotType.SHORT_SERVICE, 2010),
        _shot("r2", 1, 2, "B", "far", ShotType.PUSH, 2040),
        _shot("r2", 1, 3, "A", "near", ShotType.DRIVE, 2070,
              is_error=True, lose_reason="net"),
        # r3 (game 2: sides swapped, A far) — ball_round 2 is unattributed:
        # ingest must resolve side "near" → B from the side-switch schedule
        _shot("r3", 2, 1, "A", "far", ShotType.LONG_SERVICE, 3010),
        _shot("r3", 2, 2, None, "near", ShotType.CLEAR, 3040, landing_height="high"),
        _shot("r3", 2, 3, "A", "far", ShotType.SMASH, 3070,
              is_winner=True, shot_speed_2d_kmh=250.0, speed_conf_tier="B",
              type_probs={"smash": 0.9, "drive": 0.1}, traj_quality=0.85),
    ]
    return {
        "match": match.model_dump(mode="json"),
        # deliberately out of play order: ingest/list endpoints must impose ordering
        "rallies": [r.model_dump(mode="json") for r in (rallies[2], rallies[0], rallies[1])],
        "shots": [s.model_dump(mode="json") for s in shots],
    }


def _create_match(client: TestClient) -> dict[str, Any]:
    resp = client.post("/matches", json={"filename": "match.mp4", "discipline": "MS"})
    assert resp.status_code == 201, resp.text
    return resp.json()


def _ingest(client: TestClient, *, with_names: bool = True) -> tuple[str, dict[str, Any]]:
    match_id = _create_match(client)["match"]["id"]
    payload = build_results_payload(with_names=with_names)
    resp = client.post(f"/matches/{match_id}/results", json=payload)
    assert resp.status_code == 200, resp.text
    return match_id, resp.json()


# --- tests -------------------------------------------------------------------------


def test_create_match_returns_upload_target(client: TestClient) -> None:
    created = _create_match(client)
    match, upload = created["match"], created["upload"]
    assert match["status"] == "awaiting_upload"
    assert match["discipline"] == "MS"
    assert upload["method"] == "file"
    assert upload["video_uri"].startswith("file://")
    assert upload["video_uri"].endswith("source.mp4")
    assert match["video_uri"] == upload["video_uri"]

    got = client.get(f"/matches/{match['id']}")
    assert got.status_code == 200
    assert got.json()["status"] == "awaiting_upload"


def test_results_roundtrip_scores_and_ordering(client: TestClient) -> None:
    match_id, counts = _ingest(client)
    assert counts == {"match_id": match_id, "status": "ready", "games": 2,
                      "rallies": 3, "shots": 10}

    match = client.get(f"/matches/{match_id}").json()
    assert match["status"] == "ready"
    assert match["pipeline_match_id"] == PIPELINE_MATCH_ID
    assert match["confidence_report"]["pct_rallies_clean"] == pytest.approx(66.7)
    assert match["confidence_report"]["flags_summary"] == {"ocr_low_conf": 1}
    assert match["players"]["a"]["name"] == "Viktor Axelsen"
    assert match["players"]["b"]["name"] == "Kento Momota"
    assert match["first_server"] == "A"

    games = match["games"]
    assert [g["seq"] for g in games] == [1, 2]
    assert games[0]["final_score"] == {"a": 21, "b": 15}
    assert games[1]["final_score"] == {"a": 21, "b": 18}
    # ends swap between games: A starts near, plays game 2 from the far end
    assert games[0]["side_assignment"] == {"near": "A", "far": "B"}
    assert games[1]["side_assignment"] == {"near": "B", "far": "A"}

    rallies = client.get(f"/matches/{match_id}/rallies").json()["rallies"]
    assert [r["pipeline_rally_id"] for r in rallies] == ["r1", "r2", "r3"]  # play order
    assert [(r["game_seq"], r["seq"]) for r in rallies] == [(1, 1), (1, 2), (2, 1)]
    assert rallies[0]["score_before"] == {"a": 0, "b": 0}
    assert rallies[0]["score_after"] == {"a": 1, "b": 0}
    assert rallies[1]["score_before"] == {"a": 1, "b": 0}
    assert rallies[0]["verified"] is True
    assert rallies[2]["verified"] is False
    assert rallies[2]["qa_flags"] == ["ocr_low_conf"]
    assert rallies[0]["end_reason"] == "out"
    assert rallies[0]["clip_uri"] == "file:///clips/r1.mp4"
    # named players → server/winner FKs resolved at ingest
    player_a_id = match["players"]["a"]["id"]
    player_b_id = match["players"]["b"]["id"]
    assert rallies[0]["server_player_id"] == player_a_id
    assert rallies[0]["winner_player_id"] == player_a_id
    assert rallies[1]["winner_player_id"] == player_b_id


def test_rally_detail_shot_fields_survive(client: TestClient) -> None:
    match_id, _ = _ingest(client)
    rallies = client.get(f"/matches/{match_id}/rallies").json()["rallies"]
    by_pid = {r["pipeline_rally_id"]: r for r in rallies}

    r1 = client.get(f"/rallies/{by_pid['r1']['id']}").json()
    shots = r1["shots"]
    assert [s["seq"] for s in shots] == [1, 2, 3, 4]  # stroke order
    serve = shots[0]
    assert serve["shot_type_canonical"] == "short service"
    assert serve["shot_type_merged"] == "serve_short"
    assert serve["shot_type_coarse"] == "serve"
    assert serve["hit_xy_court"] == {"x": 0.5, "y": -2.0}
    assert serve["landing_xy_court"] == {"x": -1.0, "y": 2.5}
    assert serve["landing_area"] == 6
    assert serve["player_location"] == {"x": 0.5, "y": -2.0}
    assert serve["opponent_location"] == {"x": 0.0, "y": 3.0}
    assert shots[2]["backhand"] is True
    assert shots[3]["is_error"] is True
    assert shots[3]["lose_reason"] == "out"

    r3 = client.get(f"/rallies/{by_pid['r3']['id']}").json()
    smash = r3["shots"][2]
    assert smash["shot_type_canonical"] == "smash"
    assert smash["is_winner"] is True
    assert smash["shot_speed_2d_kmh"] == pytest.approx(250.0)
    assert smash["speed_method"] == "court_plane_2d"
    assert smash["speed_conf_tier"] == "B"
    assert smash["type_probs"] == {"smash": 0.9, "drive": 0.1}


def test_side_resolution_and_profile(client: TestClient) -> None:
    match_id, _ = _ingest(client)
    match = client.get(f"/matches/{match_id}").json()
    player_a_id = match["players"]["a"]["id"]
    player_b_id = match["players"]["b"]["id"]

    rallies = client.get(f"/matches/{match_id}/rallies").json()["rallies"]
    r3_row_id = next(r["id"] for r in rallies if r["pipeline_rally_id"] == "r3")
    r3 = client.get(f"/rallies/{r3_row_id}").json()
    unattributed = r3["shots"][1]
    # pipeline sent player=None, side="near"; in game 2 the near end is B's
    assert unattributed["player_ref"] == "B"
    assert unattributed["player_id"] == player_b_id

    profile = client.get(f"/players/{player_a_id}/profile").json()
    assert profile["player"]["name"] == "Viktor Axelsen"
    assert profile["total_shots"] == 6
    assert profile["shots_by_coarse_type"] == {"serve": 3, "lift": 1, "drive": 1, "smash": 1}
    assert profile["winners"] == 1
    assert profile["errors"] == 1
    assert profile["rallies_played"] == 3
    assert profile["rallies_won"] == 2
    assert profile["avg_rally_shots"] == pytest.approx(10 / 3)


def test_reingest_is_idempotent(client: TestClient) -> None:
    match_id, first = _ingest(client)
    resp = client.post(f"/matches/{match_id}/results", json=build_results_payload())
    assert resp.status_code == 200, resp.text
    assert resp.json() == first  # identical counts — replaced, not duplicated

    match = client.get(f"/matches/{match_id}").json()
    assert len(match["games"]) == 2
    rallies = client.get(f"/matches/{match_id}/rallies").json()["rallies"]
    assert len(rallies) == 3
    profile = client.get(f"/players/{match['players']['a']['id']}/profile").json()
    assert profile["total_shots"] == 6  # players not duplicated either


def test_identity_assignment_backfills_fks(client: TestClient) -> None:
    match_id, _ = _ingest(client, with_names=False)
    match = client.get(f"/matches/{match_id}").json()
    assert match["players"] == {"a": None, "b": None}
    rallies = client.get(f"/matches/{match_id}/rallies").json()["rallies"]
    assert all(r["server_player_id"] is None for r in rallies)

    resp = client.post(f"/matches/{match_id}/identity", json={"near": "Alice", "far": "Bob"})
    assert resp.status_code == 200, resp.text
    match = resp.json()
    # a_on_near_side_at_start is True → near name is player A
    assert match["players"]["a"]["name"] == "Alice"
    assert match["players"]["b"]["name"] == "Bob"
    alice_id = match["players"]["a"]["id"]
    bob_id = match["players"]["b"]["id"]

    rallies = client.get(f"/matches/{match_id}/rallies").json()["rallies"]
    for rally in rallies:
        assert rally["server_player_id"] == (
            alice_id if rally["server_ref"] == "A" else bob_id
        )
        assert rally["winner_player_id"] == (
            alice_id if rally["winner_ref"] == "A" else bob_id
        )
    r1 = client.get(f"/rallies/{rallies[0]['id']}").json()
    assert [s["player_id"] for s in r1["shots"]] == [alice_id, bob_id, alice_id, bob_id]

    profile = client.get(f"/players/{alice_id}/profile").json()
    assert profile["total_shots"] == 6


def test_workspace_isolation(client: TestClient) -> None:
    match_id, _ = _ingest(client)
    other = {"X-Workspace-Id": "other-team"}
    assert client.get(f"/matches/{match_id}", headers=other).status_code == 404
    assert client.get(f"/matches/{match_id}/rallies", headers=other).status_code == 404
    rallies = client.get(f"/matches/{match_id}/rallies").json()["rallies"]
    assert client.get(f"/rallies/{rallies[0]['id']}", headers=other).status_code == 404
    match = client.get(f"/matches/{match_id}").json()
    player_a_id = match["players"]["a"]["id"]
    assert client.get(f"/players/{player_a_id}/profile", headers=other).status_code == 404

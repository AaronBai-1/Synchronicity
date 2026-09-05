"""Row → JSON-dict serializers shared by the routers.

Hand-rolled (rather than ORM-coupled response models) so the wire format stays an
explicit, reviewable contract: grouped court coordinates mirror the pipeline's
XY sub-objects, enum columns surface as their ShuttleSet string values, and NULLs pass
through as JSON nulls — "null + flag beats wrong" applies to the API surface too.
"""

from __future__ import annotations

from typing import Any

from synchro_api.models import Game, Match, Rally, Shot


def _score(a: int | None, b: int | None) -> dict[str, int] | None:
    if a is None and b is None:
        return None
    return {"a": a, "b": b}  # type: ignore[dict-item]  # one-sided scores never occur


def _xy(x: float | None, y: float | None) -> dict[str, float] | None:
    if x is None and y is None:
        return None
    return {"x": x, "y": y}  # type: ignore[dict-item]


def _player_brief(player_id: str | None, name: str | None) -> dict[str, Any] | None:
    if player_id is None:
        return None
    return {"id": player_id, "name": name}


def match_to_dict(match: Match, games: list[Game] | None = None) -> dict[str, Any]:
    return {
        "id": match.id,
        "workspace_id": match.workspace_id,
        "status": match.status.value,
        "video_uri": match.video_uri,
        "discipline": match.discipline,
        "pipeline_match_id": match.pipeline_match_id,
        "pipeline_version": match.pipeline_version,
        "source_meta": match.source_meta,
        "confidence_report": match.confidence_report,
        "players": {
            "a": _player_brief(match.player_a_id, match.player_a.name if match.player_a else None),
            "b": _player_brief(match.player_b_id, match.player_b.name if match.player_b else None),
        },
        "first_server": match.first_server,
        "a_on_near_side_at_start": match.a_on_near_side_at_start,
        "games": [game_to_dict(g) for g in games] if games is not None else None,
        "created_at": match.created_at.isoformat(),
    }


def game_to_dict(game: Game) -> dict[str, Any]:
    return {
        "id": game.id,
        "match_id": game.match_id,
        "seq": game.seq,
        "final_score": _score(game.final_score_a, game.final_score_b),
        "side_assignment": game.side_assignment,
    }


def rally_to_dict(rally: Rally, *, game_seq: int | None = None) -> dict[str, Any]:
    return {
        "id": rally.id,
        "match_id": rally.match_id,
        "game_id": rally.game_id,
        "game_seq": game_seq,
        "pipeline_rally_id": rally.pipeline_rally_id,
        "seq": rally.seq,
        "start_frame": rally.start_frame,
        "end_frame": rally.end_frame,
        "duration_s": rally.duration_s,
        "score_before": _score(rally.score_before_a, rally.score_before_b),
        "score_after": _score(rally.score_after_a, rally.score_after_b),
        "server_ref": rally.server_ref,
        "server_player_id": rally.server_player_id,
        "winner_ref": rally.winner_ref,
        "winner_player_id": rally.winner_player_id,
        "serve_type": rally.serve_type,
        "end_reason": rally.end_reason,
        "shot_count": rally.shot_count,
        "verified": rally.verified,
        "confidence": rally.confidence,
        "qa_flags": rally.qa_flags or [],
        "clip_uri": rally.clip_uri,
    }


def shot_to_dict(shot: Shot) -> dict[str, Any]:
    return {
        "id": shot.id,
        "rally_id": shot.rally_id,
        "match_id": shot.match_id,
        "seq": shot.seq,
        "frame_num": shot.frame_num,
        "t_ms": shot.t_ms,
        "side": shot.side,
        "player_ref": shot.player_ref,
        "player_id": shot.player_id,
        "shot_type_canonical": shot.shot_type_canonical,
        "shot_type_merged": shot.shot_type_merged,
        "shot_type_coarse": shot.shot_type_coarse,
        "type_probs": shot.type_probs,
        "type_conf": shot.type_conf,
        "backhand": shot.backhand,
        "aroundhead": shot.aroundhead,
        "landing_height": shot.landing_height,
        "hit_xy_court": _xy(shot.hit_x, shot.hit_y),
        "landing_xy_court": _xy(shot.landing_x, shot.landing_y),
        "landing_area": shot.landing_area,
        "player_location": _xy(shot.player_x, shot.player_y),
        "player_location_area": shot.player_location_area,
        "opponent_location": _xy(shot.opponent_x, shot.opponent_y),
        "opponent_location_area": shot.opponent_location_area,
        "shot_speed_2d_kmh": shot.shot_speed_2d_kmh,
        "shot_speed_3d_kmh": shot.shot_speed_3d_kmh,
        "speed_method": shot.speed_method,
        "speed_conf_tier": shot.speed_conf_tier,
        "is_winner": shot.is_winner,
        "is_error": shot.is_error,
        "lose_reason": shot.lose_reason,
        "hit_det_conf": shot.hit_det_conf,
        "traj_quality": shot.traj_quality,
        "qa_flags": shot.qa_flags or [],
    }

"""Pipeline → API ingest: the single write path for analysis results.

This is the contract between the offline pipeline (docs/plan.md stages S0–S7, delivered
as pydantic records from synchro_pipeline.schemas) and the platform's Postgres data
model. Three rules govern every mapping decision:

    * "null + flag beats wrong" — anything the pipeline could not determine stays NULL;
      we never guess a player, score, or side.
    * idempotent per match — re-ingesting (pipeline re-run, bug-fix backfill) replaces
      that match's games/rallies/shots wholesale instead of duplicating rows.
    * near/far → player resolution reuses the side-switch schedule from
      domain.scoring: ends swap after every game and, in the deciding game, when the
      leading score first reaches 11. Where the schedule cannot be evaluated (unknown
      start side, unknown score in the deciding game) the ref — and hence the player
      FK — stays NULL.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session
from synchro_pipeline.domain.scoring import DECIDING_GAME_SWITCH_SCORE, GAMES_TO_WIN_MATCH
from synchro_pipeline.schemas.records import (
    MatchRecord,
    RallyRecord,
    ScorePair,
    ShotRecord,
)

from synchro_api.models import Game, Match, MatchStatus, Player, Rally, Shot

_DECIDING_SET_NO = 2 * GAMES_TO_WIN_MATCH - 1  # game 3 in best-of-3


# --- side → ref resolution ---------------------------------------------------------


def _a_on_near_at_game_start(a_on_near_at_match_start: bool | None, set_no: int) -> bool | None:
    """Ends swap after every game (domain.scoring), so game parity flips the start side."""
    if a_on_near_at_match_start is None:
        return None
    flipped = (set_no - 1) % 2 == 1
    return a_on_near_at_match_start != flipped


def _a_on_near_during_rally(
    a_on_near_at_match_start: bool | None, set_no: int, score_before: ScorePair | None
) -> bool | None:
    """Which end player A occupies during a given rally, or None if unknowable.

    In the deciding game the ends swap again when the leading score first reaches 11
    (DECIDING_GAME_SWITCH_SCORE); without a trusted score_before we abstain rather than
    risk attributing every late-game shot to the wrong player.
    """
    at_start = _a_on_near_at_game_start(a_on_near_at_match_start, set_no)
    if at_start is None:
        return None
    if set_no != _DECIDING_SET_NO:
        return at_start
    if score_before is None:
        return None
    if max(score_before.a, score_before.b) >= DECIDING_GAME_SWITCH_SCORE:
        return not at_start
    return at_start


def _ref_for_side(
    side: str,
    set_no: int,
    a_on_near_at_match_start: bool | None,
    score_before: ScorePair | None,
) -> str | None:
    """Resolve a physical side ("near"/"far") to a record ref ("A"/"B"), or None."""
    a_near = _a_on_near_during_rally(a_on_near_at_match_start, set_no, score_before)
    if a_near is None:
        return None
    if side == "near":
        return "A" if a_near else "B"
    return "B" if a_near else "A"


def _side_assignment(a_on_near_at_match_start: bool | None, set_no: int) -> dict[str, str | None]:
    """Game-start near/far → ref mapping stored on the game row (JSON)."""
    a_near = _a_on_near_at_game_start(a_on_near_at_match_start, set_no)
    if a_near is None:
        return {"near": None, "far": None}
    return {"near": "A" if a_near else "B", "far": "B" if a_near else "A"}


# --- players -----------------------------------------------------------------------


def _get_or_create_player(
    session: Session, workspace_id: str, name: str, handedness: str | None = None
) -> Player:
    """Players are workspace-scoped and keyed by name (auto naming is a v1 non-goal)."""
    existing = session.scalar(
        select(Player).where(Player.workspace_id == workspace_id, Player.name == name)
    )
    if existing is not None:
        if handedness is not None and existing.handedness is None:
            existing.handedness = handedness
        return existing
    player = Player(workspace_id=workspace_id, name=name, handedness=handedness)
    session.add(player)
    session.flush()
    return player


# --- main entry points -------------------------------------------------------------


def ingest_match_results(
    session: Session,
    workspace_id: str,
    match: MatchRecord,
    rallies: Sequence[RallyRecord],
    shots: Sequence[ShotRecord],
    *,
    match_row_id: str | None = None,
) -> Match:
    """Map pipeline records into rows for one match. Returns the (updated) match row.

    Targets the existing match row `match_row_id` (the upload the results belong to);
    when omitted — e.g. CLI backfill — the match is found or created by the pipeline's
    own match_id within the workspace. Re-ingest replaces the match's games, rallies
    and shots; the caller owns the transaction (this function only flushes).

    Raises ValueError on contract violations (wrong workspace, records referencing
    unknown rally/match ids) — the router surfaces these as 422s.
    """
    match_row = _resolve_match_row(session, workspace_id, match, match_row_id)

    # Idempotency: wipe this match's previous analysis before re-inserting (FK order).
    for table in (Shot, Rally, Game):
        session.execute(delete(table).where(table.match_id == match_row.id))

    # A/B → player rows, when the pipeline (or the user, earlier) knows names.
    player_ids: dict[str, str | None] = {"A": None, "B": None}
    for ref in ("A", "B"):
        info = match.players.get(ref)
        if info is not None and info.name:
            player_ids[ref] = _get_or_create_player(
                session, workspace_id, info.name, info.handedness
            ).id
    match_row.player_a_id = player_ids["A"]
    match_row.player_b_id = player_ids["B"]

    match_row.status = MatchStatus.READY
    match_row.discipline = match.discipline
    match_row.pipeline_match_id = match.match_id
    match_row.pipeline_version = match.pipeline_version
    match_row.source_meta = match.source_meta.model_dump(mode="json")
    match_row.stage_model_hashes = dict(match.stage_model_hashes)
    match_row.confidence_report = match.quality_report.model_dump(mode="json")
    match_row.first_server = match.first_server
    match_row.a_on_near_side_at_start = match.a_on_near_side_at_start

    a_near_start = match.a_on_near_side_at_start

    # Games: union of the match summary's sets and any set_no the rallies mention.
    summaries = {gs.set_no: gs for gs in match.sets}
    set_nos = sorted(set(summaries) | {r.set_no for r in rallies})
    game_rows: dict[int, Game] = {}
    for set_no in set_nos:
        summary = summaries.get(set_no)
        game_rows[set_no] = Game(
            match_id=match_row.id,
            seq=set_no,
            final_score_a=summary.final_score.a if summary else None,
            final_score_b=summary.final_score.b if summary else None,
            side_assignment=_side_assignment(a_near_start, set_no),
        )
    session.add_all(game_rows.values())
    session.flush()

    # Rallies, ordered (game, seq) regardless of payload order.
    rally_rows: dict[str, Rally] = {}
    rally_recs: dict[str, RallyRecord] = {}
    for rec in sorted(rallies, key=lambda r: (r.set_no, r.seq)):
        if rec.match_id != match.match_id:
            raise ValueError(
                f"rally {rec.rally_id!r} belongs to match {rec.match_id!r}, "
                f"payload is for {match.match_id!r}"
            )
        row = Rally(
            game_id=game_rows[rec.set_no].id,
            match_id=match_row.id,
            pipeline_rally_id=rec.rally_id,
            seq=rec.seq,
            start_frame=rec.start_frame,
            end_frame=rec.end_frame,
            duration_s=rec.duration_s,
            score_before_a=rec.score_before.a if rec.score_before else None,
            score_before_b=rec.score_before.b if rec.score_before else None,
            score_after_a=rec.score_after.a if rec.score_after else None,
            score_after_b=rec.score_after.b if rec.score_after else None,
            server_ref=rec.server,
            server_player_id=player_ids.get(rec.server) if rec.server else None,
            winner_ref=rec.winner,
            winner_player_id=player_ids.get(rec.winner) if rec.winner else None,
            serve_type=rec.serve_type,
            end_reason=rec.lose_reason,
            shot_count=rec.n_strokes,
            verified=rec.score_verified,
            confidence=rec.confidence.model_dump(mode="json"),
            qa_flags=list(rec.qa_flags),
            clip_uri=rec.clip_uri,
        )
        rally_rows[rec.rally_id] = row
        rally_recs[rec.rally_id] = rec
    session.add_all(rally_rows.values())
    session.flush()

    for shot in shots:
        rally_row = rally_rows.get(shot.rally_id)
        if rally_row is None:
            raise ValueError(f"shot at frame {shot.frame_num} references unknown rally_id "
                             f"{shot.rally_id!r}")
        rally_rec = rally_recs[shot.rally_id]
        # Prefer the pipeline's own attribution; fall back to the side-switch schedule.
        ref = shot.player or _ref_for_side(
            shot.side, rally_rec.set_no, a_near_start, rally_rec.score_before
        )
        session.add(_shot_row(shot, rally_row, match_row, ref, player_ids))
    session.flush()
    return match_row


def _resolve_match_row(
    session: Session, workspace_id: str, match: MatchRecord, match_row_id: str | None
) -> Match:
    if match_row_id is not None:
        row = session.get(Match, match_row_id)
        if row is None or row.workspace_id != workspace_id:
            raise ValueError(f"match {match_row_id!r} not found in workspace {workspace_id!r}")
        return row
    row = session.scalar(
        select(Match).where(
            Match.workspace_id == workspace_id, Match.pipeline_match_id == match.match_id
        )
    )
    if row is None:
        row = Match(workspace_id=workspace_id, pipeline_match_id=match.match_id)
        session.add(row)
        session.flush()
    return row


def _shot_row(
    shot: ShotRecord,
    rally_row: Rally,
    match_row: Match,
    ref: str | None,
    player_ids: dict[str, str | None],
) -> Shot:
    """One ShotRecord → one shots row; enum values become their ShuttleSet strings.

    speed_method documents how a printed speed was produced (plan S7: v1 speeds are 2D
    court-plane estimates surfaced as bands labeled "estimated"; a 3D lift is deferred).
    """
    if shot.shot_speed_2d_kmh is not None:
        speed_method = "court_plane_2d"
    elif shot.shot_speed_3d_kmh is not None:
        speed_method = "3d"
    else:
        speed_method = None
    return Shot(
        rally_id=rally_row.id,
        match_id=match_row.id,
        seq=shot.ball_round,
        frame_num=shot.frame_num,
        t_ms=shot.t_ms,
        side=shot.side,
        player_ref=ref,
        player_id=player_ids.get(ref) if ref else None,
        shot_type_canonical=shot.type_canonical.value if shot.type_canonical else None,
        shot_type_merged=shot.type_merged.value if shot.type_merged else None,
        shot_type_coarse=shot.type_coarse.value if shot.type_coarse else None,
        type_probs=(
            {k.value: float(v) for k, v in shot.type_probs.items()} if shot.type_probs else None
        ),
        type_conf=shot.type_conf,
        backhand=shot.backhand,
        aroundhead=shot.aroundhead,
        landing_height=shot.landing_height,
        hit_x=shot.hit_xy_court.x if shot.hit_xy_court else None,
        hit_y=shot.hit_xy_court.y if shot.hit_xy_court else None,
        landing_x=shot.landing_xy_court.x if shot.landing_xy_court else None,
        landing_y=shot.landing_xy_court.y if shot.landing_xy_court else None,
        landing_area=shot.landing_area,
        player_x=shot.player_location.x if shot.player_location else None,
        player_y=shot.player_location.y if shot.player_location else None,
        player_location_area=shot.player_location_area,
        opponent_x=shot.opponent_location.x if shot.opponent_location else None,
        opponent_y=shot.opponent_location.y if shot.opponent_location else None,
        opponent_location_area=shot.opponent_location_area,
        shot_speed_2d_kmh=shot.shot_speed_2d_kmh,
        shot_speed_3d_kmh=shot.shot_speed_3d_kmh,
        speed_method=speed_method,
        speed_conf_tier=shot.speed_conf_tier,
        is_winner=shot.is_winner,
        is_error=shot.is_error,
        lose_reason=shot.lose_reason,
        hit_det_conf=shot.hit_det_conf,
        traj_quality=shot.traj_quality,
        qa_flags=list(shot.qa_flags),
    )


def assign_identity(
    session: Session, match_row: Match, *, near: str | None, far: str | None
) -> Match:
    """Human near/far → player-name assignment (plan: one-click manual naming, not auto).

    `near`/`far` name the players as seen at the *start of the match*. If the pipeline
    never determined which record-ref started near (a_on_near_side_at_start is NULL),
    the refs "A"/"B" carry no side meaning yet, so we adopt near=A as the convention and
    persist it — this is human ground truth pinning a free variable, not a guess.

    Backfills player FKs on this match's rallies (server/winner) and shots from the
    stored "A"/"B" refs; rows whose ref could not be resolved at ingest stay NULL.
    """
    if match_row.a_on_near_side_at_start is None:
        match_row.a_on_near_side_at_start = True
    if match_row.a_on_near_side_at_start:
        a_name, b_name = near, far
    else:
        a_name, b_name = far, near

    if a_name:
        match_row.player_a_id = _get_or_create_player(
            session, match_row.workspace_id, a_name
        ).id
    if b_name:
        match_row.player_b_id = _get_or_create_player(
            session, match_row.workspace_id, b_name
        ).id

    for ref, player_id in (("A", match_row.player_a_id), ("B", match_row.player_b_id)):
        if player_id is None:
            continue
        session.execute(
            update(Rally)
            .where(Rally.match_id == match_row.id, Rally.server_ref == ref)
            .values(server_player_id=player_id)
        )
        session.execute(
            update(Rally)
            .where(Rally.match_id == match_row.id, Rally.winner_ref == ref)
            .values(winner_player_id=player_id)
        )
        session.execute(
            update(Shot)
            .where(Shot.match_id == match_row.id, Shot.player_ref == ref)
            .values(player_id=player_id)
        )
    session.flush()
    return match_row

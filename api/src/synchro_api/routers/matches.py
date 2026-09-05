"""Match lifecycle routes: create/upload, results delivery, status, rallies, identity.

POST /matches/{id}/results is how the offline pipeline delivers: the request body is
the pipeline's own pydantic records ({match, rallies, shots}) validated against
synchro_pipeline.schemas — the API and pipeline share one schema definition by import,
so the contract cannot drift silently (docs/plan.md: "FastAPI + Pydantic v2, shared
schema models with the pipeline").
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from synchro_pipeline.schemas.records import Discipline, MatchRecord, RallyRecord, ShotRecord

from synchro_api.deps import SessionDep, WorkspaceDep
from synchro_api.ingest import assign_identity, ingest_match_results
from synchro_api.models import Game, Match, MatchStatus, Rally, Shot, Workspace
from synchro_api.serialize import match_to_dict, rally_to_dict
from synchro_api.storage import get_storage

router = APIRouter(prefix="/matches", tags=["matches"])


class MatchCreate(BaseModel):
    filename: str | None = None  # original upload name; only its extension is trusted
    discipline: Discipline | None = None


class ResultsPayload(BaseModel):
    """Exactly what the pipeline emits — see synchro_pipeline.schemas.records."""

    match: MatchRecord
    rallies: list[RallyRecord] = Field(default_factory=list)
    shots: list[ShotRecord] = Field(default_factory=list)


class IdentityAssign(BaseModel):
    """Names as seen at the start of the match (near = closest to camera)."""

    near: str | None = None
    far: str | None = None


def _get_match(session: SessionDep, workspace: Workspace, match_id: str) -> Match:
    match = session.get(Match, match_id)
    if match is None or match.workspace_id != workspace.id:
        raise HTTPException(status_code=404, detail=f"match {match_id!r} not found")
    return match


@router.post("", status_code=201)
def create_match(body: MatchCreate, session: SessionDep, workspace: WorkspaceDep) -> dict:
    """Create a match and hand back an upload target for its source video.

    The client uploads directly to storage (presigned in production, file:// in dev) —
    video bytes never pass through the API (docs/plan.md: R2, zero egress).
    """
    match = Match(
        workspace_id=workspace.id,
        status=MatchStatus.AWAITING_UPLOAD,
        discipline=body.discipline,
    )
    session.add(match)
    session.flush()
    target = get_storage().create_upload_target(match.id, filename=body.filename)
    match.video_uri = target.video_uri
    session.commit()
    return {"match": match_to_dict(match, []), "upload": asdict(target)}


@router.get("/{match_id}")
def get_match(match_id: str, session: SessionDep, workspace: WorkspaceDep) -> dict:
    """Match status + confidence report + per-game summaries."""
    match = _get_match(session, workspace, match_id)
    games = list(session.scalars(select(Game).where(Game.match_id == match.id).order_by(Game.seq)))
    return match_to_dict(match, games)


@router.post("/{match_id}/results")
def post_results(
    match_id: str, payload: ResultsPayload, session: SessionDep, workspace: WorkspaceDep
) -> dict:
    """Pipeline results delivery. Idempotent: re-posting replaces this match's analysis."""
    match = _get_match(session, workspace, match_id)
    try:
        ingest_match_results(
            session,
            workspace.id,
            payload.match,
            payload.rallies,
            payload.shots,
            match_row_id=match.id,
        )
    except ValueError as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    counts = {
        "games": session.scalar(
            select(func.count()).select_from(Game).where(Game.match_id == match.id)
        ),
        "rallies": session.scalar(
            select(func.count()).select_from(Rally).where(Rally.match_id == match.id)
        ),
        "shots": session.scalar(
            select(func.count()).select_from(Shot).where(Shot.match_id == match.id)
        ),
    }
    session.commit()
    return {"match_id": match.id, "status": match.status.value, **counts}


@router.get("/{match_id}/rallies")
def list_rallies(match_id: str, session: SessionDep, workspace: WorkspaceDep) -> dict:
    """All rallies of a match in play order (game seq, then rally seq)."""
    match = _get_match(session, workspace, match_id)
    rows = session.execute(
        select(Rally, Game.seq)
        .join(Game, Rally.game_id == Game.id)
        .where(Rally.match_id == match.id)
        .order_by(Game.seq, Rally.seq)
    ).all()
    return {
        "match_id": match.id,
        "rallies": [rally_to_dict(rally, game_seq=game_seq) for rally, game_seq in rows],
    }


@router.post("/{match_id}/identity")
def post_identity(
    match_id: str, body: IdentityAssign, session: SessionDep, workspace: WorkspaceDep
) -> dict:
    """One-click near/far → player-name assignment (auto naming is a v1 non-goal)."""
    match = _get_match(session, workspace, match_id)
    assign_identity(session, match, near=body.near, far=body.far)
    session.commit()
    games = list(session.scalars(select(Game).where(Game.match_id == match.id).order_by(Game.seq)))
    return match_to_dict(match, games)

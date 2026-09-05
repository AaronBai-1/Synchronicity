"""Rally detail route — a rally with its ordered shots.

This is the payload behind the stat→clip loop (docs/plan.md dashboard: "every stat
click-through to video clips"): the rally player renders clip_uri plus the ordered shot
list for the synced top-down court animation.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from synchro_api.deps import SessionDep, WorkspaceDep
from synchro_api.models import Game, Match, Rally, Shot
from synchro_api.serialize import rally_to_dict, shot_to_dict

router = APIRouter(prefix="/rallies", tags=["rallies"])


@router.get("/{rally_id}")
def get_rally(rally_id: str, session: SessionDep, workspace: WorkspaceDep) -> dict:
    """One rally with its shots in stroke order (seq == ball_round, 1 = serve)."""
    rally = session.get(Rally, rally_id)
    if rally is not None:
        match = session.get(Match, rally.match_id)
        if match is None or match.workspace_id != workspace.id:
            rally = None
    if rally is None:
        raise HTTPException(status_code=404, detail=f"rally {rally_id!r} not found")
    game_seq = session.scalar(select(Game.seq).where(Game.id == rally.game_id))
    shots = session.scalars(
        select(Shot).where(Shot.rally_id == rally.id).order_by(Shot.seq)
    ).all()
    payload = rally_to_dict(rally, game_seq=game_seq)
    payload["shots"] = [shot_to_dict(s) for s in shots]
    return payload

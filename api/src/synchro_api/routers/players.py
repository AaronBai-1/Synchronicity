"""Player profile route — v0 cross-match aggregates computed live over the shot table.

Deliberately simple SQL (a few grouped counts + one average) recomputed per request:
at Phase 0/1 traffic this is nothing, and it guarantees the numbers always reflect the
latest ingest. Materialized per-player aggregates (and ShuttleNet-style embedding
profiles) arrive with docs/plan.md Phase 4 "player profiles"; this endpoint's shape is
the contract they must keep serving.

Aggregation policy: only shots/rallies attributed to the player (player_id set) count.
Unattributed rows — NULL by the "null + flag beats wrong" rule — are excluded, never
guessed into a profile.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import func, select

from synchro_api.deps import SessionDep, WorkspaceDep
from synchro_api.models import Player, Rally, Shot

router = APIRouter(prefix="/players", tags=["players"])


@router.get("/{player_id}/profile")
def get_profile(player_id: str, session: SessionDep, workspace: WorkspaceDep) -> dict:
    """Counts by coarse shot type, winners/errors, and average rally length."""
    player = session.get(Player, player_id)
    if player is None or player.workspace_id != workspace.id:
        raise HTTPException(status_code=404, detail=f"player {player_id!r} not found")

    total_shots = session.scalar(
        select(func.count()).select_from(Shot).where(Shot.player_id == player.id)
    )
    by_coarse_rows = session.execute(
        select(Shot.shot_type_coarse, func.count())
        .where(Shot.player_id == player.id)
        .group_by(Shot.shot_type_coarse)
    ).all()
    # NULL coarse type = stroke detected but not (confidently) classified.
    shots_by_coarse_type = {
        (coarse if coarse is not None else "unclassified"): n for coarse, n in by_coarse_rows
    }
    winners = session.scalar(
        select(func.count())
        .select_from(Shot)
        .where(Shot.player_id == player.id, Shot.is_winner.is_(True))
    )
    errors = session.scalar(
        select(func.count())
        .select_from(Shot)
        .where(Shot.player_id == player.id, Shot.is_error.is_(True))
    )

    played_rally_ids = (
        select(Shot.rally_id).where(Shot.player_id == player.id).distinct().scalar_subquery()
    )
    rallies_played = session.scalar(
        select(func.count()).select_from(Rally).where(Rally.id.in_(played_rally_ids))
    )
    # AVG ignores NULL shot_count rows (rallies whose stroke count the pipeline abstained on).
    avg_rally_shots = session.scalar(
        select(func.avg(Rally.shot_count)).where(Rally.id.in_(played_rally_ids))
    )
    rallies_won = session.scalar(
        select(func.count()).select_from(Rally).where(Rally.winner_player_id == player.id)
    )

    return {
        "player": {"id": player.id, "name": player.name, "handedness": player.handedness},
        "total_shots": total_shots,
        "shots_by_coarse_type": shots_by_coarse_type,
        "winners": winners,
        "errors": errors,
        "rallies_played": rallies_played,
        "rallies_won": rallies_won,
        "avg_rally_shots": float(avg_rally_shots) if avg_rally_shots is not None else None,
    }

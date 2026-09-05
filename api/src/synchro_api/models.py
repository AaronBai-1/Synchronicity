"""ORM tables: workspaces / players / matches / games / rallies / shots.

Mirrors the docs/plan.md "Platform" data model. The per-shot columns are a 1:1 image of
synchro_pipeline.schemas.records.ShotRecord (the ShuttleSet-compatible per-stroke
contract) so the offline pipeline's output lands without translation and public
CoachAI-style tooling can query the shot table directly.

Conventions:
    * ids are opaque uuid4 hex strings (API surface ids); the pipeline's own
      match_id/rally_id strings are kept in `pipeline_*` columns for traceability
    * anything the pipeline could not determine is NULL — "null + flag beats wrong"
    * `*_ref` columns store the pipeline's abstract player labels ("A"/"B") so that
      later identity assignment (POST /matches/{id}/identity) can backfill player FKs
    * JSON columns are portable sqlalchemy.JSON (SQLite dev / Postgres prod; switching
      hot ones to JSONB is a follow-up once we are Postgres-only)
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from synchro_api.db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class MatchStatus(enum.StrEnum):
    """Lifecycle of an uploaded match through the offline pipeline."""

    AWAITING_UPLOAD = "awaiting_upload"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


_STATUS_ENUM = SAEnum(
    MatchStatus,
    name="match_status",
    native_enum=False,
    length=32,
    values_callable=lambda e: [m.value for m in e],
)


class Workspace(Base):
    """Tenant boundary. Uploads are private-per-workspace by design (plan: legal posture).

    Ids come from the X-Workspace-Id header for now; Clerk org ids slot in unchanged.
    """

    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Player(Base):
    """A named human within a workspace (cross-match identity for profiles).

    Auto player naming is a v1 non-goal — rows are created from MatchRecord.players
    names or one-click manual assignment, never from face/jersey recognition.
    """

    __tablename__ = "players"
    __table_args__ = (UniqueConstraint("workspace_id", "name", name="uq_players_workspace_name"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"))
    name: Mapped[str] = mapped_column(String(255))
    handedness: Mapped[str | None] = mapped_column(String(8))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Match(Base):
    """One uploaded video → one match. Carries the confidence report (plan: every match
    gets one) and the A/B → player mapping plus the side facts (first_server,
    a_on_near_side_at_start) that anchor near/far → player resolution."""

    __tablename__ = "matches"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    status: Mapped[MatchStatus] = mapped_column(
        _STATUS_ENUM, default=MatchStatus.AWAITING_UPLOAD
    )
    video_uri: Mapped[str | None] = mapped_column(String(1024))
    discipline: Mapped[str | None] = mapped_column(String(4))  # MS|WS|MD|WD|XD

    pipeline_match_id: Mapped[str | None] = mapped_column(String(255), index=True)
    pipeline_version: Mapped[str | None] = mapped_column(String(64))
    source_meta: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    stage_model_hashes: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    confidence_report: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    player_a_id: Mapped[str | None] = mapped_column(ForeignKey("players.id"))
    player_b_id: Mapped[str | None] = mapped_column(ForeignKey("players.id"))
    first_server: Mapped[str | None] = mapped_column(String(1))  # "A" | "B"
    a_on_near_side_at_start: Mapped[bool | None]

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    player_a: Mapped[Player | None] = relationship(foreign_keys=[player_a_id])
    player_b: Mapped[Player | None] = relationship(foreign_keys=[player_b_id])
    games: Mapped[list[Game]] = relationship(
        back_populates="match", order_by="Game.seq", cascade="all, delete-orphan"
    )


class Game(Base):
    """One game ("set" in ShuttleSet naming) within a match."""

    __tablename__ = "games"
    __table_args__ = (UniqueConstraint("match_id", "seq", name="uq_games_match_seq"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"), index=True)
    seq: Mapped[int]  # game number, 1-based (== pipeline set_no)
    final_score_a: Mapped[int | None]
    final_score_b: Mapped[int | None]
    # {"near": "A"|"B"|None, "far": ...} at *game start*; the deciding-game mid-game
    # switch at 11 is handled per-rally at ingest, not encoded here.
    side_assignment: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    match: Mapped[Match] = relationship(back_populates="games")
    rallies: Mapped[list[Rally]] = relationship(
        back_populates="game", order_by="Rally.seq", cascade="all, delete-orphan"
    )


class Rally(Base):
    """One rally = one point. Mirrors synchro_pipeline.schemas.records.RallyRecord."""

    __tablename__ = "rallies"
    __table_args__ = (Index("ix_rallies_game_seq", "game_id", "seq"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    game_id: Mapped[str] = mapped_column(ForeignKey("games.id"))
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"), index=True)
    pipeline_rally_id: Mapped[str] = mapped_column(String(255))
    seq: Mapped[int]  # rally number within the game, 1-based

    start_frame: Mapped[int]
    end_frame: Mapped[int]
    duration_s: Mapped[float]

    score_before_a: Mapped[int | None]
    score_before_b: Mapped[int | None]
    score_after_a: Mapped[int | None]
    score_after_b: Mapped[int | None]

    server_ref: Mapped[str | None] = mapped_column(String(1))  # "A" | "B"
    server_player_id: Mapped[str | None] = mapped_column(ForeignKey("players.id"))
    winner_ref: Mapped[str | None] = mapped_column(String(1))
    winner_player_id: Mapped[str | None] = mapped_column(ForeignKey("players.id"))
    serve_type: Mapped[str | None] = mapped_column(String(8))  # short | long
    end_reason: Mapped[str | None] = mapped_column(String(64))

    shot_count: Mapped[int | None]
    verified: Mapped[bool] = mapped_column(default=False)  # scoring-state-machine cross-check
    confidence: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    qa_flags: Mapped[list[str] | None] = mapped_column(JSON)
    clip_uri: Mapped[str | None] = mapped_column(String(1024))

    game: Mapped[Game] = relationship(back_populates="rallies")
    shots: Mapped[list[Shot]] = relationship(
        back_populates="rally", order_by="Shot.seq", cascade="all, delete-orphan"
    )


class Shot(Base):
    """One stroke — 1:1 image of ShotRecord, the product's core contract.

    Fields the v1 UI ignores (backhand, aroundhead, landing_height, lose_reason) are
    stored anyway: they are cheap and unlock ShuttleSet-trained tactical models later.
    Indexes back the plan's query patterns: per-player tendency analytics
    (player_id, shot_type_canonical) and ordered rally playback (rally_id, seq).
    """

    __tablename__ = "shots"
    __table_args__ = (
        Index("ix_shots_player_type", "player_id", "shot_type_canonical"),
        Index("ix_shots_rally_seq", "rally_id", "seq"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    rally_id: Mapped[str] = mapped_column(ForeignKey("rallies.id"))
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"), index=True)
    seq: Mapped[int]  # stroke sequence within the rally (== ball_round), 1 = serve
    frame_num: Mapped[int]
    t_ms: Mapped[float]

    side: Mapped[str] = mapped_column(String(4))  # near | far
    player_ref: Mapped[str | None] = mapped_column(String(1))  # "A" | "B"
    player_id: Mapped[str | None] = mapped_column(ForeignKey("players.id"))

    shot_type_canonical: Mapped[str | None] = mapped_column(String(32))  # ShuttleSet 18-type
    shot_type_merged: Mapped[str | None] = mapped_column(String(16))  # classifier 11-class
    shot_type_coarse: Mapped[str | None] = mapped_column(String(16))  # UI 8-class
    type_probs: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    type_conf: Mapped[float | None]

    backhand: Mapped[bool | None]
    aroundhead: Mapped[bool | None]
    landing_height: Mapped[str | None] = mapped_column(String(8))  # net | mid | high

    hit_x: Mapped[float | None]  # court metres, frame per domain.court
    hit_y: Mapped[float | None]
    landing_x: Mapped[float | None]
    landing_y: Mapped[float | None]
    landing_area: Mapped[int | None]  # ShuttleSet 1-16 grid
    player_x: Mapped[float | None]
    player_y: Mapped[float | None]
    player_location_area: Mapped[int | None]
    opponent_x: Mapped[float | None]
    opponent_y: Mapped[float | None]
    opponent_location_area: Mapped[int | None]

    shot_speed_2d_kmh: Mapped[float | None]
    shot_speed_3d_kmh: Mapped[float | None]
    speed_method: Mapped[str | None] = mapped_column(String(32))  # e.g. court_plane_2d
    speed_conf_tier: Mapped[str | None] = mapped_column(String(1))  # A | B | C

    is_winner: Mapped[bool | None]
    is_error: Mapped[bool | None]
    lose_reason: Mapped[str | None] = mapped_column(String(64))

    hit_det_conf: Mapped[float | None]
    traj_quality: Mapped[float | None]
    qa_flags: Mapped[list[str] | None] = mapped_column(JSON)

    rally: Mapped[Rally] = relationship(back_populates="shots")

"""Shot taxonomy — three layers, one source of truth (docs/plan.md).

1. Canonical storage: ShuttleSet's 18 stroke types. Enum values are the exact ShuttleSet
   CSV strings so public data loads without translation.
2. Classifier output: merged 11 classes — ≥80% accuracy only exists at merged granularity
   (BST: 82.5% merged vs ~63% fine-grained). Never surface finer labels than the model
   can defend.
3. UI: 8-class coach vocabulary.

Coarse labels for ground-truth data should always derive from the canonical type
(CANONICAL_TO_COARSE); MERGED_TO_COARSE exists for model output and is lossy on DEFENSE
(defensive lob-returns and drive-returns both collapse to LIFT).
"""

from __future__ import annotations

from enum import StrEnum


class ShotType(StrEnum):
    """Canonical ShuttleSet 18-type taxonomy (values = ShuttleSet CSV strings)."""

    SHORT_SERVICE = "short service"
    LONG_SERVICE = "long service"
    NET_SHOT = "net shot"
    RETURN_NET = "return net"
    SMASH = "smash"
    WRIST_SMASH = "wrist smash"
    LOB = "lob"
    DEFENSIVE_RETURN_LOB = "defensive return lob"
    CLEAR = "clear"
    DRIVE = "drive"
    DRIVEN_FLIGHT = "driven flight"
    BACK_COURT_DRIVE = "back-court drive"
    DROP = "drop"
    PASSIVE_DROP = "passive drop"
    PUSH = "push"
    RUSH = "rush"
    DEFENSIVE_RETURN_DRIVE = "defensive return drive"
    CROSS_COURT_NET_SHOT = "cross-court net shot"


class MergedShotType(StrEnum):
    """Classifier output classes (11) — the granularity the model can stand behind."""

    SERVE_SHORT = "serve_short"
    SERVE_LONG = "serve_long"
    NET = "net"
    SMASH = "smash"
    LOB = "lob"
    DEFENSE = "defense"
    CLEAR = "clear"
    DRIVE = "drive"
    DROP = "drop"
    PUSH = "push"
    RUSH = "rush"


class CoarseShotType(StrEnum):
    """8-class coach vocabulary shown in the UI."""

    SERVE = "serve"
    CLEAR = "clear"
    DROP = "drop"
    SMASH = "smash"
    DRIVE = "drive"
    NET = "net"
    LIFT = "lift"
    PUSH_RUSH = "push_rush"


CANONICAL_TO_MERGED: dict[ShotType, MergedShotType] = {
    ShotType.SHORT_SERVICE: MergedShotType.SERVE_SHORT,
    ShotType.LONG_SERVICE: MergedShotType.SERVE_LONG,
    ShotType.NET_SHOT: MergedShotType.NET,
    ShotType.RETURN_NET: MergedShotType.NET,
    ShotType.CROSS_COURT_NET_SHOT: MergedShotType.NET,
    ShotType.SMASH: MergedShotType.SMASH,
    ShotType.WRIST_SMASH: MergedShotType.SMASH,
    ShotType.LOB: MergedShotType.LOB,
    ShotType.DEFENSIVE_RETURN_LOB: MergedShotType.DEFENSE,
    ShotType.DEFENSIVE_RETURN_DRIVE: MergedShotType.DEFENSE,
    ShotType.CLEAR: MergedShotType.CLEAR,
    ShotType.DRIVE: MergedShotType.DRIVE,
    ShotType.DRIVEN_FLIGHT: MergedShotType.DRIVE,
    ShotType.BACK_COURT_DRIVE: MergedShotType.DRIVE,
    ShotType.DROP: MergedShotType.DROP,
    ShotType.PASSIVE_DROP: MergedShotType.DROP,
    ShotType.PUSH: MergedShotType.PUSH,
    ShotType.RUSH: MergedShotType.RUSH,
}

CANONICAL_TO_COARSE: dict[ShotType, CoarseShotType] = {
    ShotType.SHORT_SERVICE: CoarseShotType.SERVE,
    ShotType.LONG_SERVICE: CoarseShotType.SERVE,
    ShotType.NET_SHOT: CoarseShotType.NET,
    ShotType.RETURN_NET: CoarseShotType.NET,
    ShotType.CROSS_COURT_NET_SHOT: CoarseShotType.NET,
    ShotType.SMASH: CoarseShotType.SMASH,
    ShotType.WRIST_SMASH: CoarseShotType.SMASH,
    ShotType.LOB: CoarseShotType.LIFT,
    ShotType.DEFENSIVE_RETURN_LOB: CoarseShotType.LIFT,
    ShotType.CLEAR: CoarseShotType.CLEAR,
    ShotType.DRIVE: CoarseShotType.DRIVE,
    ShotType.DRIVEN_FLIGHT: CoarseShotType.DRIVE,
    ShotType.BACK_COURT_DRIVE: CoarseShotType.DRIVE,
    ShotType.DEFENSIVE_RETURN_DRIVE: CoarseShotType.DRIVE,
    ShotType.DROP: CoarseShotType.DROP,
    ShotType.PASSIVE_DROP: CoarseShotType.DROP,
    ShotType.PUSH: CoarseShotType.PUSH_RUSH,
    ShotType.RUSH: CoarseShotType.PUSH_RUSH,
}

# Lossy: model output only. DEFENSE mixes lob- and drive-returns; coaches read defensive
# returns primarily as lifts, so that is where the merged class lands in the UI.
MERGED_TO_COARSE: dict[MergedShotType, CoarseShotType] = {
    MergedShotType.SERVE_SHORT: CoarseShotType.SERVE,
    MergedShotType.SERVE_LONG: CoarseShotType.SERVE,
    MergedShotType.NET: CoarseShotType.NET,
    MergedShotType.SMASH: CoarseShotType.SMASH,
    MergedShotType.LOB: CoarseShotType.LIFT,
    MergedShotType.DEFENSE: CoarseShotType.LIFT,
    MergedShotType.CLEAR: CoarseShotType.CLEAR,
    MergedShotType.DRIVE: CoarseShotType.DRIVE,
    MergedShotType.DROP: CoarseShotType.DROP,
    MergedShotType.PUSH: CoarseShotType.PUSH_RUSH,
    MergedShotType.RUSH: CoarseShotType.PUSH_RUSH,
}

SERVE_TYPES: frozenset[ShotType] = frozenset({ShotType.SHORT_SERVICE, ShotType.LONG_SERVICE})

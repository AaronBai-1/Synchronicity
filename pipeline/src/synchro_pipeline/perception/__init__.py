"""Model wrappers for the perception stages (plan S2-S4).

Each wrapper adapts an external model to a small in-repo protocol so stages and the
eval harness never depend on a specific vendor repo. Heavy deps (torch, vendored
research code) are lazy-imported inside methods — the base package must import and
test offline (docs/plan.md: torch is an optional `ml` extra).

hits_baseline is the exception: no model at all — the trajectory-only hit-detection
prototype plan risk #1 asks for (pure numpy over ShuttleTrackPoints).
"""

from synchro_pipeline.perception.hits_baseline import ProposedHit, assign_sides, propose_hits

__all__ = ["ProposedHit", "assign_sides", "propose_hits"]

"""Model wrappers for the perception stages (plan S2-S4).

Each wrapper adapts an external model to a small in-repo protocol so stages and the
eval harness never depend on a specific vendor repo. Heavy deps (torch, vendored
research code) are lazy-imported inside methods — the base package must import and
test offline (docs/plan.md: torch is an optional `ml` extra).

hits_baseline and audio_onsets are the exceptions: no model at all — hits_baseline is
the trajectory-only hit-detection prototype plan risk #1 asks for (pure numpy over
ShuttleTrackPoints), and audio_onsets is the numpy+stdlib spectral-flux hit proposer
that accelerates golden-set hit labeling (Phase 0).
"""

from synchro_pipeline.perception.audio_onsets import (
    AudioExtractionError,
    OnsetEvent,
    detect_onsets,
    extract_audio,
    next_onset,
    onset_near,
    prev_onset,
)
from synchro_pipeline.perception.hits_baseline import ProposedHit, assign_sides, propose_hits

__all__ = [
    "AudioExtractionError",
    "OnsetEvent",
    "ProposedHit",
    "assign_sides",
    "detect_onsets",
    "extract_audio",
    "next_onset",
    "onset_near",
    "prev_onset",
    "propose_hits",
]

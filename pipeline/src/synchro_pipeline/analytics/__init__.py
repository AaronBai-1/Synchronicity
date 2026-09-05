# Portions adapted from BadmintonAnalyzer (Apache-2.0), Copyright Dhyey Mavani.
"""Analytics + tactical suite over ShotRecord/RallyRecord contracts (docs/plan.md "S7").

Modules: adapter (contracts -> DataFrames with explicit QA-drop reporting),
descriptive (aggregates — every row carries n + low_sample), tactical (transition
tables and sequence-pattern mining), scouting (deterministic template text, gated on
sample size).
"""

from synchro_pipeline.analytics.adapter import (
    DEFAULT_MIN_N,
    ERROR_OUTCOMES,
    RALLY_FRAME_COLUMNS,
    SHOT_FRAME_COLUMNS,
    TERMINAL_OUTCOMES,
    UNKNOWN,
    ZONE6_LABELS,
    AdapterReport,
    build_rally_frame,
    build_shot_frame,
    outcome_from_flags,
    zone6_from_area,
    zone6_from_xy,
)
from synchro_pipeline.analytics.descriptive import (
    PRESSURE_CONTEXTS,
    conditional_shot_mix,
    error_profile,
    head_to_head,
    match_trend,
    momentum_summary,
    pressure_summary,
    serve_patterns,
    shot_distribution,
    terminal_conversion,
)
from synchro_pipeline.analytics.scouting import build_recommendations, build_scouting_report
from synchro_pipeline.analytics.tactical import (
    PATTERN_COLUMNS,
    TRANSITION_COLUMNS,
    build_transition_table,
    mine_sequence_patterns,
)

__all__ = [
    "DEFAULT_MIN_N",
    "ERROR_OUTCOMES",
    "PATTERN_COLUMNS",
    "PRESSURE_CONTEXTS",
    "RALLY_FRAME_COLUMNS",
    "SHOT_FRAME_COLUMNS",
    "TERMINAL_OUTCOMES",
    "TRANSITION_COLUMNS",
    "UNKNOWN",
    "ZONE6_LABELS",
    "AdapterReport",
    "build_rally_frame",
    "build_recommendations",
    "build_scouting_report",
    "build_shot_frame",
    "build_transition_table",
    "conditional_shot_mix",
    "error_profile",
    "head_to_head",
    "match_trend",
    "mine_sequence_patterns",
    "momentum_summary",
    "outcome_from_flags",
    "pressure_summary",
    "serve_patterns",
    "shot_distribution",
    "terminal_conversion",
    "zone6_from_area",
    "zone6_from_xy",
]

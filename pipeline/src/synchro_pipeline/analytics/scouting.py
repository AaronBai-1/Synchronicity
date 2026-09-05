# Portions adapted from BadmintonAnalyzer (Apache-2.0), Copyright Dhyey Mavani.
"""Deterministic template-over-aggregates scouting text (docs/plan.md trust rules).

Adapted from BadmintonAnalyzer's ``build_recommendations`` / ``build_scouting_report``
(their ML-model-derived recommendation was NOT ported — audit verdict). Trust fixes
applied during the port:

    * every claim embeds the sample size it rests on (``n=...``);
    * every section is gated on ``low_sample``: a thin aggregate produces an explicit
      "insufficient data" line (or is omitted from recommendations) instead of being
      asserted as if it were a finding;
    * generation is pure and deterministic — same frames in, same text out.
"""

from __future__ import annotations

import pandas as pd

from synchro_pipeline.analytics.adapter import DEFAULT_MIN_N
from synchro_pipeline.analytics.descriptive import (
    error_profile,
    pressure_summary,
    shot_distribution,
)
from synchro_pipeline.analytics.tactical import build_transition_table, mine_sequence_patterns


def _is_missing(value: object) -> bool:
    return value is None or pd.isna(value)


def _transition_claim(row: pd.Series) -> str:
    claim = (
        f"the most common response to '{row['current_state']}' is "
        f"'{row['response_state']}' ({row['probability']:.0%} of n={int(row['row_total'])} "
        "transitions)"
    )
    if not _is_missing(row["win_rate"]):
        claim += (
            f"; rallies featuring this exchange were won {row['win_rate']:.0%} "
            f"of the time (n={int(row['n_decided'])} decided)"
        )
    return claim


def build_recommendations(
    shots_df: pd.DataFrame,
    rallies_df: pd.DataFrame,
    player: str,
    *,
    min_n: int = DEFAULT_MIN_N,
) -> list[str]:
    """Up to three deterministic recommendations for ``player``, each embedding its n.

    A recommendation is only emitted when its backing aggregate row clears ``min_n``
    (``low_sample`` is False) — thin evidence is omitted, never asserted.
    """
    recommendations: list[str] = []
    if shots_df.empty:
        return recommendations

    transitions = build_transition_table(shots_df, player, min_n=min_n)
    if not transitions.empty:
        usable = transitions[~transitions["low_sample"]]
        if not usable.empty:
            top = usable.iloc[0]
            recommendations.append(
                f"Lean into '{top['current_state']}': " + _transition_claim(top) + "."
            )

    errors = error_profile(shots_df, player, min_n=min_n)
    if not errors.empty:
        usable = errors[~errors["low_sample"]]
        if not usable.empty:
            top = usable.iloc[0]
            recommendations.append(
                f"Reduce risk on '{top['shot_type']}' from {top['origin_zone']}: "
                f"it is the largest error source in this sample (n={int(top['n'])} errors)."
            )

    winner_shots = shots_df[
        (shots_df["player"] == player) & (shots_df["outcome"] == "winner")
    ]
    if not winner_shots.empty:
        finishes = (
            winner_shots.groupby(
                [winner_shots["shot_type"].fillna("unknown"), winner_shots["target_zone"].fillna("unknown")]
            )
            .size()
            .rename("n")
            .reset_index()
            .sort_values(["n", "shot_type", "target_zone"], ascending=[False, True, True])
        )
        top = finishes.iloc[0]
        if int(top["n"]) >= min_n:
            recommendations.append(
                f"Build rallies toward '{top['shot_type']}' into {top['target_zone']}: "
                f"your most frequent winning finish (n={int(top['n'])})."
            )

    return recommendations


def build_scouting_report(
    shots_df: pd.DataFrame,
    rallies_df: pd.DataFrame,
    player: str,
    *,
    min_n: int = DEFAULT_MIN_N,
    pattern_length: int = 3,
    min_support: int = 10,
) -> str:
    """Markdown scouting report for ``player``; every claim embeds its n, and each
    section falls back to an explicit "insufficient data" line when the backing
    aggregate is empty or below ``min_n``."""
    n_shots = int((shots_df["player"] == player).sum()) if not shots_df.empty else 0
    n_rallies = len(rallies_df)
    n_decided = (
        int(rallies_df["winner"].isin(("A", "B")).sum()) if not rallies_df.empty else 0
    )
    lines: list[str] = [
        f"# Scouting report — player {player}",
        "",
        f"Sample: n={n_shots} shots by player {player}, n={n_rallies} rallies "
        f"({n_decided} with a decided winner).",
        "",
        "## Shot mix",
    ]

    distribution = shot_distribution(shots_df, player, min_n=min_n)
    usable = distribution[~distribution["low_sample"]] if not distribution.empty else distribution
    if usable.empty:
        largest = int(distribution["n"].max()) if not distribution.empty else 0
        lines.append(
            f"- insufficient data: no shot type reaches n={min_n} (largest sample n={largest})."
        )
    else:
        for row in usable.head(3).itertuples(index=False):
            lines.append(f"- {row.shot_type}: {row.share:.0%} of shots (n={int(row.n)})")

    lines.extend(["", "## Response tendencies"])
    transitions = build_transition_table(shots_df, player, min_n=min_n)
    usable = transitions[~transitions["low_sample"]] if not transitions.empty else transitions
    if usable.empty:
        largest = int(transitions["row_total"].max()) if not transitions.empty else 0
        lines.append(
            f"- insufficient data: no transition row reaches n={min_n} "
            f"(largest row n={largest})."
        )
    else:
        lines.append("- " + _transition_claim(usable.iloc[0]).capitalize() + ".")

    lines.extend(["", "## Recurring patterns"])
    patterns = mine_sequence_patterns(
        shots_df, player, n=pattern_length, min_support=min_support, min_n=min_n
    )
    usable = patterns[~patterns["low_sample"]] if not patterns.empty else patterns
    if usable.empty:
        lines.append(
            f"- insufficient data: no {pattern_length}-shot pattern reaches "
            f"support n={max(min_support, min_n)}."
        )
    else:
        top = usable.iloc[0]
        claim = (
            f"- '{top['pattern']}' occurred n={int(top['n'])} times; rally win rate "
            f"{top['win_rate']:.0%} vs {top['baseline_win_rate']:.0%} baseline"
        )
        if not _is_missing(top["lift"]):
            claim += f" (lift {top['lift']:.2f})"
        lines.append(claim + ".")

    lines.extend(["", "## Pressure situations"])
    pressure = pressure_summary(rallies_df, player, min_n=min_n)
    pressure_lines: list[str] = []
    if not pressure.empty:
        for row in pressure.itertuples(index=False):
            if row.context in ("neutral", "unknown") or row.n == 0:
                continue
            if row.low_sample or _is_missing(row.win_rate):
                pressure_lines.append(f"- {row.context}: insufficient data (n={int(row.n)}).")
            else:
                pressure_lines.append(
                    f"- {row.context}: win rate {row.win_rate:.0%} "
                    f"(n={int(row.n_decided)} decided of {int(row.n)})."
                )
    if pressure_lines:
        lines.extend(pressure_lines)
    else:
        lines.append("- insufficient data: no pressure-context rallies in sample (n=0).")

    lines.extend(["", "## Recommendations"])
    recommendations = build_recommendations(shots_df, rallies_df, player, min_n=min_n)
    if recommendations:
        lines.extend(f"- {rec}" for rec in recommendations)
    else:
        lines.append(f"- insufficient data for recommendations at n>={min_n}.")

    return "\n".join(lines)

# Portions adapted from BadmintonAnalyzer (Apache-2.0), Copyright Dhyey Mavani.
"""Tactical mining over adapter frames: transition tables and sequence patterns.

Adapted from BadmintonAnalyzer's ``tactical/models.py`` (their outcome logistic model,
matchup simulation and style clustering were NOT ported — audit verdict), with trust
fixes applied during the port:

    * transition rows keep raw counts next to the row-normalized probability, carry
      the row denominator (``row_total``) and are flagged ``low_sample`` on it;
    * win rates are computed only over rallies with a known winner (``n_decided``),
      None when there are none — never a fabricated number;
    * pattern mining has a ``min_support`` guard (default 10) so one lucky rally can't
      masquerade as a tactic, and reports lift against an explicit in-sample baseline;
    * shots with an unknown type (or unknown zone in "type_zone" mode) never become
      tokens — the affected transition/window is excluded rather than guessed at, and
      rallies with an unknown winner contribute no windows to win statistics;
    * empty-input safe: every function returns an empty frame with documented columns.

Tokens (``token`` parameter):
    "type"      — the coarse shot type alone, e.g. ``smash``
    "type_zone" — type + target zone, e.g. ``smash@rear_left``
"""

from __future__ import annotations

import pandas as pd

from synchro_pipeline.analytics.adapter import DEFAULT_MIN_N

_PLAYERS = ("A", "B")

TRANSITION_COLUMNS: tuple[str, ...] = (
    "current_state",
    "response_state",
    "n",
    "row_total",
    "probability",
    "n_decided",
    "win_rate",
    "low_sample",
)

PATTERN_COLUMNS: tuple[str, ...] = (
    "pattern",
    "n",
    "n_won",
    "n_lost",
    "win_rate",
    "baseline_win_rate",
    "lift",
    "low_sample",
)


def _token(row: object, mode: str) -> str | None:
    """Token for one shot row, or None when the needed fields are unknown."""
    shot_type = row.shot_type
    if pd.isna(shot_type):
        return None
    if mode == "type":
        return str(shot_type)
    if mode == "type_zone":
        target = row.target_zone
        if pd.isna(target):
            return None
        return f"{shot_type}@{target}"
    raise ValueError(f"unknown token mode {mode!r}; expected 'type' or 'type_zone'")


def build_transition_table(
    shots_df: pd.DataFrame,
    player: str,
    *,
    token: str = "type",
    min_n: int = DEFAULT_MIN_N,
) -> pd.DataFrame:
    """Markov transition table: ``player``'s shot -> the opponent's immediate response.

    Transitions are consecutive shot pairs within a rally (ordered by shot_number,
    never crossing rally boundaries) where the current shot is ``player``'s and the
    next is the opponent's. Columns: :data:`TRANSITION_COLUMNS` — ``probability`` is
    row-normalized (``n / row_total`` per current_state) with the raw counts kept;
    ``win_rate`` is the share of the ``n_decided`` transitions whose rally ``player``
    won (None when no rally in the cell has a known winner); ``low_sample`` flags
    ``row_total < min_n`` (the probability denominator). Sorted by n desc.
    """
    columns = list(TRANSITION_COLUMNS)
    if shots_df.empty:
        return pd.DataFrame(columns=columns)
    opponent = "B" if player == "A" else "A"
    records: list[dict[str, object]] = []
    for _, rally_df in shots_df.groupby(["match_id", "rally_id"], sort=False):
        ordered = list(rally_df.sort_values("shot_number").itertuples(index=False))
        for cur, nxt in zip(ordered, ordered[1:], strict=False):
            if cur.player != player or nxt.player != opponent:
                continue
            cur_token = _token(cur, token)
            nxt_token = _token(nxt, token)
            if cur_token is None or nxt_token is None:
                continue
            decided = cur.winner in _PLAYERS
            records.append(
                {
                    "current_state": cur_token,
                    "response_state": nxt_token,
                    "decided": decided,
                    "won": bool(decided and cur.winner == player),
                }
            )
    if not records:
        return pd.DataFrame(columns=columns)
    transitions = pd.DataFrame.from_records(records)
    grouped = (
        transitions.groupby(["current_state", "response_state"])
        .agg(n=("won", "size"), n_decided=("decided", "sum"), n_won=("won", "sum"))
        .reset_index()
    )
    grouped["row_total"] = grouped.groupby("current_state")["n"].transform("sum")
    grouped["probability"] = grouped["n"] / grouped["row_total"]
    grouped["win_rate"] = [
        (int(won) / int(dec)) if dec else None
        for won, dec in zip(grouped["n_won"], grouped["n_decided"], strict=True)
    ]
    grouped["low_sample"] = grouped["row_total"] < min_n
    return (
        grouped.sort_values(
            ["n", "current_state", "response_state"], ascending=[False, True, True]
        ).reset_index(drop=True)[columns]
    )


def mine_sequence_patterns(
    shots_df: pd.DataFrame,
    player: str,
    *,
    n: int = 3,
    token: str = "type",
    min_support: int = 10,
    min_n: int = DEFAULT_MIN_N,
    top_k: int = 20,
) -> pd.DataFrame:
    """N-gram mining over ``player``'s shot-token sequences, split won vs lost.

    Windows are contiguous runs of ``n`` of the player's own shots within one rally
    (rally boundaries are never crossed). Only rallies with a known winner contribute
    windows — win statistics are never fabricated for undecided rallies. Columns:
    :data:`PATTERN_COLUMNS` — ``baseline_win_rate`` is the mean win share over ALL
    kept windows, and ``lift = win_rate / baseline_win_rate`` (None when the baseline
    is 0). Patterns below ``min_support`` occurrences are excluded (guard against
    one-rally "tactics"); ``low_sample`` additionally flags ``n < min_n``. Sorted by
    n desc then win_rate desc, truncated to ``top_k`` rows.
    """
    columns = list(PATTERN_COLUMNS)
    if shots_df.empty:
        return pd.DataFrame(columns=columns)
    windows: list[dict[str, object]] = []
    for _, rally_df in shots_df.groupby(["match_id", "rally_id"], sort=False):
        ordered = rally_df.sort_values("shot_number")
        winner = ordered.iloc[0]["winner"]
        if winner not in _PLAYERS:
            continue  # unknown rally winner: no windows, never guessed
        tokens = [
            _token(row, token)
            for row in ordered.itertuples(index=False)
            if row.player == player
        ]
        won = int(winner == player)
        for start in range(len(tokens) - n + 1):
            window = tokens[start : start + n]
            if any(t is None for t in window):
                continue  # unknown token inside the window: exclude, don't guess
            windows.append({"pattern": " -> ".join(window), "won": won})
    if not windows:
        return pd.DataFrame(columns=columns)
    window_df = pd.DataFrame.from_records(windows)
    baseline = float(window_df["won"].mean())
    grouped = (
        window_df.groupby("pattern")
        .agg(n=("won", "size"), n_won=("won", "sum"))
        .reset_index()
    )
    grouped = grouped[grouped["n"] >= min_support]
    if grouped.empty:
        return pd.DataFrame(columns=columns)
    grouped["n_lost"] = grouped["n"] - grouped["n_won"]
    grouped["win_rate"] = grouped["n_won"] / grouped["n"]
    grouped["baseline_win_rate"] = baseline
    grouped["lift"] = grouped["win_rate"] / baseline if baseline > 0 else None
    grouped["low_sample"] = grouped["n"] < min_n
    return (
        grouped.sort_values(["n", "win_rate", "pattern"], ascending=[False, False, True])
        .head(top_k)
        .reset_index(drop=True)[columns]
    )

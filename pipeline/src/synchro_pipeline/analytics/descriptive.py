# Portions adapted from BadmintonAnalyzer (Apache-2.0), Copyright Dhyey Mavani.
"""Descriptive aggregates over adapter frames (docs/plan.md "S7 Metrics").

Adapted from BadmintonAnalyzer's ``analytics.py`` with the trust architecture the
original lacked (fixed during the port, per the audit):

    * every output row carries ``n`` and ``low_sample`` (n < min_n, default
      :data:`~synchro_pipeline.analytics.adapter.DEFAULT_MIN_N`) — low-sample rows are
      flagged, never hidden and never silently included;
    * an undefined rate is None, never a fabricated 0.0 (their ``fillna(0.0)`` /
      unconditional ``.mean()`` idioms are not replicated); rate denominators are
      reported alongside (``n_decided`` where a rally winner may be unknown);
    * missing group keys become the explicit ``"unknown"`` bucket, never dropped;
    * momentum/point-run metrics respect game boundaries (theirs ran streaks across
      games and matches);
    * every function is empty-input safe, returning an empty frame with the documented
      columns.

All functions take the DataFrames produced by :mod:`synchro_pipeline.analytics.adapter`
and a ``player`` of "A" / "B" (our PlayerRef vocabulary).
"""

from __future__ import annotations

import pandas as pd

from synchro_pipeline.analytics.adapter import (
    DEFAULT_MIN_N,
    ERROR_OUTCOMES,
    TERMINAL_OUTCOMES,
    coalesce_unknown,
)
from synchro_pipeline.domain.scoring import GAME_TARGET, GameScore, Player, is_game_point

_PLAYERS = ("A", "B")


def _opponent(player: str) -> str:
    return "B" if player == "A" else "A"


def _rate_or_none(numerator: int, denominator: int) -> float | None:
    """A rate with an explicit denominator; None (never 0.0) when undefined."""
    return numerator / denominator if denominator else None


# --- shot mix ----------------------------------------------------------------------


def shot_distribution(
    shots_df: pd.DataFrame, player: str, *, min_n: int = DEFAULT_MIN_N
) -> pd.DataFrame:
    """Shot-type mix for one player: [shot_type, n, share, low_sample]."""
    columns = ["shot_type", "n", "share", "low_sample"]
    if shots_df.empty:
        return pd.DataFrame(columns=columns)
    player_shots = shots_df[shots_df["player"] == player]
    if player_shots.empty:
        return pd.DataFrame(columns=columns)
    typed = player_shots.assign(shot_type=player_shots["shot_type"].map(coalesce_unknown))
    dist = typed.groupby("shot_type").size().rename("n").reset_index()
    dist["share"] = dist["n"] / int(dist["n"].sum())
    dist["low_sample"] = dist["n"] < min_n
    return (
        dist.sort_values(["n", "shot_type"], ascending=[False, True])
        .reset_index(drop=True)[columns]
    )


def conditional_shot_mix(
    shots_df: pd.DataFrame, player: str, *, min_n: int = DEFAULT_MIN_N
) -> pd.DataFrame:
    """Response mix: what ``player`` plays after each opposing shot type.

    Pairs are consecutive shots within a rally where the previous shot belongs to the
    opponent — rally boundaries are never crossed. Columns:
    [prev_shot_type, shot_type, n, share, low_sample]; ``share`` normalizes within each
    prev_shot_type group.
    """
    columns = ["prev_shot_type", "shot_type", "n", "share", "low_sample"]
    if shots_df.empty:
        return pd.DataFrame(columns=columns)
    opponent = _opponent(player)
    records: list[dict[str, object]] = []
    for _, rally_df in shots_df.groupby(["match_id", "rally_id"], sort=False):
        ordered = list(rally_df.sort_values("shot_number").itertuples(index=False))
        for prev, cur in zip(ordered, ordered[1:], strict=False):
            if cur.player != player or prev.player != opponent:
                continue
            records.append(
                {
                    "prev_shot_type": coalesce_unknown(prev.shot_type),
                    "shot_type": coalesce_unknown(cur.shot_type),
                }
            )
    if not records:
        return pd.DataFrame(columns=columns)
    pairs = pd.DataFrame.from_records(records)
    mix = pairs.groupby(["prev_shot_type", "shot_type"]).size().rename("n").reset_index()
    mix["share"] = mix["n"] / mix.groupby("prev_shot_type")["n"].transform("sum")
    mix["low_sample"] = mix["n"] < min_n
    return (
        mix.sort_values(
            ["prev_shot_type", "n", "shot_type"], ascending=[True, False, True]
        ).reset_index(drop=True)[columns]
    )


# --- serves ------------------------------------------------------------------------


def serve_patterns(
    rallies_df: pd.DataFrame, player: str, *, min_n: int = DEFAULT_MIN_N
) -> pd.DataFrame:
    """Serve-type share and outcomes by the server's score parity (when known).

    Columns: [serve_type, parity, n, share, n_decided, win_rate, low_sample].
    ``parity`` is "even"/"odd" from the server's own score in ``score_before``
    ("unknown" when the score is missing); ``share`` normalizes within each parity
    group. ``win_rate`` is computed over the ``n_decided`` rallies with a known winner
    and is None when there are none.
    """
    columns = ["serve_type", "parity", "n", "share", "n_decided", "win_rate", "low_sample"]
    if rallies_df.empty:
        return pd.DataFrame(columns=columns)
    serves = rallies_df[rallies_df["server"] == player].copy()
    if serves.empty:
        return pd.DataFrame(columns=columns)
    own_score = serves["score_a"] if player == "A" else serves["score_b"]
    serves["parity"] = own_score.map(
        lambda v: "unknown" if pd.isna(v) else ("even" if int(v) % 2 == 0 else "odd")
    )
    serves["serve_type"] = serves["serve_type"].map(coalesce_unknown)
    serves["decided"] = serves["winner"].isin(_PLAYERS)
    serves["won"] = serves["winner"] == player
    grouped = (
        serves.groupby(["parity", "serve_type"])
        .agg(n=("rally_id", "size"), n_decided=("decided", "sum"), n_won=("won", "sum"))
        .reset_index()
    )
    grouped["share"] = grouped["n"] / grouped.groupby("parity")["n"].transform("sum")
    grouped["win_rate"] = [
        _rate_or_none(int(won), int(dec))
        for won, dec in zip(grouped["n_won"], grouped["n_decided"], strict=True)
    ]
    grouped["low_sample"] = grouped["n"] < min_n
    return (
        grouped.sort_values(["parity", "n", "serve_type"], ascending=[True, False, True])
        .reset_index(drop=True)[columns]
    )


# --- terminal shots ----------------------------------------------------------------


def terminal_conversion(
    shots_df: pd.DataFrame, player: str, *, min_n: int = DEFAULT_MIN_N
) -> pd.DataFrame:
    """Winners/errors by shot type over the player's terminal shots.

    Columns: [shot_type, n, n_winners, n_errors, winner_rate, error_rate, low_sample];
    ``n`` counts terminal shots (rally-ending) of that type.
    """
    columns = ["shot_type", "n", "n_winners", "n_errors", "winner_rate", "error_rate", "low_sample"]
    if shots_df.empty:
        return pd.DataFrame(columns=columns)
    terminal = shots_df[
        (shots_df["player"] == player) & shots_df["outcome"].isin(TERMINAL_OUTCOMES)
    ].copy()
    if terminal.empty:
        return pd.DataFrame(columns=columns)
    terminal["shot_type"] = terminal["shot_type"].map(coalesce_unknown)
    grouped = (
        terminal.groupby("shot_type")
        .agg(
            n=("outcome", "size"),
            n_winners=("outcome", lambda s: int((s == "winner").sum())),
            n_errors=("outcome", lambda s: int(s.isin(ERROR_OUTCOMES).sum())),
        )
        .reset_index()
    )
    grouped["winner_rate"] = grouped["n_winners"] / grouped["n"]
    grouped["error_rate"] = grouped["n_errors"] / grouped["n"]
    grouped["low_sample"] = grouped["n"] < min_n
    return (
        grouped.sort_values(["n", "shot_type"], ascending=[False, True])
        .reset_index(drop=True)[columns]
    )


def error_profile(
    shots_df: pd.DataFrame, player: str, *, min_n: int = DEFAULT_MIN_N
) -> pd.DataFrame:
    """Where the player's errors come from: [shot_type, origin_zone, target_zone, n,
    n_forced, n_unforced, n_unattributed, low_sample].

    The forced/unforced split is best-effort (see adapter.outcome_from_flags);
    ``n_unattributed`` counts errors whose lose_reason did not distinguish.
    """
    columns = [
        "shot_type",
        "origin_zone",
        "target_zone",
        "n",
        "n_forced",
        "n_unforced",
        "n_unattributed",
        "low_sample",
    ]
    if shots_df.empty:
        return pd.DataFrame(columns=columns)
    errors = shots_df[
        (shots_df["player"] == player) & shots_df["outcome"].isin(ERROR_OUTCOMES)
    ].copy()
    if errors.empty:
        return pd.DataFrame(columns=columns)
    for key in ("shot_type", "origin_zone", "target_zone"):
        errors[key] = errors[key].map(coalesce_unknown)
    grouped = (
        errors.groupby(["shot_type", "origin_zone", "target_zone"])
        .agg(
            n=("outcome", "size"),
            n_forced=("outcome", lambda s: int((s == "error_forced").sum())),
            n_unforced=("outcome", lambda s: int((s == "error_unforced").sum())),
            n_unattributed=("outcome", lambda s: int((s == "error").sum())),
        )
        .reset_index()
    )
    grouped["low_sample"] = grouped["n"] < min_n
    return (
        grouped.sort_values(
            ["n", "shot_type", "origin_zone", "target_zone"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)[columns]
    )


# --- pressure ----------------------------------------------------------------------

PRESSURE_CONTEXTS: tuple[str, ...] = (
    "game_point_for",
    "game_point_against",
    "game_point_both",
    "deuce",
    "neutral",
    "unknown",
)


def _pressure_context(score_a: object, score_b: object, me: Player, opp: Player) -> str:
    if pd.isna(score_a) or pd.isna(score_b):
        return "unknown"
    score = GameScore(int(score_a), int(score_b))
    gp_me = is_game_point(score, me)
    gp_opp = is_game_point(score, opp)
    if gp_me and gp_opp:
        return "game_point_both"
    if gp_me:
        return "game_point_for"
    if gp_opp:
        return "game_point_against"
    if score.a == score.b and score.a >= GAME_TARGET - 1:
        return "deuce"
    return "neutral"


def pressure_summary(
    rallies_df: pd.DataFrame, player: str, *, min_n: int = DEFAULT_MIN_N
) -> pd.DataFrame:
    """Rally outcomes in pressure contexts derived via domain.scoring.

    One row per context in :data:`PRESSURE_CONTEXTS` (always all six, so callers see
    n=0 explicitly): [context, n, n_decided, n_won, win_rate, low_sample]. Contexts are
    from ``player``'s perspective; "game_point_both" covers 29-29-style scores where
    both players hold game point; "deuce" is any tied score >= 20. ``win_rate`` is
    None when no rally in the context has a known winner.
    """
    columns = ["context", "n", "n_decided", "n_won", "win_rate", "low_sample"]
    if rallies_df.empty:
        return pd.DataFrame(columns=columns)
    me = Player(player)
    opp = me.opponent
    stats: dict[str, dict[str, int]] = {
        c: {"n": 0, "n_decided": 0, "n_won": 0} for c in PRESSURE_CONTEXTS
    }
    for row in rallies_df.itertuples(index=False):
        entry = stats[_pressure_context(row.score_a, row.score_b, me, opp)]
        entry["n"] += 1
        if row.winner in _PLAYERS:
            entry["n_decided"] += 1
            entry["n_won"] += int(row.winner == player)
    rows = [
        {
            "context": context,
            "n": e["n"],
            "n_decided": e["n_decided"],
            "n_won": e["n_won"],
            "win_rate": _rate_or_none(e["n_won"], e["n_decided"]),
            "low_sample": e["n"] < min_n,
        }
        for context, e in ((c, stats[c]) for c in PRESSURE_CONTEXTS)
    ]
    return pd.DataFrame(rows, columns=columns)


# --- momentum ----------------------------------------------------------------------


def momentum_summary(
    rallies_df: pd.DataFrame, player: str, *, min_n: int = DEFAULT_MIN_N
) -> pd.DataFrame:
    """Point-run metrics: [metric, value, n, low_sample].

    Streaks/skids are computed within each (match_id, game_number) ordered by ``seq``
    and never cross a game boundary (fixed from the original, which ran streaks across
    games). A rally with an unknown winner breaks both runs (conservative: an unknown
    result is never assumed to extend a streak). ``close_score_win_rate`` covers
    decided rallies whose score margin was <= 2; None when there are none.
    """
    columns = ["metric", "value", "n", "low_sample"]
    if rallies_df.empty:
        return pd.DataFrame(columns=columns)
    longest_streak = 0
    longest_skid = 0
    for _, game_df in rallies_df.groupby(["match_id", "game_number"], sort=False):
        streak = 0
        skid = 0
        for winner in game_df.sort_values("seq")["winner"]:
            if winner == player:
                streak += 1
                skid = 0
            elif winner in _PLAYERS:
                skid += 1
                streak = 0
            else:  # unknown winner: breaks both runs, never extends either
                streak = 0
                skid = 0
            longest_streak = max(longest_streak, streak)
            longest_skid = max(longest_skid, skid)

    n_rallies = len(rallies_df)
    margin = (
        pd.to_numeric(rallies_df["score_a"], errors="coerce")
        - pd.to_numeric(rallies_df["score_b"], errors="coerce")
    ).abs()
    close = rallies_df[(margin <= 2) & rallies_df["winner"].isin(_PLAYERS)]
    n_close = len(close)
    close_rate = float((close["winner"] == player).mean()) if n_close else None
    rows = [
        {
            "metric": "longest_point_streak",
            "value": float(longest_streak),
            "n": n_rallies,
            "low_sample": n_rallies < min_n,
        },
        {
            "metric": "longest_skid",
            "value": float(longest_skid),
            "n": n_rallies,
            "low_sample": n_rallies < min_n,
        },
        {
            "metric": "close_score_win_rate",
            "value": close_rate,
            "n": n_close,
            "low_sample": n_close < min_n,
        },
    ]
    return pd.DataFrame(rows, columns=columns)


# --- comparisons and trends --------------------------------------------------------


def head_to_head(
    shots_df: pd.DataFrame, rallies_df: pd.DataFrame, *, min_n: int = DEFAULT_MIN_N
) -> pd.DataFrame:
    """A-vs-B comparison rows: [player, metric, value, n, low_sample].

    Metrics: ``rally_win_rate`` (n = rallies with a known winner), ``winner_rate``
    (share of the player's terminal shots that are winners; n = terminal shots) and
    ``error_share`` (share of all the player's shots that are errors; n = shots).
    Values are None when the denominator is 0.
    """
    columns = ["player", "metric", "value", "n", "low_sample"]
    if shots_df.empty and rallies_df.empty:
        return pd.DataFrame(columns=columns)
    decided = (
        rallies_df[rallies_df["winner"].isin(_PLAYERS)] if not rallies_df.empty else rallies_df
    )
    n_decided = len(decided)
    rows: list[dict[str, object]] = []
    for p in _PLAYERS:
        win_rate = float((decided["winner"] == p).mean()) if n_decided else None
        rows.append(
            {
                "player": p,
                "metric": "rally_win_rate",
                "value": win_rate,
                "n": n_decided,
                "low_sample": n_decided < min_n,
            }
        )
        player_shots = shots_df[shots_df["player"] == p] if not shots_df.empty else shots_df
        n_shots = len(player_shots)
        if n_shots:
            terminal = player_shots[player_shots["outcome"].isin(TERMINAL_OUTCOMES)]
            n_terminal = len(terminal)
            winner_rate = (
                float((terminal["outcome"] == "winner").mean()) if n_terminal else None
            )
            error_share = float(player_shots["outcome"].isin(ERROR_OUTCOMES).mean())
        else:
            n_terminal = 0
            winner_rate = None
            error_share = None
        rows.append(
            {
                "player": p,
                "metric": "winner_rate",
                "value": winner_rate,
                "n": n_terminal,
                "low_sample": n_terminal < min_n,
            }
        )
        rows.append(
            {
                "player": p,
                "metric": "error_share",
                "value": error_share,
                "n": n_shots,
                "low_sample": n_shots < min_n,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def match_trend(
    shots_df: pd.DataFrame,
    rallies_df: pd.DataFrame,
    player: str,
    *,
    min_n: int = DEFAULT_MIN_N,
) -> pd.DataFrame:
    """Per-match trend rows for one player, sorted by match_id.

    Columns: [match_id, n_rallies, n_decided, rallies_won, win_rate, n_shots,
    n_terminal, winner_rate, low_sample]; ``low_sample`` gates on ``n_decided`` (the
    win-rate denominator). Rates are None when their denominator is 0.
    """
    columns = [
        "match_id",
        "n_rallies",
        "n_decided",
        "rallies_won",
        "win_rate",
        "n_shots",
        "n_terminal",
        "winner_rate",
        "low_sample",
    ]
    if rallies_df.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for match_id, match_rallies in rallies_df.groupby("match_id", sort=False):
        decided = match_rallies[match_rallies["winner"].isin(_PLAYERS)]
        n_decided = len(decided)
        rallies_won = int((decided["winner"] == player).sum()) if n_decided else 0
        if not shots_df.empty:
            match_shots = shots_df[
                (shots_df["match_id"] == match_id) & (shots_df["player"] == player)
            ]
            terminal = match_shots[match_shots["outcome"].isin(TERMINAL_OUTCOMES)]
            n_shots = len(match_shots)
            n_terminal = len(terminal)
            winner_rate = (
                float((terminal["outcome"] == "winner").mean()) if n_terminal else None
            )
        else:
            n_shots = 0
            n_terminal = 0
            winner_rate = None
        rows.append(
            {
                "match_id": match_id,
                "n_rallies": len(match_rallies),
                "n_decided": n_decided,
                "rallies_won": rallies_won,
                "win_rate": _rate_or_none(rallies_won, n_decided),
                "n_shots": n_shots,
                "n_terminal": n_terminal,
                "winner_rate": winner_rate,
                "low_sample": n_decided < min_n,
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("match_id")
        .reset_index(drop=True)
    )

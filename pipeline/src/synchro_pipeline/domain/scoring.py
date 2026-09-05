"""Badminton scoring state machine — the pipeline's trust anchor.

Rules encoded (BWF rally scoring):
    * game to 21, win by 2, hard cap at 30 (29-29 → next point wins 30-29)
    * match is best of 3 games
    * server of a rally = winner of the previous rally; first server of a game = winner
      of the previous game (first server of the match is an input — decided by toss)
    * server serves from the right service court when their own score is even, left when odd
    * players change ends after every game, and in the deciding game when the leading
      score first reaches 11

The state machine is used three ways (plan §S1c):
    1. constrain scoreboard-OCR decoding: reject transitions the rules forbid
    2. cross-validate rally segmentation: every score delta must align to a rally boundary
    3. derive the side-switch schedule that anchors player identity across a match

Everything here is pure and deterministic; keep it dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

GAME_TARGET = 21
GAME_CAP = 30
DECIDING_GAME_SWITCH_SCORE = 11
GAMES_TO_WIN_MATCH = 2


class Player(StrEnum):
    A = "A"
    B = "B"

    @property
    def opponent(self) -> Player:
        return Player.B if self is Player.A else Player.A


class ServiceCourt(StrEnum):
    RIGHT = "right"  # even own score
    LEFT = "left"  # odd own score


@dataclass(frozen=True)
class GameScore:
    a: int
    b: int

    def of(self, player: Player) -> int:
        return self.a if player is Player.A else self.b

    def incremented(self, player: Player) -> GameScore:
        return GameScore(self.a + 1, self.b) if player is Player.A else GameScore(self.a, self.b + 1)


def game_winner(score: GameScore) -> Player | None:
    """The player who has won the game at this score, or None if the game is live."""
    for player, own, other in ((Player.A, score.a, score.b), (Player.B, score.b, score.a)):
        if own == GAME_CAP:
            return player
        if own >= GAME_TARGET and own - other >= 2:
            return player
    return None


def is_valid_score(score: GameScore) -> bool:
    """Whether a score tuple can occur in a real game (used to reject OCR misreads)."""
    if score.a < 0 or score.b < 0 or (score.a > GAME_CAP or score.b > GAME_CAP):
        return False
    hi, lo = max(score.a, score.b), min(score.a, score.b)
    if hi == GAME_CAP:
        return lo >= GAME_CAP - 2  # 30 only reachable from live 29-28 or 29-29
    if hi > GAME_TARGET:
        return hi - lo <= 2  # past 21 only through deuce: margin can never exceed 2
    if hi == GAME_TARGET:
        return True
    return True


def is_valid_transition(before: GameScore, after: GameScore) -> bool:
    """Exactly one player gains exactly one point, and `before` was not already terminal."""
    if not (is_valid_score(before) and is_valid_score(after)):
        return False
    if game_winner(before) is not None:
        return False
    da, db = after.a - before.a, after.b - before.b
    return (da, db) in ((1, 0), (0, 1))


def valid_next_scores(before: GameScore) -> frozenset[GameScore]:
    """The (at most two) legal successor scores — the OCR decoder's candidate set."""
    if game_winner(before) is not None:
        return frozenset()
    return frozenset(
        s for s in (before.incremented(Player.A), before.incremented(Player.B))
        if is_valid_score(s)
    )


def is_game_point(score: GameScore, player: Player) -> bool:
    """Whether `player` wins the game by taking the next rally."""
    if game_winner(score) is not None:
        return False
    return game_winner(score.incremented(player)) is player


@dataclass(frozen=True)
class MatchState:
    """Immutable snapshot of a match; advance with `apply_rally`."""

    score: GameScore = GameScore(0, 0)
    game_number: int = 1  # 1-based
    games_a: int = 0
    games_b: int = 0
    server: Player = Player.A  # first server of the match comes from the toss
    a_on_near_side: bool = True  # physical end occupied by player A
    deciding_switch_done: bool = False
    games: tuple[GameScore, ...] = ()  # final scores of completed games

    # -- derived ---------------------------------------------------------------

    @property
    def match_winner(self) -> Player | None:
        if self.games_a >= GAMES_TO_WIN_MATCH:
            return Player.A
        if self.games_b >= GAMES_TO_WIN_MATCH:
            return Player.B
        return None

    @property
    def is_deciding_game(self) -> bool:
        return self.game_number == 2 * GAMES_TO_WIN_MATCH - 1

    def serving_court(self) -> ServiceCourt:
        own = self.score.of(self.server)
        return ServiceCourt.RIGHT if own % 2 == 0 else ServiceCourt.LEFT

    def side_of(self, player: Player) -> str:
        """'near' | 'far' — the physical end this player currently occupies."""
        on_near = self.a_on_near_side if player is Player.A else not self.a_on_near_side
        return "near" if on_near else "far"

    # -- transitions -----------------------------------------------------------

    def apply_rally(self, winner: Player) -> MatchState:
        """Advance the match by one rally won by `winner`."""
        if self.match_winner is not None:
            raise ValueError("match is already over")
        new_score = self.score.incremented(winner)
        finished = game_winner(new_score)

        if finished is None:
            state = replace(self, score=new_score, server=winner)
            # deciding-game end change when the leading score first reaches 11
            if (
                state.is_deciding_game
                and not state.deciding_switch_done
                and max(new_score.a, new_score.b) == DECIDING_GAME_SWITCH_SCORE
            ):
                state = replace(
                    state,
                    a_on_near_side=not state.a_on_near_side,
                    deciding_switch_done=True,
                )
            return state

        # game over: record it, swap ends, winner serves first in the next game
        games_a = self.games_a + (1 if finished is Player.A else 0)
        games_b = self.games_b + (1 if finished is Player.B else 0)
        return replace(
            self,
            score=GameScore(0, 0),
            game_number=self.game_number + 1,
            games_a=games_a,
            games_b=games_b,
            server=finished,
            a_on_near_side=not self.a_on_near_side,
            deciding_switch_done=False,
            games=(*self.games, new_score),
        )

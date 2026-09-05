"""Scoring state machine — the trust anchor gets the most thorough tests."""

from synchro_pipeline.domain.scoring import (
    GameScore,
    MatchState,
    Player,
    ServiceCourt,
    game_winner,
    is_game_point,
    is_valid_score,
    is_valid_transition,
    valid_next_scores,
)


class TestGameWinner:
    def test_straight_game(self):
        assert game_winner(GameScore(21, 15)) is Player.A
        assert game_winner(GameScore(10, 21)) is Player.B

    def test_live_scores(self):
        assert game_winner(GameScore(20, 19)) is None
        assert game_winner(GameScore(20, 20)) is None
        assert game_winner(GameScore(21, 20)) is None  # deuce: win by 2

    def test_deuce_win_by_two(self):
        assert game_winner(GameScore(22, 20)) is Player.A
        assert game_winner(GameScore(25, 23)) is Player.A
        assert game_winner(GameScore(23, 25)) is Player.B

    def test_cap_at_30(self):
        assert game_winner(GameScore(30, 29)) is Player.A
        assert game_winner(GameScore(29, 30)) is Player.B
        assert game_winner(GameScore(29, 29)) is None


class TestScoreValidation:
    def test_basic_valid(self):
        assert is_valid_score(GameScore(0, 0))
        assert is_valid_score(GameScore(21, 19))
        assert is_valid_score(GameScore(29, 29))
        assert is_valid_score(GameScore(30, 28))  # 29-28 -> 30-28
        assert is_valid_score(GameScore(30, 29))

    def test_impossible_scores_rejected(self):
        assert not is_valid_score(GameScore(-1, 0))
        assert not is_valid_score(GameScore(31, 20))
        assert not is_valid_score(GameScore(25, 21))  # margin >2 past 21 is unreachable
        assert not is_valid_score(GameScore(30, 20))  # cap only from 29-28 / 29-29

    def test_transition_exactly_one_point(self):
        assert is_valid_transition(GameScore(5, 3), GameScore(6, 3))
        assert is_valid_transition(GameScore(5, 3), GameScore(5, 4))
        assert not is_valid_transition(GameScore(5, 3), GameScore(7, 3))  # 2-point jump
        assert not is_valid_transition(GameScore(5, 3), GameScore(6, 4))  # both changed
        assert not is_valid_transition(GameScore(5, 3), GameScore(4, 3))  # decrease

    def test_transition_from_terminal_rejected(self):
        assert not is_valid_transition(GameScore(21, 15), GameScore(22, 15))

    def test_valid_next_scores_is_ocr_candidate_set(self):
        assert valid_next_scores(GameScore(20, 20)) == {GameScore(21, 20), GameScore(20, 21)}
        assert valid_next_scores(GameScore(21, 15)) == frozenset()

    def test_game_point(self):
        assert is_game_point(GameScore(20, 10), Player.A)
        assert not is_game_point(GameScore(20, 10), Player.B)
        assert not is_game_point(GameScore(20, 20), Player.A)  # 21-20 is not a win
        assert is_game_point(GameScore(21, 20), Player.A)
        assert is_game_point(GameScore(29, 29), Player.B)  # cap point for both
        assert is_game_point(GameScore(29, 29), Player.A)


class TestMatchState:
    def test_server_is_last_rally_winner(self):
        s = MatchState(server=Player.A)
        s = s.apply_rally(Player.B)
        assert s.server is Player.B
        s = s.apply_rally(Player.A)
        assert s.server is Player.A

    def test_service_court_parity(self):
        s = MatchState(server=Player.A)  # 0-0: even -> right
        assert s.serving_court() is ServiceCourt.RIGHT
        s = s.apply_rally(Player.A)  # A leads 1-0, serves with odd score -> left
        assert s.serving_court() is ServiceCourt.LEFT
        s = s.apply_rally(Player.B)  # B serves at 1 point (odd) -> left
        assert s.server is Player.B
        assert s.serving_court() is ServiceCourt.LEFT

    def test_game_end_swaps_sides_and_server(self):
        s = MatchState(server=Player.A, a_on_near_side=True)
        for _ in range(21):  # B wins 21 straight points
            s = s.apply_rally(Player.B)
        assert s.games_b == 1 and s.game_number == 2
        assert s.score == GameScore(0, 0)
        assert s.server is Player.B  # game winner serves first next game
        assert s.a_on_near_side is False  # ends changed
        assert s.games == (GameScore(0, 21),)

    def test_deciding_game_switch_at_11(self):
        s = MatchState(server=Player.A)
        for _ in range(21):
            s = s.apply_rally(Player.A)  # A takes game 1
        for _ in range(21):
            s = s.apply_rally(Player.B)  # B takes game 2
        assert s.game_number == 3 and s.is_deciding_game
        sides_at_start_of_g3 = s.a_on_near_side
        for _ in range(10):
            s = s.apply_rally(Player.A)
        assert s.a_on_near_side == sides_at_start_of_g3  # 10-x: not yet
        s = s.apply_rally(Player.A)  # reaches 11
        assert s.a_on_near_side != sides_at_start_of_g3
        assert s.deciding_switch_done
        s = s.apply_rally(Player.B)  # no second switch
        assert s.a_on_near_side != sides_at_start_of_g3

    def test_match_winner_and_no_play_after(self):
        s = MatchState(server=Player.A)
        for _ in range(42):
            s = s.apply_rally(Player.A)
        assert s.match_winner is Player.A
        assert s.games == (GameScore(21, 0), GameScore(21, 0))
        try:
            s.apply_rally(Player.B)
            raise AssertionError("expected ValueError after match end")
        except ValueError:
            pass

    def test_deuce_game_runs_to_cap(self):
        s = MatchState(server=Player.A)
        # alternate to 20-20, then trade to 29-29, B takes the cap point
        for _ in range(20):
            s = s.apply_rally(Player.A)
            s = s.apply_rally(Player.B)
        assert s.score == GameScore(20, 20)
        for _ in range(9):
            s = s.apply_rally(Player.A)
            s = s.apply_rally(Player.B)
        assert s.score == GameScore(29, 29)
        s = s.apply_rally(Player.B)
        assert s.games_b == 1
        assert s.games[-1] == GameScore(29, 30)

    def test_side_of_player(self):
        s = MatchState(server=Player.A, a_on_near_side=True)
        assert s.side_of(Player.A) == "near" and s.side_of(Player.B) == "far"

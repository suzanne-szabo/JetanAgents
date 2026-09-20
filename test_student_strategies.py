

import math
import unittest

from direct_llm_agent import DirectLLMAgent
from jetan import JetanBoard, Piece, PieceType, Player
from llm_client import ScriptedClient
from llm_evaluation import LLMEvaluationFunction
from student_strategies import (
    build_evaluation_prompt,
    build_move_prompt,
    choose_fallback,
    evaluate_position_1,
    evaluate_position_2,
    evaluate_position_3,
    parse_evaluation_response,
    parse_move_response,
)


class EvaluatePosition1Tests(unittest.TestCase):

    def test_hand_checked_material_only_position(self):
        # Orange: Chief (6.0) + Warrior (2.0) = 8.0 material.
        # Black: Chief (6.0) = 6.0 material.
        #
        # From ORANGE:
        #   (8.0 - 6.0) / 51.0 = 2/51
        #
        # From BLACK:
        #   (6.0 - 8.0) / 51.0 = -2/51
        board = JetanBoard.from_pieces(
            {
                (4, 4): Piece(Player.ORANGE, PieceType.CHIEF),
                (0, 0): Piece(Player.ORANGE, PieceType.WARRIOR),
                (9, 9): Piece(Player.BLACK, PieceType.CHIEF),
            },
            turn=Player.ORANGE,
        )

        expected_orange = 2.0 / 51.0
        expected_black = -2.0 / 51.0

        actual_orange = evaluate_position_1(
            board,
            Player.ORANGE,
        )
        actual_black = evaluate_position_1(
            board,
            Player.BLACK,
        )

        self.assertAlmostEqual(
            actual_orange,
            expected_orange,
            places=9,
        )
        self.assertAlmostEqual(
            actual_black,
            expected_black,
            places=9,
        )

        self.assertAlmostEqual(
            actual_orange,
            -actual_black,
            places=9,
        )


class EvaluatePosition2Tests(unittest.TestCase):

    def test_hand_checked_mobility_isolated_position(self):
        # One Warrior each means the material score is 0.
        #
        # Orange Warrior at e4 has 8 legal moves.
        # Black Warrior at j9 has 3 legal moves.
        #
        # mobility_score(ORANGE)
        #   = (8 - 3) / (8 + 3)
        #   = 5/11
        #
        # evaluate_position_2:
        #   0.70 * material_score
        #   + 0.30 * mobility_score
        #
        # = 0.70 * 0 + 0.30 * (5/11)
        # = 1.5/11
        board = JetanBoard.from_pieces(
            {
                (4, 4): Piece(
                    Player.ORANGE,
                    PieceType.WARRIOR,
                ),
                (9, 9): Piece(
                    Player.BLACK,
                    PieceType.WARRIOR,
                ),
            },
            turn=Player.ORANGE,
        )

        self.assertEqual(
            board.legal_action_count(Player.ORANGE),
            8,
        )
        self.assertEqual(
            board.legal_action_count(Player.BLACK),
            3,
        )

        expected_orange = 1.5 / 11.0
        expected_black = -1.5 / 11.0

        actual_orange = evaluate_position_2(
            board,
            Player.ORANGE,
        )
        actual_black = evaluate_position_2(
            board,
            Player.BLACK,
        )

        self.assertAlmostEqual(
            actual_orange,
            expected_orange,
            places=9,
        )
        self.assertAlmostEqual(
            actual_black,
            expected_black,
            places=9,
        )

        self.assertAlmostEqual(
            actual_orange,
            -actual_black,
            places=9,
        )


class EvaluatePosition3Tests(unittest.TestCase):

    def test_hand_checked_equal_material_mobility_position(self):
        # One Warrior each -> material difference is 0.
        #
        # Neither Warrior attacks the other, so:
        #   princess_difference = 0
        #   threat_adjustment = 0
        #   strategic_score = 0
        #
        # Orange Warrior at e4 has 8 legal moves.
        # Black Warrior at j9 has 3 legal moves.
        #
        # mobility_score(ORANGE)
        #   = (8 - 3) / (8 + 3)
        #   = 5/11
        #
        # evaluate_position_3:
        #   0.75 * strategic_score
        #   + 0.25 * mobility_score
        #
        # = 0.75 * 0 + 0.25 * (5/11)
        # = 1.25/11
        board = JetanBoard.from_pieces(
            {
                (4, 4): Piece(
                    Player.ORANGE,
                    PieceType.WARRIOR,
                ),
                (9, 9): Piece(
                    Player.BLACK,
                    PieceType.WARRIOR,
                ),
            },
            turn=Player.ORANGE,
        )

        self.assertEqual(
            board.legal_action_count(Player.ORANGE),
            8,
        )
        self.assertEqual(
            board.legal_action_count(Player.BLACK),
            3,
        )

        self.assertFalse(
            (9, 9)
            in board.attacked_locations(Player.ORANGE)
        )
        self.assertFalse(
            (4, 4)
            in board.attacked_locations(Player.BLACK)
        )

        expected_orange = 1.25 / 11.0
        expected_black = -1.25 / 11.0

        actual_orange = evaluate_position_3(
            board,
            Player.ORANGE,
        )
        actual_black = evaluate_position_3(
            board,
            Player.BLACK,
        )

        self.assertAlmostEqual(
            actual_orange,
            expected_orange,
            places=9,
        )
        self.assertAlmostEqual(
            actual_black,
            expected_black,
            places=9,
        )

        self.assertAlmostEqual(
            actual_orange,
            -actual_black,
            places=9,
        )

    def test_princess_attacked_decreases_orange_score(self):

        def make_board(black_warrior_square):
            return JetanBoard.from_pieces(
                {
                    (4, 5): Piece(
                        Player.ORANGE,
                        PieceType.PRINCESS,
                    ),
                    (0, 0): Piece(
                        Player.ORANGE,
                        PieceType.CHIEF,
                    ),
                    black_warrior_square: Piece(
                        Player.BLACK,
                        PieceType.WARRIOR,
                    ),
                },
                turn=Player.ORANGE,
            )

        safe_board = make_board((0, 9))
        attacked_board = make_board((4, 7))

        self.assertFalse(
            (4, 5)
            in safe_board.attacked_locations(
                Player.BLACK
            )
        )

        self.assertTrue(
            (4, 5)
            in attacked_board.attacked_locations(
                Player.BLACK
            )
        )

        score_safe = evaluate_position_3(
            safe_board,
            Player.ORANGE,
        )

        score_attacked = evaluate_position_3(
            attacked_board,
            Player.ORANGE,
        )

        self.assertLess(
            score_attacked,
            score_safe,
        )

    def test_used_escape_makes_attacked_princess_more_dangerous(
        self,
    ):

        def make_board(used_escape):
            return JetanBoard.from_pieces(
                {
                    (4, 5): Piece(
                        Player.ORANGE,
                        PieceType.PRINCESS,
                    ),
                    (0, 0): Piece(
                        Player.ORANGE,
                        PieceType.CHIEF,
                    ),
                    (4, 7): Piece(
                        Player.BLACK,
                        PieceType.WARRIOR,
                    ),
                },
                turn=Player.ORANGE,
                used_escape=used_escape,
            )

        escape_available = make_board(
            (False, False)
        )
        escape_used = make_board(
            (True, False)
        )

        score_escape_available = evaluate_position_3(
            escape_available,
            Player.ORANGE,
        )

        score_escape_used = evaluate_position_3(
            escape_used,
            Player.ORANGE,
        )

        self.assertLess(
            score_escape_used,
            score_escape_available,
        )

    def test_exposed_nonprincess_piece_decreases_score(self):

        def make_board(black_dwar_square):
            return JetanBoard.from_pieces(
                {
                    (4, 3): Piece(
                        Player.ORANGE,
                        PieceType.DWAR,
                    ),
                    black_dwar_square: Piece(
                        Player.BLACK,
                        PieceType.DWAR,
                    ),
                },
                turn=Player.ORANGE,
            )

        safe_board = make_board((9, 9))
        exposed_board = make_board((4, 0))

        self.assertFalse(
            (4, 3)
            in safe_board.attacked_locations(
                Player.BLACK
            )
        )

        self.assertTrue(
            (4, 3)
            in exposed_board.attacked_locations(
                Player.BLACK
            )
        )

        self.assertEqual(
            safe_board.legal_action_count(
                Player.ORANGE
            ),
            exposed_board.legal_action_count(
                Player.ORANGE
            ),
        )

        score_safe = evaluate_position_3(
            safe_board,
            Player.ORANGE,
        )

        score_exposed = evaluate_position_3(
            exposed_board,
            Player.ORANGE,
        )

        self.assertLess(
            score_exposed,
            score_safe,
        )

    def test_perspective_reversal(self):
        board = JetanBoard.from_pieces(
            {
                (4, 5): Piece(
                    Player.ORANGE,
                    PieceType.PRINCESS,
                ),
                (0, 0): Piece(
                    Player.ORANGE,
                    PieceType.CHIEF,
                ),
                (0, 9): Piece(
                    Player.BLACK,
                    PieceType.WARRIOR,
                ),
            },
            turn=Player.ORANGE,
        )

        score_orange = evaluate_position_3(
            board,
            Player.ORANGE,
        )

        score_black = evaluate_position_3(
            board,
            Player.BLACK,
        )

        self.assertAlmostEqual(
            score_orange,
            -score_black,
            places=9,
        )


class BoundsAndFiniteTests(unittest.TestCase):

    @staticmethod
    def _extreme_material_board():
        pieces = {}

        x = 0

        for _ in range(9):
            pieces[(x, 0)] = Piece(
                Player.ORANGE,
                PieceType.CHIEF,
            )
            x += 1

        pieces[(x, 0)] = Piece(
            Player.ORANGE,
            PieceType.PRINCESS,
        )

        pieces[(0, 9)] = Piece(
            Player.BLACK,
            PieceType.PANTHAN,
        )

        return JetanBoard.from_pieces(
            pieces,
            turn=Player.ORANGE,
        )

    def test_all_evaluators_clip_to_declared_bound(self):
        board = self._extreme_material_board()

        for evaluate in (
            evaluate_position_1,
            evaluate_position_2,
            evaluate_position_3,
        ):
            with self.subTest(
                evaluate=evaluate.__name__
            ):
                orange_score = evaluate(
                    board,
                    Player.ORANGE,
                )

                black_score = evaluate(
                    board,
                    Player.BLACK,
                )

                self.assertTrue(
                    math.isfinite(orange_score)
                )
                self.assertTrue(
                    math.isfinite(black_score)
                )

                self.assertLessEqual(
                    orange_score,
                    0.99,
                )
                self.assertGreaterEqual(
                    orange_score,
                    -0.99,
                )

                self.assertAlmostEqual(
                    orange_score,
                    0.99,
                    places=9,
                )

                self.assertAlmostEqual(
                    black_score,
                    -0.99,
                    places=9,
                )


class ParseEvaluationResponseTests(unittest.TestCase):

    def test_accepts_well_formed_response(self):
        self.assertEqual(
            parse_evaluation_response(
                '{"score": 0.25}'
            ),
            0.25,
        )

    def test_accepts_both_inclusive_boundary_values(
        self,
    ):
        self.assertEqual(
            parse_evaluation_response(
                '{"score": 0.99}'
            ),
            0.99,
        )

        self.assertEqual(
            parse_evaluation_response(
                '{"score": -0.99}'
            ),
            -0.99,
        )

    def test_rejects_markdown_fence(self):
        with self.assertRaises(ValueError):
            parse_evaluation_response(
                '```json\n{"score": 0.5}\n```'
            )

    def test_rejects_multiple_lines(self):
        with self.assertRaises(ValueError):
            parse_evaluation_response(
                '{"score": 0.5}\n'
                '{"score": 0.6}'
            )

    def test_rejects_surrounding_commentary(self):
        with self.assertRaises(ValueError):
            parse_evaluation_response(
                'Sure! {"score": 0.5}'
            )

    def test_rejects_invalid_json(self):
        with self.assertRaises(ValueError):
            parse_evaluation_response(
                "not json"
            )

    def test_rejects_duplicate_keys(self):
        with self.assertRaises(ValueError):
            parse_evaluation_response(
                '{"score": 0.1, '
                '"score": 0.2}'
            )

    def test_rejects_missing_key(self):
        with self.assertRaises(ValueError):
            parse_evaluation_response("{}")

    def test_rejects_additional_key(self):
        with self.assertRaises(ValueError):
            parse_evaluation_response(
                '{"score": 0.1, '
                '"confidence": 0.9}'
            )

    def test_rejects_boolean_score(self):
        with self.assertRaises(ValueError):
            parse_evaluation_response(
                '{"score": true}'
            )

    def test_rejects_string_score(self):
        with self.assertRaises(ValueError):
            parse_evaluation_response(
                '{"score": "0.5"}'
            )

    def test_rejects_nonfinite_score(self):
        with self.assertRaises(ValueError):
            parse_evaluation_response(
                '{"score": NaN}'
            )

        with self.assertRaises(ValueError):
            parse_evaluation_response(
                '{"score": Infinity}'
            )

    def test_rejects_out_of_range_score(self):
        with self.assertRaises(ValueError):
            parse_evaluation_response(
                '{"score": 1.5}'
            )

        with self.assertRaises(ValueError):
            parse_evaluation_response(
                '{"score": -1.5}'
            )


class ParseMoveResponseTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.actions = JetanBoard.initial().actions()
        cls.good_token = str(cls.actions[0])

    def test_accepts_well_formed_legal_move(self):
        move = parse_move_response(
            '{"move": "%s"}'
            % self.good_token,
            self.actions,
        )

        self.assertEqual(
            str(move),
            self.good_token,
        )

        self.assertIn(
            move,
            self.actions,
        )

    def test_rejects_markdown_fence(self):
        with self.assertRaises(ValueError):
            parse_move_response(
                '```json\n{"move": "%s"}\n```'
                % self.good_token,
                self.actions,
            )

    def test_rejects_invalid_json(self):
        with self.assertRaises(ValueError):
            parse_move_response(
                "not json",
                self.actions,
            )

    def test_rejects_duplicate_keys(self):
        with self.assertRaises(ValueError):
            parse_move_response(
                '{"move": "%s", "move": "%s"}'
                % (
                    self.good_token,
                    self.good_token,
                ),
                self.actions,
            )

    def test_rejects_missing_key(self):
        with self.assertRaises(ValueError):
            parse_move_response(
                "{}",
                self.actions,
            )

    def test_rejects_extra_key(self):
        with self.assertRaises(ValueError):
            parse_move_response(
                '{"move": "%s", '
                '"confidence": 0.9}'
                % self.good_token,
                self.actions,
            )

    def test_rejects_wrong_type(self):
        with self.assertRaises(ValueError):
            parse_move_response(
                '{"move": 5}',
                self.actions,
            )

    def test_rejects_unknown_malformed_move_token(
        self,
    ):
        with self.assertRaises(ValueError):
            parse_move_response(
                '{"move": "zz-zz"}',
                self.actions,
            )

    def test_rejects_well_formed_but_illegal_move(
        self,
    ):
        with self.assertRaises(ValueError):
            parse_move_response(
                '{"move": "e4-e5"}',
                self.actions,
            )


class LLMEvaluationFallbackAndBudgetTests(
    unittest.TestCase
):

    def test_malformed_response_triggers_fallback(
        self,
    ):
        board = JetanBoard.initial()
        client = ScriptedClient(
            ["not json"]
        )

        evaluator = LLMEvaluationFunction(
            client,
            build_evaluation_prompt,
            parse_evaluation_response,
            evaluate_position_3,
            max_model_calls=64,
        )

        value = evaluator(
            board,
            Player.ORANGE,
        )

        self.assertEqual(
            value,
            evaluate_position_3(
                board,
                Player.ORANGE,
            ),
        )

        self.assertEqual(
            evaluator.model_calls,
            1,
        )
        self.assertEqual(
            evaluator.fallback_calls,
            1,
        )
        self.assertEqual(
            evaluator.cache_hits,
            0,
        )

        self.assertEqual(
            len(client.messages_seen),
            1,
        )

    def test_zero_model_call_budget_never_invokes_client(
        self,
    ):
        board = JetanBoard.initial()

        client = ScriptedClient(
            ['{"score": 0.5}']
        )

        evaluator = LLMEvaluationFunction(
            client,
            build_evaluation_prompt,
            parse_evaluation_response,
            evaluate_position_3,
            max_model_calls=0,
        )

        value = evaluator(
            board,
            Player.ORANGE,
        )

        self.assertEqual(
            value,
            evaluate_position_3(
                board,
                Player.ORANGE,
            ),
        )

        self.assertEqual(
            evaluator.model_calls,
            0,
        )
        self.assertEqual(
            evaluator.fallback_calls,
            1,
        )
        self.assertEqual(
            len(client.messages_seen),
            0,
        )

    def test_repeated_lookup_hits_cache_instead_of_client(
        self,
    ):
        board = JetanBoard.initial()

        client = ScriptedClient(
            ['{"score": 0.4}']
        )

        evaluator = LLMEvaluationFunction(
            client,
            build_evaluation_prompt,
            parse_evaluation_response,
            evaluate_position_3,
            max_model_calls=64,
        )

        first = evaluator(
            board,
            Player.ORANGE,
        )

        second = evaluator(
            board,
            Player.ORANGE,
        )

        self.assertEqual(
            first,
            0.4,
        )
        self.assertEqual(
            second,
            0.4,
        )

        self.assertEqual(
            evaluator.model_calls,
            1,
        )
        self.assertEqual(
            evaluator.cache_hits,
            1,
        )
        self.assertEqual(
            len(client.messages_seen),
            1,
        )


class DirectLLMAgentScriptedClientTests(
    unittest.TestCase
):

    def test_valid_response_is_used_without_fallback(
        self,
    ):
        board = JetanBoard.initial()

        good_token = str(
            board.actions()[10]
        )

        client = ScriptedClient(
            [
                '{"move": "%s"}'
                % good_token
            ]
        )

        agent = DirectLLMAgent(
            client,
            build_move_prompt,
            parse_move_response,
            "Test Direct",
            choose_fallback,
        )

        move = agent.choose_action(board)

        self.assertEqual(
            str(move),
            good_token,
        )

        self.assertEqual(
            agent.model_calls,
            1,
        )
        self.assertEqual(
            agent.fallback_calls,
            0,
        )

    def test_malformed_response_falls_back_to_legal_move(
        self,
    ):
        board = JetanBoard.initial()

        client = ScriptedClient(
            ["not json at all"]
        )

        agent = DirectLLMAgent(
            client,
            build_move_prompt,
            parse_move_response,
            "Test Direct",
            choose_fallback,
        )

        move = agent.choose_action(board)

        self.assertIn(
            move,
            board.actions(),
        )

        self.assertEqual(
            agent.model_calls,
            1,
        )
        self.assertEqual(
            agent.fallback_calls,
            1,
        )

    def test_illegal_move_response_falls_back_to_legal_move(
        self,
    ):
        board = JetanBoard.initial()

        client = ScriptedClient(
            ['{"move": "z9-z9"}']
        )

        agent = DirectLLMAgent(
            client,
            build_move_prompt,
            parse_move_response,
            "Test Direct",
            choose_fallback,
        )

        move = agent.choose_action(board)

        self.assertIn(
            move,
            board.actions(),
        )

        self.assertEqual(
            agent.model_calls,
            1,
        )
        self.assertEqual(
            agent.fallback_calls,
            1,
        )

    def test_deterministic_across_short_scripted_match(
        self,
    ):
        script = [
            '{"move": "%s"}'
            % str(
                JetanBoard.initial().actions()[5]
            ),
            "garbage",
            '{"move": "a1-a2", "extra": 1}',
        ]

        def play_three_plies():
            client = ScriptedClient(
                list(script)
            )

            agent = DirectLLMAgent(
                client,
                build_move_prompt,
                parse_move_response,
                "Test Direct",
                choose_fallback,
            )

            board = JetanBoard.initial()
            moves = []

            for _ in range(3):
                move = agent.choose_action(board)

                moves.append(
                    str(move)
                )

                board = board.result(move)

            return moves

        first_run = play_three_plies()
        second_run = play_three_plies()

        self.assertEqual(
            first_run,
            second_run,
        )


if __name__ == "__main__":
    unittest.main()
"""Public tests for student-owned alpha-beta recursion and ordering keys."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from alpha_beta_agent import AlphaBetaAgent
from alpha_beta_ordering import order_actions_1, order_actions_2
from hw06_measured_minimax import MeasuredMinimaxAgent
from hw06_positions import POSITIONS
from jetan import Player


@dataclass(frozen=True)
class ToyBoard:
    node: str
    tree: dict[str, tuple[Player, dict[str, str], float | None]]
    action_calls: dict[str, int] | None = None

    def player(self) -> Player:
        return self.tree[self.node][0]

    def actions(self) -> tuple[str, ...]:
        if self.action_calls is not None:
            self.action_calls[self.node] = self.action_calls.get(self.node, 0) + 1
        return tuple(self.tree[self.node][1])

    def result(self, action: str) -> ToyBoard:
        return ToyBoard(
            self.tree[self.node][1][action],
            self.tree,
            self.action_calls,
        )

    def is_terminal(self) -> bool:
        return self.tree[self.node][2] is not None

    def utility(self, perspective: Player) -> float:
        value = self.tree[self.node][2]
        if value is None:
            raise ValueError("nonterminal state")
        return value if perspective is Player.ORANGE else -value


def toy_evaluator(board: ToyBoard, perspective: Player) -> float:
    del board, perspective
    return 0.0


def canonical_order(
    board: ToyBoard, actions: tuple[str, ...], perspective: Player
) -> tuple[str, ...]:
    del board, perspective
    return actions


class AlphaBetaRequirements(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = {
            "root": (Player.ORANGE, {"a": "a", "b": "b"}, None),
            "a": (Player.BLACK, {"a1": "a1", "a2": "a2"}, None),
            "b": (Player.BLACK, {"b1": "b1", "b2": "b2"}, None),
            "a1": (Player.ORANGE, {}, 0.3),
            "a2": (Player.ORANGE, {}, 0.5),
            "b1": (Player.ORANGE, {}, 0.2),
            "b2": (Player.ORANGE, {}, 0.4),
        }

    def test_alpha_beta_matches_reference_with_exact_work(self) -> None:
        board = ToyBoard("root", self.tree)
        reference = MeasuredMinimaxAgent("reference", 2, toy_evaluator)  # type: ignore[arg-type]
        agent = AlphaBetaAgent("alpha", 2, toy_evaluator, canonical_order)  # type: ignore[arg-type]
        self.assertEqual(agent.choose_action(board), reference.choose_action(board))  # type: ignore[arg-type]
        self.assertAlmostEqual(agent.last_value, reference.last_value)
        metrics = agent.last_metrics
        self.assertEqual(
            (metrics.visited_states, metrics.expanded_nodes,
             metrics.generated_actions, metrics.evaluated_states,
             metrics.pruned_actions, metrics.maximum_depth),
            (6, 3, 6, 3, 1, 2),
        )

    def test_terminal_precedes_cutoff_with_fixed_root_perspective(self) -> None:
        tree = {
            "root": (Player.BLACK, {"win": "terminal"}, None),
            "terminal": (Player.BLACK, {}, -1.0),
        }
        called = False

        def evaluator(board: ToyBoard, perspective: Player) -> float:
            nonlocal called
            called = True
            return 0.0

        agent = AlphaBetaAgent("terminal", 1, evaluator, canonical_order)  # type: ignore[arg-type]
        agent.choose_action(ToyBoard("root", tree))  # type: ignore[arg-type]
        self.assertFalse(called)
        self.assertEqual(agent.last_value, 1.0)

    def test_student_alpha_cutoff_case(self) -> None:
       
        tree = {
            "root": (Player.ORANGE, {"A": "A", "B": "B", "C": "C"}, None),
            "A":    (Player.BLACK,  {"a1": "a1", "a2": "a2", "a3": "a3"}, None),
            "B":    (Player.BLACK, {"b1": "b1", "b2": "b2", "b3": "b3"}, None),
            "C":    (Player.BLACK, {"c1": "c1", "c2": "c2", "c3": "c3"}, None),
          
            "a1":   (Player.ORANGE, {}, 0.7),
            "a2":   (Player.ORANGE, {}, 0.4),
            "a3":   (Player.ORANGE, {}, 0.5),

            "b1":   (Player.ORANGE, {}, 0.4),
            "b2":   (Player.ORANGE, {}, 0.3),
            "b3":   (Player.ORANGE, {}, 0.8),

            "c1":   (Player.ORANGE, {}, 0.8),
            "c2":   (Player.ORANGE, {}, 0.2),
            "c3":   (Player.ORANGE, {}, 0.3),
        }
        board = ToyBoard("root", tree)
        agent = AlphaBetaAgent("cutoff", 2, toy_evaluator, canonical_order)
        reference = MeasuredMinimaxAgent("reference", 2, toy_evaluator)
        move = agent.choose_action(board)
        self.assertEqual(move, reference.choose_action(board))
        self.assertAlmostEqual(agent.last_value, reference.last_value)
        metrics = agent.last_metrics
        self.assertEqual(
            (metrics.visited_states, metrics.expanded_nodes,
                metrics.generated_actions, metrics.evaluated_states,
                metrics.pruned_actions, metrics.maximum_depth),
            (10, 4, 12, 6, 3, 2)
        )
        self.assertEqual(move, "A")
        self.assertAlmostEqual(agent.last_value, 0.4)
        
        self.assertEqual(reference.last_metrics.visited_states, 13)
        self.assertEqual(reference.last_metrics.pruned_actions, 0)




    def test_canonical_root_tie_does_not_use_below_root_ordering(self) -> None:
        tree = {
            "root": (Player.ORANGE, {"first": "x", "second": "y"}, None),
            "x": (Player.BLACK, {}, 0.0),
            "y": (Player.BLACK, {}, 0.0),
        }

        def reverse_order(
            board: ToyBoard,
            actions: tuple[str, ...],
            perspective: Player,
        ) -> tuple[str, ...]:
            del board, perspective
            return tuple(reversed(actions))

        agent = AlphaBetaAgent("ties", 1, toy_evaluator, reverse_order)  # type: ignore[arg-type]
        self.assertEqual(
            agent.choose_action(ToyBoard("root", tree)),
            "first",
        )  # type: ignore[arg-type]

    def test_ordering_is_applied_below_root_only(self) -> None:
        calls: list[str] = []

        def recording_order(
            board: ToyBoard,
            actions: tuple[str, ...],
            perspective: Player,
        ) -> tuple[str, ...]:
            del perspective
            calls.append(board.node)
            return actions

        agent = AlphaBetaAgent("levels", 2, toy_evaluator, recording_order)  # type: ignore[arg-type]
        agent.choose_action(ToyBoard("root", self.tree))  # type: ignore[arg-type]
        self.assertNotIn("root", calls)
        self.assertTrue(calls)

    def test_each_expanded_node_generates_actions_once(self) -> None:
        action_calls: dict[str, int] = {}
        agent = AlphaBetaAgent(
            "actions-once",
            2,
            toy_evaluator,
            canonical_order,
        )  # type: ignore[arg-type]
        agent.choose_action(ToyBoard("root", self.tree, action_calls))  # type: ignore[arg-type]
        self.assertEqual(sum(action_calls.values()), agent.last_metrics.expanded_nodes)
        self.assertTrue(all(count == 1 for count in action_calls.values()))


class OrderingRequirements(unittest.TestCase):
    def test_each_ordering_is_a_repeatable_permutation_of_canonical_input(self) -> None:
        for position in POSITIONS:
            board = position.board
            canonical = board.actions()
            for ordering in (order_actions_1, order_actions_2):
                first = ordering(board, canonical, board.player())
                second = ordering(board, canonical, board.player())
                self.assertEqual(first, second, position.identifier)
                self.assertEqual(len(first), len(canonical), position.identifier)
                self.assertEqual(set(first), set(canonical), position.identifier)

    def test_orderings_differ_on_at_least_one_supplied_position(self) -> None:
        self.assertTrue(any(
            order_actions_1(p.board, p.board.actions(), p.board.player())
            != order_actions_2(p.board, p.board.actions(), p.board.player())
            for p in POSITIONS
        ))


if __name__ == "__main__":
    unittest.main()

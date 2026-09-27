"""Stable action-ordering wrappers and student ranking-key stubs."""

from collections.abc import Callable
from typing import Any

from jetan import JetanBoard, Move, Player, PieceType
from hw06_config import SELECTED_EVALUATOR

OrderingFunction = Callable[
    [JetanBoard, tuple[Move, ...], Player], tuple[Move, ...]
]

PIECE_VALUES = {
    PieceType.PANTHAN: 1.0, PieceType.WARRIOR: 2.0, PieceType.PADWAR: 2.0,
    PieceType.DWAR: 3.0, PieceType.THOAT: 3.5, PieceType.FLIER: 4.0,
    PieceType.CHIEF: 6.0, PieceType.PRINCESS: 8.0,
}

def ordering_key_1(
    board: JetanBoard, action: Move, perspective: Player
) -> Any:
    """Return the first ranking key; larger keys are searched first at MAX."""
 
    mover = board.player()
    attacker = board.at(action.source)
    victim = board.at(action.destination)
    score = 0.0

    # Tier 1: immediate win (Princess captured, or Chief takes Chief)
    if victim is not None and (
        victim.kind is PieceType.PRINCESS or 
        (victim.kind is PieceType.CHIEF and attacker.kind is PieceType.CHIEF)
    ) :
        score = 1000.0

    # Tier 2: Princess steps from an attacked square to an unattacked one
    elif attacker.kind is PieceType.PRINCESS:
        threatened = board.attacked_locations(mover.opponent)
        if action.source in threatened and action.destination not in threatened:
            score = 600.0 if board.used_escape[mover - 1] else 500.0

    # Tier 3: capture, most valuable victim / least valuable attacker
    if score == 0.0 and victim is not None:
        score = 10 * PIECE_VALUES[victim.kind] - PIECE_VALUES[attacker.kind]

    # Score is from the mover's view; flip it to the root player's view
    return score if board.player() is perspective else -score


def ordering_key_2(
    board: JetanBoard, action: Move, perspective: Player
) -> Any:
    """Return a distinct ranking key; larger keys are searched first at MAX."""
    child = board.result(action)
    if child.is_terminal() :
        return float(child.utility(perspective))
    else:
        return float(SELECTED_EVALUATOR(child,perspective))


def _stable_order(
    board: JetanBoard,
    actions: tuple[Move, ...],
    perspective: Player,
    key: Callable[[JetanBoard, Move, Player], Any],
) -> tuple[Move, ...]:
    """Rank the supplied canonical tuple while retaining input-order ties."""
    return tuple(
        sorted(
            actions,
            key=lambda action: key(board, action, perspective),
            reverse=board.player() is perspective,
        )
    )


def order_actions_1(
    board: JetanBoard, actions: tuple[Move, ...], perspective: Player
) -> tuple[Move, ...]:
    """Apply the first student key to the supplied canonical action tuple."""
    return _stable_order(board, actions, perspective, ordering_key_1)


def order_actions_2(
    board: JetanBoard, actions: tuple[Move, ...], perspective: Player
) -> tuple[Move, ...]:
    """Apply the second student key to the supplied canonical action tuple."""
    return _stable_order(board, actions, perspective, ordering_key_2)

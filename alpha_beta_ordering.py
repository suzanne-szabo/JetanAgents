"""Stable action-ordering wrappers and student ranking-key stubs."""

from collections.abc import Callable
from typing import Any

from jetan import JetanBoard, Move, Player

OrderingFunction = Callable[
    [JetanBoard, tuple[Move, ...], Player], tuple[Move, ...]
]


def ordering_key_1(
    board: JetanBoard, action: Move, perspective: Player
) -> Any:
    """Return the first ranking key; larger keys are searched first at MAX."""
    # TODO: Design the first general action-ordering key.
    del board, action, perspective
    return 0


def ordering_key_2(
    board: JetanBoard, action: Move, perspective: Player
) -> Any:
    """Return a distinct ranking key; larger keys are searched first at MAX."""
    # TODO: Design the second general action-ordering key.
    del board, action, perspective
    return 0


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

"""Implement student evaluation, prompting, parsing, and fallback functions."""

from collections.abc import Sequence

from jetan import JetanBoard, Move, Player, PieceType
import json
import math



def evaluate_position_1(board: JetanBoard, perspective: Player) -> float:
    piece_values = {
        PieceType.PANTHAN: 1.0,
        PieceType.WARRIOR: 2.0,
        PieceType.PADWAR: 2.0,
        PieceType.DWAR: 3.0,
        PieceType.THOAT: 3.5,
        PieceType.FLIER: 4.0,
        PieceType.CHIEF: 6.0,
        PieceType.PRINCESS: 8.0,
    }

    my_material = sum(
        piece_values[piece.kind] for _, piece in board.pieces(perspective)
    )
    opponent_material = sum(
        piece_values[piece.kind] for _, piece in board.pieces(perspective.opponent)
    )

    score = (my_material - opponent_material) / 51.0
    return max(-0.99, min(0.99, score))


def evaluate_position_2(board: JetanBoard, perspective: Player) -> float:
    piece_values = {
        PieceType.PANTHAN: 1.0,
        PieceType.WARRIOR: 2.0,
        PieceType.PADWAR: 2.0,
        PieceType.DWAR: 3.0,
        PieceType.THOAT: 3.5,
        PieceType.FLIER: 4.0,
        PieceType.CHIEF: 6.0,
        PieceType.PRINCESS: 8.0,
    }

    my_material = sum(
        piece_values[piece.kind]
        for _, piece in board.pieces(perspective)
    )

    opponent_material = sum(
        piece_values[piece.kind]
        for _, piece in board.pieces(perspective.opponent)
    )

    material_score = (my_material - opponent_material) / 51.0



    my_moves = board.legal_action_count(perspective)
    opponent_moves = board.legal_action_count(perspective.opponent)

    total_moves = my_moves + opponent_moves

    if total_moves == 0:
        mobility_score = 0.0
    else:
        mobility_score = (my_moves - opponent_moves) / total_moves



    score = (
        0.70 * material_score
        + 0.30 * mobility_score
    )

    return max(-0.99, min(0.99, score))


def evaluate_position_3(board: JetanBoard, perspective: Player) -> float:
    piece_values = {
        PieceType.PANTHAN: 1.0,
        PieceType.WARRIOR: 2.0,
        PieceType.PADWAR: 2.0,
        PieceType.DWAR: 3.0,
        PieceType.THOAT: 3.5,
        PieceType.FLIER: 4.0,
        PieceType.CHIEF: 6.0,
        PieceType.PRINCESS: 8.0,
    }

    my_material = sum(
        piece_values[piece.kind]
        for _, piece in board.pieces(perspective)
    )

    opponent_material = sum(
        piece_values[piece.kind]
        for _, piece in board.pieces(perspective.opponent)
    )

    material_difference = my_material - opponent_material

    def princess_danger(player: Player) -> float:
        princess_location = None

        for location, piece in board.pieces(player):
            if piece.kind is PieceType.PRINCESS:
                princess_location = location
                break

        if princess_location is None:
            return 0.0

        opponent_attacks = board.attacked_locations(player.opponent)

        if princess_location not in opponent_attacks:
            return 0.0

        if board.used_escape[player - 1]:
            return 7.0

        return 3.5

    my_princess_danger = princess_danger(perspective)
    opponent_princess_danger = princess_danger(perspective.opponent)

    princess_difference = (
        opponent_princess_danger - my_princess_danger
    )


    def exposed_material(player: Player) -> float:
        opponent_attacks = board.attacked_locations(player.opponent)
        total = 0.0

        for location, piece in board.pieces(player):
            if piece.kind is PieceType.PRINCESS:
                continue

            if location in opponent_attacks:
                total += piece_values[piece.kind]

        return total

    my_exposed_material = exposed_material(perspective)
    opponent_exposed_material = exposed_material(perspective.opponent)

    threat_adjustment = 0.20 * (
        opponent_exposed_material - my_exposed_material
    )

    threat_adjustment = max(
        -4.0,
        min(4.0, threat_adjustment),
    )

    
    strategic_points = (
        material_difference
        + princess_difference
        + threat_adjustment
    )

    strategic_score = strategic_points / 51.0

    
    my_moves = board.legal_action_count(perspective)
    opponent_moves = board.legal_action_count(perspective.opponent)

    total_moves = my_moves + opponent_moves

    if total_moves == 0:
        mobility_score = 0.0
    else:
        mobility_score = (
            my_moves - opponent_moves
        ) / total_moves

    
    score = (
        0.75 * strategic_score
        + 0.25 * mobility_score
    )

    return max(-0.99, min(0.99, score))


def build_evaluation_prompt(
    board: JetanBoard, perspective: Player
) -> Sequence[dict[str, str]]:
    orange_escape = (
        "used"
        if board.used_escape[Player.ORANGE - 1]
        else "available"
    )

    black_escape = (
        "used"
        if board.used_escape[Player.BLACK - 1]
        else "available"
    )

    orange_moves = board.legal_action_count(Player.ORANGE)
    black_moves = board.legal_action_count(Player.BLACK)

    orange_princess = None
    black_princess = None

    for location, piece in board.pieces():
        if piece.kind is PieceType.PRINCESS:
            if piece.player is Player.ORANGE:
                orange_princess = location
            else:
                black_princess = location

    orange_princess_attacked = (
        orange_princess is not None
        and orange_princess in board.attacked_locations(Player.BLACK)
    )

    black_princess_attacked = (
        black_princess is not None
        and black_princess in board.attacked_locations(Player.ORANGE)
    )

    system_message = """
You are a Jetan position evaluator.

Do not restate or transcribe the board.
Do not explain your reasoning.
Do not describe the pieces or position.

Evaluate the supplied nonterminal board from the FIXED PERSPECTIVE
and immediately return the required JSON result.

Return exactly one line of raw JSON like this:
{"score": 0.25}

Response rules:
- output JSON immediately
- exactly one key named "score"
- score must be a finite JSON number
- score must be between -0.99 and 0.99
- no markdown
- no explanation
- no commentary
- no board description
- no extra text
""".strip()

    user_message = f"""
<TASK>
Evaluate this nonterminal Jetan position.
</TASK>

<PERSPECTIVE>
fixed={perspective.name}
to_move={board.player().name}
</PERSPECTIVE>

<SCORE_MEANING>
positive=favors {perspective.name}
negative=favors {perspective.opponent.name}
zero=approximately even
+1 and -1 are reserved for terminal wins and losses
</SCORE_MEANING>

<IMPORTANT_RULES>
Capturing the opponent Princess wins the game.
A Chief capturing the opposing Chief wins the game.
An attacked Princess may use one long escape if that escape is still available.
</IMPORTANT_RULES>

<PIECES>
Uppercase=ORANGE
Lowercase=BLACK
W=Warrior
A=Padwar
D=Dwar
F=Flier
C=Chief
R=Princess
T=Thoat
P=Panthan
.=empty
</PIECES>

<STATE_FEATURES>
ORANGE legal moves={orange_moves}
BLACK legal moves={black_moves}
ORANGE Princess attacked={"yes" if orange_princess_attacked else "no"}
BLACK Princess attacked={"yes" if black_princess_attacked else "no"}
ORANGE Princess escape={orange_escape}
BLACK Princess escape={black_escape}
</STATE_FEATURES>

<STRATEGIC_PRIORITIES>
1. Princess safety and immediate serious threats
2. Material and importance of remaining pieces
3. Threatened or exposed valuable pieces
4. Mobility and activity
</STRATEGIC_PRIORITIES>

<BOARD>
{board}
</BOARD>

Do not restate or analyze the board.
Return the JSON answer immediately.

{{"score": <number>}}
""".strip()

    return (
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    )



def parse_evaluation_response(response: str) -> float:
    if "\n" in response or "\r" in response:
        raise ValueError("response must be one line")

    def reject_duplicate_keys(pairs):
        result = {}

        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value

        return result

    try:
        data = json.loads(
            response,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid evaluation response") from error

    if not isinstance(data, dict):
        raise ValueError("response must be a JSON object")

    if set(data.keys()) != {"score"}:
        raise ValueError("response must contain exactly the key 'score'")

    score = data["score"]

    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("score must be a JSON number")

    score = float(score)

    if not math.isfinite(score):
        raise ValueError("score must be finite")

    if score < -0.99 or score > 0.99:
        raise ValueError("score must be between -0.99 and 0.99")

    return score


def build_move_prompt(
    board: JetanBoard,
    actions: tuple[Move, ...],
    recent_moves: tuple[Move, ...],
) -> Sequence[dict[str, str]]:
    mover = board.player()

    orange_escape = (
        "used"
        if board.used_escape[Player.ORANGE - 1]
        else "available"
    )

    black_escape = (
        "used"
        if board.used_escape[Player.BLACK - 1]
        else "available"
    )

    orange_princess = None
    black_princess = None

    for location, piece in board.pieces():
        if piece.kind is PieceType.PRINCESS:
            if piece.player is Player.ORANGE:
                orange_princess = location
            else:
                black_princess = location

    orange_princess_attacked = (
        orange_princess is not None
        and orange_princess in board.attacked_locations(Player.BLACK)
    )

    black_princess_attacked = (
        black_princess is not None
        and black_princess in board.attacked_locations(Player.ORANGE)
    )

    legal_move_strings = [str(action) for action in actions]
    recent_move_strings = [str(move) for move in recent_moves]

    immediate_winning_moves = []

    for action in actions:
        successor = board.result(action)

        if (
            successor.is_terminal()
            and successor.utility(mover) > 0
        ):
            immediate_winning_moves.append(str(action))

    system_message = """
You are a Jetan move-selection agent.

Choose exactly one move from the supplied authoritative legal-move list.

Return exactly one line of raw JSON like this:
{"move": "a1-b2"}

Response rules:
- exactly one key named "move"
- the value must be a string copied verbatim from LEGAL_MOVES
- no markdown
- no explanation
- no extra text
""".strip()

    user_message = f"""
<TASK>
Choose one move for the player to move on this Jetan board.
</TASK>

<PLAYER_TO_MOVE>
{mover.name}
</PLAYER_TO_MOVE>

<IMPORTANT_RULES>
Capturing the opponent Princess wins the game.
A Chief capturing the opposing Chief wins the game.
An attacked Princess may use one long escape if that escape is still available.
</IMPORTANT_RULES>

<PIECES>
Uppercase=ORANGE
Lowercase=BLACK
W=Warrior
A=Padwar
D=Dwar
F=Flier
C=Chief
R=Princess
T=Thoat
P=Panthan
.=empty
</PIECES>

<BOARD>
{board}
</BOARD>

<STATE_FEATURES>
ORANGE Princess attacked={"yes" if orange_princess_attacked else "no"}
BLACK Princess attacked={"yes" if black_princess_attacked else "no"}
ORANGE Princess escape={orange_escape}
BLACK Princess escape={black_escape}
</STATE_FEATURES>

<IMMEDIATE_WINNING_MOVES>
{", ".join(immediate_winning_moves) if immediate_winning_moves else "none"}
</IMMEDIATE_WINNING_MOVES>

<STRATEGIC_PRIORITIES>
1. If IMMEDIATE_WINNING_MOVES is not "none", choose one of those moves.
2. Protect your Princess from immediate danger.
3. Prefer favorable captures and strong threats.
4. Preserve valuable pieces.
5. Prefer useful mobility and activity when no stronger tactical move exists.
</STRATEGIC_PRIORITIES>

<RECENT_MOVES oldest_first="true" limit="{len(recent_move_strings)}">
{", ".join(recent_move_strings) if recent_move_strings else "none"}
</RECENT_MOVES>

<LEGAL_MOVES count="{len(legal_move_strings)}">
{", ".join(legal_move_strings)}
</LEGAL_MOVES>

Return exactly:
{{"move": "<one token copied verbatim from LEGAL_MOVES>"}}
""".strip()

    return (
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    )


def parse_move_response(response: str, actions: tuple[Move, ...]) -> Move:
    if "\n" in response or "\r" in response:
        raise ValueError("response must be one line")

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        data = json.loads(response, object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid move response") from error

    if not isinstance(data, dict):
        raise ValueError("response must be a JSON object")

    if set(data.keys()) != {"move"}:
        raise ValueError("response must contain exactly the key 'move'")

    token = data["move"]

    if not isinstance(token, str):
        raise ValueError("move must be a JSON string")

    moves_by_token = {str(action): action for action in actions}

    if token not in moves_by_token:
        raise ValueError("move must be one of the supplied legal actions")

    return moves_by_token[token]


def choose_fallback(
    board: JetanBoard,
    actions: tuple[Move, ...],
) -> Move:
    if not actions:
        raise ValueError("fallback requires at least one legal action")

    mover = board.player()

    best_action = actions[0]
    best_score = None

    for action in actions:
        successor = board.result(action)

        if successor.is_terminal():
            score = float(successor.utility(mover))
        else:
            score = evaluate_position_3(
                successor,
                mover,
            )

        if best_score is None or score > best_score:
            best_score = score
            best_action = action

    return best_action
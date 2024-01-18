import json

import pandas as pd

from typing import Dict, List, Any

from .permute import reverse_perm, perm_to_swap, permute_with_swap


def diff_between_states(state1: List[Any], state2: List[Any]) -> int:
    assert len(state1) == len(state2)

    diff_count = 0
    for a, b in zip(state1, state2):
        if a != b:
            diff_count += 1

    return diff_count


def is_valid_final_state(
    state: List[Any], final_state: List[Any], num_wild: int
) -> bool:
    assert len(state) == len(final_state)
    diff_count = diff_between_states(state, final_state)
    return diff_count <= num_wild


def remove_prefix(s: str, prefix="-") -> str:
    return s[s.startswith(prefix) and len(prefix) :]


def get_inverse_move(move_name: str) -> str:
    """returns the inverted move name"""
    if move_name.startswith("-"):
        return move_name[1:]
    else:
        return f"-{move_name}"


def generate_state_from_moves(
    moves: List[str],
    move_dict: Dict[str, List],
    state: List[int],
    inverse: bool = False,
) -> List[int]:
    for move_name in moves:
        if inverse:
            move_name = get_inverse_move(move_name)
        move = move_dict[move_name]
        state = permute_with_swap(state, move)

    return state


def load_puzzle(
    puzzle_name: str, convert_to_swaps=True, puzzle_dir="../../puzzles"
) -> (Dict[str, List[int]], int):
    """Retrieves and returns the moves and final position of the puzzle"""
    # load the moves:
    with open(f"{puzzle_dir}/{puzzle_name}/moves.json") as f:
        moves = json.load(f)

    # add reversed moves
    reversed_moves = {}
    for move_name, perm in moves.items():
        reversed_perm = reverse_perm(perm)
        if reversed_perm == perm:
            continue
        reversed_moves[f"-{move_name}"] = reversed_perm

    moves.update(reversed_moves)

    if convert_to_swaps:
        for move_name, perm in moves.items():
            moves[move_name] = perm_to_swap(perm)

    # get final position (from the first puzzle)
    df = pd.read_csv(f"puzzles/{puzzle_name}/puzzles.csv")
    state = df.iloc[0].to_numpy()[3].split(";")
    c = -1
    mapping = {}
    for i, s in enumerate(state):
        if s not in mapping.keys():
            c += 1
            mapping[s] = c
        state[i] = mapping[s]

    return moves, state

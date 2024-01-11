from typing import List, Any


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

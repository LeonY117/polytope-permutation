from typing import List, Tuple, Any

# TODO: use jit / jax to the accelerate permutation operations


def permute(state: List[Any], perm: List[int]) -> List[Any]:
    """performs permutation on state and returns new state"""
    assert len(state) == len(perm)
    new_state = state[:]
    for i, j in enumerate(perm):
        new_state[i] = state[j]
    return new_state


def permute_with_swap(state: List[Any], swaps: List[Tuple[int, int]]) -> List[Any]:
    """performs permutation with swapping operations"""
    new_state = state[:]
    for i, j in swaps:
        new_state[i] = state[j]
    return new_state


def reverse_perm(perm: List[int]) -> List[int]:
    """computes the reversed permutation"""
    out = perm[:]
    for i, j in enumerate(perm):
        out[j] = i
    return out


def perm_to_swap(perm: List[int]) -> List[Tuple[int, int]]:
    """changes a permutation arr to a list of swaps"""
    swap = []
    for i, j in enumerate(perm):
        if i != j:
            swap.append((i, j))

    return swap

from collections import Counter
from functools import partial
from .utils import get_inverse_move, remove_prefix

__all__ = ["iterate_reduce_sequence"]


def _reduce_commutative(subsequence, n):
    """Reduce a subsequence of commutative moves given cycle length n"""
    # TODO: we don't have to sort, it can be O(n) with stacks.
    subsequence.sort()
    out = []
    curr_count, curr_move = 0, ""
    # reduce 3 or 4 in a row:
    for move in subsequence + ["x"]:
        # a dummy item 'x' to make loop run once more for leftover items
        if move != curr_move:
            if curr_count == n - 1:
                out.append(get_inverse_move(curr_move))
            else:
                out.extend([curr_move] * curr_count)
            curr_count, curr_move = 1, move
        elif move == curr_move:
            curr_count += 1
        if curr_count == n:
            # skip entirely, don't append to out
            curr_count = 0

    counts = Counter(out)

    # remove pairs
    out = []
    for move, count in counts.items():
        inv_move = get_inverse_move(move)
        inv_count = counts.get(inv_move, 0)
        if count > inv_count:
            out.extend([move] * (count - inv_count))
        else:
            out.extend([inv_move] * (inv_count - count))
        if inv_move in counts.keys():
            counts[inv_move] = 0
        counts[move] = 0
    return out


def _reduce_wreath(subsequence):
    """For wreath we can only cancel consecutive, opposite pairs"""
    # we use a simple stack to do this, can be done in one paths
    out = []
    prev = None
    for move in subsequence:
        prev = out[-1] if out else None
        if prev and get_inverse_move(prev) == move:
            out.pop()
        else:
            out.append(move)

    return out


def _reduce_globe(subsequence, n):
    """
    Reduce subsequence for globes
    r moves are commutative, f moves are not, but fx = -fx
    """
    if not subsequence:
        return []
    move_type = remove_prefix(subsequence[0], "-")[0]
    if move_type == "r":
        out = _reduce_commutative(subsequence, n)
    elif move_type == "f":
        out = []
        prev = None
        for move in subsequence:
            prev = out[-1] if out else None
            if prev and remove_prefix(prev, "-") == move:
                out.pop()
            else:
                out.append(move)

    return out


def _reduce_cube(subsequence):
    return _reduce_commutative(subsequence, n=4)


_reduce_subsequence = {
    "cube": _reduce_cube,
    "wreath": _reduce_wreath,
    "globe": _reduce_globe,
}


def iterate_reduction(sequence, method):
    """A generic template for applying method iteratively to a sequence"""
    prev = []
    curr = sequence

    while prev != curr:
        prev = curr[:]
        curr = method(curr)

    return curr


def _iterate_subsequence_reduction(subsequence, puzzle):
    """Iteratively apply the reduction method to the subsequence until its minimal"""
    if len(subsequence) <= 1:
        return subsequence

    if puzzle.startswith("globe"):
        assert len(puzzle.split("/")) > 1(
            "Globe puzzles require full puzzle name, e.g. globe_1/8"
        )
        n = int(puzzle.split("/")[1])
        reduction_method = partial(_reduce_subsequence["globe"], n=n * 2)
    elif puzzle.startswith("cube"):
        reduction_method = _reduce_subsequence["cube"]
    elif puzzle.startswith("wreath"):
        reduction_method = _reduce_subsequence["wreath"]
    else:
        print("invalid puzzle name, make sure it's either cube, wreath or globe")
        raise ValueError()

    return iterate_reduction(subsequence, reduction_method)


def _reduce_sequence(sequence, puzzle):
    """Finds subsequence of the same face and iteratively reduce each subsequence"""
    out = []
    curr_face = ""
    curr_subsequence = []
    for move in sequence + ["x"]:
        face = remove_prefix(move, "-")[0]
        if face != curr_face:
            out.extend(_iterate_subsequence_reduction(curr_subsequence, puzzle))
            curr_subsequence = []
        curr_face = face
        curr_subsequence.append(move)

    return out


def iterate_reduce_sequence(sequence, puzzle_name: str):
    """Applies greedy reduction to a sequence for a particular puzzle"""
    return iterate_reduction(sequence, partial(_reduce_sequence, puzzle=puzzle_name))

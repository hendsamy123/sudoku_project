"""NN reads the grid -> AC-3 finds a conflict -> Search tries the cheapest corrections -> CSP tests each one."""
from dataclasses import dataclass, field

import numpy as np

from csp.sudoku_csp import conflicts, solve, cell_name
from search.correction import ALGORITHMS, CorrectionProblem

BLANK_MEAN_THRESHOLD = 0.01   # blank-cell rule: mean pixel intensity below 1% (NOT decided by the NN)


def least_confident_cells(read_grid, probs, k=5):
    """The k filled cells where the network's top probability is lowest: [(cell, digit, p)]."""
    filled = [i for i, v in enumerate(read_grid) if v]
    order = sorted(filled, key=lambda i: probs[i, read_grid[i] - 1])[:k]
    return [(i, read_grid[i], float(probs[i, read_grid[i] - 1])) for i in order]


def read_cells(model, cells):
    """cells: (81,28,28) images in [0,1]. Returns (read_grid list[81], probs (81,9))."""
    flat = cells.reshape(81, -1)
    probs = model.predict_proba(flat)
    read = [0 if flat[i].mean() < BLANK_MEAN_THRESHOLD else int(probs[i].argmax()) + 1 for i in range(81)]
    return read, probs


@dataclass
class PipelineResult:
    status: str                      # solved_directly | solved_after_correction | unreadable | not_solved
    read_grid: list
    conflicts: list = field(default_factory=list)
    first_csp_message: str = ""
    corrections: list = field(default_factory=list)
    solution: list = None
    corrected_grid: list = None
    search: object = None
    least_confident: list = field(default_factory=list)
    message: str = ""


def solve_reading(read_grid, probs, algorithm="ucs", use_nn=True, max_corrections=3, n_uncertain=12, n_alts=2,
                  require_unique=False, node_limit=20000):
    cf = conflicts(read_grid)
    first = solve(read_grid, use_fc=True, use_ac3=True, order="mrv")
    res = PipelineResult("not_solved", list(read_grid), cf, first.message)
    if first.solved and (not require_unique or len(solve(read_grid, True, True, "mrv", 2).solutions) == 1):
        res.status, res.solution, res.corrected_grid, res.message = "solved_directly", first.solution, list(read_grid), \
            "The first reading is already solvable - no correction needed."
        return res
    if not use_nn:
        res.least_confident = least_confident_cells(read_grid, probs)
        res.message = ("NO SOLUTION: puzzle unreadable (the first reading is not solvable and corrections are switched off). "
                       + first.message + " Least confident cells: " +
                       ", ".join(f"{cell_name(i)} read {d} @ {p:.2f}" for i, d, p in res.least_confident))
        return res
    prob = CorrectionProblem(read_grid, probs, max_corrections, n_uncertain, n_alts, require_unique)
    s = ALGORITHMS[algorithm](prob, node_limit=node_limit)
    res.search = s
    if s.status == "solved":
        res.status, res.corrections, res.solution, res.corrected_grid = "solved_after_correction", s.corrections, s.solution, s.grid
        res.message = f"Solved after {len(s.corrections)} correction(s), total cost {s.total_cost:.3f}."
    else:
        res.status = "unreadable"
        res.least_confident = least_confident_cells(read_grid, probs)
        res.message = ("Unreadable: no solvable puzzle within "
                       f"{max_corrections} corrections. Least confident cells: " +
                       ", ".join(f"{cell_name(i)}={d} (p={p:.2f})" for i, d, p in res.least_confident))
    return res

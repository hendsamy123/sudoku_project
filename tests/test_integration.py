import numpy as np
from integration.pipeline import solve_reading
from tests.test_search import PUZ, misread, make_probs


def test_normal_pipeline_corrects_misread_and_recovers_true_solution():
    from csp.sudoku_csp import solve
    truth = solve(PUZ).solution
    read, probs = misread({0: 3, 31: 1})
    res = solve_reading(read, probs, "ucs")
    assert res.status == "solved_after_correction" and res.solution == truth and len(res.corrections) == 2


def test_normal_pipeline_nn_off_stays_unsolved():
    read, probs = misread({0: 3})
    res = solve_reading(read, probs, use_nn=False)
    assert res.status == "not_solved" and res.conflicts and "No solution" in res.message


def test_edge_pipeline_no_correction_when_reading_is_fine():
    res = solve_reading(PUZ, make_probs(PUZ, {}))
    assert res.status == "solved_directly" and res.corrections == []


def test_impossible_pipeline_reports_unreadable_with_least_confident_cells():
    read, probs = misread({0: 3, 9: 1, 19: 1, 27: 6})   # 4 misreads, budget is 3
    res = solve_reading(read, probs, "astar")
    assert res.status == "unreadable" and res.message.startswith("Unreadable") and len(res.least_confident) == 5

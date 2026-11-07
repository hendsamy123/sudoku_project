import math
import numpy as np
import pytest
from csp.sudoku_csp import solve
from search.correction import CorrectionProblem, ucs, greedy, astar, brute_force

PUZ = [int(c) for c in "530070000600195000098000060800060003400803001700020006060000280000419005000080079"]


def make_probs(read, wrong, base=0.9):
    """Synthetic network output: confident everywhere except `wrong` = {cell: (read_digit, p_read, true_digit, p_true)}."""
    probs = np.full((81, 9), (1 - base) / 8)
    for i, v in enumerate(read):
        if v:
            probs[i, v - 1] = base
    for i, (rd, pr, td, pt) in wrong.items():
        probs[i] = (1 - pr - pt) / 7
        probs[i, rd - 1], probs[i, td - 1] = pr, pt
    return probs


def misread(cells_to_wrong):
    read = list(PUZ)
    wrong = {}
    for i, new in cells_to_wrong.items():
        true = PUZ[i]
        read[i] = new
        wrong[i] = (new, 0.52, true, 0.45)
    return read, make_probs(read, wrong)


def test_normal_spec_example_7_read_as_1():
    # cell 0 holds 5; the network reads it as 3 -> duplicate 3 in row 1 (spec example, 52% vs 45%)
    read, probs = misread({0: 3})
    for algo in (ucs, greedy, astar):
        r = algo(CorrectionProblem(read, probs))
        assert r.status == "solved" and r.corrections[0]["cell"] == 0 and r.corrections[0]["new"] == 5
        assert r.total_cost == pytest.approx(math.log(0.52) - math.log(0.45))


def test_normal_cost_is_never_negative():
    read, probs = misread({0: 3, 80: 1})
    p = CorrectionProblem(read, probs)
    assert all(c >= 0 for opts in p.options.values() for _, c in opts)


def test_normal_ucs_and_astar_match_brute_force_and_greedy_is_not_better():
    read, probs = misread({0: 3, 9: 1})
    p = CorrectionProblem(read, probs, max_corrections=2)
    best_cost, _ = brute_force(p)
    u, a, g = ucs(CorrectionProblem(read, probs, max_corrections=2)), astar(CorrectionProblem(read, probs, max_corrections=2)), \
        greedy(CorrectionProblem(read, probs, max_corrections=2))
    assert u.total_cost == pytest.approx(best_cost) and a.total_cost == pytest.approx(best_cost)
    assert g.total_cost >= best_cost - 1e-9


def test_normal_astar_expands_no_more_than_ucs():
    read, probs = misread({0: 3, 9: 1})
    u = ucs(CorrectionProblem(read, probs)); a = astar(CorrectionProblem(read, probs))
    assert a.expanded <= u.expanded


def test_normal_heuristic_is_admissible_on_root():
    read, probs = misread({0: 3, 9: 1})
    p = CorrectionProblem(read, probs)
    assert p.h(()) <= ucs(CorrectionProblem(read, probs)).total_cost + 1e-9


def test_edge_clean_reading_needs_zero_corrections():
    probs = make_probs(PUZ, {})
    for algo in (ucs, greedy, astar):
        r = algo(CorrectionProblem(PUZ, probs))
        assert r.status == "solved" and r.corrections == [] and r.total_cost == 0 and r.expanded == 1


def test_edge_budget_too_small_reports_unreadable():
    read, probs = misread({0: 3, 9: 1})
    r = ucs(CorrectionProblem(read, probs, max_corrections=1))
    assert r.status == "unreadable" and "no solvable" in r.message


def test_impossible_zero_budget_with_conflict_gives_clear_message():
    read, probs = misread({0: 3})                      # duplicate 3 in row 1, but no corrections allowed
    for algo in (ucs, greedy, astar):
        r = algo(CorrectionProblem(read, probs, max_corrections=0))
        assert r.status == "unreadable" and "no solvable" in r.message and r.total_cost == math.inf

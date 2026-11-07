import pytest
from csp.sudoku_csp import solve, is_valid_solution, count_solutions, conflicts, ac3, ALL

PUZ = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
CONFIGS = [dict(), dict(use_fc=True), dict(use_fc=True, use_ac3=True), dict(use_fc=True, use_ac3=True, order="mrv")]


# ---- 5 normal ---------------------------------------------------------------
@pytest.mark.parametrize("kw", CONFIGS[:3])
def test_normal_all_switchable_variants_solve(kw):
    r = solve(PUZ, **kw)
    assert r.solved and is_valid_solution(r.solution)
    assert all(a == 0 or a == b for a, b in zip([int(c) for c in PUZ], r.solution))  # givens respected


def test_normal_fc_prunes_search():
    bt, fc = solve(PUZ), solve(PUZ, use_fc=True)
    assert fc.stats.assignments < bt.stats.assignments and fc.stats.fc_prunings > 0


def test_normal_ac3_removes_values_and_counts():
    r = solve(PUZ, use_fc=True, use_ac3=True)
    assert r.stats.ac3_removals > 0 and r.stats.ac3_revisions > 0 and r.stats.backtracks == 0


# ---- 3 edge -----------------------------------------------------------------
def test_edge_complete_grid_is_returned_unchanged():
    full = "".join(map(str, solve(PUZ).solution))
    r = solve(full)
    assert r.solved and r.stats.assignments == 0


def test_edge_empty_grid_has_solution():
    r = solve("0" * 81, use_fc=True, use_ac3=True, order="mrv")
    assert r.solved and is_valid_solution(r.solution)


def test_edge_uniqueness_counter_detects_multiple_solutions():
    assert count_solutions("0" * 81, limit=2) == 2
    assert count_solutions(PUZ, limit=2) == 1


# ---- 2 impossible -----------------------------------------------------------
def test_impossible_duplicate_givens_all_variants_report_no_solution():
    bad = "11" + PUZ[2:]
    for kw in CONFIGS:
        r = solve(bad, **kw)
        assert r.status == "no_solution" and r.message.startswith("No solution")
    assert "AC-3" in solve(bad, use_ac3=True).message


def test_impossible_consistent_but_unsolvable():
    # row 1 holds 1-8, but column 9 already contains a 9 -> cell (1,9) has an empty domain, no duplicate givens
    g = list("123456780") + ["0"] * 72
    g[9 * 1 + 8] = "9"
    r = solve("".join(g), use_fc=True, use_ac3=True)
    assert not conflicts("".join(g)) and r.status == "no_solution" and "No solution" in r.message
    assert solve("".join(g)).status == "no_solution"

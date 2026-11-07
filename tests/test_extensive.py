"""Heavy / randomised tests. Independent verifiers are written here on purpose (not imported from the code under test)."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from csp.sudoku_csp import solve, count_solutions, ac3, ALL, conflicts, parse
from data.puzzles import make_puzzle, random_full_grid, make_cell_images, build_pools, mosaic
from integration.pipeline import solve_reading, read_cells
from search.correction import CorrectionProblem, ucs, greedy, astar, brute_force
from tests.test_search import PUZ, make_probs

ROOT = Path(__file__).resolve().parents[1]


def independent_valid(sol, puzzle=None):
    """Plain re-implementation of the Sudoku rules."""
    if len(sol) != 81 or any(v not in range(1, 10) for v in sol):
        return False
    for k in range(9):
        if sorted(sol[k * 9:(k + 1) * 9]) != list(range(1, 10)): return False
        if sorted(sol[k::9]) != list(range(1, 10)): return False
        br, bc = (k // 3) * 3, (k % 3) * 3
        if sorted(sol[(br + r) * 9 + bc + c] for r in range(3) for c in range(3)) != list(range(1, 10)): return False
    return puzzle is None or all(p == 0 or p == s for p, s in zip(puzzle, sol))


# =============================== CSP (randomised) ===============================
@pytest.fixture(scope="module")
def puzzles():
    rng = np.random.default_rng(2024)
    return [make_puzzle(rng, g) for g in (35, 35, 30, 30, 30, 28, 28, 26, 25, 25) for _ in range(3)]


def test_every_variant_gives_a_valid_and_identical_solution(puzzles):
    for puz, sol in puzzles:
        for kw in (dict(), dict(use_fc=True), dict(use_fc=True, use_ac3=True), dict(use_ac3=True),
                   dict(use_fc=True, use_ac3=True, order="mrv"), dict(order="mrv")):
            r = solve(puz, node_limit=2_000_000, **kw)
            assert r.solved and independent_valid(r.solution, puz) and r.solution == sol, kw


def test_pruning_never_increases_search_with_same_variable_order(puzzles):
    for puz, _ in puzzles:
        bt = solve(puz, node_limit=2_000_000).stats.assignments
        fc = solve(puz, use_fc=True).stats.assignments
        fa = solve(puz, use_fc=True, use_ac3=True).stats.assignments
        assert fc <= bt and fa <= fc


def test_ac3_is_sound_never_removes_the_true_value(puzzles):
    for puz, sol in puzzles:
        dom = [(1 << v) if v else ALL for v in puz]
        ok, _ = ac3(dom)
        assert ok and all(dom[i] & (1 << sol[i]) for i in range(81))


def test_generated_puzzles_are_unique_with_exact_givens(puzzles):
    for puz, sol in puzzles:
        assert count_solutions(puz, 2) == 1 and independent_valid(sol, puz)
    assert sorted({sum(1 for v in p if v) for p, _ in puzzles}) == [25, 26, 28, 30, 35]


def test_solve_does_not_mutate_its_input(puzzles):
    puz, _ = puzzles[0]
    before = list(puz)
    solve(puz, use_fc=True, use_ac3=True)
    assert puz == before


def test_random_corruptions_are_reported_as_no_solution_consistently(puzzles):
    rng = np.random.default_rng(5)
    for puz, sol in puzzles[:15]:
        g = list(puz)
        filled = [i for i, v in enumerate(g) if v]
        i = int(rng.choice(filled)); j = int(rng.choice([k for k in filled if k != i]))
        g[i] = g[j]   # may or may not conflict; compare against brute truth below
        results = [solve(g, node_limit=300_000, **kw) for kw in (dict(use_fc=True), dict(use_fc=True, use_ac3=True), dict(use_fc=True, use_ac3=True, order="mrv"))]
        statuses = {r.status for r in results}
        assert len(statuses - {"limit"}) <= 1            # all variants that finish agree
        for r in results:
            if r.status == "no_solution":
                assert r.message.startswith("No solution")
            if r.status == "solved":
                assert independent_valid(r.solution, g)


def test_full_grid_generator_valid_and_seeded():
    a, b = random_full_grid(np.random.default_rng(1)), random_full_grid(np.random.default_rng(1))
    c = random_full_grid(np.random.default_rng(2))
    assert independent_valid(a) and a == b and a != c


def test_puzzle_generation_is_reproducible_with_seed():
    p1 = make_puzzle(np.random.default_rng(77), 30)
    p2 = make_puzzle(np.random.default_rng(77), 30)
    assert p1 == p2


# =============================== SEARCH (randomised vs brute force) ===============================
def random_instance(rng, n_wrong):
    filled = [i for i, v in enumerate(PUZ) if v]
    cells = rng.choice(filled, size=n_wrong, replace=False)
    read, wrong = list(PUZ), {}
    for i in cells:
        new = int(rng.choice([d for d in range(1, 10) if d != PUZ[i]]))
        pr = float(rng.uniform(0.40, 0.75)); pt = float(rng.uniform(0.05, min(0.24, 0.98 - pr)))
        read[i] = new
        wrong[int(i)] = (new, pr, PUZ[i], pt)
    return read, make_probs(read, wrong)


@pytest.mark.parametrize("seed", range(12))
def test_ucs_and_astar_equal_brute_force_greedy_not_better(seed):
    rng = np.random.default_rng(seed)
    read, probs = random_instance(rng, int(rng.integers(1, 4)))
    mk = lambda: CorrectionProblem(read, probs, max_corrections=3, n_uncertain=8)
    best_cost, _ = brute_force(mk())
    u, a, g = ucs(mk()), astar(mk()), greedy(mk())
    for r in (u, a, g):
        assert r.status == "solved" and independent_valid(r.solution)
        assert all(c["cost"] >= 0 for c in r.corrections) and len(r.corrections) <= 3
        # returned solution must respect every non-corrected read digit
        grid = list(read)
        for c in r.corrections: grid[c["cell"]] = c["new"]
        assert independent_valid(r.solution, grid)
        assert r.total_cost == pytest.approx(sum(c["cost"] for c in r.corrections))
    assert u.total_cost == pytest.approx(best_cost) and a.total_cost == pytest.approx(best_cost)
    assert g.total_cost >= best_cost - 1e-9
    assert a.expanded <= u.expanded + 1   # informed search should not do worse than blind


@pytest.mark.parametrize("seed", range(12))
def test_heuristic_is_admissible_along_the_optimal_path(seed):
    rng = np.random.default_rng(100 + seed)
    read, probs = random_instance(rng, int(rng.integers(1, 4)))
    p = CorrectionProblem(read, probs, n_uncertain=8)
    r = ucs(CorrectionProblem(read, probs, n_uncertain=8))
    corr = sorted((c["cell"], c["new"]) for c in r.corrections)
    for k in range(len(corr) + 1):
        state = tuple(corr[:k])
        g = sum(p.cost_of(c, d) for c, d in state)
        assert p.h(state) <= r.total_cost - g + 1e-9      # h never overestimates cost-to-go


def test_zero_misreads_never_triggers_corrections_in_any_algorithm():
    probs = make_probs(PUZ, {})
    for algo in (ucs, greedy, astar):
        r = algo(CorrectionProblem(PUZ, probs))
        assert r.total_cost == 0 and r.corrections == [] and r.csp_calls == 1


def test_search_counters_are_consistent():
    read, probs = random_instance(np.random.default_rng(9), 2)
    for algo in (ucs, greedy, astar):
        r = algo(CorrectionProblem(read, probs))
        assert 1 <= r.expanded <= r.generated and r.csp_calls <= r.expanded and len(r.trace) <= r.expanded


def test_require_unique_goal_returns_a_unique_puzzle():
    read, probs = random_instance(np.random.default_rng(3), 1)
    r = ucs(CorrectionProblem(read, probs, require_unique=True))
    assert r.status == "solved" and count_solutions(r.grid, 2) == 1


# =============================== PIPELINE FUZZ ===============================
def test_pipeline_fuzz_invariants():
    rng = np.random.default_rng(31337)
    seen = set()
    for _ in range(60):
        read, probs = random_instance(rng, int(rng.integers(0, 6)))
        algo = ["ucs", "greedy", "astar"][int(rng.integers(0, 3))]
        res = solve_reading(read, probs, algo)
        seen.add(res.status)
        assert res.status in {"solved_directly", "solved_after_correction", "unreadable"}
        if res.status.startswith("solved"):
            assert independent_valid(res.solution, res.corrected_grid) and len(res.corrections) <= 3
        else:
            assert res.message.startswith("Unreadable") and 1 <= len(res.least_confident) <= 5
            assert res.least_confident == sorted(res.least_confident, key=lambda t: t[2])
        off = solve_reading(read, probs, algo, use_nn=False)
        assert off.status in {"solved_directly", "not_solved"}
    assert {"solved_after_correction", "unreadable"} <= seen


# =============================== REAL DATA + REAL NETWORK ===============================
@pytest.fixture(scope="module")
def real():
    from data.make_data import load_data
    from nn.model import MLP
    d = load_data(); net = MLP.load(ROOT / "models" / "mlp.npz")
    clean, hard = build_pools(net, d["x_test"], d["y_test"])
    return d, net, clean, hard


def test_dataset_integrity(real):
    d = real[0]
    assert (len(d["y_train"]), len(d["y_val"]), len(d["y_test"])) == (44077, 10000, 9020)
    for k in ("train", "val", "test"):
        assert d["x_" + k].shape[1] == 784 and d["x_" + k].min() >= 0 and d["x_" + k].max() <= 1
        assert set(np.unique(d["y_" + k])) == set(range(1, 10))      # digit 0 dropped
    h = lambda x: {hashlib.md5(r.tobytes()).hexdigest() for r in x}
    tr, va, te = h(d["x_train"]), h(d["x_val"]), h(d["x_test"])
    assert not (tr & va) and not (tr & te) and not (va & te)        # no leakage between splits


def test_network_quality_and_beats_baseline(real):
    d, net, _, _ = real
    m = json.loads((ROOT / "results" / "nn_metrics.json").read_text())
    p = net.predict_proba(d["x_test"])
    acc = float((p.argmax(1) + 1 == d["y_test"]).mean())
    assert acc >= 0.97 and abs(acc - m["mlp_temp_scaled"]["accuracy"]) < 1e-9
    assert m["mlp_temp_scaled"]["accuracy"] > m["logistic_regression"]["accuracy"] + 0.04
    assert m["mlp_temp_scaled"]["ece"] < 0.02 and np.allclose(p.sum(1), 1, atol=1e-5)
    assert m["mlp_temp_scaled"]["nll"] <= m["mlp_uncalibrated"]["nll"]


def test_mosaic_blank_rule_and_shapes(real):
    d, net, clean, hard = real
    rng = np.random.default_rng(4)
    puz, sol = make_puzzle(rng, 30)
    cells, bad = make_cell_images(puz, d["x_test"], clean, hard, rng, 2)
    assert cells.shape == (81, 28, 28) and mosaic(cells).shape == (252, 252) and len(bad) == 2
    read, probs = read_cells(net, cells)
    assert [v == 0 for v in read] == [v == 0 for v in puz]                    # blank rule is exact
    assert all(read[i] != puz[i] for i in bad)                                # injected cells ARE misread
    assert all(read[i] == puz[i] for i in range(81) if puz[i] and i not in bad)   # all other cells read correctly
    assert np.allclose(probs.sum(1), 1, atol=1e-5)


def test_end_to_end_recovery_on_real_digits(real):
    d, net, clean, hard = real
    rec = {0: 0, 1: 0, 2: 0}
    N = 15
    for t in range(N):
        rng = np.random.default_rng(900 + t)
        puz, sol = make_puzzle(rng, 30)
        for m in rec:
            cells, bad = make_cell_images(puz, d["x_test"], clean, hard, rng, m)
            read, probs = read_cells(net, cells)
            res = solve_reading(read, probs, "astar")
            assert res.status != "not_solved"
            if res.status.startswith("solved") and res.solution == sol:
                rec[m] += 1
    assert rec[0] == N                         # no misread -> always exact
    assert rec[1] >= 0.6 * N and rec[2] >= 0.4 * N   # NN-guided correction clearly works

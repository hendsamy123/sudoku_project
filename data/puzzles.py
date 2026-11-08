"""Seeded generator of Sudoku puzzles and 9x9 handwritten-digit mosaics (GENERATED data, not real photos).

* full grid: randomised backtracking
* puzzle: remove cells in random order while the solution stays UNIQUE, until `n_givens` remain
* mosaic: every given cell gets a 28x28 MNIST *test* image of that digit; blank cells are all-zero images
* clean images  = test images the network classifies correctly (so the only misreads are the injected ones)
* hard images   = test images the network misclassifies but whose true digit is in its top-3 (documented rule)
"""
import numpy as np

from csp.sudoku_csp import PEERS, count_solutions


def random_full_grid(rng):
    grid = [0] * 81

    def rec():
        best, best_c = -1, None
        for i in range(81):
            if grid[i] == 0:
                c = [d for d in range(1, 10) if all(grid[p] != d for p in PEERS[i])]
                if best < 0 or len(c) < len(best_c):
                    best, best_c = i, c
                    if len(c) <= 1:
                        break
        if best < 0:
            return True
        for d in rng.permutation(best_c):
            grid[best] = int(d)
            if rec():
                return True
        grid[best] = 0
        return False

    rec()
    return grid


def make_puzzle(rng, n_givens, max_tries=60):
    """Returns (puzzle, solution) with exactly `n_givens` givens and a unique solution."""
    for _ in range(max_tries):
        sol = random_full_grid(rng)
        puz = list(sol)
        left = 81
        for c in rng.permutation(81):
            if left <= n_givens:
                break
            keep = puz[c]
            puz[c] = 0
            if count_solutions(puz, 2) != 1:
                puz[c] = keep
            else:
                left -= 1
        if left == n_givens:
            return puz, sol
    raise RuntimeError(f"could not reach {n_givens} givens with a unique solution")


def build_pools(model, x, y):
    """x: (N,784) float images, y: labels 1..9. Returns clean/hard index pools per digit."""
    p = model.predict_proba(x)
    pred = p.argmax(1) + 1
    rank_of_true = (p > p[np.arange(len(y)), y - 1][:, None]).sum(1)  # 0 = top-1
    clean = {d: np.where((y == d) & (pred == d))[0] for d in range(1, 10)}
    hard = {d: np.where((y == d) & (pred != d) & (rank_of_true <= 2))[0] for d in range(1, 10)}
    return clean, hard


def make_cell_images(puzzle, x, clean, hard, rng, n_misreads=0):
    """Returns (cells (81,28,28) float32, misread_cells list). Blank cells stay all-zero."""
    cells = np.zeros((81, 28, 28), np.float32)
    filled = [i for i, v in enumerate(puzzle) if v]
    can_inject = [i for i in filled if len(hard[puzzle[i]]) > 0]
    n_misreads = min(n_misreads, len(can_inject))
    bad = list(rng.choice(can_inject, size=n_misreads, replace=False)) if n_misreads else []
    for i in filled:
        pool = hard[puzzle[i]] if i in bad else clean[puzzle[i]]
        cells[i] = x[int(rng.choice(pool))].reshape(28, 28)
    return cells, sorted(int(b) for b in bad)


def mosaic(cells):
    """(81,28,28) -> (252,252) picture."""
    return cells.reshape(9, 9, 28, 28).transpose(0, 2, 1, 3).reshape(252, 252)

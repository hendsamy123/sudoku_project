# P7 – Handwritten Sudoku Reader and Solver (NN + Search + CSP)

A neural network reads a 9×9 mosaic of handwritten digits. If a misread digit breaks the puzzle, **Uniform Cost Search / Greedy / A\*** look for the
cheapest set of corrections (cost = how much less likely the new digit is for the network) and a **CSP solver** (Backtracking, Forward Checking, AC-3 –
separately switchable) tests every corrected puzzle. First solvable puzzle wins; if none exists the system answers "unreadable" and lists the least confident cells.

## Layout
| Folder | Content |
|---|---|
| `data/` | `make_data.py` (download + preprocess, seed 7), `puzzles.py` (puzzle + mosaic generator), `DATA_DICTIONARY.md` |
| `nn/` | `model.py` (NumPy MLP 784-128-64-9, Adam, temperature scaling, ECE), `train.py` (vs logistic regression) |
| `csp/` | `sudoku_csp.py` (BT / FC / AC-3 switches + counters) |
| `search/` | `correction.py` (UCS, Greedy, A\*, admissible heuristic, brute-force reference) |
| `integration/` | `pipeline.py` (NN → AC-3 → Search → CSP), `viz.py` |
| `eval/` | `experiments.py` (the three experiments, tables, plots) |
| `tests/` | 65 pytest tests: normal / edge / impossible, randomised vs brute force, admissibility, fuzzing, real-data checks |
| `check_requirements.py` | one-shot verification of the assignment checklist |
| `results/` | tables (`exp*_table.md`, raw CSVs), plots, `nn_metrics.json`, `demo.png` |

## Setup (Python 3.12)
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Exact run commands (in this order)
```
python -m data.make_data          # downloads MNIST (needs internet once), builds data/mnist_1to9.npz
python -m nn.train                # trains + calibrates the MLP, compares with logistic regression  (~30 s)
python -m pytest -q               # 65 tests (basic + randomised/fuzz + real-data tests)
python check_requirements.py      # checks every deliverable of the assignment (PASS/FAIL/TODO)
python run_demo.py                # integrated demo (seed 1); add --seed N --misreads M --givens G
python -m eval.experiments        # all 3 experiments, 5 seeds (~1.5 min); use --quick for a smoke test
```

## What `run_demo.py` prints
Sections, in order: ground-truth puzzle (`*` = digit the network will misread) -> first reading with top-3 probabilities of every misread cell ->
pipeline (AC-3 check, search trace with g and h, search counters, CSP counters, result, correction log, solution, "matches the writer's original solution")
-> NN-guided correction OFF -> UCS vs Greedy vs A\* -> BT vs BT+FC vs BT+FC+AC-3. A figure is saved to `results/demo.png`
(`--out FILE`, `--no-plot`, `--show`; try `--seed 4 --misreads 3 --givens 35`; `--misreads 4` shows the "unreadable" answer).

Counter meanings: `generated` / `expanded` = search nodes; `goal_tests` = goal tests (one per expanded node); `csp_solves` = distinct CSP runs
(a corrected grid is solved once, then cached); `pruned` = children dropped because h = inf; `max_frontier` = largest queue;
`ac3_prunings` = values AC-3 removed; `fc_prunings` = values forward checking removed.

## Design summary
* **Network** – inputs 28×28 → probabilities for digits 1–9. Temperature scaling is fitted on the validation set (does not change the arg-max).
* **CSP** – 81 variables, domains as bit-masks, `!=` between peers. `use_fc`, `use_ac3`, `order` ("static" or "mrv") are independent switches.
  Counters: assignments, backtracks, AC-3 revisions/removals, FC prunings. Impossible puzzles return a message starting with `No solution:`.
* **Search** – state = tuple of (cell, new digit); action = correct one more cell (candidates = 12 least-confident cells + cells in a conflict; top-2 alternatives each);
  max 3 corrections; goal = CSP solves the corrected puzzle (optionally exactly one solution).
  A\* heuristic = sum over cell-disjoint conflicting pairs of the cheapest still-possible fix (admissible; states re-opened when reached more cheaply).
* **Extensions included** – uniqueness goal (`require_unique=True`), A\* heuristic, temperature scaling.

## Honest limitations
* Puzzles/mosaics are synthetic; "hard digits" are real MNIST test images chosen by a documented rule.
* "Solvable" ≠ "correct": the cheapest solvable correction can be a different puzzle. This is why recovery < 100% (see Exp 1/3).
* Static-order plain BT is capped at 50,000 assignments in Exp 2 (reported as `limit_rate`), so its mean is a lower bound.
* The MNIST host (yann.lecun.com) may be unreachable; `make_data.py` then falls back to two mirrors with byte-identical files (header-checked).

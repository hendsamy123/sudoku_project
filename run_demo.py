"""One integrated demo.

    python run_demo.py                      # default: seed 1, 30 givens, 2 injected hard digits
    python run_demo.py --seed 4 --misreads 3 --givens 35 --out results/demo.png [--no-plot] [--show]

Prints, in this order:
  1. INPUT        the ground-truth puzzle (* = digit the NN will misread)
  2. FIRST READING the network's reading, with top-3 probabilities for each misread cell
  3. PIPELINE     AC-3 check -> search trace -> counters -> result -> correction log -> solution
  4. NN-GUIDED CORRECTION OFF   (same input, no repair)
  5. SEARCH STRATEGIES          UCS vs Greedy vs A*
  6. CSP PRUNING                BT vs BT+FC vs BT+FC+AC-3 on the corrected puzzle
and saves one overview figure.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LINE = "=" * 72


def rc(i):
    return f"r{i // 9 + 1}c{i % 9 + 1}"


def grid_str(values, marks=(), indent="  "):
    """ASCII Sudoku. values: 81 ints (0 = blank). marks: cells shown with a trailing '*'."""
    sep = indent + "+" + "+".join(["-" * 10] * 3) + "+"
    out = [sep]
    for r in range(9):
        parts = []
        for b in range(3):
            cells = []
            for c in range(b * 3, b * 3 + 3):
                i = r * 9 + c
                v = values[i]
                cells.append(("." if not v else str(v)) + ("*" if i in marks else " "))
            parts.append(" " + " ".join(cells) + " ")
        out.append(indent + "|" + "|".join(parts) + "|")
        if r % 3 == 2:
            out.append(sep)
    return "\n".join(out)


def top3(probs_row):
    idx = np.argsort(-probs_row)[:3]
    return ", ".join(f"{int(d) + 1}:{probs_row[d]:.2f}" for d in idx)


def make_figure(path, show, cells, bad, read, probs, cf, results, rows, on, off, truth):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from csp.sudoku_csp import domain_sizes_after_ac3
    from data.puzzles import mosaic
    from integration.viz import draw_grid

    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    ax[0, 0].imshow(mosaic(cells), cmap="gray_r"); ax[0, 0].set_xticks([]); ax[0, 0].set_yticks([])
    for i in bad:
        r_, c_ = divmod(i, 9)
        ax[0, 0].add_patch(plt.Rectangle((c_ * 28, r_ * 28), 28, 28, fill=False, ec="red", lw=2))
    ax[0, 0].set_title("Input mosaic (red = injected hard digits)")
    conf = np.array([[probs[r * 9 + c].max() if read[r * 9 + c] else np.nan for c in range(9)] for r in range(9)])
    ax[0, 1].imshow(conf, cmap="RdYlGn", vmin=0.3, vmax=1.0); ax[0, 1].set_xticks([]); ax[0, 1].set_yticks([])
    for i in range(81):
        if read[i]:
            ax[0, 1].text(i % 9, i // 9, f"{read[i]}\n{probs[i].max():.2f}", ha="center", va="center", fontsize=7)
    ax[0, 1].set_title("NN reading: digit and confidence")
    _, after, _, _ = domain_sizes_after_ac3(read)
    draw_grid(ax[0, 2], [read[i] if read[i] else after[i] for i in range(81)], {i: "red" for pair in cf for i in pair},
              "AC-3 on the read grid: conflicting digits in red\n(grey = domain size left for blank cells)", 12,
              muted={i for i in range(81) if not read[i]})
    for name, col in zip(results, ("tab:blue", "tab:orange", "tab:green")):
        r = results[name]
        ax[1, 0].plot(range(len(r.trace)), [t["g"] for t in r.trace], "o-", color=col, label=f"{name}: {r.expanded} expanded")
    ax[1, 0].set_xlabel("expansion order (first 60)"); ax[1, 0].set_ylabel("cost g of expanded state"); ax[1, 0].legend()
    ax[1, 0].set_title("Search trace: cost of each expanded correction set")
    names, w = list(rows), 0.38
    ax[1, 1].bar(np.arange(3) - w / 2, [rows[n].assignments for n in names], w, label="assignments")
    ax[1, 1].bar(np.arange(3) + w / 2, [rows[n].backtracks for n in names], w, label="backtracks")
    ax[1, 1].set_xticks(range(3)); ax[1, 1].set_xticklabels(names); ax[1, 1].set_yscale("symlog"); ax[1, 1].legend()
    ax[1, 1].set_title("CSP pruning on the corrected puzzle")
    colors = {c["cell"]: "orange" for c in on.corrections}
    colors.update({i: "red" for i in bad if i not in colors})
    exact = (on.solution == truth) if on.solution else False
    draw_grid(ax[1, 2], on.solution if on.solution else [""] * 81, colors,
              f"Final ({on.status}); orange = corrected cell\nNN ON: exact true solution = {exact} | NN OFF: {off.status}",
              11, bold={i for i in range(81) if read[i]})
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--misreads", type=int, default=2, help="number of injected hard digits (0-6)")
    ap.add_argument("--givens", type=int, default=30, help="number of given digits (25-40)")
    ap.add_argument("--out", default=str(ROOT / "results" / "demo.png"), help="figure file")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--show", action="store_true", help="also open the figure window")
    a = ap.parse_args()
    if not 0 <= a.misreads <= 6 or not 22 <= a.givens <= 45:
        sys.exit("error: --misreads must be 0..6 and --givens 22..45")

    model_path = ROOT / "models" / "mlp.npz"
    try:
        from data.make_data import load_data
        d = load_data()
    except FileNotFoundError:
        sys.exit("error: MNIST not prepared. Run:  python -m data.make_data")
    if not model_path.exists():
        sys.exit("error: trained network not found. Run:  python -m nn.train")

    from csp.sudoku_csp import cell_name, conflicts, solve
    from data.puzzles import build_pools, make_cell_images, make_puzzle
    from integration.pipeline import read_cells, solve_reading
    from nn.model import MLP
    from search.correction import ALGORITHMS, CorrectionProblem

    net = MLP.load(model_path)
    clean, hard = build_pools(net, d["x_test"], d["y_test"])
    rng = np.random.default_rng(a.seed)
    puzzle, truth = make_puzzle(rng, a.givens)
    cells, bad = make_cell_images(puzzle, d["x_test"], clean, hard, rng, a.misreads)
    read, probs = read_cells(net, cells)
    n_bad = len(bad)

    # ------------------------------------------------------------------ 1. INPUT
    print(f"ground-truth puzzle ({a.givens} givens, unique solution), digits marked * are the ones the NN will misread:")
    print(grid_str(puzzle, bad))
    print(f"\n(seed={a.seed}, injected hard digits requested={a.misreads}, injected={n_bad})")

    # ------------------------------------------------------------------ 2. FIRST READING
    print("\nFIRST READING by the network:")
    print(grid_str(read, bad))
    if not bad:
        print("  (no digit was misread in this puzzle)")
    for i in bad:
        print(f"  cell ({rc(i)}): read {read[i]} (true {puzzle[i]}); top-3: {top3(probs[i])}")

    # ------------------------------------------------------------------ 3. PIPELINE
    on = solve_reading(read, probs, "ucs", use_nn=True)
    cf = conflicts(read)
    print("\n" + LINE + "\nPIPELINE (UCS, NN ON)")
    first = solve(read, use_fc=True, use_ac3=True, order="mrv")
    if first.solved:
        print("  AC-3 / CSP: the first reading is consistent and solvable -> nothing to repair")
    else:
        print(f"  AC-3 / CSP: {first.message}  -> the reading is inconsistent")
        if cf:
            print("  duplicate-digit conflicts:", ", ".join(f"{rc(i)}={read[i]} vs {rc(j)}={read[j]}" for i, j in cf))
    s = on.search
    if s is None:
        print("  search: not needed (zero corrections)")
    else:
        print("  search trace (first expansions: #, cost so far g, heuristic h, corrections):")
        for k, t in enumerate(s.trace[:10], 1):
            print(f"    #{k:<3d} g={t['g']:6.3f} h={t['h']:6.3f}  {[(rc(c), dg) for c, dg in t['state']]}" + ("   <- goal" if t["goal"] else ""))
        if len(s.trace) > 10:
            print(f"    ... ({len(s.trace) - 10} more expansions not shown)")
        print(f"  search counters: generated={s.generated} expanded={s.expanded} pruned={s.pruned} "
              f"goal_tests={s.goal_tests} max_frontier={s.max_frontier} csp_solves={s.csp_calls}")
    final_grid = on.corrected_grid or read
    st = solve(final_grid, use_fc=True, use_ac3=True, order="mrv").stats if on.solution else None
    if st:
        print(f"  CSP counters   : assignments={st.assignments} backtracks={st.backtracks} fc_prunings={st.fc_prunings} "
              f"ac3_revisions={st.ac3_revisions} ac3_prunings={st.ac3_removals}")
    if on.status == "solved_directly":
        print("  RESULT: solved with 0 corrections")
    elif on.status == "solved_after_correction":
        print(f"  RESULT: solved after {len(on.corrections)} correction(s), total cost {s.total_cost:.3f}")
    else:
        print("  RESULT: NO SOLUTION: puzzle unreadable. " + on.message)
        for i, dgt, p in on.least_confident:
            print(f"    least confident: {rc(i)} read {dgt} @ {p:.2f}")
    print("  correction log (cell, read digit, new digit, cost):")
    if on.corrections:
        for c in on.corrections:
            print(f"    ({rc(c['cell'])})  {c['read']} -> {c['new']}   cost {c['cost']:.4f}")
    else:
        print("    (none)")
    if on.solution:
        print("  solution:")
        print(grid_str(on.solution))
        print(f"  matches the writer's original solution: {on.solution == truth}")

    # ------------------------------------------------------------------ 4. NN OFF
    off = solve_reading(read, probs, "ucs", use_nn=False)
    print("\n" + LINE + "\nNN-GUIDED CORRECTION OFF (solve the first reading as it is)")
    if off.status == "solved_directly":
        print(f"  RESULT: solved (the first reading was already fine). matches original: {off.solution == truth}")
    else:
        print("  RESULT: " + off.message)

    # ------------------------------------------------------------------ 5. SEARCH STRATEGIES
    print("\n" + LINE + "\nSEARCH STRATEGIES (nodes generated / expanded, CSP solves, total cost)")
    results = {}
    for name, f in ALGORITHMS.items():
        r = f(CorrectionProblem(read, probs))
        results[name] = r
        cost = f"{r.total_cost:.3f}" if r.status == "solved" else "no solution"
        print(f"  {name:6s} generated {r.generated:6d}  expanded {r.expanded:6d}  CSP solves {r.csp_calls:5d}  cost {cost}")

    # ------------------------------------------------------------------ 6. CSP PRUNING
    print("\n" + LINE + "\nCSP PRUNING on the " + ("corrected" if on.solution else "read") + " puzzle (row-major order)")
    rows = {}
    for label, kw in (("BT", {}), ("BT+FC", dict(use_fc=True)), ("BT+FC+AC3", dict(use_fc=True, use_ac3=True))):
        r = solve(final_grid, node_limit=500_000, **kw)
        rows[label] = r.stats
        tail = "" if r.status != "limit" else "  (node limit reached)"
        print(f"  {label:10s} assignments {r.stats.assignments:7d}  backtracks {r.stats.backtracks:7d}  "
              f"FC prunings {r.stats.fc_prunings:6d}  AC-3 prunings {r.stats.ac3_removals:4d}{tail}")

    # ------------------------------------------------------------------ figure
    if not a.no_plot:
        make_figure(a.out, a.show, cells, bad, read, probs, cf, results, rows, on, off, truth)
        print(f"\nPlot saved to {a.out}")


if __name__ == "__main__":
    main()

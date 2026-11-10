"""The three experiments. Run:  python -m eval.experiments [--quick]   (results -> results/)

Exp 1  UCS vs Greedy vs A*  : nodes + recovery rate for 0-4 misreads   (30 givens, 5 seeds x 20 puzzles = 100 per level)
Exp 2  BT vs BT+FC vs BT+FC+AC-3 on puzzles with 25/30/35 givens        (5 seeds x 10 puzzles)
Exp 3  NN-guided correction vs no correction vs random correction       (same instances as Exp 1)
All numbers are mean +- std over the per-seed means (5 seeds).
"""
import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from csp.sudoku_csp import solve
from data.make_data import load_data
from data.puzzles import build_pools, make_cell_images, make_puzzle
from integration.pipeline import read_cells
from nn.model import MLP
from search.correction import ALGORITHMS, CorrectionProblem

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEEDS = [1, 2, 3, 4, 5]
LEVELS = [0, 1, 2, 3, 4]


def ms(vals):
    v = np.asarray(vals, float)
    return float(np.nanmean(v)), float(np.nanstd(v))


def write_csv(name, rows):
    with open(RES / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def setup():
    d = load_data()
    net = MLP.load(ROOT / "models" / "mlp.npz")
    clean, hard = build_pools(net, d["x_test"], d["y_test"])
    return d, net, clean, hard


def instances(seed, n_puzzles, d, net, clean, hard, givens=30):
    """For one seed: n puzzles, each with images for every misread level (0-4)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_puzzles):
        puz, sol = make_puzzle(rng, givens)
        per_level = {}
        for lv in LEVELS:
            cells, bad = make_cell_images(puz, d["x_test"], clean, hard, rng, lv)
            read, probs = read_cells(net, cells)
            per_level[lv] = (read, probs, bad)
        out.append((puz, sol, per_level))
    return out


# ------------------------------------------------------------------ Experiment 1 + 3
def exp1_and_3(n_puzzles):
    d, net, clean, hard = setup()
    raw1, raw3 = [], []
    for seed in SEEDS:
        t0 = time.time()
        inst = instances(seed, n_puzzles, d, net, clean, hard)
        rng_rand = np.random.default_rng(1000 + seed)
        for lv in LEVELS:
            acc = {a: dict(exp=[], gen=[], csp=[], cost=[], rec=[], ok=[]) for a in ALGORITHMS}
            base_rec, rand_rec, base_ok, rand_ok = [], [], [], []
            for puz, sol, per_level in inst:
                read, probs, bad = per_level[lv]
                for a, f in ALGORITHMS.items():
                    r = f(CorrectionProblem(read, probs))
                    acc[a]["exp"].append(r.expanded); acc[a]["gen"].append(r.generated); acc[a]["csp"].append(r.csp_calls)
                    acc[a]["ok"].append(r.status == "solved")
                    acc[a]["rec"].append(r.status == "solved" and r.solution == sol)
                    if r.status == "solved":
                        acc[a]["cost"].append(r.total_cost)
                # Exp 3 baselines
                first = solve(read, use_fc=True, use_ac3=True, order="mrv")
                base_ok.append(first.solved); base_rec.append(first.solved and first.solution == sol)
                filled = [i for i, v in enumerate(read) if v]
                found = None
                if first.solved:
                    found = first
                else:
                    for _ in range(200):   # random correction: same CSP-call budget flavour, no NN guidance
                        k = int(rng_rand.integers(1, 4))
                        g = list(read)
                        for c in rng_rand.choice(filled, size=k, replace=False):
                            g[c] = int(rng_rand.choice([x for x in range(1, 10) if x != read[c]]))
                        r = solve(g, use_fc=True, use_ac3=True, order="mrv")
                        if r.solved:
                            found = r; break
                rand_ok.append(found is not None); rand_rec.append(found is not None and found.solution == sol)
            for a in ALGORITHMS:
                raw1.append(dict(seed=seed, level=lv, algo=a, expanded=np.mean(acc[a]["exp"]), generated=np.mean(acc[a]["gen"]),
                                 csp_calls=np.mean(acc[a]["csp"]), cost=np.mean(acc[a]["cost"]) if acc[a]["cost"] else np.nan,
                                 solvable_rate=np.mean(acc[a]["ok"]), recovery_rate=np.mean(acc[a]["rec"])))
            raw3.append(dict(seed=seed, level=lv, method="no_correction", recovery_rate=np.mean(base_rec), solvable_rate=np.mean(base_ok)))
            raw3.append(dict(seed=seed, level=lv, method="random_correction", recovery_rate=np.mean(rand_rec), solvable_rate=np.mean(rand_ok)))
            raw3.append(dict(seed=seed, level=lv, method="nn_guided_ucs", recovery_rate=np.mean(acc["ucs"]["rec"]), solvable_rate=np.mean(acc["ucs"]["ok"])))
        print(f"[exp1/3] seed {seed} done in {time.time() - t0:.0f}s", flush=True)
    write_csv("exp1_raw.csv", raw1); write_csv("exp3_raw.csv", raw3)

    rows1 = []
    for lv in LEVELS:
        for a in ALGORITHMS:
            sel = [r for r in raw1 if r["level"] == lv and r["algo"] == a]
            row = dict(misreads=lv, algorithm=a)
            for k in ("expanded", "generated", "csp_calls", "cost", "solvable_rate", "recovery_rate"):
                m, s = ms([r[k] for r in sel]); row[k] = f"{m:.2f} ± {s:.2f}"
            rows1.append(row)
    rows3 = []
    for lv in LEVELS:
        for m_ in ("no_correction", "random_correction", "nn_guided_ucs"):
            sel = [r for r in raw3 if r["level"] == lv and r["method"] == m_]
            a, b = ms([r["recovery_rate"] for r in sel]); c, e = ms([r["solvable_rate"] for r in sel])
            rows3.append(dict(misreads=lv, method=m_, recovery_rate=f"{a:.2f} ± {b:.2f}", solvable_rate=f"{c:.2f} ± {e:.2f}"))
    (RES / "exp1_table.md").write_text(md_table(rows1, list(rows1[0])))
    (RES / "exp3_table.md").write_text(md_table(rows3, list(rows3[0])))
    print(md_table(rows1, list(rows1[0]))); print(); print(md_table(rows3, list(rows3[0])))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, col in zip(ALGORITHMS, ("tab:blue", "tab:orange", "tab:green")):
        for k, axis, lab in (("expanded", ax[0], "nodes expanded (log)"), ("recovery_rate", ax[1], "exact-solution recovery rate")):
            mu = [ms([r[k] for r in raw1 if r["level"] == lv and r["algo"] == a]) for lv in LEVELS]
            axis.errorbar(LEVELS, [m for m, _ in mu], [s for _, s in mu], marker="o", capsize=3, label=a, color=col)
            axis.set_xlabel("injected misreads"); axis.set_ylabel(lab)
        mu = [ms([r["csp_calls"] for r in raw1 if r["level"] == lv and r["algo"] == a]) for lv in LEVELS]
        ax[2].errorbar(LEVELS, [m for m, _ in mu], [s for _, s in mu], marker="o", capsize=3, label=a, color=col)
    ax[0].set_yscale("log"); ax[2].set_yscale("log"); ax[2].set_ylabel("CSP calls (log)"); ax[2].set_xlabel("injected misreads")
    ax[0].set_title("Exp 1: search effort"); ax[1].set_title("Exp 1: recovery of the TRUE solution"); ax[2].set_title("Exp 1: CSP calls")
    ax[0].legend(); fig.tight_layout(); fig.savefig(RES / "exp1_search.png", dpi=130)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m_, col in zip(("no_correction", "random_correction", "nn_guided_ucs"), ("tab:red", "tab:gray", "tab:green")):
        mu = [ms([r["recovery_rate"] for r in raw3 if r["level"] == lv and r["method"] == m_]) for lv in LEVELS]
        ax.errorbar(LEVELS, [m for m, _ in mu], [s for _, s in mu], marker="o", capsize=3, label=m_, color=col)
    ax.set_xlabel("injected misreads"); ax.set_ylabel("exact-solution recovery rate"); ax.set_title("Exp 3: effect of the NN-guided correction")
    ax.legend(); fig.tight_layout(); fig.savefig(RES / "exp3_correction.png", dpi=130)


# ------------------------------------------------------------------ Experiment 2
CONFIGS = {"BT": dict(), "BT+FC": dict(use_fc=True), "BT+FC+AC-3": dict(use_fc=True, use_ac3=True)}


def exp2(n_puzzles, node_limit=50_000):
    raw = []
    for order in ("static", "mrv"):
        for givens in (25, 30, 35):
            for seed in SEEDS:
                rng = np.random.default_rng(seed * 100 + givens)
                puzzles = [make_puzzle(rng, givens)[0] for _ in range(n_puzzles)]
                for name, kw in CONFIGS.items():
                    A, B, T, L = [], [], [], []
                    for p in puzzles:
                        t = time.perf_counter()
                        r = solve(p, order=order, node_limit=node_limit, **kw)
                        T.append(time.perf_counter() - t); A.append(r.stats.assignments); B.append(r.stats.backtracks)
                        L.append(r.status == "limit")
                    raw.append(dict(order=order, givens=givens, seed=seed, config=name, assignments=np.mean(A), backtracks=np.mean(B),
                                    ms=1000 * np.mean(T), limit_rate=np.mean(L)))
        print(f"[exp2] order={order} done", flush=True)
    write_csv("exp2_raw.csv", raw)
    rows = []
    for order in ("static", "mrv"):
        for g in (25, 30, 35):
            for name in CONFIGS:
                sel = [r for r in raw if r["order"] == order and r["givens"] == g and r["config"] == name]
                row = dict(order=order, givens=g, config=name)
                for k in ("assignments", "backtracks", "ms", "limit_rate"):
                    m, s = ms([r[k] for r in sel]); row[k] = f"{m:.1f} ± {s:.1f}"
                rows.append(row)
    (RES / "exp2_table.md").write_text(md_table(rows, list(rows[0])))
    print(md_table(rows, list(rows[0])))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, order in zip(ax, ("static", "mrv")):
        w = 0.25
        for j, (name, col) in enumerate(zip(CONFIGS, ("tab:red", "tab:orange", "tab:green"))):
            mu = [ms([r["assignments"] for r in raw if r["order"] == order and r["givens"] == g and r["config"] == name]) for g in (25, 30, 35)]
            axis.bar(np.arange(3) + (j - 1) * w, [m for m, _ in mu], w, yerr=[s for _, s in mu], capsize=3, label=name, color=col)
        axis.set_xticks(range(3)); axis.set_xticklabels(["25", "30", "35"]); axis.set_yscale("log")
        axis.set_xlabel("givens"); axis.set_title(f"Exp 2: variable order = {order} (cap {node_limit:,})")
    ax[0].set_ylabel("assignments (log)"); ax[0].legend(); fig.tight_layout(); fig.savefig(RES / "exp2_csp.png", dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few puzzles per seed (smoke test)")
    ap.add_argument("--only", choices=["13", "2"], default=None)
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    if a.only != "2":
        exp1_and_3(n_puzzles=3 if a.quick else 20)
    if a.only != "13":
        exp2(n_puzzles=2 if a.quick else 10)

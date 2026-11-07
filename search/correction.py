"""Search over 'correction sets' of the network's first reading of a Sudoku.

State  : tuple of (cell, new_digit) pairs (sorted); each cell can be corrected at most once.
Action : change ONE more uncertain cell to one of its top-`n_alts` alternative digits.
Cost   : g = sum over corrections of  log p(read digit) - log p(new digit)   (>= 0, because read = argmax).
Goal   : the corrected puzzle is solvable by the CSP (optionally: has exactly one solution).
Max `max_corrections` corrections.

Algorithms
  UCS     : priority g                  (optimal: cheapest correction set)
  Greedy  : priority h                  (fast, not optimal; ties broken by g)
  A*      : priority g + h              (optimal when h is admissible; states are re-opened if reached cheaper)

Heuristic h (admissible): every pair of peer cells holding the same digit needs at least one of the two to change.
Take a set of cell-disjoint conflicting pairs; each needs its OWN changed cell, so
h = sum over those pairs of the cheapest change cost among their still-changeable cells.
h = inf (state pruned in A*/Greedy) if a pair has no changeable cell or needs more changes than remain.
"""
import heapq
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from csp.sudoku_csp import conflicts, solve, parse, PEERS


@dataclass
class SearchResult:
    algorithm: str
    status: str                       # "solved" | "unreadable"
    corrections: list = field(default_factory=list)   # dicts: cell, read, new, cost
    total_cost: float = math.inf
    grid: list = None
    solution: list = None
    expanded: int = 0
    generated: int = 0
    csp_calls: int = 0
    trace: list = field(default_factory=list)
    message: str = ""
    pruned: int = 0            # children discarded because h = inf (provably unrepairable)
    goal_tests: int = 0        # goal tests performed (one per expanded state)
    max_frontier: int = 0      # largest size of the priority queue


class CorrectionProblem:
    def __init__(self, read_grid, probs, max_corrections=3, n_uncertain=12, n_alts=2, require_unique=False,
                 csp_kwargs=None):
        self.read = parse(read_grid)
        self.probs = np.clip(np.asarray(probs, dtype=float), 1e-9, 1.0)   # (81, 9)
        self.max_corrections = max_corrections
        self.require_unique = require_unique
        self.csp_kwargs = csp_kwargs or dict(use_fc=True, use_ac3=True, order="mrv")
        filled = [i for i, v in enumerate(self.read) if v]
        conf = {i: self.probs[i, self.read[i] - 1] for i in filled}
        by_conf = sorted(filled, key=lambda i: conf[i])[:n_uncertain]
        in_conflict = {c for pair in conflicts(self.read) for c in pair}   # cells flagged by the CSP
        self.candidates = sorted(set(by_conf) | in_conflict)
        self.options = {}   # cell -> list of (new_digit, cost) cheapest first
        for i in self.candidates:
            order = [d for d in np.argsort(-self.probs[i]) + 1 if d != self.read[i]][:n_alts]
            self.options[i] = [(int(d), float(math.log(self.probs[i, self.read[i] - 1]) - math.log(self.probs[i, d - 1])))
                               for d in order]
        self._cache = {}
        self.csp_calls = 0

    # ---- helpers ------------------------------------------------------------
    def grid(self, state):
        g = list(self.read)
        for c, d in state:
            g[c] = d
        return g

    def cost_of(self, cell, digit):
        return dict(self.options[cell])[digit]

    def csp(self, grid):
        key = tuple(grid)
        if key not in self._cache:
            self.csp_calls += 1
            self._cache[key] = solve(grid, max_solutions=2 if self.require_unique else 1, **self.csp_kwargs)
        return self._cache[key]

    def is_goal(self, state):
        r = self.csp(self.grid(state))
        if self.require_unique:
            return len(r.solutions) == 1
        return r.solved

    def successors(self, state):
        done = {c for c, _ in state}
        if len(state) >= self.max_corrections:
            return
        for c in self.candidates:
            if c in done:
                continue
            for d, cost in self.options[c]:
                yield tuple(sorted(state + ((c, d),))), cost

    def h(self, state):
        done = {c for c, _ in state}
        pairs = conflicts(self.grid(state))
        if not pairs:
            return 0.0

        def cell_cost(c):
            return min(cost for _, cost in self.options[c]) if c in self.options and c not in done else math.inf

        scored = sorted(((min(cell_cost(a), cell_cost(b)), a, b) for a, b in pairs), reverse=True)
        taken, total, n = set(), 0.0, 0
        for cost, a, b in scored:                 # cell-disjoint pairs (greedy packing)
            if a in taken or b in taken:
                continue
            if math.isinf(cost):
                return math.inf
            taken.update((a, b))
            total += cost
            n += 1
        return math.inf if n > self.max_corrections - len(state) else total


def _run(problem, algorithm, node_limit=20000):
    start = ()
    best_g = {start: 0.0}
    cnt = itertools.count()
    prio = {"ucs": lambda g, h: g, "greedy": lambda g, h: h, "astar": lambda g, h: g + h}[algorithm]
    use_h = algorithm != "ucs"

    def mk(state, g):
        h = problem.h(state) if use_h else 0.0
        return h, (prio(g, h), g if algorithm == "greedy" else 0.0, next(cnt), state, g)

    h0, e0 = mk(start, 0.0)
    heap = [e0]
    res = SearchResult(algorithm, "unreadable")
    res.generated = 1
    res.max_frontier = 1
    seen = set()
    while heap:
        _, _, _, state, g = heapq.heappop(heap)
        if algorithm != "greedy" and g > best_g.get(state, math.inf):
            continue
        if algorithm == "greedy":
            if state in seen:
                continue
            seen.add(state)
        res.expanded += 1
        goal = problem.is_goal(state)
        res.goal_tests += 1
        if len(res.trace) < 60:
            res.trace.append(dict(state=state, g=g, h=problem.h(state), goal=goal))
        if goal:
            r = problem.csp(problem.grid(state))
            res.status, res.total_cost = "solved", g
            res.corrections = [dict(cell=c, read=problem.read[c], new=d, cost=problem.cost_of(c, d)) for c, d in state]
            res.grid, res.solution = problem.grid(state), r.solution
            break
        if res.expanded >= node_limit:
            res.message = f"node limit {node_limit} reached"
            break
        for ns, cost in problem.successors(state):
            ng = g + cost
            if algorithm == "greedy" and ns in seen:
                continue
            if ng < best_g.get(ns, math.inf) or algorithm == "greedy":
                best_g[ns] = min(ng, best_g.get(ns, math.inf))
                h, entry = mk(ns, ng)
                res.generated += 1
                if math.isinf(h):
                    res.pruned += 1
                    continue          # pruned: provably cannot be repaired within the budget
                heapq.heappush(heap, entry)
                res.max_frontier = max(res.max_frontier, len(heap))
    res.csp_calls = problem.csp_calls
    if res.status != "solved":
        res.message = res.message or "no solvable corrected puzzle found within the correction budget"
    return res


def ucs(problem, **kw):
    return _run(problem, "ucs", **kw)


def greedy(problem, **kw):
    return _run(problem, "greedy", **kw)


def astar(problem, **kw):
    return _run(problem, "astar", **kw)


ALGORITHMS = {"ucs": ucs, "greedy": greedy, "astar": astar}


def brute_force(problem):
    """Reference answer: enumerate every correction set within the same action space (used in tests)."""
    best = (math.inf, None)
    cells = problem.candidates
    for k in range(problem.max_corrections + 1):
        for chosen in itertools.combinations(cells, k):
            for digits in itertools.product(*[problem.options[c] for c in chosen]):
                state = tuple(sorted((c, d) for c, (d, _) in zip(chosen, digits)))
                cost = sum(cc for _, cc in digits)
                if cost < best[0] and problem.is_goal(state):
                    best = (cost, state)
    return best

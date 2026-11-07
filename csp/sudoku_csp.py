"""Sudoku as a CSP: 81 variables, domains {1..9} (bitmasks), 'all-different' on rows/cols/boxes.

Three independently switchable parts:
  * Backtracking (always on)
  * Forward Checking     (use_fc=True)   - prune peers' domains after every assignment
  * AC-3                 (use_ac3=True)  - arc consistency on the puzzle BEFORE search
Variable ordering: order="static" (row-major) or "mrv" (fewest candidates first).

Counters (Stats): assignments, backtracks, ac3_revisions, ac3_removals, fc_prunings, checks.
"""
from collections import deque
from dataclasses import dataclass, field

ALL = 0b1111111110  # bits 1..9


def _build():
    units = [[r * 9 + c for c in range(9)] for r in range(9)]
    units += [[r * 9 + c for r in range(9)] for c in range(9)]
    units += [[(br * 3 + r) * 9 + bc * 3 + c for r in range(3) for c in range(3)] for br in range(3) for bc in range(3)]
    peers = [set() for _ in range(81)]
    for u in units:
        for a in u:
            peers[a].update(x for x in u if x != a)
    return units, [sorted(p) for p in peers]


UNITS, PEERS = _build()
ROW = [i // 9 for i in range(81)]
COL = [i % 9 for i in range(81)]
BOX = [(i // 27) * 3 + (i % 9) // 3 for i in range(81)]


@dataclass
class Stats:
    assignments: int = 0
    backtracks: int = 0
    ac3_revisions: int = 0
    ac3_removals: int = 0
    fc_prunings: int = 0
    checks: int = 0
    limit_hit: bool = False

    def as_dict(self):
        return dict(self.__dict__)


@dataclass
class SolveResult:
    status: str  # "solved" | "no_solution" | "limit"
    solutions: list = field(default_factory=list)
    message: str = ""
    stats: Stats = field(default_factory=Stats)

    @property
    def solved(self):
        return self.status == "solved"

    @property
    def solution(self):
        return self.solutions[0] if self.solutions else None


class _Limit(Exception):
    pass


def parse(grid):
    """Accepts a string of 81 chars ('0' or '.' = blank), a flat list or a 9x9 nested list."""
    if isinstance(grid, str):
        return [0 if ch in ".0" else int(ch) for ch in grid.strip() if ch not in " \n"]
    flat = []
    for x in grid:
        flat.extend(x) if hasattr(x, "__len__") else flat.append(x)
    return [int(v) for v in flat]


def cell_name(i):
    return f"(row {ROW[i] + 1}, col {COL[i] + 1})"


def conflicts(grid):
    """All pairs (i, j), i<j, of peer cells holding the same non-zero digit."""
    g = parse(grid)
    return [(i, j) for i in range(81) if g[i] for j in PEERS[i] if j > i and g[j] == g[i]]


def is_valid_solution(grid):
    g = parse(grid)
    return len(g) == 81 and all(v in range(1, 10) for v in g) and all(
        sorted(g[i] for i in u) == list(range(1, 10)) for u in UNITS)


def ac3(domains, stats=None, queue=None):
    """AC-3 on the != constraints between peers. Mutates `domains`. Returns (ok, wiped_cell)."""
    stats = stats or Stats()
    if queue is None:
        queue = deque((xi, xj) for xi in range(81) for xj in PEERS[xi])
    while queue:
        xi, xj = queue.popleft()
        stats.ac3_revisions += 1
        dj = domains[xj]
        if dj & (dj - 1) == 0 and dj & domains[xi]:  # xj fixed -> remove its value from xi
            domains[xi] &= ~dj
            stats.ac3_removals += 1
            if domains[xi] == 0:
                return False, xi
            for xk in PEERS[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    return True, None


def solve(grid, use_fc=False, use_ac3=False, order="static", max_solutions=1, node_limit=None):
    stats = Stats()
    g = parse(grid)
    if len(g) != 81:
        raise ValueError("grid must have 81 cells")
    domains = [(1 << v) if v else ALL for v in g]

    # --- preprocessing -------------------------------------------------------
    if use_ac3:
        ok, bad = ac3(domains, stats)
        if not ok:
            return SolveResult("no_solution", [], f"No solution: AC-3 emptied the domain of cell {cell_name(bad)} "
                               "(the givens contradict each other).", stats)
    else:
        cf = conflicts(g)
        if cf:
            i, j = cf[0]
            return SolveResult("no_solution", [], f"No solution: cells {cell_name(i)} and {cell_name(j)} "
                               f"both contain {g[i]} in the same unit.", stats)
    assign = list(g)
    used = [0] * 27  # rows 0-8, cols 9-17, boxes 18-26 (digit bitmasks)
    for i, v in enumerate(g):
        if v:
            for u in (ROW[i], 9 + COL[i], 18 + BOX[i]):
                used[u] |= 1 << v
    if use_fc:  # propagate givens once
        for i, v in enumerate(g):
            if v:
                for p in PEERS[i]:
                    if not g[p] and domains[p] & (1 << v):
                        domains[p] &= ~(1 << v)
                        stats.fc_prunings += 1
                        if domains[p] == 0:
                            return SolveResult("no_solution", [], f"No solution: forward checking emptied the domain of "
                                               f"cell {cell_name(p)}.", stats)

    solutions = []

    def candidates(i, doms):
        if use_fc:
            return doms[i]
        stats.checks += 1
        return doms[i] & ~(used[ROW[i]] | used[9 + COL[i]] | used[18 + BOX[i]])

    def pick(doms):
        best, best_n = -1, 99
        for i in range(81):
            if assign[i] == 0:
                if order == "static":
                    return i
                n = bin(candidates(i, doms)).count("1")
                if n < best_n:
                    best, best_n = i, n
                    if n <= 1:
                        break
        return best

    def rec(doms):
        i = pick(doms)
        if i < 0:
            solutions.append(list(assign))
            return len(solutions) >= max_solutions
        cands = candidates(i, doms)
        for d in range(1, 10):
            bit = 1 << d
            if not cands & bit:
                continue
            stats.assignments += 1
            if node_limit is not None and stats.assignments > node_limit:
                stats.limit_hit = True
                raise _Limit
            assign[i] = d
            for u in (ROW[i], 9 + COL[i], 18 + BOX[i]):
                used[u] |= bit
            ok = True
            if use_fc:
                nd = list(doms)
                for p in PEERS[i]:
                    if assign[p] == 0 and nd[p] & bit:
                        nd[p] &= ~bit
                        stats.fc_prunings += 1
                        if nd[p] == 0:
                            ok = False
                            break
                if ok and rec(nd):
                    return True
            elif rec(doms):
                return True
            assign[i] = 0
            for u in (ROW[i], 9 + COL[i], 18 + BOX[i]):
                used[u] &= ~bit
            stats.backtracks += 1
        return False

    try:
        rec(domains)
    except _Limit:
        return SolveResult("limit", solutions, f"Node limit of {node_limit} assignments reached.", stats)
    if solutions:
        return SolveResult("solved", solutions, "Solved.", stats)
    return SolveResult("no_solution", [], "No solution: search exhausted every possible assignment "
                       "(the givens are consistent but cannot be completed).", stats)


def count_solutions(grid, limit=2):
    r = solve(grid, use_fc=True, use_ac3=True, order="mrv", max_solutions=limit)
    return len(r.solutions)


def domain_sizes_after_ac3(grid):
    """Domain size of every cell before/after AC-3 (for plots)."""
    g = parse(grid)
    dom = [(1 << v) if v else ALL for v in g]
    before = [bin(d).count("1") for d in dom]
    ok, bad = ac3(dom)
    return before, [bin(d).count("1") for d in dom], ok, bad

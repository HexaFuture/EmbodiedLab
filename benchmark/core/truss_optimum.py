"""The optimum P* of truss_bridge: the strongest bridge the kit builds on the pegboard within the mass limit, at the
kinds' nominal (hidden) properties. Pure numpy/scipy (HiGHS MILP); deterministic. Used by the evaluator (cached) and by
the reference analysis (with the properties it measured).

Ground structure: every hole pair a bar spans (truss_model.links), each with up to MAX_PER_PAIR bars (one). Variables: the
bars on each pair and kind (binary copies), a loose pin in each hole (binary), each pair-and-kind's axial force, and
the load P. Constraints: equilibrium of every joint under P at C, each force within its bars' limits (tension T,
compression the strut load at that length), the mass limit, the kit, the number of loose pins, no bar over a pin, at
most STACK eyes on a pin.

1. Determinate stage: bars on exactly 2 j - 3 distinct pairs for j joints, one kind per pair. A rigid determinate truss
   has one equilibrium solution, so the MILP's P (the plastic capacity) is the evaluator's elastic first-failure load.
   Mechanisms are cut off by Laman counts on overbraced subsets, and geometric degeneracies by a cut on the pair set.
2. Redundant stages: 2 j - 3 + r distinct pairs (r = 0, 1, 2) with mixed kinds allowed on a pair. The elastic capacity
   of such a design is at most its plastic one, so the MILP only looks above the best elastic value found so far; each
   design found is analysed elastically (truss_model.capacity) and cut off. A stage that runs out of designs above the
   floor is complete. With every stage complete the optimum is certified (over designs with at most two redundant
   pairs and one bar per pair, which a 12-instance exhaustive check on a smaller board never found beaten).
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np

try:                                      # in the bench image core is flat on the path; on the host a package
    import truss_model as M
except ImportError:
    from benchmark.core import truss_model as M

MAX_PER_PAIR = 1          # a bar laid on its twin rests on the twin's grip knob, off the pins
MIP_GAP = 1e-3
# Deterministic limits (no wall clock decides a result): branch-and-bound nodes per MILP solve and solves per stage.
# SAFETY_S only guards against a hung solver; a solve that reaches it is recorded (stage 'timed_out') and the result
# is then machine-dependent.
NODE_LIMIT = 200000
DET_MAX_ITER, RED_MAX_ITER = 200, 80
SAFETY_S = 3600.0
SOLVE_LOG = []                        # (stage, nodes, status, seconds) of every solve, for studies


class GroundStructure:
    def __init__(self):
        self.holes = sorted(M.HOLES)
        self.index = {h: i for i, h in enumerate(self.holes)}
        self.xy = np.array([M.HOLE_XY_MM[h] for h in self.holes], float)
        self.fixed = [self.index[n] for n in M.FIXED]
        self.loose = [i for i in range(len(self.holes)) if i not in self.fixed]
        self.A, self.B, self.C = self.index['A'], self.index['B'], self.index[M.LOAD_PIN]
        self.pairs = []                                   # (a, b, label, blocker hole indices)
        for (a, b), label in sorted(M.links().items()):
            self.pairs.append((self.index[a], self.index[b], label, tuple(self.index[n] for n in M.blockers(a, b))))
        self.L = np.array([M.BAR_LENGTHS_MM[p[2]] for p in self.pairs])
        self.u = np.array([(self.xy[p[1]] - self.xy[p[0]]) / np.linalg.norm(self.xy[p[1]] - self.xy[p[0]])
                           for p in self.pairs])


GS = None


def ground():
    global GS
    if GS is None:
        GS = GroundStructure()
    return GS


class _Model:
    """Variable layout and the base constraint rows of one MILP."""

    def __init__(self, gs, props, kit, mass_g, n_pins, redundancy, mixed):
        from scipy.sparse import coo_matrix  # noqa: F401  (imported here: scipy only where the optimum is solved)
        self.gs, E, C = gs, len(gs.pairs), MAX_PER_PAIR
        self.nk = NK = len(M.KINDS)
        self.nb = E * NK * C
        self.yidx = {n: self.nb + k for k, n in enumerate(gs.loose)}
        self.f0 = self.nb + len(gs.loose)
        self.w0 = self.f0 + E * NK
        self.iP = self.w0 + E
        self.nv = self.iP + 1
        rows, cols, vals, lo, hi = [], [], [], [], []
        inf = np.inf

        def add(entries, l, h):
            r = len(lo)
            for c, v in entries:
                rows.append(r); cols.append(c); vals.append(v)
            lo.append(l); hi.append(h)

        kinds = M.KINDS
        T = [float(props[k]['T_N']) for k in kinds]
        inc = {n: [] for n in range(len(gs.holes))}
        for e, (a, b, label, blk) in enumerate(gs.pairs):
            inc[a].append((e, 1.0))
            inc[b].append((e, -1.0))
            for k, kind in enumerate(kinds):
                Cc = compression_limit(props[kind], label)
                bs = [self.b(e, k, c) for c in range(C)]
                add([(self.f(e, k), 1.0)] + [(x, -T[k]) for x in bs], -inf, 0.0)
                add([(self.f(e, k), -1.0)] + [(x, -Cc) for x in bs], -inf, 0.0)
                for c in range(1, C):
                    add([(self.b(e, k, c), 1.0), (self.b(e, k, c - 1), -1.0)], -inf, 0.0)
            allb = [self.b(e, k, c) for k in range(NK) for c in range(C)]
            add([(x, 1.0) for x in allb], -inf, C)
            if not mixed:
                add([(self.b(e, k, 0), 1.0) for k in range(NK)], -inf, 1.0)
            add([(self.b(e, k, 0), 1.0) for k in range(NK)] + [(self.w(e), -1.0)], 0.0, 1.0 if mixed else 0.0)
            if mixed:          # w_e = 1 exactly when some bar lies on the pair
                for k in range(NK):
                    add([(self.b(e, k, 0), 1.0), (self.w(e), -1.0)], -inf, 0.0)
            for n in (a, b):
                if n in self.yidx:
                    add([(x, 1.0) for x in allb] + [(self.yidx[n], -C)], -inf, 0.0)
            for n in blk:
                add([(x, 1.0) for x in allb] + [(self.yidx[n], C)], -inf, C)
        for n in range(len(gs.holes)):
            for d in range(2):
                if n == gs.A or (n == gs.B and d == 1):
                    continue
                ent = [(self.f(e, k), s * gs.u[e][d]) for e, s in inc[n] for k in range(NK)]
                if n == gs.C and d == 1:
                    ent.append((self.iP, -1.0))
                if ent:
                    add(ent, 0.0, 0.0)
            eyes = [(self.b(e, k, c), 1.0) for e, _ in inc[n] for k in range(NK) for c in range(C)]
            if eyes:
                if n in self.yidx:
                    add(eyes + [(self.yidx[n], -M.STACK)], -inf, 0.0)
                else:
                    add(eyes, -inf, M.STACK)
            ws = [(self.w(e), 1.0) for e, _ in inc[n]]
            if n in self.yidx:
                add(ws + [(self.yidx[n], -2.0)], 0.0, inf)
            elif ws:
                add(ws, 2.0 if n == gs.C else 1.0, inf)
        add([(y, 1.0) for y in self.yidx.values()], -inf, n_pins)
        nf0 = 2 * len(gs.fixed) - 3 + redundancy
        add([(self.w(e), 1.0) for e in range(E)] + [(y, -2.0) for y in self.yidx.values()], nf0, nf0)
        # mirror symmetry (the span is symmetric about C and the load vertical: a design and its mirror carry alike)
        ent = []
        for n, y in self.yidx.items():
            col, row = M.HOLES[gs.holes[n]]
            if col == M.FIXED['C'][0]:
                continue
            wgt = 1.0 + (min(col, 2 * M.FIXED['C'][0] - col) * 7 + (row + 5)) * 0.01
            ent.append((y, wgt if col < M.FIXED['C'][0] else -wgt))
        if ent:
            add(ent, 0.0, inf)
        for label in M.LABELS:
            for k, kind in enumerate(kinds):
                ent = [(self.b(e, k, c), 1.0) for e in range(E) if gs.pairs[e][2] == label for c in range(C)]
                add(ent, -inf, kit.get((kind, label), 0))
        add([(self.b(e, k, c), M.bar_mass_g(kind, gs.pairs[e][2])) for e in range(E) for k, kind in enumerate(kinds)
             for c in range(C)], -inf, mass_g)
        self.rows, self.cols, self.vals, self.lo, self.hi = rows, cols, vals, lo, hi
        self.integrality = np.zeros(self.nv)
        self.integrality[:self.f0] = 1
        self.integrality[self.w0:self.iP] = 1
        self.lb, self.ub = np.zeros(self.nv), np.ones(self.nv)
        self.lb[self.f0:self.w0], self.ub[self.f0:self.w0] = -inf, inf
        self.ub[self.iP] = inf
        self.c = np.zeros(self.nv)
        self.c[self.iP] = -1.0
        for e in range(E):                 # tie-break: among equal loads the lightest design
            for k, kind in enumerate(kinds):
                for cc in range(C):
                    self.c[self.b(e, k, cc)] = 1e-4 * M.bar_mass_g(kind, gs.pairs[e][2])

    def b(self, e, k, c):
        return (e * self.nk + k) * MAX_PER_PAIR + c

    def f(self, e, k):
        return self.f0 + e * self.nk + k

    def w(self, e):
        return self.w0 + e

    def cut(self, entries, lo, hi):
        r = len(self.lo)
        for c, v in entries:
            self.rows.append(r); self.cols.append(c); self.vals.append(v)
        self.lo.append(lo); self.hi.append(hi)

    def solve(self, stage):
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
        A = coo_matrix((self.vals, (self.rows, self.cols)), shape=(len(self.lo), self.nv)).tocsr()
        t0 = time.time()
        res = milp(self.c, constraints=LinearConstraint(A, self.lo, self.hi), integrality=self.integrality,
                   bounds=Bounds(self.lb, self.ub), options={'node_limit': NODE_LIMIT, 'time_limit': SAFETY_S,
                                                             'mip_rel_gap': MIP_GAP, 'presolve': True})
        secs = time.time() - t0
        res.timed_out = res.status == 1 and secs >= SAFETY_S * 0.99
        res.limited = res.status == 1
        SOLVE_LOG.append((stage, int(getattr(res, 'mip_node_count', -1) or -1), int(res.status), round(secs, 2)))
        return res

    def design(self, x):
        """[(hole a, hole b, label, kind)] of a solution."""
        gs = self.gs
        out = []
        for i in range(self.nb):
            if x[i] > 0.5:
                e, k = (i // MAX_PER_PAIR) // self.nk, (i // MAX_PER_PAIR) % self.nk
                a, b, label, _ = gs.pairs[e]
                out.append((gs.holes[a], gs.holes[b], label, M.KINDS[k]))
        return out


def compression_limit(kind_props, label):
    """A kind's compression limit at a length: from its table 'compression_N' {label: N} if it has one, else the
    strut load of its P100_N and N_N."""
    table = kind_props.get('compression_N')
    if table:
        return float(table[label])
    return M.strut_load(kind_props['P100_N'], kind_props['N_N'], label)


def _bar(a, b, label, kind_props):
    row = {'pins': (a, b), 'label': label, 'T_N': kind_props['T_N'], 'P100_N': kind_props['P100_N']}
    if kind_props.get('compression_N'):
        row['C_N'] = compression_limit(kind_props, label)
    else:
        row['N_N'] = kind_props['N_N']
    return row


def design_capacity(design, props, detail=False):
    """Elastic failure load of a design [(a, b, label, kind)] at properties {kind: {T_N, P100_N, N_N}} (nominal) or
    {kind: {T_N, P100_N, compression_N: {label: N}}} (a table; P100_N then only sets the stiffness)."""
    P, which = M.capacity([_bar(a, b, label, props[kind]) for a, b, label, kind in design])
    return (P, which) if detail else P


def _laman_violation(design):
    pairs = sorted({(a, b) for a, b, _, _ in design})
    nodes = sorted({n for p in pairs for n in p})
    for r in range(3, len(nodes) + 1):
        for S in itertools.combinations(nodes, r):
            S = set(S)
            if sum(1 for a, b in pairs if a in S and b in S) > 2 * r - 3:
                return S
    return None


def _pair_set_cut(m, design):
    used = {(a, b) for a, b, _, _ in design}
    gs = m.gs
    ent = [(m.w(e), -1.0 if (gs.holes[p[0]], gs.holes[p[1]]) in used else 1.0) for e, p in enumerate(gs.pairs)]
    m.cut(ent, 1.0 - len(used), np.inf)


def _design_cut(m, x):
    on = {i for i in range(m.nb) if x[i] > 0.5}
    m.cut([(i, -1.0 if i in on else 1.0) for i in range(m.nb)], 1.0 - len(on), np.inf)


def determinate(props, kit, mass_g, n_pins, max_iter=DET_MAX_ITER):
    gs = ground()
    m = _Model(gs, props, kit, mass_g, n_pins, redundancy=0, mixed=False)
    t0, best, UB, it, timed_out = time.time(), (0.0, None), math.inf, 0, False
    for it in range(1, max_iter + 1):
        res = m.solve('determinate')
        timed_out |= res.timed_out
        if res.x is None:
            if res.status == 2:
                UB = min(UB, best[0])
            break
        x = res.x
        bound = x[m.iP] if res.status == 0 else -(getattr(res, 'mip_dual_bound', None) or res.fun)
        UB = min(UB, max(bound, x[m.iP]))
        d = m.design(x)
        P = design_capacity(d, props)
        if P > best[0]:
            best = (P, d)
        if UB <= best[0] * (1 + 2 * MIP_GAP) or res.timed_out:
            break
        S = _laman_violation(d)
        if S is not None:
            idx = {gs.index[n] for n in S}
            m.cut([(m.w(e), 1.0) for e, p in enumerate(gs.pairs) if p[0] in idx and p[1] in idx], -np.inf,
                  2 * len(S) - 3)
        else:
            _pair_set_cut(m, d)
    return {'P': float(best[0]), 'design': best[1], 'UB': float(UB), 'iters': it, 'secs': time.time() - t0,
            'complete': bool(UB <= best[0] * (1 + 2 * MIP_GAP)), 'timed_out': timed_out}


def redundant(props, kit, mass_g, n_pins, floor, redundancy, max_iter=RED_MAX_ITER):
    gs = ground()
    m = _Model(gs, props, kit, mass_g, n_pins, redundancy=redundancy, mixed=True)
    t0, best, complete, it, timed_out = time.time(), (floor, None), False, 0, False
    for it in range(1, max_iter + 1):
        m.lb[m.iP] = best[0] * (1 + 1e-4)
        res = m.solve(f'redundant{redundancy}')
        timed_out |= res.timed_out
        if res.x is None:
            complete = res.status == 2           # infeasible: no design above the floor is left
            break
        d = m.design(res.x)
        P = design_capacity(d, props)
        if P > best[0]:
            best = (P, d)
        if P <= 0.0:
            _pair_set_cut(m, d)
        else:
            _design_cut(m, res.x)
        if res.timed_out:
            break
    return {'P': float(best[0]), 'design': best[1], 'complete': bool(complete), 'iters': it, 'secs': time.time() - t0,
            'timed_out': timed_out}


def optimum(props, kit=None, mass_g=None, n_pins=None, redundancies=(0, 1, 2)):
    """{'P_N', 'design': [(a, b, label, kind)], 'certified', 'stages'} of the strongest bridge.
    props: {kind: {'T_N', 'P100_N', 'N_N'}} (nominal); kit {(kind, label): count} (default the full kit)."""
    kit = dict(M.KIT if kit is None else kit)
    mass_g = M.MASS_LIMIT_G if mass_g is None else mass_g
    n_pins = M.N_LOOSE_PINS if n_pins is None else n_pins
    keys = ('P', 'complete', 'iters', 'secs', 'timed_out')
    det = determinate(props, kit, mass_g, n_pins)
    best, design, stages = det['P'], det['design'], [dict(stage='determinate', **{k: det[k] for k in keys})]
    for r in redundancies:
        q = redundant(props, kit, mass_g, n_pins, best, r)
        stages.append(dict(stage=f'redundant{r}', **{k: q[k] for k in keys}))
        if q['design'] is not None and q['P'] > best:
            best, design = q['P'], q['design']
    return {'P_N': float(best), 'design': design, 'certified': all(s['complete'] for s in stages),
            'deterministic': not any(s['timed_out'] for s in stages), 'stages': stages}

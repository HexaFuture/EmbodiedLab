"""Reference values of active_filter: A_search, the ranking reference, by a search over every circuit the kit can build
on the board, beside A_ref (the success line, filter_model.optimum). The circuit structure is open (any structure the
kit and the board allow); a layout is scored by the smallest attenuation over the stopband 2 kHz - 20 kHz.

State: what stands on the board, so every state is buildable (benchmark/core/circuit_grid/filter.py's rules): pads
(c, r), c 1..7, r 1..6, row 6 the ground rail (printed copper), the analyzer at IN = (1, 3); a two-pin block (resistor,
capacitor, wire, probe) on one edge between neighbouring pads, one block per edge; an op-amp module on three collinear
pads (IN+, IN-, OUT) taking both edges it spans; at most 8 wires; each piece at most once; one probe, its - pin on the
ground node and its + pin on neither ground nor IN. The response is filter_model.Circuit's at the instance's true
values (the evaluator's solver); the objective is the stopband attenuation, with a penalty of 30 dB per dB of passband
ripple over the limit. Simulated annealing (RUNS independent runs from the two-stage unity-gain Sallen-Key layout of
filter_model.optimum, then ROUNDS - 1 rounds restarted from the best layouts so far, each run ITERS moves: add /
remove / move / revalue / swap a block, a wire, an op-amp seat), scored on a coarse grid (every third passband point,
every sixth stopband point) and re-scored on the evaluator's grid; the best layout within the limit is pruned (blocks
that change nothing taken off) and re-checked with benchmark.evaluation.filter.response_of (the scoring code itself).

The scoring code (benchmark.evaluation.filter.optimum) runs this search with the defaults RUNS = 48, ITERS = 400000,
ROUNDS = 2 when an instance's reference values are neither in a table nor in the cache; to run it ahead of scoring:

    python -m benchmark.evaluation.filter prewarm <instance.json> ...          # A_ref and A_search into the cache
    python -m benchmark.evaluation.filter_search <model_seed> ... [--procs N] [--cache] [--save DIR | --from-saved DIR]

The result depends only on the model seed and RUNS, ITERS, ROUNDS: each run has its own fixed random seed and the runs'
results are collected in order, so the number of processes changes the time, not the result. About 15 min per round
and instance on 48 cores (30 min with the default two rounds); fewer cores take proportionally longer.

Arithmetic path. The annealing meets exact ties between layouts (moves that leave the response unchanged), and whether
a tie stays a tie depends on the last bits of the circuit solves; a different rounding sends the search down another
path. So the search always runs with PINNED_ENV, set before numpy loads: single-threaded OpenBLAS with its Haswell
kernels and numpy's AVX2/FMA3 loops (no AVX-512), the same bits on every x86-64 CPU with AVX2 and FMA3, whichever numpy
release the requirements install. The command line re-executes itself with it; the scoring code calls `reference`,
which runs the command line in a child process.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from benchmark.core import filter_model as M

PINNED_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "OPENBLAS_CORETYPE": "Haswell",
              "NPY_DISABLE_CPU_FEATURES": "AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL "
                                          "AVX512_ICL AVX512_SPR"}


def pinned():
    """Whether this process runs on the pinned arithmetic path (PINNED_ENV set now and when numpy loaded)."""
    return _PINNED_AT_LOAD and all(os.environ.get(k) == v for k, v in PINNED_ENV.items())


_PINNED_AT_LOAD = all(os.environ.get(k) == v for k, v in PINNED_ENV.items())


def _require_pinned():
    if not pinned():
        raise RuntimeError("the filter search runs only on the pinned arithmetic path: use reference() or "
                           "`python -m benchmark.evaluation.filter_search`")

SEARCH_VERSION = "board-anneal-1"
RUNS, ITERS, ROUNDS = 48, 400000, 2
COLS, ROWS = 7, 6
IN = (1, 3)
GROUND = (1, ROWS)
PADS = [(c, r) for r in range(1, ROWS + 1) for c in range(1, COLS + 1)]
COPPER = [((c, ROWS), (c + 1, ROWS)) for c in range(1, COLS)]
EDGES = []
for _c, _r in PADS:
    if _c < COLS and _r < ROWS:
        EDGES.append(((_c, _r), (_c + 1, _r)))
    if _r < ROWS:
        EDGES.append(((_c, _r), (_c, _r + 1)))
LINES = []
for _c, _r in PADS:
    for _dc, _dr in ((1, 0), (0, 1)):
        _q = [(_c + k * _dc, _r + k * _dr) for k in range(3)]
        if all(1 <= a <= COLS and 1 <= b <= ROWS for a, b in _q):
            LINES.append(tuple(_q))
            LINES.append(tuple(_q[::-1]))
LIM = M.RIPPLE_DB
NR, NC = len(M.R_NOM) * M.COPIES, len(M.C_NOM) * M.COPIES
_PI = np.r_[np.arange(0, len(M.F_PASS), 3), len(M.F_PASS) - 1]
_SI = np.r_[np.arange(0, len(M.F_STOP), 6), len(M.F_STOP) - 1]
F_CO = np.r_[M.F_PASS[_PI], M.F_STOP[_SI]]
NPASS_CO = len(_PI)


def ekey(a, b):
    return (a, b) if a < b else (b, a)


def _find(par, x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


class Board:
    def __init__(self):
        self.edge = {}                 # edge -> ('R', i) | ('C', i) | ('W',) | ('P', plus_pad)
        self.seats = [None, None]      # (IN+, IN-, OUT) pads of OPAMP 1, OPAMP 2

    def blocked(self):
        out = set()
        for s in self.seats:
            if s is not None:
                out.add(ekey(s[0], s[1]))
                out.add(ekey(s[1], s[2]))
        return out

    def used(self, kind):
        return {v[1] for v in self.edge.values() if v[0] == kind}

    def wires(self):
        return sum(1 for v in self.edge.values() if v[0] == 'W')

    def free_edges(self):
        b = self.blocked()
        return [e for e in EDGES if e not in self.edge and e not in b]


def _node_of(bd):
    par = {p: p for p in PADS}
    for a, b in COPPER:
        par[_find(par, a)] = _find(par, b)
    for e, v in bd.edge.items():
        if v[0] == 'W':
            par[_find(par, e[0])] = _find(par, e[1])
    return lambda p: '%d,%d' % _find(par, p)


def circuit(bd, inst):
    name = _node_of(bd)
    els, probe = [], None
    for e, v in sorted(bd.edge.items()):
        if v[0] == 'R':
            els.append(('R', name(e[0]), name(e[1]), float(inst['r'][v[1]])))
        elif v[0] == 'C':
            els.append(('C', name(e[0]), name(e[1]), float(inst['c'][v[1]])))
        elif v[0] == 'P':
            minus = e[0] if e[1] == v[1] else e[1]
            probe = (name(v[1]), name(minus))
    for k, s in enumerate(bd.seats):
        if s is not None:
            els.append(('OA', name(s[0]), name(s[1]), name(s[2]), inst['amps'][k]))
    return M.Circuit(els, name(GROUND), source=name(IN), probe=probe), probe, name


def score(bd, inst, coarse=True):
    """(ripple dB, stopband attenuation dB, ok): ok = the probe's - on ground and + on neither ground nor IN."""
    c, probe, name = circuit(bd, inst)
    if probe is None or probe[1] != name(GROUND) or probe[0] in (name(GROUND), name(IN)):
        return 99.0, 0.0, False
    h = c.response(F_CO if coarse else M.FREQS)
    if not np.all(np.isfinite(h)) or np.any(np.abs(h) == 0):
        return 99.0, 0.0, False
    g = 20 * np.log10(np.abs(h))
    npass = NPASS_CO if coarse else len(M.F_PASS)
    return float(np.max(np.abs(g[:npass] - g[0]))), float(g[0] - np.max(g[npass:])), True


def objective(rip, att, ok):
    return (att - 30.0 * max(0.0, rip - LIM)) if ok else -200.0


def mutate(bd, rng):
    bd = copy.deepcopy(bd)
    free = bd.free_edges()
    placed = list(bd.edge)
    u = rng.random()
    if u < 0.22 and free:                                     # add a piece
        kind = rng.choice('RC')
        pool = sorted(set(range(NR if kind == 'R' else NC)) - bd.used(kind))
        if pool:
            bd.edge[rng.choice(free)] = (kind, rng.choice(pool))
    elif u < 0.34 and placed:                                 # remove a piece or a wire
        e = rng.choice(placed)
        if bd.edge[e][0] != 'P':
            del bd.edge[e]
    elif u < 0.52 and placed and free:                        # move a block to a free edge (often a neighbour)
        e = rng.choice(placed)
        near = [f for f in free if set(f) & set(e)] or free
        f = rng.choice(near if rng.random() < 0.7 else free)
        v = bd.edge.pop(e)
        if v[0] == 'P':
            v = ('P', f[0] if rng.random() < 0.5 else f[1])
        bd.edge[f] = v
    elif u < 0.72 and placed:                                 # another value
        e = rng.choice(placed)
        v = bd.edge[e]
        if v[0] in 'RC':
            others = [x for x, w in bd.edge.items() if w[0] == v[0] and x != e]
            pool = sorted(set(range(NR if v[0] == 'R' else NC)) - bd.used(v[0]))
            if pool and (rng.random() < 0.6 or not others):
                bd.edge[e] = (v[0], rng.choice(pool))
            elif others:
                x = rng.choice(others)
                bd.edge[e], bd.edge[x] = bd.edge[x], bd.edge[e]
    elif u < 0.84 and free:                                   # a wire
        if bd.wires() < 8:
            bd.edge[rng.choice(free)] = ('W',)
    elif u < 0.92 and placed and free:                        # swap two edges' contents
        e, f = rng.choice(placed), rng.choice(placed)
        bd.edge[e], bd.edge[f] = bd.edge[f], bd.edge[e]
        for x in (e, f):
            if bd.edge[x][0] == 'P' and bd.edge[x][1] not in x:
                bd.edge[x] = ('P', x[0])
    else:                                                     # move an op-amp module
        k = rng.randrange(2)
        line = rng.choice(LINES)
        other = bd.seats[1 - k] or ()
        span = {ekey(line[0], line[1]), ekey(line[1], line[2])}
        if not (set(line) & set(other)) and not (span & set(bd.edge)):
            bd.seats[k] = line
    if bd.wires() > 8 or sum(1 for v in bd.edge.values() if v[0] == 'P') != 1:
        return None
    b = bd.blocked()
    if any(e in b for e in bd.edge):
        return None
    return bd


def to_json(bd):
    return {'edges': [[list(e[0]), list(e[1]), ['P', list(v[1])] if v[0] == 'P' else list(v)]
                      for e, v in sorted(bd.edge.items())],
            'seats': [None if s is None else [list(p) for p in s] for s in bd.seats]}


def from_json(d):
    bd = Board()
    for a, b, v in d['edges']:
        bd.edge[ekey(tuple(a), tuple(b))] = ('P', tuple(v[1])) if v[0] == 'P' else tuple(v)
    bd.seats = [None if s is None else tuple(tuple(p) for p in s) for s in d['seats']]
    return bd


def to_blocks(bd, inst):
    """The evaluator's board {block id: [pad names]} of a layout (the sealed device record's form)."""
    ids = M.piece_ids(inst)
    pad = lambda p: '%d,%d' % p
    out, w = {}, 0
    for e, v in sorted(bd.edge.items()):
        if v[0] == 'R':
            out[ids[v[1]]] = [pad(e[0]), pad(e[1])]
        elif v[0] == 'C':
            out[ids[NR + v[1]]] = [pad(e[0]), pad(e[1])]
        elif v[0] == 'W':
            w += 1
            out[f'W{w}'] = [pad(e[0]), pad(e[1])]
        else:
            minus = e[0] if e[1] == v[1] else e[1]
            out['PROBE'] = [pad(v[1]), pad(minus)]
    for k, s in enumerate(bd.seats):
        if s is not None:
            out[f'OA{k + 1}'] = [pad(p) for p in s]
    return out


def start_layout(inst):
    """filter_model.optimum's two-stage unity-gain Sallen-Key, laid out as the reference controller builds it."""
    q = M.optimum(inst, second=False)
    bd = Board()
    pad = lambda s: tuple(int(x) for x in s.split(','))
    for (r1, r2, c1, c2), (pa, pb, pc, pd, pe, w) in zip(
            (q['st1'], q['st2']), (('1,3', '1,4', '1,5', '2,4', '1,6', ('2,5', '2,4', '3,4', '3,5')),
                                   ('3,4', '4,4', '4,5', '5,4', '4,6', ('5,5', '5,4', '6,4', '6,5')))):
        bd.edge[ekey(pad(pa), pad(pb))] = ('R', int(r1))
        bd.edge[ekey(pad(pb), pad(pc))] = ('R', int(r2))
        bd.edge[ekey(pad(pb), pad(pd))] = ('C', int(c1))
        bd.edge[ekey(pad(pc), pad(pe))] = ('C', int(c2))
        for x, y in zip(w[:-1], w[1:]):
            bd.edge[ekey(pad(x), pad(y))] = ('W',)
    s1 = tuple(pad(x) for x in ('1,5', '2,5', '3,5'))
    s2 = tuple(pad(x) for x in ('4,5', '5,5', '6,5'))
    bd.seats = [s1, s2] if q['order'] == 0 else [s2, s1]
    bd.edge[ekey(pad('6,5'), pad('6,6'))] = ('P', pad('6,5'))
    return bd


def anneal(args):
    seed, model_seed, iters, start, T0 = args
    rng = random.Random(seed)
    inst = M.draw(model_seed)
    bd = from_json(start) if start else start_layout(inst)
    cur = objective(*score(bd, inst))
    best = None
    T1 = 0.02
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / iters)
        nb = mutate(bd, rng)
        if nb is None:
            continue
        r2, a2, ok2 = score(nb, inst)
        j2 = objective(r2, a2, ok2)
        if j2 >= cur or rng.random() < math.exp((j2 - cur) / T):
            bd, cur = nb, j2
            if ok2 and r2 <= LIM and (best is None or a2 > best[0]):
                fr, fa, fok = score(nb, inst, coarse=False)
                if fok and fr <= LIM:
                    best = (fa, fr, to_json(nb))
    return best


def search(model_seed, runs=RUNS, iters=ITERS, procs=None, rounds=ROUNDS):
    """Round 1: every run from the Sallen-Key start at T0 = 3 dB; each later round: run k from the k % 8-th best layout
    so far at T0 = 1 dB. Returns every round's results, best first. procs (default: the CPU count, at most runs) only
    sets how many runs go at once."""
    _require_pinned()
    procs = procs or max(1, min(runs, os.cpu_count() or 1))
    res = []
    for rnd in range(rounds):
        top = [r[2] for r in res[:8]]
        jobs = [(2654435761 % (k + 97) * 31 + k + 1000003 * rnd, model_seed, iters,
                 top[k % len(top)] if rnd and top else None, 1.0 if rnd and top else 3.0) for k in range(runs)]
        with Pool(procs) as p:                    # workers inherit PINNED_ENV (forked or spawned)
            res += [r for r in p.map(anneal, jobs) if r]
        res.sort(key=lambda r: -r[0])
    return res


def prune(bd, inst):
    """Take off, one at a time, every block whose removal keeps the passband within the limit and the attenuation
    from falling (blocks with a free end, islands): the same filter with fewer blocks."""
    rip, att, ok = score(bd, inst, coarse=False)
    changed = True
    while changed:
        changed = False
        for e in sorted(bd.edge, key=lambda e: bd.edge[e][0] != 'W'):
            if bd.edge[e][0] == 'P':
                continue
            nb = copy.deepcopy(bd)
            del nb.edge[e]
            r2, a2, ok2 = score(nb, inst, coarse=False)
            if ok2 and r2 <= LIM and a2 >= att - 1e-9:
                bd, rip, att, changed = nb, r2, max(att, a2), True
        for k in range(2):
            if bd.seats[k] is not None:
                nb = copy.deepcopy(bd)
                nb.seats[k] = None
                r2, a2, ok2 = score(nb, inst, coarse=False)
                if ok2 and r2 <= LIM and a2 >= att - 1e-9:
                    bd, rip, att, changed = nb, r2, max(att, a2), True
    return bd


def entry(model_seed, res, runs=RUNS, iters=ITERS, rounds=ROUNDS):
    """The reference entry of an instance (two reference values):
    A_ref_db, the success line: filter_model.optimum, the best two-stage unity-gain Sallen-Key (either module order),
    scored on the stopband; A_search_db, the ranking: the search's best layout (pruned), re-checked with the scoring
    code (benchmark.evaluation.filter.response_of)."""
    _require_pinned()
    from benchmark.evaluation import filter as EF
    inst = M.draw(model_seed)
    q = M.optimum(inst, second=False)
    bd = prune(from_json(res[0][2]), inst)
    rip, att, _ = score(bd, inst, coarse=False)
    blocks = to_blocks(bd, inst)
    rip_e, att_e = EF.response_of({'model_seed': model_seed}, blocks)
    assert abs(rip_e - rip) < 1e-9 and abs(att_e - att) < 1e-9 and rip_e <= LIM and att_e >= res[0][0] - 1e-9, \
        (res[0][:2], rip, att, rip_e, att_e)
    assert att_e >= q['A_star_db'], (att_e, q['A_star_db'])
    return {'A_ref_db': float(q['A_star_db']), 'ref_ripple_db': float(q['ripple_db']),
            'ref_design': {'order': int(q['order']), 'st1': [int(x) for x in q['st1']],
                           'st2': [int(x) for x in q['st2']], 'design': q['design'],
                           'A_order_db': [float(x) for x in q['A_order_db']]},
            'A_search_db': att_e, 'search_ripple_db': rip_e, 'layout': blocks,
            'model_seed': int(model_seed), 'model_version': M.MODEL_VERSION,
            'search': {'version': SEARCH_VERSION, 'runs': runs, 'iters': iters, 'rounds': rounds,
                       'top_db': [round(r[0], 4) for r in res[:8]]}}


def reference(model_seed):
    """The reference entry of an instance with the default settings, computed by the command line in a child process
    on the pinned arithmetic path."""
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "entry.json"
        subprocess.run([sys.executable, "-m", "benchmark.evaluation.filter_search", str(int(model_seed)),
                        "--entry", str(out)], cwd=root, env={**os.environ, **PINNED_ENV}, check=True)
        return json.loads(out.read_text())


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not pinned():                              # numpy must load with PINNED_ENV: start again with it
        sys.stdout.flush()
        os.execve(sys.executable, [sys.executable, "-m", "benchmark.evaluation.filter_search", *argv],
                  {**os.environ, **PINNED_ENV})
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('model_seeds', nargs='+', type=int)
    ap.add_argument('--runs', type=int, default=RUNS, help='the scoring reference uses the default')
    ap.add_argument('--iters', type=int, default=ITERS, help='the scoring reference uses the default')
    ap.add_argument('--rounds', type=int, default=ROUNDS, help='the scoring reference uses the default')
    ap.add_argument('--procs', type=int, default=None, help='runs at once (default: the CPU count, at most --runs)')
    ap.add_argument('--save', help='directory for every result layout (<model_seed>.json)')
    ap.add_argument('--from-saved', help='no search: the results saved by --save in this directory')
    ap.add_argument('--cache', action='store_true',
                    help="store the entries in the scoring cache (only with the default --runs, --iters, --rounds)")
    ap.add_argument('--entry', help='write the entry of the (single) model seed to this JSON file')
    a = ap.parse_args(argv)
    if (a.cache or a.entry) and (a.runs, a.iters, a.rounds) != (RUNS, ITERS, ROUNDS):
        ap.error('--cache and --entry need the default --runs, --iters and --rounds')
    if a.entry and len(a.model_seeds) != 1:
        ap.error('--entry takes one model seed')
    from benchmark.evaluation import filter as EF
    for ms in a.model_seeds:
        t = time.time()
        if a.from_saved:
            res = [tuple(r) for r in json.loads(Path(a.from_saved, f'{ms}.json').read_text())]
        else:
            res = search(ms, a.runs, a.iters, a.procs, a.rounds)
            if a.save:
                Path(a.save, f'{ms}.json').write_text(json.dumps(res[:40]))
        e = entry(ms, res, a.runs, a.iters, a.rounds)
        print(json.dumps({'model_seed': ms, 'A_ref_db': e['A_ref_db'], 'A_search_db': e['A_search_db'],
                          'search_ripple_db': e['search_ripple_db'], 'top_db': e['search']['top_db'],
                          's': round(time.time() - t)}), flush=True)
        if a.cache:
            EF._cache_write(EF._cache_key({'model_seed': ms}), e)
        if a.entry:
            Path(a.entry).write_text(json.dumps(e))


if __name__ == '__main__':
    main()

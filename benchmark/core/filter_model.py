"""Model of active_filter: a low-pass filter built on the circuit board from resistor and capacitor
pieces and two micropower op-amp modules, measured by a frequency-response analyzer. No Isaac.

Calibration model: the kit, the hidden values of
an instance (`draw`), one unity-gain Sallen-Key stage in closed form (`stage_h`), the exact two-stage cascade by nodal
analysis (`cascade`), and a two-stage search: branch and bound over stage pairs on the separable model (stage 1 sees
the 50 ohm source, stage 2 an ideal source; `search`, bounded at f_s), re-ranked with the exact cascade (`optimum`).
Its `optimum` (the best two-stage Sallen-Key) is the success line A_ref of an instance.

Scored numbers (MODEL_VERSION 2): the passband ripple is the largest deviation of the gain over
F_PASS (100 Hz - 1 kHz) from the gain at 100 Hz; the attenuation is the gain at 100 Hz minus the highest gain over
F_STOP (f_s = 2 kHz to 20 kHz), i.e. the smallest attenuation over the stopband, so a notch at f_s alone does not
raise it (version 1 scored the attenuation at f_s only).

Stage (R1, R2, C1, C2, amp): in -R1- a -R2- b, C1 from a to out, C2 from b to ground, amp + at b, - at out.

General circuit (`Circuit`, `response`): what is seated on the board as a netlist of board nodes. Elements:
    ('R', a, b, ohm), ('C', a, b, farad), ('W', a, b) a wire (the two nodes are one), ('OA', plus, minus, out, amp)
    an op-amp: a single pole A0 / (1 + s A0 / (2 pi GBW)) behind its open-loop output resistance ro, the current
    (A (V+ - V-) - Vout) / ro into its output (supplied from its own rails, referred to the ground rail); its inputs
    draw no current. The analyzer's source (EMF 1 V peak behind R_SRC) drives the IN node against ground; its probe
    (PROBE_R || PROBE_C) reads V(+) - V(-) wherever it is seated. The response is the probe voltage over the source EMF,
    solved per frequency by complex nodal analysis. A node with no conducting path to ground (only op-amp inputs or a
    detached island) carries no signal and is held at 0 V. Elements are stamped in their list order and nodes numbered
    by first appearance, so a netlist listed as `cascade` builds it reproduces that solution bit for bit.

Instance (`hidden_of`): the registry draws one integer `model_seed`; `draw(model_seed)` gives every piece's true value
and both op-amps exactly as the calibration drew its instances (its truncated normal factors cannot be written
as registry distributions).
"""
from __future__ import annotations

import numpy as np

MODEL_VERSION = 2          # 2: attenuation = the smallest over the stopband f_s..20 kHz

FP, FS, F_LO = 1000.0, 2000.0, 100.0
FS_HI = 20000.0            # the stopband's upper end
RIPPLE_DB = 0.5
F_PASS = np.geomspace(F_LO, FP, 120)
F_STOP = np.geomspace(FS, FS_HI, 241)       # f_s and 20 kHz included, about 1 % apart
FREQS = np.concatenate([F_PASS, F_STOP])
F_SEARCH = np.concatenate([F_PASS, [FS]])   # the two-stage search's separable bound (at f_s; re-ranked on FREQS)

R_NOM = (3300., 4700., 6800., 10000., 15000., 22000., 33000.)
C_NOM = (1.0e-9, 2.2e-9, 3.3e-9, 4.7e-9, 10e-9, 22e-9, 47e-9, 100e-9)
COPIES = 2
R_SRC = 50.0
PROBE_R, PROBE_C = 1.0e6, 15e-12

HIDDEN = dict(c_sigma=0.04, c_trunc=0.10, r_sigma=0.004, r_trunc=0.01,
              gbw=(70e3, 140e3), gbw_each=0.10, ro=(100., 400.), a0_db=(90., 110.))


def _trunc(rng, n, sigma, lim):
    x = rng.normal(0.0, sigma, n)
    while np.any(np.abs(x) > lim):
        bad = np.abs(x) > lim
        x[bad] = rng.normal(0.0, sigma, int(bad.sum()))
    return 1.0 + x


def pieces(r_nom=R_NOM, c_nom=C_NOM, copies=COPIES):
    """Nominal value of every piece (resistors then capacitors), in kit order."""
    return np.repeat(r_nom, copies), np.repeat(c_nom, copies)


def draw(seed, r_nom=R_NOM, c_nom=C_NOM, copies=COPIES, hidden=HIDDEN):
    rng = np.random.default_rng(seed)
    rn, cn = pieces(r_nom, c_nom, copies)
    r = rn * _trunc(rng, len(rn), hidden['r_sigma'], hidden['r_trunc'])
    c = cn * _trunc(rng, len(cn), hidden['c_sigma'], hidden['c_trunc'])
    g = np.exp(rng.uniform(*np.log(hidden['gbw'])))
    amps = []
    for _ in range(2):
        amps.append(dict(gbw=g * rng.uniform(1 - hidden['gbw_each'], 1 + hidden['gbw_each']),
                         ro=rng.uniform(*hidden['ro']), a0=10 ** (rng.uniform(*hidden['a0_db']) / 20)))
    return dict(seed=seed, r=r, c=c, r_nom=rn, c_nom=cn, amps=amps)


def nominal(inst):
    ideal = dict(gbw=np.inf, ro=0.0, a0=np.inf)
    return dict(inst, r=inst['r_nom'].copy(), c=inst['c_nom'].copy(), amps=[ideal, ideal])


def gain(amp, s):
    if not np.isfinite(amp['gbw']):
        return np.full_like(s, 1e12)
    return amp['a0'] / (1.0 + s * amp['a0'] / (2 * np.pi * amp['gbw']))


def stage_h(R1, R2, C1, C2, amp, f=FREQS, r_src=0.0, y_load=0.0):
    """Transfer of one stage; R1..C2 broadcast against f (shape (..., 1) against (F,))."""
    s = 2j * np.pi * f
    A = gain(amp, s)
    G1, G2 = 1.0 / (R1 + r_src), 1.0 / R2
    k = G2 / (G2 + s * C2)
    if amp['ro'] == 0.0:
        m = A * k / (1.0 + A)
    else:
        go = 1.0 / amp['ro']
        m = (s * C1 + A * k * go) / (go + s * C1 + y_load + A * go)
    va = G1 / (G1 + G2 + s * C1 - G2 * k - s * C1 * m)
    return m * va


def cascade(inst, st1, st2, f=FREQS, probe=True):
    """Exact nodal solution of source(50 ohm) -> stage 1 -> stage 2 -> probe. st = (r1, r2, c1, c2) piece indices."""
    s = 2j * np.pi * np.asarray(f)
    n = 7                                   # in, a1, b1, o1, a2, b2, o2
    Y = np.zeros((len(s), n, n), complex)
    I = np.zeros((len(s), n), complex)

    def adm(i, j, y):
        Y[:, i, i] += y
        if j is not None:
            Y[:, j, j] += y
            Y[:, i, j] -= y
            Y[:, j, i] -= y

    Y[:, 0, 0] += 1.0 / R_SRC
    I[:, 0] = 1.0 / R_SRC
    nodes = ((0, 1, 2, 3), (3, 4, 5, 6))
    for (inp, a, b, o), st, amp in zip(nodes, (st1, st2), inst['amps']):
        r1, r2, c1, c2 = st
        adm(inp, a, 1.0 / inst['r'][r1])
        adm(a, b, 1.0 / inst['r'][r2])
        adm(a, o, s * inst['c'][c1])
        adm(b, None, s * inst['c'][c2])
        A = gain(amp, s)
        ro = max(amp['ro'], 1e-3)
        Y[:, o, o] += (1.0 + A) / ro          # current (A (Vb - Vo) - Vo) / ro into o
        Y[:, o, b] -= A / ro
    if probe:
        adm(6, None, 1.0 / PROBE_R + s * PROBE_C)
    v = np.linalg.solve(Y, I[..., None])[..., 0]
    return v[:, 6]


def db(h):
    return 20 * np.log10(np.abs(h))


FLOOR_DB = -120.0          # the analyzer's floor (circuit_grid/filter_bench.FLOOR_DB): any smaller gain reads as this


def shown_db(h):
    """Gains (dB) as the analyzer shows them: at least FLOOR_DB, also for a response of exactly 0 (no path)."""
    with np.errstate(divide="ignore"):
        return np.maximum(db(h), FLOOR_DB)


def silent(h):
    """No response: the gain at F_LO (h[0]) reads at the floor, so the passband shows nothing to measure (a
    board without the probe seated gives h = 0)."""
    return bool(shown_db(np.asarray(h)[:1])[0] <= FLOOR_DB)


def score_curve(h):
    """(ripple dB, stopband attenuation dB) from a response on FREQS: the largest deviation over the passband from the
    gain at F_LO, and the gain at F_LO minus the highest gain over the stopband F_STOP (its smallest attenuation). Gains
    as the analyzer shows them (shown_db), so a dead circuit scores as its flat floor reading: ripple 0, attenuation 0
    (silent() tells it apart; the passband gate fails it)."""
    g = shown_db(h)
    rel = g[:len(F_PASS)] - g[0]
    return float(np.max(np.abs(rel))), float(g[0] - np.max(g[len(F_PASS):]))


def stage_candidates(inst):
    """All (r1, r2, c1, c2) piece tuples with distinct pieces, and their piece bitmasks."""
    nr, nc = len(inst['r']), len(inst['c'])
    r1, r2, c1, c2 = np.meshgrid(np.arange(nr), np.arange(nr), np.arange(nc), np.arange(nc), indexing='ij')
    keep = (r1 != r2) & (c1 != c2)
    idx = np.stack([r1[keep], r2[keep], c1[keep], c2[keep]], 1)
    mask = (np.left_shift(1, idx[:, 0]) | np.left_shift(1, idx[:, 1])
            | np.left_shift(1, nr + idx[:, 2]) | np.left_shift(1, nr + idx[:, 3])).astype(np.int64)
    return idx, mask


def stage_curves(inst, idx, which, r_src=0.0):
    R1 = inst['r'][idx[:, 0]][:, None]
    R2 = inst['r'][idx[:, 1]][:, None]
    C1 = inst['c'][idx[:, 2]][:, None]
    C2 = inst['c'][idx[:, 3]][:, None]
    return db(stage_h(R1, R2, C1, C2, inst['amps'][which], f=F_SEARCH, r_src=r_src))


def search(inst, ripple_lim=RIPPLE_DB, top=8, idx=None, mask=None, exclude=None):
    """Branch and bound over stage pairs on the separable model (stage 1 sees the 50 ohm source); returns the top
    pairs [(att, ripple, i, j)] by separable attenuation with ripple <= ripple_lim.
    exclude: optional (code, key) with code an int array over the stage candidates and key a pair of codes; pairs with
    (code[i], code[j]) == key are skipped (the distinct second best of `optimum`). None is the calibration's search."""
    if idx is None:
        idx, mask = stage_candidates(inst)
    g1 = stage_curves(inst, idx, 0, r_src=R_SRC)
    g2 = stage_curves(inst, idx, 1)
    np_ = len(F_PASS)
    rel1 = g1[:, :np_] - g1[:, :1]
    rel2 = g2[:, :np_] - g2[:, :1]
    att1 = g1[:, 0] - g1[:, -1]
    att2 = g2[:, 0] - g2[:, -1]
    # no other stage can make up more than ~15 dB of droop or peaking (kit Q stays below ~5)
    sane1 = np.max(np.abs(rel1), axis=1) <= 15.0
    sane2 = np.max(np.abs(rel2), axis=1) <= 15.0
    o1 = np.argsort(-att1)
    o1 = o1[sane1[o1]]
    o2 = np.argsort(-att2)
    o2 = o2[sane2[o2]]
    att2s, rel2s, mask2s = att2[o2], rel2[o2], mask[o2]
    code2s = None if exclude is None else exclude[0][o2]
    coarse = np.r_[np.arange(0, np_, 8), np_ - 1]
    rel2c = rel2s[:, coarse]
    found = []
    thr = -np.inf
    for i in o1:
        if att1[i] + att2s[0] <= thr:
            break
        need = thr - att1[i]
        n = int(np.searchsorted(-att2s, -need, side='left')) if np.isfinite(need) else len(att2s)
        if n == 0:
            continue
        ok = np.max(np.abs(rel1[i, coarse][None, :] + rel2c[:n]), axis=1) <= ripple_lim
        ok &= (mask2s[:n] & mask[i]) == 0
        if exclude is not None and exclude[0][i] == exclude[1][0]:
            ok &= code2s[:n] != exclude[1][1]
        js = np.nonzero(ok)[0]
        if len(js) == 0:
            continue
        rip = np.max(np.abs(rel1[i][None, :] + rel2s[js]), axis=1)
        js = js[rip <= ripple_lim][:top]
        rip = rip[rip <= ripple_lim][:top]
        for j, rp in zip(js, rip):
            found.append((att1[i] + att2s[j], float(rp), int(i), int(o2[j])))
        found.sort(reverse=True)
        del found[top:]
        if len(found) >= top:
            thr = found[-1][0]
    return [(a, r, tuple(idx[i]), tuple(idx[j])) for a, r, i, j in found], idx, mask


def exact(inst, st1, st2):
    return score_curve(cascade(inst, st1, st2))


# --- A* -----------------------------------------------------------------------------------------------------------
TOP = 20                     # separable candidates re-ranked with the exact cascade (the calibration's study)


def best_exact(inst, top):
    """Best exact (att, ripple, st1, st2) among separable candidates with exact ripple within the limit (the
    calibration's filter_study.best_exact)."""
    out = []
    for _, _, s1, s2 in top:
        rip, att = exact(inst, s1, s2)
        if rip <= RIPPLE_DB:
            out.append((att, rip, [int(x) for x in s1], [int(x) for x in s2]))
    return max(out) if out else None


def nominal_codes(inst, idx):
    """An integer per stage candidate naming its nominal design (R1, R2, C1, C2 nominal values, not the copies)."""
    rk = np.searchsorted(np.unique(inst['r_nom']), inst['r_nom'])
    ck = np.searchsorted(np.unique(inst['c_nom']), inst['c_nom'])
    nr, nc = int(rk.max()) + 1, int(ck.max()) + 1
    return ((rk[idx[:, 0]] * nr + rk[idx[:, 1]]) * nc + ck[idx[:, 2]]) * nc + ck[idx[:, 3]]


def design_text(inst, st1, st2):
    """'R1/R2/C1/C2 | R1/R2/C1/C2' of a design in nominal values (k ohm, nF)."""
    def one(st):
        r1, r2, c1, c2 = st
        return '%g/%gk %g/%gn' % (inst['r_nom'][r1] / 1e3, inst['r_nom'][r2] / 1e3, inst['c_nom'][c1] * 1e9,
                                   inst['c_nom'][c2] * 1e9)
    return one(st1) + ' | ' + one(st2)


def optimum(inst, second=True):
    """A* of an instance over both orders of the two op-amp modules (the modules are
    kit pieces too): the better of optimum_order(inst) (module 1 in stage 1, the calibration's search) and of the same
    search with the modules swapped. Adds 'order' (0: module 1 in stage 1, 1: module 2 in stage 1), 'A_order_db' (A*
    of each order); 'second_db' / 'gap_db' (diagnostics) are those of the winning order."""
    q = [optimum_order(inst, second), optimum_order(dict(inst, amps=inst['amps'][::-1]), second)]
    k = 0 if q[0]['A_star_db'] >= q[1]['A_star_db'] else 1
    return dict(q[k], order=k, A_order_db=[q[0]['A_star_db'], q[1]['A_star_db']])


def optimum_order(inst, second=True):
    """A* with module 1 in stage 1: {'A_star_db', 'ripple_db', 'st1', 'st2', 'design'}
    and, with second, the best design whose nominal values differ from A*'s ('second_db', 'second_st1',
    'second_st2', 'gap_db' = A* - second)."""
    idx, mask = stage_candidates(inst)
    top, _, _ = search(inst, top=TOP, idx=idx, mask=mask)
    best = best_exact(inst, top)
    if best is None:
        raise ValueError('no stage pair meets the passband limit')
    out = {'A_star_db': float(best[0]), 'ripple_db': float(best[1]), 'st1': best[2], 'st2': best[3],
           'design': design_text(inst, best[2], best[3])}
    if second:
        code = nominal_codes(inst, idx)
        i = int(np.flatnonzero((idx == best[2]).all(1))[0])
        j = int(np.flatnonzero((idx == best[3]).all(1))[0])
        top2, _, _ = search(inst, top=TOP, idx=idx, mask=mask, exclude=(code, (code[i], code[j])))
        sec = best_exact(inst, top2)
        out.update({'second_db': None if sec is None else float(sec[0]),
                    'second_st1': None if sec is None else sec[2], 'second_st2': None if sec is None else sec[3],
                    'gap_db': None if sec is None else float(best[0] - sec[0])})
    return out


# --- instance ----------------------------------------------------------------------------------------------------
def hidden_of(hidden):
    """The model instance of a registry instance's hidden values ({'model_seed': int})."""
    return draw(int(hidden['model_seed']))


def piece_ids(inst=None):
    """Block id of every piece in kit order: resistors 'R<value>_<copy>', capacitors 'C<value>_<copy>'
    (e.g. R4k7_1, C2n2_2)."""
    rn, cn = pieces() if inst is None else (inst['r_nom'], inst['c_nom'])
    out = []
    for values, prefix, label in ((rn, 'R', r_label), (cn, 'C', c_label)):
        seen = {}
        for v in values:
            seen[v] = seen.get(v, 0) + 1
            out.append(f'{prefix}{label(v)}_{seen[v]}')
    return out


def r_label(ohm):
    """Printed value of a resistor: 3k3, 4k7, 10k, 22k."""
    k = ohm / 1e3
    if k >= 10:
        return f'{k:g}k'
    whole = int(k)
    frac = round((k - whole) * 10)
    return f'{whole}k{frac}' if frac else f'{whole}k'


def c_label(farad):
    """Printed value of a capacitor: 1n0, 2n2, 10n, 100n."""
    n = farad * 1e9
    if n >= 10:
        return f'{n:g}n'
    whole = int(n)
    frac = round((n - whole) * 10)
    return f'{whole}n{frac}'


def piece_values(inst):
    """{piece block id: true value (ohm or farad)}."""
    ids = piece_ids(inst)
    vals = list(inst['r']) + list(inst['c'])
    return {k: float(v) for k, v in zip(ids, vals)}


# --- general circuit ------------------------------------------------------------------------------------------------
class _UF:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class Circuit:
    """A netlist: elements (see the module docstring), the ground node, the analyzer source node (or None) and the
    probe's (+ node, - node) (or None)."""

    def __init__(self, elements, ground, source=None, probe=None):
        self.elements = [tuple(e) for e in elements]
        self.ground, self.source, self.probe = ground, source, None if probe is None else tuple(probe)
        uf = _UF()
        uf.find(ground)
        if source is not None:
            uf.find(source)
        for e in self.elements:
            for n in e[1:3] if e[0] in ('R', 'C', 'W') else e[1:4] if e[0] == 'OA' else ():
                uf.find(n)
            if e[0] == 'W':
                uf.union(e[1], e[2])
        if self.probe is not None:
            uf.find(self.probe[0]), uf.find(self.probe[1])
        self.uf = uf
        g = uf.find(ground)
        # conducting paths to ground: R, C, the probe, the source (to ground), an op-amp output (to ground)
        reach = _UF()
        reach.find(g)
        for e in self.elements:
            if e[0] in ('R', 'C'):
                reach.union(uf.find(e[1]), uf.find(e[2]))
            elif e[0] == 'OA':
                reach.union(g, uf.find(e[3]))
        if source is not None:
            reach.union(g, uf.find(source))
        if self.probe is not None:
            reach.union(uf.find(self.probe[0]), uf.find(self.probe[1]))
        live = reach.find(g)
        order = []
        for n in ([source] if source is not None else []) + [n for e in self.elements for n in _nodes(e)] + \
                list(self.probe or ()):
            r = uf.find(n)
            if r != g and r not in order and reach.find(r) == live:
                order.append(r)
        self.index = {r: k for k, r in enumerate(order)}

    def node(self, name):
        """Matrix index of a board node (None: ground or no signal)."""
        return self.index.get(self.uf.find(name))

    def same(self, a, b):
        return self.uf.find(a) == self.uf.find(b)

    def response(self, f, amps_override=None):
        """Probe reading over the source EMF at the frequencies f (complex array; zeros without a probe or source)."""
        f = np.asarray(f, float)
        s = 2j * np.pi * f
        n = len(self.index)
        out = np.zeros(len(s), complex)
        if self.probe is None or self.source is None or n == 0 or self.node(self.source) is None:
            return out
        Y = np.zeros((len(s), n, n), complex)
        I = np.zeros((len(s), n), complex)

        def adm(i, j, y):
            if i is not None:
                Y[:, i, i] += y
            if j is not None:
                Y[:, j, j] += y
                if i is not None:
                    Y[:, i, j] -= y
                    Y[:, j, i] -= y

        k = self.node(self.source)
        Y[:, k, k] += 1.0 / R_SRC
        I[:, k] = 1.0 / R_SRC
        for e in self.elements:
            kind = e[0]
            if kind == 'R':
                adm(self.node(e[1]), self.node(e[2]), 1.0 / e[3])
            elif kind == 'C':
                adm(self.node(e[1]), self.node(e[2]), s * e[3])
            elif kind == 'OA':
                p, m, o = self.node(e[1]), self.node(e[2]), self.node(e[3])
                if o is None:                         # OUT on ground (or on a dead node): it drives nothing there,
                    continue                          # and its inputs draw no current
                amp = e[4]
                A = gain(amp, s)
                ro = max(amp['ro'], 1e-3)
                if m is not None and m == o:          # follower: (1 + A) / ro as the cascade stamps it
                    Y[:, o, o] += (1.0 + A) / ro
                else:
                    Y[:, o, o] += 1.0 / ro
                    if m is not None:
                        Y[:, o, m] += A / ro
                if p is not None:
                    Y[:, o, p] -= A / ro
        pp, pm = self.node(self.probe[0]), self.node(self.probe[1])
        if pp is None and pm is not None:
            adm(pm, None, 1.0 / PROBE_R + s * PROBE_C)
        else:
            adm(pp, pm, 1.0 / PROBE_R + s * PROBE_C)
        try:
            v = np.linalg.solve(Y, I[..., None])[..., 0]
        except np.linalg.LinAlgError:
            # a singular frequency (an ideal loop that cancels exactly): a leak of 1e-12 S from every node
            v = np.linalg.solve(Y + 1e-12 * np.eye(n), I[..., None])[..., 0]
        vp = v[:, pp] if pp is not None else 0.0
        vm = v[:, pm] if pm is not None else 0.0
        return out + vp - vm


def _nodes(e):
    if e[0] in ('R', 'C', 'W'):
        return e[1:3]
    if e[0] == 'OA':
        return e[1:4]
    return ()


def cascade_circuit(inst, st1, st2, probe=True):
    """The two-stage cascade as a Circuit listed in `cascade`'s stamping order (follower wires OUT - IN-)."""
    els = []
    nodes = (('in', 'a1', 'b1', 'o1'), ('o1', 'a2', 'b2', 'o2'))
    for (inp, a, b, o), st, amp in zip(nodes, (st1, st2), inst['amps']):
        r1, r2, c1, c2 = st
        els += [('R', inp, a, float(inst['r'][r1])), ('R', a, b, float(inst['r'][r2])),
                ('C', a, o, float(inst['c'][c1])), ('C', b, 'gnd', float(inst['c'][c2])),
                ('W', o, o + 'm'), ('OA', b, o + 'm', o, amp)]
    return Circuit(els, 'gnd', source='in', probe=('o2', 'gnd') if probe else None)


def evaluate(circuit, f=FREQS):
    """(ripple dB, stopband attenuation dB) of a circuit's probe reading (the evaluator's numbers)."""
    return score_curve(circuit.response(f))

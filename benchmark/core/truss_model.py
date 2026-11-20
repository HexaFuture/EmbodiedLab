"""Truss bench mechanics (truss_bridge v2), headless: the pegboard, the bridge's failure load and the bar tester.

The board is a pegboard: a grid of holes, PITCH_MM apart, COLS columns along the span and ROWS rows across it (bench
frame: x along the span from support A, y away from the load direction; row 0 is the deck line through A, C and B).
A, B and C are fixed pins in their holes; the agent inserts up to N_LOOSE_PINS loose pins into other holes. A hole is
named by its column and row, 'x<col>y<row>' (e.g. 'x1y-1'); the fixed pins' holes are named 'A', 'C', 'B'. A bar is
in the bridge when both its eyes sit on pins standing in two holes; the pins are the joints.

The bridge is analysed as a pin-jointed planar truss (linear elastic, small displacements, direct stiffness method; two
or more bars on the same two pins act in parallel): A holds x and y, B holds y only (roller), and the test frame pulls
the load pin C in -y. Each bar carries an axial force proportional to the load; it fails in tension at its tensile
strength T and in compression at its strut load, the Perry-Robertson combination of its squash load N and its
pin-ended Euler load PE(L) = P100 (100 / L)^2 with the imperfection factor eta = ETA_PER_100MM L / 100 mm:
    Pc = s - sqrt(s^2 - N PE),  s = (N + (1 + eta) PE) / 2
(a short bar crushes near N, a long one buckles near PE, between the two the initial crookedness costs strength).
The bridge fails when its first bar fails: its capacity is the smallest load at which a bar reaches its limit. A
bridge that is a mechanism (singular stiffness), or that leaves out A, B or C, carries nothing.

All bar kinds (colours) have the same section, so their axial stiffness follows the modulus that sets the Euler load:
EA = EA_PER_P100 * P100. It only matters for redundant (statically indeterminate) bridges.

The optimum P* (the private target) is in truss_optimum (MILP over the ground structure of the board).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

PITCH_MM = 50.0
COLS = range(0, 5)                    # x = 0 .. 200 mm
ROWS = range(-2, 3)                   # y = -100 .. +100 mm (row 0: the deck line)
FIXED = {'A': (0, 0), 'C': (2, 0), 'B': (4, 0)}
SUPPORTS = {'A': (True, True), 'B': (False, True)}       # pin -> (x held, y held)
LOAD_PIN = 'C'
N_LOOSE_PINS = 4
STACK = 6                             # eyes one pin carries (its straight part / the bar thickness)
BAR_LENGTHS_MM = {50: 50.0, 71: math.hypot(50.0, 50.0), 100: 100.0, 112: math.hypot(50.0, 100.0)}  # label -> spacing
LABELS = tuple(sorted(BAR_LENGTHS_MM))
KINDS = ('blue', 'orange', 'green', 'white')     # four printed materials
KIT_PER_KIND = 5
KIT = {(kind, label): KIT_PER_KIND for kind in KINDS for label in LABELS}
LINEAR_DENSITY_G_PER_MM = {'blue': 0.055, 'orange': 0.065, 'green': 0.075, 'white': 0.085}
MASS_LIMIT_G = 50.0
EA_PER_P100 = 1351.0                 # N per N of 100 mm Euler load (5 mm thick strip of any of the materials)
ETA_PER_100MM = 0.15                 # Perry-Robertson imperfection factor at 100 mm (proportional to length)
LENGTH_TOL_MM = 1.5                  # a bar spans two holes when their distance is within this of its eye spacing
PIN_R_MM, BAR_HALF_WIDTH_MM, BAR_CLEARANCE_MM = 4.0, 6.0, 1.5


def hole_name(col, row):
    for name, cr in FIXED.items():
        if cr == (col, row):
            return name
    return f'x{col}y{row}'


HOLES = {hole_name(c, r): (c, r) for r in ROWS for c in COLS}
HOLE_XY_MM = {name: (c * PITCH_MM, r * PITCH_MM) for name, (c, r) in HOLES.items()}
LOOSE_HOLES = tuple(n for n in HOLES if n not in FIXED)


def bar_mass_g(kind, label):
    return LINEAR_DENSITY_G_PER_MM[kind] * BAR_LENGTHS_MM[label]


def label_for(a, b):
    """The printed length of a bar spanning holes a and b, or None."""
    d = math.dist(HOLE_XY_MM[a], HOLE_XY_MM[b])
    for label, length in BAR_LENGTHS_MM.items():
        if abs(d - length) <= LENGTH_TOL_MM:
            return label
    return None


def blockers(a, b):
    """Holes a pin in which a straight bar from hole a to hole b would run over (the bar could not lie on its pins)."""
    p, q = np.asarray(HOLE_XY_MM[a], float), np.asarray(HOLE_XY_MM[b], float)
    d = q - p
    out = []
    for name, r in HOLE_XY_MM.items():
        if name in (a, b):
            continue
        t = float(np.dot(np.asarray(r) - p, d) / np.dot(d, d))
        if 0.0 < t < 1.0 and np.linalg.norm(p + t * d - r) < PIN_R_MM + BAR_HALF_WIDTH_MM + BAR_CLEARANCE_MM:
            out.append(name)
    return tuple(out)


def links():
    """{(hole, hole): label} of every hole pair (sorted names) a bar can span that no fixed pin is in the way of."""
    out = {}
    for a, b in itertools.combinations(sorted(HOLES), 2):
        label = label_for(a, b)
        if label is not None and not set(blockers(a, b)) & set(FIXED):
            out[(a, b)] = label
    return out


def euler_load(P100_N, label):
    return float(P100_N) * (100.0 / BAR_LENGTHS_MM[label]) ** 2


def strut_load(P100_N, N_N, label):
    """The compression load (N) at which a bar of this length fails (Perry-Robertson, see the module docstring)."""
    PE = euler_load(P100_N, label)
    N = float(N_N)
    eta = ETA_PER_100MM * BAR_LENGTHS_MM[label] / 100.0
    s = 0.5 * (N + (1.0 + eta) * PE)
    return s - math.sqrt(max(s * s - N * PE, 0.0))


def bar_limits(T_N, P100_N, N_N, label):
    """(tension limit, compression limit) of one bar, N."""
    return float(T_N), strut_load(P100_N, N_N, label)


def unit_forces(members, stiffness):
    """Axial forces (N, tension +) under a 1 N pull on the load pin, for members [(hole, hole)] with axial stiffnesses
    EA (N); None when the truss is a mechanism or leaves out a support or the load pin."""
    nodes = sorted({n for m in members for n in m})
    if not set(SUPPORTS) | {LOAD_PIN} <= set(nodes):
        return None
    index = {n: k for k, n in enumerate(nodes)}
    K = np.zeros((2 * len(nodes), 2 * len(nodes)))
    rows = []
    for (a, b), ea in zip(members, stiffness):
        d = np.subtract(HOLE_XY_MM[b], HOLE_XY_MM[a]) / 1000.0
        length = float(np.linalg.norm(d))
        if length <= 0.0:
            return None
        v = np.concatenate([-d, d]) / length
        dofs = [2 * index[a], 2 * index[a] + 1, 2 * index[b], 2 * index[b] + 1]
        k = float(ea) / length
        K[np.ix_(dofs, dofs)] += k * np.outer(v, v)
        rows.append((dofs, v, k))
    fixed = {2 * index[n] + axis for n, held in SUPPORTS.items() for axis in (0, 1) if held[axis]}
    free = [i for i in range(len(K)) if i not in fixed]
    Kf = K[np.ix_(free, free)]
    w = np.linalg.eigvalsh(Kf)
    if w[0] <= 1e-9 * w[-1]:
        return None
    F = np.zeros(len(K))
    F[2 * index[LOAD_PIN] + 1] = -1.0
    u = np.zeros(len(K))
    u[free] = np.linalg.solve(Kf, F[free])
    return np.array([k * float(v @ u[dofs]) for dofs, v, k in rows])


def capacity(bars):
    """Failure load (N) of a bridge of bars [{'pins': (a, b), 'label', 'T_N', 'P100_N', 'N_N'}]: the load at which its
    first bar fails; 0 for a mechanism. Also returns the governing bar's index and mode. A bar may instead give its
    compression limit as 'C_N' (a measured or estimated table; then 'N_N' is not needed; 'P100_N' still sets EA)."""
    if not bars:
        return 0.0, None
    f = unit_forces([tuple(b['pins']) for b in bars], [EA_PER_P100 * b['P100_N'] for b in bars])
    if f is None:
        return 0.0, None
    best, which = math.inf, None
    for i, (fi, b) in enumerate(zip(f, bars)):
        if 'C_N' in b:
            tension, compression = float(b['T_N']), float(b['C_N'])
        else:
            tension, compression = bar_limits(b['T_N'], b['P100_N'], b['N_N'], b['label'])
        if fi > 1e-9 and tension / fi < best:
            best, which = tension / fi, (i, 'tension')
        elif fi < -1e-9 and compression / -fi < best:
            best, which = compression / -fi, (i, 'compression')
    return (best, which) if which is not None else (0.0, None)


def tester_peak(mode, T_N, P100_N, N_N, label):
    """The force (N) at which a bar fails in the tester: its tensile strength, or its strut load at its length."""
    tension, compression = bar_limits(T_N, P100_N, N_N, label)
    return tension if mode == 'tension' else compression

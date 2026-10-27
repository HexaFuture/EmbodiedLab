"""Circuit bench solver (kit circuits), headless.

The black box holds at most one capacitor, so the state is the capacitor
voltage x = V(plus plate) - V(minus plate). For every switch state the
resistive network (box resistors, capacitor leakage, the selected R_s, the
logger CH1 input resistance and G_min from every node to ground) is solved by
modified nodal analysis with the capacitor as a voltage source. That gives,
exactly and linearly, dx/dt = a*x + b*u and each node voltage as p*x + q*u,
with u the supply voltage while CHARGE and 0 otherwise. A tick advances x with
the exact exponential of this scalar system, so the result does not depend on
the tick length.

Switch state: (pair, rs, mode) with None for a selector between detents. The
selected pair's first letter is the plus terminal: CHARGE connects supply ->
R_s -> plus and minus -> ground; DISCHARGE replaces the supply by a short, so
the capacitor discharges through R_s; OFF and a selector in transition leave
R_s open. A PAIR in transition connects nothing to the box, and the logger then
reads 0 V. CH1 = V(plus) - V(minus) with the 10 MOhm input across the pair;
CH2 = V(supply side of R_s) - V(plus), an isolated channel, 0 V with R_s open.
"""
from __future__ import annotations

import math

import numpy as np

PAIRS = ('A-B', 'B-C', 'A-C')
RS_NOMINAL_OHM = (1000, 10000, 100000)
MODES = ('CHARGE', 'OFF', 'DISCHARGE')
SOURCE_V = 5.0
LOGGER_INPUT_OHM = 1.0e7
LEAK_OHM = 1.0e10
G_MIN_S = 1.0e-12

_NODES = ('A', 'B', 'C', 'S')      # S: supply side of R_s; ground is the reference

# Box networks. Elements: (kind, name, node, node); the capacitor's
# first node is its plus plate.
TOPOLOGIES = {
    'single_c': (('C', 'C1', 'A', 'B'),),
    'single_r': (('R', 'R1', 'A', 'B'),),
    'T1': (('R', 'R1', 'A', 'C'), ('R', 'R2', 'C', 'B')),
    'T2': (('R', 'R1', 'A', 'C'), ('C', 'C1', 'C', 'B')),
    'T3': (('C', 'C1', 'A', 'C'), ('R', 'R1', 'C', 'B')),
    'T4': (('R', 'R1', 'A', 'C'), ('C', 'C1', 'A', 'C'), ('R', 'R2', 'C', 'B')),
    'T5': (('R', 'R1', 'A', 'C'), ('R', 'R2', 'C', 'B'), ('C', 'C1', 'C', 'B')),
    'T6': (('R', 'R1', 'A', 'C'), ('C', 'C1', 'C', 'B'), ('R', 'R2', 'A', 'B')),
}


def box_network(topology, values):
    """Elements (kind, node, node, value) of a named topology; values maps R1/R2/C1 to ohm/farad."""
    elements = []
    for kind, name, p, n in TOPOLOGIES[topology]:
        value = float(values[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite')
        elements.append((kind, p, n, value))
    return tuple(elements)


def pair_terminals(pair):
    plus, minus = pair.split('-')
    return plus, minus


class Linear:
    """dx/dt = a*x + b*u and node voltages V = p*x + q*u for one switch state."""
    __slots__ = ('a', 'b', 'p', 'q', 'resistors', 'sources')

    def __init__(self, a, b, p, q, resistors, sources):
        self.a, self.b, self.p, self.q = a, b, p, q
        self.resistors = resistors      # (node, node, conductance) as solved, for audits
        self.sources = sources          # names of the MNA voltage sources, in order

    @property
    def tau_s(self):
        return math.inf if self.a == 0 else -1.0 / self.a


class CircuitModel:
    """Electrical model of one bench instance: box network, actual R_s values, supply and logger load."""

    def __init__(self, elements, rs_actual_ohm, *, source_v=SOURCE_V, logger_input_ohm=LOGGER_INPUT_OHM,
                 leak_ohm=LEAK_OHM, g_min_s=G_MIN_S):
        capacitors = [e for e in elements if e[0] == 'C']
        if len(capacitors) > 1:
            raise ValueError('the bench solver supports at most one capacitor')
        if set(rs_actual_ohm) != set(RS_NOMINAL_OHM):
            raise ValueError('actual values required for every R_s detent')
        self.elements = tuple(elements)
        self.capacitor = capacitors[0] if capacitors else None
        self.rs_actual_ohm = {int(k): float(v) for k, v in rs_actual_ohm.items()}
        self.source_v = float(source_v)
        self.logger_input_ohm = float(logger_input_ohm)
        self.leak_ohm = float(leak_ohm)
        self.g_min_s = float(g_min_s)
        self._cache = {}

    def linear(self, switch):
        switch = tuple(switch)
        if switch not in self._cache:
            self._cache[switch] = self._solve(*switch)
        return self._cache[switch]

    def _solve(self, pair, rs, mode):
        if pair is not None and pair not in PAIRS or rs is not None and rs not in RS_NOMINAL_OHM \
                or mode is not None and mode not in MODES:
            raise ValueError(f'invalid switch state {(pair, rs, mode)!r}')
        index = {name: i for i, name in enumerate(_NODES)}
        resistors = [(n, None, self.g_min_s) for n in _NODES]
        sources = []                    # (name, plus node, minus node or None, basis)
        for kind, p, n, value in self.elements:
            if kind == 'R':
                resistors.append((p, n, 1.0 / value))
            elif self.leak_ohm != math.inf:
                resistors.append((p, n, 1.0 / self.leak_ohm))
        if self.capacitor is not None:
            sources.append(('capacitor', self.capacitor[1], self.capacitor[2], 'x'))
        if pair is not None:
            plus, minus = pair_terminals(pair)
            if self.logger_input_ohm != math.inf:
                resistors.append((plus, minus, 1.0 / self.logger_input_ohm))
            sources.append(('ground', minus, None, None))
            if mode in ('CHARGE', 'DISCHARGE') and rs is not None:
                resistors.append(('S', plus, 1.0 / self.rs_actual_ohm[rs]))
                sources.append(('supply', 'S', None, 'u' if mode == 'CHARGE' else None))
        size = len(_NODES) + len(sources)
        matrix = np.zeros((size, size))
        for p, n, g in resistors:
            i = index[p]
            matrix[i, i] += g
            if n is not None:
                j = index[n]
                matrix[j, j] += g
                matrix[i, j] -= g
                matrix[j, i] -= g
        rhs = np.zeros((size, 2))       # columns: x = 1, u = 1
        for k, (_, p, n, basis) in enumerate(sources):
            row = len(_NODES) + k
            # Extra unknown: current entering the source at its plus node.
            matrix[index[p], row] += 1.0
            matrix[row, index[p]] += 1.0
            if n is not None:
                matrix[index[n], row] -= 1.0
                matrix[row, index[n]] -= 1.0
            if basis == 'x':
                rhs[row, 0] = 1.0
            elif basis == 'u':
                rhs[row, 1] = 1.0
        solution = np.linalg.solve(matrix, rhs)
        p = dict(zip(_NODES, solution[:len(_NODES), 0]))
        q = dict(zip(_NODES, solution[:len(_NODES), 1]))
        a = b = 0.0
        if self.capacitor is not None:
            # The source current unknown is the current leaving its plus node into the
            # source, i.e. the current charging the plus plate.
            capacitance = self.capacitor[3]
            a = float(solution[len(_NODES), 0]) / capacitance
            b = float(solution[len(_NODES), 1]) / capacitance
        return Linear(a, b, p, q, tuple(resistors), tuple(s[0] for s in sources))

    def drive_v(self, mode):
        return self.source_v if mode == 'CHARGE' else 0.0

    def node_voltages(self, switch, x):
        lin = self.linear(switch)
        u = self.drive_v(switch[2])
        return {node: lin.p[node] * x + lin.q[node] * u for node in _NODES}

    def channels(self, switch, x):
        """(CH1, CH2) in volts, noise-free and unquantised."""
        pair, rs, mode = switch
        if pair is None:
            return 0.0, 0.0
        v = self.node_voltages(switch, x)
        plus, minus = pair_terminals(pair)
        ch2 = v['S'] - v[plus] if mode in ('CHARGE', 'DISCHARGE') and rs is not None else 0.0
        return v[plus] - v[minus], ch2

    def advance(self, switch, x, dt_s):
        """Exact capacitor voltage after dt_s with the switch state held."""
        if self.capacitor is None or dt_s == 0:
            return x
        lin = self.linear(switch)
        u = self.drive_v(switch[2])
        ad = lin.a * dt_s
        gain = math.expm1(ad) / lin.a if lin.a != 0 else dt_s
        return math.exp(ad) * x + gain * lin.b * u

    def time_constant_s(self, pair, rs, mode='CHARGE'):
        """Capacitor time constant with the given selectors; inf without a capacitor."""
        return self.linear((pair, rs, mode)).tau_s if self.capacitor is not None else math.inf


def finite_taus(topology, values, rs=10000, *, limit_s=1.0e3):
    """Time constants per pair in CHARGE with nominal R_s; pairs whose time constant exceeds limit_s
    (the capacitor is not on the path and only G_min or the leakage discharges it) are omitted."""
    model = CircuitModel(box_network(topology, values), {r: float(r) for r in RS_NOMINAL_OHM})
    taus = {pair: model.time_constant_s(pair, rs) for pair in PAIRS}
    return {pair: tau for pair, tau in taus.items() if tau <= limit_s}

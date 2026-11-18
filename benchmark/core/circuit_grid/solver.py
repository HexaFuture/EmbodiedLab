"""General modified-nodal-analysis (MNA) solver of the circuit board (no Isaac, no numpy-free shortcuts).

A Netlist holds two-terminal elements between named nodes:
    resistor(a, b, ohm)              linear
    vsource(a, b, volts)             V(a) - V(b) = volts; its current (a -> b through the source) is an unknown
    isource(a, b, amps)              `amps` flow from a through the source to b (they leave the source at b)
    capacitor(a, b, farad, v0)       open in a DC solve; in a transient step backward Euler: G = C/dt in parallel
                                     with a current source G * v_previous (implicit, unconditionally stable)
    diode(anode, cathode, i_s, n)    Shockley junction, solved by damped Newton iteration with junction-voltage
                                     limiting (LEDs are diodes with their own i_s, n and an optional series resistor)
A time-dependent source (AC) is a vsource/isource whose value the caller sets for the time of the step.
Every node has G_MIN to the reference node, so a floating part of the board (a meter lying in the tray, a node with
nothing on it) solves to 0 V instead of making the matrix singular. Solution: node voltages and, for every element,
its current from its first to its second node.

supply_solve wraps a netlist builder with a bench supply: constant voltage up to its current limit, constant
current (at the limit, CC) beyond it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G_MIN = 1.0e-12                 # S from every node to the reference
VT_300K = 0.025852              # thermal voltage kT/q at 300 K
NEWTON_MAX = 200
NEWTON_TOL_V = 1.0e-9


@dataclass
class Element:
    kind: str
    name: str
    a: str
    b: str
    value: float = 0.0
    params: dict = field(default_factory=dict)


class Netlist:
    def __init__(self, reference='0'):
        self.reference = reference
        self.elements: list[Element] = []
        self._names = set()

    def _add(self, kind, name, a, b, value=0.0, **params):
        if name in self._names:
            raise ValueError('circuit model: duplicate element')
        if a == b:
            raise ValueError('circuit model: an element has both terminals on one node')
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('circuit model: non-finite element value')
        self._names.add(name)
        self.elements.append(Element(kind, name, str(a), str(b), value, params))
        return name

    def resistor(self, name, a, b, ohm):
        if not ohm > 0:
            raise ValueError('circuit model: element resistance must be positive')
        return self._add('R', name, a, b, ohm)

    def vsource(self, name, a, b, volts):
        return self._add('V', name, a, b, volts)

    def isource(self, name, a, b, amps):
        return self._add('I', name, a, b, amps)

    def capacitor(self, name, a, b, farad, v0=0.0):
        if not farad > 0:
            raise ValueError('circuit model: element capacitance must be positive')
        return self._add('C', name, a, b, farad, v0=float(v0))

    def diode(self, name, anode, cathode, i_s, n=1.0, vt=VT_300K):
        if not (i_s > 0 and n > 0 and vt > 0):
            raise ValueError('circuit model: element diode parameters must be positive')
        return self._add('D', name, anode, cathode, 0.0, i_s=float(i_s), n=float(n), vt=float(vt))

    def nodes(self):
        out = [self.reference]
        for e in self.elements:
            for node in (e.a, e.b):
                if node not in out:
                    out.append(node)
        return out


@dataclass
class Solution:
    v: dict                      # node -> volts (reference 0)
    i: dict                      # element name -> amps from its first to its second node
    iterations: int = 1

    def vab(self, a, b):
        return self.v.get(a, 0.0) - self.v.get(b, 0.0)


def solve(netlist: Netlist, *, dt=None, previous=None, g_min=G_MIN, guess=None) -> Solution:
    """DC operating point (dt None: capacitors open) or one backward-Euler step of length dt.

    previous: the Solution of the step before (capacitor voltages); a capacitor without one starts at its v0.
    guess: node voltages to start Newton from (diodes), e.g. the previous solution's.
    """
    nodes = netlist.nodes()
    index = {n: k - 1 for k, n in enumerate(nodes)}          # reference -> -1
    sources = [e for e in netlist.elements if e.kind == 'V']
    n, m = len(nodes) - 1, len(sources)
    size = n + m
    diodes = [e for e in netlist.elements if e.kind == 'D']
    v = np.zeros(n)
    if guess:
        for node, value in guess.items():
            if node in index and index[node] >= 0:
                v[index[node]] = value

    def vd(e, x):
        a, b = index[e.a], index[e.b]
        return (x[a] if a >= 0 else 0.0) - (x[b] if b >= 0 else 0.0)

    cap_v = {}
    for e in netlist.elements:
        if e.kind == 'C':
            cap_v[e.name] = (previous.vab(e.a, e.b) if previous is not None else e.params['v0'])

    iterations = 0
    while True:
        iterations += 1
        A = np.zeros((size, size))
        z = np.zeros(size)

        def stamp_g(a, b, g):
            ia, ib = index[a], index[b]
            if ia >= 0:
                A[ia, ia] += g
            if ib >= 0:
                A[ib, ib] += g
            if ia >= 0 and ib >= 0:
                A[ia, ib] -= g
                A[ib, ia] -= g

        def stamp_i(a, b, amps):
            """A current `amps` flowing from a to b through the element (leaves node a, enters node b)."""
            ia, ib = index[a], index[b]
            if ia >= 0:
                z[ia] -= amps
            if ib >= 0:
                z[ib] += amps

        for k in range(n):
            A[k, k] += g_min
        for e in netlist.elements:
            if e.kind == 'R':
                stamp_g(e.a, e.b, 1.0 / e.value)
            elif e.kind == 'I':
                stamp_i(e.a, e.b, e.value)
            elif e.kind == 'C' and dt is not None:
                g = e.value / dt
                stamp_g(e.a, e.b, g)
                stamp_i(e.b, e.a, g * cap_v[e.name])          # companion source G*v_prev from b to a
            elif e.kind == 'D':
                u = vd(e, v)
                nvt = e.params['n'] * e.params['vt']
                u = min(u, 80 * nvt)                            # exp overflow guard (steps are limited below)
                expo = math.exp(u / nvt)
                current = e.params['i_s'] * (expo - 1.0)
                g = e.params['i_s'] * expo / nvt + g_min
                stamp_g(e.a, e.b, g)
                stamp_i(e.a, e.b, current - g * u)
        for k, e in enumerate(sources):
            row = n + k
            ia, ib = index[e.a], index[e.b]
            if ia >= 0:
                A[ia, row] += 1.0
                A[row, ia] += 1.0
            if ib >= 0:
                A[ib, row] -= 1.0
                A[row, ib] -= 1.0
            z[row] = e.value
        x = np.linalg.solve(A, z)
        new_v = x[:n]
        if not diodes:
            v = new_v
            break
        # Junction-voltage limiting (SPICE pnjlim): above its critical voltage a junction's step grows only
        # logarithmically; the whole Newton step is shortened to the most limited junction.
        scale = 1.0
        for e in diodes:
            nvt = e.params['n'] * e.params['vt']
            vcrit = nvt * math.log(nvt / (math.sqrt(2.0) * e.params['i_s']))
            old, new = vd(e, v), vd(e, new_v)
            if new > vcrit and abs(new - old) > 2 * nvt:
                if old > 0:
                    arg = 1.0 + (new - old) / nvt
                    capped = old + nvt * math.log(arg) if arg > 0 else vcrit
                else:
                    capped = nvt * math.log(new / nvt)
                if new != old:
                    scale = min(scale, max((capped - old) / (new - old), 1e-3))
        v_prev, v = v, v + scale * (new_v - v)
        if np.max(np.abs(v - v_prev)) < NEWTON_TOL_V:
            break
        if iterations >= NEWTON_MAX:
            raise RuntimeError('diode Newton iteration did not converge')
    volts = {node: (0.0 if index[node] < 0 else float(v[index[node]])) for node in nodes}
    currents = {}
    for e in netlist.elements:
        u = volts[e.a] - volts[e.b]
        if e.kind == 'R':
            currents[e.name] = u / e.value
        elif e.kind == 'I':
            currents[e.name] = e.value
        elif e.kind == 'C':
            currents[e.name] = 0.0 if dt is None else e.value * (u - cap_v[e.name]) / dt
        elif e.kind == 'D':
            nvt = e.params['n'] * e.params['vt']
            currents[e.name] = e.params['i_s'] * (math.exp(min(u / nvt, 80.0)) - 1.0)
    if sources:
        branch = x[n:]
        for k, e in enumerate(sources):
            # The MNA unknown is the current entering the source at its + node a from outside, i.e. flowing a -> b
            # through the source.
            currents[e.name] = float(branch[k])
    return Solution(volts, currents, iterations)


@dataclass
class SupplyResult:
    solution: Solution
    mode: str                    # 'CV', 'CC' or 'OFF'
    current_a: float             # delivered from the + terminal into the circuit
    voltage_v: float             # V(+) - V(-)


def supply_solve(build, plus, minus, *, on, v_set, i_limit, **kwargs) -> SupplyResult:
    """Solve the netlist that build(netlist) fills, with a bench supply between plus and minus.

    Constant voltage v_set while the load draws at most i_limit; beyond that constant current i_limit (the output
    voltage drops below v_set). OFF: the output is open."""
    net = Netlist(reference=minus)
    build(net)
    if not on:
        sol = solve(net, **kwargs)
        return SupplyResult(sol, 'OFF', 0.0, sol.vab(plus, minus))
    cv = Netlist(reference=minus)
    build(cv)
    cv.vsource('__supply', plus, minus, v_set)
    sol = solve(cv, **kwargs)
    delivered = -sol.i['__supply']          # current through the source from + to - is -delivered
    if delivered <= i_limit:
        return SupplyResult(sol, 'CV', delivered, sol.vab(plus, minus))
    cc = Netlist(reference=minus)
    build(cc)
    cc.isource('__supply', minus, plus, i_limit)
    sol = solve(cc, **kwargs)
    if sol.vab(plus, minus) > v_set:            # cannot happen with a passive load; stay in CV then
        sol = solve(cv, **kwargs)
        return SupplyResult(sol, 'CV', -sol.i['__supply'], sol.vab(plus, minus))
    return SupplyResult(sol, 'CC', i_limit, sol.vab(plus, minus))


class UnionFind:
    """Ideal connections (wires, a closed switch) merge board nodes into one electrical node."""

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
            # deterministic root: the smaller name
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra
        return ra

"""Beam bench mechanics (beam_compliance), headless: a cantilever on a compliant clamp.

The beam (length L, bending stiffness EI) leaves a clamp that turns by M / k_theta under the root moment M. Point loads
F_i at distances a_i from the clamp (small deflections, Euler-Bernoulli superposition):
    w(x) = sum_i F_i [ x^2 (3 a_i - x) / (6 EI)  (x <= a_i),  a_i^2 (3 x - a_i) / (6 EI)  (x > a_i) ] + x M_root / k_theta,
    M_root = sum_i F_i a_i.
The loads are whatever presses on the strip each physics tick (contact normal forces at their contact points, a knife-edge
saddle's at its edge). The deflected shape follows the loads with a first-order lag TAU_S (the strip's damped settling);
as the deflection is linear in the loads, lagging every station's deflection equals lagging the load distribution.
A rubber stop under the free end limits its travel to STOP_M: a target shape beyond it is scaled back to it.
"""
from __future__ import annotations

import math

import numpy as np

G = 9.81
TAU_S = 0.25
STOP_M = 0.030                 # travel of the free end (x = length) to the rubber stop under it


def deflection(x, loads, EI, k_theta):
    """Downward deflection (m) at x for loads [(a_m, F_N)]."""
    w = 0.0
    for a, F in loads:
        w += F * (x * x * (3 * a - x) if x <= a else a * a * (3 * x - a)) / (6 * EI)
        w += F * a * x / k_theta
    return w


def slope(x, loads, EI, k_theta):
    s = 0.0
    for a, F in loads:
        s += F * ((2 * a * x - x * x) / (2 * EI) if x <= a else a * a / (2 * EI))
        s += F * a / k_theta
    return s


def deflections(xs, loads, EI, k_theta):
    """Deflection (m) and slope (rad) at every x of the array xs for loads [(a_m, F_N)] (vectorised)."""
    xs = np.asarray(xs, float)
    w, s = np.zeros_like(xs), np.zeros_like(xs)
    for a, F in loads:
        near = xs <= a
        w += F * np.where(near, xs * xs * (3 * a - xs), a * a * (3 * xs - a)) / (6 * EI) + F * a * xs / k_theta
        s += F * np.where(near, (2 * a * xs - xs * xs) / 2, a * a / 2) / EI + F * a / k_theta
    return w, s


class Beam:
    """Lagged deflections (and slopes) at a set of stations (seats, gauges, the strip's collider pieces).

    length: the free end's distance from the clamp, where the travel stop acts (None: no stop)."""

    def __init__(self, EI, k_theta, stations, length=None):
        self.EI, self.k_theta = float(EI), float(k_theta)
        self.stations = dict(stations)                 # name -> x (m from the clamp)
        self.names = list(self.stations)
        self.x = np.array([self.stations[k] for k in self.names], float)
        self.length = length
        self._w = np.zeros(len(self.names))
        self._s = np.zeros(len(self.names))
        self.stopped = False

    @property
    def w(self):
        return dict(zip(self.names, self._w.tolist()))

    def value(self, name):
        return float(self._w[self.names.index(name)])

    def slope_at(self, name):
        return float(self._s[self.names.index(name)])

    def _goal(self, loads):
        loads = [(min(max(float(a), 0.0), self.length if self.length else float(a)), float(F)) for a, F in loads]
        w, s = deflections(self.x, loads, self.EI, self.k_theta)
        self.stopped = False
        if self.length is not None and loads:
            end = deflection(self.length, loads, self.EI, self.k_theta)
            if end > STOP_M:
                w, s = w * (STOP_M / end), s * (STOP_M / end)
                self.stopped = True
        return w, s

    def target(self, loads):
        return dict(zip(self.names, self._goal(loads)[0].tolist()))

    def step(self, dt, loads):
        w, s = self._goal(loads)
        f = 1.0 - math.exp(-dt / TAU_S)
        self._w += (w - self._w) * f
        self._s += (s - self._s) * f
        return self.w

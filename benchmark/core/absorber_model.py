"""Vibration absorber bench mechanics (vibration_absorber), headless: the cantilever's modes, the absorber and the sweep.

For configurations in the hidden ranges the sweep follows the calibration model operation for operation.

The primary: a steel strip clamped at x = 0 (the clamp turns with rotational stiffness k_theta), free at x = L, cubic
Hermite beam elements; lumped masses: the exciter's moving parts at X_F and the absorber mount at X_A (each at its
nearest node). Its first N_MODES modes (mass-normalised) carry the response, each with modal damping (zeta_1, then
ZETA_RATIO x zeta_1). The exciter pushes at X_F with F0 sin(phase(t)), a linear sweep F_LO..F_HI over SWEEP_S.

The absorber at X_A: a mass m (carrier + weights) on a blade spring k = 3 E_b I_b / L_b^3 (free length L_b, the clamp
position), a viscous damper c(n) = c0 exp(a n) at valve setting n, and rubber stops at +-GAP from the unloaded blade
line (contact stiffness K_STOP, damping C_STOP). Gravity sags the mass by m g / k (it may rest on the lower stop).
z = absorber displacement relative to the mount (up +). The sweep starts from the static state.

The reading: the tip's peak displacement from its static position over the sweep (mm), and the sweep frequency at
that moment; with curve=True also the peak-hold curve, the largest displacement in each BIN_HZ band of the sweep
frequency (the analyzer's amplitude-frequency curve). Many configurations integrate together (numpy arrays over the
last axis), RK4 at DT_S. A configuration whose absorber is too stiff or too damped for RK4 at DT_S (lambda DT_S above
RK4_LAMBDA_DT, see substeps) is integrated with substeps; every configuration in the hidden ranges needs none.

Sweep also advances step by step (advance_to) for the bench, which shows the strip and the absorber moving: its state
gives the strip's displacement and slope anywhere (mode_shapes) and the absorber's. Between sweeps the bench runs
Plant: the same equations without the exciter, with the configuration changing as weights come and go and with the
external loads of whatever presses on the strip, the mount or the absorber.

The optimum A* (the private target): the smallest peak over every configuration the bench allows (configs: the weight
subsets' total masses x the blade detents x the valve settings), optimum(h).
"""
from __future__ import annotations

import math

import numpy as np

MODEL_VERSION = 2                        # 2: the moving mass is the carrier's and its four slides'
G = 9.81
L_BEAM, B_BEAM, T_BEAM, RHO_STEEL = 0.400, 0.030, 0.004, 7850.0
X_F, X_A = 0.080, 0.370                  # exciter push rod, absorber mount (from the clamp edge)
M_EXCITER, M_MOUNT = 0.020, 0.040        # lumped on the strip
N_ELEM, N_MODES, ZETA_RATIO = 24, 3, 1.5
F_LO, F_HI, SWEEP_S = 5.0, 40.0, 60.0
F0_N = 1.5
DT_S = 1.0 / 2000
BIN_HZ = 0.1                             # the analyzer's curve: peak hold per 0.1 Hz of the sweep frequency
# Absorber: blade (spring steel), clamp positions, weights, damper valve, stops.
BLADE_W, BLADE_T, E_BLADE = 0.012, 0.0005, 200e9
BLADE_L_MM = tuple(range(30, 81, 5))     # 11 detents
M_CARRIER = 0.060                        # the parts riding on the blade: the carrier (POM clamp and tray, pins; 40 g)
                                         # and its four hold-down slides (aluminium, 5 g each), as their physics
                                         # bodies (absorber_bench.CARRIER_MASS, HOLD_MASS); x mass_factor
WEIGHTS_G = (10.0, 20.0, 30.0, 40.0)
VALVE = tuple(range(10))
GAP_M = 0.003
K_STOP, C_STOP = 5e4, 5.0
RK4_LAMBDA_DT = 2.6                      # RK4 is stable for |lambda dt| < 2.78 (real axis) / 2.83 (imaginary axis)

HIDDEN = {                               # name: (low, high); LOG ones log-uniform
    'E_beam': (190e9, 210e9),
    't_factor': (0.97, 1.03),
    'k_theta': (200.0, 800.0),
    'zeta1': (0.003, 0.010),
    'blade_factor': (0.80, 1.20),
    'c0': (0.20, 0.80),                  # N s/m at valve 0
    'c_span': (15.0, 80.0),              # c(9) / c(0)
    'gap_factor': (0.90, 1.10),
    'force_factor': (0.90, 1.10),
    'mass_factor': (0.95, 1.05),         # carrier (and mount) mass against nominal
}
LOG = ('zeta1', 'c0', 'c_span', 'k_theta')


def draw(rng):
    return {k: float(np.exp(rng.uniform(math.log(lo), math.log(hi))) if k in LOG else rng.uniform(lo, hi))
            for k, (lo, hi) in HIDDEN.items()}


def nominal():
    """The published / mid values (what a textbook agent would take)."""
    out = {k: (math.sqrt(lo * hi) if k in LOG else (lo + hi) / 2) for k, (lo, hi) in HIDDEN.items()}
    out.update(t_factor=1.0, blade_factor=1.0, gap_factor=1.0, force_factor=1.0, mass_factor=1.0)
    return out


def hidden_of(values):
    """The model's hidden dict (floats) from a mapping holding at least HIDDEN's keys (an instance's hidden)."""
    return {k: float(values[k]) for k in HIDDEN}


# --- the primary -------------------------------------------------------------------------------------------------
def _node(x):
    return int(round(x / (L_BEAM / N_ELEM)))


def _eig(h):
    """(omega^2 (all), mode vectors (nd, all)) of the clamped strip with the lumped masses; row 2 i is node i's
    deflection, 2 i + 1 its slope; w(0) = 0."""
    t = T_BEAM * h['t_factor']
    ei = h['E_beam'] * B_BEAM * t ** 3 / 12
    mu = RHO_STEEL * B_BEAM * t
    le = L_BEAM / N_ELEM
    nd = 2 * (N_ELEM + 1)
    K, M = np.zeros((nd, nd)), np.zeros((nd, nd))
    ke = ei / le ** 3 * np.array([[12, 6 * le, -12, 6 * le], [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
                                  [-12, -6 * le, 12, -6 * le], [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]])
    me = mu * le / 420 * np.array([[156, 22 * le, 54, -13 * le], [22 * le, 4 * le ** 2, 13 * le, -3 * le ** 2],
                                   [54, 13 * le, 156, -22 * le], [-13 * le, -3 * le ** 2, -22 * le, 4 * le ** 2]])
    for e in range(N_ELEM):
        i = slice(2 * e, 2 * e + 4)
        K[i, i] += ke
        M[i, i] += me
    K[1, 1] += h['k_theta']
    for x, m in ((X_F, M_EXCITER), (X_A, M_MOUNT * h['mass_factor'])):
        M[2 * _node(x), 2 * _node(x)] += m
    free = np.arange(1, nd)                       # w(0) = 0
    from scipy.linalg import eigh
    w2, V = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)])
    shapes = np.zeros((nd, len(w2)))
    shapes[free] = V
    return w2, shapes


def beam_modes(h, n_modes=N_MODES):
    """(omega rad/s (n,), shapes at (X_F, X_A, tip) (3, n)) of the clamped strip with the lumped masses."""
    w2, shapes = _eig(h)
    pick = lambda x: shapes[2 * _node(x), :n_modes]
    return np.sqrt(w2[:n_modes]), np.stack([pick(X_F), pick(X_A), pick(L_BEAM)])


def mode_shapes(h, xs, n_modes=N_MODES):
    """(deflection (len(xs), n), slope (len(xs), n)) of the first n modes at distances xs (m) from the clamp edge, by
    the elements' Hermite interpolation (up +; slope dw/dx)."""
    _, shapes = _eig(h)
    le = L_BEAM / N_ELEM
    xs = np.clip(np.asarray(xs, float), 0.0, L_BEAM)
    e = np.minimum((xs / le).astype(int), N_ELEM - 1)
    s = xs / le - e
    n = np.stack([1 - 3 * s ** 2 + 2 * s ** 3, le * (s - 2 * s ** 2 + s ** 3), 3 * s ** 2 - 2 * s ** 3,
                  le * (-s ** 2 + s ** 3)], axis=1)
    dn = np.stack([(-6 * s + 6 * s ** 2) / le, 1 - 4 * s + 3 * s ** 2, (6 * s - 6 * s ** 2) / le,
                   -2 * s + 3 * s ** 2], axis=1)
    dofs = np.stack([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3], axis=1)
    local = shapes[:, :n_modes][dofs]                        # (len(xs), 4, n)
    return np.einsum('ij,ijk->ik', n, local), np.einsum('ij,ijk->ik', dn, local)


def blade_k(h, length_mm):
    i_b = BLADE_W * BLADE_T ** 3 / 12
    return 3 * E_BLADE * i_b * h['blade_factor'] / (np.asarray(length_mm, float) * 1e-3) ** 3


def valve_c(h, n):
    a = math.log(h['c_span']) / 9
    return h['c0'] * np.exp(a * np.asarray(n, float))


def absorber_mass(h, mass_g):
    """The absorber's moving mass (kg): the carrier and the weights on its tray."""
    return M_CARRIER * h['mass_factor'] + np.asarray(mass_g, float) * 1e-3


def static_z(h, mass_g, length_mm):
    """The absorber's static displacement (m, up +) from the unloaded blade line: the sag m g / k, or resting on the
    lower stop."""
    m, k, gap = absorber_mass(h, mass_g), blade_k(h, length_mm), GAP_M * h['gap_factor']
    return np.where(m * G / np.maximum(k, 1e-9) <= gap, -m * G / np.maximum(k, 1e-9),
                    -(m * G + K_STOP * gap) / (k + K_STOP))


def substeps(h, mass_g, length_mm, valve, dt=DT_S):
    """RK4 substeps per DT_S for each configuration: the absorber's fastest rate, its damping (c + C_STOP) and
    stiffness (k + K_STOP) over the effective mass (1/m + sum of the modes' squared mount shape), kept below
    RK4_LAMBDA_DT; and the highest beam mode."""
    om, ph = beam_modes(h)
    inv_m = 1.0 / absorber_mass(h, mass_g) + float(np.sum(ph[1] ** 2))
    lam = np.maximum((valve_c(h, valve) + C_STOP) * inv_m, np.sqrt((blade_k(h, length_mm) + K_STOP) * inv_m))
    lam = np.maximum(lam, float(om.max()))
    return np.maximum(1, np.ceil(lam * dt / RK4_LAMBDA_DT)).astype(int)


# --- the sweep ---------------------------------------------------------------------------------------------------
class Sweep:
    """One sweep of n configurations (1-d arrays, or scalars), integrated step by step.

    step() advances DT_S (n_sub RK4 substeps); advance_to(t) steps while the next step ends by t; result() gives the
    peak (mm), its frequency (Hz) and, with curve=True, the peak-hold curve (mm per BIN_HZ band). All configurations
    share n_sub: sweep() groups them by substeps()."""

    def __init__(self, h, mass_g, length_mm, valve, *, dt=DT_S, sweep_s=SWEEP_S, f_lo=F_LO, f_hi=F_HI, absorber=True,
                 curve=False, n_sub=1, initial=None):
        mass_g, length_mm, valve = (np.atleast_1d(np.asarray(a, float)) for a in
                                    np.broadcast_arrays(np.asarray(mass_g, float), np.asarray(length_mm, float),
                                                        np.asarray(valve, float)))
        n = mass_g.size
        self.n, self.absorber = n, absorber
        self.dt, self.sweep_s, self.f_lo, self.f_hi = float(dt), float(sweep_s), float(f_lo), float(f_hi)
        self.n_sub = int(n_sub)
        om, ph = beam_modes(h)
        self.om, self.ph = om, ph
        nm = len(om)
        self.zeta = h['zeta1'] * np.r_[1.0, np.full(nm - 1, ZETA_RATIO)]
        self.pf, self.pa, self.pt = ph                           # (nm,)
        self.m = absorber_mass(h, mass_g) if absorber else np.ones(n)
        self.k = blade_k(h, length_mm) if absorber else np.zeros(n)
        self.c = valve_c(h, valve) if absorber else np.zeros(n)
        self.gap = GAP_M * h['gap_factor']
        self.f0 = F0_N * h['force_factor']
        m, k = self.m, self.k
        # Static state: the absorber's weight on the mount; z on the spring (or on the lower stop).
        z0 = np.where(m * G / np.maximum(k, 1e-9) <= self.gap, -m * G / np.maximum(k, 1e-9),
                      -(m * G + K_STOP * self.gap) / (k + K_STOP)) if absorber else np.zeros(n)
        fa0 = -m * G if absorber else np.zeros(n)                # force of the absorber on the mount (up +)
        self.q0 = (self.pa[:, None] * fa0[None]) / (om[:, None] ** 2)
        self.tip0 = self.pt @ self.q0
        self.z0 = z0
        # State: q (nm, n), qd (nm, n), z (n), zd (n) - z absolute relative to the unloaded blade line.
        self.q, self.qd = self.q0.copy(), np.zeros((nm, n))
        self.z, self.zd = z0.copy(), np.zeros(n)
        if initial is not None:                                  # a sweep started while the bench still moved
            self.q, self.qd = (np.asarray(initial[k], float).reshape(nm, n).copy() for k in ('q', 'qd'))
            self.z, self.zd = (np.asarray(initial[k], float).reshape(n).copy() for k in ('z', 'zd'))
        self.rate = (self.f_hi - self.f_lo) / self.sweep_s
        self.steps = int(round(self.sweep_s / self.dt))
        self.i = 0
        self.t = 0.0
        self.peak = np.zeros(n)
        self.peak_f = np.zeros(n)
        self.bins = int(round((self.f_hi - self.f_lo) / BIN_HZ))
        self.curve = np.zeros((n, self.bins)) if curve else None

    def _stop(self, z, zd):
        over = np.maximum(z - self.gap, 0.0) + np.minimum(z + self.gap, 0.0)
        return np.where(over != 0.0, K_STOP * over + C_STOP * zd, 0.0)

    def _acc(self, t, q, qd, z, zd):
        om, zeta, pf, pa = self.om, self.zeta, self.pf, self.pa
        phase = 2 * math.pi * (self.f_lo * t + 0.5 * self.rate * t * t)
        f = self.f0 * math.sin(phase)
        fa = (self.k * z + self.c * zd + self._stop(z, zd)) if self.absorber else 0.0  # pulls the mount up when z > 0
        qdd = -2 * (zeta * om)[:, None] * qd - (om ** 2)[:, None] * q + pf[:, None] * f + pa[:, None] * fa
        if not self.absorber:
            return qdd, np.zeros(self.n)
        # absolute absorber acceleration = -(k z + c zd + stop) / m - g ; z = u - w_a  =>  zdd = udd - wa_dd
        wa_dd = pa @ qdd
        zdd = -fa / self.m - G - wa_dd
        return qdd, zdd

    def _f_all(self, t, q, qd, z, zd):
        qdd, zdd = self._acc(t, q, qd, z, zd)
        return qd, qdd, zd, zdd

    def _rk4(self, dt):
        t, q, qd, z, zd = self.t, self.q, self.qd, self.z, self.zd
        f_all = self._f_all
        k1 = f_all(t, q, qd, z, zd)
        k2 = f_all(t + dt / 2, q + dt / 2 * k1[0], qd + dt / 2 * k1[1], z + dt / 2 * k1[2], zd + dt / 2 * k1[3])
        k3 = f_all(t + dt / 2, q + dt / 2 * k2[0], qd + dt / 2 * k2[1], z + dt / 2 * k2[2], zd + dt / 2 * k2[3])
        k4 = f_all(t + dt, q + dt * k3[0], qd + dt * k3[1], z + dt * k3[2], zd + dt * k3[3])
        self.q = q + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self.qd = qd + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self.z = z + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        self.zd = zd + dt / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        self.t = t + dt

    @property
    def done(self):
        return self.i >= self.steps

    def step(self):
        if self.done:
            raise ValueError('the sweep has ended')
        if self.n_sub == 1:
            self._rk4(self.dt)
        else:
            for _ in range(self.n_sub):
                self._rk4(self.dt / self.n_sub)
        self.i += 1
        dev = np.abs(self.pt @ self.q - self.tip0)
        better = dev > self.peak
        self.peak = np.where(better, dev, self.peak)
        f_now = self.f_lo + self.rate * self.t
        self.peak_f = np.where(better, f_now, self.peak_f)
        if self.curve is not None:
            b = min(int((f_now - self.f_lo) / BIN_HZ), self.bins - 1)
            np.maximum(self.curve[:, b], dev, out=self.curve[:, b])

    def advance_to(self, t_s):
        """Step while the next step ends by t_s (s since the sweep started); returns the steps taken."""
        n = 0
        while not self.done and (self.i + 1) * self.dt <= t_s + 1e-9:
            self.step()
            n += 1
        if n and not (np.all(np.isfinite(self.q)) and np.all(np.isfinite(self.z))):
            raise FloatingPointError('absorber sweep diverged')
        return n

    def frequency(self):
        """The sweep frequency now (Hz)."""
        return self.f_lo + self.rate * self.t

    def tip_mm(self):
        """The tip's displacement from its static position now (mm, up +), per configuration."""
        return (self.pt @ self.q - self.tip0) * 1e3

    def strip(self, w_shapes, s_shapes):
        """(deflection, slope) of the strip now at the stations whose mode shapes are given (mode_shapes): absolute
        (static sag included), up +, (len(stations), n)."""
        return w_shapes @ self.q, s_shapes @ self.q

    def state(self):
        """The integration state {'q', 'qd', 'z', 'zd'} (copies)."""
        return {'q': self.q.copy(), 'qd': self.qd.copy(), 'z': self.z.copy(), 'zd': self.zd.copy()}

    def result(self):
        if not (np.all(np.isfinite(self.peak)) and np.all(np.isfinite(self.q))):
            raise FloatingPointError('absorber sweep diverged')
        out = {'peak_mm': self.peak * 1e3, 'peak_hz': self.peak_f.copy()}
        if self.curve is not None:
            out['curve_mm'] = self.curve * 1e3
            out['curve_hz'] = self.f_lo + BIN_HZ * (np.arange(self.bins) + 0.5)
        return out


def sweep(h, mass_g, length_mm, valve, dt=DT_S, sweep_s=SWEEP_S, f_lo=F_LO, f_hi=F_HI, absorber=True):
    """Peak tip displacement (mm) and its frequency (Hz) over a sweep, per configuration (arrays broadcast)."""
    mass_g, length_mm, valve = np.broadcast_arrays(np.asarray(mass_g, float), np.asarray(length_mm, float),
                                                   np.asarray(valve, float))
    shape = mass_g.shape
    mass_g, length_mm, valve = mass_g.ravel(), length_mm.ravel(), valve.ravel()
    n_sub = substeps(h, mass_g, length_mm, valve, dt) if absorber else np.ones(mass_g.size, int)
    peak, peak_f = np.zeros(mass_g.size), np.zeros(mass_g.size)
    for s in np.unique(n_sub):
        pick = n_sub == s
        run = Sweep(h, mass_g[pick], length_mm[pick], valve[pick], dt=dt, sweep_s=sweep_s, f_lo=f_lo, f_hi=f_hi,
                    absorber=absorber, n_sub=int(s))
        for _ in range(run.steps):
            run.step()
        r = run.result()
        peak[pick], peak_f[pick] = r['peak_mm'], r['peak_hz']
    return peak.reshape(shape), peak_f.reshape(shape)


def sweep_curve(h, mass_g, length_mm, valve):
    """One configuration's sweep as the analyzer records it: {'peak_mm', 'peak_hz', 'curve_hz', 'curve_mm'}."""
    n_sub = int(np.max(substeps(h, mass_g, length_mm, valve)))
    run = Sweep(h, mass_g, length_mm, valve, curve=True, n_sub=n_sub)
    for _ in range(run.steps):
        run.step()
    r = run.result()
    return {'peak_mm': float(r['peak_mm'][0]), 'peak_hz': float(r['peak_hz'][0]), 'curve_hz': r['curve_hz'],
            'curve_mm': r['curve_mm'][0]}


# --- configurations and the optimum ----------------------------------------------------------------------------
def mass_steps():
    """The total masses (g) the weights make, sorted (subset sums of WEIGHTS_G)."""
    return sorted({sum(w for w, b in zip(WEIGHTS_G, bits) if b) for bits in np.ndindex(*(2,) * len(WEIGHTS_G))})


def configs():
    """Every configuration the bench allows: (total weight g, blade length mm, valve) arrays, weights as subset sums."""
    mg, lm, vv = np.meshgrid(np.array(mass_steps()), np.array(BLADE_L_MM, float), np.array(VALVE, float), indexing='ij')
    return mg, lm, vv


def grid(h):
    """Peak (mm) and its frequency (Hz) of every configuration (configs() shape)."""
    mg, lm, vv = configs()
    return sweep(h, mg, lm, vv)


def optimum_from_grid(peaks):
    """{'A_star_mm', 'mass_g', 'length_mm', 'valve', 'edge'} of a configs()-shaped grid of peaks; edge: the optimum's
    blade length on an end detent or its valve at 0 or 9."""
    peaks = np.asarray(peaks, float)
    mg, lm, vv = configs()
    if peaks.shape != mg.shape:
        raise ValueError(f'grid shape {peaks.shape}, expected {mg.shape}')
    i = np.unravel_index(int(np.argmin(peaks)), peaks.shape)
    L, v = float(lm[i]), int(vv[i])
    return {'A_star_mm': float(peaks[i]), 'mass_g': float(mg[i]), 'length_mm': L, 'valve': v,
            'edge': L in (BLADE_L_MM[0], BLADE_L_MM[-1]) or v in (VALVE[0], VALVE[-1])}


def optimum(h):
    """The optimum over the full grid (optimum_from_grid) and the grid itself (peaks mm)."""
    peaks, _ = grid(h)
    return optimum_from_grid(peaks) | {'grid_mm': peaks}


# --- the bench between sweeps -----------------------------------------------------------------------------------
class Plant:
    """The strip and the absorber between sweeps (one configuration, scalars): the sweep's equations without the
    exciter, plus external loads held over each advance: modal forces (nm,) from loads on the strip (sum of F phi(x),
    up +) and a vertical force on the absorber mass (up +). set_config changes the mass, blade and valve in place (the
    state carries on: a weight put on the tray makes the absorber sag and ring). The state hands over to and from a
    Sweep (state(), set_state)."""

    def __init__(self, h, mass_g=0.0, length_mm=55.0, valve=0, dt=DT_S):
        self.h = dict(h)
        self.dt = float(dt)
        self.om, self.ph = beam_modes(h)
        nm = len(self.om)
        self.zeta = h['zeta1'] * np.r_[1.0, np.full(nm - 1, ZETA_RATIO)]
        self.pf, self.pa, self.pt = self.ph
        self.gap = GAP_M * h['gap_factor']
        self.set_config(mass_g, length_mm, valve)
        st = self.static_state()
        self.q, self.qd, self.z, self.zd = st['q'], st['qd'], st['z'], st['zd']
        self.t = 0.0
        self.f_modal = np.zeros(nm)
        self.f_abs = 0.0
        self.c_touch = 0.0          # a finger on the absorber: damping on its motion in the world (N s/m, the bench sets it)

    def set_config(self, mass_g, length_mm, valve):
        """The absorber's mass (weights on the tray, g), blade length (mm, where the carrier stands: it slides
        continuously between the detents) and valve setting."""
        self.mass_g, self.length_mm, self.valve = float(mass_g), float(length_mm), int(valve)
        self.m = float(absorber_mass(self.h, mass_g))
        self.k = float(blade_k(self.h, length_mm))
        self.c = float(valve_c(self.h, valve))
        inv_m = 1.0 / self.m + float(np.sum(self.pa ** 2))       # as substeps(), without recomputing the modes
        lam = max((self.c + C_STOP) * inv_m, math.sqrt((self.k + K_STOP) * inv_m), float(self.om.max()))
        self.n_sub = max(1, int(math.ceil(lam * self.dt / RK4_LAMBDA_DT)))

    def static_state(self):
        """The rest state of the present configuration without external loads (the sweep's start)."""
        z0 = float(static_z(self.h, self.mass_g, self.length_mm))
        q0 = self.pa * (-self.m * G) / self.om ** 2
        return {'q': q0.copy(), 'qd': np.zeros_like(q0), 'z': z0, 'zd': 0.0}

    def state(self):
        return {'q': self.q.copy(), 'qd': self.qd.copy(), 'z': float(self.z), 'zd': float(self.zd)}

    def set_state(self, state):
        self.q = np.asarray(state['q'], float).reshape(-1).copy()
        self.qd = np.asarray(state['qd'], float).reshape(-1).copy()
        self.z, self.zd = float(np.asarray(state['z']).reshape(-1)[0]), float(np.asarray(state['zd']).reshape(-1)[0])

    def _stop(self, z, zd):
        over = max(z - self.gap, 0.0) + min(z + self.gap, 0.0)
        return K_STOP * over + C_STOP * zd if over != 0.0 else 0.0

    def _f(self, q, qd, z, zd):
        fa = self.k * z + self.c * zd + self._stop(z, zd)
        qdd = -2 * self.zeta * self.om * qd - self.om ** 2 * q + self.pa * fa + self.f_modal
        f_touch = -self.c_touch * (zd + float(self.pa @ qd)) if self.c_touch else 0.0
        zdd = (-fa + self.f_abs + f_touch) / self.m - G - float(self.pa @ qdd)
        return qd, qdd, zd, zdd

    def _rk4(self, dt):
        q, qd, z, zd = self.q, self.qd, self.z, self.zd
        k1 = self._f(q, qd, z, zd)
        k2 = self._f(q + dt / 2 * k1[0], qd + dt / 2 * k1[1], z + dt / 2 * k1[2], zd + dt / 2 * k1[3])
        k3 = self._f(q + dt / 2 * k2[0], qd + dt / 2 * k2[1], z + dt / 2 * k2[2], zd + dt / 2 * k2[3])
        k4 = self._f(q + dt * k3[0], qd + dt * k3[1], z + dt * k3[2], zd + dt * k3[3])
        self.q = q + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self.qd = qd + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self.z = z + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        self.zd = zd + dt / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

    def advance_to(self, t_s, f_modal=None, f_abs=0.0):
        """Step (DT_S, with the configuration's substeps) while the next step ends by t_s, under the given loads."""
        self.f_modal = np.zeros(len(self.om)) if f_modal is None else np.asarray(f_modal, float)
        self.f_abs = float(f_abs)
        n = 0
        while self.t + self.dt <= t_s + 1e-9:
            for _ in range(self.n_sub):
                self._rk4(self.dt / self.n_sub)
            self.t += self.dt
            n += 1
        if n and not (np.all(np.isfinite(self.q)) and math.isfinite(self.z)):
            raise FloatingPointError('absorber plant diverged')
        return n

    def deviation(self):
        """How far the state is from the configuration's rest (tip m, absorber m, tip m/s, absorber m/s)."""
        st = self.static_state()
        return (abs(float(self.pt @ (self.q - st['q']))), abs(self.z - st['z']), abs(float(self.pt @ self.qd)),
                abs(self.zd))

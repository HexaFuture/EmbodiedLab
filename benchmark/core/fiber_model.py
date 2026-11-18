"""Model of fiber_coupling: a laser beam steered by two kinematic mirrors into an aspheric coupling
lens and a single-mode fiber on a focus stage, read by a power meter. No Isaac.

Calibration model (from `LAM` to `textbook_lens`): the bench constants, the hidden values of an instance (`draw`), the beam and its central ray, the
coupling into the fiber with its cladding light (`coupling`) and the optimum eta* over the five lenses (`optimum`).

Geometry (beam path from the laser exit, metres): M1 at L1, M2 at L2, lens at L3, fiber face at L3 + f + dz.
A mirror tilted by phi deflects the beam by 2 phi. At the lens the central ray has offset x_L and angle a_L; after
the lens (focal f) at distance f + dz from it the ray is at x_L + (f + dz)(a_L - x_L / f) with angle a_L - x_L / f.
Coupling into the fiber: overlap of the focused Gaussian beam (complex q through ABCD) with the fiber mode (waist
w_m at the face), with the ray's offset and angle relative to the core (the core's own offset and pointing are
hidden), times the lens Strehl factor (residual spherical aberration rms kappa w_L^4 / f^3) and the transmission.
Cladding light: while the spot lies on the cladding (|r| < R_CLAD) a fraction of the light outside the core mode
reaches the meter (short patch cord), falling to zero at the cladding's edge, as in a real first-light search.

Knobs: fine adjusters (0.1 mm/turn, about 254 TPI) at LEVER from the pivot: tilt g = 0.1e-3 / (360 LEVER) rad per
degree; each knob has backlash (degrees of dead travel after a reversal) and a small cross-talk into the other axis.
Focus knob: FOCUS_UM_PER_DEG micrometres of fiber travel per degree, with its own backlash.

Bench mechanics (`Adjusters`, added for the bench): the knob angles the physics reports (degrees, unwrapped, from the
episode's start) drive the screws through the backlash as a true dead band (the screw stays put until the knob has
taken up the play; the play's initial phase is hidden, drawn from the instance seed on its own stream so that `draw`
stays the calibration's), the screws tilt the mirrors with the cross-talk and move the fiber (`Adjusters.phis`,
`Adjusters.dz`). The calibration's offline bench applied the backlash per turn instead (all of it on the first turn
after a reversal).

Robot operations measured on the knob (with its own inertia, fiber_knob.py): the grasp
hold turns a held knob by the wrist's turn less a lost motion of GRIP_LAG_DEG (a dead band between hand and knob that
starts centred at each new grasp: a stroke loses GRIP_LAG_DEG, a reversal while holding 2 GRIP_LAG_DEG); the times
(sim seconds) below were measured there too.

Instance (`hidden_of`): the registry draws one integer `model_seed`; `draw(model_seed)` gives every hidden value as the
calibration drew its instances.
"""
from __future__ import annotations

import numpy as np

LAM = 633e-9
K = 2 * np.pi / LAM
L1, L2, L3 = 0.20, 0.45, 0.70
LENSES_MM = (4.51, 6.24, 8.00, 11.0, 15.29)
W0_NOM, MFD_NOM = 0.80e-3, 4.0e-6
LEVER = 0.030
PITCH_M = 0.1e-3
G_RAD_PER_DEG = PITCH_M / 360 / LEVER          # mirror tilt per knob degree (9.3 urad)
FOCUS_UM_PER_DEG = 0.5e3 / 360                  # 0.5 mm per turn: 1.39 um per degree
TRANSMISSION = 0.99 * 0.965
R_CLAD = 62.5e-6
P_LASER_UW = 1000.0
KAPPA_NOM = 0.0073

HIDDEN = dict(w0=(0.7, 1.3), z_w=(-0.3, 0.3), mfd=(0.85, 1.15), kappa=(0.5, 3.0), f_tol=0.01,
              backlash_deg=(2.0, 8.0), crosstalk=(0.0, 0.05), clad=(0.05, 0.20),
              core_off_um=5.0, core_tilt_mrad=3.0, cap_off_mm=(1.0, 4.0), focus_off_um=100.0)


def draw(seed, hidden=HIDDEN):
    rng = np.random.default_rng(seed)
    h = hidden
    inst = dict(seed=seed,
                w0=W0_NOM * rng.uniform(*h['w0']), z_w=rng.uniform(*h['z_w']),
                w_m=MFD_NOM / 2 * rng.uniform(*h['mfd']),
                kappa=KAPPA_NOM * np.exp(rng.uniform(*np.log(h['kappa']), len(LENSES_MM))),
                f=np.array(LENSES_MM) * 1e-3 * (1 + rng.uniform(-h['f_tol'], h['f_tol'], len(LENSES_MM))),
                backlash=rng.uniform(*h['backlash_deg'], 5),       # M1x, M1y, M2x, M2y, focus
                crosstalk=rng.uniform(*h['crosstalk'], 4),
                clad=rng.uniform(*h['clad']),
                core=rng.normal(0, h['core_off_um'] * 1e-6, 2), core_tilt=rng.normal(0, h['core_tilt_mrad'] * 1e-3, 2))
    # initial beam at the lens: offset 1-4 mm in a random direction, angle as the core's pointing plus a random error
    ang = rng.uniform(0, 2 * np.pi)
    r = rng.uniform(*h['cap_off_mm']) * 1e-3
    inst['x0'] = np.array([r * np.cos(ang), r * np.sin(ang)])
    inst['a0'] = inst['core_tilt'] + rng.normal(0, 2e-3, 2)
    # each lens cartridge is made to put its focus near the fiber face; the residual is per cartridge
    inst['dz0'] = rng.uniform(-h['focus_off_um'], h['focus_off_um'], len(LENSES_MM)) * 1e-6
    return inst


def beam_at_lens(inst):
    zr = np.pi * inst['w0'] ** 2 / LAM
    q = (L3 - inst['z_w']) + 1j * zr
    w = np.sqrt(-LAM / (np.pi * np.imag(1 / q)))
    return q, w


def ray(inst, phi, axis):
    """Central ray at the lens (offset, angle) for mirror tilts phi = (phi1, phi2) on one axis."""
    x = inst['x0'][axis] + 2 * phi[0] * (L3 - L1) + 2 * phi[1] * (L3 - L2)
    a = inst['a0'][axis] + 2 * phi[0] + 2 * phi[1]
    return x, a


L_T = 0.48                    # removable alignment target, just after M2


def ray_at(inst, phi, axis, z):
    """Central ray offset at path position z (z > L2) for mirror tilts phi on one axis."""
    x, a = ray(inst, phi, axis)
    return x + a * (z - L3)


def _overlap_1d(q1, q2, a, th):
    A = 1j * K / (2 * q1) - 1j * K / (2 * np.conj(q2))
    B = 1j * K * a / q1 + 1j * K * th
    C = -1j * K * a ** 2 / (2 * q1)
    I = np.sqrt(np.pi / A) * np.exp(B ** 2 / (4 * A) + C)
    n1 = np.sqrt(np.pi / (2 * np.real(1j * K / (2 * q1))))
    n2 = np.sqrt(np.pi / (2 * np.real(1j * K / (2 * q2))))
    return np.abs(I) ** 2 / (n1 * n2)


def coupling(inst, lens, phi_x, phi_y, dz, cladding=True):
    """Fraction of the laser power on the meter: (core, total)."""
    f = inst['f'][lens]
    q_l, w_l = beam_at_lens(inst)
    q_f = 1 / (1 / q_l - 1 / f) + f + dz
    q_m = 1j * np.pi * inst['w_m'] ** 2 / LAM
    eta = TRANSMISSION
    off = []
    for axis, phi in ((0, phi_x), (1, phi_y)):
        x, a = ray(inst, phi, axis)
        xf = x + (f + dz) * (a - x / f) - inst['core'][axis]
        af = (a - x / f) - inst['core_tilt'][axis]
        eta = eta * _overlap_1d(q_f, q_m, xf, af)
        off.append(xf)
    sigma = inst['kappa'][lens] * w_l ** 4 / f ** 3
    core = eta * np.exp(-(2 * np.pi * sigma / LAM) ** 2)
    if not cladding:
        return core
    r = np.hypot(*off)
    # cladding modes take less of the light the nearer the spot is to the cladding's edge
    clad = np.where(r < R_CLAD, inst['clad'] * (TRANSMISSION - core) * (1 - (r / R_CLAD) ** 2), 0.0)
    return core, core + clad


def aligned_phi(inst, lens, dz, axis):
    """Mirror tilts that put the ray on the core with the core's pointing (two equations, two tilts)."""
    f = inst['f'][lens]
    # want a - x/f = t (core tilt) and x + (f+dz) t = c (core offset)  ->  x = c - (f+dz) t, a = t + x/f
    t, c = inst['core_tilt'][axis], inst['core'][axis]
    x = c - (f + dz) * t
    a = t + x / f
    # x = x0 + 2 p1 (L3-L1) + 2 p2 (L3-L2), a = a0 + 2 p1 + 2 p2
    M = np.array([[2 * (L3 - L1), 2 * (L3 - L2)], [2.0, 2.0]])
    return np.linalg.solve(M, [x - inst['x0'][axis], a - inst['a0'][axis]])


def best_for_lens(inst, lens):
    dzs = np.linspace(-60e-6, 60e-6, 241)
    best = (0.0, 0.0)
    for dz in dzs:
        e = coupling(inst, lens, aligned_phi(inst, lens, dz, 0), aligned_phi(inst, lens, dz, 1), dz, cladding=False)
        if e > best[0]:
            best = (float(e), float(dz))
    lo, hi = best[1] - 1e-6, best[1] + 1e-6           # refine
    for dz in np.linspace(lo, hi, 81):
        e = coupling(inst, lens, aligned_phi(inst, lens, dz, 0), aligned_phi(inst, lens, dz, 1), dz, cladding=False)
        if e > best[0]:
            best = (float(e), float(dz))
    return best


def optimum(inst):
    per = [best_for_lens(inst, k) for k in range(len(LENSES_MM))]
    k = int(np.argmax([p[0] for p in per]))
    return dict(eta_star=per[k][0], lens=k, dz=per[k][1], per_lens=[p[0] for p in per])


def textbook_lens():
    """Mode-matching focal length from the nominal waist and mode field, collimated beam: f = pi w0 w_m / lambda."""
    f = np.pi * W0_NOM * (MFD_NOM / 2) / LAM
    return int(np.argmin([abs(np.log(x * 1e-3 / f)) for x in LENSES_MM])), f


# --- bench (added to the calibration model) -------------------------------------------------------------------------
MODEL_VERSION = 1
KNOBS = ('M1x', 'M1y', 'M2x', 'M2y', 'focus')

GRIP_LAG_DEG = 0.45            # hand-to-knob lost motion per grasp (measured: 0.44-0.47 per stroke, scatter 0.005; held
                               # steps fit a dead band of +-0.448; 0.47-0.55 with a grip 2-4 mm / 2-4 deg off)
TURN_NOISE_DEG = 0.05          # random error of a held turn, knob against command: measured (own inertia) residual
                               # sd 0.020 deg (max 0.07) about the dead band (fitted 0.448); 0.05 keeps a margin
# times (sim seconds), measured on the knob over execute_joint_trajectory and set_gripper
WRIST_DEG_S = 25.8             # wrist turn while holding (0.45 rad/s)
TURN_MIN_S = 0.2               # shortest wrist trajectory
METER_S = 0.2                  # the meter refreshes every 0.2 s: one refresh after a turn before reading it
TAKE_HOLD_S = 6.1              # from the 5 cm stand-off: open 0.6, in at 2 cm/s 4.7, close 0.8
RELEASE_S = 2.5                # open 0.6, out to the stand-off 1.9
TRANSIT_S = 25.0               # stand-off to stand-off of another knob (move_hand 22-28)
MAX_GRIP_DEG = 300.0           # wrist travel per hold (q7 range 332 deg less margins): then let go, turn back, re-take


def turn_s(deg):
    """Sim time of one wrist turn of `deg` degrees while holding a knob, with the meter refresh after it."""
    return max(TURN_MIN_S, abs(deg) / WRIST_DEG_S) + METER_S


def change_knob_s():
    """Let go of one knob and take hold of another."""
    return RELEASE_S + TRANSIT_S + TAKE_HOLD_S


def regrasp_s(wrist_deg):
    """Let go, turn the open hand back by wrist_deg and take hold of the same knob again."""
    return RELEASE_S + abs(wrist_deg) / WRIST_DEG_S + TAKE_HOLD_S


def hidden_of(hidden):
    """The model instance of a registry instance's hidden values ({'model_seed': int})."""
    return draw(int(hidden['model_seed']))


def play_phase(inst):
    """Initial position of each knob within its backlash: 0 engaged for a - turn .. 1 engaged for a + turn."""
    return np.random.default_rng([int(inst['seed']), 7]).uniform(0.0, 1.0, 5)


class Adjusters:
    """Knob angles (degrees, unwrapped, + = right-handed about the knob's outward axis = the screw backs out, right-hand thread) to screw positions
    through each knob's backlash (a dead band of width backlash), to mirror tilts with the cross-talk and to the
    fiber's focus offset. screw[k] is the screw's travel in knob degrees since the episode's start."""

    def __init__(self, inst, phase=None):
        self.i = inst
        self.b = np.asarray(inst['backlash'], float)
        u = play_phase(inst) if phase is None else np.asarray(phase, float)
        self.knob = np.zeros(5)
        self.gap = u * self.b          # knob - screw - (initial offset) within [0, b]: b engaged forward, 0 backward
        self.screw = np.zeros(5)

    def set_knobs(self, angles):
        """New knob angles (5,); returns the screw positions."""
        a = np.asarray(angles, float)
        g = self.gap + (a - self.knob)
        lo, hi = np.zeros(5), self.b
        moved = np.clip(g, lo, hi)
        self.screw = self.screw + (g - moved)
        self.gap, self.knob = moved, a.copy()
        return self.screw.copy()

    def turn(self, k, deg):
        a = self.knob.copy()
        a[k] += deg
        return self.set_knobs(a)

    def phis(self):
        return screw_phis(self.i, self.screw)

    def dz(self, lens):
        return screw_dz(self.i, lens, self.screw)

    def coupling(self, lens, cladding=True):
        px, py = self.phis()
        return coupling(self.i, lens, px, py, self.dz(lens), cladding=cladding)


def screw_phis(inst, screw):
    """Mirror tilts (phi_x (M1, M2), phi_y (M1, M2)) of screw positions (knob degrees), with the cross-talk (the
    calibration bench's phis)."""
    g = G_RAD_PER_DEG
    s, c = screw, inst['crosstalk']
    px = np.array([s[0] + c[1] * s[1], s[2] + c[3] * s[3]]) * g
    py = np.array([s[1] + c[0] * s[0], s[3] + c[2] * s[2]]) * g
    return px, py


def screw_dz(inst, lens, screw):
    return inst['dz0'][lens] + screw[4] * FOCUS_UM_PER_DEG * 1e-6


def meter_uW(total, rng=None, noise=0.002):
    """Power meter reading of a coupled fraction: µW, three significant digits, 0.2 % noise with rng."""
    p = P_LASER_UW * float(total) * (1 + (noise * rng.normal() if rng is not None else 0.0))
    return float(f'{p:.3g}') if p > 0 else 0.0

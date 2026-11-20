"""Reaction bench chemistry (reaction_yield), headless: kinetics, spectra, heat and the private optimum.

Four reactions run in every cuvette compartment (layer) holding R, each at the compartment's cuvette temperature:
    S + R -> P       r1 = k1 [S][R]          (wanted)
    P + R -> Q       r2 = k2 [P][R]          (over-reaction)
    S + R -> D       rD = kD [S][R]          (competing path)
    2 P -> X         rX = kX [P]^2 [R]       (dimerisation of P, catalysed by R: R is not used up)
    K + R -> (inert) at once                 (the quench: K from the QUENCH line destroys R, mole for mole)
S, P and the stocks are stable on their own (no path runs without R). At a hidden temperature T0_C, k1 = k1_ref and
the ratios k2/k1 = rho, kD/k1 = delta, kX/k1 = gamma; away from it k1 follows Arrhenius (E1) and the ratios change by
the activation-energy differences dE2 = E2 - E1, dED = E1 - ED, dEX = EX - E1. S, P and Q absorb in the visible
(registered as kit dyes under neutral names, register_spectra), R, D, X and K do not. The yield of a cuvette is n(P) /
n(S added); S atoms are kept as S + P + Q + D + 2 X.

Heat: the dry block heats towards its set point at up to HEAT_C_PER_S and cools by itself (no cooling, time
constant BLOCK_COOL_TAU_S); its true temperature is the display plus a hidden offset. A cuvette relaxes to the block
(in a well) or to the room (anywhere else) with first-order time constants.

The optimum y* (the private target) is the largest attainable yield over every recipe the dispensers can make for one
product cuvette (S_ML of S STOCK, R and BUFFER in 0.1 mL steps, total RECIPE_MAX_ML at most, READ_MIN_ML at least) and
every set point (1 C steps), made by the protocol this bench and robot attain (yields: the larger of the S and R
parts, with the buffer, preheated in the block, then PRE_AIR_S in the air until MIX, the other added at the station
from its room-temperature bottle, STATION_S from MIX until back in the block), taken out of the block at the best
moment within WINDOW_S of mixing, at least MIN_STAY_S after it went in, and quenched QUENCH_AFTER_S later (carried to
the station in the air). The protocol times are the fastest the reference robot achieved. Stocks are at room temperature; liquids poured together take
their volume-weighted temperature.
"""
from __future__ import annotations

import math

import numpy as np

R_GAS = 8.314
T_REF_K = 323.15
ROOM_C = 22.0
SPECIES = ('rx_s', 'rx_r', 'rx_p', 'rx_q', 'rx_d', 'rx_x')  # neutral internal names (leak scan)
S, RG, P, Q, D, X = SPECIES
K = 'rx_k'                          # the quench (not integrated: it only takes R out, at once)
S_ATOMS = {S: 1, P: 1, Q: 1, D: 1, X: 2}
SET_RANGE_C = (20, 80)
HEAT_C_PER_S = 0.5
BLOCK_COOL_TAU_S = 240.0
TAU_IN_BLOCK_S, TAU_IN_AIR_S = 20.0, 90.0
STEP_ML, RECIPE_MAX_ML, READ_MIN_ML = 0.1, 3.3, 2.0   # 3.3: room for a 0.1 mL quench and the scatter
S_ML = 1.0                          # S STOCK in the product cuvette (the batch size the task fixes)
WINDOW_S = 600.0                    # the optimum takes the cuvette out of the block within this long of mixing
PULL_EVERY_S = 5.0                  # moments considered for taking it out (s after mixing; the reference plans on 5 s)
# The attainable protocol (y* is what this bench and robot can attain). Each time is the fastest the reference
# robot achieved over 51 reaction trials, from the private logs: the block heat log (a cuvette in a well, 1 s rows; in/out half a row outside) and the MIX events.
PRE_AIR_S = 54.0                    # preheated part out of its block well -> MIX at the station (median 58.7 s)
STATION_S = 44.5                    # MIX -> the mixed cuvette back in a block well (median 47.3 s)
MIN_STAY_S = 23.0                   # shortest stay in the well: set down, back to the start pose, grasped (23-26 s)
QUENCH_AFTER_S = 56.7               # out of the well -> the quench's MIX at the station (median 58.6 s)

# name: (lambda_max nm, eps(lambda_max), bands) as chem_solver.DYES; P's eps is drawn per instance (eps_factor)
SPECTRA = {
    S: (430.0, 2.0e4, [(430.0, 35.0, 1.0), (370.0, 25.0, 0.20)]),
    P: (520.0, 3.0e4, [(522.0, 30.0, 1.0), (485.0, 24.0, 0.25)]),
    Q: (585.0, 4.0e4, [(587.0, 25.5, 1.0), (555.0, 20.0, 0.10)]),
}


def register_spectra():
    """Register S, P and Q as kit dyes (chem_solver), once per process (the bench and the backend call it)."""
    import chem_solver as CS
    for name, (lam, eps, bands) in SPECTRA.items():
        CS.register_dye(name, lam, eps, bands)

HIDDEN = {                          # name: (low, high); LOG ones log-uniform, the others uniform
    'k1_ref': (25.0, 100.0),        # M^-1 s^-1 at T0
    'E1': (65e3, 85e3),
    'T0_C': (30.0, 70.0),           # where k1_ref and the ratios below hold
    'rho': (0.15, 0.5),             # k2 / k1 at T0
    'delta': (0.05, 0.3),           # kD / k1 at T0
    'gamma': (500.0, 5000.0),       # kX / k1 at T0, M^-1
    'dE2': (40e3, 70e3),            # E2 - E1
    'dED': (35e3, 50e3),            # E1 - ED (ED >= 15 kJ/mol)
    'dEX': (-30e3, 30e3),           # EX - E1
    'block_offset_C': (-0.5, 0.5),
}
LOG = ('k1_ref', 'rho', 'delta', 'gamma')


def draw(rng):
    return {k: float(np.exp(rng.uniform(math.log(lo), math.log(hi))) if k in LOG else rng.uniform(lo, hi))
            for k, (lo, hi) in HIDDEN.items()}


def rate_constants(h, t_c):
    """(k1, k2, kD, kX) at t_c (C; scalar or array) for hidden parameters h."""
    inv = 1.0 / (np.asarray(t_c, float) + 273.15)
    x0 = -(inv - 1.0 / (h['T0_C'] + 273.15)) / R_GAS
    k1 = h['k1_ref'] * np.exp(h['E1'] * x0)
    return k1, h['rho'] * k1 * np.exp(h['dE2'] * x0), h['delta'] * k1 * np.exp(-h['dED'] * x0), \
        h['gamma'] * k1 * np.exp(h['dEX'] * x0)


def _deriv(c, k1, k2, kd, kx):
    s, r, p = c[0], c[1], c[2]
    r1, r2, rd, rx = k1 * s * r, k2 * p * r, kd * s * r, kx * p * p * r
    return np.stack([-r1 - rd, -r1 - r2 - rd, r1 - r2 - 2 * rx, r2, rd, rx])


def advance(c, k, dt_s):
    """RK4 over dt_s for concentrations c (6, ...) (M), sub-stepped so no rate turns over more than 10 % a step."""
    k1, k2, kd, kx = k
    fastest = np.max((k1 + kd) * (c[0] + c[1]) + k2 * (c[1] + c[2]) + kx * c[2] * (4 * c[1] + c[2]))
    n = max(1, int(math.ceil(float(fastest) * dt_s / 0.1)))
    h = dt_s / n
    for _ in range(n):
        a = _deriv(c, k1, k2, kd, kx)
        b = _deriv(c + h / 2 * a, k1, k2, kd, kx)
        d = _deriv(c + h / 2 * b, k1, k2, kd, kx)
        e = _deriv(c + h * d, k1, k2, kd, kx)
        c = np.maximum(c + h / 6 * (a + 2 * b + 2 * d + e), 0.0)
    return c


def react(composition, t_c, h, dt_s):
    """One compartment over dt_s at t_c: returns the new composition dict (other species pass through). The
    quench K first takes out as much R as it can (mole for mole), then the reactions run."""
    out = dict(composition)
    if out.get(K, 0.0) > 0 and out.get(RG, 0.0) > 0:
        n = min(out[K], out[RG])
        out[K], out[RG] = out[K] - n, out[RG] - n
    c = np.array([out.get(s, 0.0) for s in SPECIES])
    if c[1] <= 0 or (c[0] <= 0 and c[2] <= 0):
        return out
    c = advance(c, rate_constants(h, t_c), dt_s)
    out.update({s: float(v) for s, v in zip(SPECIES, c)})
    return out


def s_moles(moles):
    """Moles of S added, from the moles of every species ({species: mol})."""
    return sum(n * moles.get(k, 0.0) for k, n in S_ATOMS.items())


def yield_of(composition_M, volume_mL, s_added_mol):
    return composition_M.get(P, 0.0) * volume_mL * 1e-3 / s_added_mol if s_added_mol > 0 else 0.0


class Block:
    """Dry block heater: set point (display units), true temperature = display + offset."""

    def __init__(self, offset_c, set_c=ROOM_C):
        self.offset = offset_c
        self.set_c = set_c
        self.display_c = ROOM_C

    @property
    def true_c(self):
        return self.display_c + self.offset

    def nudge(self, step_c):
        self.set_c = int(min(max(self.set_c + step_c, SET_RANGE_C[0]), SET_RANGE_C[1]))

    def update(self, dt_s):
        target = max(self.set_c, ROOM_C)
        if self.display_c < target:
            self.display_c = min(target, self.display_c + HEAT_C_PER_S * dt_s)
        else:                                   # no cooling: it drifts down to the set point (or the room)
            self.display_c = target + (self.display_c - target) * math.exp(-dt_s / BLOCK_COOL_TAU_S)


def cuvette_temperature(t_c, env_c, in_block, dt_s):
    tau = TAU_IN_BLOCK_S if in_block else TAU_IN_AIR_S
    return env_c + (t_c - env_c) * np.exp(-dt_s / tau)


def stock_concentrations(labels):
    """Bottle labels (uM) -> composition of each line (M)."""
    return {'S STOCK': {S: labels['c_S_uM'] * 1e-6}, 'R STOCK': {RG: labels['c_R_uM'] * 1e-6},
            'BUFFER': {}, 'P STD': {P: labels['c_P_uM'] * 1e-6}}


def recipes(s_ml=S_ML, max_ml=RECIPE_MAX_ML, step=STEP_ML):
    """Every (V_S, V_R, V_buffer) mL of a product cuvette: V_S = s_ml, R and BUFFER in steps, within capacity and with
    enough liquid to read."""
    n, i = int(round(max_ml / step)), int(round(s_ml / step))
    return [(round(i * step, 1), round(j * step, 1), round(b * step, 1))
            for j in range(1, n + 1 - i) for b in range(0, n + 1 - i - j) if (i + j + b) * step >= READ_MIN_ML - 1e-9]


def mix_temperature(parts):
    """Temperature of liquids poured together: [(mL, C), ...] (heat capacities alike)."""
    v = sum(p[0] for p in parts)
    return sum(p[0] * p[1] for p in parts) / v if v > 0 else ROOM_C


def yields(h, labels, set_c, recipe_arr, window_s=WINDOW_S, after_s=QUENCH_AFTER_S, pull_every_s=PULL_EVERY_S, dt_s=1.0,
           station_s=STATION_S, pre_air_s=PRE_AIR_S, min_stay_s=MIN_STAY_S):
    """Attainable yields, array (len(set_c), len(recipes)): the larger of the S and R parts (with the buffer) waits in a
    block well until it has the block's true temperature, then spends pre_air_s in the air (nothing reacts yet); the
    other is added from its room-temperature bottle at the station and mixed; the mixed cuvette is back in the well
    station_s after mixing (relaxing to the room until then). It is taken out at the best of the moments pull_every_s
    apart (s after mixing) within window_s of mixing and at least min_stay_s after it went in, and quenched after_s
    later, the cuvette relaxing to the room meanwhile. Each set point on its own, all recipes together. Returns
    (yields, best moment to take it out, s after mixing), both (len(set_c), len(recipes))."""
    vs, vr, vb = recipe_arr[:, 0], recipe_arr[:, 1], recipe_arr[:, 2]
    vt = vs + vr + vb
    warm = (np.where(vs >= vr, vs, vr) + vb) / vt               # fraction that was preheated
    s0 = labels['c_S_uM'] * 1e-6 * vs / vt
    every = int(round(pull_every_s / dt_s))
    out, pull = np.empty((len(set_c), len(recipe_arr))), np.empty((len(set_c), len(recipe_arr)))
    for i, t in enumerate(np.asarray(set_c, float)):
        block = t + h['block_offset_C']
        preheated = cuvette_temperature(block, ROOM_C, False, pre_air_s)
        temp = warm * preheated + (1 - warm) * ROOM_C
        c = np.zeros((len(SPECIES), len(recipe_arr)))
        c[0], c[1] = s0, labels['c_R_uM'] * 1e-6 * vr / vt
        pulled_c, pulled_t = [], []
        for n in range(1, int(round(window_s / dt_s)) + 1):
            in_block = n * dt_s > station_s
            temp = cuvette_temperature(temp, block if in_block else ROOM_C, in_block, dt_s)
            c = advance(c, rate_constants(h, temp), dt_s)
            if n % every == 0 and n * dt_s >= station_s + min_stay_s - 1e-9:
                pulled_c.append(c)
                pulled_t.append(temp)
        c, temp = np.concatenate(pulled_c, axis=1), np.concatenate(pulled_t)
        for _ in range(int(round(after_s / dt_s))):
            temp = cuvette_temperature(temp, ROOM_C, False, dt_s)
            c = advance(c, rate_constants(h, temp), dt_s)
        y = (c[2] / np.tile(s0, len(pulled_c))).reshape(len(pulled_c), -1)
        out[i], best = y.max(0), y.argmax(0)
        first = int(np.ceil((station_s + min_stay_s - 1e-9) / pull_every_s)) * pull_every_s
        pull[i] = first + best * pull_every_s
    return out, pull


def optimum(h, labels, set_grid=None, recipe_list=None):
    """(y*, set_c*, recipe*, moment to take it out (s after mixing)) and the full yield table."""
    set_grid = np.arange(SET_RANGE_C[0], SET_RANGE_C[1] + 1) if set_grid is None else np.asarray(set_grid)
    rec = np.array(recipes() if recipe_list is None else recipe_list, float)
    y, pull = yields(h, labels, set_grid, rec)
    i, j = np.unravel_index(int(np.argmax(y)), y.shape)
    return float(y[i, j]), int(set_grid[i]), tuple(float(v) for v in rec[j]), float(pull[i, j]), y

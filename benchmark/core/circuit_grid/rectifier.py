"""rectifier_supply on the circuit board: a fixed AC SOURCE + BRIDGE module (12 V peak, 50 Hz, 25 ohm source
resistance, a bridge of four equal Shockley diodes; an ON/OFF push button and an ON lamp) whose DC output + and - are
strapped to two board pads, three bipolar capacitor blocks 100/220/470 uF (+-10 %, hidden), load resistor blocks
150/330/470/680/820/1k/3k3 (+-1 %, hidden), five wires, the two voltmeters, the ammeter, the logger channels and the
logger unit with kHz rates (the scope of this bench).

RectBench (variants.NetBench): the bridge port and the linear network of the seated blocks with up to three
capacitor states, reduced at every change of the circuit (MNA unit solves: each capacitor a voltage source, the bridge
a current source) to
    C dv/dt = G v + h i,   u = a . v + z i,   instrument values = M [v; i],
v the capacitor (group) voltages, i the bridge current, u the output (port) voltage. Capacitors seated on the same
two nodes are one state (their charges are shared when they meet); a loop of capacitors that is not that (singular
with ideal capacitors) is solved with an ESR of ESR_FALLBACK_OHM per capacitor and backward Euler. The state moves at
INTEGRATION_HZ by the trapezoidal rule with the bridge current implicit at each step (rect_kernel.c): the bridge
conducts while the rectified source voltage exceeds the output, 25 i + 2 n V_T ln(1 + i / I_s) = |12 sin(2 pi 50 t)|
- u. A capacitor off the board keeps its charge; one whose pins are joined by a wire is discharged at once.
Meters show the mean over their conversion (since the last change of the circuit, at most CONVERSION_S): a DC meter
on a rectified output reads its average. The logger samples the instantaneous channel voltages at its rate.

Submission: nothing runs after it; the gate rectifier_grid_protocol judges the final supply from the logger record the
submission cites (benchmark/evaluation/circuit_grid_variants.py).
"""
from __future__ import annotations

import ctypes
import math
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import layout as L
from .instruments import (CONVERSION_S, DISPLAY_S, LOGGER_INPUT_OHM, LOGGER_NOISE_V, LOGGER_RESOLUTION_V, Logger,
                          VOLTMETER_OHM)
from .solver import Netlist, solve
from .variants import INSTRUMENTS, NetBench, Profile, _Result, build_scene, register

HERE = Path(__file__).resolve().parent
SOURCE_PEAK_V, SOURCE_HZ, SOURCE_R_OHM = 12.0, 50.0, 25.0
DIODE_IS_A, DIODE_T_K = 1.0e-8, 298.0
K_B_EV = 8.617333262145e-5
INTEGRATION_HZ = 24000
ESR_FALLBACK_OHM = 0.01
CAPS = {'C100': 1.0e-4, 'C220': 2.2e-4, 'C470': 4.7e-4}
CAP_PRINT = {'C100': '100uF', 'C220': '220uF', 'C470': '470uF'}
CAP_FACTOR = {'C100': 'c100_factor', 'C220': 'c220_factor', 'C470': 'c470_factor'}
LOADS = {'R150': 150.0, 'R330': 330.0, 'R470': 470.0, 'R680': 680.0, 'R820': 820.0, 'R1k': 1000.0, 'R3k3': 3300.0}
LOAD_PRINT = {'R150': '150', 'R330': '330', 'R470': '470', 'R680': '680', 'R820': '820', 'R1k': '1k',
              'R3k3': '3k3'}
CAP_COLOUR = (0.10, 0.22, 0.55)
SCOPE_RATES_HZ = (1000, 2000, 4000, 8000)
SCOPE_RECORD_S = (1, 5)
SCOPE_RANGE_V = 20.0
FINAL_LOAD_OHM, FINAL_LOAD_TOL = 820.0, 0.03
OPEN_OHM = 1.0e9


# --- kernel -----------------------------------------------------------------------------------------------------------
_LIB = None


def kernel():
    """rect_kernel.so next to this file (built from rect_kernel.c; rebuilt into /tmp when it does not load)."""
    global _LIB
    if _LIB is None:
        path = HERE / 'rect_kernel.so'
        try:
            lib = ctypes.CDLL(str(path))
        except OSError:
            path = Path('/tmp/embodiedlab_rect_kernel.so')
            subprocess.run(['cc', '-O2', '-shared', '-fPIC', str(HERE / 'rect_kernel.c'), '-o', str(path), '-lm'],
                           check=True)
            lib = ctypes.CDLL(str(path))
        D = ctypes.POINTER(ctypes.c_double)
        lib.rect_advance.argtypes = ([ctypes.c_int, D, D, D, D, D, D] + [ctypes.c_double] * 7
                                     + [ctypes.c_longlong, ctypes.c_int, ctypes.c_double, ctypes.c_int, D])
        lib.rect_advance.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def _ptr(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class Reduced:
    """The circuit at one configuration, reduced to the bridge port and the capacitor states (see the module
    docstring). groups: [(+ root, - root, farad, [(block, sign)])]; inst: instrument -> coefficient row over
    [v..., i]; port: coefficients of the output voltage."""

    def __init__(self, groups, G, h, a, z, inst, scheme, dt):
        n = len(groups)
        self.groups, self.n, self.scheme = groups, n, scheme
        self.inst = inst
        self.port = np.concatenate([a, [z]])
        if n:
            C = np.array([g[2] for g in groups])
            K, k = G / C[:, None], h / C
            eye = np.eye(n)
            if scheme == 'trapezoid':
                Mx = eye - dt / 2 * K
                Pm = np.linalg.solve(Mx, eye + dt / 2 * K)
                p = q = np.linalg.solve(Mx, dt / 2 * k)
            else:
                Mx = eye - dt * K
                Pm = np.linalg.inv(Mx)
                p, q = np.zeros(n), np.linalg.solve(Mx, dt * k)
        else:
            Pm, p, q = np.zeros((0, 0)), np.zeros(0), np.zeros(0)
        self.Pm = np.ascontiguousarray(Pm, float)
        self.p, self.q = np.ascontiguousarray(p, float), np.ascontiguousarray(q, float)
        self.aPm = np.ascontiguousarray(a @ Pm if n else np.zeros(0), float)
        self.ap = float(a @ p) if n else 0.0
        self.c1 = float((a @ q if n else 0.0) + z)


# --- profile ----------------------------------------------------------------------------------------------------------
@register
class RectProfile(Profile):
    recipe = 'circuit_grid_rectifier'
    blocks = {**{k: L.BLOCKS[k] for k in ('V1', 'V2', 'A', 'CH1', 'CH2')},
              **{k: ('capacitor', CAP_PRINT[k], k, 0.040) for k in CAPS},
              **{k: ('resistor', LOAD_PRINT[k], k, 0.040) for k in LOADS},
              **{f'W{k}': ('wire', '', 'wire', 0.035) for k in range(1, 6)}}
    tray_layout = (('V1', 'V2', 'A', 'CH1', 'CH2'),
                   ('C100', 'C220', 'C470', 'R150', 'R330'),
                   ('R470', 'R680', 'R820', 'R1k', 'R3k3'),
                   ('W1', 'W2', 'W3', 'W4', 'W5'))
    supply = True
    dut = 'output'
    terminals = {}
    colours = {'capacitor': CAP_COLOUR}

    def configuration(self, instance):
        from instance_episodes import appearance_seed
        h = instance['hidden']
        return {'recipe': self.recipe,
                'cap_f': {k: c * float(h[CAP_FACTOR[k]]) for k, c in CAPS.items()},
                'block_ohm': {k: v * float(h[f'load_factor_{LOAD_PRINT[k]}']) for k, v in LOADS.items()},
                'diode_n': float(h['diode_n']), 'supply_v': 0.0, 'supply_limit_a': 0.0,
                'noise_seed': appearance_seed('grid-noise:' + instance['instance_id'])}

    def fixture(self, config):
        return []

    def device(self, config):
        return {'cap_nominal_f': dict(CAPS), 'load_nominal_ohm': dict(LOADS), 'source_peak_v': SOURCE_PEAK_V,
                'source_hz': SOURCE_HZ, 'source_r_ohm': SOURCE_R_OHM, 'integration_hz': INTEGRATION_HZ,
                'final_load_ohm': FINAL_LOAD_OHM, 'final_load_tolerance': FINAL_LOAD_TOL}

    def make_bench(self, episode_id, config, **kwargs):
        return RectBench(episode_id, self, config, **kwargs)

    def build(self, stage):
        built = build_scene(stage, self, build_ac_module)
        for block in CAPS:
            dress_capacitor(stage, block, built['blocks'][block])
        rebuild_scope_display(stage)
        return built

    def sync_displays(self, backend):
        """The logger's CH1 on this bench reads +-19.99 V (two decimals); returns what it shows."""
        from . import asset as A
        bench = backend.bench
        if bench is None:
            return None
        v = max(-19.99, min(19.99, bench.logger.live_v))
        digits = f'{abs(v):05.2f}'.replace('.', '')
        backend.displays._readout(A.ROOT + '/Logger/Ch1', 4).show('-' if v < 0 else ' ', digits, 1)
        return digits

    def terminal_extra(self, backend):
        bench = backend.bench
        return {'capacitor_v': {k: round(float(v), 6) for k, v in bench.cap_volts.items()},
                'output': deepcopy(bench.analysis['output'])}

    def public_apparatus(self):
        looks = {'voltmeter': L.BLOCK_LOOKS['voltmeter'], 'ammeter': L.BLOCK_LOOKS['ammeter'],
                 'channel': 'red band at the + end of the base', 'wire': 'grey, no resistance',
                 'resistor': 'tan handle printed with its value in ohm',
                 'capacitor': 'blue handle printed with its value; not polarised'}
        handles = {'CH1': 'yellow handle', 'CH2': 'cyan handle'}
        return {
            'board': {'what': 'dark green plate with brass node pads; column numbers and row numbers are printed on '
                              'its edges',
                      'connection': 'everything with a pin or a terminal on one pad is connected there; pads are '
                                    'otherwise insulated from each other'},
            'fixed_terminals': {'output_plus': 'red strap from the module DC + terminal to a board pad',
                                'output_minus': 'black strap from the module DC - terminal to a board pad'},
            'source_module': {'what': 'grey unit printed AC SOURCE + BRIDGE with an ON/OFF push button and an ON lamp'},
            'blocks': {b: {'kind': k, 'printed': p or 'none', 'looks': handles.get(b, looks[k]), 'tray_seat': b,
                           'polarised': k in L.POLARISED} for b, (k, p, _, _) in self.blocks.items()},
            'tray': {'what': 'green-grey tray with one labelled well per block',
                     'seats': {b: {'printed': p or 'W', 'takes': c} for b, (_, p, c, _) in self.blocks.items()}},
            'logger': {'what': 'dark blue two-channel logger unit printed LOGGER',
                       'keys': [L.LOGGER_KEY_PRINT.get(k, k) for row in L.LOGGER_KEYS for k in row if k is not None],
                       'display': ['CH1 live (V)', 'SET (s)', 'RATE (kHz)', 'NO', 'LEFT', 'REC lamp', 'ERR lamp']},
        }


# --- scene ------------------------------------------------------------------------------------------------------------
def build_ac_module(stage, material, key_material):
    """The AC SOURCE + BRIDGE module where the platform's DC supply stands: the same housing, push button (its
    ON/OFF), the ON lamp, prints, and the DC output straps to the supply pads."""
    from pxr import UsdGeom
    from . import asset as A
    from .display import INK, Frame, text
    A._housing(stage, A.ROOT + '/Supply/Housing', L.SUPPLY_BOX, A.SUPPLY_COLOUR, material)
    top = L.supply_top_z()
    bx, by = L.SUPPLY_BUTTON_XY
    A.spring_cap(stage, A.SUPPLY_BUTTON, (bx, by, top + L.CAP_LIFT + L.CAP_H / 2), (L.KEY_SIZE, L.KEY_SIZE, L.CAP_H),
                 (0.92, 0.92, 0.90), key_material)
    A._box(stage, A.ROOT + '/Supply/ButtonCollar', (L.KEY_SIZE + 0.004, L.KEY_SIZE + 0.004, L.CAP_LIFT - 0.001),
           (bx, by, top + (L.CAP_LIFT - 0.001) / 2), (0.10, 0.10, 0.11), collide=False)
    frame = lambda x, y: Frame((x, y), A.READ, A.UP, top)       # noqa: E731
    text(stage, A.SUPPLY_BUTTON + '/Print', 'I/O', Frame((0, 0), A.READ, A.UP, L.CAP_H / 2), 0, 0, 0.007, INK,
         align='centre')
    text(stage, A.ROOT + '/Supply/Title', 'AC SOURCE + BRIDGE', frame(0.628, -0.070), 0, 0, 0.0055, INK,
         align='centre')
    text(stage, A.ROOT + '/Supply/ButtonLabel', 'ON/OFF', frame(bx + 0.021, by), 0, 0, 0.005, INK, align='centre')
    lx, ly = L.SUPPLY_LAMPS['ON']
    A._disc(stage, A.ROOT + '/Supply/LampOnDim', 0.0045, (lx, ly, top + 0.0004), (0.10, 0.10, 0.09))
    A._disc(stage, A.LAMP_PATHS['ON'], 0.0046, (lx, ly, top + 0.0007), A.LAMP_ON['ON'], visible=False,
            material=A._render_material(stage, A.ROOT + '/Supply/Looks/LampON', A.LAMP_ON['ON'], emission=4.0))
    text(stage, A.ROOT + '/Supply/LampONLabel', 'ON', frame(lx, ly + 0.010), 0, 0, 0.006, INK)
    UsdGeom.Xform.Define(stage, A.LAMP_PATHS['CC'])          # no CC lamp on this unit (the platform toggles it off)
    text(stage, A.ROOT + '/Supply/Spec1', '12 V PEAK 50 HZ', frame(0.712, -0.070), 0, 0, 0.0050, INK, align='centre')
    text(stage, A.ROOT + '/Supply/Spec2', 'DC OUT', frame(0.724, -0.070), 0, 0, 0.0050, INK, align='centre')
    x_edge = L.SUPPLY_BOX[0][1]
    for node, colour, mark in ((L.SUPPLY_PLUS, A.RED, '+'), (L.SUPPLY_MINUS, (0.03, 0.03, 0.03), '-')):
        nx, ny = L.node_xy(node)
        name = 'Plus' if mark == '+' else 'Minus'
        A._strap(stage, A.ROOT + f'/Supply/Strap{name}', x_edge, nx - L.PAD_HALF + 0.004, ny, colour)
        text(stage, A.ROOT + f'/Supply/Mark{name}', mark, frame(x_edge - 0.008, ny), 0, 0, 0.009, colour,
             align='centre')
    return A.SUPPLY_BUTTON


def dress_capacitor(stage, block, path):
    """A capacitor block as asset.build_block builds kind 'capacitor' (blue base and handle): its value printed on
    the handle top."""
    from .display import WHITE, Frame, text
    local = Frame((0, 0), (1, 0), (0, 1), L.BLOCK_TOP)
    text(stage, path + '/Value', L.BLOCKS[block][1], local, 0, 0, 0.0045, WHITE, align='centre')


def rebuild_scope_display(stage):
    """The logger's CH1 readout with a second decimal-point position (+-19.99 V) and the rate unit kHz."""
    from . import asset as A
    from .display import WHITE, Frame, Readout, text
    (wx0, wx1), (wy0, wy1) = L.LOGGER_DISPLAY
    frame = Frame(((wx0 + wx1) / 2, (wy0 + wy1) / 2), A.READ, A.UP, L.logger_top_z())
    stage.RemovePrim(A.ROOT + '/Logger/Ch1')
    d = L.LOGGER_CH1
    Readout(stage, A.ROOT + '/Logger/Ch1', frame, u0=d['u0'], v0=d['v0'], digit_w=d['digit_w'], digit_h=d['digit_h'],
            stroke=d['stroke'], pitch=d['pitch'], count=d['count'], points=(0, 1),
            material=A.display_looks(stage)['DisplayLit'])
    stage.RemovePrim(A.ROOT + '/Logger/RateUnit')
    text(stage, A.ROOT + '/Logger/RateUnit', 'KHZ', frame, 0.049, -0.006 + 0.003, 0.005, WHITE, dz=0.0008)


# --- logger -----------------------------------------------------------------------------------------------------------
class ScopeLogger(Logger):
    """The platform logger with this bench's rates (kHz), lengths (s) and range (+-20 V); it samples the
    instantaneous channel voltages between physics ticks (INTEGRATION_HZ sub-steps)."""

    def __init__(self, episode_id, *, noise_seed, physics_hz=240, max_records=20):
        super().__init__(episode_id, noise_seed=noise_seed, physics_hz=physics_hz, max_records=max_records)
        self.rate_index = 0
        self.substeps = INTEGRATION_HZ // physics_hz

    @property
    def rate_hz(self):
        return SCOPE_RATES_HZ[self.rate_index]

    def press(self, tick, key, remaining_ticks):
        if key == 'rate' and self.active is None:
            self.error = False
            self.rate_index = (self.rate_index + 1) % len(SCOPE_RATES_HZ)
            effect = f'rate_{self.rate_hz}'
            self.presses.append({'tick': tick, 'key': key, 'effect': effect})
            return effect
        return super().press(tick, key, remaining_ticks)

    def _rec(self, tick, remaining_ticks):
        value = int(self.entry) if self.entry else None
        if value is None or not SCOPE_RECORD_S[0] <= value <= SCOPE_RECORD_S[1]:
            self.error, self.entry = True, ''
            return 'error_length'
        if self.records_left <= 0:
            self.error = True
            return 'error_no_records_left'
        ticks = value * self.physics_hz
        if ticks > remaining_ticks:
            self.error = True
            return 'error_budget'
        self.number = len(self.records) + 1
        eid = f'{self.episode_id}:logger:{self.number}'
        start = tick * self.substeps
        self.active = dict(evidence_id=eid, number=self.number, start_tick=tick, end_tick=tick + ticks,
                           interval=INTEGRATION_HZ // self.rate_hz, start_sub=start,
                           end_sub=start + ticks * self.substeps, rate_hz=self.rate_hz, duration_s=float(value),
                           rows=[], counts={'ok': 0, 'over_range': 0})
        return 'start'

    def _measure(self, value):
        v = value + self._rng.normal(0.0, LOGGER_NOISE_V)
        status = 'ok'
        if abs(v) > SCOPE_RANGE_V:
            status, v = 'over_range', math.copysign(SCOPE_RANGE_V, v)
        return round(round(v / LOGGER_RESOLUTION_V) * LOGGER_RESOLUTION_V, 3), status

    def samples(self, tick, ch1, ch2):
        """One physics tick's sub-step values of the two channels (arrays of `substeps`, the last at `tick`)."""
        if self.live_tick is None or tick % round(DISPLAY_S * self.physics_hz) == 0:
            self.live_v, _ = self._measure(float(ch1[-1]))
            self.live_tick = tick
        a = self.active
        if a is None:
            return
        first = (tick - 1) * self.substeps + 1                 # global sub-step index of ch1[0]
        for j in range(len(ch1)):
            g = first + j
            if g <= a['start_sub'] or g > a['end_sub'] or (g - a['start_sub']) % a['interval']:
                continue
            values, status = [], 'ok'
            for v in (float(ch1[j]), float(ch2[j])):
                if not math.isfinite(v):
                    raise ValueError('non-finite channel voltage')
                m, s = self._measure(v)
                values.append(m)
                status = 'over_range' if s == 'over_range' else status
            a['rows'].append(dict(sample_id=len(a['rows']) + 1, t_s=round(g / INTEGRATION_HZ, 7), status=status,
                                  ch1_V=values[0], ch2_V=values[1]))
            a['counts'][status] += 1
        if tick >= a['end_tick']:
            eid = a['evidence_id']
            self._finish()
            self.records[eid]['range_V'] = [-SCOPE_RANGE_V, SCOPE_RANGE_V]

    def open_record(self):
        out = super().open_record()
        if out is not None and out['rows']:
            out['last_tick'] = math.ceil(out['rows'][-1]['t_s'] * self.physics_hz - 1e-9)
        return out

    def display(self):
        shown = super().display()
        shown['rate_hz'] = self.rate_hz // 1000            # the RATE field shows kHz on this bench
        return shown


# --- bench ------------------------------------------------------------------------------------------------------------
class RectBench(NetBench):
    def __init__(self, episode_id, profile, config, *, physics_hz=240, max_records=20):
        if INTEGRATION_HZ % physics_hz:
            raise ValueError('the integration clock must divide into physics ticks')
        self.cap_f = {k: float(v) for k, v in config['cap_f'].items()}
        self.cap_volts = {k: 0.0 for k in self.cap_f}
        self.diode_a = 2 * float(config['diode_n']) * K_B_EV * DIODE_T_K
        self.i_bridge = np.zeros(1)
        self.window = []
        self.capture = None
        self.reduced = None
        super().__init__(episode_id, profile, config, physics_hz=physics_hz, max_records=max_records)
        self.logger = ScopeLogger(episode_id, noise_seed=int(self.logger._rng.integers(2 ** 32)),
                                  physics_hz=physics_hz, max_records=max_records)
        self._dummy = np.zeros(1)
        self.substeps = INTEGRATION_HZ // physics_hz

    # --- circuit ----------------------------------------------------------------------------------------------------
    def _port(self, uf):
        return uf.find(L.node_name(L.SUPPLY_PLUS)), uf.find(L.node_name(L.SUPPLY_MINUS))

    def _resistive(self, uf, ammeter_range=None):
        """[(name, + root, - root, ohm)] of the resistive elements, [(cap block, + root, - root)] of the capacitors
        whose pins are on two nodes; capacitors with both pins on one node are listed in `shorted`."""
        res, caps, shorted = [], [], []
        for block, place in self.placement.items():
            if place[0] != 'board':
                continue
            kind = self.blocks[block][0]
            if kind in ('wire', 'switch', 'changeover'):
                continue
            plus, minus = uf.find(L.node_name(place[1])), uf.find(L.node_name(place[2]))
            if kind == 'capacitor':
                (shorted if plus == minus else caps).append((block, plus, minus))
                continue
            if plus == minus:
                continue
            if kind in ('voltmeter', 'channel'):
                res.append((block, plus, minus, VOLTMETER_OHM if kind == 'voltmeter' else LOGGER_INPUT_OHM))
            elif kind == 'ammeter':
                meter = self.meters[block]
                res.append((block, plus, minus, meter.ranges[meter.range if ammeter_range is None else
                                                              ammeter_range][4]))
            else:
                res.append((block, plus, minus, self.ohm[block]))
        return res, caps, shorted

    def _groups(self, caps):
        """Capacitors on the same two nodes as one state; their charges shared (the group's voltage set)."""
        groups = {}
        for block, plus, minus in caps:
            key = frozenset((plus, minus))
            if key not in groups:
                groups[key] = [plus, minus, 0.0, []]
            g = groups[key]
            g[3].append((block, 1.0 if plus == g[0] else -1.0))
            g[2] += self.cap_f[block]
        out = []
        for plus, minus, farad, members in groups.values():
            v = sum(self.cap_f[b] * s * self.cap_volts[b] for b, s in members) / farad
            for b, s in members:
                self.cap_volts[b] = s * v
            out.append((plus, minus, farad, members))
        return out

    def _reduce(self, uf, res, groups, ammeter_range=None, esr=0.0):
        P, M = self._port(uf)
        n = len(groups)
        inst_nodes = {}
        for block in INSTRUMENTS:
            place = self.placement[block]
            if place[0] == 'board':
                inst_nodes[block] = (uf.find(L.node_name(place[1])), uf.find(L.node_name(place[2])))

        def unit(j):
            net = Netlist(reference=M)
            for name, a, b, ohm in res:
                net.resistor(name, a, b, ohm)
            for k, (a, b, _, _) in enumerate(groups):
                if esr:
                    net.resistor(f'C:{k}:esr', a, f'C:{k}:int', esr)
                    net.vsource(f'C:{k}', f'C:{k}:int', b, 1.0 if k == j else 0.0)
                else:
                    net.vsource(f'C:{k}', a, b, 1.0 if k == j else 0.0)
            if P != M:
                net.isource('__bridge', M, P, 1.0 if j == n else 0.0)
            sol = solve(net)
            cap_i = [float(sol.i[f'C:{k}']) for k in range(n)]
            row = {}
            for block, (a, b) in inst_nodes.items():
                if a == b:
                    row[block] = 0.0
                elif self.blocks[block][0] == 'ammeter':
                    row[block] = float(sol.i.get(block, 0.0))
                else:
                    row[block] = sol.vab(a, b)
            return sol.vab(P, M) if P != M else 0.0, cap_i, row

        cols = [unit(j) for j in range(n + 1)]
        a = np.array([c[0] for c in cols[:n]])
        z = cols[n][0]
        G = np.array([[cols[j][1][k] for j in range(n)] for k in range(n)]).reshape(n, n)
        h = np.array(cols[n][1])
        inst = {block: np.array([cols[j][2].get(block, 0.0) for j in range(n + 1)]) for block in INSTRUMENTS}
        dt = 1.0 / INTEGRATION_HZ
        return Reduced(groups, G, h, a, z, inst, 'backward_euler' if esr else 'trapezoid', dt)

    def _load_ohm(self, uf, res):
        """DC resistance across the output with the capacitors removed (None: open)."""
        P, M = self._port(uf)
        if P == M:
            return 0.0
        net = Netlist(reference=M)
        for name, a, b, ohm in res:
            net.resistor(name, a, b, ohm)
        net.isource('__test', M, P, 1.0)
        r = solve(net).vab(P, M)
        return None if r > OPEN_OHM else float(r)

    def _analyse(self, full=True, ammeter_range=None):
        uf = self._merge()
        res, caps, shorted = self._resistive(uf, ammeter_range)
        for block, _, _ in shorted:
            self.cap_volts[block] = 0.0                           # its pins joined: discharged at once
        groups = self._groups(caps)
        try:
            reduced = self._reduce(uf, res, groups, ammeter_range)
        except np.linalg.LinAlgError:                        # a loop of capacitors: each with a small ESR
            reduced = self._reduce(uf, res, groups, ammeter_range, esr=ESR_FALLBACK_OHM)
        P, M = self._port(uf)
        across = sorted(b for b, plus, minus in caps if {plus, minus} == {P, M})
        topology = {}
        for block in INSTRUMENTS:
            place = self.placement[block]
            if place[0] != 'board':
                topology[block] = {'seated': False, 'across_output': False}
                continue
            a, b = uf.find(L.node_name(place[1])), uf.find(L.node_name(place[2]))
            topology[block] = {'seated': True, 'across_output': P != M and {a, b} == {P, M}}
        output = {'shorted': P == M, 'caps_on_board': sorted(b for b, _, _ in caps + shorted),
                  'caps_across': across, 'load_ohm': self._load_ohm(uf, res), 'scheme': reduced.scheme}
        self.reduced = reduced
        self.window = []
        values = self.analysis['values'] if self.analysis is not None else {k: 0.0 for k in INSTRUMENTS}
        return {'uf': uf, 'elements': [(n, p, m, 'R', o) for n, p, m, o in res], 'result':
                _Result(None, 'CV' if self.supply_on else 'OFF'), 'values': dict(values), 'currents': {},
                'i_dut': 0.0, 'i_ab': 0.0, 'sensitivity': 0.0, 'topology': topology, 'output': output,
                'groups': [[p, m, f, [list(x) for x in members]] for p, m, f, members in groups]}

    # --- time -------------------------------------------------------------------------------------------------------
    def advance(self, tick):
        """Integrate one physics tick; returns the sub-step rows [v..., i] (substeps x (n + 1))."""
        red = self.reduced
        n = red.n
        v = np.ascontiguousarray([g[3][0][1] * self.cap_volts[g[3][0][0]] for g in red.groups], float)
        out = np.zeros((self.substeps, n + 1))
        d = self._dummy
        kernel().rect_advance(n, _ptr(v if n else d), _ptr(self.i_bridge), _ptr(red.Pm if n else d),
                              _ptr(red.p if n else d), _ptr(red.q if n else d), _ptr(red.aPm if n else d),
                              red.ap, red.c1, SOURCE_PEAK_V, SOURCE_HZ, SOURCE_R_OHM, self.diode_a, DIODE_IS_A,
                              (tick - 1) * self.substeps, self.substeps, float(INTEGRATION_HZ), int(self.supply_on),
                              _ptr(out))
        if not np.isfinite(out).all():
            raise RuntimeError('invalid rectifier circuit state')
        for k, (_, _, _, members) in enumerate(red.groups):
            for b, s in members:
                self.cap_volts[b] = s * float(v[k])
        return out

    def step(self, tick, remaining_ticks):
        if self.analysis is None:
            raise RuntimeError('bench state not initialised')
        rows = self.advance(tick)
        red = self.reduced
        inst = {k: rows @ c for k, c in red.inst.items()}
        means = {k: float(x.mean()) for k, x in inst.items()}
        self.window.append(means)
        self.analysis['values'] = means
        if self.capture is not None:
            self.capture.append(rows @ red.port)
        changed_range = False
        for block, meter in self.meters.items():
            if not meter.due(tick):
                continue
            before = meter.text()
            recent = self.window[-meter.interval:]
            value = sum(w[block] for w in recent) / len(recent)
            old_range = meter.range
            if meter.convert(tick, lambda r, value=value: value):
                if meter.kind == 'ammeter':
                    changed_range = True
                    self.actions.append({'tick': tick, 'event': 'range', 'block': block,
                                         'from': meter.ranges[old_range][0], 'to': meter.ranges[meter.range][0]})
            if meter.text() != before:
                self.display_log.append({'tick': tick, 'block': block, 'text': meter.text(),
                                         'true_value': meter.true_value})
        self.logger.samples(tick, inst['CH1'], inst['CH2'])
        if changed_range:
            self.last_change_tick = tick
            self.analysis = self._analyse()
            self.analysis['values'] = means
            self._snapshot(tick, 'ammeter_range')

    # --- logs -------------------------------------------------------------------------------------------------------
    def _snapshot(self, tick, reason):
        a = self.analysis
        from .board import placement_text
        self.state_log.append({
            'tick': tick, 'reason': reason, 'change_tick': self.last_change_tick,
            'supply': {'on': self.supply_on}, 'held': self.held, 'all_placed': self.all_placed(),
            'placement': {b: placement_text(p) for b, p in self.placement.items()},
            'netlist': [[name, p, m, value] for name, p, m, _, value in a['elements']],
            'groups': a['groups'], 'output': dict(a['output']),
            'capacitor_v': {k: round(v, 9) for k, v in self.cap_volts.items()},
            'ammeter_range': self.meters['A'].ranges[self.meters['A'].range][0],
            'instruments': {k: dict(a['topology'][k]) for k in INSTRUMENTS},
            'shown': self.shown()})

    def observation_state(self, tick):
        from .board import placement_text
        a = self.analysis
        return {'tick': tick, 'supply_on': self.supply_on, 'held': self.held, 'all_placed': self.all_placed(),
                'last_change_tick': self.last_change_tick, 'output': dict(a['output']),
                'capacitor_v': {k: round(v, 9) for k, v in self.cap_volts.items()},
                'placement': {b: placement_text(p) for b, p in self.placement.items()},
                'instruments': {k: dict(a['topology'][k], true_value=a['values'].get(k, 0.0)) for k in INSTRUMENTS},
                'displays': deepcopy(self.shown()), 'logger_recording': self.logger.active is not None}

    def record_states(self, start_tick, end_tick):
        rows = [s for s in self.state_log if s['tick'] <= end_tick]
        before = [s for s in rows if s['tick'] <= start_tick]
        first = rows.index(before[-1]) if before else 0
        keep = ('tick', 'reason', 'change_tick', 'supply', 'held', 'all_placed', 'instruments', 'placement', 'output',
                'capacitor_v')
        return [{k: deepcopy(s[k]) for k in keep if k in s} for s in rows[first:]]

    def shown(self):
        out = {k: m.shown() for k, m in self.meters.items()}
        out['CH1'] = {'value': self.logger.live_v, 'count_si': 0.01, 'sample_tick': self.logger.live_tick,
                      'unit': 'V'}
        return out

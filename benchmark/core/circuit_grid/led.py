"""led_planck on the circuit board: the DC supply, five LED blocks (L1-L4 with tinted lenses and their peak wavelength
printed, X with a water-clear lens), series resistor blocks 100/330/1k (+-0.1 %), five wires, the two voltmeters, the
ammeter, the logger channels and the logger unit. No black box: the devices under test are the LED blocks the Agent
seats.

Supply SWEEP key (a second push key on the supply unit, next to ON/OFF):
with the output ON a press sets the output to 0 V and raises it at SWEEP_V_PER_S to the 5.00 V set point, where it
stays; a press during a sweep starts again from 0 V; OFF ends a sweep (ON again gives the set point at once); while the
output is OFF the key does nothing. The current limit holds throughout. The press is a change of the bench state (the
state log's change_tick); the rising voltage is not: the circuit is solved again at every tick of the sweep and a
state-log entry with the same change_tick is written every SWEEP_LOG_TICKS (and at the end), so a logger window during
a sweep lies in one bench state whose entries carry the LED currents as the voltage rises.

LED: a Shockley junction with ideality factor n and a series resistance r (both hidden, per LED),
    I = I_s (exp(V_j / (n V_T)) - 1),   V_LED = V_j + I r,   V_T = k T / e at 25 degC,
whose saturation current puts the junction voltage at E/e at I = 1 mA, E being the LED's turn-on energy
E = hc/lambda - delta + epsilon (one offset of the manufacturing process and a small scatter per LED; hidden). The
anode is the block's + pin (the red band); reversed, an LED blocks (I = -I_s). An LED glows in the colour of its
peak wavelength while more than 0.5 mA flows through it (render only).

LedBench (variants.NetBench): the LEDs are diode elements of the board netlist (solver.solve's damped Newton). The
device under test of the measuring-state gate is every seated LED: i_dut is the largest LED current, and an instrument
is on the DUT when its pins are on the two nodes of a seated LED (across) or removing it breaks the loop through one
(series). Per instrument the bench also records which LEDs' voltage it measures (v_of: across the LED) and
which LEDs' current (i_of: an ammeter in series with the LED, or a voltmeter or channel across a resistor block in
series with it) for the led_planck gates and milestones (benchmark/evaluation/circuit_grid_variants.py).
"""
from __future__ import annotations

import math

from . import layout as L
from .variants import NetBench, Profile, build_scene, register

H_EV_S = 4.135667696e-15
C_M_S = 299792458.0
HC_EV_NM = H_EV_S * C_M_S * 1e9            # 1239.84 eV nm
K_B_EV = 8.617333262e-5
T_K = 298.15
V_T = K_B_EV * T_K
I_REF_A = 1.0e-3
GLOW_A = 0.5e-3
LEDS = ('L1', 'L2', 'L3', 'L4', 'X')
KNOWN_NM = {'L1': 470, 'L2': 525, 'L3': 590, 'L4': 630}
RS = {'R100': 100.0, 'R330': 330.0, 'R1k': 1000.0}
RS_PRINT = {'R100': '100', 'R330': '330', 'R1k': '1k'}
RS_TAG = {'R100': '100', 'R330': '330', 'R1k': '1k'}
SUPPLY_V_NOMINAL, SUPPLY_LIMIT_A = 5.0, 0.10
SWEEP_V_PER_S = 0.25                       # the SWEEP ramp: 0 V to the set point in 20 s
SWEEP_LOG_S = 0.1                          # state-log cadence during a sweep
SWEEP_KEY_XY = (0.652, -0.115)             # on the supply top beside the ON/OFF button (layout.SUPPLY_BUTTON_XY)
SWEEP_KEY = '/World/CircuitGrid/Supply/SweepCap'  # asset.ROOT + '/Supply/SweepCap' (asset needs pxr)
# lens tints (dim, as seen unlit) and the colour each glows in
LENS = {'L1': (0.10, 0.22, 0.75), 'L2': (0.10, 0.62, 0.22), 'L3': (0.85, 0.62, 0.05), 'L4': (0.80, 0.10, 0.06),
        'X': (0.86, 0.88, 0.90)}
LED_BASE = (0.12, 0.12, 0.13)


def glow_colour(wavelength_nm):
    """Approximate sRGB of a narrow-band emitter (X glows in its own colour)."""
    w = float(wavelength_nm)
    if w < 490:
        r, g, b = 0.0, (w - 440) / 50, 1.0
    elif w < 510:
        r, g, b = 0.0, 1.0, (510 - w) / 20
    elif w < 580:
        r, g, b = (w - 510) / 70, 1.0, 0.0
    elif w < 645:
        r, g, b = 1.0, (645 - w) / 65, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    return tuple(0.15 + 0.85 * max(0.0, min(1.0, c)) for c in (r, g, b))


def led_parameters(energy_ev, n, r_ohm):
    """(I_s, n, r) of an LED whose junction voltage is E/e at I_REF_A."""
    if not 1.0 < energy_ev < 4.0 or not 1.0 <= n <= 4.0 or not 0 <= r_ohm < 100:
        raise ValueError('LED parameters out of range')
    return I_REF_A / math.expm1(energy_ev / (n * V_T)), float(n), float(r_ohm)


def led_voltage(params, current_a):
    i_s, n, r = params
    return n * V_T * math.log1p(max(current_a, 0.0) / i_s) + current_a * r


@register
class LedProfile(Profile):
    recipe = 'circuit_grid_led'
    blocks = {**{k: L.BLOCKS[k] for k in ('V1', 'V2', 'A', 'CH1', 'CH2')},
              **{k: ('led', ('X' if k == 'X' else str(KNOWN_NM[k])), k, 0.040) for k in LEDS},
              **{k: ('resistor', RS_PRINT[k], k, 0.040) for k in RS},
              **{f'W{k}': ('wire', '', 'wire', 0.035) for k in range(1, 6)}}
    tray_layout = (('V1', 'V2', 'A', 'CH1', 'CH2'),
                   LEDS,
                   tuple(RS) + (None, None),
                   ('W1', 'W2', 'W3', 'W4', 'W5'))
    supply = True
    dut = 'led'
    terminals = {}
    colours = {'led': LED_BASE}

    def configuration(self, instance):
        from instance_episodes import appearance_seed
        h = instance['hidden']
        nm = dict(KNOWN_NM, X=float(h['lambda_x_nm']))
        leds = {k: led_parameters(HC_EV_NM / nm[k] - float(h['delta_eV']) + float(h[f'eps_{k}_eV']),
                                  float(h[f'n_{k}']), float(h[f'r_{k}_ohm'])) for k in LEDS}
        return {'recipe': self.recipe, 'leds': {k: list(v) for k, v in leds.items()},
                'glow': {k: list(glow_colour(nm[k])) for k in LEDS},
                'block_ohm': {k: v * float(h[f'rs_factor_{RS_TAG[k]}']) for k, v in RS.items()},
                'supply_v': SUPPLY_V_NOMINAL, 'supply_limit_a': SUPPLY_LIMIT_A,
                'noise_seed': appearance_seed('grid-noise:' + instance['instance_id'])}

    def fixture(self, config):
        return []

    def device(self, config):
        return {'known_nm': dict(KNOWN_NM), 'rs_nominal_ohm': dict(RS), 'supply_v_nominal': SUPPLY_V_NOMINAL,
                'supply_limit_a': SUPPLY_LIMIT_A, 'sweep_v_per_s': SWEEP_V_PER_S, 'leds': list(LEDS)}

    def make_bench(self, episode_id, config, **kwargs):
        return LedBench(episode_id, self, config, **kwargs)

    def build(self, stage):
        built = build_scene(stage, self, build_supply_sweep)
        built['keys'] = dict(built['keys'], sweep=SWEEP_KEY)      # pressed like a logger key; LedBench.press_key
        glow = (getattr(self, 'scene_config', None) or {}).get('glow') or {}
        for block in LEDS:
            dress_led(stage, block, built['blocks'][block],
                      glow.get(block) or (glow_colour(KNOWN_NM[block]) if block in KNOWN_NM else (0.9, 0.9, 0.9)))
        return built

    def sync_displays(self, backend):
        """LED glow from the bench currents (render only); returns what is lit (for the display history reset)."""
        from pxr import UsdGeom
        bench = backend.bench
        lit = set()
        if bench is not None and bench.analysis is not None:
            lit = {k for k, i in bench.analysis['currents'].items() if k in LEDS and i > GLOW_A}
        shown = getattr(backend, '_led_lit', None)
        if shown != lit:
            for block in LEDS:
                prim = UsdGeom.Imageable(backend.stage.GetPrimAtPath(backend.paths[block] + '/Glow'))
                prim.MakeVisible() if block in lit else prim.MakeInvisible()
            backend._led_lit = lit
        return sorted(lit)

    def public_apparatus(self):
        looks = {'voltmeter': L.BLOCK_LOOKS['voltmeter'], 'ammeter': L.BLOCK_LOOKS['ammeter'],
                 'channel': 'red band at the + end of the base', 'wire': 'grey, no resistance',
                 'resistor': 'tan handle printed with its value in ohm'}
        handles = {'CH1': 'yellow handle', 'CH2': 'cyan handle'}
        blocks = {}
        for b, (k, p, _, _) in self.blocks.items():
            if k == 'led':
                look = ('water-clear lens printed X' if b == 'X'
                        else f'tinted lens printed {p} (its peak wavelength in nm)') + '; red band at the + (anode) end of the base'
            else:
                look = handles.get(b, looks[k])
            blocks[b] = {'kind': k, 'printed': p or 'none', 'looks': look, 'tray_seat': b,
                         'polarised': k in L.POLARISED or k == 'led'}
        return {
            'board': {'what': 'dark green plate with brass node pads; column numbers and row numbers are printed on '
                              'its edges',
                      'connection': 'everything with a pin or a terminal on one pad is connected there; pads are '
                                    'otherwise insulated from each other'},
            'fixed_terminals': {'supply_plus': 'red strap from the supply + terminal to a board pad',
                                'supply_minus': 'black strap from the supply - terminal to a board pad'},
            'supply': {'what': 'grey unit printed DC SUPPLY with an ON/OFF push button, a SWEEP push button and ON and '
                               'CC lamps'},
            'blocks': blocks,
            'tray': {'what': 'green-grey tray with one labelled well per block',
                     'seats': {b: {'printed': ('W' if k == 'wire' else (p if k != 'led' else p)), 'takes': c}
                               for b, (k, p, c, _) in self.blocks.items()}},
            'logger': {'what': 'dark blue two-channel logger unit printed LOGGER',
                       'keys': [L.LOGGER_KEY_PRINT.get(k, k) for row in L.LOGGER_KEYS for k in row if k is not None],
                       'display': ['CH1 live (V)', 'SET (s)', 'RATE (Hz)', 'NO', 'LEFT', 'REC lamp', 'ERR lamp']},
        }


def build_supply_sweep(stage, material, key_material):
    """The platform supply (ON/OFF button, lamps, straps) with a second spring key printed SWEEP beside the button."""
    from . import asset as A
    from .display import INK, Frame, text
    A.build_supply(stage, material, key_material)
    top = L.supply_top_z()
    sx, sy = SWEEP_KEY_XY
    A.spring_cap(stage, SWEEP_KEY, (sx, sy, top + L.CAP_LIFT + L.CAP_H / 2), (L.KEY_SIZE, L.KEY_SIZE, L.CAP_H),
                 (0.62, 0.76, 0.92), key_material)
    A._box(stage, A.ROOT + '/Supply/SweepCollar', (L.KEY_SIZE + 0.004, L.KEY_SIZE + 0.004, L.CAP_LIFT - 0.001),
           (sx, sy, top + (L.CAP_LIFT - 0.001) / 2), (0.10, 0.10, 0.11), collide=False)
    text(stage, SWEEP_KEY + '/Print', '/', Frame((0, 0), A.READ, A.UP, L.CAP_H / 2), 0, 0, 0.007, INK, align='centre')
    text(stage, A.ROOT + '/Supply/SweepLabel', 'SWEEP', Frame((sx + 0.021, sy), A.READ, A.UP, top), 0, 0, 0.005, INK,
         align='centre')
    return A.SUPPLY_BUTTON


def dress_led(stage, block, path, colour):
    """An LED block as built by asset.build_block (kind 'led': dark base and handle), dressed: the handle is the
    tinted (X: water-clear) lens body, a red band marks the + (anode) end of the base, the print is on the lens top,
    and a glowing copy of the lens (hidden until the LED conducts)."""
    from pxr import Gf, UsdGeom
    from . import asset as A
    from .display import INK, Frame, quad_mesh, text
    handle = UsdGeom.Gprim(stage.GetPrimAtPath(path + '/Handle'))
    handle.GetDisplayColorAttr().Set([Gf.Vec3f(*LENS[block])])
    bx, by, bz = L.BASE
    band = Frame((0, 0), (1, 0), (0, 1), bz)
    quad_mesh(stage, path + '/PlusBand', band, [(L.BASE_PLUS_BAND[0], -by / 2, L.BASE_PLUS_BAND[1], by / 2)], A.RED,
              0.0004)
    local = Frame((0, 0), (1, 0), (0, 1), L.BLOCK_TOP)
    printed = L.BLOCKS[block][1]
    text(stage, path + '/Value', printed, local, 0, 0, 0.0055, INK, align='centre', dz=0.0012)
    # the glow: an emissive shell just outside the lens body, shown while the LED conducts
    hx, hy, hz = L.HANDLE
    glow = A._box(stage, path + '/Glow', (hx + 0.0008, hy + 0.0008, hz + 0.0004), (0, 0, bz + hz / 2), (1, 1, 1),
                  collide=False)
    mat = A._render_material(stage, f'{A.ROOT}/Looks/LedGlow{block}', colour, emission=3.0)
    from pxr import UsdShade
    UsdShade.MaterialBindingAPI.Apply(glow.GetPrim()).Bind(mat)
    glow.GetDisplayColorAttr().Set([Gf.Vec3f(*colour)])
    glow.MakeInvisible()
    return path


class LedBench(NetBench):
    """NetBench with LED blocks as diode elements; the device under test is every seated LED. The supply's SWEEP key
    (module docstring) ramps the set voltage."""

    def __init__(self, episode_id, profile, config, **kwargs):
        self.leds = {k: tuple(v) for k, v in config['leds'].items()}
        super().__init__(episode_id, profile, config, **kwargs)
        self.v_set = self.supply_v
        self.sweep = None                     # {'start_tick'} while the output sweeps
        self.sweep_log_ticks = max(1, round(SWEEP_LOG_S * self.physics_hz))

    # --- the SWEEP key --------------------------------------------------------------------------------------------------
    def press_key(self, tick, key, remaining_ticks):
        if key != 'sweep':
            return super().press_key(tick, key, remaining_ticks)
        if not self.supply_on:
            effect = 'output_off'
        else:
            effect = 'restart' if self.sweep is not None else 'sweep'
            self.sweep = {'start_tick': tick}
            self.supply_v = 0.0
            self.last_change_tick = tick
            self.analysis = self._analyse()
            self._snapshot(tick, 'sweep_start')
        self.actions.append({'tick': tick, 'event': 'sweep_key', 'effect': effect})
        return effect

    def update(self, tick, placement, held, supply_on, switch_closed, reason=None, throws=None):
        if not supply_on and self.sweep is not None:      # OFF ends a sweep; ON again gives the set point
            self.sweep = None
        if not supply_on or self.sweep is None:
            self.supply_v = self.v_set
        return super().update(tick, placement, held, supply_on, switch_closed, reason=reason, throws=throws)

    def step(self, tick, remaining_ticks):
        if self.sweep is not None and tick > self.sweep['start_tick']:
            since = tick - self.sweep['start_tick']
            v = min(self.v_set, SWEEP_V_PER_S * since / self.physics_hz)
            done = v >= self.v_set
            self.supply_v = v
            self.analysis = self._analyse(full=False)
            if done:
                self.sweep = None
            if done or since % self.sweep_log_ticks == 0:
                self._snapshot(tick, 'sweep_end' if done else 'sweep')
        super().step(tick, remaining_ticks)

    def _elements(self, uf, ammeter_range=None, r_box=None):
        from .variants import HIGH_Z, LOGGER_INPUT_OHM, VOLTMETER_OHM
        out = []
        for block, place in self.placement.items():
            if place[0] != 'board':
                continue
            kind = self.blocks[block][0]
            if kind in ('wire', 'switch', 'changeover'):
                continue
            plus, minus = uf.find(L.node_name(place[1])), uf.find(L.node_name(place[2]))
            if plus == minus:
                continue
            if kind == 'led':
                out.append((block, plus, minus, 'D', self.leds[block]))
            elif kind in HIGH_Z:
                out.append((block, plus, minus, 'R', VOLTMETER_OHM if kind == 'voltmeter' else LOGGER_INPUT_OHM))
            elif kind == 'ammeter':
                meter = self.meters[block]
                out.append((block, plus, minus, 'R',
                            meter.ranges[meter.range if ammeter_range is None else ammeter_range][4]))
            else:
                out.append((block, plus, minus, 'R', self.ohm[block]))
        return out

    def _builder(self, elements, cap_v, zero=False):
        def build(net):
            for name, a, b, kind, value in elements:
                if kind == 'R':
                    net.resistor(name, a, b, value)
                elif kind == 'D':
                    i_s, n, r = value
                    net.diode(name, a, name + ':j', i_s, n, V_T)
                    net.resistor(name + ':r', name + ':j', b, r)
        return build

    def _currents(self, elements, result):
        sol = result.solution
        cur = {name: round(float(sol.i.get(name, 0.0)), 15) for name, _, _, kind, _ in elements if kind == 'D'}
        return cur, max((abs(i) for i in cur.values()), default=0.0)

    def _series_resistors(self, uf, elements, result, led, p, m):
        """Resistor blocks in series with one LED: every external path between its nodes runs through the block (the
        high-impedance instrument inputs do not count as paths), and there is such a path."""
        from .variants import HIGH_Z, INSTRUMENTS
        high_z = {k for k in INSTRUMENTS if self.blocks[k][0] in HIGH_Z}
        edges = [(name, a, b) for name, a, b, kind, _ in elements if name != led and name not in high_z]
        if self.profile.supply and result.mode != 'OFF':
            plus, minus = self._supply_roots(uf)
            if plus != minus:
                edges.append(('__supply', plus, minus))

        def connected(without):
            seen, todo = {p}, [p]
            while todo:
                n = todo.pop()
                for name, a, b in edges:
                    if name in without:
                        continue
                    for u, w in ((a, b), (b, a)):
                        if u == n and w not in seen:
                            seen.add(w)
                            todo.append(w)
            return m in seen

        if not connected(set()):
            return {}
        return {name: {a, b} for name, a, b, kind, _ in elements
                if kind == 'R' and self.blocks.get(name, ('',))[0] == 'resistor' and not connected({name})}

    def _topology(self, uf, elements, result):
        seated = [(name, p, m) for name, p, m, kind, _ in elements if kind == 'D']
        flags = None
        measured = {}                          # instrument: ({LEDs whose voltage}, {LEDs whose current})
        for name, p, m in seated:
            self.terminals, self.dut_names = {'+': p, '-': m}, [name]
            try:
                one = NetBench._topology(self, uf, elements, result)
            finally:
                self.terminals, self.dut_names = {}, []
            resistors = self._series_resistors(uf, elements, result, name, p, m)
            for k, f in one.items():
                v_of, i_of = measured.setdefault(k, (set(), set()))
                if f.get('across'):
                    v_of.add(name)
                if f.get('series') and self.blocks[k][0] == 'ammeter':
                    i_of.add(name)
                if f.get('seated') and self.blocks[k][0] != 'ammeter':
                    place = self.placement[k]
                    pins = {uf.find(L.node_name(place[1])), uf.find(L.node_name(place[2]))}
                    if any(pins == nodes for nodes in resistors.values()):
                        i_of.add(name)
            if flags is None:
                flags = one
            else:
                for k, f in one.items():
                    for key in ('across', 'series', 'on_dut', 'touches_dut'):
                        if key in f:
                            flags[k][key] = bool(flags[k].get(key)) or bool(f[key])
        if flags is not None:
            for k, (v_of, i_of) in measured.items():
                flags[k]['v_of'], flags[k]['i_of'] = sorted(v_of), sorted(i_of)
        if flags is None:
            from .variants import INSTRUMENTS
            flags = {}
            for block in INSTRUMENTS:
                seated_ = self.placement[block][0] == 'board'
                flags[block] = {'seated': seated_, 'across': False, 'series': False, 'on_dut': False}
                if seated_:
                    flags[block]['touches_dut'] = False
        return flags

    def _snapshot(self, tick, reason):
        super()._snapshot(tick, reason)
        self.state_log[-1]['led_currents_a'] = dict(self.analysis['currents'])
        self.state_log[-1]['supply']['sweep'] = self.sweep is not None

    def observation_state(self, tick):
        state = super().observation_state(tick)
        state['supply_v'] = round(self.analysis['result'].voltage_v, 9)
        state['supply_sweep'] = self.sweep is not None
        return state


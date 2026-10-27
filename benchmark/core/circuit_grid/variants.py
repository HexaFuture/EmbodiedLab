"""Circuit-board experiments beyond resistor_blackbox (battery_emf, rc_capacitance, circuit_blackbox3): the shared
machinery. The platform (layout, asset, board, backend, registered) is the resistor_blackbox bench; an experiment here
is a Profile that names its blocks, tray layout, fixed parts on the board and the device under test (DUT), and builds
its scene from the platform's builders.

Profile.apply sets the experiment's block set and tray layout on the layout module (one experiment per bench
process; tests restore it with Profile.restore). Everything else is subclassed:

NetBench (board.GridBench): the circuit from the seated blocks and the profile's fixed elements (a cell with internal
resistance, a black-box network of resistors and at most one capacitor), the DC supply if the bench has one.
    A capacitor is solved exactly between ticks: at each tick it is a voltage source at its present voltage; the
    circuit around it is linear, so over one tick its voltage moves exponentially toward the Thevenin voltage
    v + i R_th with time constant R_th C (R_th from the same circuit with its sources zeroed, recomputed at every
    change and every supply CV/CC switch). The capacitor keeps its charge while it is isolated.
    Supply: the bench supply of solver.supply_solve (CV up to its current limit, CC beyond). A wiring the ideal
    model cannot solve (the supply in parallel with the capacitor, i.e. two voltage sources in parallel) is solved
    with the supply's output resistance SUPPLY_R_OUT_OHM in series, which is the same supply in the limit; every
    wiring the ideal model solves is solved exactly as before. In CC the capacitor's voltage moves toward its CC
    final value (a ramp at I / C when nothing but the supply's current reaches it) but within a tick never past the
    CV operating point, where the supply has returned to CV.
    DUT: the cell or the box. Its current is the largest current entering one of its terminals from the board. An
    instrument is `across` the DUT when its pins are on two of its terminals, `series` when removing it (and the
    high-impedance inputs) breaks the only external path between two terminals that carry the loop.
VariantBackend (backend.GridBackend): no supply button where the bench has none, the changeover lever (a three-node
    block), placement of three-node blocks, displays without supply lamps.
variant_episode: the registered episode (registered.grid_episode with the profile's apparatus, device constants,
    terminal state and evidence).
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

import numpy as np

from . import layout as L
from .board import GridBench, _rotation, classify, classify_three, placement_text
from .instruments import (AMP_RANGES, CONVERSION_S, LOGGER_INPUT_OHM, LOGGER_NOISE_V, LOGGER_RANGE_V,
                          LOGGER_RESOLUTION_V, VOLT_RANGES, VOLTMETER_OHM)
from .solver import Netlist, SupplyResult, UnionFind, solve, supply_solve

I_MIN_A = 1.0e-6
STEADY_S = 0.2
INSTRUMENTS = L.METERS + L.CHANNELS
HIGH_Z = ('voltmeter', 'channel')
BOX_TERMINALS = dict(L.BOX_TERMINALS)       # the platform's box terminals (A, B, C) before any profile
SUPPLY_R_OUT_OHM = 0.01                     # output resistance of the bench supply (used only where the ideal
                                            # supply cannot be solved: in parallel with the capacitor)
SUPPLY_INT = '__supply:int'                 # the supply's internal node behind its output resistance
# the experiment of each profile whose objects roles.py looks at (rectifier_supply: none; its flags are across_output)
ROLES_EXP = {'circuit_grid_battery': 'battery_emf', 'circuit_grid_rc': 'rc_capacitance',
             'circuit_grid_network': 'circuit_blackbox3', 'circuit_grid_led': 'led_planck'}


# --- profiles ---------------------------------------------------------------------------------------------------------
class Profile:
    """One circuit-board experiment. Subclasses set the class attributes and fixture/build/public_apparatus."""
    recipe = None
    blocks = {}                 # two-pin blocks: id -> (kind, printed, seat class, mass)
    three = {}                  # three-node blocks: id -> (kind, printed, seat class, mass)
    tray_layout = ()            # rows of block ids (None: an empty position) on the platform tray grid
    tray_rows_x = L.TRAY_ROW_X
    tray_cols_y = L.TRAY_COL_Y
    supply = False              # the DC supply with its ON/OFF button on the board
    dut = 'box'                 # 'cell' or 'box'
    terminals = {}              # DUT terminal name -> board node
    colours = {}                # extra block kinds -> handle colour (asset.BLOCK_COLOURS)
    ground = L.SUPPLY_MINUS     # reference node of the solution on a bench without a supply
    box_terminals = None        # the box terminals used (layout.BOX_TERMINALS by default: A, B, C)
    _saved = None

    def apply(self):
        if self._saved is None:
            self._saved = {k: getattr(L, k) for k in ('BLOCKS', 'TRAY_LAYOUT', 'TRAY_ROW_X', 'TRAY_COL_Y',
                                                        'CHANGEOVER_BLOCKS', 'BOX_TERMINALS')}
        L.BLOCKS = dict(self.blocks)
        L.TRAY_LAYOUT = tuple(tuple(r) for r in self.tray_layout)
        L.TRAY_ROW_X, L.TRAY_COL_Y = tuple(self.tray_rows_x), tuple(self.tray_cols_y)
        L.CHANGEOVER_BLOCKS = dict(L.CHANGEOVER_BLOCKS, **self.three)
        if self.box_terminals is not None:
            L.BOX_TERMINALS = {k: BOX_TERMINALS[k] for k in self.box_terminals}
        return self

    def restore(self):
        if self._saved is not None:
            for k, v in self._saved.items():
                setattr(L, k, v)
            self._saved = None

    # experiment-specific
    def configuration(self, instance):
        raise NotImplementedError

    def fixture(self, config):
        """[(name, kind, node a, node b, value)] of the fixed elements: kind 'R' (ohm), 'C' (farad; one at most),
        'cell' ((emf V, internal ohm), a = + terminal). Nodes are board node names 'c,r'."""
        raise NotImplementedError

    def values(self, config):
        """{block id: ohm} of the resistive blocks (actual values)."""
        return dict(config['block_ohm'])

    def build(self, stage):
        raise NotImplementedError

    def public_apparatus(self):
        raise NotImplementedError

    def device(self, config):
        return {}

    def make_bench(self, episode_id, config, **kwargs):
        """The electrical bench of this profile (NetBench unless the profile has its own)."""
        return NetBench(episode_id, self, config, **kwargs)

    def before_seal(self, episode, submission):
        """Device work at submission before the episode is sealed; returns extra sealed device fields."""
        return {}

    def terminal_extra(self, backend):
        """Extra sealed terminal-state fields of this profile."""
        return {}

    def three_seats(self):
        """{three-node block: (seat centre xy, yaw deg)} where it starts (its own long tray well, yaw 90: the block's
        +x along world +y like every tray seat)."""
        return {}

    def build_tray(self, stage, material):
        """The platform tray with a long well for each three-node block."""
        from . import asset as A
        if not self.three:
            return A.build_tray(stage, material)
        return build_tray_long(stage, material, {b: xy for b, (xy, _) in self.three_seats().items()})


PROFILES = {}


def register(profile_cls):
    PROFILES[profile_cls.recipe] = profile_cls
    return profile_cls


MODULES = {'circuit_grid_battery': 'battery', 'circuit_grid_rc': 'rc', 'circuit_grid_network': 'network',
           'circuit_grid_led': 'led', 'circuit_grid_rectifier': 'rectifier'}


def profile_for(recipe):
    if recipe not in PROFILES:
        import importlib
        importlib.import_module(f'{__package__}.{MODULES[recipe]}')     # the module registers its profile
    return PROFILES[recipe]()


# --- electrical bench ---------------------------------------------------------------------------------------------------
class _Result:
    def __init__(self, solution, mode, current_a=0.0, voltage_v=0.0):
        self.solution, self.mode, self.current_a, self.voltage_v = solution, mode, current_a, voltage_v


class NetBench(GridBench):
    def __init__(self, episode_id, profile, config, *, physics_hz=240, max_records=20):
        super().__init__(episode_id, r_box_ohm=1.0, rs_actual_ohm={}, supply_v=config.get('supply_v', 0.0),
                         supply_limit_a=config.get('supply_limit_a', 0.0), noise_seed=config['noise_seed'],
                         physics_hz=physics_hz, max_records=max_records,
                         extra_blocks=dict(profile.three))
        self.profile = profile
        self.config = config
        self.ohm = profile.values(config)
        self.fixed = profile.fixture(config)
        caps = [f for f in self.fixed if f[1] == 'C']
        if len(caps) > 1:
            raise ValueError('at most one capacitor')
        self.cap = caps[0] if caps else None
        self.cap_v = float(config.get('cap_v0', 0.0)) if self.cap else None
        self.terminals = {k: L.node_name(v) for k, v in profile.terminals.items()}
        self.dut_names = [f[0] for f in self.fixed]
        self.throw_source = None
        self.roles_exp = ROLES_EXP.get(profile.recipe)       # roles.py (private record only)
        self._rth = {}
        self._tick = 0
        self.dt = 1.0 / physics_hz

    # --- circuit ----------------------------------------------------------------------------------------------------
    def update(self, tick, placement, held, supply_on, switch_closed, reason=None, throws=None):
        if throws is None and self.throw_source is not None:
            throws = self.throw_source()
        self._tick = tick
        return super().update(tick, placement, held, supply_on, switch_closed, reason=reason, throws=throws)

    def _elements(self, uf, ammeter_range=None, r_box=None):
        """[(name, + root, - root, kind, value)]: fixed elements, then seated blocks (pins on one node omitted)."""
        out = []
        for name, kind, a, b, value in self.fixed:
            ra, rb = uf.find(a), uf.find(b)
            if ra != rb or kind == 'cell':          # a shorted cell still drives its short through r
                out.append((name, ra, rb, kind, value))
        for block, place in self.placement.items():
            if place[0] != 'board':
                continue
            kind = self.blocks[block][0]
            if kind in ('wire', 'switch', 'changeover'):
                continue
            plus, minus = uf.find(L.node_name(place[1])), uf.find(L.node_name(place[2]))
            if plus == minus:
                continue
            if kind in HIGH_Z:
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
                elif kind == 'cell':
                    emf, r = value
                    net.resistor(name + ':r', a, name + ':int', r)
                    net.vsource(name + ':emf', name + ':int', b, 0.0 if zero else emf)
                elif kind == 'C':
                    net.vsource('C:' + name, a, b, cap_v)
        return build

    def _supply_roots(self, uf):
        return uf.find(L.node_name(L.SUPPLY_PLUS)), uf.find(L.node_name(L.SUPPLY_MINUS))

    def _solve(self, ammeter_range=None, r_box=None, cap_v=None):
        uf = self._merge()
        elements = self._elements(uf, ammeter_range)
        cap_v = self.cap_v if cap_v is None else cap_v
        build = self._builder(elements, cap_v or 0.0)
        if not self.profile.supply:
            net = Netlist(reference=uf.find(L.node_name(self.profile.ground)))
            build(net)
            return uf, elements, _Result(solve(net), 'NONE')
        plus, minus = self._supply_roots(uf)
        if plus == minus:
            net = Netlist(reference=minus)
            build(net)
            sol = solve(net)
            return uf, elements, _Result(sol, 'CC' if self.supply_on else 'OFF',
                                         self.supply_limit if self.supply_on else 0.0, 0.0)
        kwargs = dict(on=self.supply_on, v_set=self.supply_v, i_limit=self.supply_limit)
        try:
            r = supply_solve(build, plus, minus, **kwargs)
        except np.linalg.LinAlgError:
            # the ideal supply in parallel with the capacitor: the real supply, its output resistance in series
            def real(net):
                build(net)
                net.resistor('__supply:r_out', SUPPLY_INT, plus, SUPPLY_R_OUT_OHM)
            r = supply_solve(real, SUPPLY_INT, minus, **kwargs)
            r = SupplyResult(r.solution, r.mode, r.current_a, r.solution.vab(plus, minus))
        return uf, elements, _Result(r.solution, r.mode, r.current_a, r.voltage_v)

    def _thevenin_r(self, uf, elements, mode):
        """Resistance the capacitor sees with every source zeroed (CV supply: short; CC / OFF: open)."""
        name, _, a, b, _ = self.cap
        ra, rb = uf.find(a), uf.find(b)

        def thevenin(r_out):
            net = Netlist(reference=rb)
            for e_name, p, m, kind, value in elements:
                if kind == 'R':
                    net.resistor(e_name, p, m, value)
                elif kind == 'cell':
                    net.resistor(e_name + ':r', p, e_name + ':int', value[1])
                    net.vsource(e_name + ':emf', e_name + ':int', m, 0.0)
            if self.profile.supply and mode == 'CV':
                plus, minus = self._supply_roots(uf)
                if plus != minus:
                    if r_out:
                        net.resistor('__supply:r_out', SUPPLY_INT, plus, r_out)
                    net.vsource('__supply0', SUPPLY_INT if r_out else plus, minus, 0.0)
            net.vsource('__test', ra, rb, 1.0)
            return solve(net)
        try:
            sol = thevenin(0.0)
        except np.linalg.LinAlgError:          # the capacitor directly across the CV supply (see _solve)
            sol = thevenin(SUPPLY_R_OUT_OHM)
        i = abs(sol.i['__test'])
        return 1.0 / i if i > 0 else math.inf

    def _cv_voltage(self, uf, elements):
        """The capacitor's terminal voltage with the supply in CV and the capacitor removed (where a CC charge
        through nothing but the supply's current ends)."""
        name, _, a, b, _ = self.cap
        plus, minus = self._supply_roots(uf)
        if plus == minus:
            return self.cap_v
        net = Netlist(reference=minus)
        self._builder([e for e in elements if e[0] != name], 0.0)(net)
        net.vsource('__supply', plus, minus, self.supply_v)
        try:
            sol = solve(net)
        except np.linalg.LinAlgError:
            return self.cap_v
        return sol.vab(uf.find(a), uf.find(b))

    def _readings(self, uf, elements, result):
        sol = result.solution
        values = {}
        for block in INSTRUMENTS:
            place = self.placement[block]
            if place[0] != 'board':
                values[block] = 0.0
                continue
            plus, minus = uf.find(L.node_name(place[1])), uf.find(L.node_name(place[2]))
            if self.blocks[block][0] == 'ammeter':
                values[block] = float(sol.i.get(block, 0.0))
            else:
                values[block] = sol.v.get(plus, 0.0) - sol.v.get(minus, 0.0)
        return values

    def _currents(self, elements, result):
        """{fixed element: current from its first to its second node} and the DUT current (the largest current
        entering one DUT terminal from the board)."""
        sol = result.solution
        cur = {}
        for name, a, b, kind, value in elements:
            if name not in self.dut_names:
                continue
            if kind == 'cell':
                cur[name] = float(sol.i.get(name + ':r', 0.0))
            elif kind == 'C':
                cur[name] = float(sol.i.get('C:' + name, 0.0))
            else:
                cur[name] = float(sol.i.get(name, 0.0))
        into = {}
        for name, kind, a, b, _ in self.fixed:
            i = cur.get(name, 0.0)
            into[a] = into.get(a, 0.0) + i
            into[b] = into.get(b, 0.0) - i
        dut = max((abs(v) for k, v in into.items() if k in self.terminals.values()), default=0.0)
        return {k: round(v, 15) for k, v in cur.items()}, dut

    def _topology(self, uf, elements, result):
        roots = {t: uf.find(n) for t, n in self.terminals.items()}
        edges = [(name, p, m) for name, p, m, kind, _ in elements if name not in self.dut_names]
        if self.profile.supply and result.mode not in ('OFF',):
            plus, minus = self._supply_roots(uf)
            if plus != minus:
                edges.append(('__supply', plus, minus))
        high_z = {k for k in INSTRUMENTS if self.blocks[k][0] in HIGH_Z}

        def connected(x, y, without, through=()):
            seen, todo = {x}, [x]
            while todo:
                n = todo.pop()
                for name, p, m in list(edges) + list(through):
                    if name in without:
                        continue
                    for u, w in ((p, m), (m, p)):
                        if u == n and w not in seen:
                            seen.add(w)
                            todo.append(w)
            return y in seen

        pairs = [(roots[s], roots[t]) for i, s in enumerate(sorted(roots)) for t in sorted(roots)[i + 1:]
                 if roots[s] != roots[t]]
        loops = [(a, b) for a, b in pairs if connected(a, b, set())]
        dut_edges = [('__dut', a, b) for a, b in pairs]       # the DUT itself closes a loop through the board
        out = {}
        for block in INSTRUMENTS:
            place = self.placement[block]
            if place[0] != 'board':
                out[block] = {'seated': False, 'across': False, 'series': False, 'on_dut': False}
                continue
            plus, minus = uf.find(L.node_name(place[1])), uf.find(L.node_name(place[2]))
            across = plus != minus and any({plus, minus} == {a, b} for a, b in pairs)
            series = False
            if plus != minus and not across and connected(plus, minus, {block}, dut_edges):
                series = any(not connected(a, b, {block} | high_z) for a, b in loops)
            touches = plus in roots.values() or minus in roots.values()
            out[block] = {'seated': True, 'across': bool(across), 'series': bool(series),
                          'on_dut': bool(across or series), 'touches_dut': bool(touches)}
        return out

    def _analyse(self, full=True):
        uf, elements, result = self._solve()
        values = self._readings(uf, elements, result)
        currents, dut = self._currents(elements, result)
        a = {'uf': uf, 'elements': elements, 'result': result, 'values': values, 'currents': currents,
             'i_dut': dut, 'i_ab': dut, 'sensitivity': 0.0}
        cap_in = self.cap is not None and any(e[0] == self.cap[0] for e in elements)
        fresh = False
        if full or self.analysis is None:
            a['topology'] = self._topology(uf, elements, result)
            a['rth'] = None
            if cap_in:
                a['rth'] = self._thevenin_r(uf, elements, result.mode)
                self._rth = {'mode': result.mode, 'rth': a['rth']}
                fresh = True
        else:
            a['topology'] = self.analysis['topology']
            a['rth'] = self.analysis.get('rth')
            a['final_currents'] = self.analysis.get('final_currents')
            if cap_in and result.mode != self._rth.get('mode'):
                a['rth'] = self._thevenin_r(uf, elements, result.mode)
                self._rth = {'mode': result.mode, 'rth': a['rth']}
                fresh = True
        if self.cap is not None and not cap_in:
            self.cap_v = 0.0                        # its terminals joined by a wire: discharged at once
            a['rth'] = None
        a['cap_cv_bound'] = None
        if self.cap is not None:
            i = currents.get(self.cap[0], 0.0)
            rth = a['rth'] if a['rth'] is not None else math.inf
            v_final = (self.cap_v + i * rth) if math.isfinite(rth) else self.cap_v
            if cap_in and result.mode == 'CC':
                # in CC the capacitor charges toward the CV operating point at most (the supply returns to CV first)
                bound = self._cv_voltage(a['uf'], elements)
                a['cap_cv_bound'] = bound
                if not math.isfinite(rth) or (v_final - bound) * (self.cap_v - bound) < 0:
                    v_final = bound
            if fresh or a.get('final_currents') is None:
                # the fixed elements' currents once the capacitor has settled (the state held long enough)
                a['final_currents'] = self._currents(*self._solve(cap_v=v_final)[1:])[0] if cap_in else currents
            a['cap'] = {'v': self.cap_v, 'i_a': i,
                        'rth_ohm': rth if math.isfinite(rth) else None,
                        'tau_s': (rth * self.cap[4]) if math.isfinite(rth) else None,
                        'v_final': v_final, 'dut_currents_final_a': a['final_currents']}
        return a

    def _advance_cap(self):
        """Move the capacitor voltage over one tick (exact for the linear circuit around it)."""
        a = self.analysis
        rth = a.get('rth')
        i = a['currents'].get(self.cap[0], 0.0)
        bound = a.get('cap_cv_bound')
        if rth is not None and math.isfinite(rth) and rth > 0:
            new = self.cap_v - i * rth * math.expm1(-self.dt / (rth * self.cap[4]))
        elif bound is not None:
            new = self.cap_v + i * self.dt / self.cap[4]       # CC with no other path: a ramp at I / C
        else:
            return
        if bound is not None and (new - bound) * (self.cap_v - bound) < 0:
            new = bound                                       # CC ends at the CV operating point at the latest
        self.cap_v = new

    def step(self, tick, remaining_ticks):
        if self.cap is not None:
            if tick != self.last_change_tick:
                self.analysis = self._analyse(full=False)
            super().step(tick, remaining_ticks)
            self._advance_cap()
            return
        super().step(tick, remaining_ticks)

    # --- logs and evidence ------------------------------------------------------------------------------------------
    def _snapshot(self, tick, reason):
        a = self.analysis
        res = a['result']
        entry = {
            'tick': tick, 'reason': reason, 'change_tick': self.last_change_tick,
            'supply': {'on': self.supply_on, 'mode': res.mode, 'v': round(res.voltage_v, 9),
                       'i_a': round(res.current_a, 12)},
            'switch_closed': self.switch_closed, 'throws': dict(self.throws), 'held': self.held,
            'placement': {b: placement_text(p) for b, p in self.placement.items()},
            'netlist': [[name, p, m, kind, value if kind != 'cell' else list(value)]
                        for name, p, m, kind, value in a['elements']],
            'i_dut_a': a['i_dut'], 'i_ab_a': a['i_dut'], 'dut_currents_a': a['currents'],
            'ammeter_range': self.meters['A'].ranges[self.meters['A'].range][0],
            'instruments': {k: dict(a['topology'][k], true_value=a['values'][k]) for k in INSTRUMENTS},
            'shown': self.shown(), 'all_placed': self.all_placed(), 'r_sensitivity': 0.0}
        if self.cap is not None:
            entry['cap'] = dict(a['cap'])
        if self.roles_exp:
            entry['roles'] = self._roles(entry)
        self.state_log.append(entry)

    def observation_state(self, tick):
        a = self.analysis
        state = {'tick': tick, 'supply_on': self.supply_on, 'supply_mode': a['result'].mode,
                 'switch_closed': self.switch_closed, 'throws': dict(self.throws), 'held': self.held,
                 'all_placed': self.all_placed(), 'last_change_tick': self.last_change_tick,
                 'i_dut_a': a['i_dut'], 'i_ab_a': a['i_dut'], 'dut_currents_a': a['currents'],
                 'placement': {b: placement_text(p) for b, p in self.placement.items()},
                 'instruments': {k: dict(a['topology'][k], true_value=a['values'][k]) for k in INSTRUMENTS},
                 'displays': deepcopy(self.shown()), 'r_sensitivity': 0.0,
                 'logger_recording': self.logger.active is not None}
        if self.cap is not None:
            state['cap'] = dict(a['cap'])
        if self.roles_exp and self.state_log:
            state['roles'] = deepcopy(self.state_log[-1].get('roles'))
        return state

    def record_states(self, start_tick, end_tick):
        rows = [s for s in self.state_log if s['tick'] <= end_tick]
        before = [s for s in rows if s['tick'] <= start_tick]
        first = rows.index(before[-1]) if before else 0
        keep = ('tick', 'reason', 'change_tick', 'supply', 'held', 'all_placed', 'i_dut_a', 'i_ab_a',
                'dut_currents_a', 'instruments', 'placement', 'throws', 'cap', 'r_sensitivity', 'roles')
        return [{k: deepcopy(s[k]) for k in keep if k in s} for s in rows[first:]]


# --- Isaac backend ------------------------------------------------------------------------------------------------------
def _backend_class():
    from .backend import GridBackend

    class VariantBackend(GridBackend):
        """GridBackend with the profile's scene, three-node blocks (changeover lever) and no supply lamps where the
        bench has no supply."""

        def __init__(self, args, app, configuration):
            from . import asset as A
            self.profile = profile_for(configuration['recipe']).apply()
            self.profile.scene_config = dict(configuration)       # build-time looks that follow the instance
            self.throws = {b: None for b in self.profile.three}
            self.lever_up = {b: 'T1' for b in self.profile.three}          # the end that is raised: starts on T1
            original = A.build
            A.build = self.profile.build
            try:
                super().__init__(args, app, configuration)
            finally:
                A.build = original
            self.throws = {b: self._lever_throw(b) for b in self.profile.three}

        # changeover levers
        def lever_angle_deg(self, block):
            from . import asset as A
            _, quats = self.poses()
            rb = _rotation(quats[self.block_ids.index(block)])
            m = self._transform(L.block_path(A.ROOT, block) + '_Lever')
            rot = np.array([[m[i][j] for j in range(3)] for i in range(3)], float)
            xl = rot[0] / (np.linalg.norm(rot[0]) or 1.0)       # USD rows: the lever's local axes in world
            return math.degrees(math.atan2(-float(rb[:, 2] @ xl), float(rb[:, 0] @ xl)))

        def _lever_throw(self, block):
            return L.lever_position(self.lever_angle_deg(block))

        def _rocker(self):
            from . import asset as A
            from pxr import UsdPhysics
            if 'SW' in self.block_ids:
                super()._rocker()
            for block in self.profile.three:
                angle = self.lever_angle_deg(block)
                side = self.lever_up[block]
                flip = None
                if side == 'T1' and angle > 2.0:
                    flip = 'T2'
                elif side == 'T2' and angle < -2.0:
                    flip = 'T1'
                if flip:
                    self.lever_up[block] = flip
                    target = (L.LEVER_ANGLE_DEG + 2.0) * (1 if flip == 'T2' else -1)
                    joint = self.stage.GetPrimAtPath(A.PHYSICS_ROOT + f'/{block}LeverJoint')
                    UsdPhysics.DriveAPI.Get(joint, 'angular').GetTargetPositionAttr().Set(float(target))
                    self.audit_file.write(json.dumps({'event': 'lever', 'tick': self.tick_index, 'block': block,
                                                      'to': flip}) + '\n')
                self.throws[block] = L.lever_position(angle)

        def rocker_angle_deg(self):
            if 'SW' not in self.block_ids:
                return 0.0
            return super().rocker_angle_deg()

        # placement of three-node blocks
        def placement(self):
            positions, quats = self.poses()
            busy = {self.held, self.releasing}
            out = {}
            seats = self.profile.three_seats()
            for k, b in enumerate(self.block_ids):
                if b in self.profile.three:
                    place = classify_three(positions[k], quats[k], held=b in busy)
                    if place[0] != 'board' and b not in busy:
                        xy, _ = seats[b]
                        tilt = math.degrees(math.acos(max(-1.0, min(1.0, _rotation(quats[k])[2, 2]))))
                        if (math.hypot(positions[k][0] - xy[0], positions[k][1] - xy[1]) <= L.SEAT_HOME_DXY_M
                                and abs(positions[k][2] - (L.TABLE_Z + L.TRAY_FLOOR)) <= L.SEAT_HOME_DZ_M
                                and tilt <= L.SEAT_HOME_TILT_DEG):
                            place = ('tray', b, True)
                    out[b] = place
                else:
                    out[b] = classify(b, positions[k], quats[k], held=b in busy)
            return out

        def sync_displays(self):
            from .backend import RESET_HOLD_RENDERS
            if self.profile.supply:
                super().sync_displays()
            elif self.bench is not None:                 # no supply: meters and logger only (battery_emf)
                shown = {block: meter.shown() for block, meter in self.bench.meters.items()}
                for block, value in shown.items():
                    self.displays.meter(block, value)
                logger = self.bench.logger.display()
                self.displays.logger(logger)
                key = json.dumps([shown, logger], sort_keys=True, default=str)
                if key != self.display_key:               # a display change resets the denoiser history (backend)
                    self.display_key = key
                    self.reset_until = self.render_index + RESET_HOLD_RENDERS
                    self.reset_edge = True
            extra = getattr(self.profile, 'sync_displays', None)
            if extra is not None:                     # the profile's own displays and lamps (LED glow, ...)
                key = json.dumps(extra(self), sort_keys=True, default=str)
                if key != getattr(self, '_profile_display_key', None):
                    self._profile_display_key = key
                    self.reset_until = self.render_index + RESET_HOLD_RENDERS
                    self.reset_edge = True

    return VariantBackend


def variant_backend(args, app, configuration):
    return _backend_class()(args, app, configuration)


# --- registered episode ---------------------------------------------------------------------------------------------
def variant_configuration(instance):
    return profile_for(instance['recipe']).configuration(instance)


def variant_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    from device_runtime import DeviceRuntime
    from instance_episodes import InstanceMixin
    from .registered import TOOLS, canonical_sha256, public_record

    config = backend.configuration
    profile = backend.profile
    max_records = int(instance['public_task']['public_constants']['logger']['max_records'])

    class VariantEpisode(InstanceMixin, DeviceRuntime):
        def public_task(self):
            return self.instance_public_task({'apparatus': profile.public_apparatus()})

        def robot_state(self):
            return super().robot_state() | {'gripper_holding': backend.held is not None}

        def observation(self):
            value = super().observation()
            bench = backend.bench
            n = len(self.captures) + 1
            eid = f'{self.episode_id}:obs:{n}'
            views = backend.view_pngs
            images = {camera: hashlib.sha256(png).hexdigest() for camera, png in views.items()}
            public = {'evidence_id': eid, 'physics_tick': self.tick, 'sim_time_s': self.tick / physics_hz,
                      'images_sha256': images}
            directory = backend.private / 'evidence'
            directory.mkdir(exist_ok=True)
            for camera, png in views.items():
                (directory / f'obs_{n}_{camera}.png').write_bytes(png)
            self.captures[eid] = {
                'kind': 'screen_capture', 'public': public, 'sha256': canonical_sha256(public),
                'state': bench.observation_state(self.tick),
                'images': {camera: {'file': f'evidence/obs_{n}_{camera}.png', 'sha256': digest}
                           for camera, digest in images.items()}}
            value['evidence_id'] = eid
            return value

        def terminal_state(self):
            bench = backend.bench
            placement = backend.placement()
            seats = {}
            for block, place in placement.items():
                if place[0] == 'tray' and place[2]:
                    seats.setdefault(place[1], []).append(block)

            def text(p):
                if p[0] == 'board':
                    return ('board',) + tuple(L.node_name(n) for n in p[1:])
                if p[0] == 'tray':
                    return ('tray', p[1], bool(p[2]))
                return ('loose',)
            state = {'supply_on': bool(backend.supply_on), 'switch_closed': bool(backend.switch_closed),
                     'throws': dict(backend.throws), 'logger_recording': bench.logger.active is not None,
                     'held': backend.held or backend.releasing,
                     'placement': {b: text(p) for b, p in placement.items()}, 'tray_seats': seats,
                     'all_in_tray': all(p[0] == 'tray' and p[2] for p in placement.values())}
            if bench.cap is not None:
                state['box_capacitor_v'] = round(float(bench.cap_v), 6)
            state.update(profile.terminal_extra(backend))
            return state

        def seal_payload(self, submission=None):
            bench = backend.bench
            logger = bench.logger
            evidence = {eid: dict(public_record(record), kind='circuit_logger', **deepcopy(logger.private[eid]),
                                  states=bench.record_states(logger.private[eid]['start_tick'],
                                                             logger.private[eid]['end_tick']))
                        for eid, record in logger.records.items()}
            evidence.update(deepcopy(self.captures))
            cited = set()
            if isinstance(submission, dict) and isinstance(submission.get('measurements'), list):
                cited = {m.get('evidence_id') for m in submission['measurements'] if isinstance(m, dict)}
            if isinstance(submission, dict) and isinstance(submission.get('evidence_ids'), list):
                cited |= {e for e in submission['evidence_ids'] if isinstance(e, str)}
            directory = backend.private / 'evidence'
            files = [(image['file'], (directory / image['file'].rsplit('/', 1)[-1]).read_bytes())
                     for eid, record in self.captures.items() if eid in cited for image in record['images'].values()]
            device = {'voltmeter_input_ohm': VOLTMETER_OHM, 'logger_input_ohm': LOGGER_INPUT_OHM,
                      'voltmeter_ranges': [list(r[:4]) for r in VOLT_RANGES],
                      'ammeter_ranges': [list(r) for r in AMP_RANGES], 'conversion_s': CONVERSION_S,
                      'logger': {'range_v': LOGGER_RANGE_V, 'resolution_v': LOGGER_RESOLUTION_V,
                                 'noise_v': LOGGER_NOISE_V},
                      'i_min_a': I_MIN_A, 'steady_s': STEADY_S, 'physics_hz': physics_hz,
                      'recipe': profile.recipe, **profile.device(config)}
            return self.terminal_state(), device, evidence, files

        def process_extra(self):
            bench = backend.bench
            return {'logger_open': bench.logger.open_record(), 'display_log': list(bench.display_log),
                    'logger_presses': list(bench.logger.presses)}

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if name not in TOOLS or not isinstance(args, dict):
                raise ValueError('unknown circuit board tool')
            if name == 'get_task_spec' and args == {}:
                return self.public_task()
            if name == 'export_logger':
                if set(args) != {'evidence_id'}:
                    raise ValueError('export_logger takes evidence_id')
                return public_record(backend.bench.logger.export(args['evidence_id']))
            if name == 'submit_result':
                extra = profile.before_seal(self, args)
                terminal, device, evidence, files = self.seal_payload(args)
                device.update(extra)
                return self.freeze_and_seal(args, terminal, device, evidence, files)
            return super()._dispatch(name, args)

        def evaluation(self):
            return self.instance_evaluation()

    # Apparatus preparation (not charged): open the gripper and let the blocks settle in their tray seats.
    backend.set_gripper(.08)
    for _ in range(40 * (physics_hz // 30)):
        backend.step(None)
        backend.after_step({}, backend.tick_index)
    if not backend.at_rest():
        raise RuntimeError('circuit board blocks failed to settle')
    placement = backend.placement()
    away = [b for b, p in placement.items() if not (p[0] == 'tray' and p[1] == b)]
    if away:
        raise RuntimeError(f'blocks not in their tray seats after set-up: {away}')
    backend.tick_index = 0
    backend.history.clear()
    episode = VariantEpisode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    episode.captures = {}
    backend.horizon_ticks = horizon_ticks
    bench = profile.make_bench(episode.episode_id, config, physics_hz=physics_hz, max_records=max_records)
    bench.throw_source = lambda: dict(backend.throws)
    backend.bench = bench
    bench.update(0, placement, None, backend.supply_on, backend.switch_closed, reason='start')
    return episode


def grid_bench_for(args, app, instance):
    """Backend and episode of a circuit-board instance (server.py): resistor_blackbox on the platform bench, the
    other recipes through their profile."""
    if instance['recipe'] == 'circuit_grid_bench':
        from .backend import GridBackend
        from .registered import grid_episode
        backend = GridBackend(args, app, args.grid_configuration)
        episode = grid_episode(backend, instance, args.output, physics_hz=args.physics_hz,
                               horizon_ticks=args.horizon_ticks)
        return backend, episode
    backend = variant_backend(args, app, args.grid_configuration)
    episode = variant_episode(backend, instance, args.output, physics_hz=args.physics_hz,
                              horizon_ticks=args.horizon_ticks)
    return backend, episode


def grid_configuration_for(instance):
    if instance['recipe'] == 'circuit_grid_bench':
        from .registered import grid_configuration
        return grid_configuration(instance)
    return variant_configuration(instance)


# --- scene ------------------------------------------------------------------------------------------------------------
def build_scene(stage, profile, fixtures):
    """The platform scene with the profile's blocks: board, `fixtures(stage, fixed material, key material)` (the
    supply, the box or the cell), tray, logger unit, every block in its tray well (three-node blocks in their own
    wells, see the profile). Returns asset.build's dict ('button': the supply button or None; 'rocker': the switch
    rocker or, without a switch, the first changeover lever)."""
    from pxr import PhysxSchema, UsdGeom
    from . import asset as A
    from .display import Frame, INK, text
    A.BLOCK_COLOURS.update(profile.colours)       # handle colours of the profile's own block kinds
    A.retire_hooke_apparatus(stage)
    A.repair_cube_extents(stage)
    z = A.table_top_z(stage)
    if abs(z - L.TABLE_Z) > 0.002:
        raise RuntimeError(f'table top at {z:.4f} m, the circuit board layout expects {L.TABLE_Z} m')
    UsdGeom.Xform.Define(stage, A.ROOT)
    UsdGeom.Xform.Define(stage, A.PHYSICS_ROOT)
    fixed = A._physics_material(stage, '/World/PhysicsMaterials/CircuitGridFixed', 0.5, 0.4)
    blocks_mat = A._physics_material(stage, '/World/PhysicsMaterials/CircuitGridBlock', 0.5, 0.4)
    keys_mat = A._physics_material(stage, '/World/PhysicsMaterials/CircuitGridKey', 0.8, 0.7)
    A.build_board(stage, fixed)
    button = fixtures(stage, fixed, keys_mat)
    profile.build_tray(stage, fixed)
    keys = A.build_logger(stage, fixed, keys_mat)
    paths = {}
    UsdGeom.Xform.Define(stage, A.ROOT + '/Blocks')
    rocker = None
    for label, seat in L.tray_seats().items():
        pose = (seat['xy'][0], seat['xy'][1], L.TABLE_Z + L.TRAY_FLOOR + 0.0005)
        paths[label] = A.build_block(stage, label, pose, L.TRAY_YAW_DEG, blocks_mat)
        kind, printed = L.BLOCKS[label][:2]
        if kind == 'shunt':        # printed value on the handle top, as the platform prints a resistor's
            text(stage, paths[label] + '/Value', printed, Frame((0, 0), (1, 0), (0, 1), L.BLOCK_TOP), 0, 0, 0.0055,
                 INK, align='centre')
        if label == 'SW':
            rocker = A.build_rocker(stage, pose, L.TRAY_YAW_DEG, blocks_mat)
    for block, (xy, yaw) in profile.three_seats().items():
        path, lever, _ = A.build_changeover(stage, block, (xy[0], xy[1], L.TABLE_Z + L.TRAY_FLOOR + 0.0005), yaw,
                                            blocks_mat)
        paths[block] = path
        rocker = rocker or lever
    # a bench with neither a switch nor a changeover (led_planck, rectifier_supply): the backend's rocker view goes on
    # the first block, never read (rocker_angle_deg is 0 and _rocker skips the switch without SW)
    rocker = rocker or next(iter(paths.values()))
    table = stage.GetPrimAtPath(A.TABLE_TOP)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(table)
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    return {'blocks': paths, 'keys': keys, 'button': button, 'rocker': rocker}


def build_tray_long(stage, material, long_wells):
    """asset.build_tray with extra long wells {three-node block: centre xy}: as a platform well (mouth, chamfer, floor
    either side of the pin groove) but long enough along world y for a three-node base (layout.BASE3), labelled with
    the block's print."""
    from pxr import Gf, UsdGeom
    from . import asset as A
    from .display import Frame, text
    (x0, x1), (y0, y1) = L.tray_extent()
    top, floor, groove = L.TABLE_Z + L.TRAY_H, L.TABLE_Z + L.TRAY_FLOOR, L.TABLE_Z + L.TRAY_GROOVE_FLOOR
    throat = top - L.TRAY_CHAMFER_DEPTH
    gh = L.TRAY_GROOVE_HALF
    seats = L.tray_seats()
    grow = L.BASE3[0] / 2 + (L.TRAY_WELL_HALF[1] - L.BASE[0] / 2) - L.TRAY_WELL_HALF[1]
    wells = [(label, s['xy'], L.TRAY_WELL_HALF, L.TRAY_MOUTH_HALF, s['printed']) for label, s in seats.items()]
    wells += [(b, xy, (L.TRAY_WELL_HALF[0], L.TRAY_WELL_HALF[1] + grow), (L.TRAY_MOUTH_HALF[0], L.TRAY_MOUTH_HALF[1] + grow),
               L.CHANGEOVER_BLOCKS[b][1]) for b, xy in long_wells.items()]
    for _, (cx, cy), _, (mx, my), _ in wells:
        if cx - mx < x0 or cx + mx > x1 or cy - my < y0 or cy + my > y1:
            raise ValueError(f'tray well at {(cx, cy)} outside the tray')
    holes = [(cx - mx, cx + mx, cy - my, cy + my) for _, (cx, cy), _, (mx, my), _ in wells]
    quads = [q for q, _ in A._grid_cells((x0, x1), (y0, y1), holes, top)]
    for _, (cx, cy), (wx, wy), (mx, my), _ in wells:
        quads += A._pocket(cx, cy, [(mx, my, top), (wx, wy, throat), (wx, wy, floor)])
        quads.append(((cx - wx, cy - wy, floor), (cx - gh, cy - wy, floor), (cx - gh, cy + wy, floor),
                      (cx - wx, cy + wy, floor)))
        quads.append(((cx + gh, cy - wy, floor), (cx + wx, cy - wy, floor), (cx + wx, cy + wy, floor),
                      (cx + gh, cy + wy, floor)))
        for sx in (-gh, gh):
            quads.append(((cx + sx, cy - wy, floor), (cx + sx, cy + wy, floor), (cx + sx, cy + wy, groove),
                          (cx + sx, cy - wy, groove)))
        for sy in (-wy, wy):
            quads.append(((cx - gh, cy + sy, floor), (cx + gh, cy + sy, floor), (cx + gh, cy + sy, groove),
                          (cx - gh, cy + sy, groove)))
        quads.append(((cx - gh, cy - wy, groove), (cx + gh, cy - wy, groove), (cx + gh, cy + wy, groove),
                      (cx - gh, cy + wy, groove)))
    bottom = L.TABLE_Z
    quads += [((x0, y0, bottom), (x1, y0, bottom), (x1, y0, top), (x0, y0, top)),
              ((x0, y1, bottom), (x1, y1, bottom), (x1, y1, top), (x0, y1, top)),
              ((x0, y0, bottom), (x0, y1, bottom), (x0, y1, top), (x0, y0, top)),
              ((x1, y0, bottom), (x1, y1, bottom), (x1, y1, top), (x1, y0, top))]
    UsdGeom.Xform.Define(stage, A.ROOT + '/Tray')
    A._quads(stage, A.ROOT + '/Tray/Body', quads, A.TRAY_COLOUR, collide=True, material=material)
    for label, (cx, cy), _, _, printed in wells:
        text(stage, f'{A.ROOT}/Tray/Label{label}', printed, Frame((cx + L.TRAY_LABEL_DX, cy), A.READ, A.UP, top), 0, 0,
             0.0075, A.WHITE, align='centre')
        seat = UsdGeom.Xform.Define(stage, f'{A.ROOT}/Tray/Seat{label}')
        UsdGeom.XformCommonAPI(seat).SetTranslate(Gf.Vec3d(cx, cy, floor))

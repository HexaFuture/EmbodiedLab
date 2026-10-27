"""Electrical bench of the circuit board (no Isaac): where every block is, the circuit that makes, its solution, the
instruments and the private logs.

Placement of a block (from its pose, `classify`): seated on the board on one edge (both pins within SEAT_TOL_M of the
two facing sockets of that edge, the base bottom within SEAT_DZ_M of the board surface, tilted at most SEAT_TILT_DEG,
not held), in a tray seat (within 20 mm / 20 mm / 10 degrees of a block standing in that seat, not held), or loose
(anywhere else, or held). Only seated blocks are in the circuit; everything else is open.

Circuit: board nodes joined by seated wires (and the switch when closed) are one electrical node. The supply's + and
- terminals, the black box's A, B, C and the pins of every seated block sit on board nodes. Elements: the box resistor
A-B, the R_s blocks, the voltmeters and logger channels (10 MOhm), the ammeter (its range's shunt) and the supply
(CV up to its current limit, then CC; OFF: open). The netlist is rebuilt and solved whenever a placement, the switch,
the supply or the ammeter range changes; between changes the DC solution holds.

Logs (private, process record): `state_log` gets a snapshot at every change (netlist, supply state, placements, every
instrument's true value and shown reading, measurement-relevant flags); `actions` gets every seat and board event,
button and key press and range change.
"""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

from . import layout as L
from .instruments import Logger, Meter, LOGGER_INPUT_OHM, VOLTMETER_OHM
from .solver import Netlist, SupplyResult, UnionFind, solve, supply_solve

I_MIN_A = 1.0e-6                 # a measurement needs at least this current through A-B
STEADY_S = 0.2                   # no placement / switch / supply change within this time before a reading
SENSITIVITY_MIN = 0.1            # |d ln(y_i / y_j) / d ln R| for a reading pair to determine R (process rule M2)
INSTRUMENTS = L.METERS + L.CHANNELS
_SOCKETS = None


def _sockets():
    global _SOCKETS
    if _SOCKETS is None:
        s = L.sockets()
        _SOCKETS = (np.array([(x, y) for x, y, _, _ in s]), [(node, edge) for _, _, node, edge in s])
    return _SOCKETS


def _rotation(quat_wxyz):
    w, x, y, z = (float(v) for v in quat_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _pin_sockets(p, R, pins):
    """{pin: (node, edge index)} of the pins within SEAT_TOL_M of a socket (each pin its nearest socket)."""
    centres, owners = _sockets()
    out = {}
    for name, local in pins.items():
        pin = p + R @ np.asarray(local)
        d = np.hypot(*(centres - pin[:2]).T)
        k = int(np.argmin(d))
        if d[k] <= L.SEAT_TOL_M:
            out[name] = owners[k]
    return out


def classify_three(xyz, quat_wxyz, held=False, table_z=L.TABLE_Z):
    """Placement of a three-node block (changeover): ('board', T1 node, C node, T2 node) when its base lies on the
    board (SEAT_DZ_M, SEAT_TILT_DEG), every pin is within SEAT_TOL_M of a socket, T1 and C in the two sockets of one
    edge and T2 on the next node along the same line; else ('loose',) (tray seats for these blocks belong to the
    experiments that use them)."""
    if held:
        return ('loose',)
    R = _rotation(quat_wxyz)
    p = np.asarray(xyz, float)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
    if tilt > L.SEAT_TILT_DEG or abs(p[2] - (table_z + L.BOARD_H)) > L.SEAT_DZ_M:
        return ('loose',)
    pins = _pin_sockets(p, R, L.changeover_pins())
    if len(pins) != 3:
        return ('loose',)
    (t1, e1), (c, ec), (t2, _) = pins['T1'], pins['C'], pins['T2']
    if e1 != ec or t1 == c or (t2[0] - c[0], t2[1] - c[1]) != (c[0] - t1[0], c[1] - t1[1]):
        return ('loose',)
    return ('board', t1, c, t2)


def classify(block_id, xyz, quat_wxyz, held=False, table_z=L.TABLE_Z):
    """Placement of one block from its world pose: ('board', plus node, minus node), ('tray', seat label, class
    matches) or ('loose',)."""
    if held:
        return ('loose',)
    R = _rotation(quat_wxyz)
    p = np.asarray(xyz, float)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
    board_top = table_z + L.BOARD_H
    if tilt <= L.SEAT_TILT_DEG and abs(p[2] - board_top) <= L.SEAT_DZ_M:
        centres, owners = _sockets()
        pins = {}
        for sign in (+1, -1):
            pin = p + R @ np.asarray(L.pin_local(sign))
            d = np.hypot(*(centres - pin[:2]).T)
            k = int(np.argmin(d))
            if d[k] <= L.SEAT_TOL_M:
                pins[sign] = owners[k]
        if len(pins) == 2 and pins[1][1] == pins[-1][1] and pins[1][0] != pins[-1][0]:
            return ('board', pins[1][0], pins[-1][0])
    if tilt <= L.SEAT_HOME_TILT_DEG:
        for label, seat in L.tray_seats().items():
            if (math.hypot(p[0] - seat['xy'][0], p[1] - seat['xy'][1]) <= L.SEAT_HOME_DXY_M
                    and abs(p[2] - (table_z + L.TRAY_FLOOR)) <= L.SEAT_HOME_DZ_M):
                return ('tray', label, seat['class'] == L.BLOCKS[block_id][2])
    return ('loose',)


def placement_text(placement):
    if placement[0] == 'board' and len(placement) == 4:
        return 'board:' + '|'.join(f'{L.node_name(n)}{p}' for n, p in zip(placement[1:], ('T1', 'C', 'T2')))
    if placement[0] == 'board':
        return f'board:{L.node_name(placement[1])}+|{L.node_name(placement[2])}-'
    if placement[0] == 'tray':
        return f'tray:{placement[1]}' + ('' if placement[2] else ':mismatch')
    return 'loose'


class GridBench:
    def __init__(self, episode_id, *, r_box_ohm, rs_actual_ohm, supply_v, supply_limit_a, noise_seed,
                 physics_hz=240, max_records=20, extra_blocks=None):
        self.episode_id = episode_id
        self.physics_hz = physics_hz
        self.r_box = float(r_box_ohm)
        self.rs_actual = {k: float(v) for k, v in rs_actual_ohm.items()}
        self.supply_v, self.supply_limit = float(supply_v), float(supply_limit_a)
        rng = np.random.default_rng(noise_seed)
        self.meters = {b: Meter(b, L.BLOCKS[b][0], rng=np.random.default_rng(rng.integers(2 ** 32)),
                                phase_ticks=k * 32, physics_hz=physics_hz) for k, b in enumerate(L.METERS)}
        self.logger = Logger(episode_id, noise_seed=int(rng.integers(2 ** 32)), physics_hz=physics_hz,
                             max_records=max_records)
        # blocks beyond the resistor_blackbox set (three-node changeovers): {id: (kind, printed, seat class, mass)}
        self.blocks = dict(L.BLOCKS, **(extra_blocks or {}))
        self.placement = {b: ('loose',) for b in self.blocks}
        self.throws = {b: None for b, v in self.blocks.items() if v[0] == 'changeover'}
        self.held = None
        self.supply_on = False
        self.switch_closed = False
        self.last_change_tick = 0
        self.state_log = []
        self.actions = []
        self.display_log = []
        self.result = None
        self.analysis = None
        self.started = False

    # --- circuit ----------------------------------------------------------------------------------------------------
    def _merge(self):
        uf = UnionFind()
        for node in L.all_nodes():
            uf.find(L.node_name(node))
        for block, place in self.placement.items():
            kind = self.blocks[block][0]
            if place[0] == 'board' and (kind == 'wire' or (kind == 'switch' and self.switch_closed)):
                uf.union(L.node_name(place[1]), L.node_name(place[2]))
            if place[0] == 'board' and kind == 'changeover' and self.throws.get(block):
                t1, c, t2 = place[1:4]
                uf.union(L.node_name(c), L.node_name(t1 if self.throws[block] == 'T1' else t2))
        return uf

    def _elements(self, uf, ammeter_range=None, r_box=None):
        """[(name, + root, - root, ohm)] of the resistive elements on the board (pins on one merged node omitted)."""
        out = []
        a, b = uf.find(L.node_name(L.BOX_TERMINALS['A'])), uf.find(L.node_name(L.BOX_TERMINALS['B']))
        if a != b:
            out.append(('box_AB', a, b, self.r_box if r_box is None else r_box))
        for block, place in self.placement.items():
            if place[0] != 'board':
                continue
            kind = self.blocks[block][0]
            if kind in ('wire', 'switch', 'changeover'):
                continue
            plus, minus = uf.find(L.node_name(place[1])), uf.find(L.node_name(place[2]))
            if plus == minus:
                continue
            if kind == 'resistor':
                ohm = self.rs_actual[block]
            elif kind in ('voltmeter', 'channel'):
                ohm = VOLTMETER_OHM if kind == 'voltmeter' else LOGGER_INPUT_OHM
            else:
                meter = self.meters[block]
                ohm = meter.ranges[meter.range if ammeter_range is None else ammeter_range][4]
            out.append((block, plus, minus, ohm))
        return out

    def _solve(self, ammeter_range=None, r_box=None):
        uf = self._merge()
        elements = self._elements(uf, ammeter_range, r_box)
        plus, minus = uf.find(L.node_name(L.SUPPLY_PLUS)), uf.find(L.node_name(L.SUPPLY_MINUS))

        def build(net):
            for name, a, b, ohm in elements:
                net.resistor(name, a, b, ohm)

        if plus == minus:                            # supply terminals joined by wires: a short, CC at 0 V
            net = Netlist(reference=minus)
            build(net)
            sol = solve(net)
            result = SupplyResult(sol, 'CC' if self.supply_on else 'OFF', self.supply_limit if self.supply_on else 0.0,
                                  0.0)
        else:
            result = supply_solve(build, plus, minus, on=self.supply_on, v_set=self.supply_v,
                                  i_limit=self.supply_limit)
        return uf, elements, result

    def _readings(self, uf, elements, result):
        """True (noise-free) value of every instrument: V(+) - V(-) or the ammeter current; 0 off the board."""
        sol = result.solution
        values = {}
        for block in INSTRUMENTS:
            place = self.placement[block]
            if place[0] != 'board':
                values[block] = 0.0
                continue
            plus, minus = uf.find(L.node_name(place[1])), uf.find(L.node_name(place[2]))
            if L.BLOCKS[block][0] == 'ammeter':
                values[block] = float(sol.i.get(block, 0.0))
            else:
                values[block] = sol.v.get(plus, 0.0) - sol.v.get(minus, 0.0)
        return values

    def _topology(self, uf, elements, result):
        """Per instrument on the board: across A-B (its pins on A's and B's nodes) or in series with A-B (removing it,
        the box resistor and the high-impedance instrument inputs separates A from B, both lying on loops)."""
        a, b = uf.find(L.node_name(L.BOX_TERMINALS['A'])), uf.find(L.node_name(L.BOX_TERMINALS['B']))
        edges = [(name, p, m) for name, p, m, _ in elements]
        if result.mode != 'OFF':
            edges.append(('__supply', uf.find(L.node_name(L.SUPPLY_PLUS)), uf.find(L.node_name(L.SUPPLY_MINUS))))

        def connected(x, y, without):
            seen, todo = {x}, [x]
            while todo:
                n = todo.pop()
                for name, p, m in edges:
                    if name in without:
                        continue
                    for u, w in ((p, m), (m, p)):
                        if u == n and w not in seen:
                            seen.add(w)
                            todo.append(w)
            return y in seen

        out = {}
        box_loop = a != b and connected(a, b, {'box_AB'})
        # The meters' own 10 MOhm inputs do not break a series connection (an ammeter in series with A-B and a
        # voltmeter across A-B carries the box current plus the voltmeter's).
        high_z = {k for k in INSTRUMENTS if L.BLOCKS[k][0] in ('voltmeter', 'channel')}
        for block in INSTRUMENTS:
            place = self.placement[block]
            if place[0] != 'board':
                out[block] = {'seated': False, 'across_ab': False, 'series_ab': False}
                continue
            plus, minus = uf.find(L.node_name(place[1])), uf.find(L.node_name(place[2]))
            across = a != b and {plus, minus} == {a, b}
            series = False
            if box_loop and plus != minus and not across and connected(plus, minus, {block}):
                series = not connected(a, b, {block, 'box_AB'} | high_z)
            out[block] = {'seated': True, 'across_ab': bool(across), 'series_ab': bool(series)}
        return out

    def _sensitivity(self, values):
        """max over pairs of seated instruments with a signal of |d ln(y_i / y_j) / d ln R| (0 if fewer than two)."""
        live = {k: v for k, v in values.items() if self.placement[k][0] == 'board'
                and abs(v) >= (1e-7 if L.BLOCKS[k][0] == 'ammeter' else 1e-3)}
        if len(live) < 2:
            return 0.0
        eps = 1e-3
        uf, elements, result = self._solve(r_box=self.r_box * (1 + eps))
        shifted = self._readings(uf, elements, result)
        best = 0.0
        keys = sorted(live)
        for i, p in enumerate(keys):
            for q in keys[i + 1:]:
                if shifted[p] == 0 or shifted[q] == 0 or (shifted[p] / live[p]) <= 0 or (shifted[q] / live[q]) <= 0:
                    continue
                change = math.log(shifted[p] / live[p]) - math.log(shifted[q] / live[q])
                best = max(best, abs(change) / math.log1p(eps))
        return round(best, 6)

    def _analyse(self):
        uf, elements, result = self._solve()
        values = self._readings(uf, elements, result)
        topo = self._topology(uf, elements, result)
        i_ab = float(result.solution.i.get('box_AB', 0.0))
        return {'uf': uf, 'elements': elements, 'result': result, 'values': values, 'topology': topo, 'i_ab': i_ab,
                'sensitivity': self._sensitivity(values)}

    # --- state ----------------------------------------------------------------------------------------------------
    def all_placed(self):
        """Every block seated on the board or standing in a tray seat, and none held."""
        return self.held is None and all(p[0] in ('board', 'tray') for p in self.placement.values())

    def update(self, tick, placement, held, supply_on, switch_closed, reason=None, throws=None):
        """The physical state at `tick`; on any change the circuit is rebuilt, solved and logged. throws: {changeover
        block: 'T1', 'T2' or None} (the throw its lever selects)."""
        changes = []
        for block, throw in (throws or {}).items():
            if throw != self.throws[block]:
                self.actions.append({'tick': tick, 'event': 'changeover', 'block': block, 'from': self.throws[block],
                                     'to': throw})
                self.throws[block] = throw
                changes.append('changeover')
        for block, new in placement.items():
            old = self.placement[block]
            if new != old:
                changes.append(block)
                self.actions.append({'tick': tick, 'event': 'placement', 'block': block, 'from': placement_text(old),
                                     'to': placement_text(new)})
        if held != self.held:
            self.actions.append({'tick': tick, 'event': 'held', 'from': self.held, 'to': held})
            changes.append('held')
        if supply_on != self.supply_on:
            self.actions.append({'tick': tick, 'event': 'supply', 'on': bool(supply_on)})
            changes.append('supply')
        if switch_closed != self.switch_closed:
            self.actions.append({'tick': tick, 'event': 'switch', 'closed': bool(switch_closed)})
            changes.append('switch')
        if not changes and self.started:
            return False
        self.placement = dict(placement)
        self.held, self.supply_on, self.switch_closed = held, bool(supply_on), bool(switch_closed)
        self.last_change_tick = tick
        self.analysis = self._analyse()
        self._snapshot(tick, reason or ('start' if not self.started else ','.join(changes)))
        self.started = True
        return True

    roles_exp = 'resistor_blackbox'      # the experiment whose objects roles.py looks at (NetBench: the profile's)

    def _roles(self, entry):
        """What each instrument measures in this state (roles.py; private process record only). A
        failure here never stops the bench: the entry then records the error."""
        try:
            from .roles import from_state
            return from_state(entry, self.roles_exp, getattr(self, 'dut_names', None))
        except Exception as exc:                                    # noqa: BLE001
            return {'error': f'{type(exc).__name__}: {exc}'}

    def _snapshot(self, tick, reason):
        a = self.analysis
        res = a['result']
        self.state_log.append({
            'tick': tick, 'reason': reason, 'change_tick': self.last_change_tick,
            'supply': {'on': self.supply_on, 'mode': res.mode, 'v': round(res.voltage_v, 9),
                       'i_a': round(res.current_a, 12)},
            'switch_closed': self.switch_closed, 'held': self.held,
            'placement': {b: placement_text(p) for b, p in self.placement.items()},
            'netlist': [[name, p, m, ohm] for name, p, m, ohm in a['elements']],
            'i_ab_a': a['i_ab'], 'ammeter_range': self.meters['A'].ranges[self.meters['A'].range][0],
            'instruments': {k: dict(a['topology'][k], true_value=a['values'][k]) for k in INSTRUMENTS},
            'shown': self.shown(), 'all_placed': self.all_placed(), 'r_sensitivity': a['sensitivity']})
        self.state_log[-1]['roles'] = self._roles(self.state_log[-1])

    def shown(self):
        out = {k: m.shown() for k, m in self.meters.items()}
        out['CH1'] = {'value': self.logger.live_v, 'count_si': 0.001, 'sample_tick': self.logger.live_tick,
                      'unit': 'V'}
        return out

    def step(self, tick, remaining_ticks):
        """Instruments at one physics tick: meter conversions that are due, the logger."""
        if self.analysis is None:
            raise RuntimeError('bench state not initialised')
        changed_range = False
        for block, meter in self.meters.items():
            if not meter.due(tick):
                continue
            before = meter.text()
            if meter.kind == 'ammeter':
                def read(r, block=block):
                    if r == meter.range:
                        return self.analysis['values'][block]
                    uf, elements, result = self._solve(ammeter_range=r)
                    return self._readings(uf, elements, result)[block]
                old_range = meter.range
                if meter.convert(tick, read):
                    changed_range = True
                    self.actions.append({'tick': tick, 'event': 'range', 'block': block,
                                         'from': meter.ranges[old_range][0], 'to': meter.ranges[meter.range][0]})
            else:
                meter.convert(tick, lambda r, block=block: self.analysis['values'][block])
            if meter.text() != before:
                self.display_log.append({'tick': tick, 'block': block, 'text': meter.text(),
                                         'true_value': meter.true_value})
        if changed_range:
            # A range change switches the ammeter's shunt: the circuit changes like any other change of state.
            self.last_change_tick = tick
            self.analysis = self._analyse()
            self._snapshot(tick, 'ammeter_range')
        values = self.analysis['values']
        self.logger.tick(tick, values['CH1'], values['CH2'])

    def press_key(self, tick, key, remaining_ticks):
        effect = self.logger.press(tick, key, remaining_ticks)
        self.actions.append({'tick': tick, 'event': 'logger_key', 'key': key, 'effect': effect})
        return effect

    # --- evidence ------------------------------------------------------------------------------------------------
    def observation_state(self, tick):
        """Private physical state at an observation (sealed with its evidence record)."""
        a = self.analysis
        return {'tick': tick, 'supply_on': self.supply_on, 'supply_mode': a['result'].mode,
                'switch_closed': self.switch_closed, 'held': self.held, 'all_placed': self.all_placed(),
                'last_change_tick': self.last_change_tick, 'i_ab_a': a['i_ab'],
                'placement': {b: placement_text(p) for b, p in self.placement.items()},
                'instruments': {k: dict(a['topology'][k], true_value=a['values'][k]) for k in INSTRUMENTS},
                'displays': deepcopy(self.shown()), 'r_sensitivity': a['sensitivity'],
                'logger_recording': self.logger.active is not None,
                'roles': deepcopy(self.state_log[-1].get('roles')) if self.state_log else None}

    def record_states(self, start_tick, end_tick):
        """The state snapshots in force during [start_tick, end_tick] (the one before start_tick included)."""
        rows = [s for s in self.state_log if s['tick'] <= end_tick]
        before = [s for s in rows if s['tick'] <= start_tick]
        first = rows.index(before[-1]) if before else 0
        keep = ('tick', 'reason', 'change_tick', 'supply', 'held', 'all_placed', 'i_ab_a', 'instruments',
                'placement', 'r_sensitivity', 'roles')
        return [{k: deepcopy(s[k]) for k in keep if k in s} for s in rows[first:]]

    def tray_state(self):
        seats = {label: None for label in L.tray_seats()}
        for block, place in self.placement.items():
            if place[0] == 'tray' and place[2]:
                seats[place[1]] = block
        return seats

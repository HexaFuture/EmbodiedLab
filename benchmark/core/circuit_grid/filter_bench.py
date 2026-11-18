"""Electrical bench of active_filter (no Isaac): the circuit the seated blocks make on the 7 x 6 board, the
frequency-response analyzer (keypad state machine, sweeps, screen state) and the private logs.

Circuit: board nodes joined by seated wires and by the ground rail's printed copper (the far row) are one electrical
node. Resistor and capacitor pieces (true values from filter_model.draw(model_seed)), wires, the two op-amp modules
(three-node blocks: IN+ on the first node, IN- on the middle one, OUT on the third; module OA1 is op-amp 0 of the
instance, OA2 op-amp 1) and the probe (a polarised two-pin block, + pin and - pin) make a filter_model.Circuit with
the analyzer's source at the IN pad and ground on the rail. Only seated blocks count (layout/board.classify,
classify_three); the netlist is rebuilt whenever a placement changes.

Analyzer keypad (each press acts once; keys do nothing while a sweep runs):
    0-9      append a digit to the entry (at most ENTRY_DIGITS)
    START / STOP / PTS   set that field to the entry (Err if out of range or START >= STOP) and clear the entry;
             with an empty entry nothing happens
    CLR      clear the entry and Err
    SWEEP    start a sweep with START, STOP, PTS (Err if it would not end within the budget)
A sweep measures PTS log-spaced frequencies from START to STOP, one every POINT_S, the lowest first; each point is
the probe reading over the source EMF of the circuit as the board is when that point is taken (gain dB rounded to
0.001, phase deg to 0.01; below the floor of -120 dB the gain reads -120.000). A seat or unseat while a sweep runs
does not stop it (the instrument does not know the board): later points read the changed circuit; the private record
keeps which board signature each point was taken on. Err lights only for a keypad entry out of range and a sweep the
budget refuses. Sweep n is evidence `<episode>:sweep:<n>`; its public record (export_sweeps) carries what the
instrument knows: fields, times, status, the points (no probe position, no board state).

Logs (private): state_log (a snapshot at every change: placements, netlist, analyzer), actions (placement, held,
key events), display_log (screen field changes).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from copy import deepcopy

import numpy as np

from . import layout as L
from .board import placement_text

import filter_model as M

POINT_S = 0.4
START_RANGE_HZ = (10, 100000)
PTS_RANGE = (2, 200)
ENTRY_DIGITS = 6
DEFAULTS = {'start_hz': 100, 'stop_hz': 10000, 'pts': 50}
FLOOR_DB = -120.0
GAIN_DECIMALS, PHASE_DECIMALS, FREQ_DECIMALS = 3, 2, 3
TRACE = bool(__import__('os').environ.get('EMBODIEDLAB_FILTER_TRACE'))     # probes: board changes and keys to stderr
KEYS = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'start', 'stop', 'pts', 'clr', 'sweep')


def sweep_freqs(start, stop, pts):
    """The PTS frequencies of a sweep (Hz, rounded to FREQ_DECIMALS; the ends exact)."""
    k = np.arange(pts)
    f = start * (stop / start) ** (k / (pts - 1))
    f = np.round(f, FREQ_DECIMALS)
    f[0], f[-1] = float(start), float(stop)
    return f


def reading(h):
    """(gain dB, phase deg) as the analyzer shows a complex response."""
    mag = np.abs(h)
    gain = np.where(mag > 10 ** (FLOOR_DB / 20), 20 * np.log10(np.maximum(mag, 1e-300)), FLOOR_DB)
    phase = np.where(mag > 10 ** (FLOOR_DB / 20), np.degrees(np.angle(h)), 0.0)
    return np.round(gain, GAIN_DECIMALS) + 0.0, np.round(phase, PHASE_DECIMALS) + 0.0


def sweep_csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['point', 'f_hz', 'gain_db', 'phase_deg'])
    for k, (f, g, p) in enumerate(rows):
        w.writerow([k, f'{f:.{FREQ_DECIMALS}f}', f'{g:.{GAIN_DECIMALS}f}', f'{p:.{PHASE_DECIMALS}f}'])
    return buf.getvalue()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Analyzer:
    def __init__(self, episode_id, physics_hz):
        self.episode_id = episode_id
        self.physics_hz = physics_hz
        self.point_ticks = round(POINT_S * physics_hz)
        self.fields = dict(DEFAULTS)
        self.entry = ''
        self.error = False
        self.active = None                  # the running sweep
        self.sweeps = []                    # every sweep, in order (records)
        self.private = {}
        self.presses = []

    # --- keys ---------------------------------------------------------------------------------------------------
    def press(self, tick, key, remaining_ticks, circuit_fn):
        if self.active is not None:
            effect = 'ignored_running'
        elif key in '0123456789' and len(key) == 1:
            if len(self.entry) < ENTRY_DIGITS:
                self.entry += key
                effect = 'digit'
            else:
                effect = 'entry_full'
            self.error = False
        elif key == 'clr':
            self.entry, self.error = '', False
            effect = 'clear'
        elif key in ('start', 'stop', 'pts'):
            if not self.entry:
                effect = 'no_entry'
            else:
                value = int(self.entry)
                self.entry = ''
                f = dict(self.fields, **{key + ('_hz' if key != 'pts' else ''): value})
                ok = (START_RANGE_HZ[0] <= f['start_hz'] <= START_RANGE_HZ[1]
                      and START_RANGE_HZ[0] <= f['stop_hz'] <= START_RANGE_HZ[1]
                      and f['start_hz'] < f['stop_hz'] and PTS_RANGE[0] <= f['pts'] <= PTS_RANGE[1])
                if ok:
                    self.fields, self.error = f, False
                    effect = f'set_{key}'
                else:
                    self.error = True
                    effect = f'err_{key}'
        elif key == 'sweep':
            need = self.fields['pts'] * self.point_ticks
            if need > remaining_ticks:
                self.error = True
                effect = 'err_budget'
            else:
                self.error = False
                self._start(tick, circuit_fn)
                effect = f'sweep_{len(self.sweeps)}'
        else:
            effect = 'unknown'
        self.presses.append({'tick': tick, 'key': key, 'effect': effect})
        return effect

    def _start(self, tick, circuit_fn):
        _, info = circuit_fn()
        f = sweep_freqs(self.fields['start_hz'], self.fields['stop_hz'], self.fields['pts'])
        n = len(self.sweeps) + 1
        eid = f'{self.episode_id}:sweep:{n}'
        self.active = {'n': n, 'evidence_id': eid, 'start_tick': tick, 'freqs': f, 'gain': np.zeros(len(f)),
                       'phase': np.zeros(len(f)), 'done': 0, 'fields': dict(self.fields), 'circuit_fn': circuit_fn}
        # private: the board at the start (signature, netlist, probe pads) and, per point, the board it was taken on
        self.private[eid] = {'start_tick': tick, 'end_tick': None, 'netlist': info['netlist'],
                             'signature': info['signature'], 'probe': deepcopy(info['probe']),
                             'point_signatures': [], 'board_changed_during': False, 'true_gain_db': []}
        self.sweeps.append(None)

    def _measure(self, upto):
        """Take points done..upto-1 on the circuit as the board is now."""
        a = self.active
        k0 = a['done']
        if upto <= k0:
            return
        circuit, info = a['circuit_fn']()
        h = circuit.response(a['freqs'][k0:upto])
        g, p = reading(h)
        a['gain'][k0:upto], a['phase'][k0:upto] = g, p
        priv = self.private[a['evidence_id']]
        priv['point_signatures'] += [info['signature']] * (upto - k0)
        priv['board_changed_during'] = priv['board_changed_during'] or info['signature'] != priv['signature']
        priv['true_gain_db'] += [float(x) for x in M.db(np.where(h == 0, 1e-300, h))]
        a['done'] = upto

    # --- time -------------------------------------------------------------------------------------------------
    def tick(self, tick):
        a = self.active
        if a is None:
            return False
        done = min(len(a['freqs']), (tick - a['start_tick']) // self.point_ticks)
        changed = done != a['done']
        self._measure(done)
        if done >= len(a['freqs']):
            self._finish(tick, 'complete')
            return True
        return changed

    def _finish(self, tick, status):
        a = self.active
        k = a['done']
        rows = [(float(f), float(g), float(p)) for f, g, p in zip(a['freqs'][:k], a['gain'][:k], a['phase'][:k])]
        text = sweep_csv(rows)
        hz = self.physics_hz
        public = {'evidence_id': a['evidence_id'], 'sweep': a['n'], 'status': status,
                  'started_s': a['start_tick'] / hz, 'ended_s': tick / hz,
                  'start_hz': a['fields']['start_hz'], 'stop_hz': a['fields']['stop_hz'], 'points': a['fields']['pts'],
                  'points_measured': k,
                  'csv': text, 'csv_sha256': hashlib.sha256(text.encode()).hexdigest()}
        self.sweeps[a['n'] - 1] = public
        self.private[a['evidence_id']]['end_tick'] = tick
        self.private[a['evidence_id']]['status'] = status
        self.active = None

    # --- records ---------------------------------------------------------------------------------------------
    def export(self):
        """Every sweep that has ended (public records) and whether one runs now."""
        done = [deepcopy(s) for s in self.sweeps if s is not None]
        for s in done:
            s['rows'] = [[float(r['f_hz']), float(r['gain_db']), float(r['phase_deg'])]
                         for r in csv.DictReader(io.StringIO(s['csv']))]
        return {'sweeps': done, 'running': self.active is not None,
                'running_sweep': None if self.active is None else self.active['n']}

    def display(self):
        """What the screen shows: fields, entry, Err, RUN, sweep number and the trace (points measured so far of the
        running sweep, else the last sweep's)."""
        trace = None
        if self.active is not None:
            a = self.active
            trace = {'start_hz': a['fields']['start_hz'], 'stop_hz': a['fields']['stop_hz'],
                     'f': a['freqs'][:a['done']].tolist(), 'g': a['gain'][:a['done']].tolist(), 'n': a['n']}
        else:
            last = next((s for s in reversed(self.sweeps) if s is not None), None)
            if last is not None:
                rows = list(csv.DictReader(io.StringIO(last['csv'])))
                trace = {'start_hz': last['start_hz'], 'stop_hz': last['stop_hz'],
                         'f': [float(r['f_hz']) for r in rows], 'g': [float(r['gain_db']) for r in rows],
                         'n': last['sweep']}
        return {'start_hz': self.fields['start_hz'], 'stop_hz': self.fields['stop_hz'], 'pts': self.fields['pts'],
                'entry': self.entry, 'error': self.error, 'running': self.active is not None,
                'number': len(self.sweeps), 'trace': trace}


class FilterBench:
    """The bench state between physics ticks (the backend's interface of circuit_grid.board.GridBench)."""

    def __init__(self, episode_id, profile, config, *, physics_hz=240, max_records=None):
        self.episode_id = episode_id
        self.profile = profile
        self.physics_hz = physics_hz
        inst = M.hidden_of({'model_seed': config['model_seed']})
        self.inst = inst
        self.values = M.piece_values(inst)
        self.amps = {'OA1': inst['amps'][0], 'OA2': inst['amps'][1]}
        self.blocks = dict(L.BLOCKS, **profile.three)
        self.placement = {b: ('loose',) for b in self.blocks}
        self.held = None
        self.analyzer = Analyzer(episode_id, physics_hz)
        self.state_log, self.actions, self.display_log = [], [], []
        self.transitions = []               # board changes: tick, block, from, to
        self.started = False
        self.last_change_tick = 0
        self.analysis = None
        self.current = None                 # (circuit, info) of the board as it is now
        self._display_key = None

    # --- circuit ----------------------------------------------------------------------------------------------
    def board_state(self, placement=None):
        """{block: tuple of node names} of the blocks seated on the board."""
        placement = self.placement if placement is None else placement
        return {b: tuple(L.node_name(n) for n in p[1:]) for b, p in placement.items() if p[0] == 'board'}

    def netlist(self, board=None):
        """filter_model elements in kit order (fixed rail copper first), the probe's (+ node, - node) or None."""
        board = self.board_state() if board is None else board
        els = [('W', L.node_name(a), L.node_name(b)) for a, b in self.profile.rail_edges()]
        probe = None
        for b in self.blocks:
            if b not in board:
                continue
            nodes = board[b]
            kind = self.blocks[b][0]
            if kind == 'resistor':
                els.append(('R', nodes[0], nodes[1], self.values[b]))
            elif kind == 'capacitor':
                els.append(('C', nodes[0], nodes[1], self.values[b]))
            elif kind == 'wire':
                els.append(('W', nodes[0], nodes[1]))
            elif kind == 'opamp':
                els.append(('OA', nodes[0], nodes[1], nodes[2], self.amps[b]))
            elif kind == 'probe':
                probe = (nodes[0], nodes[1])
        return els, probe

    def circuit(self, board=None):
        els, probe = self.netlist(board)
        return M.Circuit(els, L.node_name(self.profile.ground_node), source=L.node_name(self.profile.in_node),
                         probe=probe)

    def _current(self):
        if self.current is None:
            self.current = self._circuit_info()
            self.analysis = self.current[1]
        return self.current

    def _circuit_info(self):
        board = self.board_state()
        circuit = self.circuit(board)
        els, probe = self.netlist(board)
        return circuit, {'probe': None if probe is None else {'plus': probe[0], 'minus': probe[1]},
                         'signature': canonical_sha256(sorted([b, list(n)] for b, n in board.items())),
                         'netlist': [[e[0], *e[1:-1], e[-1]] if e[0] in ('R', 'C') else
                                     [e[0], *e[1:4], [k for k, v in self.amps.items() if v is e[4]][0]]
                                     if e[0] == 'OA' else list(e) for e in els]}

    # --- state ------------------------------------------------------------------------------------------------
    def all_placed(self):
        return self.held is None and all(p[0] in ('board', 'tray') for p in self.placement.values())

    def update(self, tick, placement, held, supply_on=False, switch_closed=False, reason=None, throws=None):
        changes = []
        board_change = False
        for block, new in placement.items():
            old = self.placement.get(block, ('loose',))
            if new != old:
                changes.append(block)
                self.actions.append({'tick': tick, 'event': 'placement', 'block': block,
                                     'from': placement_text(old), 'to': placement_text(new)})
                if old[0] == 'board' or new[0] == 'board':
                    board_change = True
                    self.transitions.append({'tick': tick, 't_s': tick / self.physics_hz, 'block': block,
                                             'from': placement_text(old), 'to': placement_text(new)})
        if held != self.held:
            self.actions.append({'tick': tick, 'event': 'held', 'from': self.held, 'to': held})
            changes.append('held')
        if not changes and self.started:
            return False
        self.placement = dict(placement)
        self.held = held
        if board_change or not self.started:
            self.last_change_tick = tick
            if board_change and self.analyzer.active is not None:     # the sweep runs on; later points see the change
                self.actions.append({'tick': tick, 'event': 'board_change_during_sweep',
                                     'sweep': self.analyzer.active['n']})
            self.current = self._circuit_info()
            self.analysis = self.current[1]
        self._snapshot(tick, reason or ('start' if not self.started else ','.join(changes)))
        if TRACE:
            import sys
            print('filter-trace', json.dumps({'tick': tick, 'held': held, 'board': self.board_state(),
                                              'changes': changes}), file=sys.stderr, flush=True)
        self.started = True
        return True

    def step(self, tick, remaining_ticks):
        if self.analyzer.tick(tick):
            self._log_display(tick)

    def press_key(self, tick, key, remaining_ticks):
        effect = self.analyzer.press(tick, key, remaining_ticks, self._current)
        self.actions.append({'tick': tick, 'event': 'analyzer_key', 'key': key, 'effect': effect})
        if TRACE:
            import sys
            print('filter-trace', json.dumps({'tick': tick, 'key': key, 'effect': effect}), file=sys.stderr, flush=True)
        self._log_display(tick)
        return effect

    def _log_display(self, tick):
        d = self.analyzer.display()
        key = json.dumps({k: d[k] for k in d if k != 'trace'} | {'trace_n': (d['trace'] or {}).get('n'),
                                                                 'trace_len': len((d['trace'] or {}).get('f', []))})
        if key != self._display_key:
            self._display_key = key
            self.display_log.append({'tick': tick, **json.loads(key)})

    def _snapshot(self, tick, reason):
        self.state_log.append({
            'tick': tick, 'reason': reason, 'change_tick': self.last_change_tick, 'held': self.held,
            'placement': {b: placement_text(p) for b, p in self.placement.items()},
            'board': {b: list(n) for b, n in self.board_state().items()},
            'netlist': deepcopy(self.analysis['netlist']), 'signature': self.analysis['signature'],
            'probe': deepcopy(self.analysis['probe']), 'sweep_running': self.analyzer.active is not None,
            'all_placed': self.all_placed()})

    # --- evidence ---------------------------------------------------------------------------------------------
    def observation_state(self, tick):
        d = self.analyzer.display()
        return {'tick': tick, 'held': self.held, 'all_placed': self.all_placed(),
                'last_change_tick': self.last_change_tick,
                'placement': {b: placement_text(p) for b, p in self.placement.items()},
                'signature': self.analysis['signature'], 'probe': deepcopy(self.analysis['probe']),
                'analyzer': {k: d[k] for k in d if k != 'trace'}}

    def tray_state(self):
        seats = {label: None for label in list(L.tray_seats()) + list(self.profile.three)}
        for block, place in self.placement.items():
            if place[0] == 'tray' and place[2]:
                seats[place[1]] = block
        return seats

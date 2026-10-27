"""Gate recorder runs: the timer memory of both photogates, run by run, and its private validity witnesses.

A run starts at the release (the recorder READY from REC on the keypad) and lasts the keyed record length. While it
runs the two gate timers record every pass; both gate screens list the passes of the current (or last) run, and REC
clears them. export_recording returns a run's pass records (gate, index, direction, beam times, pass time), never
which glider made a pass. Each run is evidence `<episode_id>:collision:<n>`; the sealed record also keeps, privately,
the glider of every pass, the notch the launcher was fired from, B's largest speed from the release to the first
contact of the gliders, the ticks at which the gliders touched, and per glider the ticks with any external contact and
with its catch engaged (measurement.reduce_record).
"""
from copy import deepcopy
import hashlib
import json

from .measurement import BEAM_SPACING_M, CONTACT_GAP_M, GLIDER_LENGTH_M, GateTimer, public_pass

KIND = 'airtrack_recorder'


def digest(public):
    return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()


def _mark(intervals, tick, on):
    """Extend or start an interval [first, last] of ticks at which a condition holds."""
    if not on:
        return
    if intervals and intervals[-1][1] == tick - 1:
        intervals[-1][1] = tick
    else:
        intervals.append([tick, tick])


class GateRecorder:
    def __init__(self, episode_id, hz):
        self.episode_id, self.hz = episode_id, hz
        self.runs = {}              # evidence id -> run record (public + private), in run order
        self.current = None         # the run in progress
        self.timer = None
        self.cleared = True         # screens blank (power-on or REC) until the next run records a pass

    # --- run lifecycle --------------------------------------------------------------------------------------------
    def clear(self):
        """REC: both screens clear."""
        self.cleared = True

    def begin(self, tick, length_s, positions, *, notch):
        number = len(self.runs) + 1
        eid = f'{self.episode_id}:collision:{number}'
        self.timer = GateTimer(self.hz, tick, positions)
        self.current = {'kind': KIND, 'evidence_id': eid, 'run': number, 'length_s': length_s,
                        'start_tick': tick, 'end_tick': tick + round(length_s * self.hz), 'finished': False,
                        'notch': notch, 'pass_gliders': [], 'B_speed_before_contact_m_s': 0.0,
                        'contact_ticks': [], 'external_ticks': {'A': [], 'B': []}, 'catch_ticks': {'A': [], 'B': []}}
        self.runs[eid] = self.current
        self.cleared = False
        return eid

    @property
    def active(self):
        return self.current is not None

    def sample(self, tick, positions, speeds, external, caught):
        """One physics tick after the run's start tick. positions/speeds: {'A','B'} track X (m) and X speed (m/s);
        external/caught: {'A','B'} any external contact (not the other glider) / catch engaged. Returns the run's
        evidence id when the run ends at this tick."""
        run = self.current
        if run is None or tick <= run['start_tick']:
            return None
        for p in self.timer.step(tick, positions):
            run['pass_gliders'].append({k: p[k] for k in ('gate', 'index', 'glider', 'tick_first', 'tick_second')})
        touching = positions['B'] - positions['A'] - GLIDER_LENGTH_M <= CONTACT_GAP_M
        if not run['contact_ticks'] and not touching:
            run['B_speed_before_contact_m_s'] = max(run['B_speed_before_contact_m_s'], abs(speeds['B']))
        _mark(run['contact_ticks'], tick, touching)
        for glider in ('A', 'B'):
            _mark(run['external_ticks'][glider], tick, external[glider])
            _mark(run['catch_ticks'][glider], tick, caught[glider])
        if tick >= run['end_tick']:
            return self.finish(tick)
        return None

    def finish(self, tick):
        run = self.current
        run['finished'] = True
        run['end_tick'] = min(run['end_tick'], tick)
        run['_public_passes'] = [public_pass(p) for p in self.timer.passes]
        self.current, self.timer = None, None
        return run['evidence_id']

    # --- instrument views ------------------------------------------------------------------------------------------
    def _passes(self, run):
        timer = self.timer if run is self.current else None
        if timer is not None:
            return [public_pass(p) for p in timer.passes]
        return run.get('_public_passes', [])

    def public(self, run):
        passes = self._passes(run)
        return {'evidence_id': run['evidence_id'], 'run': run['run'],
                'status': 'complete' if run['finished'] else 'recording', 'length_s': run['length_s'],
                'beam_spacing_m': BEAM_SPACING_M, 'time_resolution_ms': 0.01,
                'passes': sorted(passes, key=lambda p: (p['t_second_beam_s'], p['gate']))}

    def screens(self):
        """{gate: [(index, direction, dt_ms), ...]} shown on the gate screens: the latest run's passes (blank after REC)."""
        out = {1: [], 2: []}
        if self.cleared or not self.runs:
            return out
        run = self.current or list(self.runs.values())[-1]
        for p in self.public(run)['passes']:
            out[p['gate']].append((p['index'], p['direction'], p['dt_ms']))
        for gate in out:
            out[gate].sort()
        return out

    def find(self, key):
        """A run by evidence id, run number (int or digits) or 'latest'."""
        if not self.runs:
            raise ValueError('no gate run recorded yet')
        if key == 'latest':
            return list(self.runs.values())[-1]
        if isinstance(key, bool):
            raise ValueError('unknown run')
        if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
            number = int(key)
            for run in self.runs.values():
                if run['run'] == number:
                    return run
            raise ValueError(f'no run number {number}')
        if isinstance(key, str) and key in self.runs:
            return self.runs[key]
        raise ValueError('unknown evidence_id')

    def export(self, key):
        return deepcopy(self.public(self.find(key)))

    def evidence(self):
        """Sealed evidence: every run (a run still recording is sealed as it stands)."""
        out = {}
        for eid, run in self.runs.items():
            public = self.public(run)
            out[eid] = {'kind': KIND, 'public': deepcopy(public), 'sha256': digest(public),
                        **{k: deepcopy(v) for k, v in run.items()
                           if k not in ('kind', 'evidence_id', '_public_passes')}}
        return out

    def ticks(self):
        """{evidence id: [start tick, end tick (None while recording)]} for the process record."""
        return {eid: [r['start_tick'], r['end_tick'] if r['finished'] else None] for eid, r in self.runs.items()}

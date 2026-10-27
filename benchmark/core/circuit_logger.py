"""Two-channel voltage logger of the circuit bench, driven by the physics tick loop.

CH1 is the selected terminal pair's voltage, CH2 the voltage across R_s (see
circuit_solver). Range +/-10 V; white noise (sigma 0.5 mV) is added per sample
and the result quantised to 1 mV; a sample beyond the range is clipped and
marked over_range. Rates 30/60/120/240 Hz on integer physics ticks. Each arm
records 2-60 simulated seconds; one arm at a time; the number of arms per
episode is limited. Rows carry the episode time and the public selector
positions (nominal R_s only). No element value, actual R_s or initial voltage
is part of this interface.
"""
import csv
from copy import deepcopy
import hashlib
import io
import math

import numpy as np

RANGE_V = 10.0
RESOLUTION_V = 0.001
NOISE_V = 0.0005
INPUT_OHM = 1.0e7
RATES_HZ = (30, 60, 120, 240)
FIELDS = ('sample_id', 't_s', 'status', 'ch1_V', 'ch2_V', 'pair', 'rs', 'mode')


def selector_label(value):
    return 'transition' if value is None else value


class CircuitLogger:
    arm_s = (2.0, 60.0)

    def __init__(self, episode_id, *, max_arms, noise_seed, physics_hz=240, noise_v=NOISE_V):
        if physics_hz not in (240, 480):
            raise ValueError('unsupported physics clock')
        if type(max_arms) is not int or max_arms < 1:
            raise ValueError('max_arms must be a positive integer')
        self.episode_id = episode_id
        self.physics_hz = physics_hz
        self.max_arms = max_arms
        self.noise_v = float(noise_v)
        self._rng = np.random.default_rng(noise_seed)
        self.active = None
        self.records = {}
        self.private = {}

    @property
    def remaining_arms(self):
        return self.max_arms - len(self.records) - (self.active is not None)

    def arm(self, tick, duration_s, rate_hz, remaining_ticks):
        if self.active is not None:
            raise ValueError('logger acquisition already active')
        if self.remaining_arms <= 0:
            raise ValueError('no logger arms left')
        if (isinstance(duration_s, bool) or not isinstance(duration_s, (int, float))
                or not math.isfinite(duration_s) or not self.arm_s[0] <= duration_s <= self.arm_s[1]):
            raise ValueError(f'arm duration must be {self.arm_s[0]:g} to {self.arm_s[1]:g} simulated seconds')
        if isinstance(rate_hz, bool) or rate_hz not in RATES_HZ:
            raise ValueError(f'rate_hz must be one of {list(RATES_HZ)}')
        interval = self.physics_hz // int(rate_hz)
        samples = math.ceil(duration_s * rate_hz)
        ticks = samples * interval
        if ticks > remaining_ticks:
            raise ValueError('arm duration exceeds the remaining episode budget')
        eid = f'{self.episode_id}:logger:{len(self.records) + 1}'
        self.active = dict(evidence_id=eid, start_tick=tick, end_tick=tick + ticks, interval=interval,
                           rate_hz=int(rate_hz), duration_s=float(duration_s), rows=[],
                           counts={'ok': 0, 'over_range': 0})
        return dict(arm_id=eid, t_start_s=round(tick / self.physics_hz, 6),
                    t_end_s=round((tick + ticks) / self.physics_hz, 6), remaining_arms=self.remaining_arms)

    def sample(self, tick, ch1_v, ch2_v, switch):
        """One physics tick with the trusted noise-free channel voltages and the public selector state."""
        a = self.active
        if a is None:
            return
        if tick != a.setdefault('next_tick', a['start_tick'] + 1):
            raise ValueError('logger requires every consecutive physics tick')
        a['next_tick'] += 1
        if (tick - a['start_tick']) % a['interval'] == 0:
            status = 'ok'
            values = []
            for v in (ch1_v, ch2_v):
                if not math.isfinite(v):
                    raise ValueError('non-finite channel voltage')
                v = v + self._rng.normal(0.0, self.noise_v) if self.noise_v else v
                if abs(v) > RANGE_V:
                    status = 'over_range'
                    v = math.copysign(RANGE_V, v)
                values.append(round(round(v / RESOLUTION_V) * RESOLUTION_V, 3))
            pair, rs, mode = switch
            a['rows'].append(dict(sample_id=len(a['rows']) + 1, t_s=round(tick / self.physics_hz, 6), status=status,
                                  ch1_V=values[0], ch2_V=values[1], pair=selector_label(pair),
                                  rs=selector_label(rs), mode=selector_label(mode)))
            a['counts'][status] += 1
        if tick >= a['end_tick']:
            self._finish()

    def _finish(self):
        a = self.active
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(a['rows'])
        data = buffer.getvalue()
        eid = a['evidence_id']
        self.records[eid] = dict(
            schema_version='embodiedlab.circuit-logger/v1', episode_id=self.episode_id, evidence_id=eid,
            status='complete', clock='simulation_time since episode start', duration_s=a['duration_s'],
            sample_hz=a['rate_hz'], t_start_s=round(a['start_tick'] / self.physics_hz, 6),
            t_end_s=round(a['end_tick'] / self.physics_hz, 6), range_V=[-RANGE_V, RANGE_V],
            resolution_V=RESOLUTION_V, input_ohm=INPUT_OHM, status_counts=dict(a['counts']),
            rows=a['rows'], csv=data, sha256=hashlib.sha256(data.encode()).hexdigest())
        self.private[eid] = dict(start_tick=a['start_tick'], end_tick=a['end_tick'])
        self.active = None

    def _find(self, evidence_id):
        if self.active is not None and self.active['evidence_id'] == evidence_id:
            return self.active, 'recording'
        if evidence_id in self.records:
            return self.records[evidence_id], 'complete'
        raise ValueError('unknown or cross-episode acquisition')

    def read(self, evidence_id, t_from_s=None, t_to_s=None, max_points=600):
        """Evenly thinned samples of an acquisition (running or complete) plus per-segment statistics."""
        record, status = self._find(evidence_id)
        if type(max_points) is not int or not 2 <= max_points <= 600:
            raise ValueError('max_points must be an integer from 2 to 600')
        for bound in (t_from_s, t_to_s):
            if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (int, float))
                                      or not math.isfinite(bound)):
                raise ValueError('t_from_s and t_to_s must be finite numbers')
        rows = [r for r in record['rows'] if (t_from_s is None or r['t_s'] >= t_from_s)
                and (t_to_s is None or r['t_s'] <= t_to_s)]
        if len(rows) > max_points:
            picks = np.unique(np.round(np.linspace(0, len(rows) - 1, max_points)).astype(int))
            thinned = [rows[i] for i in picks]
        else:
            thinned = rows
        return dict(arm_id=evidence_id, status=status, samples_in_window=len(rows),
                    samples=deepcopy(thinned), segments=segments(rows))

    def export(self, evidence_id):
        if evidence_id not in self.records:
            raise ValueError('unknown, unfinished or cross-episode acquisition')
        return deepcopy(self.records[evidence_id])

    def reading(self):
        a = self.active
        return dict(display='ARMED' if a else 'IDLE', remaining_arms=self.remaining_arms,
                    active_arm_id=a['evidence_id'] if a else None,
                    completed_arm_ids=list(self.records))

    def open_record(self):
        """The acquisition still recording, with its samples so far (process record only; None when idle)."""
        a = self.active
        if a is None:
            return None
        return dict(evidence_id=a['evidence_id'], status='open', sample_hz=a['rate_hz'], duration_s=a['duration_s'],
                    start_tick=a['start_tick'], planned_end_tick=a['end_tick'],
                    last_tick=a.get('next_tick', a['start_tick'] + 1) - 1,
                    status_counts=dict(a['counts']), rows=deepcopy(a['rows']))


def segments(rows):
    """Runs of consecutive samples with the same selector state and their channel statistics."""
    out = []
    for row in rows:
        key = (row['pair'], row['rs'], row['mode'])
        if not out or out[-1]['key'] != key:
            out.append(dict(key=key, rows=[]))
        out[-1]['rows'].append(row)
    result = []
    for seg in out:
        rs = seg['rows']
        entry = dict(pair=seg['key'][0], rs=seg['key'][1], mode=seg['key'][2], t_first_s=rs[0]['t_s'],
                     t_last_s=rs[-1]['t_s'], samples=len(rs))
        for ch in ('ch1_V', 'ch2_V'):
            values = [r[ch] for r in rs]
            entry[ch] = dict(mean=round(sum(values) / len(values), 4), min=min(values), max=max(values))
        result.append(entry)
    return result

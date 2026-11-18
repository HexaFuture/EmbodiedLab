"""Instruments of the circuit board (no Isaac): autoranging DC meters, the two-channel logger and its keypad panel.

Meters (voltmeter, ammeter): 4 1/2 digits (+-19999 counts), autoranging, one conversion every CONVERSION_S; the
display holds the last conversion. A conversion reads the true value at that tick, adds white noise (NOISE_COUNTS
rms counts of the range in use) and rounds to a count. Range up above 19999 counts, down below 1800 counts (decided
within the conversion, so a shown reading is always on its proper range); above 19999 counts on the top range the
display shows OL. A voltmeter is a 10 MOhm resistor between its pins and shows V(+) - V(-); an ammeter is its range's
shunt resistance between its pins and shows the current entering its + pin.

Logger: channels CH1 and CH2 are blocks (10 MOhm each) that measure V(+) - V(-) between their pins; a channel not on
the board reads 0 V. Range +-10 V, 1 mV resolution, noise 0.5 mV rms. Records are started from the panel (REC) and
run their full length; rows are sample_id, t_s, status, ch1_V, ch2_V. The panel's display shows the live CH1 voltage
(refreshed every DISPLAY_S), the REC lamp, the entered length (SET), the rate, the record number (NO), the records
left (LEFT) and an ERR lamp.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
from copy import deepcopy

import numpy as np

CONVERSION_S = 0.4
NOISE_COUNTS = 0.5
FULL_COUNTS = 19999
DOWN_COUNTS = 1800
VOLTMETER_OHM = 1.0e7
# (name, count in SI units, decimals shown, unit, shunt ohm)
VOLT_RANGES = (('200mV', 1e-5, 2, 'mV', None), ('2V', 1e-4, 4, 'V', None), ('20V', 1e-3, 3, 'V', None))
AMP_RANGES = (('200uA', 1e-8, 2, 'uA', 1000.0), ('2mA', 1e-7, 4, 'mA', 100.0), ('20mA', 1e-6, 3, 'mA', 10.0),
              ('200mA', 1e-5, 2, 'mA', 1.0))
UNIT_SCALE = {'mV': 1e-3, 'V': 1.0, 'uA': 1e-6, 'mA': 1e-3}


def display_text(counts, decimals):
    """Sign and five digit cells with the decimal point after cell 4 - decimals (0-based), leading zeros blank
    up to the cell before the point: (sign, digits, point cell)."""
    if counts is None:
        return ' ', ' OL  ', None
    digits = f'{abs(int(counts)):05d}'
    point = 4 - decimals
    cells = list(digits)
    for k in range(point):
        if cells[k] == '0':
            cells[k] = ' '
        else:
            break
    return ('-' if counts < 0 else ' '), ''.join(cells), point


class Meter:
    def __init__(self, block_id, kind, *, rng, phase_ticks=0, physics_hz=240):
        if kind not in ('voltmeter', 'ammeter'):
            raise ValueError(kind)
        self.block_id, self.kind = block_id, kind
        self.ranges = VOLT_RANGES if kind == 'voltmeter' else AMP_RANGES
        self.range = 0
        self.rng = rng
        self.interval = round(CONVERSION_S * physics_hz)
        self.phase = int(phase_ticks) % self.interval
        self.counts = 0
        self.overload = False
        self.sample_tick = None      # tick of the conversion shown
        self.true_value = 0.0        # SI value at that conversion (private)

    def resistance(self):
        return VOLTMETER_OHM if self.kind == 'voltmeter' else self.ranges[self.range][4]

    def due(self, tick):
        return tick % self.interval == self.phase

    def convert(self, tick, read_value):
        """One conversion at `tick`. read_value(range index) -> true SI value with that range in the circuit (the
        ammeter's shunt depends on its range). Returns True when the range changed."""
        start = self.range
        for _ in range(len(self.ranges) + 1):
            value = read_value(self.range)
            step = self.ranges[self.range][1]
            noisy = value / step + self.rng.normal(0.0, NOISE_COUNTS)
            counts = int(round(noisy))
            if abs(counts) > FULL_COUNTS and self.range < len(self.ranges) - 1:
                self.range += 1
                continue
            if abs(counts) < DOWN_COUNTS and self.range > 0:
                lower = value / self.ranges[self.range - 1][1]
                if abs(lower) <= FULL_COUNTS * 0.95:
                    self.range -= 1
                    continue
            break
        self.overload = abs(counts) > FULL_COUNTS
        self.counts = None if self.overload else counts
        self.sample_tick, self.true_value = tick, float(value)
        return self.range != start

    def shown(self):
        """What the display shows: {text cells, unit, range, counts, value (SI) or None for OL, count size (SI)}."""
        name, step, decimals, unit, _ = self.ranges[self.range]
        sign, digits, point = display_text(self.counts, decimals)
        value = None if self.counts is None else round(self.counts * step, 12)
        return {'sign': sign, 'digits': digits, 'point': point, 'unit': unit, 'range': name, 'counts': self.counts,
                'value': value, 'count_si': step, 'sample_tick': self.sample_tick}

    def text(self):
        s = self.shown()
        if s['point'] is None:
            return f"{s['sign']}{s['digits']} {s['unit']}"
        d = s['digits']
        return f"{s['sign']}{d[:s['point'] + 1]}.{d[s['point'] + 1:]} {s['unit']}"


# --- logger ---------------------------------------------------------------------------------------------------------
LOGGER_RANGE_V = 10.0
LOGGER_RESOLUTION_V = 0.001
LOGGER_NOISE_V = 0.0005
LOGGER_INPUT_OHM = 1.0e7
RATES_HZ = (30, 60, 120, 240)
RECORD_S = (2, 60)
MAX_RECORDS = 20
DISPLAY_S = 0.2
FIELDS = ('sample_id', 't_s', 'status', 'ch1_V', 'ch2_V')
ENTRY_DIGITS = 2
KEYS = tuple(str(d) for d in range(10)) + ('clear', 'rate', 'rec')


class Logger:
    """Two-channel logger driven by the physics tick loop and its panel keys."""

    def __init__(self, episode_id, *, noise_seed, physics_hz=240, max_records=MAX_RECORDS):
        self.episode_id = episode_id
        self.physics_hz = physics_hz
        self.max_records = max_records
        self._rng = np.random.default_rng(noise_seed)
        self.active = None
        self.records = {}
        self.private = {}
        self.entry = ''
        self.rate_index = 2                  # 120 Hz at power-on
        self.error = False
        self.number = 0                      # record being made or last finished
        self.live_v = 0.0                    # live CH1 shown, refreshed every DISPLAY_S
        self.live_tick = None
        self.presses = []

    @property
    def rate_hz(self):
        return RATES_HZ[self.rate_index]

    @property
    def records_left(self):
        return self.max_records - len(self.records) - (self.active is not None)

    def press(self, tick, key, remaining_ticks):
        """A key press (closing edge of its switch); returns its effect (for the log)."""
        if key not in KEYS:
            raise ValueError(f'unknown logger key {key!r}')
        if self.active is not None:
            effect = 'ignored_recording'
        else:
            self.error = False
            if key == 'clear':
                self.entry, effect = '', 'cleared'
            elif key == 'rate':
                self.rate_index = (self.rate_index + 1) % len(RATES_HZ)
                effect = f'rate_{self.rate_hz}'
            elif key == 'rec':
                effect = self._rec(tick, remaining_ticks)
            elif len(self.entry) >= ENTRY_DIGITS:
                effect = 'ignored_full'
            else:
                self.entry += key
                effect = 'digit'
        self.presses.append({'tick': tick, 'key': key, 'effect': effect})
        return effect

    def _rec(self, tick, remaining_ticks):
        value = int(self.entry) if self.entry else None
        interval = self.physics_hz // self.rate_hz
        if value is None or not RECORD_S[0] <= value <= RECORD_S[1]:
            self.error, self.entry = True, ''
            return 'error_length'
        if self.records_left <= 0:
            self.error = True
            return 'error_no_records_left'
        samples = math.ceil(value * self.rate_hz)
        ticks = samples * interval
        if ticks > remaining_ticks:
            self.error = True
            return 'error_budget'
        self.number = len(self.records) + 1
        eid = f'{self.episode_id}:logger:{self.number}'
        self.active = dict(evidence_id=eid, number=self.number, start_tick=tick, end_tick=tick + ticks,
                           interval=interval, rate_hz=self.rate_hz, duration_s=float(value), rows=[],
                           counts={'ok': 0, 'over_range': 0})
        return 'start'

    def _measure(self, value):
        v = value + self._rng.normal(0.0, LOGGER_NOISE_V)
        status = 'ok'
        if abs(v) > LOGGER_RANGE_V:
            status, v = 'over_range', math.copysign(LOGGER_RANGE_V, v)
        return round(round(v / LOGGER_RESOLUTION_V) * LOGGER_RESOLUTION_V, 3), status

    def tick(self, tick, ch1_v, ch2_v):
        """Every physics tick, with the noise-free channel voltages (0 for a channel off the board)."""
        if self.live_tick is None or tick % round(DISPLAY_S * self.physics_hz) == 0:
            self.live_v, _ = self._measure(ch1_v)
            self.live_tick = tick
        a = self.active
        if a is None:
            return
        if (tick - a['start_tick']) % a['interval'] == 0 and tick > a['start_tick']:
            values, status = [], 'ok'
            for v in (ch1_v, ch2_v):
                if not math.isfinite(v):
                    raise ValueError('non-finite channel voltage')
                m, s = self._measure(v)
                values.append(m)
                status = 'over_range' if s == 'over_range' else status
            a['rows'].append(dict(sample_id=len(a['rows']) + 1, t_s=round(tick / self.physics_hz, 6), status=status,
                                  ch1_V=values[0], ch2_V=values[1]))
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
            schema_version='embodiedlab.circuit-logger/v2', episode_id=self.episode_id, evidence_id=eid,
            record_number=a['number'], status='complete', clock='simulation time since episode start',
            duration_s=a['duration_s'], sample_hz=a['rate_hz'], t_start_s=round(a['start_tick'] / self.physics_hz, 6),
            t_end_s=round(a['end_tick'] / self.physics_hz, 6), range_V=[-LOGGER_RANGE_V, LOGGER_RANGE_V],
            resolution_V=LOGGER_RESOLUTION_V, input_ohm=LOGGER_INPUT_OHM, status_counts=dict(a['counts']),
            rows=a['rows'], csv=data, sha256=hashlib.sha256(data.encode()).hexdigest())
        self.private[eid] = dict(start_tick=a['start_tick'], end_tick=a['end_tick'])
        self.active = None

    def export(self, evidence_id):
        if not isinstance(evidence_id, str) or evidence_id not in self.records:
            if self.active is not None and evidence_id == self.active['evidence_id']:
                raise ValueError('record still running')
            raise ValueError('unknown or cross-episode logger record')
        return deepcopy(self.records[evidence_id])

    def display(self):
        a = self.active
        return {'ch1_V': self.live_v, 'rec': a is not None,
                'set_s': (int(a['duration_s']) if a is not None else (int(self.entry) if self.entry else None)),
                'rate_hz': self.rate_hz, 'number': self.number, 'left': self.records_left, 'error': self.error}

    def open_record(self):
        a = self.active
        if a is None:
            return None
        return dict(evidence_id=a['evidence_id'], status='open', sample_hz=a['rate_hz'], duration_s=a['duration_s'],
                    start_tick=a['start_tick'], planned_end_tick=a['end_tick'],
                    last_tick=round(a['rows'][-1]['t_s'] * self.physics_hz) if a['rows'] else a['start_tick'],
                    status_counts=dict(a['counts']), rows=deepcopy(a['rows']))

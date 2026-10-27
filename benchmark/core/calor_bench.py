"""Calorimetry bench (calorimetry_specific_heat), independent of Isaac: where the sample is, the heater controller,
the stirrer, the thermal state, the displays and the recorder.

Every physics tick the backend reports where the sample is: 'heater' (standing in the heater well), 'cup' (its body
inside the calorimeter cup below the water line, held or not), 'air' (anywhere else), and which buttons are down.
Pressing STIR switches the cup's stirrer (it runs at the start); UP and DN move the heater's set point by 5 C within
25-95 C (it starts at 25 C). The displays refresh at LOG_HZ (the water thermometer and the heater's block thermometer;
every refresh is kept privately in rows / private, with the stirrer, the set point, the sample's place, the true
temperatures and the running record).

Recorder (the REC cap in front of the water thermometer): nothing is recorded at the start. The first press starts
record n (the LCD shows the REC mark and n), the next press ends it. While a record runs, every display refresh is
recorded (sample_id within the record, t_s, T_C the water thermometer, heater_C the heater's block thermometer).
export_temperature_log returns the recorded segments only, each with its record id <episode>:thermo_rec:<n>, its
start and end time (end null while it runs), its rows and their CSV with SHA-256; the latest export of each record is
what the sealed evidence keeps.
"""
from __future__ import annotations

import csv
import hashlib
import io
from copy import deepcopy

from calor_model import Calorimeter

LOG_HZ = 1
PLACES = ('heater', 'cup', 'air')
BUTTONS = ('stir', 'up', 'down', 'rec')
FIELDS = ('sample_id', 't_s', 'T_C', 'heater_C')


class CalorimetryBench:
    def __init__(self, episode_id, config, *, physics_hz=240, noise_seed=0):
        if physics_hz not in (240, 480):
            raise ValueError('unsupported physics clock')
        self.episode_id = episode_id
        self.physics_hz = physics_hz
        self.model = Calorimeter(noise_seed=noise_seed, **config['thermal'])
        self.tick = 0
        self.place = 'heater'
        self.stirring = True
        self.pressed = set()
        self.rows, self.private = [], []
        self.records = []                 # {'n', 'started_s', 'ended_s', 'rows', 'private'}
        self.open = None                  # the running record
        self.exports = {}                 # record id -> (its latest export as returned, export tick)
        self.transitions, self.actions = [], []
        self._next_log = 0

    @property
    def t_s(self):
        return self.tick / self.physics_hz

    def step(self, tick, world):
        """world: {'place': 'heater' | 'cup' | 'air', 'pressed': set of button ids}."""
        if tick != self.tick + 1:
            raise ValueError('calorimetry bench requires every consecutive physics tick')
        self.tick = tick
        place = world.get('place', 'air')
        if place not in PLACES:
            raise ValueError(place)
        pressed = set(world.get('pressed', ()))
        for button in sorted((pressed - self.pressed) & set(BUTTONS)):
            if button == 'stir':
                self.stirring = not self.stirring
                self.actions.append({'tick': tick, 't_s': round(self.t_s, 6), 'action': 'stir', 'stirring': self.stirring})
            elif button == 'rec':
                if self.open is None:
                    self.open = {'n': len(self.records) + 1, 'started_s': round(self.t_s, 6), 'ended_s': None,
                                 'rows': [], 'private': []}
                    self.records.append(self.open)
                    self.actions.append({'tick': tick, 't_s': round(self.t_s, 6), 'action': 'rec', 'recording': True,
                                         'record': self.open['n']})
                else:
                    self.open['ended_s'] = round(self.t_s, 6)
                    self.actions.append({'tick': tick, 't_s': round(self.t_s, 6), 'action': 'rec', 'recording': False,
                                         'record': self.open['n']})
                    self.open = None
            else:
                changed = self.model.heater.press(button)
                self.actions.append({'tick': tick, 't_s': round(self.t_s, 6), 'action': button, 'changed': changed,
                                     'set_C': self.model.heater.set_c})
        self.pressed = pressed
        if place != self.place:
            m = self.model
            self.transitions.append(dict(tick=tick, t_s=round(self.t_s, 6), place=place, sample_C=round(m.Ts, 4),
                                         water_C=round(m.Tw, 4), heater_C=round(m.heater.T, 4)))
            self.place = place
        self.model.step(1.0 / self.physics_hz, place, self.stirring)
        if tick >= self._next_log:
            self._next_log = tick + self.physics_hz // LOG_HZ
            m = self.model
            row = {'sample_id': len(self.rows) + 1, 't_s': round(self.t_s, 3), 'T_C': m.reading(),
                   'heater_C': m.heater_reading()}
            private = {'place': place, 'stirring': self.stirring, 'set_C': m.heater.set_c, 'water_C': round(m.Tw, 5),
                       'sample_C': round(m.Ts, 5), 'block_C': round(m.heater.T, 5),
                       'record': None if self.open is None else self.open['n']}
            self.rows.append(row)
            self.private.append(private)
            if self.open is not None:
                self.open['rows'].append(dict(row, sample_id=len(self.open['rows']) + 1))
                self.open['private'].append(dict(private, tick=tick))

    @property
    def recording(self):
        return self.open is not None

    def display(self):
        """What the bench shows now: the water thermometer (with the REC mark and the running record's number), the
        heater screen (SET, block) and the stirrer lamp."""
        last = self.rows[-1] if self.rows else {}
        return {'water_C': last.get('T_C'), 'heater_C': last.get('heater_C'), 'set_C': self.model.heater.set_c,
                'stirring': self.stirring, 'recording': self.recording,
                'record': None if self.open is None else self.open['n']}

    def record_id(self, n):
        return f'{self.episode_id}:thermo_rec:{n}'

    def public_record(self, record):
        """One record as export_temperature_log returns it (its rows so far)."""
        kept = record['rows']
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        for row in kept:
            writer.writerow(row)
        text = buffer.getvalue()
        return {'record_id': self.record_id(record['n']), 'record': record['n'], 'started_s': record['started_s'],
                'ended_s': record['ended_s'], 'rows': deepcopy(kept), 'csv': text,
                'sha256': hashlib.sha256(text.encode()).hexdigest()}

    def export(self):
        """The recorded segments so far (a running record with its rows so far, ended_s null)."""
        out = []
        for record in self.records:
            public = self.public_record(record)
            self.exports[public['record_id']] = (deepcopy(public), self.tick)
            out.append(public)
        return {'records': out}

    def state(self):
        """Control state for the terminal record: the stirrer, the heater set point and the recorder."""
        return {'stirring': self.stirring, 'heater_set_C': self.model.heater.set_c, 'recording': self.recording}

    def private_record(self):
        m = self.model
        return {'transitions': deepcopy(self.transitions), 'actions': deepcopy(self.actions), 'rows': len(self.rows),
                'Cs_J_K': m.Cs, 'Cw_J_K': m.Cw, 'sample_C': m.Ts, 'water_C': m.Tw, 'block_C': m.heater.T,
                'set_C': m.heater.set_c, 'stirring': self.stirring, 'records': deepcopy(self.records)}

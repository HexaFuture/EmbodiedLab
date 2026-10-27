"""Angle/position recorder of the incline bench, operated from its keypad (the shared recorder panel).

A record starts when REC takes a valid length (the bench calls start with the panel's record number and length)
and runs for exactly that many seconds of physics ticks; it cannot be stopped early. The recorder accepts only
sensor observations; it cannot access material parameters or decide whether a submitted coefficient is correct.
"""
from copy import deepcopy
import math


class InclineRecorder:
    angle_resolution_deg = .01
    position_resolution_m = .0001

    def __init__(self, physics_hz=240, prefix=None):
        if type(physics_hz) is not int or physics_hz <= 0:
            raise ValueError('positive integer sample rate required')
        self.physics_hz = physics_hz
        self.records = {}
        self.private = {}          # evidence id -> one audit row per public sample (never exported)
        self.ticks = {}            # evidence id -> [first tick, last tick] (process record only)
        self.prefix = prefix
        self.active = None
        self.active_private = []
        self.active_length = 0     # samples the running record takes
        self.latest = None
        self.last_tick = None

    def evidence_id(self, number):
        return f'incline-{number:04d}' if self.prefix is None else f'{self.prefix}:incline:{number}'

    def start(self, number, length_s):
        """Start record `number` for `length_s` seconds (the keypad's REC)."""
        if self.active is not None:
            raise ValueError('recorder already recording')
        if type(number) is not int or number <= 0:
            raise ValueError('positive record number required')
        if isinstance(length_s, bool) or not isinstance(length_s, (int, float)) or not math.isfinite(length_s) or length_s <= 0:
            raise ValueError('positive record length required')
        key = self.evidence_id(number)
        if key in self.records:
            raise ValueError('record number already used')
        self.active = {'evidence_id': key, 'record_number': number, 'length_s': length_s,
                       'physics_hz': self.physics_hz, 'samples': []}
        self.active_private = []
        self.active_length = max(1, round(length_s * self.physics_hz))
        return key

    @property
    def elapsed_s(self):
        return 0. if self.active is None else len(self.active['samples']) / self.physics_hz

    def sample(self, tick, angle_deg, position_m, private=None):
        """One physics tick of the sensors. Returns the evidence id of a record this sample completed, else None."""
        if type(tick) is not int or tick < 0:
            raise ValueError('nonnegative integer physics tick required')
        if self.last_tick is not None and tick != self.last_tick + 1:
            raise ValueError('instrument samples must follow consecutive physics ticks')
        angle, position = float(angle_deg), None if position_m is None else float(position_m)
        if not math.isfinite(angle) or position is not None and not math.isfinite(position):
            raise ValueError('nonfinite sensor reading')
        self.last_tick = tick
        self.latest = {'physics_tick': tick, 'time_s': tick/self.physics_hz,
                       'angle_deg': round(angle/self.angle_resolution_deg)*self.angle_resolution_deg,
                       'position_m': None if position is None else round(position/self.position_resolution_m)*self.position_resolution_m}
        if self.active is None:
            return None
        self.active['samples'].append(deepcopy(self.latest))
        self.active_private.append(dict(private or {}))
        if len(self.active['samples']) < self.active_length:
            return None
        key = self.active['evidence_id']
        self.records[key] = self.active
        self.private[key] = self.active_private
        self.ticks[key] = [self.active['samples'][0]['physics_tick'], tick]
        self.active, self.active_private, self.active_length = None, [], 0
        return key

    def export(self, evidence_id):
        if evidence_id not in self.records:
            raise ValueError('unknown or unfinished record')
        return deepcopy(self.records[evidence_id])

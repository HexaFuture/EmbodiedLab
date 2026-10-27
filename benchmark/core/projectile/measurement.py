"""Two optical beams at the muzzle and the receiving light curtain.

Instruments consume position samples, not engine velocity or private spring settings. A spherical ball blocks a beam
with its leading surface; matched leading-edge crossings give the centre speed over the beam separation. The curtain
records where the ball centre crosses the public receiving plane, measured from the muzzle. The public record holds the
raw edges and the crossing only; the latched notch, the plunger travel and the derived gate speed stay private.
"""
import math
from .geometry import BALL_RADIUS, GATE_X, EXIT_X, RECEIVER_CENTER_Z


class FlightRecorder:
    def __init__(self, initial_position, dt, notch_m, plunger_travel_m):
        if len(initial_position) != 3 or not all(math.isfinite(float(v)) for v in initial_position):
            raise ValueError('invalid initial optical sample')
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('invalid sample interval')
        self.previous = tuple(initial_position)
        self.notch = notch_m
        self.travel = round(plunger_travel_m / .0001) * .0001
        self.dt = float(dt)
        self.tick = 0
        self.events = []
        self.landing = None

    def step(self, position):
        p = tuple(float(v) for v in position)
        if len(p) != 3 or not all(math.isfinite(v) for v in p):
            raise ValueError("non-finite optical position")
        before = self.previous
        t0 = self.tick * self.dt
        self.tick += 1
        for index, x in enumerate(GATE_X):
            plane = x - BALL_RADIUS
            if before[0] < plane <= p[0] and not any(e['channel'] == index for e in self.events):
                u = (plane - before[0]) / (p[0] - before[0])
                # Public resolution 10 microseconds; raw samples stay private.
                self.events.append({'channel': index, 'edge': 'leading', 't_s': round((t0 + u * self.dt) / 1e-5) * 1e-5})
        if self.landing is None and before[2] > RECEIVER_CENTER_Z >= p[2] and p[0] > EXIT_X:
            u = (before[2] - RECEIVER_CENTER_Z) / (before[2] - p[2])
            self.landing = {'distance_m': round((before[0] + u * (p[0] - before[0]) - EXIT_X) / .0001) * .0001,
                            'lateral_m': round((before[1] + u * (p[1] - before[1])) / .0001) * .0001,
                            't_s': round((t0 + u * self.dt) / 1e-5) * 1e-5}
        self.previous = p

    def export(self):
        """The timer memory's raw record: gate edges and the curtain crossing, nothing derived."""
        if len(self.events) == 2 and self.events[1]['t_s'] - self.events[0]['t_s'] <= 0:
            raise ValueError('unordered optical edges')
        return {'gate_events': self.events, 'gate_separation_m': GATE_X[1] - GATE_X[0], 'landing': self.landing,
                'distance_origin': 'muzzle', 'receiver_center_z_m': RECEIVER_CENTER_Z}

    def private(self):
        """Launcher state of the shot for the evaluator and the process rule only (never returned to the Agent)."""
        speed = None
        if len(self.events) == 2:
            speed = (GATE_X[1] - GATE_X[0]) / (self.events[1]['t_s'] - self.events[0]['t_s'])
        return {'notch_m': self.notch, 'plunger_travel_m': self.travel, 'gate_speed_m_s': speed}

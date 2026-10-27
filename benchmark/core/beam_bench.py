"""Beam bench (beam_compliance), independent of Isaac: the loads on the strip, its deflected shape and the digital
indicator on its rail carriage.

Every physics tick the backend reports what presses on the strip: each body's contact normal force on the strip's
collider pieces at its contact points, and on a knife-edge saddle at the saddle's edge (the seat distance); the weights
standing in the saddles (by contact, not held); the buttons held down; the carriage's index (its distance from the
clamp edge) and speed. The shape follows the loads with the beam model's lag. The indicator's plunger tip rides on the
strip's underside at the carriage's index and presses on nothing; it reads the strip's height there against the rail:
the load deflection plus the unloaded strip's own shape over the rail (a tilt along the rail and a bow, hidden), from an
arbitrary zero (0.001 mm, plunger in = +). ZERO sets the reading to 0.000 where the carriage stands. The LCD refreshes
DISPLAY_HZ times per simulated second, each value with Gaussian noise NOISE_MM. Each refresh is logged privately (rows,
with what loaded the strip and where the carriage stood); an observation's private record keeps what the LCD showed
then (observation_record) for the readings gate.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from beam_model import Beam, G

DISPLAY_HZ = 10
RESOLUTION_MM = 0.001
NOISE_MM = 0.0015
GRID_MM = (30, 420)              # the carriage's index range on a 1 mm grid of beam stations (read by interpolation)
HOLE_TOL_M = 0.0003              # the carriage stands in a hole: its index within this of the hole, ...
HOLE_SPEED_M_S = 0.001           # ... moving slower than this (the private log's hole; the gates take their own limits)
CONFIG_MIN_N = 0.02            # a body pressing with less than this is not part of a load configuration
CONFIG_STEP_MM = 5             # off-seat load positions are recorded to this step


def lcd_text(value_mm):
    """Six seven-segment cells of an indicator LCD: sign, tens, units, then three decimals (the point stands after the
    units cell); blank tens below 10 mm; 'Err' style overflow beyond +-99.999 mm."""
    if value_mm is None:
        return '      '
    v = round(float(value_mm), 3)
    if abs(v) > 99.999:
        return ' Err  ' if v > 0 else '-Err  '
    digits = f'{abs(v):06.3f}'.replace('.', '')           # 'ttuddd'
    tens = ' ' if digits[0] == '0' else digits[0]
    sign = '-' if v < 0 and digits != '00000' else ' '
    return sign + tens + digits[1:]


def configuration(contacts):
    """Sorted [(who, where)] of the bodies pressing on the strip (a held weight is '<id>/held', the arm 'robot')."""
    out = []
    for who, c in contacts.items():
        if c['F_N'] < CONFIG_MIN_N:
            continue
        where = c['support'] if c['support'].startswith('seat:') else \
            f"x:{int(round(c['x_m'] * 1000 / CONFIG_STEP_MM) * CONFIG_STEP_MM)}"
        out.append([who + ('/held' if c.get('held') else ''), where])
    return sorted(out)


def configuration_forces(contacts):
    """{who: F_N} (0.01 N) of the configuration's entries, named as in configuration: private record only (process
    score). A weight standing on another weight is not in the configuration; the one under it carries both."""
    return {who + ('/held' if c.get('held') else ''): round(float(c['F_N']), 2)
            for who, c in contacts.items() if c['F_N'] >= CONFIG_MIN_N}



# --- loads from contact points (pure; BeamBackend feeds them PhysX's contact report) ---------------------------------
def strip_point_load(position, impulse, centre, theta, clamp_x, dt):
    """(a_m, F_N) of one contact point on a strip piece: the force normal to the piece (centre `centre`, turned by
    `theta` rad about +y) at the point's distance from the clamp. F > 0 presses down; a point below the piece's centre
    plane presses up (F < 0)."""
    up = np.array([np.sin(theta), 0.0, np.cos(theta)])
    p = np.asarray(position, float)
    f = abs(float(np.asarray(impulse, float) @ up)) / dt
    if float((p - np.asarray(centre, float)) @ up) < 0:
        f = -f
    return float(p[0]) - clamp_x, f


def saddle_point_load(position, impulse, edge_z, dt):
    """F_N of one contact point on a knife-edge saddle: its vertical force, applied at the saddle's knife edge; a point
    below the edge pushes the saddle up (F < 0)."""
    f = abs(float(impulse[2])) / dt
    return -f if float(position[2]) < edge_z else f


def summarize_points(points, seats_m, piece_frame, clamp_x, dt, saddle_edge_z, settled=None):
    """Loads of one physics step from its contact points [(kind 'piece'|'saddle', key, who, position, impulse)]: a
    piece point loads the strip at the point (piece_frame(key) -> (centre, theta)), all points of one body on one saddle
    together load it at that saddle's knife edge (seats_m[key]). A weight settled in a saddle (settled: {who: (key,
    m g)}, see BeamBackend.settled) loads it with its weight instead of its points' vertical force. Returns ({who:
    {'F_N', 'x_m', 'support', 'saddle_N'}}, [(who, a_m, F_N)]); support is 'seat:<k>' when one saddle carries at least
    half of the body's load, else 'strip'."""
    bodies, loads, on_saddle = {}, [], {}
    settled = settled or {}
    for who, (key, weight) in settled.items():
        a = float(seats_m[key])
        bodies[who] = {'F_N': weight, 'Fx': weight * a, 'saddle_N': {key: weight}}
        on_saddle[(who, key)] = weight
    for kind, key, who, position, impulse in points:
        if who in settled:
            continue
        body = bodies.setdefault(who, {'F_N': 0.0, 'Fx': 0.0, 'saddle_N': {}})
        if kind == 'piece':
            centre, theta = piece_frame(key)
            a, f = strip_point_load(position, impulse, centre, theta, clamp_x, dt)
            loads.append((who, a, f))
        else:
            a, f = float(seats_m[key]), saddle_point_load(position, impulse, saddle_edge_z[key], dt)
            on_saddle[(who, key)] = on_saddle.get((who, key), 0.0) + f
            body['saddle_N'][key] = body['saddle_N'].get(key, 0.0) + f
        body['F_N'] += f
        body['Fx'] += f * a
    loads += [(who, float(seats_m[key]), f) for (who, key), f in on_saddle.items()]
    for body in bodies.values():
        F = body['F_N']
        body['x_m'] = body.pop('Fx') / F if abs(F) > 1e-9 else 0.0
        best = max(body['saddle_N'].items(), key=lambda kv: kv[1], default=(None, 0.0))
        body['support'] = f'seat:{best[0]}' if best[0] is not None and best[1] > 0 and best[1] >= 0.5 * F else 'strip'
    return bodies, loads


def rest_loads(loads, bodies, masses, seats_m, moving, held):
    """The 'rest' load model: each weight that presses on the strip, is not held and is not moving contributes its
    weight m g at its support (a saddle's knife edge, or its contact centroid on the bare strip) instead of its contact
    forces; every other contact (the robot, a moving or held weight) keeps its contact loads. Returns [(a_m, F_N)]."""
    resting = {}
    for who, body in bodies.items():
        if who not in masses or who in held or who in moving or body['F_N'] <= 0:
            continue
        support = body['support']
        resting[who] = float(seats_m[support.split(':', 1)[1]]) if support.startswith('seat:') else body['x_m']
    return [(a, F) for who, a, F in loads if who not in resting] + [(a, masses[w] * G) for w, a in resting.items()]

def unloaded_shape_m(x_m, profile):
    """The unloaded strip's own height over the rail at x (m, down = +): a tilt along the rail and a bow."""
    return (float(profile.get('tilt_mm_per_m', 0.0)) * x_m + float(profile.get('bow_mm', 0.0)) * (x_m / 0.4) ** 2) / 1000.0


class BeamBench:
    def __init__(self, episode_id, config, *, physics_hz=240, noise_seed=0, segment_x=()):
        if physics_hz not in (240, 480):
            raise ValueError('unsupported physics clock')
        self.episode_id = episode_id
        self.physics_hz = physics_hz
        self.seat_x = {k: float(v) for k, v in config['seats_m'].items()}
        self.holes = {int(mm): int(mm) / 1000.0 for mm in config['holes_mm']}
        self.masses = {k: float(v) for k, v in config['masses_kg'].items()}
        self.length = float(config.get('length_m', 0.415))
        self.segment_x = [float(x) for x in segment_x]
        self.profile = dict(config.get('profile') or {})
        self.grid_x = np.arange(GRID_MM[0], GRID_MM[1] + 1) / 1000.0
        stations = {**{f'seat:{k}': x for k, x in self.seat_x.items()},
                    **{f'seg:{i}': x for i, x in enumerate(self.segment_x)},
                    **{f'p:{i}': x for i, x in enumerate(self.grid_x)}}
        self.beam = Beam(config['EI'], config['k_theta'], stations, length=self.length)
        self.grid_index = np.array([self.beam.names.index(f'p:{i}') for i in range(len(self.grid_x))])
        start = config.get('carriage_hole_mm', min(self.holes))
        self.carriage = {'x_m': self.holes[int(start)], 'v_m_s': 0.0}
        self.zero = float(config.get('gauge_zero_mm', 0.0)) / 1000.0      # raw height that reads 0.000
        self.rng = np.random.default_rng(noise_seed)
        self.tick = 0
        self.seated = {k: None for k in self.seat_x}
        self.config = []
        self.hole = self._hole()
        self.loads_now, self.contacts_now = [], {}
        self.pressed = set()
        self.rows, self.private = [], []
        self.transitions, self.actions = [], []
        self._next_log = 0
        self.reading = None

    @property
    def t_s(self):
        return self.tick / self.physics_hz

    def loads(self):
        """The loads [(a_m, F_N)] of the last tick."""
        return list(self.loads_now)

    def load_n(self):
        return float(sum(F for _, F in self.loads_now))

    def deflection(self, station):
        """Present deflection (m) at a station ('seat:<k>' or 'seg:<i>')."""
        return self.beam.value(station)

    def slope(self, station):
        return self.beam.slope_at(station)

    def deflection_at(self, x_m):
        """Present (lagged) load deflection (m) at x, interpolated on the 1 mm grid."""
        return float(np.interp(x_m, self.grid_x, self.beam._w[self.grid_index]))

    def plunger_w(self):
        """The load deflection at the carriage's index (where the plunger rod is drawn)."""
        return self.deflection_at(self.carriage['x_m'])

    def raw_m(self):
        """The strip's height over the rail at the carriage's index (m, down = +): load deflection and own shape."""
        x = self.carriage['x_m']
        return self.deflection_at(x) + unloaded_shape_m(x, self.profile)

    def _hole(self):
        """The hole the carriage stands in (mm), or None: within HOLE_TOL_M of it and slower than HOLE_SPEED_M_S."""
        x, v = self.carriage['x_m'], self.carriage['v_m_s']
        mm = min(self.holes, key=lambda h: abs(self.holes[h] - x))
        return mm if abs(self.holes[mm] - x) <= HOLE_TOL_M and abs(v) <= HOLE_SPEED_M_S else None

    def _read(self):
        v = (self.raw_m() - self.zero) * 1000 + self.rng.normal(0.0, NOISE_MM)
        return round(round(v / RESOLUTION_MM) * RESOLUTION_MM, 3)

    def step(self, tick, world):
        """world: {'loads': [(a_m, F_N)], 'contacts': {body: {'F_N', 'x_m', 'support', 'held'}}, 'seated': {seat:
        weight id or None}, 'pressed': set of button ids, 'carriage': {'x_m' from the clamp edge, 'v_m_s'}}."""
        if tick != self.tick + 1:
            raise ValueError('beam bench requires every consecutive physics tick')
        self.tick = tick
        self.loads_now = [(float(a), float(F)) for a, F in world.get('loads', ())]
        self.contacts_now = deepcopy(world.get('contacts', {}))
        if world.get('carriage') is not None:
            self.carriage = {'x_m': float(world['carriage']['x_m']), 'v_m_s': float(world['carriage']['v_m_s'])}
        seated = {k: world.get('seated', {}).get(k) for k in self.seat_x}
        config = configuration(self.contacts_now)
        hole = self._hole()
        if seated != self.seated or config != self.config or hole != self.hole:
            self.transitions.append({'tick': tick, 't_s': round(self.t_s, 6), 'seated': dict(seated), 'config': config,
                                     'hole_mm': hole})
            self.seated, self.config, self.hole = seated, config, hole
        pressed = set(world.get('pressed', ()))
        if 'zero' in pressed - self.pressed:
            self.zero = self.raw_m()
            self.actions.append({'tick': tick, 't_s': round(self.t_s, 6), 'action': 'zero',
                                 'carriage_mm': round(self.carriage['x_m'] * 1000, 3), 'hole_mm': self.hole,
                                 'config': deepcopy(self.config), 'config_F_N': configuration_forces(self.contacts_now)})
        self.pressed = pressed
        self.beam.step(1.0 / self.physics_hz, self.loads_now)
        if tick >= self._next_log:
            self._next_log = tick + self.physics_hz // DISPLAY_HZ
            self.reading = self._read()
            self.rows.append({'sample_id': len(self.rows) + 1, 't_s': round(self.t_s, 3), 'reading_mm': self.reading})
            self.private.append({'seated': dict(self.seated), 'config': deepcopy(self.config),
                                 'load_n': round(self.load_n(), 5), 'hole_mm': self.hole,
                                 'carriage_mm': round(self.carriage['x_m'] * 1000, 3),
                                 'w_mm': round(self.plunger_w() * 1000, 6)})

    def display(self):
        """What the indicator LCD shows: its latest refresh (mm), None before the first."""
        return self.reading

    def lcd(self):
        return lcd_text(self.reading)

    def config_since_t_s(self):
        """Since when (s) the present load configuration has stood (0 for the configuration at the start)."""
        since = 0.0
        for x in reversed(self.transitions):
            if x['config'] != self.config:
                return since
            since = x['t_s']
        return since if self.config else 0.0

    def observation_record(self, tick):
        """Private record of one observation: what the LCD showed at that moment (the latest refresh, mm), where the
        carriage stood and how fast it moved, and the load state (configuration and since when it stood)."""
        x = self.carriage['x_m']
        nearest = min(self.holes, key=lambda h: abs(self.holes[h] - x))
        return {'tick': tick, 't_s': round(tick / self.physics_hz, 6), 'display_mm': self.reading, 'lcd': self.lcd(),
                'refreshed_t_s': self.rows[-1]['t_s'] if self.rows else None,
                'carriage': {'x_mm': round(x * 1000, 4), 'v_mm_s': round(self.carriage['v_m_s'] * 1000, 4),
                             'nearest_hole_mm': nearest, 'offset_mm': round((x - self.holes[nearest]) * 1000, 4)},
                'seated': dict(self.seated), 'config': deepcopy(self.config), 'config_since_t_s': self.config_since_t_s(),
                'config_F_N': configuration_forces(self.contacts_now),
                'load_n': round(self.load_n(), 5), 'w_mm': round(self.plunger_w() * 1000, 6)}

    def private_record(self):
        return {'transitions': deepcopy(self.transitions), 'actions': deepcopy(self.actions), 'rows': len(self.rows),
                'EI': self.beam.EI, 'k_theta': self.beam.k_theta}

    def strip_state(self):
        """What presses on the strip now (private: the tidy-up gate's record)."""
        return {'load_n': round(self.load_n(), 5), 'touching': sorted(self.contacts_now),
                'config': deepcopy(configuration(self.contacts_now))}

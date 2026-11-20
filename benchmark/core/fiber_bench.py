"""Fiber coupling bench (fiber_coupling), independent of Isaac: what the knobs, seats and the POWER
button do to the light.

Every physics tick the backend hands over the world as the physics has it: the five knob angles (degrees, unwrapped,
fiber_knob's convention, in fiber_model.KNOBS order), the seats (which holder stands in the coupler's lens seat and
cap slot and in the target's slot), whether the cartridge in the coupler stands the wrong way round, the POWER cap's
state, and where (path coordinate) something not of the optics blocks the beam (the robot, a holder set down in it).
The knobs drive the model's adjusters (fiber_model.Adjusters: backlash dead bands, mirror tilts with cross-talk, the
fiber's focus travel). Each press of POWER toggles the laser (it starts off).

Light (fiber_model): the beam's central ray on the final leg (M2 to the coupler) for the mirror tilts; the spot on the
target (L_T) and on the cap at their true positions with the beam's radius there; the target's 2 mm hole passes the
part of the Gaussian beam inside it; a cartridge the wrong way round puts its lens 2 d nearer the fiber (d: its lens's
offset in the cartridge), i.e. a focus offset of -2 d. The meter reads, every METER_S (0.2 s), the coupled power
(core and cladding light, fiber_model.coupling) when the laser is on, nothing blocks the beam before the lens, the cap
is off and a lens is in, with 0.2 % noise and three significant digits (fiber_model.meter_uW); it holds the reading
until the next refresh.

record(tick) is the private record an observation is bound to (knobs, screws, mirror tilts, focus, seats, the light
and the meter as shown); actions and transitions log what changed, knob_moves every knob that turned by KNOB_LOG_DEG or
more since its last entry (for the process ladder).
"""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

import fiber_model as M

LENS_SEAT, CAP_SEAT, TARGET_SEAT = 'coupler:lens', 'coupler:cap', 'bench:target'
POWER = 'power'
KNOB_LOG_DEG = 1.0


def beam_radius(inst, p):
    """1/e^2 radius (m) of the laser beam at path coordinate p."""
    zr = math.pi * inst['w0'] ** 2 / M.LAM
    q = (p - inst['z_w']) + 1j * zr
    return math.sqrt(-M.LAM / (math.pi * (1 / q).imag))


def hole_transmission(d, w, radius, n=200):
    """Fraction of a Gaussian beam (1/e^2 radius w) through a round hole of `radius`, the beam's centre d off the
    hole's centre."""
    r = np.linspace(0.0, radius, n + 1)
    z = 4 * r * d / w ** 2
    f = (4 * r / w ** 2) * np.exp(-2 * (r - d) ** 2 / w ** 2) * (np.i0(z) * np.exp(-z))
    return float(np.clip(np.trapezoid(f, r) if hasattr(np, 'trapezoid') else np.trapz(f, r), 0.0, 1.0))


class FiberBench:
    def __init__(self, episode_id, config, *, physics_hz=240):
        """config: {'model_seed', 'noise_seed', 'cartridges': {holder id: lens index}, 'lens_offsets_m': [d per lens],
        'cap_id', 'target_id', 'p_cap', 'p_target', 'p_fiber', 'hole_r'}."""
        if physics_hz not in (240, 480):
            raise ValueError('unsupported physics clock')
        self.episode_id = episode_id
        self.physics_hz = physics_hz
        self.model_seed = int(config['model_seed'])
        self.inst = M.draw(self.model_seed)
        self.adj = M.Adjusters(self.inst)
        self.cartridges = {k: int(v) for k, v in config['cartridges'].items()}
        self.lens_offset = [float(v) for v in config['lens_offsets_m']]
        self.cap_id, self.target_id = config['cap_id'], config['target_id']
        self.p_cap, self.p_target, self.hole_r = float(config['p_cap']), float(config['p_target']), float(config['hole_r'])
        self.p_fiber = float(config['p_fiber'])
        self.rng = np.random.default_rng(int(config['noise_seed']))
        self.meter_ticks = int(round(M.METER_S * physics_hz))
        self.tick = 0
        self.power = False
        self.power_pressed = False
        self.knob_deg = np.zeros(5)
        self.seats = {LENS_SEAT: None, CAP_SEAT: None, TARGET_SEAT: None}
        self.reversed = False
        self.blocked_p = None
        self.reading_uW = 0.0
        self.read_tick = None
        self.state_version = 0
        self._last_state = None
        self.actions = []
        self.transitions = []
        self.knob_moves = []
        self._knob_logged = np.zeros(5)
        self.light = self._light()

    # --- state -------------------------------------------------------------------------------------------------
    @property
    def t_s(self):
        return self.tick / self.physics_hz

    def lens(self):
        """Index of the lens in the coupler (None)."""
        oid = self.seats.get(LENS_SEAT)
        return self.cartridges.get(oid) if oid is not None else None

    def cap_on(self):
        return self.seats.get(CAP_SEAT) is not None

    def target_in(self):
        return self.seats.get(TARGET_SEAT) is not None

    def step(self, tick, world):
        self.tick = int(tick)
        angles = np.asarray(world['knob_deg'], float)
        if angles.shape != (5,):
            raise ValueError('five knob angles expected')
        self.knob_deg = angles
        self.adj.set_knobs(angles)
        moved = np.abs(angles - self._knob_logged) >= KNOB_LOG_DEG
        if moved.any():
            self.knob_moves.append({'tick': self.tick, 'knobs': [M.KNOBS[k] for k in np.flatnonzero(moved)],
                                    'knob_deg': [round(float(v), 3) for v in angles]})
            self._knob_logged = np.where(moved, angles, self._knob_logged)
        pressed = POWER in world.get('pressed', ())
        if pressed and not self.power_pressed:
            self.power = not self.power
            self.actions.append({'tick': self.tick, 'action': 'power', 'on': self.power})
        self.power_pressed = pressed
        seats = world.get('seats', {})
        self.seats = {sid: seats.get(sid) for sid in (LENS_SEAT, CAP_SEAT, TARGET_SEAT)}
        self.reversed = bool(world.get('reversed', False)) and self.seats[LENS_SEAT] is not None
        self.blocked_p = world.get('blocked_p')
        state = (self.power, tuple(sorted((k, v) for k, v in self.seats.items() if v is not None)), self.reversed,
                 None if self.blocked_p is None else round(float(self.blocked_p), 3))
        if state != self._last_state:
            self.transitions.append({'tick': self.tick, 'power': self.power,
                                     'seats': {k: v for k, v in self.seats.items()}, 'reversed': self.reversed,
                                     'blocked_p': self.blocked_p})
            self._last_state = state
        self.light = self._light()
        if self.read_tick is None or self.tick - self.read_tick >= self.meter_ticks:
            self.reading_uW = M.meter_uW(self.light['meter_fraction'], self.rng)
            self.read_tick = self.tick
            self.state_version += 1

    # --- light ---------------------------------------------------------------------------------------------------
    def _ray(self):
        """The central ray at the lens plane (L3): offsets (m) and angles (rad), model x (world +x) and y (up)."""
        px, py = self.adj.phis()
        x, ax = M.ray(self.inst, px, 0)
        y, ay = M.ray(self.inst, py, 1)
        return np.array([x, y]), np.array([ax, ay]), (px, py)

    def offset_at(self, p):
        off, ang, _ = self._ray()
        return off + ang * (p - M.L3)

    def _light(self):
        off, ang, (px, py) = self._ray()
        lens = self.lens()
        on = self.power
        blocked = self.blocked_p
        reach = math.inf if blocked is None else float(blocked)
        out = {'power': on, 'phi_x': [float(v) for v in px], 'phi_y': [float(v) for v in py],
               'dz_m': None, 'core': 0.0, 'total': 0.0, 'eta_lens': None, 'target_T': 1.0, 'target_spot': None,
               'cap_spot': None, 'meter_fraction': 0.0, 'beam_end_p': None, 'lens': lens, 'reversed': self.reversed,
               'ray_lens_m': [float(v) for v in off], 'ray_angle_rad': [float(v) for v in ang]}
        level = 1.0
        end = M.L3 if lens is not None else self.p_fiber
        if self.target_in():
            o = off + ang * (self.p_target - M.L3)
            w = beam_radius(self.inst, self.p_target)
            T = hole_transmission(float(np.hypot(*o)), w, self.hole_r)
            out['target_T'] = T
            if on and reach > self.p_target:
                out['target_spot'] = {'offset_m': [float(v) for v in o], 'w_m': w, 'level': 1.0 - T}
            level *= T
            if T < 0.02:
                end = self.p_target
        if self.cap_on():
            o = off + ang * (self.p_cap - M.L3)
            w = beam_radius(self.inst, self.p_cap)
            if on and reach > self.p_cap and level > 1e-4:
                out['cap_spot'] = {'offset_m': [float(v) for v in o], 'w_m': w, 'level': level}
            end = min(end, self.p_cap)
        if on:
            out['beam_end_p'] = min(end, reach)
        if lens is not None:
            dz = float(self.adj.dz(lens))
            if self.reversed:
                dz -= 2 * self.lens_offset[lens]
            core, total = M.coupling(self.inst, lens, px, py, dz)
            out.update(dz_m=dz, core=float(core), total=float(total), eta_lens=float(core))
            if on and not self.cap_on() and reach > M.L3:
                out['meter_fraction'] = float(total) * level
        return out

    # --- records -------------------------------------------------------------------------------------------------
    def meter_shown(self):
        return self.reading_uW

    def record(self, tick):
        """Private record bound to an observation at `tick`."""
        light = deepcopy(self.light)
        return {'tick': int(tick), 't_s': tick / self.physics_hz, 'knob_deg': [round(float(v), 4) for v in self.knob_deg],
                'screw_deg': [round(float(v), 4) for v in self.adj.screw],
                'gap_deg': [round(float(v), 4) for v in self.adj.gap],
                'seats': dict(self.seats), 'reversed': self.reversed, 'power': self.power,
                'blocked_p': self.blocked_p, 'light': light, 'meter_uW': self.reading_uW,
                'meter_read_tick': self.read_tick,
                'true_uW': M.P_LASER_UW * light['meter_fraction']}

    def snapshot(self):
        return self.record(self.tick)

    def process_extra(self):
        return {'knob_moves': deepcopy(self.knob_moves)}

    def private_record(self):
        return {'model_version': M.MODEL_VERSION, 'model_seed': self.model_seed, 'final': self.snapshot(),
                'actions': deepcopy(self.actions), 'transitions': deepcopy(self.transitions)}

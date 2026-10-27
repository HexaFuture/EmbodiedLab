"""Optics bench (optics kit), independent of Isaac: motorised carriages on the rail, the stage controller's
keypad and display, which lens sits on which carriage, the screen image and the instrument records.

Positions are rail coordinates in millimetres from the object plane (the light box's illuminated "F"). The
controller moves a carriage at MOTOR_MM_S toward a typed target (axis key, digits, optional decimal point, GO; CLR
clears the entry); a target outside the axis travel, or one that would cross another carriage or come closer to it
than MIN_GAP_MM, is refused (ERR). The screen image is recomputed from the
lenses seated on the lens carriages (the backend reports seats) whenever every carriage has stopped; while one
moves the screen shows the last image. Jog keys (-10, -1, +1, +10 mm) move the selected axis relative to its
target. A lens's principal plane sits offset_mm from its carriage's index mark (holder and lens thickness; hidden):
the optics use the principal plane, the controller shows the mark. capture_screen returns that image with the positions and seats it was
computed for (an instrument record, cited as <episode>:screen:<n>); the 8-bit image carries sensor noise.
arm_sweep(every_mm) arms the screen camera's position trigger for the next carriage move: from the tick a carriage
starts moving until every carriage stands still it records a capture at the start, after each every_mm of travel and
at the stop (the same records as capture_screen, computed for the positions at that tick); get_capture(n) returns any
capture's record and image.
"""
from __future__ import annotations

import hashlib
import io
from copy import deepcopy

import numpy as np

import optics_model as M

MOTOR_MM_S = 25.0
MIN_GAP_MM = 60.0                  # carriages cannot pass each other; their blocks keep this distance
DISPLAY_DECIMALS = 1
KEYS = tuple(str(d) for d in range(10)) + ('point', 'clear', 'go', 'm10', 'm1', 'p1', 'p10')
JOG_MM = {'m10': -10.0, 'm1': -1.0, 'p1': 1.0, 'p10': 10.0}
NOISE_DN = 2.0                     # sensor noise of the screen camera, 8-bit counts
SWEEP_MM = (0.5, 20.0)             # allowed trigger spacing of a sweep


class OpticsBench:
    def __init__(self, episode_id, config, *, physics_hz=240, noise_seed=0):
        if physics_hz not in (240, 480):
            raise ValueError('unsupported physics clock')
        self.episode_id = episode_id
        self.physics_hz = physics_hz
        self.travel = {k: tuple(v) for k, v in config['travel_mm'].items()}       # axis -> (min, max)
        self.position = {k: float(v) for k, v in config['initial_mm'].items()}
        self.target = dict(self.position)
        # Elements in holders: lenses (f_m, aperture_m), gratings (lines_per_mm) and slits (width_m); offset_mm of each
        # from its mark.
        self.lenses = {k: {'kind': v.get('kind', 'lens'), 'f_m': float(v.get('f_m', 0.0)),
                           'aperture_m': float(v.get('aperture_m', 0.0)), 'lines_per_mm': float(v.get('lines_per_mm', 0.0)),
                           'width_m': float(v.get('width_m', 0.0)),
                           'thickness_m': float(v.get('thickness_m', 0.0)), 'n': float(v.get('n', 1.0)),
                           'offset_mm': float(v.get('offset_mm', 0.0))} for k, v in config['lenses'].items()}
        self.source = config.get('source', 'object')                 # 'object' (the F) or 'laser'
        self.laser_m = float(config.get('laser_nm', 0.0)) * 1e-9
        self.lens_axes = tuple(config['lens_axes'])                                  # carriages that carry a lens seat
        self.screen_axis = config['screen_axis']
        self.rng = np.random.default_rng(noise_seed)
        self.tick = 0
        self.pressed = set()
        self.axis = None
        self.entry = ''
        self.error = False
        self.seated = {a: None for a in self.lens_axes}                              # axis -> lens id
        self.image = None
        self.image_state = None
        self.captures = []
        self.pngs = []
        self.private = []
        self.sweep = None                  # armed trigger: {'every_mm', 'active', 'travel', 'captures'}
        self.last_sweep = None
        self.actions = []
        self.state_log = []

    @property
    def t_s(self):
        return self.tick / self.physics_hz

    def moving(self):
        return any(abs(self.target[a] - self.position[a]) > 1e-9 for a in self.position)

    def step(self, tick, world):
        """world: {'pressed': set of button ids down, 'seated': {lens axis: lens id or None}}."""
        if tick != self.tick + 1:
            raise ValueError('optics bench requires every consecutive physics tick')
        self.tick = tick
        down = set(world.get('pressed', ()))
        for button in sorted(down - self.pressed):
            self._press(button)
        self.pressed = down
        step_mm = MOTOR_MM_S / self.physics_hz
        moved = 0.0
        for a in self.position:
            d = self.target[a] - self.position[a]
            new = self.target[a] if abs(d) <= step_mm else self.position[a] + np.sign(d) * step_mm
            moved = max(moved, abs(new - self.position[a]))
            self.position[a] = new
        seated = {a: world.get('seated', {}).get(a) for a in self.lens_axes}
        if seated != self.seated:
            self.state_log.append(dict(tick=tick, event='seats', seated=dict(seated)))
            self.seated = seated
        if self.sweep is not None and moved > 0:
            self._sweep_tick(moved)
        if not self.moving():
            if self.sweep is not None and self.sweep['active']:
                self._sweep_capture()                                  # at the stop
                self.last_sweep = {'every_mm': self.sweep['every_mm'], 'captures': list(self.sweep['captures'])}
                self.state_log.append(dict(tick=tick, event='sweep', captures=list(self.sweep['captures'])))
                self.sweep = None
            state = self._optical_state()
            if state != self.image_state:
                self.image = self._compute(state)
                self.image_state = state

    # --- position-triggered captures -------------------------------------------------------------------
    def arm_sweep(self, every_mm):
        if isinstance(every_mm, bool) or not isinstance(every_mm, (int, float)) or not SWEEP_MM[0] <= every_mm <= SWEEP_MM[1]:
            raise ValueError(f'every_mm must be within {SWEEP_MM[0]}-{SWEEP_MM[1]} mm')
        if self.moving():
            raise ValueError('carriages are moving: arm the sweep before GO')
        self.sweep = {'every_mm': float(every_mm), 'active': False, 'travel': 0.0, 'captures': []}
        self.state_log.append(dict(tick=self.tick, event='arm_sweep', every_mm=float(every_mm)))
        return {'armed': True, 'every_mm': float(every_mm)}

    def _sweep_tick(self, moved):
        sw = self.sweep
        if not sw['active']:
            sw['active'] = True
            self._sweep_capture()                                      # at the start
            return
        sw['travel'] += moved
        if sw['travel'] >= sw['every_mm'] - 1e-9 and self.moving():
            sw['travel'] -= sw['every_mm']
            self._sweep_capture()

    def _sweep_capture(self):
        state = self._optical_state()
        public, _ = self._record(self._compute(state), state, trigger='sweep')
        self.sweep['captures'].append(public['capture'])
    def _press(self, button):
        self.actions.append(dict(tick=self.tick, button=button))
        kind, _, key = button.partition(':')
        if kind == 'axis':
            if key not in self.target:           # no such carriage on this bench: ERR and no carriage selected
                self.axis, self.entry, self.error = None, '', True
            else:
                self.axis, self.entry, self.error = key, '', False
        elif kind == 'key':
            self.error = False
            if key == 'clear':
                self.entry = ''
            elif key == 'go':
                self._go()
            elif key in JOG_MM:
                self._jog(JOG_MM[key])
            elif key == 'point':
                if '.' not in self.entry and len(self.entry) < 5:
                    self.entry += '.'
            elif len(self.entry.replace('.', '')) < 4 and not (self.entry.count('.') and len(self.entry.split('.')[1]) >= 1):
                self.entry += key
        else:
            raise ValueError(f'unknown button {button!r}')

    def _allowed(self, axis, value):
        if axis not in self.position:                         # no such carriage on this bench
            return False
        lo, hi = self.travel.get(axis, (None, None))
        if value is None or not lo <= value <= hi:
            return False
        for other in self.position:
            if other == axis:
                continue
            side = np.sign(self.position[axis] - self.position[other])
            for there in (self.position[other], self.target[other]):
                if np.sign(value - there) != side or abs(value - there) < MIN_GAP_MM:
                    return False
        return True

    def _go(self):
        try:
            value = float(self.entry)
        except ValueError:
            value = None
        if not self._allowed(self.axis, value):
            self.error = True
        else:
            self.target[self.axis] = value
            self.state_log.append(dict(tick=self.tick, event='go', axis=self.axis, target_mm=value))
        self.entry = ''

    def _jog(self, mm):
        if self.axis not in self.target or not self._allowed(self.axis, self.target[self.axis] + mm):  # no such carriage: ERR
            self.error = True
        else:
            self.target[self.axis] = round(self.target[self.axis] + mm, 4)
            self.state_log.append(dict(tick=self.tick, event='jog', axis=self.axis, target_mm=self.target[self.axis]))

    def _optical_state(self):
        lenses = tuple(sorted((round(self.position[a] + self.lenses[lens]['offset_mm'], 4), lens)
                              for a, lens in self.seated.items() if lens is not None))
        return lenses, round(self.position[self.screen_axis], 4)

    def _compute(self, state):
        lenses, screen = state
        if self.source == 'laser':
            # The laser beam along the axis; the first grating (orders) or slit (a diffraction pattern) in the beam
            # spreads it on the screen.
            gratings = [(x, name) for x, name in lenses
                        if self.lenses[name]['kind'] in ('grating', 'slit') and 0 < x < screen]
            if not gratings:                                  # the bare beam: one spot on the axis
                return M.grating_image(self.laser_m, 1.0, (screen - 0.0) / 1000, max_order=0)
            x, name = gratings[0]
            if self.lenses[name]['kind'] == 'slit':
                return M.slit_image(self.laser_m, self.lenses[name]['width_m'], (screen - x) / 1000)
            return M.grating_image(self.laser_m, self.lenses[name]['lines_per_mm'], (screen - x) / 1000)
        inside = [(x / 1000, self.lenses[lens]) for x, lens in lenses if 0 < x < screen]
        usable, screen_m = M.fold_plates(inside, screen / 1000)          # glass plates (plate_index)
        first = next((spec for _, spec in inside if spec['kind'] == 'lens'), None)
        aperture = first['aperture_m'] if first else 0.0
        return M.screen_image(usable, screen_m, aperture)

    # --- instruments ---------------------------------------------------------------------------------
    def display(self):
        """What the controller screen shows."""
        sweep = ({'armed': True, 'every_mm': self.sweep['every_mm'], 'captures_so_far': list(self.sweep['captures'])}
                 if self.sweep else {'armed': False, 'last_captures': list(self.last_sweep['captures']) if self.last_sweep else []})
        return {'axis': self.axis, 'entry': self.entry or None, 'error': 'ERR' if self.error else None,
                'positions_mm': {a: round(self.position[a], DISPLAY_DECIMALS) for a in sorted(self.position)},
                'moving': self.moving(), 'sweep': sweep}

    def capture(self):
        """Screen image now (8-bit grey PNG bytes) and the record; refused while a carriage moves."""
        if self.moving() or self.image is None:
            raise ValueError('carriages are moving: capture after they stop (or arm a sweep before GO)')
        return self._record(self.image, self.image_state, trigger='manual')

    def get_capture(self, n):
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(self.captures):
            raise ValueError('no such capture')
        return deepcopy(self.captures[n - 1]), self.pngs[n - 1]

    def _record(self, image, image_state, *, trigger):
        import cv2
        img = np.clip(np.round(image * 255 + self.rng.normal(0, NOISE_DN, image.shape)), 0, 255).astype(np.uint8)
        ok, png = cv2.imencode('.png', img)
        if not ok:
            raise RuntimeError('screen image encoding failed')
        data = png.tobytes()
        n = len(self.captures) + 1
        public = {'capture': n, 't_s': round(self.t_s, 6),
                  'positions_mm': {a: round(self.position[a], DISPLAY_DECIMALS) for a in sorted(self.position)},
                  'image_sha256': hashlib.sha256(data).hexdigest(), 'pixel_mm': M.SCREEN_M * 1000 / M.SCREEN_PX,
                  'trigger': trigger}
        self.captures.append(public)
        self.pngs.append(data)
        self.private.append({'capture': n, 'seated': dict(self.seated), 'image_state': deepcopy(image_state)})
        return deepcopy(public), data

    def private_record(self):
        """What the backend writes to private/chem_state.json at close (the chemistry backend's hook)."""
        return {'positions_mm': dict(self.position), 'targets_mm': dict(self.target), 'seated': dict(self.seated),
                'captures': deepcopy(self.private), 'state_log': deepcopy(self.state_log)}

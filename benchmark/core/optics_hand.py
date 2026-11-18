"""Hand-slid optics bench (recipes optics_hand_*: lens_focal, grating_wavelength, plate_index, slit_width), independent
of Isaac.

The robot slides the riders along the rail by hand (the backend reads their positions from the physics every tick)
and switches the source with its POWER push button (each press toggles it; it starts off). Positions are rail
coordinates in millimetres of each rider's index mark from the object plane (the light box's illuminated F, or the
laser's exit aperture). An element on a lens rider sits at the rider's index mark plus its hidden offset; the screen
surface is at the screen rider's index mark.

The light on the screen is recomputed whenever the source, a seated element or a rider position has changed (a
rider by more than CHANGE_MM), at most RECOMPUTE_HZ times per simulated second, and always for an observation
(current), so an observation shows the screen of its own physical state. With the source off the screen is dark.

record(tick) is the private record an observation is bound to: the riders' true index positions, the seats, the
source, the optical state the screen shows and what it shows (blur and magnification of the F image, the positions of
the laser spots on the screen scale, or those of the single slit's dark fringes; for the slit also every geometric dark
fringe on the card, shown or not).

Process record only (private; nothing the Agent sees and nothing the bench does depends on it): screen_log,
one row per screen recomputation with the optical state and what the screen shows from then on (process_extra).
"""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

import optics_model as M
import optics_screen as S

RECOMPUTE_HZ = 5.0
CHANGE_MM = 0.1
FRINGE_MARGIN_MM = 1.0             # a dark fringe this close to the screen edge is not on the screen
FRINGE_VISIBLE = 0.05              # a dark fringe shows only while the side lobe beyond it lights the card this much
TRACE_MM = 0.05                    # rider trace (process record): a row at 30 Hz when a rider moved this far
POWER = 'power'


class HandOpticsBench:
    def __init__(self, episode_id, config, *, physics_hz=240):
        if physics_hz not in (240, 480):
            raise ValueError('unsupported physics clock')
        self.episode_id = episode_id
        self.physics_hz = physics_hz
        self.frame_ticks = physics_hz // 30
        self.travel = {k: tuple(v) for k, v in config['travel_mm'].items()}
        self.position = {k: float(v) for k, v in config['initial_mm'].items()}
        self.start_mm = dict(self.position)
        self.lenses = {k: {'kind': v.get('kind', 'lens'), 'f_m': float(v.get('f_m', 0.0)),
                           'aperture_m': float(v.get('aperture_m', 0.0)), 'lines_per_mm': float(v.get('lines_per_mm', 0.0)),
                           'width_m': float(v.get('width_m', 0.0)),
                           'thickness_m': float(v.get('thickness_m', 0.0)), 'n': float(v.get('n', 1.0)),
                           'offset_mm': float(v.get('offset_mm', 0.0))} for k, v in config['lenses'].items()}
        self.source = config.get('source', 'object')
        self.laser_m = float(config.get('laser_nm', 0.0)) * 1e-9
        self.lens_axes = tuple(config['lens_axes'])
        self.screen_axis = config['screen_axis']
        self.screen_kind = 'scale' if self.source == 'laser' else 'grid'
        self.tick = 0
        self.power = False
        self.pressed = set()
        self.seated = {a: None for a in self.lens_axes}
        self.image = None                  # light on the screen (0..1), None before the first computation
        self.image_state = None
        self.image_tick = None
        self.image_version = 0
        self.actions = []
        self.state_log = []
        self.trace = [{'tick': 0, 'positions_mm': {a: round(p, 3) for a, p in self.position.items()}}]
        self._traced = dict(self.position)
        self.spots = []                    # laser spots the light shows: [(order, screen mm, level)]
        self.screen_log = []               # process record: what the screen shows after each recomputation
        self._compute_now(self._optical_state())

    @property
    def t_s(self):
        return self.tick / self.physics_hz

    # --- the physical world, every tick ---------------------------------------------------------------------
    def step(self, tick, world):
        """world: {'positions_mm': {axis: index mark mm}, 'seated': {lens axis: element or None}, 'pressed': set}."""
        if tick != self.tick + 1:
            raise ValueError('optics bench requires every consecutive physics tick')
        self.tick = tick
        down = set(world.get('pressed', ()))
        for button in sorted(down - self.pressed):
            self._press(button)
        self.pressed = down
        for a, s in (world.get('positions_mm') or {}).items():
            if a in self.position:
                self.position[a] = float(s)
        seated = {a: world.get('seated', {}).get(a) for a in self.lens_axes}
        if seated != self.seated:
            self.state_log.append(dict(tick=tick, event='seats', seated=dict(seated)))
            self.seated = seated
        if tick % self.frame_ticks == 0 and any(abs(self.position[a] - self._traced[a]) >= TRACE_MM for a in self.position):
            self.trace.append({'tick': tick, 'positions_mm': {a: round(p, 3) for a, p in self.position.items()}})
            self._traced = dict(self.position)
        state = self._optical_state()
        if self._changed(state, CHANGE_MM) and (self.image_tick is None or
                                                tick - self.image_tick >= self.physics_hz / RECOMPUTE_HZ - 1e-9):
            self._compute_now(state)

    def _press(self, button):
        self.actions.append(dict(tick=self.tick, button=button))
        if button == POWER:
            self.power = not self.power
            self.state_log.append(dict(tick=self.tick, event='power', on=self.power))
        else:
            raise ValueError(f'unknown button {button!r}')

    def current(self):
        """Make the screen show the present state exactly (an observation is taken now)."""
        state = self._optical_state()
        if self._changed(state, 1e-3):
            self._compute_now(state)
        return self.image_state

    # --- optics ----------------------------------------------------------------------------------------------
    def _optical_state(self):
        elements = tuple(sorted((round(self.position[a] + self.lenses[name]['offset_mm'], 4), name)
                                for a, name in self.seated.items() if name is not None))
        return self.power, elements, round(self.position[self.screen_axis], 4)

    def _changed(self, state, tol_mm):
        if self.image_state is None:
            return True
        power, elements, screen = state
        power0, elements0, screen0 = self.image_state
        if power != power0 or [n for _, n in elements] != [n for _, n in elements0]:
            return True
        return abs(screen - screen0) > tol_mm or any(abs(x - x0) > tol_mm for (x, _), (x0, _) in zip(elements, elements0))

    def _compute_now(self, state):
        self.image = self.light(state)
        self.spots = self.laser_spots(state)
        self.image_state = state
        self.image_tick = self.tick
        self.image_version += 1
        self._log_screen(state)

    def _log_screen(self, state):
        """Process record (private): the screen as recomputed at this tick. It only records; a failure is recorded."""
        power, elements, screen = state
        try:
            shown = self.optics(state)
        except Exception as exc:                                      # pragma: no cover - the record never stops the bench
            shown = {'error': f'{type(exc).__name__}: {exc}'}
        self.screen_log.append({'tick': self.tick, 'source_on': bool(power), 'elements_mm': [[x, n] for x, n in elements],
                                'screen_mm': screen, 'optics': shown})

    def light(self, state):
        """The light on the screen (0..1 above the room light) for an optical state."""
        power, elements, screen = state
        if not power:
            return np.zeros((8, 8))
        if self.source == 'laser':
            inside = [(x, name) for x, name in elements
                      if self.lenses[name]['kind'] in ('grating', 'slit') and 0 < x < screen]
            if not inside:
                img = S.spots_image(self.laser_m, 1.0, screen / 1000, max_order=0)
            else:
                x, name = inside[0]
                spec = self.lenses[name]
                if spec['kind'] == 'slit':          # at the card's resolution: the dark fringes are narrow
                    img = M.slit_image(self.laser_m, spec['width_m'], (screen - x) / 1000, px=S.TEXTURE_PX)
                else:
                    img = S.spots_image(self.laser_m, spec['lines_per_mm'], (screen - x) / 1000)
        else:
            inside = [(x / 1000, self.lenses[name]) for x, name in elements if 0 < x < screen]
            usable, screen_m = M.fold_plates(inside, screen / 1000)
            first = next((spec for _, spec in inside if spec['kind'] == 'lens'), None)
            # at the card's resolution: near the focus the blur is a fraction of a millimetre
            img = M.screen_image(usable, screen_m, first['aperture_m'] if first else 0.0, px=S.TEXTURE_PX)
        return np.clip(img - M.AMBIENT, 0.0, 1.0)

    def laser_spots(self, state):
        """The laser spots in the light of an optical state, for the renderer's glow: [(order, screen mm, level)], the
        same orders, centres and levels as the light image (optics_screen.spots_image); none while the laser is off or
        with the slit on the beam (its pattern is the light image alone)."""
        power, elements, screen = state
        if not power or self.source != 'laser':
            return []
        inside = [(x, name) for x, name in elements
                  if self.lenses[name]['kind'] in ('grating', 'slit') and 0 < x < screen]
        if not inside:
            lines, distance_m, top = 1.0, screen / 1000, 0
        else:
            x, name = inside[0]
            if self.lenses[name]['kind'] == 'slit':
                return []
            lines, distance_m, top = self.lenses[name]['lines_per_mm'], (screen - x) / 1000, 3
        if distance_m <= 0:
            return []
        return [(m, 1000.0 * x, 0.9 * M.ORDER_LEVELS.get(abs(m), 0.05))
                for m, x in M.grating_spots(self.laser_m, lines, distance_m, top)
                if abs(x) <= M.SCREEN_M / 2 + 5 * M.SPOT_SIGMA_M]

    def optics(self, state):
        """What the screen shows in an optical state: F-image blur and magnification, the laser spots, or the dark
        fringes of the single slit ({order: screen mm} of those on the screen; none while the laser is off)."""
        power, elements, screen = state
        out = {'source_on': bool(power)}
        if self.source == 'laser':
            slit = next(((x, n) for x, n in elements if self.lenses[n]['kind'] == 'slit' and 0 < x < screen), None)
            if slit is not None:
                x, name = slit
                width, d = self.lenses[name]['width_m'], (screen - x) / 1000
                out['dark_fringes_mm'] = {str(k): round(1000 * v, 4) for k, v in M.slit_dark_fringes(self.laser_m, width, d)
                                          if power and abs(1000 * v) <= S.SCREEN_MM / 2 - FRINGE_MARGIN_MM
                                          and self._lobe_beyond(abs(k), width, d) >= FRINGE_VISIBLE}
                out['slit_screen_mm'] = round(screen - x, 4)
                return out
            grating = next(((x, n) for x, n in elements if self.lenses[n]['kind'] == 'grating' and 0 < x < screen), None)
            if grating is None:
                out['spots_mm'] = {'0': 0.0} if power else {}
                return out
            x, name = grating
            spots = S.spot_positions_mm(self.laser_m, self.lenses[name]['lines_per_mm'], (screen - x) / 1000, max_order=3)
            out['spots_mm'] = {str(m): round(v, 4) for m, v in spots.items() if power and S.on_screen(v, 0.0)}
            out['grating_screen_mm'] = round(screen - x, 4)
            return out
        inside = [(x / 1000, self.lenses[n]) for x, n in elements if 0 < x < screen]
        lenses, screen_m = M.fold_plates(inside, screen / 1000)          # a glass block shifts the image back
        if lenses:
            first = next(spec for _, spec in inside if spec['kind'] == 'lens')
            m, blur = M.magnification_and_blur(lenses, screen_m, first['aperture_m'])
            out.update(magnification=round(float(m), 6), blur_mm=round(float(blur) * 1000, 4))
        return out

    def dark_fringes_on_card(self, state):
        """Every geometric dark fringe of the single slit on the card, shown or not ({order: screen mm}; none while the
        laser is off or without the slit in the beam). Private record for the readings gate."""
        power, elements, screen = state
        slit = next(((x, n) for x, n in elements if self.lenses[n]['kind'] == 'slit' and 0 < x < screen), None)
        if not power or slit is None:
            return {}
        x, name = slit
        return {str(k): round(1000 * v, 4) for k, v in M.slit_dark_fringes(self.laser_m, self.lenses[name]['width_m'],
                                                                         (screen - x) / 1000)
                if abs(1000 * v) <= S.SCREEN_MM / 2}

    @classmethod
    def screen_model(cls, lenses, source, laser_nm):
        """The bench reduced to what its screen shows for an optical state (optics, dark_fringes_on_card), for
        offline use on a record: lenses as the record's device keeps them."""
        bench = cls.__new__(cls)
        bench.lenses, bench.source, bench.laser_m = lenses, source, float(laser_nm or 0.0) * 1e-9
        return bench

    def _lobe_beyond(self, k, width_m, distance_m):
        """Light on the card at the side lobe just outside dark fringe k (half way to fringe k + 1)."""
        s = (k + 0.5) * self.laser_m / width_m
        if s >= 1:
            return 0.0
        return float(M.slit_profile(self.laser_m, width_m, distance_m, distance_m * s / math.sqrt(1 - s * s)))

    def record(self, tick):
        """Private record of an observation at `tick`: the state the screen shows (current first)."""
        state = self.current()
        power, elements, screen = state
        optics = self.optics(state)
        if 'dark_fringes_mm' in optics:
            optics['dark_fringes_all_mm'] = self.dark_fringes_on_card(state)
        return {'tick': tick, 't_s': round(tick / self.physics_hz, 6),
                'positions_mm': {a: round(p, 4) for a, p in self.position.items()},
                'seated': dict(self.seated), 'source_on': bool(power),
                'elements_mm': [[x, n] for x, n in elements], 'screen_mm': screen,
                'optics': optics}

    # --- records -------------------------------------------------------------------------------------------
    def snapshot(self):
        """Terminal state of the bench (seal and process record)."""
        return {'source_on': bool(self.power), 'positions_mm': {a: round(p, 4) for a, p in self.position.items()},
                'seated': dict(self.seated)}

    def private_record(self):
        return {'positions_mm': dict(self.position), 'seated': dict(self.seated), 'source_on': self.power,
                'state_log': deepcopy(self.state_log), 'trace_rows': len(self.trace)}

    def process_extra(self):
        return {'rider_trace': deepcopy(self.trace), 'start_mm': dict(self.start_mm),
                'screen_log': deepcopy(self.screen_log)}

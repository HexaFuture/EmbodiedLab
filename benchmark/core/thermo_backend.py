"""Dry-block bench physics device: the calorimetry backend with the dry-block apparatus. The gripper carries the
probe. Every physics tick the probe's place is read from its pose (standing in the block's well or on its stand,
anywhere else in the air), the UP and DOWN switches (and the ohmmeter's RANGE and AUTO keys) from their caps, and the
dry-block bench advances one tick. The displays are redrawn when what they show changes. Every observation keeps its
frames (last_frames): the episode seals them as the observation's evidence."""
import numpy as np

import thermal_lcd as L

from calor_backend import CalorBackend
from chem_backend import SEAT_TILT_DEG, SEAT_XY_TOL_M, SEAT_Z_TOL_M, _tilt_deg


class ThermoBackend(CalorBackend):
    ASSET = 'thermo_apparatus'

    def _support_paths(self, key, built):
        root = self.asset.ROOT
        return list(built['seats'].values()) + [root + '/Block', root + '/Stand']

    def _place(self, position, orientation, key):
        if key in (self.held, self.releasing) or _tilt_deg(orientation) > SEAT_TILT_DEG:
            return 'air'
        for place in ('well', 'stand'):
            xy, floor = self.seats[place]
            if np.linalg.norm(position[:2] - xy) <= SEAT_XY_TOL_M + 0.002 and abs(position[2] - floor) <= SEAT_Z_TOL_M:
                return place
        return 'air'

    def _world(self, positions, orientations):
        tilts = {key: _tilt_deg(orientations[k]) for k, key in enumerate(self.ids)}
        k = self.ids.index('probe')
        place = self._place(positions[k], orientations[k], 'probe')
        depth = self.cap_rest_z - np.asarray(self.caps.get_world_poses()[0], float)[:, 2]
        for b, d in zip(self.button_ids, depth):
            if d >= self.asset.C.SWITCH_DOWN_M:
                self.pressed.add(b)
            elif d <= self.asset.C.SWITCH_UP_M:
                self.pressed.discard(b)
        seats = {sid: None for sid in self.seats}
        if place in seats:
            seats[place] = 'probe'
        return {'place': place, 'seats': seats, 'held': self.held, 'pressed': set(self.pressed)}, depth, tilts

    def _redraw(self):
        """The controller screen, the ohmmeter or the probe thermometer and the indicator follow the bench (the next
        capture resets the renderer's history)."""
        if self.bench is None:
            return
        shown = self.bench.display()
        states = {'controller': L.controller_state(shown['set_C'], shown.get('block_C'), 4, 2),
                  'ohmmeter': L.ohmmeter_state(shown.get('ohmmeter')),
                  'thermometer': L.thermometer_state(shown.get('probe_C')),
                  'indicator': L.indicator_state(shown.get('indicator_mm'))}
        for name, (path, rows) in self.panels.items():
            if states[name] != self.panels_drawn[name]:
                self.panels_drawn[name] = L.show(self.stage, path, rows, states[name], self.panels_drawn[name])
                self.history_stale = True

    def observation(self):
        frames = super().observation()
        self.last_frames = frames               # the episode seals the observation's images as its evidence
        return frames

    def seat_audit(self):
        positions, orientations = self.cuvettes.get_world_poses()
        world, _, tilts = self._world(np.asarray(positions, float), np.asarray(orientations, float))
        return {'probe_place': world['place'], 'held': self.held, 'tilt_deg': {k: round(t, 2) for k, t in tilts.items()}}

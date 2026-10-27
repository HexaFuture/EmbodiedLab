"""Calorimetry bench physics device: the chemistry backend with the calorimetry apparatus. The gripper carries the
sample. Every physics tick the sample's place is read from its pose (standing in the heater well: seated, not held,
tilted <= 10 deg; in the cup: the centre of its metal body inside the cup below the water line, at any tilt, held
or not; anywhere else in the air), the STIR, UP and DN switches from their caps,
and the calorimetry bench advances one tick. The displays (the heater's controller screen, the water thermometer's
LCD with the recorder's REC mark and record number, the stirrer lamp) are redrawn when what they show changes."""
import numpy as np

import thermal_lcd as L
from chem_backend import SEAT_TILT_DEG, SEAT_XY_TOL_M, SEAT_Z_TOL_M, ChemBackend, _tilt_deg


class CalorBackend(ChemBackend):
    ASSET = 'calor_apparatus'

    def _build(self, asset):
        built = asset.build(self.stage, self.configuration['bench'])
        self.table_z = None
        self.panels = built.get('panels', {})
        self.panels_drawn = {name: None for name in self.panels}
        self.lamp_drawn = True
        return {'objects': built['objects'], 'buttons': built['buttons'], 'seats': built['seats'], 'screens': {}}

    def _support_paths(self, key, built):
        root = self.asset.ROOT
        return list(built['seats'].values()) + [root + '/Heater', root + '/CupFloor']

    def _seated(self, position, orientation, key, seat):
        xy, floor = self.seats[seat]
        return key not in (self.held, self.releasing) and _tilt_deg(orientation) <= SEAT_TILT_DEG and \
            np.linalg.norm(position[:2] - xy) <= SEAT_XY_TOL_M + 0.002 and abs(position[2] - floor) <= SEAT_Z_TOL_M

    def _place(self, position, orientation, key):
        asset = self.asset
        if self.table_z is None:
            self.table_z = self.seats['heater'][1] - (asset.HEATER['size'][2] + asset.SEAT_FLOOR)
        if self._seated(position, orientation, key, 'heater'):
            return 'heater'
        w, x, y, z = [float(v) for v in orientation]
        up = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        body = position + up * asset.body_centre_local()
        cx, cy = asset.CUP['centre']
        inside = np.hypot(body[0] - cx, body[1] - cy) < asset.CUP['inner_r']
        if inside and body[2] < self.table_z + asset.water_top_local(self.configuration['bench']['water_g']):
            return 'cup'
        return 'air'

    def _world(self, positions, orientations):
        tilts = {key: _tilt_deg(orientations[k]) for k, key in enumerate(self.ids)}
        k = self.ids.index('sample')
        place = self._place(positions[k], orientations[k], 'sample')
        depth = self.cap_rest_z - np.asarray(self.caps.get_world_poses()[0], float)[:, 2]
        for b, d in zip(self.button_ids, depth):
            if d >= self.asset.C.SWITCH_DOWN_M:
                self.pressed.add(b)
            elif d <= self.asset.C.SWITCH_UP_M:
                self.pressed.discard(b)
        seats = {sid: None for sid in self.seats}
        for sid in seats:
            if self._seated(positions[k], orientations[k], 'sample', sid):
                seats[sid] = 'sample'
        return {'place': place, 'seats': seats, 'held': self.held, 'pressed': set(self.pressed)}, depth, tilts

    def seat_audit(self):
        positions, orientations = self.cuvettes.get_world_poses()
        world, _, tilts = self._world(np.asarray(positions, float), np.asarray(orientations, float))
        seat = next((sid for sid, who in world['seats'].items() if who == 'sample'), None)
        return {'sample_place': world['place'], 'sample_seat': seat, 'held': self.held,
                'tilt_deg': {k: round(t, 2) for k, t in tilts.items()}}

    def _redraw(self):
        """The heater screen, the thermometer LCD (with the recorder's REC mark and record number) and the stirrer lamp
        follow the bench (the next capture resets the renderer's history)."""
        if self.bench is None:
            return
        shown = self.bench.display()
        states = {'controller': L.controller_state(shown['set_C'], shown['heater_C'], 3, 1),
                  'thermometer': L.recording_thermometer_state(shown['water_C'], shown['recording'], shown['record'])}
        for name, state in states.items():
            if name in self.panels and state != self.panels_drawn[name]:
                path, rows = self.panels[name]
                self.panels_drawn[name] = L.show(self.stage, path, rows, state, self.panels_drawn[name])
                self.history_stale = True
        if shown['stirring'] != self.lamp_drawn:
            self.asset.set_lamp(self.stage, shown['stirring'])
            self.lamp_drawn = shown['stirring']
            self.history_stale = True

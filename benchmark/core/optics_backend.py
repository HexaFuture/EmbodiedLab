"""Optics bench physics device (optics kit): the chemistry backend with the optics apparatus. Objects the gripper
carries: the element holders.

Keypad bench (recipes optics_rail_bench, slit_rail_bench, plate_rail_bench): every physics tick the carriages are
moved to the positions the optics bench computed (kinematic bodies), the element seated on each carriage is read
from the poses, the pressed controller keys from the caps, and the optics bench advances one tick. After the screen
image changes it is written as a new PNG and the screen material points at it.

Hand-slid bench (recipe optics_hand_bench, bench['drive'] == 'hand'): the riders are dynamic bodies on their rail
joints; every tick their index positions are read from the physics, each lens rider's seat collar (a kinematic body)
is kept on its rider, the seated elements and the POWER button are read, and the hand bench (optics_hand) advances
one tick. The screen is drawn from two textures: the printed card (written once) and the light on it (written when
the hand bench recomputed it); the pilot lamp and the source follow the power state."""
import numpy as np

from chem_backend import SEAT_TILT_DEG, SEAT_XY_TOL_M, SEAT_Z_TOL_M, ChemBackend, _tilt_deg

SEAT_HYST_XY_M, SEAT_HYST_Z_M = 0.003, 0.006   # a seated element stays seated within these beyond the tolerance
KEEP_LIGHT_PNGS = 4                 # older screen light textures are deleted (the renderer has loaded them)


class OpticsBackend(ChemBackend):
    ASSET = 'optics_apparatus'

    def _build(self, asset):
        bench = self.configuration['bench']
        self.hand = bench.get('drive') == 'hand'
        built = asset.build(self.stage, {'axes': bench['initial_mm'], 'lens_axes': bench['lens_axes'],
                                         'screen_axis': bench['screen_axis'], 'lenses': bench['lens_seats'],
                                         'lens_colours': bench.get('lens_colours', {}),
                                         'kinds': {k: v.get('kind', 'lens') for k, v in bench['lenses'].items()},
                                         'radii': {k: v['aperture_m'] for k, v in bench['lenses'].items()
                                                   if v.get('kind', 'lens') == 'lens'},
                                         'source': bench.get('source', 'object'), 'laser_nm': bench.get('laser_nm', 0.0),
                                         'drive': bench.get('drive', 'motor'), 'travel_mm': bench['travel_mm']})
        self.carriage_axes = list(bench['initial_mm'])
        self.carriage_bodies = None
        self.image_drawn = None
        self.image_version = 0
        self.drawn_mm = dict(bench['initial_mm'])
        self.table_z = None
        if self.hand:
            import optics_screen as S
            self.rider_bodies = None
            self.seat_bodies = None
            self.rider_mm = {a: float(v) for a, v in bench['initial_mm'].items()}
            self.seat_mm = dict(self.rider_mm)
            self.screen_ink = S.ink('scale' if bench.get('source') == 'laser' else 'grid')
            self.card_png = self.private / 'screen_card.png'
            self.cv2.imwrite(str(self.card_png), S.lit_card(np.zeros((8, 8)), self.screen_ink))
            self.light_pngs = []
            self.power_drawn = None
            self.light_drawn = None
        return {'objects': built['objects'], 'buttons': built['buttons'], 'seats': built['seats'], 'screens': {}}

    def _support_paths(self, key, built):
        root = self.asset.ROOT
        bench = self.configuration['bench']
        paths = list(built['seats'].values()) + [root + '/Stand'] + \
            [self.asset.carriage_path(a) for a in bench['initial_mm']]
        if getattr(self, 'hand', False):
            paths += [self.asset.hand_seat_path(a) for a in bench['lens_axes']]
        return paths

    # --- keypad bench: kinematic carriages ----------------------------------------------------------------------
    def _move_carriages(self):
        if self.bench is None:
            return
        if self.carriage_bodies is None:
            from isaacsim.core.prims import SingleRigidPrim
            self.carriage_bodies = {}
            for a in self.carriage_axes:
                body = SingleRigidPrim(self.asset.carriage_path(a), name=f'carriage_{a.lower()}', reset_xform_properties=False)
                body.initialize()
                self.carriage_bodies[a] = body
        for a in self.carriage_axes:
            s = self.bench.position[a]
            if abs(s - self.drawn_mm[a]) > 1e-6:
                self.asset.set_carriage(self.stage, a, s, self.carriage_bodies[a])
                self.drawn_mm[a] = s

    # --- hand bench: riders on their rail joints ---------------------------------------------------------------
    def _read_riders(self):
        """Index positions (mm) of the riders from the physics; each lens rider's seat collar follows its rider."""
        from isaacsim.core.prims import SingleRigidPrim
        asset = self.asset
        if self.rider_bodies is None:
            self.rider_bodies, self.seat_bodies = {}, {}
            for a in self.carriage_axes:
                body = SingleRigidPrim(asset.rider_path(a), name=f'rider_{a.lower()}', reset_xform_properties=False)
                body.initialize()
                self.rider_bodies[a] = body
            for a in self.configuration['bench']['lens_axes']:
                body = SingleRigidPrim(asset.seat_body_path(a), name=f'rider_seat_{a.lower()}', reset_xform_properties=False)
                body.initialize()
                self.seat_bodies[a] = body
        for a, body in self.rider_bodies.items():
            x = float(np.asarray(body.get_world_pose()[0], float)[0])
            self.rider_mm[a] = (x - asset.OBJECT_X) * 1000.0
        for a, body in self.seat_bodies.items():
            if abs(self.rider_mm[a] - self.seat_mm[a]) > 1e-4:
                asset.set_seat(self.stage, a, self.rider_mm[a], body)
                self.seat_mm[a] = self.rider_mm[a]
        return dict(self.rider_mm)

    def rider_speeds_mm_s(self):
        if not getattr(self, 'rider_bodies', None):
            return {}
        return {a: round(float(np.asarray(b.get_linear_velocity(), float)[0]) * 1000.0, 3)
                for a, b in self.rider_bodies.items()}

    def _world(self, positions, orientations):
        asset = self.asset
        bench_cfg = self.configuration['bench']
        if getattr(self, 'hand', False):
            riders = self._read_riders()
            seat_s = dict(self.seat_mm)
        else:
            self._move_carriages()
            riders = None
            seat_s = {a: (self.bench.position[a] if self.bench is not None else bench_cfg['initial_mm'][a])
                      for a in bench_cfg['lens_axes']}
        if self.table_z is None:
            self.table_z = float(np.mean([floor for _, floor in self.seats.values()])) - (asset.STAND['base'][2] + asset.SEAT_FLOOR)
        frames = dict(self.seats)
        for a in bench_cfg['lens_axes']:
            frames[f'carriage:{a}'] = (np.array([asset.rail_x(seat_s[a]), asset.RAIL_Y]),
                                       self.table_z + asset.CARRIAGE_TOP + asset.SEAT_FLOOR)
        seated = {sid: None for sid in frames}
        tilts = {}
        before = getattr(self, 'carriage_seated', {})      # carriage seat -> element seated there at the last test
        for k, key in enumerate(self.ids):
            tilts[key] = _tilt_deg(orientations[k])
            if key in (self.held, self.releasing) or tilts[key] > SEAT_TILT_DEG:
                continue
            for sid, (xy, floor) in frames.items():
                tol = SEAT_XY_TOL_M + (0.002 if sid.startswith('carriage:') else 0.0)     # the carriage moves under it
                # hysteresis: an element seated on a rider stays seated while it rocks in the moving well, so a
                # holder carried on a pushed rider does not toggle seated / unseated from tick to tick
                keep = before.get(sid) == key
                if np.linalg.norm(positions[k][:2] - xy) <= tol + (SEAT_HYST_XY_M if keep else 0.0) and \
                        abs(positions[k][2] - floor) <= SEAT_Z_TOL_M + (SEAT_HYST_Z_M if keep else 0.0):
                    seated[sid] = key
                    break
        self.carriage_seated = {sid: v for sid, v in seated.items() if sid.startswith('carriage:') and v is not None}
        depth = self.cap_rest_z - np.asarray(self.caps.get_world_poses()[0], float)[:, 2]
        for b, d in zip(self.button_ids, depth):
            if d >= asset.C.SWITCH_DOWN_M:
                self.pressed.add(b)
            elif d <= asset.C.SWITCH_UP_M:
                self.pressed.discard(b)
        lens_on = {a: seated.get(f'carriage:{a}') for a in bench_cfg['lens_axes']}
        world = {'seats': seated, 'seated': lens_on, 'held': self.held, 'pressed': set(self.pressed)}
        if riders is not None:
            world['positions_mm'] = riders
        return world, depth, tilts

    def seat_audit(self):
        positions, orientations = self.cuvettes.get_world_poses()
        world, _, tilts = self._world(np.asarray(positions, float), np.asarray(orientations, float))
        where = {key: None for key in self.ids}
        for sid, key in world['seats'].items():
            if key is not None:
                where[key] = sid
        return {'lens_seat': where, 'held': self.held, 'tilt_deg': {k: round(t, 2) for k, t in tilts.items()}}

    def observation(self):
        frames = super().observation()
        self.last_frames = frames
        return frames

    def _redraw(self):
        bench = self.bench
        if getattr(self, 'hand', False):
            return self._redraw_hand()
        if bench.image is None or bench.image_state == self.image_drawn:
            return
        self.image_version += 1
        path = self.private / f'screen_{self.image_version:04d}.png'
        img = np.clip(np.round(bench.image * 255), 0, 255).astype(np.uint8)
        self.cv2.imwrite(str(path), img)
        self.asset.set_screen_image(self.stage, path)
        self.image_drawn = bench.image_state
        self._reset_history()

    def _redraw_hand(self):
        import optics_screen as S
        bench = self.bench
        changed = False
        if bench.power != self.power_drawn:
            cfg = self.configuration['bench']
            self.asset.set_power(self.stage, bench.power, cfg.get('source', 'object'), cfg.get('laser_nm', 0.0))
            self.power_drawn = bench.power
            changed = True
        if bench.image_version != self.light_drawn:
            self.image_version += 1
            path = self.private / f'screen_light_{self.image_version:05d}.png'
            self.cv2.imwrite(str(path), S.lit_card(bench.image, self.screen_ink), [self.cv2.IMWRITE_PNG_COMPRESSION, 1])
            self.asset.set_screen_textures(self.stage, path)
            cfg = self.configuration['bench']
            if cfg.get('source') == 'laser':
                self.asset.set_laser_spots(self.stage, cfg['screen_axis'], bench.spots, cfg.get('laser_nm', 0.0))
            self.light_drawn = bench.image_version
            self.light_pngs.append(path)
            while len(self.light_pngs) > KEEP_LIGHT_PNGS:
                old = self.light_pngs.pop(0)
                try:
                    old.unlink()
                except OSError:
                    pass
            changed = True
        if changed:
            self._reset_history()

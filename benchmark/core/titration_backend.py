"""Titration bench physics device (titration_basic, titration_weak_acid): the chemistry backend with the titration
apparatus. Objects the gripper carries: the five beaker carriers (three samples, two pH buffers) and the pH electrode. Every physics tick the
carrier on the stirrer seat and the electrode's place (in the bridge of that carrier, on its park seat, held or
elsewhere) are read from the poses and the titration bench (chem_titration) advances one tick."""
import numpy as np

from chem_backend import SEAT_TILT_DEG, SEAT_XY_TOL_M, SEAT_Z_TOL_M, ChemBackend

SCREEN_RESET_EVERY_RENDERS = 30         # the pH readout flickers at 2 Hz: reset the denoiser at most once a second


class TitrationBackend(ChemBackend):
    ASSET = 'titration_apparatus'

    def _build(self, asset):
        bench = self.configuration['bench']
        built = asset.build(self.stage, {'titrant_mL': bench['titrant']['volume_mL'],
                                         'titrant_label': bench.get('titrant_label_text', '0.1'),
                                         'beaker_mL': {label: b['volume_mL'] for label, b in bench['beakers'].items()}})
        self.carriers = dict(built['carriers'])
        self.park_path = built['park']
        self.shown_screens = {name: None for name in asset.SCREENS}
        self.last_screen_reset = -10 ** 9
        return {'objects': dict(built['carriers'], electrode=built['electrode']), 'buttons': built['buttons'],
                'seats': built['seats']}

    def _support_paths(self, key, built):
        root = self.asset.ROOT
        fixed = list(built['seats'].values()) + [self.park_path, root + '/Rack', root + '/StationBase', root + '/Meter']
        return fixed + (list(self.carriers.values()) if key == 'electrode' else [])

    # --- world as the titration bench sees it -----------------------------------------------------------
    def _world(self, positions, orientations):
        from chem_backend import _tilt_deg
        asset = self.asset
        if not hasattr(self, 'park'):
            self.park = asset.park_frame(self.stage)
        seated = {sid: None for sid in self.seats}
        tilts, at = {}, {}
        for k, key in enumerate(self.ids):
            tilts[key] = _tilt_deg(orientations[k])
            at[key] = positions[k]
            if key == 'electrode' or key in (self.held, self.releasing) or tilts[key] > SEAT_TILT_DEG:
                continue
            for sid, (xy, floor) in self.seats.items():
                if np.linalg.norm(positions[k][:2] - xy) <= SEAT_XY_TOL_M and abs(positions[k][2] - floor) <= SEAT_Z_TOL_M:
                    seated[sid] = key
                    break
        stirrer = seated.get('stirrer')
        p = at['electrode']
        if 'electrode' in (self.held, self.releasing):
            electrode = 'held'
        elif stirrer is not None and np.linalg.norm(p[:2] - at[stirrer][:2]) <= asset.ELECTRODE_XY_TOL_M \
                and abs(p[2] - (at[stirrer][2] + asset.BRIDGE_TOP_Z)) <= SEAT_Z_TOL_M and tilts['electrode'] <= SEAT_TILT_DEG:
            electrode = 'holder'
        elif np.linalg.norm(p[:2] - self.park[0]) <= asset.ELECTRODE_XY_TOL_M and abs(p[2] - self.park[1]) <= SEAT_Z_TOL_M:
            electrode = 'park'
        else:
            electrode = 'free'
        depth = self.cap_rest_z - np.asarray(self.caps.get_world_poses()[0], float)[:, 2]
        for b, d in zip(self.button_ids, depth):
            if d >= asset.SWITCH_DOWN_M:
                self.pressed.add(b)
            elif d <= asset.SWITCH_UP_M:
                self.pressed.discard(b)
        return {'seats': seated, 'stirrer': stirrer, 'electrode': electrode, 'held': self.held,
                'pressed': set(self.pressed), 'tilts': {k: t for k, t in tilts.items() if k != 'electrode'},
                'electrode_tilt_deg': tilts['electrode']}, depth, tilts    # tilt: private record only

    def seat_audit(self):
        """Private: seat of each carrier, where the electrode is and whether a record runs."""
        positions, orientations = self.cuvettes.get_world_poses()
        world, _, tilts = self._world(np.asarray(positions, float), np.asarray(orientations, float))
        where = {key: None for key in self.carriers}
        for sid, key in world['seats'].items():
            if key is not None:
                where[key] = sid
        bench = getattr(self, 'bench', None)
        return {'seats': world['seats'], 'carrier_seat': where, 'electrode': world['electrode'], 'held': self.held,
                'recording': bool(bench.recorder.recording) if bench is not None else False,
                'tilt_deg': {key: round(t, 2) for key, t in tilts.items()}}

    # --- rendering of the chemical state ----------------------------------------------------------------
    def _draw_cable(self):
        """Redraw the electrode's cable (drawn only) from the electrode's pose; a change marks the denoiser history
        stale, as a liquid change does."""
        positions, orientations = self.cuvettes.get_world_poses()
        k = self.ids.index('electrode')
        p, (w, x, y, z) = np.asarray(positions[k], float), np.asarray(orientations[k], float)
        if getattr(self, 'cable_root_inv', None) is None:
            from pxr import UsdGeom
            m = UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.asset.ROOT))
            self.cable_root_inv, self.cable_drawn = np.linalg.inv(np.asarray(m, float).T), None
        axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])   # electrode's z axis
        attach = (self.cable_root_inv @ np.r_[p + self.asset.CABLE_ATTACH_Z * axis, 1.0])[:3]
        if self.cable_drawn is not None and np.max(np.abs(attach - self.cable_drawn)) < 2e-4:
            return
        self.asset.set_cable(self.stage, attach)
        self.cable_drawn = attach
        self.history_stale = True

    def _redraw(self):
        self._draw_cable()
        asset, bench = self.asset, self.bench
        levels = {label: round(b.volume_mL, 2) for label, b in bench.beakers.items()}
        fill = round(bench.titrant.remaining_mL / bench.titrant.capacity_mL, 3)
        if levels != self.drawn:
            for label, volume in levels.items():
                if self.drawn.get(label) != volume:
                    asset.set_beaker_liquid(self.stage, label, volume)
            self.drawn = levels
            self.history_stale = True
        if fill != self.bottles_drawn:
            asset.set_titrant(self.stage, fill)
            self.bottles_drawn = fill
            self.history_stale = True
        shown = bench.display()
        changed = False
        for name in asset.SCREENS:
            text = asset.screen_text(name, shown)
            if text != self.shown_screens[name]:
                self.shown_screens[name] = asset.show(self.stage, name, text, self.shown_screens[name])
                changed = True
        if changed:
            if self.render_index - self.last_screen_reset >= SCREEN_RESET_EVERY_RENDERS:
                self._reset_history()
                self.last_screen_reset = self.render_index
            else:
                self.history_stale = True

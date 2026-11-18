"""Truss bench physics device: the chemistry backend with the truss apparatus. The gripper carries the bars (their real
masses; each bar is two bodies, its main body and its tail, joined) and the loose pins. Every physics tick the bodies'
poses are read, the loose pins classified (seated in a board hole, in the stand, held, loose) and the bars
(truss_bench.classify: in the bridge on two seated pins, mounted in the tester, in a rack slot, in the scrap tray,
held, loose); the crosshead carriage and the MODE slider are read from the physics and get their ball
detents' forces, and the carriage the tester's drive force during and after a run; START is read from its cap. The
truss bench advances one tick with that world. A bar that fails in the tester fails in the physics at that tick: in
tension its joint goes (two pieces, the carriage springs on), in compression the joint turns over KINK_S (the bar
kinks, the carriage follows it in). At each rendered frame the tester's display shows its state.

The loading frame (truss_frame): the gripper also carries the weights; every tick the LoadMonitor reads the rope's pull
(the platform's guide drive), starts the loading at the first weight on the platform, fails the bridge when the pull
reaches its capacity, keeps the held load and the touches on the bridge (from a contact view built with the bench:
the platform, the weights, the bars and the loose pins against the robot's links and the weights). After the failure
the episode seals and the bridge collapses on camera (truss_loading, collapse); the bench is not stepped then."""
import numpy as np

from chem_backend import ChemBackend
import truss_bench as B
import truss_frame as F
import truss_loading as L

MODE_TOL_M = 0.004              # the slider selects a mode within this of its detent (between: no mode)
SLIDER_WINDOW_M = 0.008         # the slider's ball catches it within this of a detent
SLIDER_STIFFNESS, SLIDER_PEAK_N = 1500.0, 3.0
SETTLED_DRIFT_M = 0.0002


def _quiet_face_index_warnings():
    """Keep omni.physx.plugin's errors in the device log but not its warnings. A loose pin's shank, collar and tips
    are cylinders and cones (PhysX custom geometry) and are contact-reported (grasped parts); against the pegboard and
    the stand (static triangle meshes) each contact's report asks for the mesh's material with no face index, and
    PhysX warns (PxShape::getMaterialFromInternalFaceIndex received 0xFFFFffff) for every contact at every tick:
    ~2700 lines/s, 9.6 GB in a 4 h episode. The contact and its material are unaffected
    (the board has one material)."""
    try:
        import carb
        carb.logging.acquire_logging().set_level_threshold_for_source(
            'omni.physx.plugin', carb.logging.LogSettingBehavior.OVERRIDE, carb.logging.LEVEL_ERROR)
    except Exception as exc:                # the log keeps the warnings; nothing else depends on this
        print(f'truss_backend: physx warnings not quietened ({exc!r})', flush=True)


class TrussBackend(ChemBackend):
    ASSET = 'truss_apparatus'

    def _build(self, asset):
        _quiet_face_index_warnings()
        bench = self.configuration['bench']
        built = asset.build(self.stage, {'bars': bench['bars'], 'pins': bench['pins'],
                                         'carriage_label': bench['carriage_label'], 'mode': bench['mode']})
        self.labels = {b: spec['label'] for b, spec in bench['bars'].items()}
        self.pin_ids = sorted(bench['pins'])
        self.bar_ids = set(self.labels) | {B.tail_id(b) for b in self.labels}
        self.weight_ids = sorted(k for k in built['objects'] if F.is_weight(k))
        self._contact_view(built['objects'])
        self.frame = F.LoadMonitor(self)
        self.views = {}
        self.table_z = None
        self.screen_drawn = None
        self.failed_done = set()
        self.drive = None               # the crosshead drive of the present run (see _drive_force)
        self.kinks = {}                 # bar -> tick its buckle began
        self._tester_clear = True
        self.where = {}
        self.loading = None             # the collapse after the failure (collapse)
        return {'objects': built['objects'], 'buttons': built['buttons'], 'seats': built['seats']}

    def _contact_view(self, objects):
        """The loading's contact view (truss_frame.contact_bodies), authored with the bench (before the physics starts:
        its contact reports are authored attributes); initialised on first use."""
        from isaacsim.core.prims import RigidPrim
        row_paths, self.contact_rows, col_paths, self.contact_cols = F.contact_bodies(objects, self.pin_ids)
        self.contacts = RigidPrim(row_paths, name='truss_load_contacts', reset_xform_properties=False,
                                  track_contact_forces=True, contact_filter_prim_paths_expr=[list(col_paths) for _ in row_paths],
                                  max_contact_count=len(row_paths) * 8)
        self._contacts_ready = False

    def contact_matrix(self):
        """This tick's contact forces (rows x columns x 3) of the loading's contact view."""
        if not self._contacts_ready:
            self.contacts.initialize()
            self._contacts_ready = True
        return np.asarray(self.contacts.get_contact_force_matrix(dt=self.dt), float)

    def _support_paths(self, key, built):
        A = self.asset
        if F.is_weight(key):
            return ([f'{F.FRAME}/CaseFloor', F.CARRIAGE] + [f'{F.FRAME}/CaseWall{k}' for k in range(4)] + F.case_peg_paths()
                    + [p for k, p in built['objects'].items() if F.is_weight(k) and k != key])
        import truss_model as M
        collars = [f'{A.ROOT}/Pin{n}Collar/Collider' for n in M.FIXED]
        pins = [A.pin_path(p) for p in self.configuration['bench']['pins'] if p != key]
        return [A.ROOT + '/Board', A.ROOT + '/PinStand'] + [f'{A.ROOT}/Rack{name.title()}' for name in B.RACKS] + [
                A.ROOT + '/TrayFloor', A.ROOT + '/Pedestal', A.ROOT + '/TesterBedL', A.ROOT + '/TesterBedR',
                A.ROOT + '/TesterBedEnd', A.CARRIAGE + '/Web'] + collars + pins

    def _table_z(self):
        if self.table_z is None:
            from torsion_apparatus import table_top_z
            self.table_z = table_top_z(self.stage)
        return self.table_z

    # --- the crosshead carriage and the MODE slider ------------------------------------------------------------
    def _view(self, path, name):
        view = self.views.get(path)
        if view is None:
            from isaacsim.core.prims import RigidPrim
            view = self.views[path] = RigidPrim(path, name=name, reset_xform_properties=False)
            view.initialize()
        return view

    def _slide(self, path, name, axis):
        view = self._view(path, name)
        return (float(np.asarray(view.get_world_poses()[0], float)[0, axis]),
                float(np.asarray(view.get_linear_velocities(), float)[0, axis]))

    def carriage_state(self):
        y, v = self._slide(self.asset.CARRIAGE, 'truss_crosshead', 1)
        return {'y_m': y, 'v_m_s': v}

    def slider_state(self):
        x, v = self._slide(self.asset.SLIDER_PATH, 'truss_mode_slider', 0)
        return {'x_m': x, 'v_m_s': v}

    def mode(self):
        x = self.slider_state()['x_m']
        for m in B.MODES:
            if abs(x - self.asset.slider_x(m)) <= MODE_TOL_M:
                return m
        return None

    def _drive_force(self, y, v):
        """The tester's crosshead drive (truss_apparatus DRIVE_*), or None while the carriage is free. A run starts it
        from the hole nearest the carriage; after the run it holds until the tester's pins are clear, then returns the
        carriage to that hole and lets it go."""
        A, bench = self.asset, self.bench
        run = bench.run if bench is not None else None
        t = self.tick_index * self.dt
        if run is not None and bench.state == 'RUN' and (self.drive is None or self.drive['run'] != run['run']):
            home = min((B.detent_y(mm) for mm in B.TESTER_SPACINGS_MM), key=lambda d: abs(d - y))
            self.drive = {'run': run['run'], 'home': home, 'sign': 1.0 if run['mode'] == 'tension' else -1.0,
                          'phase': 'run', 'from': home, 'to': home + (1.0 if run['mode'] == 'tension' else -1.0)
                          * A.DRIVE_RUN_STROKE_M, 'speed': A.DRIVE_TAKEUP_M_S, 't0': t}
        d = self.drive
        if d is None:
            return None
        if d['phase'] in ('run', 'after', 'unload', 'parked') and bench.state != 'RUN' and self._tester_clear:
            d.update(phase='return', **{'from': y, 'to': d['home'], 'speed': A.DRIVE_RETURN_M_S, 't0': t})
        if d['phase'] == 'return' and abs(y - d['home']) < 3e-4 and abs(v) < 0.005 and d['target'] == d['home']:
            self.drive = None
            return None
        if d['phase'] == 'parked':
            return 0.0
        if d['phase'] == 'after' and d.get('target') == d['to'] and t - d['t0'] > A.KINK_S \
                and (abs(v) < A.DRIVE_PARK_M_S or t - d['t0'] > A.KINK_S + A.DRIVE_SETTLE_S):
            if d.get('unload'):              # after a buckle: back off the kinked bar first (unload it)
                d.update(phase='unload', **{'from': y, 'to': y + d['unload'], 'speed': A.DRIVE_TAKEUP_M_S, 't0': t})
            else:
                d['phase'] = 'parked'        # stopped: the drive holds still, pushing nothing (no detent either)
                return 0.0
        if d['phase'] == 'unload' and d.get('target') == d['to'] and (
                (abs(v) < A.DRIVE_PARK_M_S and abs(y - d['to']) < 6e-4) or t - d['t0'] > A.DRIVE_SETTLE_S):
            d['phase'] = 'parked'
            return 0.0
        step = d['to'] - d['from']
        reach = d['speed'] * (t - d['t0']) if d['speed'] else abs(step)
        d['target'] = d['to'] if reach >= abs(step) else d['from'] + np.sign(step) * reach
        return float(np.clip(A.DRIVE_STIFFNESS * (d['target'] - y) - A.DRIVE_DAMPING * v, -A.DRIVE_N, A.DRIVE_N))

    def _drive_after(self, mode):
        """The run ended in a failure of its bar in `mode`: the carriage springs on (tension) or follows the kink."""
        A, d, t = self.asset, self.drive, self.tick_index * self.dt
        if d is None:
            return
        y = self.carriage_state()['y_m']
        if mode == 'tension':
            d.update(phase='after', **{'from': y, 'to': d['home'] + A.DRIVE_SNAP_M, 'speed': None, 't0': t})
        else:
            d.update(phase='after', **{'from': y, 'to': d['home'] - A.DRIVE_BUCKLE_M,
                                       'speed': A.DRIVE_BUCKLE_M / A.KINK_S, 't0': t, 'unload': A.DRIVE_UNLOAD_M})

    def _detents(self):
        """Ball detents: within a window of a detent a force toward it, a stiffness times the offset, capped; while
        the tester drives the crosshead, its drive force instead (applied for the next physics step)."""
        A = self.asset
        carriage = self.carriage_state()
        y = carriage['y_m']
        f = self._drive_force(y, carriage['v_m_s'])
        if f is None:
            d = min((B.detent_y(mm) - y for mm in B.TESTER_SPACINGS_MM), key=abs)
            f = float(np.clip(A.DETENT_STIFFNESS * d, -A.DETENT_PEAK_N, A.DETENT_PEAK_N)) if abs(d) <= A.DETENT_WINDOW_M else 0.0
        self.views[A.CARRIAGE].apply_forces(np.array([[0.0, f, 0.0]], dtype=np.float32), is_global=True)
        x = self.slider_state()['x_m']
        d = min((A.slider_x(m) - x for m in B.MODES), key=abs)
        f = float(np.clip(SLIDER_STIFFNESS * d, -SLIDER_PEAK_N, SLIDER_PEAK_N)) if abs(d) <= SLIDER_WINDOW_M else 0.0
        self.views[A.SLIDER_PATH].apply_forces(np.array([[f, 0.0, 0.0]], dtype=np.float32), is_global=True)

    # --- world as the bench sees it -------------------------------------------------------------------------------
    def _world(self, positions, orientations):
        poses = {key: (positions[i], orientations[i]) for i, key in enumerate(self.ids)}
        held = {self.held, self.releasing} - {None}
        carriage = self.carriage_state()
        split = {b for b, m in self.bench.failed.items() if m == 'tension'} if self.bench is not None else set()
        where = B.classify(poses, self.labels, held, carriage['y_m'], self._table_z(), split=split, pins=self.pin_ids)
        self._tester_clear = B.tester_clear(poses, self.labels, carriage['y_m'], self._table_z())
        self.where = where
        depth = self.cap_rest_z - np.asarray(self.caps.get_world_poses()[0], float)[:, 2]
        for b, d in zip(self.button_ids, depth):
            if d >= self.asset.C.SWITCH_DOWN_M:
                self.pressed.add(b)
            elif d <= self.asset.C.SWITCH_UP_M:
                self.pressed.discard(b)
        tilts = {key: B.tilt_deg(orientations[i]) for i, key in enumerate(self.ids)}
        seats = {key: (w[0] + (':' + '-'.join(w[1]) if w[0] == 'bridge' else ':' + w[1] if w[1] else ''))
                 for key, w in where.items() if w[0] != 'held'}
        where = {k: w for k, w in where.items() if k not in self.pin_ids}      # the tester's bench sees the bars
        return ({'where': where, 'seats': seats, 'held': self.held, 'pressed': set(self.pressed), 'mode': self.mode(),
                 'carriage': carriage}, depth, tilts)

    def _table_poses(self, positions, orientations):
        z = self._table_z()
        return {key: (np.asarray(positions[i], float) - (0.0, 0.0, z), orientations[i]) for i, key in enumerate(self.ids)}

    def seat_audit(self):
        positions, orientations = self.cuvettes.get_world_poses()
        world, _, tilts = self._world(np.asarray(positions, float), np.asarray(orientations, float))
        carriage, slider = self.carriage_state(), self.slider_state()
        pins = {p: [w[0], w[1]] for p, w in self.where.items() if p in self.pin_ids}
        return {'where': {b: [w[0], list(w[1]) if isinstance(w[1], tuple) else w[1]] for b, w in world['where'].items()},
                'pins': pins, 'held': self.held, 'releasing': self.releasing,
                'tilt_deg': {k: round(t, 2) for k, t in tilts.items()},
                'xyz_mm': {k: [round(float(c) * 1000, 2) for c in positions[i]] for i, k in enumerate(self.ids)},
                'carriage': {'y_mm': round(carriage['y_m'] * 1000, 3), 'v_mm_s': round(carriage['v_m_s'] * 1000, 3)},
                'slider': {'x_mm': round(slider['x_m'] * 1000, 3), 'mode': world['mode']},
                'failed': dict(self.bench.failed) if self.bench is not None else {},
                'weights': F.classify_weights(self._table_poses(positions, orientations),
                                              {self.held, self.releasing} - {None}),
                'loading': self.frame.state,
                'drive': None if self.drive is None else {k: (round(v, 5) if isinstance(v, float) else v)
                                                          for k, v in self.drive.items()}}

    # --- grasp diagnostics (EMBODIEDLAB_TRUSS_GRIPDEBUG=1 only; private audit) ----------------------------------
    def _gripper_mechanism(self, positions, velocities):
        import os
        if os.environ.get('EMBODIEDLAB_TRUSS_GRIPDEBUG') == '1' and self.held is None:
            self._grip_debug(positions, velocities)
        return super()._gripper_mechanism(positions, velocities)

    def _grip_debug(self, positions, velocities):
        import json
        import grip_contact
        q = np.asarray(self.robot.get_joint_positions())
        target = getattr(self, "width_target", None)
        if target is None or float(q[7] + q[8]) <= target + grip_contact.GRIP_MARGIN_M:
            return
        self._gd_tick = getattr(self, '_gd_tick', 0) + 1
        if self._gd_tick % 6:
            return
        rows = {}
        quats = None
        for k, i in enumerate(self.ids):
            left, right = grip_contact.fingers_touching(self.finger_sensors[i], self.dt)
            if left or right:
                if quats is None:
                    quats = np.asarray(self.cuvettes.get_world_poses()[1], float)
                forces = np.asarray(self.finger_sensors[i].get_contact_force_matrix(dt=self.dt), float).reshape(-1, 3)
                rows[i] = {'touch': [left, right], 'v': np.round(np.asarray(velocities[k], float), 4).tolist(),
                           'p': np.round(np.asarray(positions[k], float), 5).tolist(),
                           'quat': np.round(quats[k], 5).tolist(), 'f': np.round(forces, 3).tolist()}
        lock = self.grip.finger_lock
        self.audit_file.write(json.dumps({'event': 'grip_debug', 'tick': int(self.tick_index), 'width': round(float(q[7] + q[8]), 5),
                                          'q': np.round(q[7:9], 5).tolist(),
                                          'lock': None if lock is None else np.round(np.asarray(lock, float), 5).tolist(),
                                          'qd': np.round(np.asarray(self.robot.get_joint_velocities())[7:9], 4).tolist(),
                                          'stalled': self.grip.stalled, 'bodies': rows,
                                          'drive': None if self.drive is None else self.drive.get('phase'),
                                          'carriage': {k: round(float(v), 5) for k, v in self.carriage_state().items()
                                                       if isinstance(v, (int, float))}}) + '\n')

    # --- stepping and drawing -----------------------------------------------------------------------------------
    def step(self, target, velocity=None):
        self._detents()
        if self.loading is not None:
            self.loading.apply()
            bench, self.bench = self.bench, None        # the sealed bench stands still
            try:
                out = super().step(target, velocity)
            finally:
                self.bench = bench
            self.loading.tick(self.dt)
            self.frame.pull = self.frame.rope_pull()[0]
            return out
        out = super().step(target, velocity)
        self._failures()
        positions, orientations = self.cuvettes.get_world_poses()
        self.frame.tick(self.tick_index, self._table_poses(np.asarray(positions, float), np.asarray(orientations, float)))
        return out

    def _failures(self):
        """A bar the tester failed this tick fails in the physics: broken in two (tension) or set to kink (compression);
        a kink turns the bar's joint over KINK_S."""
        if self.bench is None:
            return
        for bar, mode in self.bench.failed.items():
            if bar in self.failed_done:
                continue
            self.failed_done.add(bar)
            if mode == 'tension':
                self.asset.break_bar(self.stage, bar)
            else:
                self.kinks[bar] = self.tick_index
            self._drive_after(mode)
        ticks = max(1, int(round(self.asset.KINK_S / self.dt)))
        for bar, tick0 in list(self.kinks.items()):
            u = min(1.0, (self.tick_index - tick0) / ticks)
            if (self.tick_index - tick0) % 2 == 0 or u >= 1.0:
                self.asset.bend_bar(self.stage, bar, u * self.asset.kink_deg(self.labels[bar]))
            if u >= 1.0:
                del self.kinks[bar]

    def _redraw(self):
        self.frame.draw(self.stage, self.tick_index * self.dt)
        if self.bench is None:
            return
        screen = self.bench.screen()
        if screen != self.screen_drawn:
            self.screen_drawn = self.asset.show(self.stage, screen, self.screen_drawn)
            # a changed display resets the DLSS history at once (ChemBackend._render_for_capture waits for it)
            self._reset_history()

    def after_step(self, timers, tick):
        out = super().after_step(timers, tick)
        if tick % self.frame_ticks == 0:            # a frame row was written: keep the exact positions with it
            if getattr(self, '_frame_positions', None) is None:
                from collections import deque
                self._frame_positions = deque(maxlen=30)
            self._frame_positions.append(np.asarray(self.cuvettes.get_world_poses()[0], float))
        return out

    def drift_m(self):
        """The largest displacement of a bar over the last 30 frames (1 s), None before 30 frames."""
        rest = getattr(self, '_frame_positions', None)
        if rest is None or len(rest) < 30:
            return None
        stack = np.stack(list(rest))
        return float(np.max(np.linalg.norm(stack - stack[-1], axis=2)))

    def at_rest(self):
        """At rest by displacement (as the beam bench: a light body resting on a contact can report a velocity while
        it does not move): over the last 30 frames no bar moved more than SETTLED_DRIFT_M and no button was held."""
        rows = list(self.history)[-30:]
        drift = self.drift_m()
        return len(rows) >= 30 and drift is not None and drift < SETTLED_DRIFT_M and not any(r['pressed'] for r in rows)

    # --- the collapse after the failure (post-seal) -------------------------------------------------------------
    def collapse(self, plan, load_n):
        """The failed bridge gives way on camera (plan: its frozen record with the bars failed before, load_n: the pull
        at the failure): the rope goes slack at the platform (it falls onto its catch), the governing bar fails and the
        jointed bridge folds under the rope's pull (truss_loading). Stepped and recorded until done; logged privately,
        never raised."""
        import json
        loading = None
        try:
            F.slacken(self.stage)
            self.frame.slack = True
            if plan.get('capacity_N', 0) > 0 and plan.get('members'):
                loading = L.Loading(self, plan, load_n)
                loading.start()
                self.loading = loading
            ticks = 0
            while (loading is None and ticks < int(L.AFTER_S / self.dt)) or (loading is not None and not loading.done):
                self.step(None)
                self.after_step({}, self.tick_index)
                ticks += 1
        except Exception:
            import traceback
            self.audit_file.write(json.dumps({'event': 'collapse_error', 'traceback': traceback.format_exc()}) + '\n')
        finally:
            self.loading = None
            if loading is not None:
                self.audit_file.write(json.dumps({'event': 'collapse', **loading.log}, default=str) + '\n')
            self.audit_file.flush()

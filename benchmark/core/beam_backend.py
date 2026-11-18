"""Beam bench physics device: the chemistry backend with the beam apparatus. The gripper carries the weights (real
masses). Every physics tick the contacts on the strip are read from PhysX's contact report: each contact of any body
(a weight, the robot, anything) with a strip collider piece loads the strip with its normal force at its contact point;
a contact with a knife-edge saddle loads it with its vertical force at the saddle's edge (the seat distance). The beam
bench advances one tick with those loads, and the strip pieces and saddles get kinematic targets at the lagged deflected
shape (position and slope), so whatever stands on the strip moves with it. The ZERO switch is read from its cap. At each
rendered frame the same poses are written to USD for the renderer, the plunger follows the strip and the indicator LCD
shows its latest refresh.

The indicator's carriage is a dynamic body on its rail joint (rail friction: the joint's saturated damper drive). Every
physics tick its position and speed are read from the physics and, within the detent window of a hole, the ball
detent's force toward the hole is applied (beam_apparatus: DETENT_*); the bench reads the indicator at the carriage's
index (its distance from the clamp edge).

Load model (configuration bench.load_model): 'contact' (the contact normal forces, above) or 'rest' (the fallback of the
Design: a weight that is not held, touches the strip and is at rest contributes m g at its support point, a
saddle's edge or its contact centroid; the robot and moving weights still load by contact force)."""
from collections import deque

import numpy as np

from beam_model import G
from chem_backend import ChemBackend, SEAT_TILT_DEG, SEAT_XY_TOL_M, SEAT_Z_TOL_M, _tilt_deg
from beam_bench import rest_loads, summarize_points

REST_SPEED_M_S = 0.005          # 'rest' model: a weight slower than this is at rest
SETTLED_DRIFT_M = 0.0002        # at_rest: no weight moves more than this over the last 30 frames (1 s)
# Speeds come from the weights' position changes, not PhysX's reported velocities: a weight resting on a compliant
# contact reports a constant vertical velocity (about 9.5 mm/s at 100 g, 3.6 mm/s at 50 g) while it does not move.
SEATED_MIN_N = 0.02             # a weight presses on a saddle with at least this to stand in it
ROBOT_TOUCH_N = 1e-4            # the robot touches a weight: a contact point with more than this force
SETTLE_TICKS = 24               # settled in a saddle: in contact with it only, untouched, and still relative to it ...
SETTLE_M = 0.0005               # ... within this over the last SETTLE_TICKS ticks (0.1 s)


class BeamBackend(ChemBackend):
    ASSET = 'beam_apparatus'

    def _build(self, asset):
        bench = self.configuration['bench']
        built = asset.build(self.stage, {'seats_m': bench['seats_m'], 'carriage_hole_mm': bench['carriage_hole_mm'],
                                         'weights': bench['weight_labels'], 'weight_seats': bench['weight_seats'],
                                         'masses_kg': bench['masses_kg']})
        self.load_model = bench.get('load_model', 'contact')
        self.strip_view, self.strip_order = None, None
        self.strip_paths = asset.strip_bodies(bench['seats_m'])
        self.piece_index = {asset.segment_path(i): i for i in range(asset.SEGMENTS)}
        self.saddle_seat = {asset.seat_path(k): k for k in bench['seats_m']}
        self.weight_of = {path: w for w, path in built['objects'].items()}
        self.poses = None                       # the strip's latest kinematic targets (strip_paths order)
        self.poses_drawn = None
        self.lcd_drawn = None
        self.carriage_view = None
        self.table_z = None
        self._path_cache = {}
        self._contacts_tick, self._contacts = None, ({}, [])
        self._settle = {}
        return {'objects': built['objects'], 'buttons': built['buttons'], 'seats': built['seats'], 'screens': {}}

    def _support_paths(self, key, built):
        return list(built['seats'].values()) + [self.asset.ROOT + '/Rack'] + list(self.strip_paths)

    def _table_z(self):
        if self.table_z is None:
            floor = float(np.mean([f for _, f in self.seats.values()]))
            self.table_z = floor - (self.asset.RACK['base'][2] + self.asset.SEAT_FLOOR)
        return self.table_z

    # --- contacts on the strip ------------------------------------------------------------------------------------
    def _path(self, code):
        path = self._path_cache.get(code)
        if path is None:
            from pxr import PhysicsSchemaTools
            path = self._path_cache[code] = str(PhysicsSchemaTools.intToSdfPath(code))
        return path

    def _who(self, path):
        if path in self.weight_of:
            return self.weight_of[path]
        if path.startswith(self.lab.ROBOT_PATH):
            return 'robot'
        return 'other:' + path.rsplit('/', 1)[-1]

    def strip_contacts(self):
        """({body: {'F_N', 'x_m', 'support', 'saddle_N': {k: N}}}, [(body, a_m, F_N)]) of the last physics step, from
        PhysX's contact report (beam_bench.summarize_points). F > 0 presses the strip down."""
        if self._contacts_tick == self.tick_index:
            return self._contacts
        from omni.physx import get_physx_simulation_interface
        headers, data = get_physx_simulation_interface().get_contact_report()[:2]
        points, touched = [], set()
        for h in headers:
            n = int(h.num_contact_data)
            if not n:
                continue
            a0, a1 = self._path(h.actor0), self._path(h.actor1)
            for w, o in ((a0, a1), (a1, a0)):
                if w in self.weight_of and o.startswith(self.lab.ROBOT_PATH):
                    offset = int(h.contact_data_offset)
                    if any(float(np.linalg.norm(d.impulse)) / self.dt > ROBOT_TOUCH_N for d in data[offset:offset + n]):
                        touched.add(self.weight_of[w])
            part, other = (a0, a1) if (a0 in self.piece_index or a0 in self.saddle_seat) else (a1, a0)
            if part not in self.piece_index and part not in self.saddle_seat:
                continue
            if other in self.piece_index or other in self.saddle_seat:
                continue
            kind, key = ('piece', self.piece_index[part]) if part in self.piece_index else ('saddle', self.saddle_seat[part])
            who = self._who(other)
            offset = int(h.contact_data_offset)
            for d in data[offset:offset + n]:
                p, j = d.position, d.impulse
                points.append((kind, key, who, (p[0], p[1], p[2]), (j[0], j[1], j[2])))
        seats_m = self.configuration['bench']['seats_m']
        edge_z = {k: self._table_z() + self.asset.BEAM_TOP - (self.bench.deflection(f'seat:{k}') if self.bench else 0.0)
                  for k in seats_m}
        bodies, loads = summarize_points(points, seats_m, self._piece_frame, self.asset.CLAMP_X, self.dt, edge_z,
                                         self.settled(points, touched))
        self._contacts_tick, self._contacts = self.tick_index, (bodies, loads)
        return self._contacts

    def settled(self, points, touched):
        """{weight: (saddle, m g)} of the weights settled in a saddle: not held or being released, not touched by the
        robot, in contact with that saddle only, and still relative to it (SETTLE_M) over the last SETTLE_TICKS ticks.
        Such a weight loads its saddle with its weight: PhysX can report its resting contact along a well wall's normal
        (a weight that slid down the collar wall), whose vertical component is about zero."""
        masses = self.configuration['bench']['masses_kg']
        where = {}
        for kind, key, who, _, _ in points:
            if who in masses:
                where.setdefault(who, set()).add((kind, key))
        positions = np.asarray(self.cuvettes.get_world_poses()[0], float)
        out = {}
        for i, w in enumerate(self.ids):
            if w not in masses:
                continue
            track = self._settle.setdefault(w, deque(maxlen=SETTLE_TICKS))
            spots = where.get(w, set())
            if w in (self.held, self.releasing) or w in touched or len(spots) != 1 or next(iter(spots))[0] != 'saddle':
                track.clear()
                continue
            k = next(iter(spots))[1]
            relative = positions[i] + np.array([0.0, 0.0, self.bench.deflection(f'seat:{k}') if self.bench else 0.0])
            if track and track[0][0] != k:
                track.clear()
            track.append((k, relative))
            if len(track) == SETTLE_TICKS and max(float(np.max(np.abs(r - relative))) for _, r in track) <= SETTLE_M:
                out[w] = (k, float(masses[w]) * G)
        return out

    def _piece_frame(self, i):
        """(centre, theta) of strip piece i at the last kinematic target (the authored pose before the first one)."""
        asset = self.asset
        if self.poses is None:
            return (asset.CLAMP_X + asset.segment_x(i), asset.BEAM_Y,
                    self._table_z() + asset.BEAM_TOP - asset.BEAM[2] / 2), 0.0
        pose = self.poses[i]
        return tuple(pose[:3]), 2 * float(np.arctan2(pose[4], pose[6]))

    def _loads(self, bodies, loads, velocities):
        """The loads [(a_m, F_N)] the beam model gets this tick (see the module doc for the two load models)."""
        if self.load_model == 'contact' or velocities is None:
            return [(a, F) for _, a, F in loads]
        moving = {w for w in bodies if w in self.ids
                  and float(np.linalg.norm(velocities[self.ids.index(w)])) > REST_SPEED_M_S}
        return rest_loads(loads, bodies, self.configuration['bench']['masses_kg'], self.configuration['bench']['seats_m'],
                          moving, {self.held, self.releasing} - {None})

    # --- the indicator carriage on its rail ------------------------------------------------------------------
    def _carriage(self):
        """(world x, x speed) of the carriage from the physics."""
        if self.carriage_view is None:
            from isaacsim.core.prims import RigidPrim
            self.carriage_view = RigidPrim(self.asset.CARRIAGE, name='beam_carriage', reset_xform_properties=False)
            self.carriage_view.initialize()
        x = float(np.asarray(self.carriage_view.get_world_poses()[0], float)[0, 0])
        v = float(np.asarray(self.carriage_view.get_linear_velocities(), float)[0, 0])
        return x, v

    def _detent(self):
        """The ball detent: within DETENT_WINDOW_M of a hole a force toward it, DETENT_STIFFNESS times the offset, at
        most DETENT_PEAK_N (applied for the next physics step)."""
        A = self.asset
        x, _ = self._carriage()
        d = min((A.hole_x(mm) - x for mm in A.HOLES_MM), key=abs)
        f = float(np.clip(A.DETENT_STIFFNESS * d, -A.DETENT_PEAK_N, A.DETENT_PEAK_N)) if abs(d) <= A.DETENT_WINDOW_M else 0.0
        self.carriage_view.apply_forces(np.array([[f, 0.0, 0.0]], dtype=np.float32), is_global=True)

    def carriage_state(self):
        """The carriage's index (m from the clamp edge) and speed (m/s)."""
        x, v = self._carriage()
        return {'x_m': x - self.asset.CLAMP_X, 'v_m_s': v}

    # --- world as the bench sees it ---------------------------------------------------------------------------
    def _world(self, positions, orientations):
        asset = self.asset
        self._table_z()
        seated_rack = {sid: None for sid in self.seats}
        tilts = {}
        for i, key in enumerate(self.ids):
            tilts[key] = _tilt_deg(orientations[i])
            if key in (self.held, self.releasing) or tilts[key] > SEAT_TILT_DEG:
                continue
            for sid, (xy, floor) in self.seats.items():
                if np.linalg.norm(positions[i][:2] - xy) <= SEAT_XY_TOL_M and abs(positions[i][2] - floor) <= SEAT_Z_TOL_M:
                    seated_rack[sid] = key
                    break
        bodies, loads = self.strip_contacts()
        for who, body in bodies.items():
            body['held'] = who in (self.held, self.releasing)
        on_beam = {k: None for k in self.configuration['bench']['seats_m']}
        for k in on_beam:
            best = max(((b['saddle_N'].get(k, 0.0), who) for who, b in bodies.items()
                        if who in self.weight_of.values() and not b['held']), default=(0.0, None))
            if best[0] >= SEATED_MIN_N:
                on_beam[k] = best[1]
        depth = self.cap_rest_z - np.asarray(self.caps.get_world_poses()[0], float)[:, 2]
        for b, d in zip(self.button_ids, depth):
            if d >= asset.C.SWITCH_DOWN_M:
                self.pressed.add(b)
            elif d <= asset.C.SWITCH_UP_M:
                self.pressed.discard(b)
        model_loads = self._loads(bodies, loads, getattr(self, '_velocities', None))
        contacts = {who: {'F_N': b['F_N'], 'x_m': b['x_m'], 'support': b['support'], 'held': b['held']}
                    for who, b in bodies.items()}
        seats = dict(seated_rack)
        seats.update({f'beam:{k}': w for k, w in on_beam.items()})
        return ({'seats': seats, 'seated': on_beam, 'held': self.held, 'pressed': set(self.pressed),
                 'loads': model_loads, 'contacts': contacts, 'carriage': self.carriage_state()}, depth, tilts)

    def seat_audit(self):
        positions, orientations = self.cuvettes.get_world_poses()
        world, _, tilts = self._world(np.asarray(positions, float), np.asarray(orientations, float))
        where = {key: None for key in self.ids}
        for sid, key in world['seats'].items():
            if key is not None:
                where[key] = sid
        strip = self.bench.strip_state() if self.bench is not None else {'load_n': 0.0, 'touching': [], 'config': []}
        carriage = self.carriage_state()
        return {'weight_seat': where, 'held': self.held, 'releasing': self.releasing,
                'tilt_deg': {k: round(t, 2) for k, t in tilts.items()}, 'strip': strip,
                'carriage': {'x_mm': round(carriage['x_m'] * 1000, 4), 'v_mm_s': round(carriage['v_m_s'] * 1000, 4)}}

    def after_step(self, timers, tick):
        out = super().after_step(timers, tick)
        if tick % self.frame_ticks == 0:            # a frame row was written: keep the exact positions with it
            rest = getattr(self, '_frame_positions', None)
            if rest is None:
                from collections import deque
                rest = self._frame_positions = deque(maxlen=30)
            rest.append(np.asarray(self.cuvettes.get_world_poses()[0], float))
        return out

    def at_rest(self):
        """At rest by displacement: over the last 30 frames no weight moved more than SETTLED_DRIFT_M and no button
        was held down (see SETTLED_DRIFT_M for why not by reported velocity)."""
        rows, rest = list(self.history)[-30:], getattr(self, '_frame_positions', None)
        if len(rows) < 30 or rest is None or len(rest) < 30 or any(r['pressed'] for r in rows):
            return False
        stack = np.stack(list(rest))
        return float(np.max(np.linalg.norm(stack - stack[-1], axis=2))) < SETTLED_DRIFT_M

    # --- stepping: the strip follows the lagged shape --------------------------------------------------------
    def step(self, target, velocity=None):
        positions = np.asarray(self.cuvettes.get_world_poses()[0], float)
        previous = getattr(self, '_last_positions', None)
        self._velocities = (positions - previous) / self.dt if previous is not None else None
        self._last_positions = positions
        self._detent()
        out = super().step(target, velocity)
        if self.bench is not None:
            self._drive_strip()
        return out

    def _shape_poses(self):
        asset, bench = self.asset, self.bench
        seg_w = [bench.deflection(f'seg:{i}') for i in range(asset.SEGMENTS)]
        seg_s = [bench.slope(f'seg:{i}') for i in range(asset.SEGMENTS)]
        saddles = [(a, bench.deflection(f'seat:{k}')) for k, a in self.configuration['bench']['seats_m'].items()]
        return asset.strip_poses(self._table_z(), seg_w, seg_s, saddles)

    def _drive_strip(self):
        poses = self._shape_poses()
        if self.strip_view is None:
            from isaacsim.core.prims import RigidPrim
            view = RigidPrim(list(self.strip_paths), name='beam_strip', reset_xform_properties=False)
            view.initialize()
            self.strip_view = view
            order = [str(p) for p in view.prim_paths]
            self.strip_order = np.array([self.strip_paths.index(p) for p in order])
        self.poses = poses
        data = np.asarray(poses, np.float32)[self.strip_order]
        self.strip_view._physics_view.set_kinematic_targets(data, np.arange(len(data), dtype=np.int32))

    def observation(self):
        frames = super().observation()
        self.last_frames = frames               # the episode seals the observation's images as its evidence
        return frames

    def _redraw(self):
        if self.bench is None:
            return
        asset = self.asset
        if self.poses is not None:
            key = np.round(np.asarray(self.poses)[:, [2, 4]], 6).tobytes()
            if key != self.poses_drawn:
                asset.set_render_poses(self.stage, self.strip_paths, self.poses, self._table_z())
                self.poses_drawn = key
                self.history_stale = True
        text = self.bench.lcd()
        if text != self.lcd_drawn:
            self.lcd_drawn = asset.show_lcd(self.stage, text, self.lcd_drawn)
            # A changed LCD resets the DLSS history at once (held RESET_HOLD_RENDERS renders), as the chem screens and
            # the torsion/hooke displays: otherwise stale digits linger in the cameras. An observation then renders
            # until settled (ChemBackend._render_for_capture: >= CAPTURE_RENDERS, SETTLE_RENDERS after a reset).
            self._reset_history()
        w = round(self.bench.plunger_w(), 6)            # the plunger rod on the strip's underside at the carriage
        if w != getattr(self, 'plunger_drawn', None):
            asset.set_render_poses(self.stage, [], [], self._table_z(), w)
            self.plunger_drawn = w

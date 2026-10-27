"""Spectrophotometry bench physics device (the spectrophotometry experiments). No SDK, IK, planner or policy.

Robot, cameras and the constrained gripper follow the torsion device (torsion_backend.py) on the same lab
stage; the apparatus is chem_apparatus. Every physics tick the cuvette poses and button caps are read: a
cuvette standing upright on a well floor is seated there, a cap pressed 4 mm down closes its switch (open again
above 2 mm), and the chemistry bench (chem_bench) advances one tick with that world. Liquid levels, colours,
reservoir levels and the spectrophotometer screen are redrawn when they change.
"""
from collections import deque
from pathlib import Path
import json
import math
import numpy as np
import collision_reflex
import grip_contact

SEAT_XY_TOL_M = 0.004
SEAT_Z_TOL_M = 0.003
SEAT_TILT_DEG = 10.0
# The real-time path tracer's denoiser (DLSS ray reconstruction) keeps a history per camera: under a still camera a
# segment switched off or a liquid level that moved stays drawn at full brightness. Its history is reset for a few
# renders after the drawing changed; a changed prim reaches the camera output about six renders later (measured:
# three reset renders, then every segment at its final level from the eighth render on). The reset acts when the
# setting turns on, so a reset asked for while one is on gets a fresh off-on edge (measured).
HISTORY_RESET = '/rtx-transient/post/dlss/forceParamReset'
RESET_HOLD_RENDERS = 3
SETTLE_RENDERS = 9               # renders after the reset before an image is taken
CAPTURE_RENDERS = 6              # renders before any image: the camera output lags the stage by a few renders


class _SupportSum:
    """Support sensor view for the SDK: summed contact force of a cuvette on every seat, base and the table."""

    def __init__(self, sensor):
        self.sensor = sensor

    def get_contact_force_matrix(self, dt):
        force = np.asarray(self.sensor.get_contact_force_matrix(dt=dt), float)
        return force.sum(axis=1, keepdims=True)

    def get_contact_force_data(self, dt):
        return self.sensor.get_contact_force_data(dt=dt)


def _tilt_deg(quat_wxyz):
    w, x, y, z = [float(v) for v in quat_wxyz]
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


class ChemBackend:
    pendulum_ids = ()  # Shared fixed-tick runtime has no pendulum gates here.
    ASSET = 'chem_apparatus'    # apparatus module; a subclass names another bench of the kit

    def __init__(self, args, app, configuration):
        import carb
        import cv2
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.physics_context import PhysicsContext
        from isaacsim.core.prims import SingleRigidPrim, RigidPrim
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot.manipulators import SingleManipulator
        from isaacsim.robot.manipulators.grippers import ParallelGripper
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, Sdf, Usd, UsdGeom
        import importlib
        asset = importlib.import_module(self.ASSET)
        import hooke_apparatus as lab
        import hooke_physics
        self.Gf, self.UsdGeom, self.Action, self.asset, self.lab, self.cv2 = Gf, UsdGeom, ArticulationAction, asset, lab, cv2
        self.dt, self.tick_index = 1 / args.physics_hz, 0
        self.frame_ticks = args.physics_hz // 30
        self.output = Path(args.output); self.private = self.output / 'private'
        self.private.mkdir(parents=True, exist_ok=True)
        self.audit_file = (self.private / 'physics.jsonl').open('w')
        self.history = deque(maxlen=30)
        self.held_trace = deque(maxlen=48)    # private: the held cuvette's last ticks (height, contact forces)
        self.held, self.releasing = None, None
        self.previous_jaw = None
        self.grasp_broken = False
        self.width_target = .08
        self.closed = False
        self.frame_index = 0
        self.bench = None
        self.shown = None
        self.render_index, self.reset_until, self.history_stale, self.resetting = 0, 0, False, False
        self.reset_edge = False
        self.settings = carb.settings.get_settings()
        self.drawn = {}                 # container id -> drawn layers
        self.bottles_drawn = None
        self.pressed = set()
        self.configuration = dict(configuration)
        source = Usd.Stage.Open(args.stage)
        for prim in source.Traverse():
            if str(prim.GetPath()).startswith(lab.ROBOT_PATH):
                if prim.IsInstanceable(): prim.SetInstanceable(False)
                visibility = prim.GetAttribute('visibility')
                if visibility: visibility.Set('inherited')
            for attribute in prim.GetAttributes():
                if attribute.GetTypeName() != Sdf.ValueTypeNames.Asset: continue
                value = attribute.Get()
                if value and str(value.path).lower().endswith(('.png', '.jpg', '.exr', '.hdr')):
                    if str(value.path).endswith('normal.png'): attribute.Set(Sdf.AssetPath('/workspace/normal.png'))
                    elif not value.resolvedPath or str(value.resolvedPath).startswith(('http:', 'https:')): attribute.Set(Sdf.AssetPath(''))
        prepared = self.private / 'initial.usda'; source.Flatten().Export(str(prepared))
        omni.usd.get_context().open_stage(str(prepared))
        for _ in range(240):
            app.update(); self.stage = omni.usd.get_context().get_stage()
            if self.stage and self.stage.GetPrimAtPath(lab.HANGER_PATH): break
        else: raise RuntimeError('lab stage failed to load')
        built = self._build(asset)
        self.paths = dict(built['objects'])             # object id -> prim path (the SDK geometry adapter reads it)
        self.button_paths = dict(built['buttons'])
        self.seat_paths = dict(built['seats'])
        self.placement = None
        if getattr(args, 'layout_candidates', None):
            import layout
            self.placement = layout.place_chem(self.stage, args.layout_candidates, self.private, asset, lab)
        asset.author_joints(self.stage, self.button_paths)
        self.seats = asset.seat_frames(self.stage, self.seat_paths)
        hooke_physics.configure_arm(self.stage, lab.ROBOT_PATH)
        lab.setup_neutral_lab_lighting(self.stage)
        carb.settings.get_settings().set_bool('/physics/updateToUsd', True)
        carb.settings.get_settings().set_bool('/physics/updateVelocitiesToUsd', True)
        self.world = World(stage_units_in_meters=1., physics_dt=self.dt, rendering_dt=1 / 30)
        self.ids = sorted(self.paths)
        self.cuvettes = self.world.scene.add(RigidPrim([self.paths[i] for i in self.ids], name='chem_cuvettes',
                                                       reset_xform_properties=False))
        self.button_ids = sorted(self.button_paths)
        self.caps = self.world.scene.add(RigidPrim([self.button_paths[b] for b in self.button_ids], name='chem_caps',
                                                   reset_xform_properties=False))
        self.contact_sensors = {i: self.world.scene.add(RigidPrim(self.paths[i], name=f'cuvette{i}_support',
            reset_xform_properties=False, track_contact_forces=True,
            contact_filter_prim_paths_expr=self._support_paths(i, built) + [lab_table_top()],
            max_contact_count=64)) for i in self.ids}
        self.support_sensors = {i: _SupportSum(sensor) for i, sensor in self.contact_sensors.items()}
        self.bodies = {i: self.world.scene.add(SingleRigidPrim(self.paths[i], name=f'cuvette{i}', reset_xform_properties=False))
                       for i in self.ids}
        self.finger_sensors = grip_contact.finger_contact_sensors(self.world.scene, RigidPrim, lab.ROBOT_PATH, self.paths, 'chem')
        self.grip = grip_contact.GripState()
        gripper = ParallelGripper(end_effector_prim_path=lab.ROBOT_PATH + '/panda_rightfinger',
            joint_prim_names=['panda_finger_joint1', 'panda_finger_joint2'], joint_opened_positions=np.array([.04, .04]),
            joint_closed_positions=np.array([.009, .009]), action_deltas=np.array([.01, .01]))
        self.robot = self.world.scene.add(SingleManipulator(prim_path=lab.ROBOT_PATH, name='chem_franka',
            end_effector_prim_path=lab.ROBOT_PATH + '/panda_rightfinger', gripper=gripper))
        self.cameras = [self.world.scene.add(Camera(prim_path=p, name='chem_cam_' + str(i), resolution=(512, 512))) for i, p in enumerate(lab.CAMERA_PATHS)]
        if getattr(self.world, '_physics_context', None) is None:
            self.world._physics_context = PhysicsContext(physics_dt=self.dt, set_defaults=False)
            self.world.set_simulation_dt(physics_dt=self.dt, rendering_dt=1 / 30)
        collision_reflex.prepare_scene(self.stage, lab.ROBOT_PATH, self.private, self.paths.values())   # colliders as drawn; contact reports
        self.world.initialize_physics(); self.world.reset()
        self.reflex = collision_reflex.Reflex(lab.ROBOT_PATH, self.dt, self.private)   # collision reflex
        standby = np.array([0., -.785, 0., -2.356, 0., 1.571, .785, .04, .04])
        self.robot.set_joints_default_state(positions=standby)
        self.robot.set_joint_positions(standby)  # Initial conditions only.
        self.robot.initialize(); self.target = standby[:7].copy()
        for camera in self.cameras:
            camera.initialize()
        for camera in self.cameras:
            camera.add_distance_to_image_plane_to_frame()
        for _ in range(60):
            self.robot.get_articulation_controller().apply_action(self.Action(joint_positions=self.target, joint_indices=np.arange(7)))
            self.world.step(render=True)
        self.cap_rest_z = np.asarray(self.caps.get_world_poses()[0], float)[:, 2].copy()
        self.video_path = self.output / 'chem_three_views_raw.mp4'
        self.writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1536, 512))
        if not self.writer.isOpened(): raise RuntimeError('chem video writer unavailable')
        for sensor in self.contact_sensors.values():
            data = sensor.get_contact_force_data(dt=self.dt)
            if data is None or np.asarray(data[4]).size == 0: raise RuntimeError('chem contact sensor unavailable')
        self.last_q = np.asarray(self.robot.get_joint_positions()[:7]).copy()
        from omni.physx import get_physx_interface
        from omni.physx.bindings._physx import SimulationEvent
        from pxr import PhysicsSchemaTools
        def on_joint_event(event):
            if event.type == int(SimulationEvent.JOINT_BREAK):
                path = str(PhysicsSchemaTools.decodeSdfPath(*event.payload['jointPath']))
                if path == '/World/HookePhysics/GraspConstraint':
                    self.grasp_broken = True
        self._subscription = get_physx_interface().get_simulation_event_stream_v2().create_subscription_to_pop(on_joint_event)
        hand = [np.asarray(self._transform(lab.ROBOT_PATH + '/' + n).ExtractTranslation()).round(4).tolist()
                for n in ('panda_leftfinger', 'panda_rightfinger')]
        (self.private / 'configuration.json').write_text(json.dumps(self.configuration | {'finger_world_m_at_start': hand},
                                                                    indent=2))
        self.stage.GetRootLayer().Export(str(self.private / 'runtime_scene.usda'))

    # --- apparatus hooks (the spectrophotometry bench; other benches of the kit override them) ------------------
    def _build(self, asset):
        """Author the apparatus; returns {'objects': id -> prim path of everything the gripper may carry, 'buttons',
        'seats'}. Instrument screens are read from the three standard cameras only."""
        import chem_render
        from chem_solver import Dye, register_dye
        bench = self.configuration['bench']
        for name, spec in (bench.get('dye_specs') or {}).items():
            register_dye(name, *spec)
        dyes = {name: Dye(name, factor) for name, factor in bench['dyes'].items()}
        built = asset.build(self.stage, {
            'bottle_mL': {n: s['volume_mL'] for n, s in bench['sources'].items()},
            'bottle_rgb': {n: chem_render.linear_rgb(s['composition_M'], dyes, path_cm=asset.bottle_inner(s['volume_mL'])[0] * 100).tolist()
                           for n, s in bench['sources'].items()},
            'bottle_fill': {n: 1.0 for n in bench['sources']},
            'labels': bench.get('labels'), 'scan_key': bool(bench.get('scan_key', False)),
            'extra_key': bench.get('extra_key'), 'dual_display': bool(bench.get('dual_wavelength', False))})
        return {'objects': built['cuvettes'], 'buttons': built['buttons'], 'seats': built['seats']}

    def _support_paths(self, key, built):
        """Prims whose contact with object `key` counts as support (the table is added by the caller)."""
        return list(built['seats'].values()) + [self.asset.ROOT + '/Rack', self.asset.ROOT + '/StationBase',
                                                self.asset.ROOT + '/Spectro']

    # --- state ------------------------------------------------------------------------------------
    def _transform(self, path):
        return self.UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))

    def public_calibration(self):
        from robot_calibration import public_robot_calibration
        return public_robot_calibration(self) | {'gripper_mechanism': grip_contact.public_gripper_mechanism()}

    def robot_state(self):
        q = np.asarray(self.robot.get_joint_positions())
        return dict(joint_positions_rad=q[:7].tolist(), gripper_width_m=float(sum(q[7:9])), gripper_holding=self.held is not None)

    # --- gripper (as the torsion device) ------------------------------------------------------------
    def _grasp_event(self, event, key):
        hand = np.asarray(self._transform(self.lab.HAND_PATH).ExtractTranslation())
        body = np.asarray(self.bodies[key].get_world_pose()[0]) if key in self.bodies else None
        row = {'event': event, 'tick': self.tick_index, 'object': key,
               'hand_world_m': hand.round(5).tolist(), 'object_world_m': None if body is None else body.round(5).tolist(),
               'width_target_m': self.width_target}
        if event == 'grasp_break':
            row['held_trace'] = list(self.held_trace)
        self.audit_file.write(json.dumps(row) + chr(10)); self.audit_file.flush()

    def set_gripper(self, width_m):
        import hooke_physics
        opens = self.grip.opens(width_m, np.asarray(self.robot.get_joint_positions())[7:9])
        self.width_target = width_m
        if opens: self.grasp_broken = False
        if opens and self.held is not None:
            self._grasp_event('grasp_release', self.held)
            hooke_physics.release_grasp_joint(self.stage, self.paths[self.held])
            self.releasing, self.held = self.held, None

    def set_magnet(self, enabled):
        raise ValueError('chemistry bench has no switchable magnetic probe')

    def _gripper_mechanism(self, positions, velocities):
        import hooke_physics
        q = np.asarray(self.robot.get_joint_positions())
        if self.releasing is not None and self.grip.break_cleared(q[7:9]):
            hooke_physics.finish_release(self.stage, self.paths[self.releasing])   # broken hold: fingers collide again
        if self.releasing is not None and self.grip.cleared(q[7:9]):
            hooke_physics.finish_release(self.stage, self.paths[self.releasing]); self.releasing = None
        fingers = [self._transform(self.lab.ROBOT_PATH + '/' + n).ExtractTranslation() for n in ('panda_leftfinger', 'panda_rightfinger')]
        jaw = np.asarray((fingers[0] + fingers[1]) / 2)
        jaw_velocity = np.zeros(3) if self.previous_jaw is None else (jaw - self.previous_jaw) / self.dt
        self.previous_jaw = jaw.copy()
        if self.grasp_broken and self.held is not None:
            self._grasp_event('grasp_break', self.held)
            hooke_physics.release_grasp_joint(self.stage, self.paths[self.held])
            self.releasing, self.held = self.held, None
            self.grip.broke = True
        # Contact-conditioned grasp (grip_contact): the fingers close onto a body, both touch it and it
        # stops them. The rigid hold stands in for friction; there is no hidden grasp point.
        pressing = self.grip.pressing(q[7:9], np.asarray(self.robot.get_joint_velocities())[7:9], self.width_target)
        if self.held is not None or self.releasing is not None or self.grasp_broken or not pressing: return
        key = grip_contact.pinched_body({i: (self.finger_sensors[i], positions[k], velocities[k]) for k, i in enumerate(self.ids)},
                                        jaw, jaw_velocity, self.dt)
        if key is None: return
        hooke_physics.attach_grasp_joint(self.stage, self.lab.HAND_PATH, self.paths[key], jaw)
        self.grip.hold(q[7:9])
        self.held = key
        self._grasp_event('grasp_attach', key)

    # --- world as the chemistry bench sees it --------------------------------------------------------
    def _world(self, positions, orientations):
        seated = {sid: None for sid in self.seats}
        tilts = {}
        for k, i in enumerate(self.ids):
            tilt = _tilt_deg(orientations[k])
            tilts[i] = tilt
            if i in (self.held, self.releasing) or tilt > SEAT_TILT_DEG:
                continue
            for sid, (xy, floor) in self.seats.items():
                if np.linalg.norm(positions[k][:2] - xy) <= SEAT_XY_TOL_M and abs(positions[k][2] - floor) <= SEAT_Z_TOL_M:
                    seated[sid] = i
                    break
        depth = self.cap_rest_z - np.asarray(self.caps.get_world_poses()[0], float)[:, 2]
        for b, d in zip(self.button_ids, depth):
            if d >= self.asset.SWITCH_DOWN_M:
                self.pressed.add(b)
            elif d <= self.asset.SWITCH_UP_M:
                self.pressed.discard(b)
        return {'seats': seated, 'held': self.held, 'tilt_deg': tilts.get(self.held, 0.0) if self.held else 0.0,
                'pressed': set(self.pressed), 'tilts': dict(tilts)}, depth, tilts

    def seat_audit(self):
        """Private: seat of each cuvette (or where it lies)."""
        positions, orientations = self.cuvettes.get_world_poses()
        world, _, tilts = self._world(np.asarray(positions, float), np.asarray(orientations, float))
        where = {i: None for i in self.ids}
        for sid, cid in world['seats'].items():
            if cid is not None:
                where[cid] = sid
        return {'seats': world['seats'], 'cuvette_seat': where, 'held': self.held,
                'tilt_deg': {i: round(t, 2) for i, t in tilts.items()}}

    # --- rendering of chemical state ----------------------------------------------------------------
    def _layers(self, container):
        import chem_render
        layers = [(v, comp) for _, v, comp in container.layers if v > 0]
        if len(layers) > self.asset.MAX_LAYERS:
            head, tail = layers[:self.asset.MAX_LAYERS - 1], layers[self.asset.MAX_LAYERS - 1:]
            volume = sum(v for v, _ in tail)
            moles = {}
            for v, comp in tail:
                for k, c in comp.items():
                    moles[k] = moles.get(k, 0.0) + c * v
            layers = head + [(volume, {k: n / volume for k, n in moles.items()})]
        return [(round(v * 0.010, 6), tuple(round(float(c), 4) for c in chem_render.linear_rgb(comp, self.bench.dyes)))
                for v, comp in layers]

    def _redraw(self):
        """Author the chemical state into the stage. A screen change resets the denoiser history at once (the video
        shows it too); a liquid change marks the history stale and the next image resets it (dosing changes the
        level every frame)."""
        import chem_render
        before = (dict(self.drawn), self.bottles_drawn, self.shown)
        for cid, container in self.bench.containers.items():
            layers = self._layers(container)
            if self.drawn.get(cid) != layers:
                self.asset.set_liquid(self.stage, cid, layers)
                self.drawn[cid] = layers
        fill = tuple(round(s.remaining_mL / s.capacity_mL, 3) for s in self.bench.sources.values())
        if fill != self.bottles_drawn:
            for name, source in self.bench.sources.items():
                rgb = chem_render.linear_rgb(source.composition_M, self.bench.dyes, path_cm=self.asset.bottle_inner(source.capacity_mL)[0] * 100)
                self.asset.set_bottle(self.stage, name, source.remaining_mL / source.capacity_mL, rgb)
            self.bottles_drawn = fill
        self.shown = self.asset.show(self.stage, self.asset.display_text(self.bench.display()), self.shown)
        if self.shown != before[2]:
            self._reset_history()
        elif (self.drawn, self.bottles_drawn) != before[:2]:
            self.history_stale = True

    def _reset_history(self):
        self.reset_until, self.history_stale = self.render_index + RESET_HOLD_RENDERS, False
        self.reset_edge = True

    def _render(self):
        reset = self.render_index < self.reset_until
        if reset and (self.reset_edge or not self.resetting):
            if self.resetting: self.settings.set_bool(HISTORY_RESET, False)
            self.settings.set_bool(HISTORY_RESET, True); self.resetting, self.reset_edge = True, False
        elif not reset and self.resetting:
            self.settings.set_bool(HISTORY_RESET, False); self.resetting = False
        self.world.render(); self.render_index += 1

    def _render_for_capture(self):
        """Render until the cameras show the stage as it is now, without the denoiser's memory of earlier drawings."""
        if self.bench is not None:
            self._redraw()
        if self.history_stale:
            self._reset_history()
        until = max(self.render_index + CAPTURE_RENDERS, self.reset_until + SETTLE_RENDERS)
        while self.render_index < until or self.resetting: self._render()

    # --- stepping ---------------------------------------------------------------------------------
    def step(self, target, velocity=None):
        if self.closed: raise RuntimeError('chem backend closed')
        previous = self.target.copy()
        if target is not None: self.target = np.asarray(target)
        controller = self.robot.get_articulation_controller()
        controller.apply_action(self.Action(joint_positions=self.target, joint_velocities=(self.target - previous) / self.dt if velocity is None else np.asarray(velocity), joint_indices=np.arange(7)))
        controller.apply_action(self.Action(joint_positions=np.full(2, self.grip.drive_width(self.width_target, self.robot) / 2), joint_indices=np.array([7, 8])))
        self.world.step(render=False); self.tick_index += 1
        positions, orientations = self.cuvettes.get_world_poses()
        positions, orientations = np.asarray(positions, float), np.asarray(orientations, float)
        velocities = np.asarray(self.cuvettes.get_linear_velocities(), float)
        if self.held is not None:
            k = self.ids.index(self.held); sensor = self.contact_sensors[self.held]
            net = np.asarray(sensor.get_net_contact_forces(dt=self.dt), float).reshape(-1)
            support = np.asarray(sensor.get_contact_force_matrix(dt=self.dt), float).reshape(-1, 3).sum(axis=0)
            hand = self._transform(self.lab.HAND_PATH).ExtractTranslation()
            self.held_trace.append({'tick': self.tick_index, 'xyz': positions[k].round(5).tolist(),
                                    'v': velocities[k].round(4).tolist(), 'hand_z': round(float(hand[2]), 5),
                                    'net_n': net.round(3).tolist(), 'support_n': support.round(3).tolist()})
        self._gripper_mechanism(positions, velocities)
        world, depth, tilts = self._world(positions, orientations)
        if self.bench is not None:
            self.bench.step(self.tick_index, world)
        q = np.asarray(self.robot.get_joint_positions()[:7])
        rates = np.abs(q - self.last_q) / self.dt; speed = float(np.max(rates))
        if speed > getattr(self, 'frame_peak', (-1., 0))[0]: self.frame_peak = (speed, int(np.argmax(rates)))
        from control_contract import reset_window, window_rates; mean_rates = window_rates(self, q, self.dt)
        if float(np.max(mean_rates)) > .9:
            self.audit_file.write(json.dumps({'event': 'arm_speed_guard', 'tick': self.tick_index, 'joint_rates_rad_s': rates.tolist(),
                'q_rad': q.tolist(), 'previous_q_rad': self.last_q.tolist(), 'target_rad': np.asarray(self.target).tolist(),
                'held': self.held, 'releasing': self.releasing}) + '\n'); self.audit_file.flush()
            from control_contract import protective_stop
            self.target = q.copy(); self.last_q = q.copy()   # protective stop: hold the measured pose
            self.pending_protective_stop=protective_stop(mean_rates); reset_window(self, q, self.dt)   # raised by the runtime after this tick
        stop = self.reflex.check(self.tick_index, collision_reflex.held_path(self))
        if stop is not None:   # collision reflex: hold the measured pose; raised by the runtime after this tick
            self.target = q.copy()
            if getattr(self, 'pending_protective_stop', None) is None:
                self.pending_protective_stop = stop
        self.last_q = q.copy()
        if self.tick_index % self.frame_ticks == 0:
            row = {'frame': self.tick_index // self.frame_ticks, 'held': self.held, 'releasing': self.releasing,
                   'seats': {k: v for k, v in world['seats'].items() if v is not None},
                   'pressed': sorted(world['pressed']),
                   'cap_depth_mm': {b: round(float(d) * 1000, 2) for b, d in zip(self.button_ids, depth) if abs(d) > 5e-4},
                   'max_speed_m_s': round(float(np.max(np.linalg.norm(velocities, axis=1))), 4),
                   'held_tilt_deg': round(tilts[self.held], 2) if self.held else None,
                   'xyz_mm': {k: [round(float(c) * 1000) for c in positions[i]] for i, k in enumerate(self.ids)},
                   'grasp_broken': self.grasp_broken, 'width_target_m': self.width_target,
                   'q_rad': [round(float(v), 5) for v in q],
                   'peak_joint_rate_rad_s': round(self.frame_peak[0], 4), 'peak_joint': self.frame_peak[1]}
            self.frame_peak = (-1., 0)
            self.history.append(row); self.audit_file.write(json.dumps(row) + '\n')
        return {}

    def after_step(self, timers, tick):
        if tick != self.tick_index: raise RuntimeError('chem physics clock mismatch')
        if tick % self.frame_ticks: return
        if self.bench is not None:
            self._redraw()
        try:
            views = self._video_frame()
        except RuntimeError:
            # The render runs Kit's update, whose renderer and annotator callbacks also run on Kit's own threads; a
            # Python error raised there surfaces here ("deque mutated during iteration" ended an episode).
            # The video frame is not physics: log it privately and render once more; a second failure ends the episode.
            import traceback
            self.audit_file.write(json.dumps({'event': 'video_frame_retry', 'tick': self.tick_index,
                                              'traceback': traceback.format_exc()}) + '\n'); self.audit_file.flush()
            views = self._video_frame()
        self.writer.write(self.cv2.cvtColor(np.hstack(views), self.cv2.COLOR_RGB2BGR)); self.frame_index += 1

    def _video_frame(self):
        self._render()
        views = [np.asarray(c.get_rgba())[..., :3].astype(np.uint8) for c in self.cameras]
        if any(v.shape != (512, 512, 3) for v in views): raise RuntimeError('chem camera frame unavailable')
        return views

    def at_rest(self):
        rows = list(self.history)
        return len(rows) >= 30 and all(r['max_speed_m_s'] < 0.005 and not r['pressed'] for r in rows[-30:])

    def observation(self):
        self._render_for_capture()
        frames = []
        for path, camera in zip(self.lab.CAMERA_PATHS, self.cameras):
            rgba = np.asarray(camera.get_rgba()); depth = np.asarray(camera.get_depth(device='cpu'))
            if rgba.shape != (512, 512, 4) or depth.shape != (512, 512): raise RuntimeError('chem RGB-D unavailable')
            valid = np.isfinite(depth) & (depth > 0) & (depth < 65.535)
            mm = np.zeros(depth.shape, dtype='<u2'); mm[valid] = np.maximum(1, np.rint(depth[valid] * 1000)).astype('<u2')
            ok, png = self.cv2.imencode('.png', self.cv2.cvtColor(rgba[..., :3], self.cv2.COLOR_RGB2BGR))
            if not ok: raise RuntimeError('chem camera encoding failed')
            frames.append(dict(camera=path.rsplit('/', 1)[-1], resolution=[512, 512], rgb_png=png.tobytes(), depth_u16le_mm=mm.tobytes(),
                intrinsics=np.asarray(camera.get_intrinsics_matrix(device='cpu')).tolist(), world_from_camera_ros=np.linalg.inv(camera.get_view_matrix_ros(device='cpu')).tolist()))
        self.last_frames = frames               # the episode seals the observation's images with its screen record
        return frames

    def close(self):
        if self.closed: return
        self.closed = True; self.writer.release(); self.audit_file.close()
        if getattr(self, 'reflex', None) is not None: self.reflex.close()   # private contact log
        if self.bench is not None:
            record = self.bench.private_record()
            (self.private / 'chem_actions.json').write_text(json.dumps(self.bench.actions, indent=1))
            (self.private / 'chem_state.json').write_text(json.dumps(record, indent=1, default=str))


def lab_table_top():
    from torsion_apparatus import TABLE_TOP
    return TABLE_TOP

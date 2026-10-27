"""Circuit panel physics device (the circuit experiments). No SDK, IK, planner or scripted policy.

Robot, cameras and gripper follow the torsion device (torsion_backend.py) on the
same lab stage; the apparatus is circuit_apparatus. Every physics tick the
carriage positions along their rails are read, each detent spring is aimed at
the nearest detent, and the electrical bench (circuit_bench) advances one tick
with the selector state those positions imply.
"""
from collections import deque
from pathlib import Path
import json
import math
import numpy as np
import collision_reflex


class CircuitBackend:
    pendulum_ids = ()  # Shared fixed-tick runtime has no pendulum gates here.

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
        import circuit_apparatus as asset
        import hooke_apparatus as lab
        import hooke_physics
        from circuit_bench import DETENT_LABELS, SELECTORS
        self.Gf, self.UsdGeom, self.Action, self.asset, self.lab, self.cv2 = Gf, UsdGeom, ArticulationAction, asset, lab, cv2
        asset.configure(configuration.get('panel'))
        self.selectors, self.detent_labels = SELECTORS, asset.PANEL['detent_labels']
        self.leds_shown = None
        self.dt, self.tick_index = 1 / args.physics_hz, 0
        self.frame_ticks = args.physics_hz // 30
        self.output = Path(args.output); self.private = self.output / 'private'
        self.private.mkdir(parents=True, exist_ok=True)
        self.audit_file = (self.private / 'physics.jsonl').open('w')
        self.history = deque(maxlen=30)
        self.width_target = .08
        self.closed = False
        self.frame_index = 0
        self.bench = None
        self.display = None              # (text, armed) currently shown on the logger screen
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
        self.initial = self.configuration['initial']
        self.paths = asset.build(self.stage, initial=self.initial)
        self.placement = None
        if getattr(args, 'layout_candidates', None):
            import layout
            self.placement = layout.place_circuit(self.stage, args.layout_candidates, self.private, asset, lab)
        # Rails are authored at the (possibly jittered) panel pose, so no world joint needs retargeting.
        asset.author_joints(self.stage, self.initial)
        self.rails = asset.rails(self.stage)
        self.detent_index = {name: self.detent_labels[name].index(self.initial[name]) for name in self.selectors}
        hooke_physics.configure_arm(self.stage, lab.ROBOT_PATH)
        lab.setup_neutral_lab_lighting(self.stage)
        carb.settings.get_settings().set_bool('/physics/updateToUsd', True)
        carb.settings.get_settings().set_bool('/physics/updateVelocitiesToUsd', True)
        self.world = World(stage_units_in_meters=1., physics_dt=self.dt, rendering_dt=1 / 30)
        self.carriages = {name: self.world.scene.add(SingleRigidPrim(path, name='carriage_' + name.lower(), reset_xform_properties=False))
                          for name, path in self.paths.items()}
        robot_parts = [lab.ROBOT_PATH + '/' + n for n in ('panda_hand', 'panda_leftfinger', 'panda_rightfinger')] + [lab.ROBOT_PATH + '/panda_link' + str(i) for i in range(8)]
        robot_parts = [p for p in robot_parts if self.stage.GetPrimAtPath(p)]
        self.robot_sensors = {name: self.world.scene.add(RigidPrim(path, name=name.lower() + '_robot_contact', reset_xform_properties=False,
            track_contact_forces=True, contact_filter_prim_paths_expr=robot_parts, max_contact_count=128)) for name, path in self.paths.items()}
        gripper = ParallelGripper(end_effector_prim_path=lab.ROBOT_PATH + '/panda_rightfinger',
            joint_prim_names=['panda_finger_joint1', 'panda_finger_joint2'], joint_opened_positions=np.array([.04, .04]),
            joint_closed_positions=np.array([.009, .009]), action_deltas=np.array([.01, .01]))
        self.robot = self.world.scene.add(SingleManipulator(prim_path=lab.ROBOT_PATH, name='circuit_franka',
            end_effector_prim_path=lab.ROBOT_PATH + '/panda_rightfinger', gripper=gripper))
        self.cameras = [self.world.scene.add(Camera(prim_path=p, name='circuit_cam_' + str(i), resolution=(512, 512))) for i, p in enumerate(lab.CAMERA_PATHS)]
        if getattr(self.world, '_physics_context', None) is None:
            self.world._physics_context = PhysicsContext(physics_dt=self.dt, set_defaults=False)
            self.world.set_simulation_dt(physics_dt=self.dt, rendering_dt=1 / 30)
        collision_reflex.prepare_scene(self.stage, lab.ROBOT_PATH, self.private, ())   # colliders as drawn; contact reports
        self.world.initialize_physics(); self.world.reset()
        self.reflex = collision_reflex.Reflex(lab.ROBOT_PATH, self.dt, self.private)   # collision reflex
        standby = np.array([0., -.785, 0., -2.356, 0., 1.571, .785, .04, .04])
        self.robot.set_joints_default_state(positions=standby)
        self.robot.set_joint_positions(standby)  # Initial conditions only.
        self.robot.initialize(); self.target = standby[:7].copy()
        for camera in self.cameras:
            camera.initialize(); camera.add_distance_to_image_plane_to_frame()
        for _ in range(60):
            self.robot.get_articulation_controller().apply_action(self.Action(joint_positions=self.target, joint_indices=np.arange(7)))
            self.world.step(render=True)
        self.video_path = self.output / 'circuit_three_views_raw.mp4'
        self.writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1536, 512))
        if not self.writer.isOpened(): raise RuntimeError('circuit video writer unavailable')
        for sensor in self.robot_sensors.values():
            data = sensor.get_contact_force_data(dt=self.dt)
            if data is None or np.asarray(data[4]).size == 0: raise RuntimeError('circuit contact sensor unavailable')
        self.last_q = np.asarray(self.robot.get_joint_positions()[:7]).copy()
        (self.private / 'configuration.json').write_text(json.dumps(self.configuration, indent=2))
        self.stage.GetRootLayer().Export(str(self.private / 'runtime_scene.usda'))

    # --- state ------------------------------------------------------------------------------------
    def _transform(self, path):
        return self.UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))

    def public_calibration(self):
        from robot_calibration import public_robot_calibration
        return public_robot_calibration(self) | {'gripper_mechanism': {
            'type': 'parallel_gripper', 'description': 'Plain parallel fingers; the circuit bench has nothing to grasp.'}}

    def robot_state(self):
        q = np.asarray(self.robot.get_joint_positions())
        return dict(joint_positions_rad=q[:7].tolist(), gripper_width_m=float(sum(q[7:9])), gripper_holding=False)

    def positions_m(self):
        """Trusted carriage travel from detent 0 along each rail (private)."""
        out = {}
        for name, body in self.carriages.items():
            origin, direction = self.rails[name]
            position = np.asarray(body.get_world_pose()[0], float)[:2]
            out[name] = float(np.dot(position - origin, direction))
        return out

    def set_gripper(self, width_m):
        self.width_target = width_m

    def set_magnet(self, enabled):
        raise ValueError('circuit bench has no switchable magnetic probe')

    def _touching(self, sensor):
        count = int(np.asarray(sensor.get_contact_force_data(dt=self.dt)[4]).sum())
        force = float(np.linalg.norm(np.asarray(sensor.get_contact_force_matrix(dt=self.dt))))
        if count < 0 or not np.isfinite(force): raise RuntimeError('circuit contact buffer invalid')   # a full buffer counts as contact
        return count > 0 or force > 1e-5

    def _detents(self, positions):
        """Aim each detent spring at the detent nearest to its carriage."""
        from circuit_apparatus import DETENT_SPACING_M, detent_offset_m, set_detent_target
        for name, p in positions.items():
            index = min(len(self.detent_labels[name]) - 1, max(0, int(round(p / DETENT_SPACING_M))))
            if index != self.detent_index[name]:
                self.detent_index[name] = index
                set_detent_target(self.stage, name, detent_offset_m(index))

    # --- stepping ---------------------------------------------------------------------------------
    def step(self, target, velocity=None):
        if self.closed: raise RuntimeError('circuit backend closed')
        previous = self.target.copy()
        if target is not None: self.target = np.asarray(target)
        controller = self.robot.get_articulation_controller()
        controller.apply_action(self.Action(joint_positions=self.target, joint_velocities=(self.target - previous) / self.dt if velocity is None else np.asarray(velocity), joint_indices=np.arange(7)))
        controller.apply_action(self.Action(joint_positions=np.full(2, self.width_target / 2), joint_indices=np.array([7, 8])))
        self.world.step(render=False); self.tick_index += 1
        positions = self.positions_m()
        self._detents(positions)
        if self.bench is not None:
            self.bench.step(self.tick_index, positions)
        q = np.asarray(self.robot.get_joint_positions()[:7])
        rates = np.abs(q - self.last_q) / self.dt; speed = float(np.max(rates))
        if speed > getattr(self, 'frame_peak', (-1., 0))[0]: self.frame_peak = (speed, int(np.argmax(rates)))
        from control_contract import reset_window, window_rates; mean_rates = window_rates(self, q, self.dt)
        if float(np.max(mean_rates)) > .9:
            self.audit_file.write(json.dumps({'event': 'arm_speed_guard', 'tick': self.tick_index, 'joint_rates_rad_s': rates.tolist(),
                'q_rad': q.tolist(), 'previous_q_rad': self.last_q.tolist(), 'target_rad': np.asarray(self.target).tolist()}) + '\n')
            self.audit_file.flush()
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
            contacts = {name: self._touching(sensor) for name, sensor in self.robot_sensors.items()}
            row = {'frame': self.tick_index // self.frame_ticks, 'positions_mm': {k: round(v * 1000, 3) for k, v in positions.items()},
                   'switch': list(self.bench.switch) if self.bench is not None else None,
                   'vc_V': round(self.bench.vc_v, 6) if self.bench is not None else None,
                   'robot_contacts': contacts, 'width_target_m': self.width_target, 'q_rad': [round(float(v), 5) for v in q],
                   'peak_joint_rate_rad_s': round(self.frame_peak[0], 4), 'peak_joint': self.frame_peak[1]}
            self.frame_peak = (-1., 0)
            self.history.append(row); self.audit_file.write(json.dumps(row) + '\n')
        return {}

    def after_step(self, timers, tick):
        if tick != self.tick_index: raise RuntimeError('circuit physics clock mismatch')
        if tick % self.frame_ticks: return
        if self.bench is not None and tick % (6 * self.frame_ticks) == 0:      # screen refresh 5 Hz
            ch1, _ = self.bench.model.channels(self.bench.switch, self.bench.vc_v)
            self.display = self.asset.show_reading(self.stage, ch1, self.bench.logger.active is not None, self.display)
            if (self.asset.PANEL['board'] or {}).get('leds'):
                lit = {self.bench.switch[0]} if self.bench.model.current(self.bench.switch, self.bench.vc_v) > 5e-4 else set()
                self.leds_shown = self.asset.show_leds(self.stage, lit, self.leds_shown)
        self.world.render()
        views = [np.asarray(c.get_rgba())[..., :3].astype(np.uint8) for c in self.cameras]
        if any(v.shape != (512, 512, 3) for v in views): raise RuntimeError('circuit camera frame unavailable')
        self.writer.write(self.cv2.cvtColor(np.hstack(views), self.cv2.COLOR_RGB2BGR)); self.frame_index += 1

    def at_rest(self):
        rows = list(self.history)
        if len(rows) < 30: return False
        return all(max(abs(r['positions_mm'][k] - rows[-1]['positions_mm'][k]) for r in rows) < 0.05 for k in self.selectors)

    def observation(self):
        for _ in range(4): self.world.render()
        frames = []
        for path, camera in zip(self.lab.CAMERA_PATHS, self.cameras):
            rgba = np.asarray(camera.get_rgba()); depth = np.asarray(camera.get_depth(device='cpu'))
            if rgba.shape != (512, 512, 4) or depth.shape != (512, 512): raise RuntimeError('circuit RGB-D unavailable')
            valid = np.isfinite(depth) & (depth > 0) & (depth < 65.535)
            mm = np.zeros(depth.shape, dtype='<u2'); mm[valid] = np.maximum(1, np.rint(depth[valid] * 1000)).astype('<u2')
            ok, png = self.cv2.imencode('.png', self.cv2.cvtColor(rgba[..., :3], self.cv2.COLOR_RGB2BGR))
            if not ok: raise RuntimeError('circuit camera encoding failed')
            frames.append(dict(camera=path.rsplit('/', 1)[-1], resolution=[512, 512], rgb_png=png.tobytes(), depth_u16le_mm=mm.tobytes(),
                intrinsics=np.asarray(camera.get_intrinsics_matrix(device='cpu')).tolist(), world_from_camera_ros=np.linalg.inv(camera.get_view_matrix_ros(device='cpu')).tolist()))
        return frames

    def close(self):
        if self.closed: return
        self.closed = True; self.writer.release(); self.audit_file.close()
        if getattr(self, 'reflex', None) is not None: self.reflex.close()   # private contact log
        if self.bench is not None:
            (self.private / 'circuit_actions.json').write_text(json.dumps(self.bench.actions, indent=1))
            (self.private / 'circuit_state.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in self.bench.state_log))

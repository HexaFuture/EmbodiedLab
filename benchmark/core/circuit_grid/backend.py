"""Circuit board physics device (resistor_blackbox; later the other circuit experiments). No SDK, IK, planner or
scripted policy.

Robot, cameras and the contact-conditioned gripper follow the other benches on the same lab stage; the apparatus is
circuit_grid.asset. Every physics tick: the gripper mechanism (grasp and release of a block), the push buttons (the
supply's ON/OFF button toggles the output; the logger keys go to the logger panel), the switch rocker (it snaps to
the side it was pushed past), the placement of every block (seated on the board, in a tray seat, or loose), and then
the electrical bench (circuit_grid.board) with that state. The displays are updated at the frame rate and before
every observation.
"""
from collections import deque
from pathlib import Path
import json
import math

import numpy as np
import collision_reflex

import grip_contact

from . import asset as A
from . import layout as L
from .board import classify

# The real-time path tracer's denoiser (DLSS ray reconstruction) keeps a history per camera: under a still camera a
# display segment switched off stays drawn (a meter stepping 4866 -> 4867 -> 4865 showed 4868). The history is reset
# for a few renders after any display changed; a changed prim reaches the camera output about six renders later
# (chem_backend measured: three reset renders, then every segment at its final level from the eighth render on).
# The reset acts when the setting turns on: displays that change every few frames (logger 0.2 s, meters 0.4 s) kept it
# on and the history drew both states (a meter stepping 1.0969 <-> 1.0970 showed 1.0988), so every display change
# gives it a fresh off-on edge (measured: union 0.7 -> 0.0).
HISTORY_RESET = '/rtx-transient/post/dlss/forceParamReset'
RESET_HOLD_RENDERS = 3
SETTLE_RENDERS = 9               # renders after a reset before an observation's images are taken
CAPTURE_RENDERS = 6              # renders before any observation: the camera output lags the stage by a few renders


class _SupportSum:
    """Support sensor view for the SDK's support-controlled release: summed contact force of a block on the board,
    the tray and the table."""

    def __init__(self, sensor):
        self.sensor = sensor

    def get_contact_force_matrix(self, dt):
        return np.asarray(self.sensor.get_contact_force_matrix(dt=dt), float).sum(axis=1, keepdims=True)

    def get_contact_force_data(self, dt):
        return self.sensor.get_contact_force_data(dt=dt)


class GridBackend:
    pendulum_ids = ()  # Shared fixed-tick runtime has no pendulum gates here.

    def __init__(self, args, app, configuration):
        import carb
        import cv2
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.physics_context import PhysicsContext
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot.manipulators import SingleManipulator
        from isaacsim.robot.manipulators.grippers import ParallelGripper
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, Sdf, Usd, UsdGeom
        import hooke_apparatus as lab
        import hooke_physics
        self.Gf = Gf
        self.UsdGeom, self.Action, self.lab, self.cv2, self.physical = UsdGeom, ArticulationAction, lab, cv2, hooke_physics
        self.dt, self.tick_index = 1 / args.physics_hz, 0
        self.physics_hz = args.physics_hz
        self.frame_ticks = args.physics_hz // 30
        self.output = Path(args.output)
        self.private = self.output / 'private'
        self.private.mkdir(parents=True, exist_ok=True)
        self.audit_file = (self.private / 'physics.jsonl').open('w')
        self.history = deque(maxlen=30)
        self.width_target = .08
        self.closed = False
        self.frame_index = 0
        self.bench = None
        self.horizon_ticks = None
        self.held, self.releasing = None, None
        self.previous_jaw = None
        self.grasp_broken = False
        self.supply_on = False
        self.switch_closed = False
        self.configuration = dict(configuration)
        source = Usd.Stage.Open(args.stage)
        for prim in source.Traverse():
            if str(prim.GetPath()).startswith(lab.ROBOT_PATH):
                if prim.IsInstanceable():
                    prim.SetInstanceable(False)
                visibility = prim.GetAttribute('visibility')
                if visibility:
                    visibility.Set('inherited')
            for attribute in prim.GetAttributes():
                if attribute.GetTypeName() != Sdf.ValueTypeNames.Asset:
                    continue
                value = attribute.Get()
                if value and str(value.path).lower().endswith(('.png', '.jpg', '.exr', '.hdr')):
                    if str(value.path).endswith('normal.png'):
                        attribute.Set(Sdf.AssetPath('/workspace/normal.png'))
                    elif not value.resolvedPath or str(value.resolvedPath).startswith(('http:', 'https:')):
                        attribute.Set(Sdf.AssetPath(''))
        prepared = self.private / 'initial.usda'
        source.Flatten().Export(str(prepared))
        omni.usd.get_context().open_stage(str(prepared))
        for _ in range(240):
            app.update()
            self.stage = omni.usd.get_context().get_stage()
            if self.stage and self.stage.GetPrimAtPath(lab.HANGER_PATH):
                break
        else:
            raise RuntimeError('lab stage failed to load')
        built = A.build(self.stage)
        self.paths = built['blocks']
        self.block_ids = list(self.paths)
        self.key_names = list(built['keys'])
        hooke_physics.configure_arm(self.stage, lab.ROBOT_PATH)
        lab.setup_neutral_lab_lighting(self.stage)
        carb.settings.get_settings().set_bool('/physics/updateToUsd', True)
        carb.settings.get_settings().set_bool('/physics/updateVelocitiesToUsd', True)
        self.world = World(stage_units_in_meters=1., physics_dt=self.dt, rendering_dt=1 / 30)
        self.block_view = self.world.scene.add(RigidPrim([self.paths[b] for b in self.block_ids], name='grid_blocks',
                                                         reset_xform_properties=False))
        self.finger_sensors = grip_contact.finger_contact_sensors(self.world.scene, RigidPrim, lab.ROBOT_PATH,
                                                                  self.paths, 'grid')
        supports = [A.ROOT + '/Board/Plate', A.ROOT + '/Board/Brass', A.ROOT + '/Tray/Body', A.TABLE_TOP]
        self.support_sensors = {b: _SupportSum(self.world.scene.add(RigidPrim(
            self.paths[b], name='grid_' + b + '_support', reset_xform_properties=False, track_contact_forces=True,
            contact_filter_prim_paths_expr=supports, max_contact_count=64))) for b in self.block_ids}
        buttons = [built['button']] if built.get('button') else []      # a bench without the DC supply (battery_emf)
        self.cap_names = ['supply'] * len(buttons) + ['key_' + k for k in self.key_names]
        self.cap_view = self.world.scene.add(RigidPrim(buttons + [built['keys'][k] for k in self.key_names],
                                                       name='grid_caps', reset_xform_properties=False))
        self.rocker_view = self.world.scene.add(RigidPrim([built['rocker']], name='grid_rocker',
                                                          reset_xform_properties=False))
        gripper = ParallelGripper(end_effector_prim_path=lab.ROBOT_PATH + '/panda_rightfinger',
                                  joint_prim_names=['panda_finger_joint1', 'panda_finger_joint2'],
                                  joint_opened_positions=np.array([.04, .04]),
                                  joint_closed_positions=np.array([.009, .009]), action_deltas=np.array([.01, .01]))
        self.robot = self.world.scene.add(SingleManipulator(prim_path=lab.ROBOT_PATH, name='grid_franka',
                                                            end_effector_prim_path=lab.ROBOT_PATH + '/panda_rightfinger',
                                                            gripper=gripper))
        self.cameras = [self.world.scene.add(Camera(prim_path=p, name='grid_cam_' + str(i), resolution=(512, 512)))
                        for i, p in enumerate(lab.CAMERA_PATHS)]
        if getattr(self.world, '_physics_context', None) is None:
            self.world._physics_context = PhysicsContext(physics_dt=self.dt, set_defaults=False)
            self.world.set_simulation_dt(physics_dt=self.dt, rendering_dt=1 / 30)
        collision_reflex.prepare_scene(self.stage, lab.ROBOT_PATH, self.private, self.paths.values())   # colliders as drawn; contact reports
        self.world.initialize_physics()
        self.world.reset()
        self.reflex = collision_reflex.Reflex(lab.ROBOT_PATH, self.dt, self.private)   # collision reflex
        standby = np.array([0., -.785, 0., -2.356, 0., 1.571, .785, .04, .04])
        self.robot.set_joints_default_state(positions=standby)
        self.robot.set_joint_positions(standby)  # Initial conditions only.
        self.robot.initialize()
        self.target = standby[:7].copy()
        for camera in self.cameras:
            camera.initialize()
            camera.add_distance_to_image_plane_to_frame()
        for _ in range(60):
            self.robot.get_articulation_controller().apply_action(
                self.Action(joint_positions=self.target, joint_indices=np.arange(7)))
            self.world.step(render=True)
        self.video_path = self.output / 'circuit_grid_three_views_raw.mp4'
        self.writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1536, 512))
        if not self.writer.isOpened():
            raise RuntimeError('circuit board video writer unavailable')
        for sensor in self.finger_sensors.values():
            data = sensor.get_contact_force_data(dt=self.dt)
            if data is None or np.asarray(data[4]).size == 0:
                raise RuntimeError('circuit board contact sensor unavailable')
        self.cap_rest_z = np.asarray(self.cap_view.get_world_poses()[0], float)[:, 2].copy()
        self.cap_down = [False] * len(self.cap_names)
        self.displays = A.Displays(self.stage)
        self.settings = carb.settings.get_settings()
        self.render_index, self.reset_until, self.resetting, self.display_key = 0, 0, False, None
        self.reset_edge = False
        self.grip = grip_contact.GripState()
        self.last_q = np.asarray(self.robot.get_joint_positions()[:7]).copy()
        from omni.physx import get_physx_interface
        from omni.physx.bindings._physx import SimulationEvent
        from pxr import PhysicsSchemaTools

        def on_joint_event(event):
            if event.type == int(SimulationEvent.JOINT_BREAK):
                path = str(PhysicsSchemaTools.decodeSdfPath(*event.payload['jointPath']))
                if path == '/World/HookePhysics/GraspConstraint':
                    self.grasp_broken = True
        self._subscription = get_physx_interface().get_simulation_event_stream_v2().create_subscription_to_pop(
            on_joint_event)
        (self.private / 'configuration.json').write_text(json.dumps(self.configuration, indent=2))
        self.stage.GetRootLayer().Export(str(self.private / 'runtime_scene.usda'))

    # --- state ------------------------------------------------------------------------------------------------------
    def _transform(self, path):
        return self.UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))

    def public_calibration(self):
        from robot_calibration import public_robot_calibration
        return public_robot_calibration(self) | {'gripper_mechanism': grip_contact.public_gripper_mechanism()}

    def robot_state(self):
        q = np.asarray(self.robot.get_joint_positions())
        return dict(joint_positions_rad=q[:7].tolist(), gripper_width_m=float(sum(q[7:9])), gripper_holding=self.held is not None)

    def set_gripper(self, width_m):
        opens = self.grip.opens(width_m, np.asarray(self.robot.get_joint_positions())[7:9])
        self.width_target = width_m
        if opens:
            self.grasp_broken = False
        if opens and self.held is not None:
            self.physical.release_grasp_joint(self.stage, self.paths[self.held])
            self.releasing, self.held = self.held, None

    def set_magnet(self, enabled):
        raise ValueError('the circuit board has no switchable magnetic probe')

    def poses(self):
        """(positions (N, 3), quaternions wxyz (N, 4)) of the blocks, in block_ids order."""
        p, q = self.block_view.get_world_poses()
        return np.asarray(p, float), np.asarray(q, float)

    def placement(self):
        positions, quats = self.poses()
        busy = {self.held, self.releasing}
        return {b: classify(b, positions[k], quats[k], held=b in busy) for k, b in enumerate(self.block_ids)}

    # --- mechanisms -------------------------------------------------------------------------------------------------
    def _gripper_mechanism(self):
        q = np.asarray(self.robot.get_joint_positions())
        if self.releasing is not None and self.grip.break_cleared(q[7:9]):
            self.physical.finish_release(self.stage, self.paths[self.releasing])   # broken hold: fingers collide again
        if self.releasing is not None and self.grip.cleared(q[7:9]):
            self.physical.finish_release(self.stage, self.paths[self.releasing])
            self.releasing = None
        fingers = [self._transform(self.lab.ROBOT_PATH + '/' + n).ExtractTranslation()
                   for n in ('panda_leftfinger', 'panda_rightfinger')]
        jaw = np.asarray((fingers[0] + fingers[1]) / 2)
        jaw_velocity = np.zeros(3) if self.previous_jaw is None else (jaw - self.previous_jaw) / self.dt
        self.previous_jaw = jaw.copy()
        if self.grasp_broken and self.held is not None:
            self.physical.release_grasp_joint(self.stage, self.paths[self.held])
            self.releasing, self.held = self.held, None
            self.grip.broke = True
        # Contact-conditioned grasp (grip_contact): the fingers close onto a block, both touch it and it stops them.
        pressing = self.grip.pressing(q[7:9], np.asarray(self.robot.get_joint_velocities())[7:9], self.width_target)
        if self.held is not None or self.releasing is not None or self.grasp_broken or not pressing:
            return
        positions, _ = self.poses()
        velocities = np.asarray(self.block_view.get_linear_velocities(), float)
        key = grip_contact.pinched_body({b: (self.finger_sensors[b], positions[k], velocities[k])
                                         for k, b in enumerate(self.block_ids)}, jaw, jaw_velocity, self.dt)
        if key is None:
            return
        self.physical.attach_grasp_joint(self.stage, self.lab.HAND_PATH, self.paths[key], jaw)
        self.grip.hold(q[7:9])
        self.held = key

    def _buttons(self):
        """Push buttons: each acts once per press (down >= SWITCH_DOWN_M, up again <= SWITCH_UP_M)."""
        depth = self.cap_rest_z - np.asarray(self.cap_view.get_world_poses()[0], float)[:, 2]
        for k, (name, d) in enumerate(zip(self.cap_names, depth)):
            if not self.cap_down[k] and d >= L.SWITCH_DOWN_M:
                self.cap_down[k] = True
                self._press(name)
            elif self.cap_down[k] and d <= L.SWITCH_UP_M:
                self.cap_down[k] = False

    def _press(self, name):
        event = {'event': 'button', 'tick': self.tick_index, 'button': name}
        if name == 'supply':
            self.supply_on = not self.supply_on
            event['supply_on'] = self.supply_on
        elif self.bench is not None:
            remaining = (self.horizon_ticks or 0) - self.tick_index
            event['effect'] = self.bench.press_key(self.tick_index, name[4:], remaining)
        else:
            event['effect'] = 'before_episode'
        self.audit_file.write(json.dumps(event) + '\n')

    def rocker_angle_deg(self):
        _, quats = self.poses()
        block = quats[self.block_ids.index('SW')]
        rocker = np.asarray(self.rocker_view.get_world_poses()[1], float)[0]
        from .board import _rotation
        rel = _rotation(block).T @ _rotation(rocker)
        return math.degrees(math.atan2(-rel[2, 0], rel[0, 0]))

    def _rocker(self):
        angle = self.rocker_angle_deg()
        if not self.switch_closed and angle > 2.0:
            self.switch_closed = True
            A.set_rocker_target(self.stage, True)
        elif self.switch_closed and angle < -2.0:
            self.switch_closed = False
            A.set_rocker_target(self.stage, False)

    # --- stepping ---------------------------------------------------------------------------------------------------
    def step(self, target, velocity=None):
        if self.closed:
            raise RuntimeError('circuit board backend closed')
        previous = self.target.copy()
        if target is not None:
            self.target = np.asarray(target)
        controller = self.robot.get_articulation_controller()
        controller.apply_action(self.Action(joint_positions=self.target,
                                            joint_velocities=(self.target - previous) / self.dt if velocity is None
                                            else np.asarray(velocity), joint_indices=np.arange(7)))
        controller.apply_action(self.Action(joint_positions=np.full(2, self.grip.drive_width(self.width_target, self.robot) / 2),
                                            joint_indices=np.array([7, 8])))
        self.world.step(render=False)
        self.tick_index += 1
        self._gripper_mechanism()
        self._buttons()
        self._rocker()
        if self.bench is not None:
            self.bench.update(self.tick_index, self.placement(), self.held or self.releasing, self.supply_on,
                              self.switch_closed)
            self.bench.step(self.tick_index, (self.horizon_ticks or 0) - self.tick_index)
        q = np.asarray(self.robot.get_joint_positions()[:7])
        rates = np.abs(q - self.last_q) / self.dt
        speed = float(np.max(rates))
        if speed > getattr(self, 'frame_peak', (-1., 0))[0]:
            self.frame_peak = (speed, int(np.argmax(rates)))
        from control_contract import reset_window, window_rates; mean_rates = window_rates(self, q, self.dt)
        if float(np.max(mean_rates)) > .9:
            self.audit_file.write(json.dumps({'event': 'arm_speed_guard', 'tick': self.tick_index,
                                              'joint_rates_rad_s': rates.tolist(), 'q_rad': q.tolist(),
                                              'previous_q_rad': self.last_q.tolist(),
                                              'target_rad': np.asarray(self.target).tolist(), 'held': self.held}) + '\n')
            self.audit_file.flush()
            from control_contract import protective_stop
            self.target = q.copy()
            self.last_q = q.copy()   # protective stop: hold the measured pose
            self.pending_protective_stop = protective_stop(mean_rates); reset_window(self, q, self.dt)   # raised by the runtime after this tick
        stop = self.reflex.check(self.tick_index, collision_reflex.held_path(self))
        if stop is not None:   # collision reflex: hold the measured pose; raised by the runtime after this tick
            self.target = q.copy()
            if getattr(self, 'pending_protective_stop', None) is None:
                self.pending_protective_stop = stop
        self.last_q = q.copy()
        if self.tick_index % self.frame_ticks == 0:
            positions, quats = self.poses()
            row = {'frame': self.tick_index // self.frame_ticks, 'q_rad': [round(float(v), 5) for v in q],
                   'width_target_m': self.width_target, 'held': self.held, 'releasing': self.releasing,
                   'supply_on': self.supply_on, 'switch_closed': self.switch_closed,
                   'peak_joint_rate_rad_s': round(self.frame_peak[0], 4), 'peak_joint': self.frame_peak[1],
                   'xyz_mm': {b: [round(float(c) * 1000, 1) for c in positions[k]] for k, b in enumerate(self.block_ids)}}
            self.frame_peak = (-1., 0)
            self.history.append(row)
            self.audit_file.write(json.dumps(row) + '\n')
        return {}

    def sync_displays(self):
        if self.bench is None:
            self.displays.lamp('ON', self.supply_on)
            key = json.dumps([self.supply_on])
        else:
            shown = {block: meter.shown() for block, meter in self.bench.meters.items()}
            for block, value in shown.items():
                self.displays.meter(block, value)
            logger = self.bench.logger.display()
            self.displays.logger(logger)
            cc = self.supply_on and self.bench.analysis['result'].mode == 'CC'
            self.displays.lamp('ON', self.supply_on)
            self.displays.lamp('CC', cc)
            key = json.dumps([shown, logger, self.supply_on, cc], sort_keys=True, default=str)
        if key != self.display_key:
            self.display_key = key
            self.reset_until = self.render_index + RESET_HOLD_RENDERS
            self.reset_edge = True

    def _render(self):
        reset = self.render_index < self.reset_until
        if reset and (self.reset_edge or not self.resetting):
            if self.resetting:
                self.settings.set_bool(HISTORY_RESET, False)
            self.settings.set_bool(HISTORY_RESET, True)
            self.resetting, self.reset_edge = True, False
        elif not reset and self.resetting:
            self.settings.set_bool(HISTORY_RESET, False)
            self.resetting = False
        self.world.render()
        self.render_index += 1

    def after_step(self, timers, tick):
        if tick != self.tick_index:
            raise RuntimeError('circuit board physics clock mismatch')
        if tick % self.frame_ticks:
            return
        self.sync_displays()
        self._render()
        views = [np.asarray(c.get_rgba())[..., :3].astype(np.uint8) for c in self.cameras]
        if any(v.shape != (512, 512, 3) for v in views):
            raise RuntimeError('circuit board camera frame unavailable')
        self.writer.write(self.cv2.cvtColor(np.hstack(views), self.cv2.COLOR_RGB2BGR))
        self.frame_index += 1

    def at_rest(self):
        rows = list(self.history)
        if len(rows) < 10:
            return False
        last = rows[-1]['xyz_mm']
        return all(max(abs(r['xyz_mm'][b][i] - last[b][i]) for r in rows[-10:] for i in range(3)) < 0.2 for b in last)

    def observation(self):
        self.sync_displays()
        until = max(self.render_index + CAPTURE_RENDERS, self.reset_until + SETTLE_RENDERS)
        while self.render_index < until or self.resetting:
            self._render()
        frames = []
        for path, camera in zip(self.lab.CAMERA_PATHS, self.cameras):
            rgba = np.asarray(camera.get_rgba())
            depth = np.asarray(camera.get_depth(device='cpu'))
            if rgba.shape != (512, 512, 4) or depth.shape != (512, 512):
                raise RuntimeError('circuit board RGB-D unavailable')
            valid = np.isfinite(depth) & (depth > 0) & (depth < 65.535)
            mm = np.zeros(depth.shape, dtype='<u2')
            mm[valid] = np.maximum(1, np.rint(depth[valid] * 1000)).astype('<u2')
            ok, png = self.cv2.imencode('.png', self.cv2.cvtColor(rgba[..., :3], self.cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError('circuit board camera encoding failed')
            frames.append(dict(camera=path.rsplit('/', 1)[-1], resolution=[512, 512], rgb_png=png.tobytes(),
                               depth_u16le_mm=mm.tobytes(),
                               intrinsics=np.asarray(camera.get_intrinsics_matrix(device='cpu')).tolist(),
                               world_from_camera_ros=np.linalg.inv(camera.get_view_matrix_ros(device='cpu')).tolist()))
        # Evidence images of this state: the three standard views (camera name -> PNG), nothing else.
        self.view_pngs = {f['camera']: f['rgb_png'] for f in frames}
        return frames

    def close(self):
        if self.closed:
            return
        self.closed = True
        if getattr(self, 'reflex', None) is not None: self.reflex.close()   # private contact log
        self.writer.release()
        self.audit_file.close()
        if self.bench is not None:
            (self.private / 'grid_state.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in self.bench.state_log))
            (self.private / 'grid_actions.json').write_text(json.dumps(self.bench.actions, indent=1))
            (self.private / 'grid_displays.jsonl').write_text(''.join(json.dumps(e) + '\n'
                                                                      for e in self.bench.display_log))

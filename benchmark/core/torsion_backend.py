"""Torsion-pendulum physics device (torsion_inertia; torsion_centroid via centroid_backend). No SDK, IK, planner or
scripted policy.

Robot, cameras, constrained gripper and arm guard follow the Hooke device
(hooke_backend.py) on the same lab stage; the apparatus is torsion_apparatus.

Hand-operated controls: the photogate's recorder panel and the eddy-current damper lever. Every
physics tick reads the key caps (a key acts once per press: down >= SWITCH_DOWN_M, up again <= SWITCH_UP_M; the
shared RecorderPanel with start_on_rec: a valid REC starts a photogate record of the keyed length at once) and the lever
carriage (damper_lever.Lever): the turntable's viscous damping is the spring's own plus the brake damping times the
magnet's overlap with the copper vane. Objects start in the storage box; its seat occupancy is logged.
"""
from collections import deque
from pathlib import Path
import json
import math
import numpy as np
import collision_reflex
import grip_contact

# Displays: RTX's DLSS keeps a history of static pixels, so redrawn digits stayed stale in the cameras. As
# chem/circuit_grid: hold the history reset for a few renders after a display change, and take an observation only
# after the reset has settled.
HISTORY_RESET = '/rtx-transient/post/dlss/forceParamReset'
DISPLAY_RESET_RENDERS = 3
DISPLAY_SETTLE_RENDERS = 9


class _SupportSum:
    """Support sensor view for the SDK: summed contact force of a body on the turntable and the table."""

    def __init__(self, sensor):
        self.sensor = sensor

    def get_contact_force_matrix(self, dt):
        force = np.asarray(self.sensor.get_contact_force_matrix(dt=dt), float)
        return force.sum(axis=1, keepdims=True)

    def get_contact_force_data(self, dt):
        return self.sensor.get_contact_force_data(dt=dt)


class TorsionBackend:
    pendulum_ids = ()  # Shared fixed-tick runtime has no pendulum gates here.

    def apparatus_module(self):
        import torsion_apparatus
        return torsion_apparatus

    def build_apparatus(self):
        return self.asset.build(self.stage, turntable_inertia_kgm2=self.dynamics['turntable_inertia_kgm2'],
                                unknown_inertia_kgm2=self.dynamics['unknown_inertia_kgm2'],
                                mark_rgb=self.dynamics['mark_rgb'], storage_box=self.asset.BOX_SEATS)

    def __init__(self, args, app, dynamics):
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
        import hooke_apparatus as lab
        import hooke_physics
        asset = self.apparatus_module()
        self.Gf, self.UsdGeom, self.Action, self.asset, self.lab = Gf, UsdGeom, ArticulationAction, asset, lab
        self.cv2 = cv2
        self.dt, self.tick_index = 1 / args.physics_hz, 0
        self.frame_ticks = args.physics_hz // 30
        self.output = Path(args.output); self.private = self.output / 'private'
        self.private.mkdir(parents=True, exist_ok=True)
        self.audit_file = (self.private / 'physics.jsonl').open('w')
        self.history = deque(maxlen=30)
        self.held, self.releasing = None, None
        self.previous_jaw = None
        self.grasp_broken = False
        self.width_target = .08
        self.closed = False
        self.frame_index = 0
        self.brake = False             # turntable damped: the preparation settling, or the damper lever off OFF
        self.settling = False
        self.photogate = None
        # Bench controls (set up by _init_controls after the warm-up). Runtimes that subclass this backend without its
        # panel, lever or storage box (friction, projectile) may skip this __init__: every hook below reads these
        # attributes with getattr and does nothing when they are absent or None.
        self.panel = None
        self.lever = None
        self.seat_state = None
        self.seat_log = None
        self.shown = None
        self.dynamics = dict(dynamics)
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
        self.paths = self.build_apparatus()
        self.masses = {key: float(value[2]) for key, value in asset.OBJECTS.items()}
        self.placement = None
        if getattr(args, 'layout_candidates', None):
            import layout
            self.placement = layout.place_torsion(self.stage, args.layout_candidates, self.private, asset, lab)
        self.brake_damping = self.dynamics['brake_damping_nms_per_rad']
        self.free_damping = self.dynamics['damping_nms_per_rad']
        # Prepared at rest with the brake damping (preparation only, before the episode clock starts); the runtime
        # releases it before the first action. From then on the damper lever alone adds damping.
        asset.author_joint(self.stage, self.dynamics['kappa_nm_per_rad'], self.brake_damping)
        self.joint_damping = self.brake_damping
        self.brake = self.settling = True
        hooke_physics.configure_arm(self.stage, lab.ROBOT_PATH)
        lab.setup_neutral_lab_lighting(self.stage)
        carb.settings.get_settings().set_bool('/physics/updateToUsd', True)
        carb.settings.get_settings().set_bool('/physics/updateVelocitiesToUsd', True)
        self.world = World(stage_units_in_meters=1., physics_dt=self.dt, rendering_dt=1 / 30)
        self.turntable = self.world.scene.add(SingleRigidPrim(asset.TURNTABLE, name='turntable', reset_xform_properties=False))
        self.bodies = {key: self.world.scene.add(SingleRigidPrim(path, name=key, reset_xform_properties=False)) for key, path in self.paths.items()}
        # Filter order: [platter collider, table top, storage box]; the SDK view sums them, load bookkeeping reads the
        # platter column. The filter names the collider prim itself: the nested platter collider is not matched by its
        # body path.
        self.contact_sensors = {key: self.world.scene.add(RigidPrim(path, name=key + '_support', reset_xform_properties=False,
            track_contact_forces=True, contact_filter_prim_paths_expr=[asset.TURNTABLE + '/Platter/Collider', asset.TABLE_TOP,
                                                                       asset.BOX_PATH + '/Body'], max_contact_count=64))
            for key, path in self.paths.items()}
        self.support_sensors = {key: _SupportSum(sensor) for key, sensor in self.contact_sensors.items()}
        robot_parts = [lab.ROBOT_PATH + '/' + n for n in ('panda_hand', 'panda_leftfinger', 'panda_rightfinger')] + [lab.ROBOT_PATH + '/panda_link' + str(i) for i in range(8)]
        robot_parts = [p for p in robot_parts if self.stage.GetPrimAtPath(p)]
        self.finger_sensors = grip_contact.finger_contact_sensors(self.world.scene, RigidPrim, lab.ROBOT_PATH, self.paths, 'torsion')
        self.grip = grip_contact.GripState()
        self.robot_sensors = {key: self.world.scene.add(RigidPrim(path, name=key + '_robot_contact', reset_xform_properties=False,
            track_contact_forces=True, contact_filter_prim_paths_expr=robot_parts, max_contact_count=128)) for key, path in self.paths.items()}
        self.turntable_contact = self.world.scene.add(RigidPrim(asset.TURNTABLE, name='turntable_robot_contact', reset_xform_properties=False,
            track_contact_forces=True, contact_filter_prim_paths_expr=robot_parts, max_contact_count=128))
        # Hand-operated controls: the recorder keypad caps and the damper lever carriage (positions only).
        from coupled_oscillator.controls import KEYS
        self.cap_names = list(KEYS)
        self.cap_view = self.world.scene.add(RigidPrim([asset.PANEL_PATH + '/Keypad/Key_' + k for k in KEYS],
                                                       name='torsion_keys', reset_xform_properties=False))
        self.lever_view = self.world.scene.add(RigidPrim(asset.LEVER_PATH + '/Carriage', name='torsion_lever',
                                                         reset_xform_properties=False))
        # Storage box seat markers (floor centres; they move with the layout jitter).
        self.seat_paths = {f'{i + 1}': asset.seat_path(i) for i in range(len(asset.BOX_SEATS))}
        gripper = ParallelGripper(end_effector_prim_path=lab.ROBOT_PATH + '/panda_rightfinger',
            joint_prim_names=['panda_finger_joint1', 'panda_finger_joint2'], joint_opened_positions=np.array([.04, .04]),
            joint_closed_positions=np.array([.009, .009]), action_deltas=np.array([.01, .01]))
        self.robot = self.world.scene.add(SingleManipulator(prim_path=lab.ROBOT_PATH, name='torsion_franka',
            end_effector_prim_path=lab.ROBOT_PATH + '/panda_rightfinger', gripper=gripper))
        self.cameras = [self.world.scene.add(Camera(prim_path=p, name='torsion_cam_' + str(i), resolution=(512, 512))) for i, p in enumerate(lab.CAMERA_PATHS)]
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
            camera.initialize(); camera.add_distance_to_image_plane_to_frame()
        for _ in range(60):
            self.robot.get_articulation_controller().apply_action(self.Action(joint_positions=self.target, joint_indices=np.arange(7)))
            self.world.step(render=True)
        self._init_controls()
        # Rest orientation of the turntable: the joint frame's zero (flag centred in the beam).
        self.rest_yaw = self._yaw()
        self.unwrapped = 0.0
        self.last_yaw = self.rest_yaw
        self.axis_xy = np.asarray(self.turntable.get_world_pose()[0][:2], float)
        self.video_path = self.output / 'torsion_three_views_raw.mp4'
        self.writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1536, 512))
        if not self.writer.isOpened(): raise RuntimeError('torsion video writer unavailable')
        for sensor in list(self.contact_sensors.values()) + list(self.robot_sensors.values()) + [self.turntable_contact]:
            data = sensor.get_contact_force_data(dt=self.dt)
            if data is None or np.asarray(data[4]).size == 0: raise RuntimeError('torsion contact sensor unavailable')
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
        (self.private / 'configuration.json').write_text(json.dumps(self.dynamics, indent=2))
        self.stage.GetRootLayer().Export(str(self.private / 'runtime_scene.usda'))

    # --- state ------------------------------------------------------------------------------------
    def _yaw(self):
        w, x, y, z = [float(v) for v in self.turntable.get_world_pose()[1]]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def angle(self):
        """Turntable angle relative to rest (rad), unwrapped; updated every physics tick."""
        return self.unwrapped

    def _transform(self, path):
        return self.UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))

    def public_calibration(self):
        from robot_calibration import public_robot_calibration
        return public_robot_calibration(self) | {'gripper_mechanism': grip_contact.public_gripper_mechanism()}

    def robot_state(self):
        q = np.asarray(self.robot.get_joint_positions())
        return dict(joint_positions_rad=q[:7].tolist(), gripper_width_m=float(sum(q[7:9])), gripper_holding=self.held is not None)

    # --- gripper ----------------------------------------------------------------------------------
    def _grasp_event(self, event, key):
        """Private diagnostics of the constrained gripper: attach, release and break with poses."""
        hand = np.asarray(self._transform(self.lab.HAND_PATH).ExtractTranslation())
        body = np.asarray(self.bodies[key].get_world_pose()[0]) if key in self.bodies else None
        self.audit_file.write(json.dumps({'event': event, 'tick': self.tick_index, 'object': key,
            'hand_world_m': hand.round(5).tolist(), 'object_world_m': None if body is None else body.round(5).tolist(),
            'width_target_m': self.width_target}) + chr(10)); self.audit_file.flush()

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
        raise ValueError('torsion apparatus has no switchable magnetic probe')

    def _gripper_mechanism(self):
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
        key = grip_contact.pinched_body({k: (self.finger_sensors[k], np.asarray(b.get_world_pose()[0]), np.asarray(b.get_linear_velocity()))
                                         for k, b in self.bodies.items()}, jaw, jaw_velocity, self.dt)
        if key is None: return
        joint_path = hooke_physics.attach_grasp_joint(self.stage, self.lab.HAND_PATH, self.paths[key], jaw)
        joint = self.stage.GetPrimAtPath(joint_path)
        if not joint or not joint.IsActive():
            raise RuntimeError('grasp constraint is not active')
        self.grip.hold(q[7:9])
        self.held = key
        self._grasp_event('grasp_attach', key)

    # --- apparatus controls -----------------------------------------------------------------------
    def _init_controls(self):
        """After the warm-up: key cap rest heights, the lever's detent 0 where its carriage stands, the panel."""
        from coupled_oscillator.controls import RecorderPanel
        from damper_lever import Lever, lever_info
        self.cap_rest_z = np.asarray(self.cap_view.get_world_poses()[0], float)[:, 2].copy()
        self.cap_down = [False] * len(self.cap_names)
        carriage = np.asarray(self.lever_view.get_world_poses()[0], float)[0]
        info = lever_info(self.asset.LEVER_PATH, **dict(self.asset.LEVER, detent0=(float(carriage[0]), float(carriage[1]))))
        self.lever = Lever(info, self.stage)
        self.panel = RecorderPanel(start_on_rec=True, record_range=self.asset.RECORD_S_RANGE)
        self.panel_state = self.panel.state

    def _event(self, kind, fields):
        if self.audit_file.closed: return
        self.audit_file.write(json.dumps(dict(event=kind, tick=self.tick_index, **fields)) + '\n'); self.audit_file.flush()

    def _apply_damping(self):
        """Joint damping: the brake damping while settling, else the spring's own plus the lever's share."""
        lever = getattr(self, 'lever', None)
        if not hasattr(self, 'free_damping') or (lever is None and not getattr(self, 'settling', False)): return
        fraction = 1.0 if getattr(self, 'settling', False) else lever.fraction
        value = self.free_damping + fraction * (self.brake_damping - self.free_damping)
        if abs(value - self.joint_damping) > 1e-9:
            self.asset.set_joint_damping(self.stage, value)
            self.joint_damping = value
        self.brake = fraction > 0

    def release_settling_damping(self):
        """End of the apparatus preparation: the damper lever (at OFF) alone sets the extra damping from now on."""
        self.settling = False
        self._apply_damping()

    def _controls(self):
        """Key caps (each acts once per press) and the damper lever, every physics tick."""
        if getattr(self, 'lever', None) is None or getattr(self, 'panel', None) is None: return
        tick = self.tick_index
        depth = self.cap_rest_z - np.asarray(self.cap_view.get_world_poses()[0], float)[:, 2]
        from coupled_oscillator.controls_asset import SWITCH_DOWN_M, SWITCH_UP_M
        for k, (name, d) in enumerate(zip(self.cap_names, depth)):
            if not self.cap_down[k] and d >= SWITCH_DOWN_M:
                self.cap_down[k] = True
                self._key(name)
            elif self.cap_down[k] and d <= SWITCH_UP_M:
                self.cap_down[k] = False
        carriage = np.asarray(self.lever_view.get_world_poses()[0], float)[0]
        for change in self.lever.update(tick, carriage[:2]):
            self._event('damper', {k: v for k, v in change.items() if k != 'tick'})   # the event row has its own tick
        self._apply_damping()

    def _key(self, key):
        if getattr(self, 'panel', None) is None: return
        effect = self.panel.press(key)
        self._event('key', dict(key=key, effect=effect, entry=self.panel.entry, ready_s=self.panel.ready_s,
                                angle_rad=round(self.angle(), 6), damper=round(getattr(getattr(self, 'lever', None), 'fraction', 0.0), 3),
                                held=self.held))
        if effect == 'start':
            gate = self.photogate
            if gate is None or gate.active is not None:
                raise RuntimeError('recorder started while the photogate is unavailable or recording')
            eid = gate.arm(self.tick_index, float(self.panel.ready_s), None, self.loads())
            self.panel.start(int(eid.rsplit(':', 1)[-1]))
            self._event('record_start', dict(evidence_id=eid, length_s=self.panel.recording_s))
        self._panel_changed()

    def _panel_changed(self):
        if getattr(self, 'panel', None) is None: return
        if self.panel.state != self.panel_state:
            self._event('recorder', dict(old=self.panel_state, new=self.panel.state, number=self.panel.number,
                                         recording_s=self.panel.recording_s))
            self.panel_state = self.panel.state

    def _recorder_tick(self, active_before):
        gate = getattr(self, 'photogate', None)
        if gate is None or getattr(self, 'panel', None) is None: return
        if active_before is not None and gate.active is None:
            self.panel.finish()
            record = gate.records[active_before['evidence_id']]
            self._event('record_end', dict(evidence_id=active_before['evidence_id'], transits=len(record['rows'])))
        elif gate.active is not None:
            self.panel.tick((self.tick_index - gate.active['start_tick']) * self.dt)
        self._panel_changed()

    # --- storage box ------------------------------------------------------------------------------
    def seat_occupancy(self):
        """{seat number: id of the object seated in it, or None} by the public rule: within SEAT_DXY_M horizontally
        and SEAT_DZ_M vertically of an object standing on the seat floor, tilted at most SEAT_TILT_DEG, not held."""
        asset = self.asset
        out = {}
        for seat, path in self.seat_paths.items():
            home = np.asarray(self._transform(path).ExtractTranslation(), float)
            out[seat] = None
            for key, body in self.bodies.items():
                p, q = body.get_world_pose()
                w, x, y, z = np.asarray(q, float) / max(float(np.linalg.norm(q)), 1e-12)
                tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1., 1.))))
                if key in (self.held, self.releasing) or tilt > asset.SEAT_TILT_DEG:
                    continue
                xyz = np.asarray(p, float)
                if np.hypot(*(xyz[:2] - home[:2])) <= asset.SEAT_DXY_M and abs(xyz[2] - home[2]) <= asset.SEAT_DZ_M:
                    out[seat] = key
                    break
        return out

    def start_seat_log(self):
        """At the episode clock's zero: the occupancy, then seat_in / seat_out events at 30 Hz (as the Hooke box)."""
        self.seat_state = self.seat_occupancy()
        self.seat_log = {'thresholds': {'dxy_m': self.asset.SEAT_DXY_M, 'abs_dz_m': self.asset.SEAT_DZ_M,
                                        'tilt_deg': self.asset.SEAT_TILT_DEG},
                         'start': {'tick': self.tick_index, 'seats': dict(self.seat_state)}, 'events': [], 'end': None}
        self._event('seat_occupancy', dict(phase='start', seats=dict(self.seat_state)))

    def _seat_frame(self):
        if getattr(self, 'seat_state', None) is None: return
        try:
            now = self.seat_occupancy()
            for seat, before in self.seat_state.items():
                if now[seat] == before: continue
                for kind, key in (('seat_out', before), ('seat_in', now[seat])):
                    if key is not None:
                        event = {'event': kind, 'tick': self.tick_index, 'seat': seat, 'object': key}
                        self.seat_log['events'].append(event)
                        self._event(kind, dict(seat=seat, object=key))
            self.seat_state = now
        except Exception as exc:
            self.seat_log.setdefault('errors', []).append(f'{self.tick_index}: {type(exc).__name__}: {exc}')

    def end_seat_log(self):
        if getattr(self, 'seat_state', None) is None or getattr(self, 'seat_log', None) is None: return None
        if self.seat_log['end'] is None:
            self.seat_log['end'] = {'tick': self.tick_index, 'seats': self.seat_occupancy()}
            self._event('seat_occupancy', dict(phase='end', seats=self.seat_log['end']['seats']))
        return json.loads(json.dumps(self.seat_log))

    def bench_state(self, reason=None):
        """State of the bench for the tidy-up gate (sealed as part of the terminal state; logged at seal and stop)."""
        lever, panel, gate = getattr(self, 'lever', None), getattr(self, 'panel', None), getattr(self, 'photogate', None)
        state = dict(storage_box_seats=self.seat_occupancy() if getattr(self, 'seat_paths', None) else None,
                     damper=lever.record() if lever is not None else None,
                     recorder=None if panel is None else dict(panel.record(), recording=panel.recording,
                                                              photogate_recording=gate is not None and gate.active is not None),
                     held=getattr(self, 'held', None))
        if reason is not None: self._event('bench_state', dict(reason=reason, **state))
        return state

    # --- loads ------------------------------------------------------------------------------------
    def object_audit(self, key):
        """Trusted pose of an object relative to the turntable axis (private)."""
        position, orientation = self.bodies[key].get_world_pose()
        w, x, y, z = [float(v) for v in orientation]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
        com = np.asarray(position, float)[:2]
        return dict(eccentricity_mm=round(float(np.linalg.norm(com - self.axis_xy)) * 1000, 3),
                    tilt_deg=round(tilt, 3))

    def loads(self):
        """Objects resting on the turntable in each of the last 10 frames, not held or touched, with their audit."""
        rows = list(self.history)[-10:]
        if len(rows) < 10: return {}
        out = {}
        for key in self.bodies:
            if key in (self.held, self.releasing): continue
            weight = self.masses[key] * 9.81
            if all(r['turntable_force'][key] > .5 * weight and not r['robot_contacts'][key] for r in rows):
                out[key] = self.object_audit(key)
        return out

    def frame_extra(self):
        """Process-score log fields of the frame row (read-only): robot contact with the turntable this tick."""
        sensor = self.turntable_contact
        count = int(np.asarray(sensor.get_contact_force_data(dt=self.dt)[4]).sum())
        force = float(np.linalg.norm(np.asarray(sensor.get_contact_force_matrix(dt=self.dt))))
        valid = 0 <= count < 128 and np.isfinite(force)
        return {'turntable_contact': (count > 0 or force > 1e-5) if valid else None}

    def _touching(self, sensor):
        count = int(np.asarray(sensor.get_contact_force_data(dt=self.dt)[4]).sum())
        force = float(np.linalg.norm(np.asarray(sensor.get_contact_force_matrix(dt=self.dt))))
        if count < 0 or not np.isfinite(force): raise RuntimeError('torsion contact buffer invalid')   # a full buffer counts as contact
        return count > 0 or force > 1e-5

    def _gate_tick(self):
        gate = self.photogate
        if gate.active is not None:
            loaded = gate.active['loads_at_arm']
            contact = self._touching(self.turntable_contact) or any(self._touching(s) for s in self.robot_sensors.values())
            # An object arriving on or leaving the turntable changes the load while the gate records.
            current = sorted(k for k in self.bodies if abs(float(self.contact_sensors[k].get_contact_force_matrix(dt=self.dt)[0, 0, 2])) > .02)
            # Compared with what touched the turntable at the first recorded tick (any contact), so an object only
            # resting lightly on it at arming does not flag every transit; loads_at_arm stays the evaluator's record.
            start = gate.active.setdefault('contacts_at_arm', current)
            changing = current != start or self.held is not None and (self.held in loaded or self.held in start)
            conditions = self._transit_conditions(set(current) | set(start) | set(loaded or ()), changing)
        else:
            contact = changing = False
            conditions = None
        gate.sample(self.tick_index, self.angle(), self.asset.FLAG_HALF_ANGLE_RAD, contact=bool(contact), brake=bool(self.brake),
                    load_changing=bool(changing), loads=self.loads, conditions=conditions)

    def _transit_conditions(self, on_turntable, changing):
        """Private record only (six levels): the conditions of this tick by source, kept per transit by the
        photogate (`transit_conditions`): robot contact with the turntable, with an object on it (pressing on it now,
        at the first recorded tick or at REC) or with an object elsewhere; the damper; a load change. Never raises (an
        invalid contact buffer is recorded as 'contact_unknown'); no effect on the bench or on transit_status."""
        try:
            def flag(sensor):
                count = int(np.asarray(sensor.get_contact_force_data(dt=self.dt)[4]).sum())
                force = float(np.linalg.norm(np.asarray(sensor.get_contact_force_matrix(dt=self.dt))))
                return None if count < 0 or not np.isfinite(force) else (count > 0 or force > 1e-5)
            out = set()
            touched = flag(self.turntable_contact)
            if touched is not False:
                out.add('contact_turntable' if touched else 'contact_unknown')
            for key, sensor in self.robot_sensors.items():
                touched = flag(sensor)
                if touched is None:
                    out.add('contact_unknown')
                elif touched:
                    out.add('contact_on_turntable_object' if key in on_turntable else 'contact_off_turntable_object')
            if self.brake:
                out.add('brake')
            if changing:
                out.add('load_changing')
            return out
        except Exception:
            return {'contact_unknown'}

    def _gate_and_controls(self):
        """Photogate sample of this tick, then the controls (a REC keyed now starts a record at this tick)."""
        gate = getattr(self, 'photogate', None)
        active = gate.active if gate is not None else None
        if gate is not None: self._gate_tick()
        if gate is not None: self._recorder_tick(active)
        self._controls()

    # --- stepping ---------------------------------------------------------------------------------
    def step(self, target, velocity=None):
        if self.closed: raise RuntimeError('torsion backend closed')
        previous = self.target.copy()
        if target is not None: self.target = np.asarray(target)
        controller = self.robot.get_articulation_controller()
        controller.apply_action(self.Action(joint_positions=self.target, joint_velocities=(self.target - previous) / self.dt if velocity is None else np.asarray(velocity), joint_indices=np.arange(7)))
        controller.apply_action(self.Action(joint_positions=np.full(2, self.grip.drive_width(self.width_target, self.robot) / 2), joint_indices=np.array([7, 8])))
        self.world.step(render=False); self.tick_index += 1
        yaw = self._yaw()
        self.unwrapped += math.remainder(yaw - self.last_yaw, 2 * math.pi)
        self.last_yaw = yaw
        self._gripper_mechanism()
        self._gate_and_controls()
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
            matrices = {key: np.asarray(sensor.get_contact_force_matrix(dt=self.dt), float) for key, sensor in self.contact_sensors.items()}
            counts = {key: int(np.asarray(sensor.get_contact_force_data(dt=self.dt)[4]).sum()) for key, sensor in self.robot_sensors.items()}
            if any(count < 0 for count in counts.values()): raise RuntimeError('torsion contact buffer invalid')
            row = {'frame': self.tick_index // self.frame_ticks, 'angle_rad': round(self.angle(), 6),
                   'turntable_force': {k: abs(float(m[0, 0, 2])) for k, m in matrices.items()},
                   'table_force': {k: abs(float(m[0, 1, 2])) + abs(float(m[0, 2, 2])) for k, m in matrices.items()},
                   'released': self.held is None and self.releasing is None and not any(counts.values()),
                   'robot_contacts': counts, 'brake': self.brake, 'damper': round(getattr(getattr(self, 'lever', None), 'fraction', 0.0), 3),
                   'recording': self.photogate is not None and self.photogate.active is not None,
                   'held': self.held, 'releasing': self.releasing,
                   'grasp_broken': self.grasp_broken, 'width_target_m': self.width_target, 'q_rad': [round(float(v), 5) for v in q],
                   'peak_joint_rate_rad_s': round(self.frame_peak[0], 4), 'peak_joint': self.frame_peak[1]}
            self.frame_peak = (-1., 0)
            self.history.append(row)
            try:
                row.update(self.frame_extra())
            except Exception as exc:     # a log field never ends the episode
                row['log_error'] = f'{type(exc).__name__}: {exc}'
            self.audit_file.write(json.dumps(row) + '\n')
            self._seat_frame()
        return {}

    def _display_render(self):
        """world.render with the DLSS history reset held for a few renders after a display change."""
        import carb
        settings = carb.settings.get_settings()
        index = getattr(self, 'render_index', 0)
        reset = index < getattr(self, 'display_reset_until', 0); resetting = getattr(self, 'display_resetting', False)
        if reset and (getattr(self, 'display_reset_edge', False) or not resetting):
            if resetting: settings.set_bool(HISTORY_RESET, False)
            settings.set_bool(HISTORY_RESET, True); self.display_resetting, self.display_reset_edge = True, False
        elif not reset and resetting:
            settings.set_bool(HISTORY_RESET, False); self.display_resetting = False
        self.world.render(); self.render_index = index + 1

    def _display_changed(self):
        self.display_reset_until, self.display_reset_edge = getattr(self, 'render_index', 0) + DISPLAY_RESET_RENDERS, True

    def _render_for_capture(self):
        """Render until the cameras show the displays as they are now (at least 4 renders, as before)."""
        until = max(getattr(self, 'render_index', 0) + 4, getattr(self, 'display_reset_until', 0) + DISPLAY_SETTLE_RENDERS)
        while getattr(self, 'render_index', 0) < until or getattr(self, 'display_resetting', False): self._display_render()

    def after_step(self, timers, tick):
        if tick != self.tick_index: raise RuntimeError('torsion physics clock mismatch')
        if tick % self.frame_ticks: return
        self._update_display()
        self._display_render()
        views = [np.asarray(c.get_rgba())[..., :3].astype(np.uint8) for c in self.cameras]
        if any(v.shape != (512, 512, 3) for v in views): raise RuntimeError('torsion camera frame unavailable')
        self.writer.write(self.cv2.cvtColor(np.hstack(views), self.cv2.COLOR_RGB2BGR)); self.frame_index += 1

    def at_rest(self):
        rows = list(self.history)
        if len(rows) < 30: return False
        angles = np.array([r['angle_rad'] for r in rows])
        return bool(np.ptp(angles) < 2e-4)

    def _update_display(self):
        if getattr(self, 'panel', None) is None: return
        shown = self.panel.display()
        if shown != getattr(self, 'shown', None):
            from coupled_oscillator.controls_asset import update_recorder_display
            update_recorder_display(self.stage, self.asset.PANEL_PATH, shown)
            self.shown = shown
            self._display_changed()

    def observation(self):
        self._update_display()
        self._render_for_capture()
        frames = []
        for path, camera in zip(self.lab.CAMERA_PATHS, self.cameras):
            rgba = np.asarray(camera.get_rgba()); depth = np.asarray(camera.get_depth(device='cpu'))
            if rgba.shape != (512, 512, 4) or depth.shape != (512, 512): raise RuntimeError('torsion RGB-D unavailable')
            valid = np.isfinite(depth) & (depth > 0) & (depth < 65.535)
            mm = np.zeros(depth.shape, dtype='<u2'); mm[valid] = np.maximum(1, np.rint(depth[valid] * 1000)).astype('<u2')
            ok, png = self.cv2.imencode('.png', self.cv2.cvtColor(rgba[..., :3], self.cv2.COLOR_RGB2BGR))
            if not ok: raise RuntimeError('torsion camera encoding failed')
            frames.append(dict(camera=path.rsplit('/', 1)[-1], resolution=[512, 512], rgb_png=png.tobytes(), depth_u16le_mm=mm.tobytes(),
                intrinsics=np.asarray(camera.get_intrinsics_matrix(device='cpu')).tolist(), world_from_camera_ros=np.linalg.inv(camera.get_view_matrix_ros(device='cpu')).tolist()))
        return frames

    def close(self):
        if self.closed: return
        self.closed = True; self.writer.release(); self.audit_file.close()
        if getattr(self, 'reflex', None) is not None: self.reflex.close()   # private contact log

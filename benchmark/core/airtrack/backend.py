"""airtrack_collision physical backend: robot, free guided gliders and the hand-operated bench (no policy, no IK).

Each physics tick reads the recorder key caps, the release lever carriage, the launcher plunger and both gliders and
applies their effects (mechanism.py): key presses on the recorder panel, the lever's trip (launcher fired, catches
lifted, the READY recorder started) and return (catches lowered, pawl re-armed), the pawl latching a notch, the
catches capturing or letting go of a glider. The gate recorder (recorder.py) times the passes of a running run. Every
change is an event row of private/physics.jsonl; a compact state row follows every video frame (30 Hz).
"""
from pathlib import Path
import json
import numpy as np
import collision_reflex

# Displays: RTX's DLSS keeps a history of static pixels, so redrawn digits/states stayed stale in the cameras. As
# chem/circuit_grid: hold the history reset for a few renders after a display change, and take an observation only
# after the reset has settled.
HISTORY_RESET = '/rtx-transient/post/dlss/forceParamReset'
DISPLAY_RESET_RENDERS = 3
DISPLAY_SETTLE_RENDERS = 9

from . import mechanism as M
from .asset import ROOT, CENTERS, HEIGHT, TRACK_ORIGIN_WORLD, PLUNGER_REST_X, CATCH_DOWN_Z, CATCH_LIFT_M, CATCH_Y, \
    CATCH_PIN_DX, SCREEN_X, lever_info

ROBOT = '/World/Franka'
HAND = ROBOT + '/panda_hand'
FINGERS = [ROBOT + '/' + n for n in ('panda_leftfinger', 'panda_rightfinger')]
CAMERAS = [ROBOT + '/RoboCasaCameras/robot0_agentview_left', ROBOT + '/RoboCasaCameras/robot0_agentview_right',
           HAND + '/robot0_eye_in_hand']
GLIDERS = ('A', 'B')
MOVED_M = 0.001          # a glider displaced this far by the robot (one continuous contact): robot_moved


class AirtrackBackend:
    pendulum_ids = ()

    def __init__(self, args, app):
        import carb, cv2, omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.physics_context import PhysicsContext
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot.manipulators import SingleManipulator
        from isaacsim.robot.manipulators.grippers import ParallelGripper
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
        from hooke_physics import configure_arm
        from coupled_oscillator import controls_asset as CA
        from coupled_oscillator.controls import KEYS, RecorderPanel
        from damper_lever import Lever
        from .asset import build_apparatus
        from .screens import Screens
        self.CA = CA
        self.Gf, self.UsdGeom, self.Action, self.UsdPhysics, self.cv2 = Gf, UsdGeom, ArticulationAction, UsdPhysics, cv2
        self.output = Path(args.output)
        self.private = self.output / 'private'
        self.private.mkdir(parents=True, exist_ok=True)
        params = args.airtrack_parameters
        self.masses = (.2, float(params['mass_B_kg']))
        self.dt = 1 / args.physics_hz
        self.hz = args.physics_hz
        self.tick_index = 0
        self.frame_ticks = args.physics_hz // 30
        self.frame_index = 0
        self.closed = False
        self.width_target = .08
        self.recorder = None                     # GateRecorder, attached by the episode
        self.audit = None
        self.grip_ticks = [0, 0, 0]
        self.pending_protective_stop = None
        # Hand-operated state at the start: both gliders caught at their marks, lever at HOLD, launcher uncocked.
        self.caught = [True, True]
        self.pawl = M.Pawl()
        self.release_tick = None                 # last trip of the lever (dashpot window)
        self.drive_mode = None
        self.panel = RecorderPanel(record_range=(5, 60))
        self.panel_state = self.panel.state
        self.lever_index = 0
        self.robot_moved = False
        self.contact_start = [None, None]        # glider X when a continuous robot contact began
        self.cocked_logged = None
        self.snapshot = None
        self.tongue_z = None

        asset_path = self.private / 'apparatus.usda'
        stage = Usd.Stage.CreateNew(str(asset_path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, 'Z')
        build_apparatus(stage, params, seed=args.seed)
        stage.SetDefaultPrim(stage.GetPrimAtPath(ROOT))
        stage.GetRootLayer().Save()
        source = Usd.Stage.Open(args.stage)
        for path in ('/World/HookeApparatus', '/World/HookePhysics'):
            prim = source.GetPrimAtPath(path)
            if prim:
                prim.SetActive(False)
        riser = source.GetPrimAtPath('/World/RobotRiser')
        if not riser:
            raise RuntimeError('original robot mounting base missing from source stage')
        riser.SetActive(True)
        for prim in source.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                prim.SetActive(False)
        source.DefinePrim(ROOT, 'Xform').GetReferences().AddReference(str(asset_path))
        for prim in list(Usd.PrimRange(source.GetPrimAtPath(ROBOT))):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        configure_arm(source, ROBOT)
        material = UsdShade.Material(source.GetPrimAtPath(ROOT + '/Materials/ContactMaterial'))
        for prim in Usd.PrimRange(source.GetPrimAtPath(ROBOT)):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                api.CreateSolverPositionIterationCountAttr(64)
                api.CreateSolverVelocityIterationCountAttr(32)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api = UsdPhysics.RigidBodyAPI(prim)
                api.CreateVelocityAttr(Gf.Vec3f(0))
                api.CreateAngularVelocityAttr(Gf.Vec3f(0))
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                api.CreateContactOffsetAttr(.0005)
                api.CreateRestOffsetAttr(0.)
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
            if prim.IsA(UsdPhysics.PrismaticJoint) and 'finger' in prim.GetName():
                for api in list(prim.GetAppliedSchemas()):
                    if api.startswith('PhysxMimicJointAPI:'):
                        prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, api.split(':', 1)[1])
                drive = UsdPhysics.DriveAPI.Apply(prim, 'linear')
                drive.CreateStiffnessAttr(400.)
                drive.CreateDampingAttr(20.)
                drive.CreateMaxForceAttr(3.)
        for prim in source.Traverse():       # resolve textures without modifying the frozen input stage
            for attr in prim.GetAttributes():
                if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                    continue
                v = attr.Get()
                if v and str(v.path).lower().endswith(('.png', '.jpg', '.exr', '.hdr')):
                    if str(v.path).endswith('normal.png'):
                        attr.Set(Sdf.AssetPath('/workspace/normal.png'))
                    elif not v.resolvedPath or str(v.resolvedPath).startswith(('http:', 'https:')):
                        attr.Set(Sdf.AssetPath(''))
        scene_path = self.private / 'initial.usda'
        source.Flatten().Export(str(scene_path))
        omni.usd.get_context().open_stage(str(scene_path))
        for _ in range(30):
            app.update()
        self.stage = omni.usd.get_context().get_stage()
        carb.settings.get_settings().set_bool('/physics/updateToUsd', True)
        carb.settings.get_settings().set_bool('/physics/updateVelocitiesToUsd', True)
        self.world = World(stage_units_in_meters=1., physics_dt=self.dt, rendering_dt=1 / 30)
        self.views = [self.world.scene.add(RigidPrim(ROOT + f'/Slider{i}', name=f'airtrack_slider{i}',
                      reset_xform_properties=False, track_contact_forces=True,
                      contact_filter_prim_paths_expr=FINGERS, max_contact_count=1024)) for i in (1, 2)]
        self.lever_info = lever_info()
        self.lever = Lever(self.lever_info, self.stage)
        self.bar = self.world.scene.add(RigidPrim(self.lever_info['carriage'], name='airtrack_lever',
                    reset_xform_properties=False, track_contact_forces=True,
                    contact_filter_prim_paths_expr=FINGERS, max_contact_count=1024))
        self.plunger = self.world.scene.add(RigidPrim(ROOT + '/Plunger', name='airtrack_plunger',
                       reset_xform_properties=False))
        self.cap_names = list(KEYS)
        self.cap_view = self.world.scene.add(RigidPrim([ROOT + '/Recorder/Keypad/Key_' + k for k in KEYS],
                        name='airtrack_caps', reset_xform_properties=False))
        # Whole-scene contact sensors per glider (robot links and apparatus; the other glider excluded).
        contact_targets = set()
        for prim in self.stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            target = prim
            ancestor = prim
            while ancestor and str(ancestor.GetPath()) != '/':
                if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    target = ancestor
                    break
                ancestor = ancestor.GetParent()
            contact_targets.add(str(target.GetPath()))
        self.external_views, self.external_filters = [], []
        for i in (1, 2):
            path = ROOT + f'/Slider{i}'
            filters = sorted(p for p in contact_targets if p != path and not p.startswith(path + '/')
                             and p != ROOT + f'/Slider{3 - i}')
            self.external_filters.append(filters)
            self.external_views.append(self.world.scene.add(RigidPrim(path, name=f'airtrack_external{i}',
                reset_xform_properties=False, track_contact_forces=True,
                contact_filter_prim_paths_expr=filters, max_contact_count=1024)))
        gripper = ParallelGripper(end_effector_prim_path=HAND, joint_prim_names=['panda_finger_joint1', 'panda_finger_joint2'],
                                  joint_opened_positions=np.array([.04, .04]), joint_closed_positions=np.array([.008, .008]),
                                  action_deltas=np.array([.005, .005]))
        self.robot = self.world.scene.add(SingleManipulator(prim_path=ROBOT, name='airtrack_franka',
                                                            end_effector_prim_path=HAND, gripper=gripper))
        self.cameras = [self.world.scene.add(Camera(prim_path=p, name=f'airtrack_camera{i}', resolution=(512, 512)))
                        for i, p in enumerate(CAMERAS)]
        if getattr(self.world, '_physics_context', None) is None:
            self.world._physics_context = PhysicsContext(physics_dt=self.dt, set_defaults=False)
        collision_reflex.prepare_scene(self.stage, ROBOT, self.private, ())   # colliders as drawn; contact reports
        self.world.initialize_physics()
        self.world.reset()
        self.reflex = collision_reflex.Reflex(ROBOT, self.dt, self.private)   # collision reflex
        standby = np.array([0., -.785, 0., -2.356, 0., 1.571, .785, .04, .04])
        self.robot.set_joints_default_state(positions=standby)
        self.robot.set_joint_positions(standby)
        self.robot.initialize()
        self.target = standby[:7].copy()
        self.velocity = np.zeros(7)
        self.rail_joints = [UsdPhysics.PrismaticJoint(self.stage.GetPrimAtPath(ROOT + f'/ReturnJoint{i}')) for i in (1, 2)]
        for j in self.rail_joints:
            if j.GetBody0Rel().GetTargets():
                raise RuntimeError('rail must be world anchored')
        self.catch_drives = [UsdPhysics.DriveAPI.Get(j.GetPrim(), 'linear') for j in self.rail_joints]
        self.plunger_drive = UsdPhysics.DriveAPI.Get(self.stage.GetPrimAtPath(ROOT + '/LauncherJoint'), 'linear')
        self.tongues = [self.stage.GetPrimAtPath(ROOT + f'/CatchTongue{n}') for n in GLIDERS]
        for i in (0, 1):
            self._set_catch(i, True)
        self._launcher_drive()
        for c in self.cameras:
            c.initialize()
            c.add_distance_to_image_plane_to_frame()
        for _ in range(60):
            self._apply_robot()
            self.world.step(render=True)
        self.last_q = np.asarray(self.robot.get_joint_positions()[:7]).copy()
        self.cap_rest_z = np.asarray(self.cap_view.get_world_poses()[0], float)[:, 2].copy()
        self.cap_down = [False] * len(self.cap_names)
        for sensor in [*self.views, self.bar]:
            self._finger_forces(sensor)
        self.screens = Screens(self.stage, {g: ROOT + f'/Gate{g}Screen' for g in SCREEN_X})
        self.video_path = self.output / 'airtrack_three_views_raw.mp4'
        self.writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1536, 512))
        if not self.writer.isOpened():
            raise RuntimeError('video writer unavailable')
        self.audit = (self.private / 'physics.jsonl').open('w')
        self._event('bench_state', dict(reason='start', **self.bench_state()))
        self._render()

    # --- readings ---------------------------------------------------------------------------------------------------
    def _transform(self, path):
        return self.UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))

    def _state(self, view):
        return np.asarray(view.get_world_poses()[0][0]), np.asarray(view.get_linear_velocities()[0])

    def _glider(self, i):
        """(track X of the centre, X speed) of glider i (0 = A, 1 = B); private."""
        p, v = self._state(self.views[i])
        return float(p[0] - TRACK_ORIGIN_WORLD[0]), float(v[0])

    def compression(self):
        """Plunger travel behind free length (m; > 0 = cocked back)."""
        return -float(self._state(self.plunger)[0][0] - TRACK_ORIGIN_WORLD[0] - PLUNGER_REST_X)

    def _finger_forces(self, view):
        v = np.asarray(view.get_contact_force_matrix(dt=self.dt))
        if v.size != 6 or not np.isfinite(v).all():
            raise RuntimeError('invalid finger contact sensor')
        return np.linalg.norm(v.reshape(2, 3), axis=1)

    def public_calibration(self):
        from robot_calibration import public_robot_calibration
        return public_robot_calibration(self)

    def robot_state(self):
        q = np.asarray(self.robot.get_joint_positions())
        # Proprioception only, the same fields as on every bench: gripper_holding; no forces, no part names.
        return dict(joint_positions_rad=q[:7].tolist(), gripper_width_m=float(sum(q[7:9])),
                    gripper_holding=any(n >= round(.1 / self.dt) for n in self.grip_ticks))

    def contact_channels(self):
        """Finger-force sensors on the two glider handles (CH1 blue A, CH2 red B) and the release lever post."""
        pairs = [self._finger_forces(v) for v in [*self.views, self.bar]]
        return {name: dict(finger_force_n=[float(x) for x in force], sustained_bilateral=bool(ticks >= round(.1 / self.dt)))
                for name, force, ticks in zip(('CH1', 'CH2', 'release_bar'), pairs, self.grip_ticks)}

    def set_gripper(self, width_m):
        self.width_target = width_m

    def _apply_robot(self):
        c = self.robot.get_articulation_controller()
        c.apply_action(self.Action(joint_positions=self.target, joint_velocities=self.velocity, joint_indices=np.arange(7)))
        c.apply_action(self.Action(joint_positions=np.array([self.width_target / 2] * 2), joint_indices=np.array([7, 8])))

    # --- hand-operated mechanisms -------------------------------------------------------------------------------------
    def _window(self):
        return self.release_tick is not None and (self.tick_index - self.release_tick) * self.dt < M.DASHPOT_DELAY_S

    def _launcher_drive(self):
        mode, stiffness, target, damping = self.pawl.drive(self.compression(), self._window())
        key = (mode, target, damping)
        if key == self.drive_mode:
            return
        self.plunger_drive.GetStiffnessAttr().Set(float(stiffness))
        self.plunger_drive.GetTargetPositionAttr().Set(float(target))
        self.plunger_drive.GetDampingAttr().Set(float(damping))
        self.drive_mode = key

    def _set_catch(self, i, on, cause=None):
        d = self.catch_drives[i]
        d.GetStiffnessAttr().Set(M.CATCH_STIFFNESS_N_PER_M if on else 0.)
        d.GetDampingAttr().Set(M.CATCH_DAMPING_NS_PER_M if on else 0.)
        d.GetTargetPositionAttr().Set(0.)
        self.caught[i] = on
        if cause is not None:
            x, v = self._glider(i)
            self._event('catch', dict(glider=GLIDERS[i], state='caught' if on else 'free', cause=cause,
                                      offset_mm=round((x - CENTERS[i]) * 1000, 2), speed_mm_s=round(v * 1000, 1)))

    def catches_down(self):
        return self.lever_index == 0

    def _trip(self, tick):
        """The lever passed to RELEASE: the pawl lets go, both catches lift, a READY recorder starts its run."""
        notch = self.pawl.trip()
        self.release_tick = tick
        for i in (0, 1):
            if self.caught[i]:
                self._set_catch(i, False, 'release')
        positions = {g: self._glider(i)[0] for i, g in enumerate(GLIDERS)}
        fields = dict(notch=notch, x_mm=[round(x * 1000, 2) for x in positions.values()])
        if self.panel.ready:
            eid = self.recorder.begin(tick, self.panel.ready_s, positions, notch=notch)
            self.panel.start(int(eid.rsplit(':', 1)[-1]))
            fields.update(evidence_id=eid, length_s=self.panel.recording_s)
        self._event('release', fields)

    def _controls(self, tick):
        depth = self.cap_rest_z - np.asarray(self.cap_view.get_world_poses()[0], float)[:, 2]
        for k, (name, d) in enumerate(zip(self.cap_names, depth)):
            if not self.cap_down[k] and d >= self.CA.SWITCH_DOWN_M:
                self.cap_down[k] = True
                self._key(name)
            elif self.cap_down[k] and d <= self.CA.SWITCH_UP_M:
                self.cap_down[k] = False
        carriage = np.asarray(self.bar.get_world_poses()[0], float)[0]
        for change in self.lever.update(tick, carriage[:2]):
            self._event('lever', {k: v for k, v in change.items() if k != 'tick'})
        if self.lever.index != self.lever_index:
            self.lever_index = self.lever.index
            if self.lever_index == 1:
                self._trip(tick)
            else:
                self._event('hold', dict(notch=self.pawl.notch()))

    def _key(self, key):
        effect = self.panel.press(key)
        if effect == 'ready':
            self.recorder.clear()
        self._event('key', dict(key=key, effect=effect, entry=self.panel.entry, ready_s=self.panel.ready_s,
                                caught=list(self.caught), notch=self.pawl.notch()))
        self._panel_changed()

    def _panel_changed(self):
        if self.panel.state != self.panel_state:
            self._event('recorder', dict(old=self.panel_state, new=self.panel.state, number=self.panel.number,
                                         ready_s=self.panel.ready_s, recording_s=self.panel.recording_s))
            self.panel_state = self.panel.state

    # --- physics tick -------------------------------------------------------------------------------------------------
    def step(self, target, velocity=None):
        if self.closed:
            raise RuntimeError('backend closed')
        if target is not None:
            q = np.asarray(target)
            self.velocity = (q - self.target) / self.dt if velocity is None else np.asarray(velocity)
            self.target = q
        else:
            self.velocity = np.zeros(7)
        self._launcher_drive()
        self._apply_robot()
        self.world.step(render=False)
        self.tick_index += 1
        tick = self.tick_index
        q = np.asarray(self.robot.get_joint_positions()[:7])
        rates = np.abs(q - self.last_q) / self.dt
        speed = float(np.max(rates))
        self.last_q = q.copy()
        if not np.isfinite(q).all():
            raise RuntimeError('robot joint state is not finite')
        from control_contract import reset_window, window_rates; mean_rates = window_rates(self, q, self.dt)
        if float(np.max(mean_rates)) > .9:
            from control_contract import protective_stop
            self.target = q.copy()
            self.velocity = np.zeros(7)
            self.pending_protective_stop = protective_stop(mean_rates); reset_window(self, q, self.dt)
        stop = self.reflex.check(self.tick_index, collision_reflex.held_path(self))
        if stop is not None:   # collision reflex: hold the measured pose; raised by the runtime after this tick
            self.target = q.copy(); self.velocity = np.zeros(7)
            if getattr(self, 'pending_protective_stop', None) is None:
                self.pending_protective_stop = stop
        pairs = [self._finger_forces(v) for v in [*self.views, self.bar]]
        self.grip_ticks = [n + 1 if min(f) > .02 else 0 for n, f in zip(self.grip_ticks, pairs)]
        self._controls(tick)
        # Catches: capture a glider at rest on its mark while lowered; let go of one dragged out of the notch.
        gliders = [self._glider(i) for i in (0, 1)]
        for i, (x, v) in enumerate(gliders):
            offset = x - CENTERS[i]
            if self.caught[i] and not M.catch_holds(offset):
                self._set_catch(i, False, 'pushed_out')
            elif not self.caught[i] and M.catch_engages(self.catches_down(), offset, v):
                self._set_catch(i, True, 'captured')
        # Pawl: drops into the deepest notch passed while the lever is at HOLD (not in the release window).
        compression = self.compression()
        notch = self.pawl.update(compression, self.catches_down() and not self._window())
        if notch is not None:
            self._event('cocked', dict(notch=notch, travel_mm=round(compression * 1000, 2)))
        # Contacts on the gliders (anything but the other glider); robot displacement for the process record.
        from coupled_oscillator.contact_summary import summarize_contacts
        external = {}
        for i, (sensor, filters) in enumerate(zip(self.external_views, self.external_filters)):
            count, summary = summarize_contacts(sensor.get_contact_force_data(dt=self.dt), filters, ROBOT)
            external[GLIDERS[i]] = count > 0
            x = gliders[i][0]
            if 'robot' in summary['categories']:
                if self.contact_start[i] is None:
                    self.contact_start[i] = x
                if not self.robot_moved and abs(x - self.contact_start[i]) > MOVED_M:
                    self.robot_moved = True
                    self._event('robot_moved', dict(glider=GLIDERS[i], x_mm=[round(g[0] * 1000, 2) for g in gliders]))
            else:
                self.contact_start[i] = None
            p = self._state(self.views[i])[0]
            if max(abs(float(p[1])), abs(float(p[2]) - HEIGHT)) > .0002:
                raise RuntimeError('glider left rail')
        if self.recorder is not None:
            positions = {g: gliders[i][0] for i, g in enumerate(GLIDERS)}
            speeds = {g: gliders[i][1] for i, g in enumerate(GLIDERS)}
            run = self.recorder.current
            ended = self.recorder.sample(tick, positions, speeds, external,
                                         {g: self.caught[i] for i, g in enumerate(GLIDERS)})
            if ended is not None:
                self.panel.finish()
                record = self.recorder.runs[ended]
                self._event('run_end', dict(evidence_id=ended, passes=len(record['_public_passes'])))
            elif run is not None:
                self.panel.tick((tick - run['start_tick']) * self.dt)
        self._panel_changed()
        self.snapshot = dict(gliders=gliders, compression=compression)
        if tick % self.frame_ticks == 0:
            self.audit.write(json.dumps(dict(
                tick=tick, x_mm=[round(g[0] * 1000, 2) for g in gliders], v_mm_s=[round(g[1] * 1000, 1) for g in gliders],
                caught=self.caught, plunger_mm=round(compression * 1000, 2), notch=self.pawl.notch(),
                lever=self.lever.state, rec=self.panel.state, grip=list(self.grip_ticks),
                ext=[external[g] for g in GLIDERS], speed=round(speed, 3))) + '\n')
        return {}

    def _event(self, kind, fields):
        if self.audit is None or self.audit.closed:
            return
        self.audit.write(json.dumps(dict(event=kind, tick=self.tick_index, **fields)) + '\n')

    def log_event(self, event, **fields):
        self._event(event, {k: v for k, v in fields.items() if k != 'tick'})

    def bench_state(self, reason=None):
        """State of the hand-operated bench (sealed as the terminal state; logged at start, stop and submission)."""
        snap = self.snapshot
        if snap is None:
            snap = dict(gliders=[self._glider(i) for i in (0, 1)], compression=self.compression())
        state = dict(
            released=not any(n >= round(.1 / self.dt) for n in self.grip_ticks),
            gliders=[dict(glider=g, offset_mm=round((x - CENTERS[i]) * 1000, 2), speed_mm_s=round(abs(v) * 1000, 2),
                          caught=bool(self.caught[i])) for i, (g, (x, v)) in enumerate(zip(GLIDERS, snap['gliders']))],
            launcher=dict(cocked=self.pawl.cocked(), notch=self.pawl.notch(),
                          travel_mm=round(snap['compression'] * 1000, 2)),
            lever=dict(state=self.lever.state, position_mm=round(self.lever.position_m * 1000, 1)),
            recorder=dict(self.panel.record(), ready=self.panel.ready, recording=self.panel.recording))
        if reason is not None:
            self._event('bench_state', dict(reason=reason, **state))
        return state

    # --- frames -------------------------------------------------------------------------------------------------------
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

    def _render_for_capture(self, renders=4):
        """Render until the cameras show the displays as they are now (at least `renders`, as before)."""
        until = max(getattr(self, 'render_index', 0) + renders, getattr(self, 'display_reset_until', 0) + DISPLAY_SETTLE_RENDERS)
        while getattr(self, 'render_index', 0) < until or getattr(self, 'display_resetting', False): self._display_render()

    def after_step(self, timers, tick):
        if tick != self.tick_index:
            raise RuntimeError('clock mismatch')
        if tick % self.frame_ticks == 0:
            self._render()

    def _visuals(self):
        z = CATCH_DOWN_Z + (0. if self.catches_down() else CATCH_LIFT_M)
        if z != self.tongue_z:
            for i, prim in enumerate(self.tongues):
                prim.GetAttribute('xformOp:translate').Set(self.Gf.Vec3d(CENTERS[i] + CATCH_PIN_DX, CATCH_Y, z))
            self.tongue_z = z
        shown = self.panel.display()
        self.CA.update_recorder_display(self.stage, ROOT + '/Recorder', shown)
        screens = self.recorder.screens() if self.recorder is not None else None
        if screens is not None:
            self.screens.update(screens)
        key = json.dumps([shown, screens], sort_keys=True, default=str)
        if key != getattr(self, 'display_key', None):
            self.display_key = key
            self._display_changed()

    def _render(self):
        self._visuals()
        for _ in range(4):
            self._display_render()
        rgba = [np.asarray(c.get_rgba()) for c in self.cameras]
        if any(a.shape != (512, 512, 4) for a in rgba):
            raise RuntimeError('camera frames unavailable')
        self.writer.write(self.cv2.cvtColor(np.concatenate([a[:, :, :3] for a in rgba], axis=1), self.cv2.COLOR_RGB2BGR))
        self.frame_index += 1

    def _log_observed_displays(self):
        """Private frame-log event: what the displays drew for this observation, as row text.
        gate_screens: per gate the ROWS screen rows (index, sign, time digits; blank rows included); recorder: the
        panel's display. Record only: nothing is returned to the Agent."""
        try:
            from .screens import ROWS, row_text
            screens = self.recorder.screens() if self.recorder is not None else {1: [], 2: []}
            shown = {}
            for gate, entries in screens.items():
                entries = list(entries)[-ROWS:]
                shown[str(gate)] = [list(row_text(e)) for e in entries + [None] * (ROWS - len(entries))]
            self._event('observation_displays', dict(gate_screens=shown, recorder=self.panel.display()))
        except Exception as exc:
            self._event('observation_displays', dict(log_error=f'{type(exc).__name__}: {exc}'))

    def observation(self):
        self._visuals()
        self._log_observed_displays()
        self._render_for_capture()
        result = []
        for path, c in zip(CAMERAS, self.cameras):
            rgba = np.asarray(c.get_rgba())
            depth = np.asarray(c.get_depth(device='cpu'))
            if rgba.shape != (512, 512, 4) or depth.shape != (512, 512):
                raise RuntimeError('RGB-D unavailable')
            valid = np.isfinite(depth) & (depth > 0) & (depth < 65.535)
            mm = np.zeros(depth.shape, dtype='<u2')
            mm[valid] = np.maximum(1, np.rint(depth[valid] * 1000)).astype('<u2')
            ok, png = self.cv2.imencode('.png', self.cv2.cvtColor(rgba[:, :, :3], self.cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError('image encoding failed')
            result.append(dict(camera=path.rsplit('/', 1)[-1], resolution=[512, 512], rgb_png=png.tobytes(),
                               depth_u16le_mm=mm.tobytes(),
                               intrinsics=np.asarray(c.get_intrinsics_matrix(device='cpu')).tolist(),
                               world_from_camera_ros=np.linalg.inv(c.get_view_matrix_ros(device='cpu')).tolist()))
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        if getattr(self, 'reflex', None) is not None: self.reflex.close()   # private contact log
        self.writer.release()
        if self.audit is not None:
            self.audit.close()

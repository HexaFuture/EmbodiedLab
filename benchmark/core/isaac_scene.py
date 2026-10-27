"""Neutral Isaac scene, extracted from the validated demo asset setup.

No demo policy, inverse kinematics, motion planner or parameter fitting lives
here. Construct only after SimulationApp; participant commands run elsewhere.
GPU acceptance of this extraction remains pending.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import numpy as np

# Displays: RTX's DLSS keeps a history of static pixels, so redrawn digits/states stayed stale in the cameras. As
# chem/circuit_grid: hold the history reset for a few renders after a display change, and take an observation only
# after the reset has settled.
HISTORY_RESET = '/rtx-transient/post/dlss/forceParamReset'
DISPLAY_RESET_RENDERS = 3
DISPLAY_SETTLE_RENDERS = 9
import collision_reflex
from gravity_contract import configure_gravity, verify_gravity

PACKAGE_ROOT = Path(__file__).resolve().parent
ROBOT_PATH = "/World/Franka"
HAND_PATH = ROBOT_PATH + "/panda_hand"
STANDBY = np.array([0, -.785, 0, -2.356, 0, 1.571, .785, .012, .012])
def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(type(value).__name__)

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_default) + "\n")

def quat_array(q):
    return np.array([float(q.GetReal()), *map(float, q.GetImaginary())])


class PendulumScene:
    def __init__(self, args, app):
        # Isaac imports deliberately occur after SimulationApp construction.
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
        from omni.physx import get_physx_scene_query_interface
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdRender, PhysxSchema
        from scripts.pendulum_instruments import GateConfig, OpticalGateTimer
        from pendulum_asset import build_apparatus, update_display, validate_apparatus

        self.args, self.app = args, app
        self.cv2, self.Gf, self.Usd, self.UsdGeom = cv2, Gf, Usd, UsdGeom
        self.Action, self.update_display = ArticulationAction, update_display
        self.dt, self.frame_ticks = 1 / args.physics_hz, args.physics_hz // 30
        self.output = Path(args.output)
        self.private = self.output / "private"
        self.frames_dir = self.output / "frames"
        self.private.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)
        # Bench events (experiment start, gate state machine) go to private/ with the physics audit, so that the
        # process record carries them.
        self.events_file = (self.private / "events.jsonl").open("w")
        self.audit_file = (self.private / "physics.jsonl").open("w")
        self.tick_index = self.frame_index = 0
        self.phase = "initializing"
        self.gripper_target = .012
        self.active_index = None
        self.released_indices = set()
        self.max_angles = {}
        self.rows, self.prediction = [], None
        self.writer = None
        self.last_display = {}
        self.last_action = None
        self.last_joint_positions = None
        self.contact_samples = {}
        self.gate_hit_paths = {}
        self.gate_obstructions = {}
        self.stage = None
        self.world = None
        self.cameras = []
        self.view_paths = [ROBOT_PATH + "/RoboCasaCameras/robot0_agentview_left",
                           ROBOT_PATH + "/RoboCasaCameras/robot0_agentview_right",
                           HAND_PATH + "/robot0_eye_in_hand"]

        if not Path(args.stage).is_file():
            raise FileNotFoundError(args.stage)
        # Prepare the layer before the renderer sees it. The inherited lab
        # contains machine-local HDR and network-only robot detail textures;
        # unavailable decorative maps must not stall camera initialization.
        source_stage = Usd.Stage.Open(args.stage)
        for prim in source_stage.Traverse():
            for attr in prim.GetAttributes():
                if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                    continue
                value = attr.Get()
                if not value or not str(value.path).lower().endswith((".png", ".jpg", ".exr", ".hdr")):
                    continue
                local_normal = Path(os.environ.get("PENDULUM_NORMAL_TEXTURE", "/workspace/normal.png"))
                if str(value.path).endswith("normal.png") and local_normal.is_file():
                    attr.Set(Sdf.AssetPath(str(local_normal)))
                elif not value.resolvedPath or str(value.resolvedPath).startswith(("http:", "https:")):
                    attr.Set(Sdf.AssetPath(""))
        local_stage = self.private / "base_scene.usda"
        source_stage.Flatten().Export(str(local_stage))
        omni.usd.get_context().open_stage(str(local_stage))
        for _ in range(180):
            app.update()
            stage = omni.usd.get_context().get_stage()
            if stage and stage.GetPrimAtPath(ROBOT_PATH):
                self.stage = stage
                break
        if self.stage is None:
            raise RuntimeError("Franka scene failed to load")
        camera_calibration = []
        for path in self.view_paths:
            camera = UsdGeom.Camera(stage.GetPrimAtPath(path))
            camera_calibration.append({"path": path,
                "mount_transform": np.array(UsdGeom.Xformable(camera).GetLocalTransformation()).tolist(),
                "focal_length": camera.GetFocalLengthAttr().Get(),
                "horizontal_aperture": camera.GetHorizontalApertureAttr().Get(),
                "vertical_aperture": camera.GetVerticalApertureAttr().Get(),
                "clipping_range": list(camera.GetClippingRangeAttr().Get())})
        write_json(self.output / "camera_calibration.json", camera_calibration)
        stage.GetPrimAtPath("/World/HookeApparatus").SetActive(False)
        # Hooke needed a 10 cm riser to reach its high spring tray. Here the
        # work is close to the tabletop: seat the robot directly on the table
        # and centre its reach across the three pendula before initialization.
        riser = stage.GetPrimAtPath("/World/RobotRiser")
        if riser:
            riser.SetActive(False)
        UsdGeom.XformCommonAPI(stage.GetPrimAtPath(ROBOT_PATH)).SetTranslate(Gf.Vec3d(.85, .40, .90))
        if stage.GetPrimAtPath("/World/HookePhysics"):
            stage.GetPrimAtPath("/World/HookePhysics").SetActive(False)
        self.descriptors = build_apparatus(stage)
        validate_apparatus(stage, self.descriptors)
        # Registry instances move the pendulum bench rigidly; descriptors follow.
        self.placement = None
        if getattr(args, "layout_candidates", None):
            import layout
            self.placement = layout.place_pendulum(stage, self.descriptors, args.layout_candidates, self.private,
                                                   "/World/PendulumLab", self.view_paths, ROBOT_PATH)
        for prim in stage.Traverse():
            if prim.IsA(UsdLux.DomeLight):
                UsdLux.DomeLight(prim).GetTextureFileAttr().Set(Sdf.AssetPath(""))
            # A content-identical archived texture removes a network dependency.
            for attr in prim.GetAttributes():
                if attr.GetTypeName() == Sdf.ValueTypeNames.Asset:
                    value = attr.Get()
                    local_normal = Path(os.environ.get("PENDULUM_NORMAL_TEXTURE", "/workspace/normal.png"))
                    if value and str(value.path).endswith("normal.png") and local_normal.is_file():
                        attr.Set(Sdf.AssetPath(str(local_normal)))
        for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PATH)):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
            visibility = prim.GetAttribute("visibility")
            if visibility:
                visibility.Set("inherited")
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                api.CreateSolverPositionIterationCountAttr(32)
                api.CreateSolverVelocityIterationCountAttr(32)
            if prim.IsA(UsdPhysics.RevoluteJoint):
                PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(float(np.degrees(.8)))
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                proximal = prim.GetName() in ("panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4")
                drive.CreateStiffnessAttr(3200 if proximal else 800)
                drive.CreateDampingAttr(120 if proximal else 80)
        import robot_physics
        robot_physics.configure_robot_contacts(stage, ROBOT_PATH)
        settings = carb.settings.get_settings()
        settings.set_bool("/physics/updateToUsd", True)
        settings.set_bool("/physics/updateVelocitiesToUsd", True)
        key = UsdLux.RectLight.Define(stage, "/World/Lights/PendulumKey")
        key.CreateIntensityAttr(3500)
        key.CreateWidthAttr(2.5)
        key.CreateHeightAttr(2.0)
        UsdGeom.XformCommonAPI(key).SetTranslate(Gf.Vec3d(.8, .0, 3))
        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/PendulumFill")
        dome.CreateIntensityAttr(450)
        dome.CreateColorAttr(Gf.Vec3f(.88, .93, 1))

        self.world = World(stage_units_in_meters=1.0, physics_dt=self.dt, rendering_dt=1/30)
        self.bodies = [self.world.scene.add(SingleRigidPrim(d["body_path"], name=d["id"],
                       reset_xform_properties=False)) for d in self.descriptors]
        gripper = ParallelGripper(end_effector_prim_path=ROBOT_PATH + "/panda_rightfinger",
            joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
            joint_opened_positions=np.array([.04, .04]), joint_closed_positions=np.array([.012, .012]),
            action_deltas=np.array([.01, .01]))
        self.robot = self.world.scene.add(SingleManipulator(prim_path=ROBOT_PATH,
            name="pendulum_franka", end_effector_prim_path=ROBOT_PATH + "/panda_rightfinger", gripper=gripper))
        for path in self.view_paths:
            if not stage.GetPrimAtPath(path).IsA(UsdGeom.Camera):
                raise RuntimeError(f"Required RoboCasa camera missing: {path}")
        if getattr(self.world, "_physics_context", None) is None:
            self.world._physics_context = PhysicsContext(physics_dt=self.dt, set_defaults=False)
            self.world.set_simulation_dt(physics_dt=self.dt, rendering_dt=1/30)
        # World/PhysicsContext initialization applies defaults. Bind the actual
        # context to the private episode parameter before any physical steps.
        configure_gravity(self.world.get_physics_context(), args.gravity)
        collision_reflex.prepare_scene(self.stage, ROBOT_PATH, self.private, ())   # colliders as drawn; contact reports
        self.world.initialize_physics()
        self.world.reset()
        self.reflex = collision_reflex.Reflex(ROBOT_PATH, self.dt, self.private)   # collision reflex
        verify_gravity(self.world.get_physics_context(), args.gravity)
        self.robot.set_joints_default_state(positions=STANDBY)
        self.robot.set_joint_positions(STANDBY)
        self.robot.initialize()
        self.last_action = ArticulationAction(joint_positions=STANDBY[:7], joint_velocities=np.zeros(7),
                                              joint_indices=np.arange(7))
        for _ in range(4):
            self.robot.get_articulation_controller().apply_action(self.last_action)
            self.world.step(render=False)
        hand_transform = self._transform(HAND_PATH)
        self.orientation = quat_array(hand_transform.ExtractRotationQuat())
        # A filter entry must identify one PhysX actor. A broad wildcard can
        # match several actors and silently leave the tensor view unusable.
        # Discover this private sensor configuration from the authored physics
        # model; none of these object paths are published to participants.
        actors = set()
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            actor, parent = prim, prim
            while parent and str(parent.GetPath()) != '/':
                if parent.HasAPI(UsdPhysics.RigidBodyAPI):
                    actor = parent
                    break
                parent = parent.GetParent()
            actors.add(str(actor.GetPath()))
        self.contact_filters = [sorted(actors-{d['body_path']}) for d in self.descriptors]
        self.contacts = [self.world.scene.add(RigidPrim(d['body_path'], name=d['id']+'_contact',
                         reset_xform_properties=False, track_contact_forces=True,
                         contact_filter_prim_paths_expr=filters,
                         max_contact_count=512)) for d, filters in zip(self.descriptors, self.contact_filters)]
        # The robot excites the pendula directly (fingers or any other link): its contact columns.
        self.robot_contact_columns = [[i for i, path in enumerate(filters) if path.startswith(ROBOT_PATH + '/')]
                                      for filters in self.contact_filters]
        # Reinitialize after authoring physical shapes; the only explicit arm
        # pose writes are initial conditions, before any recording or trial.
        self.world.reset()
        verify_gravity(self.world.get_physics_context(), args.gravity)
        self.robot.set_joint_positions(STANDBY)
        self.query = get_physx_scene_query_interface()
        self.timers = [OpticalGateTimer(GateConfig(target_cycles=args.cycles)) for _ in self.descriptors]
        self.gate_armed = [False for _ in self.descriptors]
        if args.record or args.preview_only:
            self.cameras = [self.world.scene.add(Camera(prim_path=p, name=f"pendulum_cam_{i}",
                            resolution=(640, 640))) for i,p in enumerate(self.view_paths)]
            for camera in self.cameras:
                camera.initialize()
                camera.add_distance_to_image_plane_to_frame()
            # RealTimePathTracing's temporal reconstruction retains old LCD
            # segments when the camera is stationary. Render each observation
            # with spatial denoising and no history reconstruction instead.
            # These are render-product settings, not camera/lens changes.
            for prim in stage.Traverse():
                if prim.IsA(UsdRender.Product):
                    for name, value_type, value in (
                        ("rendermode", Sdf.ValueTypeNames.Token, "PathTracing"),
                        ("pt:samplesPerPixel", Sdf.ValueTypeNames.UInt, 8),
                        ("pt:samplesPerIteration", Sdf.ValueTypeNames.Int, 8),
                        ("pt:clampSpp", Sdf.ValueTypeNames.Int, 8),
                        ("pt:denoising:enabled", Sdf.ValueTypeNames.Bool, True),
                        ("pt:denoising:optix:temporal", Sdf.ValueTypeNames.Bool, False),
                        ("post:aa:op", Sdf.ValueTypeNames.Token, "none"),
                    ):
                        prim.CreateAttribute("omni:rtx:" + name, value_type).Set(value)
            # Preserve the archived RoboCasa camera mounts, orientations and
            # lenses. Visibility is a scene/control-layout responsibility.
            for calibration in camera_calibration:
                camera = UsdGeom.Camera(stage.GetPrimAtPath(calibration["path"]))
                if not np.allclose(np.array(UsdGeom.Xformable(camera).GetLocalTransformation()),
                                   calibration["mount_transform"], atol=1e-6):
                    raise RuntimeError("RoboCasa camera mount changed during initialization")
                if abs(camera.GetFocalLengthAttr().Get()-calibration["focal_length"]) > 1e-6:
                    raise RuntimeError("RoboCasa camera lens changed during initialization")
                for key, attr in (("horizontal_aperture", camera.GetHorizontalApertureAttr()),
                                  ("vertical_aperture", camera.GetVerticalApertureAttr()),
                                  ("clipping_range", camera.GetClippingRangeAttr())):
                    if not np.allclose(attr.Get(), calibration[key], atol=1e-6):
                        raise RuntimeError(f"RoboCasa camera {key} changed during initialization")
        for _ in range(30 if self.cameras else 4):
            self.robot.get_articulation_controller().apply_action(self.last_action)
            self.world.step(render=bool(self.cameras))
        self.tick_index = 0
        self.sim_epoch = float(self.world.current_time)
        self.last_joint_positions = np.asarray(self.robot.get_joint_positions()[:7])
        if args.record:
            self.video_path = self.output / "pendulum_three_views_raw.mp4"
            self.writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 640))
            if not self.writer.isOpened():
                raise RuntimeError("video writer could not open")
        gravity_readback = verify_gravity(self.world.get_physics_context(), args.gravity)
        self.stage.GetRootLayer().Export(str(self.private / "runtime_scene.usda"))
        public_geometry = [{key: d[key] for key in ("id", "length_m", "bob_radius_m", "bob_mass_kg",
            "rod_mass_kg", "rod_radius_m", "effective_length_m")} for d in self.descriptors]
        self.public = {"schema_version": "embodiedlab.pendulum_measurements/v1", "policy": "external_participant",
            "measurement_source": "raycast_optical_gate", "clock": "simulation_time", "physics_hz": args.physics_hz,
            "geometry": public_geometry, "measurements": self.rows, "held_out_prediction": None, "estimate": None,
            "valid": False}
        write_json(self.output / "measurements.json", self.public)
        write_json(self.private / "run_manifest.json", {"gravity_m_s2": args.gravity, "physics_gravity": gravity_readback, "physics_hz": args.physics_hz,
            "descriptors": self.descriptors, "excitation_model": "direct robot contact (fingers or arm)",
            "source_sha256": {str(path.relative_to(PACKAGE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__), PACKAGE_ROOT / "pendulum_asset.py", PACKAGE_ROOT / "gravity_contract.py", PACKAGE_ROOT / "scripts/pendulum_instruments.py")}})
        self.event("experiment_start", instruction="Measure gravitational acceleration using the pendulum apparatus",
                   policy="external_participant", target_cycles=args.cycles)
        print("PENDULUM_SCENE_READY", flush=True)

    def _transform(self, path):
        return self.UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))

    def event(self, event, **data):
        row = {"event": event, "sim_time_s": self.tick_index*self.dt,
               "video_time_s": self.frame_index/30, "phase": self.phase, **data}
        self.events_file.write(json.dumps(row, default=json_default) + "\n")
        self.events_file.flush()
        print("PENDULUM_EVENT=" + json.dumps(row, default=json_default), flush=True)

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

    def _render(self):
        for index, d in enumerate(self.descriptors):
            r = self.timers[index].reading()
            # IDLE: no acquisition on this gate (the backend's shared idle timer); ARM: started, waiting for the first
            # crossing or in the warm-up cycle.
            idle = self.timers[index] is getattr(self, "idle_timer", None)
            panel = (round(r.elapsed_s, 3), r.cycles, "IDLE" if idle else "ERR" if r.status == "invalid" else
                     "DONE" if r.valid else "RUN" if r.status == "timing" else "ARM")
            if self.last_display.get(index) != panel:
                self.update_display(self.stage, d["display"], *panel)
                self.last_display[index] = panel
                self._display_changed()
        if not self.cameras:
            return
        # Drain the renderer/annotator pipeline without advancing physics.
        # A single render can return the preceding observation's RGB buffer.
        for _ in range(3):
            self._display_render()
        views = []
        for i, camera in enumerate(self.cameras):
            rgba = np.asarray(camera.get_rgba())
            if rgba.shape != (640, 640, 4):
                raise RuntimeError(f"camera {i} missing frame {self.frame_index}")
            views.append(rgba[..., :3].astype(np.uint8))
        if self.writer:
            self.writer.write(self.cv2.cvtColor(np.concatenate(views, axis=1), self.cv2.COLOR_RGB2BGR))
        self.frame_index += 1
        if self.writer and self.frame_index % 300 == 0:
            print(f"PENDULUM_VIDEO_FRAME={self.frame_index}", flush=True)

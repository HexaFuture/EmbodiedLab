"""Isaac device backend. No IK, planning, demo execution or answer fitting.

Requires GPU integration validation. Contact-force thresholds are diagnostic
development values and must be checked against recorded raw contact tensors.
"""
import math
import numpy as np

from isaac_scene import PendulumScene, json_default
from contact_reporting import contact_flags


def idle_at_start(scene, idle_timer):
    """Every gate starts idle, so the first render (before any physics step) draws IDLE, 00.000 and 00 as the card
    says (the scene's own fresh timers drew ARM at tick 0)."""
    scene.idle_timer = idle_timer
    scene.timers = [idle_timer for _ in scene.descriptors]


class IsaacBackend:
    def __init__(self, args, app):
        self.scene = PendulumScene(args, app)
        self.pendulum_ids = tuple(d['id'] for d in self.scene.descriptors)
        self._closed = False
        self.validate_contact_sensors()
        # Gate push buttons: the switch of a cap closes when the cap is SWITCH_DOWN_M below its rest position
        # (its kinematic anchor) and opens again above SWITCH_UP_M (hysteresis, as the chemistry keys).
        from pxr import UsdGeom
        from pendulum_asset import SWITCH_DOWN_M, SWITCH_UP_M
        from scripts.pendulum_instruments import GateConfig, OpticalGateTimer, TIMED_CYCLES
        self._switch_down, self._switch_up = SWITCH_DOWN_M, SWITCH_UP_M
        cache = UsdGeom.XformCache()
        self._buttons = []          # (pendulum id, button, cap path, anchor world position, anchor world up axis)
        for d in self.scene.descriptors:
            for name, button in d['buttons'].items():
                anchor = cache.GetLocalToWorldTransform(self.scene.stage.GetPrimAtPath(button['anchor_path']))
                up = np.asarray(anchor.TransformDir(self.scene.Gf.Vec3d(0, 0, 1)), float)
                self._buttons.append((d['id'], name, button['cap_path'],
                                      np.asarray(anchor.ExtractTranslation(), float), up/np.linalg.norm(up)))
        self.switch_closed = {(key, name): False for key, name, *_ in self._buttons}
        self.cap_depth_m = {(key, name): 0.0 for key, name, *_ in self._buttons}
        # An idle gate's timer panel reads zero (IDLE 00.000, 00 cycles); the scene shows IDLE for this timer.
        self._idle_timer = OpticalGateTimer(GateConfig(target_cycles=TIMED_CYCLES))
        idle_at_start(self.scene, self._idle_timer)

    def validate_contact_sensors(self):
        """Fail startup before a public ready endpoint can be constructed."""
        for sensor in self.scene.contacts:
            detail = sensor.get_contact_force_data(dt=self.scene.dt)
            if (detail is None or len(detail) != 6 or np.asarray(detail[4]).size == 0
                    or not np.isfinite(np.asarray(detail[4])).all()
                    or np.any(np.asarray(detail[4]) < 0)):
                raise RuntimeError('contact sensor initialization failed')

    def robot_state(self):
        q = np.asarray(self.scene.robot.get_joint_positions())
        return dict(joint_positions_rad=q[:7].tolist(), gripper_width_m=float(sum(q[7:9])))

    def public_calibration(self):
        from robot_calibration import public_robot_calibration
        return public_robot_calibration(self.scene)

    def set_gripper(self, width_m):
        self.scene.gripper_target = width_m/2

    def log_event(self, event, **data):
        """A bench event (gate state machine, button press) in the private event log."""
        self.scene.event(event, **data)

    def _read_buttons(self):
        """Cap travel below rest of every gate button and its debounced switch state."""
        from pxr import UsdGeom
        cache = UsdGeom.XformCache()
        for key, name, path, rest, up in self._buttons:
            position = np.asarray(cache.GetLocalToWorldTransform(self.scene.stage.GetPrimAtPath(path)).ExtractTranslation(), float)
            depth = float(np.dot(rest - position, up))
            if not math.isfinite(depth):
                raise RuntimeError('gate button state invalid')
            self.cap_depth_m[(key, name)] = depth
            if depth >= self._switch_down:
                self.switch_closed[(key, name)] = True
            elif depth <= self._switch_up:
                self.switch_closed[(key, name)] = False

    def step(self, target, velocity=None):
        import json
        sim = self.scene
        if self._closed:
            raise RuntimeError('physics backend closed')
        if target is not None:
            previous = np.asarray(sim.last_action.joint_positions)
            positions = np.asarray(target)
            sim.last_action = sim.Action(joint_positions=positions,
                joint_velocities=(positions-previous)/sim.dt if velocity is None else np.asarray(velocity), joint_indices=np.arange(7))
        else:
            sim.last_action = sim.Action(joint_positions=sim.last_action.joint_positions,
                joint_velocities=np.zeros(7), joint_indices=np.arange(7))
        sim.robot.get_articulation_controller().apply_action(sim.last_action)
        sim.robot.get_articulation_controller().apply_action(sim.Action(
            joint_positions=np.array([sim.gripper_target]*2), joint_indices=np.array([7, 8])))
        sim.world.step(render=False)
        sim.tick_index += 1
        q = np.asarray(sim.robot.get_joint_positions()[:7])
        previous_q = np.asarray(sim.last_joint_positions)
        speed = float(np.max(np.abs(q-previous_q))/sim.dt)
        sim.last_joint_positions = q.copy()
        samples = {}
        self._read_buttons()
        for index, descriptor in enumerate(sim.descriptors):
            transform = sim._transform(descriptor['body_path'])
            bob = np.array(transform.Transform(sim.Gf.Vec3d(*descriptor['bob_local_point'])))
            pivot = np.asarray(descriptor['pivot'])
            # Swing-plane angle; the axis is world X unless the bench was placed with a yaw.
            axis = np.asarray(descriptor.get('swing_axis_world', (1.0, 0.0, 0.0)))
            angle = math.degrees(math.atan2(float(np.dot(bob-pivot, axis)), pivot[2]-bob[2]))
            net = np.asarray(sim.contacts[index].get_net_contact_forces(dt=sim.dt))
            if not np.isfinite(net).all():
                raise RuntimeError('contact sensor data invalid')
            contacts = sim.contacts[index].get_contact_force_data(dt=sim.dt)
            if contacts is None:
                raise RuntimeError('contact detail sensor unavailable')
            # Counts detect simultaneous opposing contacts even when their
            # aggregate force cancels. A full buffer fails closed.
            pair_counts = np.asarray(contacts[4]).reshape(-1)
            count = int(pair_counts.sum())
            robot_count = int(pair_counts[sim.robot_contact_columns[index]].sum()) if sim.robot_contact_columns[index] else 0
            flags = contact_flags(count, float(np.linalg.norm(net)), robot_count)
            ray = sim.query.raycast_closest(tuple(descriptor['gate_ray_origin']),
                tuple(descriptor['gate_ray_direction']), descriptor['gate_ray_distance'])
            hit = str(ray.get('collision', '')) if ray.get('hit') else None
            samples[descriptor['id']] = dict(blocked=bool(ray.get('hit')),
                angle_deg=round(angle, 4), **flags,
                unexpected_occluder=bool(hit and hit != descriptor['rod_path']),
                start_button=self.switch_closed[(descriptor['id'], 'start')],
                reset_button=self.switch_closed[(descriptor['id'], 'reset')])
        self._audit(sim, samples, q, speed)
        if not np.isfinite(q).all():
            from control_contract import SimDiverged
            raise SimDiverged('robot joint state is not finite')
        from control_contract import reset_window, window_rates; mean_rates = window_rates(self, q, sim.dt)
        if float(np.max(mean_rates)) > .9:
            from control_contract import protective_stop
            # Protective stop: hold the measured pose; the command ends, the episode continues.
            sim.last_action = sim.Action(joint_positions=q.copy(), joint_velocities=np.zeros(7), joint_indices=np.arange(7))
            sim.robot.get_articulation_controller().apply_action(sim.last_action)
            self.pending_protective_stop = protective_stop(mean_rates); reset_window(self, q, sim.dt)
        stop = sim.reflex.check(sim.tick_index)
        if stop is not None:   # collision reflex: hold the measured pose; the command ends, the episode continues
            sim.last_action = sim.Action(joint_positions=q.copy(), joint_velocities=np.zeros(7), joint_indices=np.arange(7))
            sim.robot.get_articulation_controller().apply_action(sim.last_action)
            if getattr(self, 'pending_protective_stop', None) is None:
                self.pending_protective_stop = stop
        return samples

    def _audit(self, sim, samples, q, speed):
        """Frame audit row (30 Hz, physics.jsonl) for the process record, small enough for a full episode: per
        pendulum the angle at the frame tick and the largest |angle| of the frame (peak_deg), contact flags true if
        set on any tick of the frame, the gate ray and buttons at the frame tick; joints and the largest measured
        joint speed of the frame."""
        import json
        acc = getattr(self, '_frame', None)
        if acc is None:
            acc = self._frame = {'speed': 0.0, 'pendula': {}}
        acc['speed'] = max(acc['speed'], speed)
        for key, p in samples.items():
            a = acc['pendula'].setdefault(key, {'peak_deg': 0.0, 'robot_contact': False, 'any_contact': False,
                                                'unexpected_occluder': False})
            a['peak_deg'] = max(a['peak_deg'], abs(p['angle_deg']))
            for flag in ('robot_contact', 'any_contact', 'unexpected_occluder'):
                a[flag] = a[flag] or p[flag]
        if sim.tick_index % sim.frame_ticks:
            return
        pendula = {key: {'angle_deg': round(p['angle_deg'], 3), 'peak_deg': round(acc['pendula'][key]['peak_deg'], 3),
                         'robot_contact': acc['pendula'][key]['robot_contact'],
                         'any_contact': acc['pendula'][key]['any_contact'],
                         'unexpected_occluder': acc['pendula'][key]['unexpected_occluder'],
                         'blocked': p['blocked'], 'start_button': p['start_button'], 'reset_button': p['reset_button']}
                   for key, p in samples.items()}
        sim.audit_file.write(json.dumps(dict(physics_tick=sim.tick_index, pendula=pendula,
            joint_positions_rad=[round(float(v), 4) for v in q], max_joint_speed_rad_s=round(acc['speed'], 4)),
            default=json_default, allow_nan=False)+'\n')
        self._frame = None

    def after_step(self, timers, tick):
        sim = self.scene
        if tick != sim.tick_index:
            raise RuntimeError('runtime and physics clock diverged')
        for index, key in enumerate(self.pendulum_ids):
            # A gate without a current acquisition (never started, or RESET) shows zero.
            sim.timers[index] = timers.get(key, self._idle_timer)
        if tick % sim.frame_ticks == 0:
            sim._render()

    def observation(self):
        sim = self.scene
        # Read-only rendering, separate from the 30 Hz video writer.
        sim._render_for_capture(3)   # the gate displays as they are now (DLSS history reset settled)
        images = []
        for path, camera in zip(sim.view_paths, sim.cameras):
            rgba = np.asarray(camera.get_rgba())
            if rgba.shape != (640, 640, 4):
                raise RuntimeError('camera frame unavailable')
            depth = np.asarray(camera.get_depth(device='cpu'))
            if depth.shape != rgba.shape[:2]:
                raise RuntimeError('depth frame does not match RGB camera')
            valid = np.isfinite(depth) & (depth > 0) & (depth < 65.535)
            millimetres = np.zeros(depth.shape, dtype='<u2')
            millimetres[valid] = np.maximum(1, np.rint(depth[valid]*1000)).astype('<u2')
            ok, encoded = sim.cv2.imencode('.png', sim.cv2.cvtColor(rgba[..., :3], sim.cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError('camera PNG encoding failed')
            images.append({'camera': path.rsplit('/', 1)[-1], 'resolution': [640, 640],
                'rgb_png': encoded.tobytes(), 'depth_u16le_mm': millimetres.tobytes(),
                'intrinsics': np.asarray(camera.get_intrinsics_matrix(device='cpu')).tolist(),
                'world_from_camera_ros': np.linalg.inv(camera.get_view_matrix_ros(device='cpu')).tolist()})
        return images

    def close(self):
        if self._closed:
            return
        self._closed = True
        sim = self.scene
        if getattr(sim, 'reflex', None) is not None:
            sim.reflex.close()   # private contact log
        if sim.writer:
            sim.writer.release()
        sim.audit_file.close()
        sim.events_file.close()

"""Contact-conditioned grasp shared by every bench with a gripper.

Friction grasps of small parts are unreliable in the simulator, so a held part
is carried by a rigid constraint instead (hooke_physics.attach_grasp_joint).
What decides that a part is held is physical and visible: the fingers are
closing onto something, both fingers touch the same part, and that part stops
them. There is no hidden grasp point and no width threshold; how and where to
grip is the Agent's choice. Commanding the fingers wider than the held part
releases it. The public description below is what the Agent reads.
"""

import numpy as np

FINGER_NAMES = ('panda_leftfinger', 'panda_rightfinger')
GRIP_MARGIN_M = 0.002           # commanded at least this much narrower than the opening: the fingers press
GRIP_STALL_M_S = 0.005          # finger joints slower than this count as stopped
GRIP_STALL_TICKS = 6            # stopped this many physics ticks in a row (25 ms at 240 Hz)
CAPTURE_RELATIVE_SPEED_M_S = 0.05
HOLD_BREAK_FORCE_N = 35.0       # must match hooke_physics.attach_grasp_joint
HOLD_BREAK_TORQUE_N_M = 3.0
CONTACT_BUFFER = 64
FINGER_LOCK_SLACK_M = 0.0005    # a locked finger yields at most this much (per finger) before its stop
FINGER_CLOSING_FORCE_N = 7.2    # published only: the hand asset's finger drive force limit (panda_finger_joint1 maxForce)


def public_gripper_mechanism():
    """The gripper facts every task spec publishes (device_calibration.gripper_mechanism)."""
    return {
        'type': 'contact_conditioned_constrained_gripper',
        'grip_margin_m': GRIP_MARGIN_M,
        'maximum_capture_relative_speed_m_s': CAPTURE_RELATIVE_SPEED_M_S,
        'hold_break_force_n': HOLD_BREAK_FORCE_N,
        'hold_break_torque_n_m': HOLD_BREAK_TORQUE_N_M,
        'finger_closing_force_n': FINGER_CLOSING_FORCE_N,
        'finger_give_under_load_m': FINGER_LOCK_SLACK_M,
        'description': (
            'Simulated grasp: finger friction is replaced by a rigid hold. A part is held once both fingers '
            'close onto it and it stops them: the commanded width is at least 2 mm narrower than the part, '
            'the fingers have come to rest on it, and the part moves with the hand (within 0.05 m/s). The hold '
            'is made where the hand is, the part is not moved. Where and how to grip is up to you; the fingers '
            'must enclose the part, a touch from one side or from above holds nothing. Commanding the fingers more '
            'than 2 mm wider than the held part releases it; the release is complete, and the part counts as set down, once '
            'the fingers have opened 2 mm beyond it and nothing of the robot touches it. A load above 35 N or 3 N m breaks the hold; open the '
            'fingers before gripping again. There is no hidden grasp point on any part. The gripper width in the robot '
            'state is the actual finger opening; while a part is held the fingers rest on it, so the width is the part\'s '
            'width between them. The fingers close with at most 7.2 N and are not backdrivable: an outside load does not '
            'open them (each gives at most 0.5 mm); they open only when commanded wider.'),
    }


def finger_contact_sensors(scene, rigid_prim, robot_path, paths, prefix):
    """One contact view per graspable body, filtered to the two fingers (columns: left, right)."""
    filters = [robot_path + '/' + name for name in FINGER_NAMES]
    return {key: scene.add(rigid_prim(path, name=f'{prefix}_{key}_finger_contact', reset_xform_properties=False,
                                      track_contact_forces=True, contact_filter_prim_paths_expr=filters,
                                      max_contact_count=CONTACT_BUFFER))
            for key, path in paths.items()}


def fingers_touching(sensor, dt):
    """(left, right): whether each finger touches the sensor's body (a contact pair or any non-zero force)."""
    counts = np.asarray(sensor.get_contact_force_data(dt=dt)[4]).reshape(-1)
    forces = np.asarray(sensor.get_contact_force_matrix(dt=dt), float).reshape(-1, 3)
    # A full buffer (count at CONTACT_BUFFER) is contact: only malformed data is an error.
    if counts.size != 2 or forces.shape[0] != 2 or (counts < 0).any() or not np.isfinite(forces).all():
        raise RuntimeError('finger contact buffer invalid')
    touching = (counts > 0) | (np.linalg.norm(forces, axis=1) > 1e-5)
    return bool(touching[0]), bool(touching[1])


def pinched_body(bodies, jaw, jaw_velocity, dt):
    """The body both fingers touch and that moves with the hand, nearest the jaw; None if there is none.

    bodies: {key: (finger contact sensor, world position, world linear velocity)}.
    """
    candidates = []
    for key, (sensor, position, velocity) in bodies.items():
        if not all(fingers_touching(sensor, dt)):
            continue
        if np.linalg.norm(np.asarray(velocity, float) - np.asarray(jaw_velocity, float)) > CAPTURE_RELATIVE_SPEED_M_S:
            continue
        candidates.append((float(np.linalg.norm(np.asarray(position, float) - np.asarray(jaw, float))), str(key), key))
    return min(candidates)[2] if candidates else None


class GripState:
    """Finger bookkeeping for one gripper: when the fingers press, the width of the held part, when they cleared it."""

    def __init__(self):
        self.stalled = 0
        self.hold_width = None
        self.finger_lock = None
        self._dof_limits = None
        self.broke = False

    @staticmethod
    def _opening(finger_q):
        return float(finger_q[0] + finger_q[1])

    def pressing(self, finger_q, finger_qd, width_target):
        """True once the fingers have been commanded narrower than they are and stopped, for GRIP_STALL_TICKS ticks."""
        stopped = float(np.max(np.abs(np.asarray(finger_qd, float)))) < GRIP_STALL_M_S
        blocked = width_target + GRIP_MARGIN_M < self._opening(finger_q) and stopped
        self.stalled = self.stalled + 1 if blocked else 0
        return self.stalled >= GRIP_STALL_TICKS

    def hold(self, finger_q):
        self.hold_width = self._opening(finger_q)

    def drive_width(self, width_target, robot=None):
        """The finger drive's target opening: the commanded width, but never narrower than a held part.

        Once the hold forms the fingers stop where they met the part (the opening at the hold) instead of closing on
        through it, so the reported gripper width is the opening at the part's surfaces. The hold itself does not
        depend on the fingers (the constraint carries the part; finger contacts with it are filtered); a command wider
        than the part opens them as before. From the hold until the fingers have cleared the part after a release.

        With the robot given, the finger stops are updated for this target as well (lock_fingers).
        """
        width = float(width_target) if self.hold_width is None else max(float(width_target), self.hold_width)
        if robot is not None:
            self.lock_fingers(robot, width)
        return width

    def finger_stops(self, finger_q, width):
        """Upper stops of the two finger joints for a drive target opening `width`: the fingers are not backdrivable.

        Like a real hand's self-locking finger drive, a finger closes under its drive and opens only when commanded:
        each finger's stop follows it inward (its narrowest position since the target last widened) and is never
        closer than the commanded half width. An outside load (an edge pressed between the fingertips, a part levered
        against them) cannot pry the fingers open by more than FINGER_LOCK_SLACK_M each. Closing force and speed are
        the drive's own; a stop acts only against opening.
        """
        q = np.asarray(finger_q, float)
        half = width / 2
        lock = q if self.finger_lock is None else np.minimum(self.finger_lock, q)
        self.finger_lock = np.maximum(half, lock)
        return self.finger_lock + FINGER_LOCK_SLACK_M

    def lock_fingers(self, robot, width):
        """Write finger_stops into the finger joints' upper limits, never past the hand's own travel."""
        view = robot._articulation_view._physics_view
        if self._dof_limits is None:
            self._dof_limits = np.array(view.get_dof_limits(), dtype=np.float32).reshape(view.count, -1, 2)
            self._finger_travel = self._dof_limits[0, 7:9, 1].copy()
        stops = np.minimum(self._finger_travel, self.finger_stops(np.asarray(robot.get_joint_positions())[7:9], width))
        if np.abs(stops - self._dof_limits[0, 7:9, 1]).max() > 1e-6:
            self._dof_limits[:, 7:9, 1] = stops
            view.set_dof_limits(self._dof_limits, np.arange(view.count, dtype=np.int32))

    def opens(self, width_m, finger_q):
        """Whether a width command opens the fingers: wider than the held part, or than the fingers are now."""
        reference = self.hold_width if self.hold_width is not None else self._opening(finger_q)
        return width_m > reference + GRIP_MARGIN_M

    def cleared(self, finger_q):
        """After a release: the fingers are clear of the part again (finger collisions may be restored)."""
        done = self.hold_width is None or self._opening(finger_q) >= self.hold_width + GRIP_MARGIN_M
        if done:
            self.hold_width = None
            self.broke = False
        return done

    def break_cleared(self, finger_q):
        """After the hold broke under load: True once, on the first tick the fingers are not inside the part.

        A broken hold no longer carries the part, so the fingers must collide with it again at once (finger collisions
        restored now, not when the fingers have opened past it): otherwise a hand still pushing would pass through the
        part it had held. The fingers rest where they held it (drive_width keeps them at the part's width), so this is
        normally the next tick. The release itself still completes in cleared.
        """
        if not self.broke or self.hold_width is None or self._opening(finger_q) < self.hold_width - FINGER_LOCK_SLACK_M:
            return False
        self.broke = False
        return True

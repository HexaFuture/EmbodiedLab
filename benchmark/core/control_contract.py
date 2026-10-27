"""Benchmark-owned control validation. No IK, planning, or experiment policy.

Trajectories contain one seven-joint drive target per physics tick. The
benchmark executes drives, never writes rigid-body poses. It checks shape,
finiteness and budget only; a target beyond a joint's physical range is
clamped to that range, exactly as the drive would saturate, and the robot's
own measured state is never a reason to refuse a command. What the arm can
reach, and what it collides with, is left to the physics. The same rules
apply to every participant, scripted or not.
"""
from collections import deque
from dataclasses import dataclass
import math


JOINT_LIMITS = ((-2.8973, 2.8973), (-1.7628, 1.7628),
                (-2.8973, 2.8973), (-3.0718, -0.0698),
                (-2.8973, 2.8973), (-0.0175, 3.7525),
                (-2.8973, 2.8973))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def joint_vector(value):
    """Seven finite joint targets, clamped to the physical joint range."""
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        raise ValueError("expected seven arm joint positions in radians")
    for index, q in enumerate(value):
        if not finite(q):
            raise ValueError(f"non-finite joint target for joint {index + 1}")
    return tuple(min(max(float(q), lo), hi) for q, (lo, hi) in zip(value, JOINT_LIMITS))


@dataclass(frozen=True)
class ControlLimits:
    physics_hz: int = 240
    max_command_seconds: float = 30.0
    max_joint_speed_rad_s: float = 0.65

    def __post_init__(self):
        if type(self.physics_hz) is not int or self.physics_hz not in (240, 480):
            raise ValueError("supported fixed physics rates are 240 and 480 Hz")
        if any(not finite(v) or v <= 0 for v in (self.max_command_seconds, self.max_joint_speed_rad_s)):
            raise ValueError("control limits must be finite and positive")


def validate_trajectory(targets, current_joints, remaining_ticks, limits=ControlLimits()):
    """Validate finite drive setpoints and budget before mutation.

    Setpoint differences are not measured robot velocity: contact and servo
    lag can separate encoders from the previous target. Motion generation
    belongs to participants; the backend enforces actual physical limits.
    `current_joints` is accepted for the call sites that pass it and is not
    judged: a joint resting on its limit (which the simulator reports a few
    nanoradians beyond it) used to make every later command unrecoverable.
    """
    if type(remaining_ticks) is not int or remaining_ticks < 0:
        raise ValueError("invalid remaining tick budget")
    if not isinstance(targets, (list, tuple)) or not targets:
        raise ValueError("trajectory must contain at least one physics tick")
    max_ticks = int(limits.max_command_seconds * limits.physics_hz)
    if len(targets) > max_ticks:
        raise ValueError(f"a single command may last at most {limits.max_command_seconds:g} s ({max_ticks} ticks at "
                         f"{limits.physics_hz} Hz); this trajectory has {len(targets)} ticks")
    if len(targets) > remaining_ticks:
        raise ValueError(f"trajectory exceeds the remaining episode budget "
                         f"({remaining_ticks / limits.physics_hz:.1f} s left)")
    validated = []
    for row in targets:
        q = joint_vector(row)
        validated.append(q)
    return tuple(validated)


def duration_ticks(seconds, remaining_ticks, limits=ControlLimits()):
    if not finite(seconds) or not 0 < seconds <= limits.max_command_seconds:
        raise ValueError(f"a single command may last at most {limits.max_command_seconds:g} s "
                         f"(duration must be > 0 and <= {limits.max_command_seconds:g} s)")
    ticks = math.ceil(seconds * limits.physics_hz)
    if ticks > remaining_ticks:
        raise ValueError(f"duration exceeds the remaining episode budget "
                         f"({remaining_ticks / limits.physics_hz:.1f} s left)")
    return ticks


def validate_velocities(velocities, count, limits=ControlLimits()):
    """Optional drive feed-forward; never computes IK or chooses a path."""
    if not isinstance(velocities, (list, tuple)) or len(velocities) != count:
        raise ValueError('one velocity vector per trajectory tick required')
    if any(not isinstance(row, (list, tuple)) or len(row) != 7
           or any(not finite(v) or abs(v) > limits.max_joint_speed_rad_s+1e-9 for v in row)
           for row in velocities):
        raise ValueError('invalid joint velocity feed-forward')
    return tuple(tuple(float(v) for v in row) for row in velocities)


PUBLIC_TOOLS = (
    "get_task_spec", "get_observation", "get_robot_state",
    "execute_joint_trajectory", "set_gripper", "wait",
    "record_prediction", "submit_result",
)


PROTECTIVE_STOP_RAD_S = 0.9     # mean joint speed over PROTECTIVE_STOP_WINDOW_S that triggers a protective stop
PROTECTIVE_STOP_WINDOW_S = 0.1  # window of that mean: a joint's net travel over the window divided by its length
PROTECTIVE_STOP_BRAKE_S = 0.25          # after a stop the arm brakes at the stop pose for at most this long ...
PROTECTIVE_STOP_REST_RAD_S = 0.1        # ... or until every arm joint moves slower than this ...
PROTECTIVE_STOP_REST_S = 0.025          # ... for this long (device_runtime)
PHYSICAL_JOINT_SPEED_CAP_RAD_S = 0.8   # PhysX joint velocity cap (hooke_physics.configure_arm)


class ProtectiveStop(RuntimeError):
    """A joint's mean speed over the last PROTECTIVE_STOP_WINDOW_S exceeded PROTECTIVE_STOP_RAD_S.

    Like a real arm's protective stop: the target is set to the measured pose of the stop tick, the arm brakes
    toward it (device_runtime: ordinary charged ticks, at most PROTECTIVE_STOP_BRAKE_S; it may not reach rest),
    then the running command ends early and returns this as an error, and the episode continues.
    """


class SimDiverged(RuntimeError):
    """The physics state became non-finite: the simulation diverged. The runtime ends the episode with end_reason
    sim_diverged (no further tick runs); the process record and video up to that tick are finalized as usual."""


def reset_window(owner, q, dt):
    """Start the stop window at pose q, as if the arm had rested there for the whole window (episode start and
    every protective stop: the stop pose is the new target)."""
    import numpy as np
    n = max(1, round(PROTECTIVE_STOP_WINDOW_S / dt))
    owner._stop_window = deque([np.array(q, dtype=float)] * (n + 1), maxlen=n + 1)
    return owner._stop_window


def window_rates(owner, q, dt):
    """Mean joint speeds (rad/s) over the last PROTECTIVE_STOP_WINDOW_S, including this tick's pose q: each joint's
    net travel over the window divided by the window length. A single-tick reading jumps when fingers chatter in
    contact while the joint stays where it is (measured: J7 read 3 rad/s while it moved 5 mrad); the net
    travel does not. It is never larger than the fastest single-tick rate in the window, so every stop of this rule
    is also a stop of the per-tick rule, and a run without a per-tick stop is unchanged."""
    import numpy as np
    window = getattr(owner, '_stop_window', None) or reset_window(owner, q, dt)
    window.append(np.array(q, dtype=float))
    return np.abs(window[-1] - window[0]) / ((window.maxlen - 1) * dt)


def protective_stop(rates):
    """The ProtectiveStop for these mean joint speeds over the stop window (rad/s, seven joints)."""
    joint = max(range(len(rates)), key=lambda i: abs(float(rates[i])))
    return ProtectiveStop(f'protective stop: mean speed of joint {joint + 1} over the last {PROTECTIVE_STOP_WINDOW_S:g} s '
                          f'reached {abs(float(rates[joint])):.2f} rad/s '
                          f'(limit {PROTECTIVE_STOP_RAD_S} rad/s); the arm braked toward the stop pose for at most '
                          f'{PROTECTIVE_STOP_BRAKE_S:g} s (charged to the budget) and now holds that pose as its target (it '
                          'may not be fully at rest); the gripper keeps its state (a held part stays held); the rest of '
                          'this command was not executed; the episode continues. This is not a failure: nothing is '
                          'damaged, it is not scored, and the next command runs normally from the current pose')


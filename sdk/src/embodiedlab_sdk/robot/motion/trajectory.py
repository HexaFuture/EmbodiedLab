"""Timing and execution of joint paths: planned transits (move_hand, plan_return_to_start) and contact moves.

The planner is a EmbodiedLab dependency, not a benchmark tool; it plans with the held part's
geometry. This module never imports the simulator.
The benchmark transport exposes only state and fixed-tick joint controls.
"""
from __future__ import annotations

import math
from typing import Protocol

import numpy as np


class BenchmarkTransport(Protocol):
    def call(self, name: str, arguments: dict) -> dict: ...


class MotionPlanner(Protocol):
    def plan(self, *, start_joints, target_pose, attached_geometry) -> dict: ...


def retime_joint_path(path, physics_hz=240, speed_limit=.60, acceleration_limit=1.2):
    """Stop at each planner waypoint using a bounded minimum-jerk profile.

Only time-parameterizes joint-space segments already checked by the planner.
It cannot turn an unchecked IK waypoint sequence into a collision-free plan.
"""
    if physics_hz not in (240, 480) or isinstance(physics_hz, bool):
        raise ValueError("unsupported physics rate")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
           for v in (speed_limit, acceleration_limit)):
        raise ValueError("invalid motion limits")
    if not isinstance(path, (list, tuple)) or len(path) < 2:
        raise ValueError("planner must return at least two waypoints")
    for row in path:
        if len(row) != 7 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row):
            raise ValueError("invalid planner joint vector")
    targets = []
    for start, end in zip(path, path[1:]):
        distance = max(abs(a-b) for a, b in zip(start, end))
        duration = max(1/physics_hz, 1.875*distance/speed_limit,
                       math.sqrt(5.774*distance/acceleration_limit))
        ticks = math.ceil(duration*physics_hz)
        if len(targets) + ticks > 30*physics_hz:
            raise ValueError("motion exceeds a single benchmark command; replan a shorter transit")
        for tick in range(1, ticks+1):
            t = tick/ticks
            s = 10*t**3 - 15*t**4 + 6*t**5
            targets.append([a+s*(b-a) for a, b in zip(start, end)])
    return targets


def trajectory_velocities(start, targets, physics_hz):
    """Feed-forward from this path, independent of the device's old setpoint."""
    previous=list(start)
    result=[]
    for row in targets:
        result.append([(b-a)*physics_hz for a,b in zip(previous,row)])
        previous=row
    return result


class PlannedMotion:
    """Plans externally; executes against a version-bound benchmark state."""

    def __init__(self, transport: BenchmarkTransport, planner: MotionPlanner):
        self.transport = transport
        self.planner = planner

    def move(self, target_pose, attached_geometry=None):
        state = self.transport.call('get_robot_state', {})
        plan = self.planner.plan(start_joints=state['joint_positions_rad'],
                                 target_pose=target_pose, attached_geometry=attached_geometry)
        if plan.get('success') is not True or plan.get('collision_checked') is not True:
            raise RuntimeError('planner did not produce a collision-checked path')
        path = plan['joint_path']
        if not path or len(path[0]) != 7 or any(
                not math.isfinite(a) or abs(a-b) > 1e-6
                for a, b in zip(path[0], state['joint_positions_rad'])):
            raise RuntimeError('plan starts at a different robot state')
        targets = (retime_sampled_path(path, state['physics_hz']) if plan.get('sampled_path')
                   else retime_joint_path(path, state['physics_hz']))
        return self.transport.call('execute_joint_trajectory', {
            'expected_state_id': state['state_id'], 'targets_rad': targets,
            'velocities_rad_s':trajectory_velocities(state['joint_positions_rad'],targets,state['physics_hz'])})


def retime_sampled_path(path, physics_hz=240, speed_limit=.60, acceleration_limit=1.2):
    """Slow an already sampled collision-checked path without stopping at every
    tiny interpolation point. Bound discrete speed/acceleration including both
    zero-speed endpoints, while staying on the checked joint-space segments.
    """
    if physics_hz not in (240,480) or len(path)<2:
        raise ValueError('supported clock and sampled path required')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
           for v in (speed_limit, acceleration_limit)):
        raise ValueError('invalid motion limits')
    if any(len(row)!=7 or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in row) for row in path):
        raise ValueError('finite seven-joint path required')
    delta=[[b-a for a,b in zip(left,right)] for left,right in zip(path,path[1:])]
    changes=[[b-a for a,b in zip(left,right)] for left,right in zip([[0.]*7]+delta,delta+[[0.]*7])]
    ticks=math.ceil(max(1,max(abs(v) for row in delta for v in row)*physics_hz/speed_limit,
                         max(abs(v) for row in changes for v in row)*physics_hz**2/acceleration_limit))
    if ticks*len(delta)>30*physics_hz:
        # Linear interpolation introduces one-tick velocity jumps at the
        # endpoints/corners. Uniformly slowing every segment to satisfy that
        # worst jump can make an ordinary CuRobo path take minutes. A bounded
        # stop-at-waypoint profile follows exactly the same checked segments,
        # with zero corner velocity and no geometric shortcut.
        return retime_joint_path(path, physics_hz, speed_limit, acceleration_limit)
    return [[a+(b-a)*i/ticks for a,b in zip(left,right)]
            for left,right in zip(path,path[1:]) for i in range(1,ticks+1)]


def minimum_jerk(t):
    t = np.clip(t, 0., 1.)
    return (10*t**3-15*t**4+6*t**5,
            30*t**2-60*t**3+30*t**4,
            60*t-180*t**2+120*t**3)


def transit_targets(path, physics_hz):
    """Cumulative-distance minimum-jerk profile over the whole path, without duplicate waypoints."""
    path = np.asarray(path, float)
    if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError('finite seven-joint planner path required')
    if physics_hz not in (240, 480):
        raise ValueError('unsupported physics clock')
    path = path[np.r_[True, np.max(np.abs(np.diff(path, axis=0)), axis=1) > 1e-9]]
    if len(path) < 2:
        return np.empty((0, 7))
    distance = np.r_[0., np.cumsum(np.max(np.abs(np.diff(path, axis=0)), axis=1))]
    duration = max(2., 1.875*distance[-1]/.35)
    count = int(np.ceil(duration*30))*(physics_hz//30)
    progress = minimum_jerk(np.arange(1, count+1)/count)[0]*distance[-1]
    return np.column_stack([np.interp(progress, distance, path[:, j]) for j in range(7)])


def transit(transport, path, state):
    """Execute a planned path with transit_targets, then let the arm settle for 0.4 s."""
    targets = transit_targets(path, state['physics_hz'])
    # Transport has a bounded packet size. Splitting does not retime, insert
    # stops, or change a target. Encoder feedback remains state-versioned.
    limit = state['physics_hz']*30
    for offset in range(0, len(targets), limit):
        state = transport.call('execute_joint_trajectory', {
            'expected_state_id': state['state_id'],
            'targets_rad': targets[offset:offset+limit].tolist()})
    return transport.call('wait', {'seconds': 12/30})

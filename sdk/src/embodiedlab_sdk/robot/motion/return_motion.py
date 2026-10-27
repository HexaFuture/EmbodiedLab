"""Explicit SDK planning back to the initial public hand pose.

The initial pose is a destination, not a saved trajectory. Every call plans
against current scene geometry; execution uses the existing task controller.
The destination is the initial posture too: the planner solves the hand pose on
any arm branch, so a plan that ends on another branch (shoulder or elbow turned
far from the initial joints) is planned again, and the plan ending nearest the
initial joints is used. An arm parked on another branch can lie over apparatus.
The planner keeps the arm near its current posture, so after many transits the
start hand pose is reached with the elbow swivelled ever further; on arrival the
arm is brought back to the initial joints in joint space, high above the table:
the hand stays near the start pose and every elbow/wrist frame stays well above
the robot base plane along the whole move (checked by forward kinematics).
"""
import math
import numpy as np
from .trajectory import retime_joint_path, retime_sampled_path, trajectory_velocities, transit
from .kinematics import rotation_vector
from .payload import payload_clearance


class ReturnMotion:
    def __init__(self, owner):
        self.owner = owner
        state, _, world = owner.hand.pose()
        self.world = world.copy()
        self.joints = np.asarray(state['joint_positions_rad'], float)

    BRANCH_RAD = .8          # largest shoulder/elbow joint change still counted as the initial posture
    ATTEMPTS = 4
    RESTORE_RAD = .05        # posture difference left as it is
    HAND_DRIFT_M = .20       # largest hand excursion from the start pose during a posture restore
    CLEARANCE_M = .30        # elbow, wrist and hand frames stay this far above the robot base plane

    def _restore_path(self, joints):
        """Straight joint-space path to the initial joints, or None when it would leave the free space above."""
        kin = self.owner.hand.kinematics
        start_hand = kin.forward(self.joints)[:3, 3]
        path = np.linspace(np.asarray(joints, float), self.joints, 40)
        for q in path:
            transform = np.eye(4)
            for index, (angle, (parent, child, axis)) in enumerate(zip(q, kin.transforms)):
                rot = np.eye(4); i, j = ((1, 2), (2, 0), (0, 1))[axis]; c, s = math.cos(angle), math.sin(angle)
                rot[i, i] = rot[j, j] = c; rot[i, j] = -s; rot[j, i] = s
                transform = transform@parent@rot@child
                if index >= 3 and transform[2, 3] < self.CLEARANCE_M:
                    return None
            hand = (transform@kin.tip)[:3, 3]
            if hand[2] < self.CLEARANCE_M or np.linalg.norm(hand-start_hand) > self.HAND_DRIFT_M:
                return None
        return path

    def plan(self, execute=False):
        if type(execute) is not bool:
            raise ValueError('execute must be boolean')
        owner = self.owner
        state, _, _ = owner.hand.pose()
        payload = owner.attached_geometry(state)
        target = owner.hand.target_pose(self.world[:3, 3], self.world[:3, :3])
        best = None
        for _ in range(self.ATTEMPTS):
            plan = owner.planner.plan(start_joints=state['joint_positions_rad'],
                                      target_pose=target, attached_geometry=payload)
            if plan.get('success') is not True or plan.get('collision_checked') is not True:
                raise RuntimeError('no collision-checked return path; no movement')
            if state.get('gripper_holding') and (
                    not payload or plan.get('attached_geometry_checked') is not True):
                raise RuntimeError('return path must check the complete attached payload')
            candidate = np.asarray(plan.get('joint_path'), float)
            if candidate.ndim != 2 or candidate.shape[1] != 7 or len(candidate) < 2 or not np.isfinite(candidate).all():
                raise RuntimeError('invalid return path')
            if not np.allclose(candidate[0], state['joint_positions_rad'], atol=1e-6, rtol=0):
                raise RuntimeError('return path starts at different joints')
            branch = float(np.max(np.abs(candidate[-1][:4]-self.joints[:4])))
            if best is None or branch < best[0]:
                best = (branch, candidate)
            if branch <= self.BRANCH_RAD:
                break
        path = best[1]
        if owner.transport.call('get_robot_state', {})['state_id'] != state['state_id']:
            raise ValueError('state changed while planning; no movement')
        if owner.contact_hold and payload:
            payload_clearance(owner, path, payload)
        if execute:
            if owner.contact_hold:
                transit(owner.transport, path.tolist(), state)
            else:
                retime = retime_sampled_path if plan.get('sampled_path') else retime_joint_path
                targets = retime(path.tolist(), state['physics_hz'])
                owner.transport.call('execute_joint_trajectory', {
                    'expected_state_id': state['state_id'], 'targets_rad': targets,
                    'velocities_rad_s': trajectory_velocities(state['joint_positions_rad'], targets, state['physics_hz'])})
            arrival, _, actual = owner.hand.pose()
            owner.state_id = arrival['state_id']
            if (np.linalg.norm(actual[:3, 3]-self.world[:3, 3]) > .015 or
                    np.linalg.norm(rotation_vector(self.world[:3, :3]@actual[:3, :3].T)) > .08):
                raise RuntimeError('return tracking failed; reobserve')
            restored = False
            current = np.asarray(arrival['joint_positions_rad'], float)
            if np.max(np.abs(current-self.joints)) > self.RESTORE_RAD:
                restore = self._restore_path(current)
                if restore is not None:
                    if owner.contact_hold:
                        transit(owner.transport, restore.tolist(), arrival)
                    else:
                        targets = retime_joint_path(restore.tolist(), arrival['physics_hz'])
                        owner.transport.call('execute_joint_trajectory', {
                            'expected_state_id': arrival['state_id'], 'targets_rad': targets,
                            'velocities_rad_s': trajectory_velocities(current.tolist(), targets, arrival['physics_hz'])})
                    arrival, _, actual = owner.hand.pose()
                    owner.state_id = arrival['state_id']
                    if (np.linalg.norm(actual[:3, 3]-self.world[:3, 3]) > .015 or
                            np.linalg.norm(rotation_vector(self.world[:3, :3]@actual[:3, :3].T)) > .08):
                        raise RuntimeError('posture restore tracking failed; reobserve')
                    restored = True
        return dict(success=True, executed=execute, planner='curobo',
                    destination='initial public hand pose', target_pose=target,
                    waypoint_count=len(path),
                    **({'initial_posture_restored': restored} if execute else {}))

"""The hand pose by forward kinematics of the public robot calibration (get_robot_pose and the T1 motion tools)."""
import numpy as np
from .kinematics import RobotKinematics


def quaternion_from_matrix(r):
    # Stable symmetric eigendecomposition, returning wxyz.
    from scipy.spatial.transform import Rotation
    x,y,z,w=Rotation.from_matrix(r).as_quat()
    return [float(w),float(x),float(y),float(z)]


class Hand:
    def __init__(self,participant):
        self.owner=participant
        self.kinematics=RobotKinematics(participant.task['device_calibration'])
        self.jaw=np.asarray(participant.task['device_calibration']['jaw_center_hand_m'])

    def pose(self):
        state=self.owner.transport.call('get_robot_state',{})
        base=self.kinematics.forward(state['joint_positions_rad'])
        world=self.kinematics.base@base
        return state,base,world

    def public_pose(self):
        state,base,world=self.pose()
        return {'state_id':state['state_id'],'hand_pose_base_wxyz_xyz':quaternion_from_matrix(base[:3,:3])+base[:3,3].tolist(),
                'hand_pose_world_wxyz_xyz':quaternion_from_matrix(world[:3,:3])+world[:3,3].tolist(),
                'jaw_center_world_m':(world[:3,3]+world[:3,:3]@self.jaw).tolist(),
                'source':'EmbodiedLab forward kinematics from public robot calibration'}

    def target_pose(self,position_world,orientation_world):
        target=np.eye(4);target[:3,3]=position_world;target[:3,:3]=orientation_world
        target=np.linalg.inv(self.kinematics.base)@target
        return quaternion_from_matrix(target[:3,:3])+target[:3,3].tolist()

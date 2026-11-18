"""EmbodiedLab planning client. Geometry is supplied by public perception/calibration.

No access to benchmark USD, rigid-body handles or evaluation state. The service
is an optional EmbodiedLab deployment, never a prerequisite of the benchmark image.
"""
import json
import math
import urllib.request


class CuRoboPlanner:
    def __init__(self, url, scene_provider, *, timeout=90):
        self.url = url.rstrip('/')
        self.scene_provider = scene_provider
        self.timeout = timeout
        self.kinematics = None

    def plan(self, *, start_joints, target_pose, attached_geometry):
        # target_pose is [qw,qx,qy,qz,x,y,z] in the public robot base frame.
        if len(target_pose) != 7 or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in target_pose):
            raise ValueError('finite base-frame pose required')
        if abs(sum(x*x for x in target_pose[:4])-1) > .002:
            raise ValueError('normalized target quaternion required')
        obstacles = self.scene_provider()
        if not isinstance(obstacles, list) or not obstacles:
            raise ValueError('publicly observed or calibrated collision scene required')
        spheres = [] if attached_geometry is None else attached_geometry
        if not isinstance(spheres, list) or len(spheres) > 96:
            raise ValueError('payload must be up to 96 hand-frame spheres')
        for row in spheres:
            if len(row) != 4 or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in row) or row[3] <= 0:
                raise ValueError('invalid payload collision sphere')
        body = {'start_joint_positions': start_joints, 'goal_pose_wxyz_xyz': target_pose,
                'obstacles': obstacles, 'payload_spheres_hand': spheres,
                'max_attempts': 6, 'timeout': 25., 'enable_graph': True}
        request = urllib.request.Request(self.url+'/motion_plan',
            data=json.dumps(body, allow_nan=False).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        if result.get('success') is not True:
            raise RuntimeError('CuRobo did not find a path: '+str(result.get('status')))
        if result.get('payload_collision_checked') is not True or result.get('payload_sphere_count') != len(spheres):
            raise RuntimeError('planner did not check the requested payload')
        if result.get('obstacle_count') != len(obstacles):
            raise RuntimeError('planner did not check the supplied obstacle scene')
        if self.kinematics is not None:
            import numpy as np
            from .kinematics import quaternion_matrix
            endpoint=self.kinematics.forward(result['waypoints'][-1])
            if (np.linalg.norm(endpoint[:3,3]-np.asarray(target_pose[4:]))>.015
                    or np.linalg.norm(endpoint[:3,:3]-quaternion_matrix(target_pose[:4]))>.15):
                raise RuntimeError('planner endpoint differs from public robot calibration; no motion executed')
        return {'success': True, 'collision_checked': True,
                'attached_geometry_checked': True, 'sampled_path': True,
                'joint_path': result['waypoints']}

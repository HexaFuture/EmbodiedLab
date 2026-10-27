"""EmbodiedLab participant: public visual targeting and basic device execution.

The optional EmbodiedLab native geometry provider supplies planning/contact geometry
through a host-only integration hook. It never supplies task truth or scores.
Lifecycle credentials are absent from the Agent-facing tool API.
"""
import base64
import hashlib
import math
import numpy as np
from ..motion.trajectory import PlannedMotion
from ..motion.curobo_planner import CuRoboPlanner
from ..motion.perception import VisualTargeting, unproject
from ..motion.contact import ContactController
from ..motion.hand import Hand, quaternion_from_matrix
from ..motion.payload import held_spheres
from ..motion.return_motion import ReturnMotion
from .tools import PRIMITIVES, family


class Participant:
    def __init__(self, transport, *, sam_url='', curobo_url='', geometry_provider=None):
        self.transport = transport
        self.geometry_provider = geometry_provider
        self.frames = {}
        self.state_id = None
        self.scene = None
        self.scene_state = None
        self.task = transport.task()
        self.targeting = VisualTargeting(lambda:self.state_id, sam_url)
        self.planner = CuRoboPlanner(curobo_url, self.collision_scene)
        self.motion = PlannedMotion(transport, self.planner)
        self.sam_url, self.curobo_url = sam_url, curobo_url
        self.contact = ContactController(transport, self.task['device_calibration'])
        self.planner.kinematics = self.contact.kinematics
        self.hand = Hand(self)
        # Benches whose fingers hold a part by contact: transits plan with the held part's geometry.
        # The pendulum and rail benches have no contact hold.
        self.contact_hold = family(self.task['task_id']) not in ('pendulum', 'coupled')
        self.return_motion = ReturnMotion(self)

    def geometry_snapshot(self):
        if self.geometry_provider is None:
            raise RuntimeError('EmbodiedLab native geometry adapter unavailable; no unprotected fallback')
        value = self.geometry_provider()
        state = self.transport.call('get_robot_state', {})
        if value['physics_tick'] != state['physics_tick']:
            raise ValueError('geometry and robot state differ')
        return value

    def attached_geometry(self, state=None):
        state = state or self.transport.call('get_robot_state', {})
        if not state.get('gripper_holding'):
            return None
        if not self.contact_hold:
            raise ValueError('held payload has no participant geometry; register it before transit')
        return held_spheres(self)

    def collision_scene(self):
        if self.geometry_provider is not None:
            calibration = self.contact.kinematics.base
            inverse = calibration[:3, :3].T
            quat = quaternion_from_matrix(inverse)
            snapshot = self.geometry_snapshot()
            if snapshot.get('oriented_boxes'):
                # A tilting bench: world-aligned bounds of a raised ramp would fill the space above it.
                return [{'name': 'scene_'+str(i), 'dims': (np.asarray(b['size'])+.002).tolist(),
                         'pose': [*(inverse@(np.asarray(b['centre'])-calibration[:3, 3])).tolist(),
                                  *quaternion_from_matrix(inverse@np.asarray(b['axes']))]}
                        for i, b in enumerate(snapshot['oriented_boxes'])]
            boxes = snapshot['world_boxes']
            return [{'name': 'scene_'+str(i), 'dims': (np.asarray(b['hi'])-b['lo']+.002).tolist(),
                     'pose': [*(inverse@((np.asarray(b['lo'])+b['hi'])/2-calibration[:3, 3])).tolist(), *quat]}
                    for i, b in enumerate(boxes)]
        if self.scene is None or self.scene_state != self.transport.call('get_robot_state', {})['state_id']:
            raise ValueError('EmbodiedLab collision adapter unavailable or explicit scene stale')
        return self.scene

    def hand_pose(self, joints):
        """Forward kinematics of the public robot calibration: the hand (ee_link panda_hand) pose in the robot base and
        world frames, and the jaw and fingertip centres in the world frame."""
        kinematics = self.hand.kinematics
        base = kinematics.forward(joints)
        world = kinematics.base @ base
        calibration = self.task['device_calibration']
        point = lambda key: (world[:3, 3] + world[:3, :3] @ np.asarray(calibration[key], float)).tolist()
        return {'hand_pose_base_wxyz_xyz': quaternion_from_matrix(base[:3, :3]) + base[:3, 3].tolist(),
                'hand_pose_world_wxyz_xyz': quaternion_from_matrix(world[:3, :3]) + world[:3, 3].tolist(),
                'jaw_center_world_m': point('jaw_center_hand_m'),
                'fingertip_center_world_m': point('fingertip_center_hand_m')}

    def with_hand_pose(self, state):
        if isinstance(state, dict) and len(state.get('joint_positions_rad') or ()) == 7:
            return state | self.hand_pose(state['joint_positions_rad'])
        return state

    def solve_ik(self, target_pose, frame='base', seed_joints=None):
        """Kinematics only: joints for a hand pose. Does not move the robot, check collisions or plan a path."""
        from ..motion.kinematics import quaternion_matrix
        pose = np.asarray(target_pose, float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError('target_pose must be [qw,qx,qy,qz,x,y,z]')
        if frame not in ('base', 'world'):
            raise ValueError("frame must be 'base' or 'world'")
        target = np.eye(4)
        target[:3, :3] = quaternion_matrix(pose[:4] / np.linalg.norm(pose[:4])) if abs(np.linalg.norm(pose[:4]) - 1) <= .001 \
            else quaternion_matrix(pose[:4])
        target[:3, 3] = pose[4:]
        kinematics = self.hand.kinematics
        if frame == 'world':
            target = np.linalg.inv(kinematics.base) @ target
        current = self.transport.call('get_robot_state', {})['joint_positions_rad']
        if seed_joints is not None:
            seed = np.asarray(seed_joints, float)
            if seed.shape != (7,) or not np.isfinite(seed).all():
                raise ValueError('seed_joints must be seven finite joint angles')
        from ..motion.kinematics import LOWER, UPPER
        seeds = [s for s in (seed_joints, current) if s is not None]
        seeds += [[0, -.785, 0, -2.356, 0, 1.571, .785], (LOWER + UPPER) / 2]
        # Then fixed spread-out restarts (same every call): a local solver can stall far from a reachable pose.
        seeds += list(np.random.default_rng(7).uniform(LOWER + .1, UPPER - .1, size=(16, 7)))
        joints, position, rotation = kinematics.solve(target, seeds)
        result = {'converged': joints is not None, 'frame': frame,
                  'position_residual_m': position, 'orientation_residual_deg': math.degrees(rotation),
                  'joint_lower_rad': LOWER.tolist(), 'joint_upper_rad': UPPER.tolist(),
                  'source': 'kinematics of the public robot calibration; no collision check, no motion'}
        if joints is not None:
            result['joint_positions_rad'] = joints
        return result

    def observation(self):
        value = self.transport.call('get_observation', {})
        self.state_id = value['state_id']
        self.frames = {}
        images, refs = [], []
        for f in value['frames']:
            for key in ('rgb','depth'):
                blob = base64.b64decode(f[key]['data'], validate=True)
                if hashlib.sha256(blob).hexdigest()!=f[key]['sha256']:
                    raise ValueError('camera transport hash mismatch')
            import cv2
            rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(f['rgb']['data']),np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            w,h = f['resolution']
            depth = np.frombuffer(base64.b64decode(f['depth']['data']),dtype='<u2').reshape(h,w)*f['depth']['unit_m']
            metadata = self.targeting.register(rgb,depth,f['intrinsics'],f['world_from_camera_ros'],f['camera'])
            # SDK cache identifiers are distinct from immutable benchmark evidence.
            self.frames[metadata['rgb_ref']] = self.targeting.frames[metadata['rgb_ref']]
            refs.append(metadata | {'source_rgb_ref': f['rgb_ref'], 'state_id': self.state_id})
            images.append({'type':'image','mime_type':'image/png','encoding':'base64','data':f['rgb']['data']})
        public = {'episode_id':value['episode_id'], 'state_id':self.state_id,
                'physics_tick':value['physics_tick'],'robot':self.with_hand_pose(value['robot']),'rgb_refs':refs}
        if 'evidence_id' in value:
            public['evidence_id']=value['evidence_id']   # bound to this state; its images are the views above
        return {'__embodiedlab_tool_result__': public,
                '__embodiedlab_tool_images__': images}

    def call(self,name,args):
        if name=='get_observation':
            if args: raise ValueError('get_observation takes no arguments')
            return self.observation()
        if name=='solve_ik':
            return self.solve_ik(**args)
        if name not in PRIMITIVES:
            # Bench tools are forwarded unchanged; any physical or instrument command may invalidate perception.
            result=self.transport.call(name,args)
            if name=='get_robot_state':result=self.with_hand_pose(result)
            if isinstance(result,dict) and 'state_id' in result:self.state_id=result['state_id']
            return result
        state=self.transport.call('get_robot_state',{})
        self.state_id=state['state_id']
        if name=='plan_return_to_start':
            return self.return_motion.plan(**args)
        if name=='localize_rgbd_point':
            frame=self.targeting.fresh(args['rgb_ref'])
            point=args['point_xy']
            if len(point)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in point):raise ValueError('finite [x,y] required')
            x,y=map(int,point);h,w=frame['depth'].shape
            if not 0<=x<w or not 0<=y<h:raise ValueError('pixel outside image')
            mask=np.zeros((h,w),bool);mask[max(0,y-1):min(h,y+2),max(0,x-1):min(w,x+2)]=True
            xyz=unproject(mask,frame['depth'],frame['intrinsics'],frame['pose'])
            if not len(xyz):raise ValueError('selected pixel has no valid depth')
            return {'world_point_m':np.median(xyz,axis=0).tolist(),'state_id':self.state_id,
                    'source':'participant_rgbd_estimate','sample_count':len(xyz)}
        if name=='segment_sam3_point_prompt':
            if not self.sam_url:raise RuntimeError('EmbodiedLab SAM service not configured; public benchmark remains available')
            return self.targeting.segment_sam3_point_prompt(**args)
        if name=='inspect_mask_image':
            return self.targeting.inspect_mask_image(**args)
        if name=='get_robot_pose':
            if args:raise ValueError('get_robot_pose takes no arguments')
            return self.hand.public_pose()
        if name=='mask_geometry':
            record=self.targeting.masks[args['mask_id']]
            frame=self.targeting.fresh(record['rgb_ref'])
            xyz=unproject(record['mask'],frame['depth'],frame['intrinsics'],frame['pose'])
            if len(xyz)<4:raise ValueError('insufficient masked depth')
            return {'state_id':self.state_id,'world_min_m':xyz.min(axis=0).tolist(),
                    'world_max_m':xyz.max(axis=0).tolist(),'world_median_m':np.median(xyz,axis=0).tolist(),
                    'point_count':len(xyz),'source':'visible RGB-D points; hidden surfaces are not certified'}
        if name=='translate_contact':
            try:return self.contact.translate(args['delta_base_m'])
            finally:self.scene_state=None;self.state_id=None
        if name=='move_hand':
            if not self.curobo_url:raise RuntimeError('EmbodiedLab CuRobo service not configured')
            payload=self.attached_geometry(state)
            try:return self.motion.move(args['target_pose'],payload)
            finally:self.scene_state=None;self.state_id=None
        raise ValueError('unknown EmbodiedLab participant tool')

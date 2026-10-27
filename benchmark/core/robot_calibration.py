"""Public robot CAD and installation calibration; never object-state truth."""
import numpy as np


def public_robot_calibration(sim):
    from pxr import UsdPhysics
    def joint_frame(joint, suffix):
        position = getattr(joint, 'GetLocalPos'+suffix+'Attr')().Get()
        rotation = getattr(joint, 'GetLocalRot'+suffix+'Attr')().Get()
        transform = sim.Gf.Matrix4d(1)
        transform.SetRotate(sim.Gf.Quatd(rotation))
        transform.SetTranslateOnly(sim.Gf.Vec3d(position))
        return np.asarray(transform).T.tolist()
    joints = {p.GetName(): p for p in sim.stage.Traverse()
              if p.IsA(UsdPhysics.RevoluteJoint) and str(p.GetPath()).startswith('/World/Franka/')}
    chain = []
    for index in range(1, 8):
        joint = UsdPhysics.RevoluteJoint(joints['panda_joint'+str(index)])
        chain.append({'axis': str(joint.GetAxisAttr().Get()),
                      'parent_to_joint': joint_frame(joint, '0'),
                      'child_to_joint': joint_frame(joint, '1')})
    first = UsdPhysics.RevoluteJoint(joints['panda_joint1'])
    last = UsdPhysics.RevoluteJoint(joints['panda_joint7'])
    base = sim._transform(str(first.GetBody0Rel().GetTargets()[0]))
    last_body = sim._transform(str(last.GetBody1Rel().GetTargets()[0]))
    hand = sim._transform('/World/Franka/panda_hand')
    fingers = [sim._transform('/World/Franka/'+name).ExtractTranslation()
               for name in ('panda_leftfinger','panda_rightfinger')]
    tcp = hand.GetInverse().Transform((fingers[0]+fingers[1])/2)
    # Lowest point of the finger geometry along the hand's approach axis (robot CAD, not object state).
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    to_hand = hand.GetInverse()
    reach = 0.0
    for name in ('panda_leftfinger', 'panda_rightfinger'):
        box = cache.ComputeWorldBound(sim.stage.GetPrimAtPath('/World/Franka/'+name))
        lo, hi = box.GetRange().GetMin(), box.GetRange().GetMax()
        for corner in [sim.Gf.Vec3d(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]:
            reach = max(reach, float(to_hand.Transform(box.GetMatrix().Transform(corner))[2]))
    out = {'robot_model': 'franka_panda', 'ee_link': 'panda_hand',
            'world_from_robot_base': np.asarray(base).T.tolist(),
            'joint_chain': chain,
            'last_link_to_hand': np.asarray(hand * last_body.GetInverse()).T.tolist(),
            'jaw_center_hand_m': list(tcp),
            'fingertip_center_hand_m': [float(tcp[0]), float(tcp[1]), reach]}
    import collision_reflex
    if collision_reflex.ENABLED:
        out['collision_reflex'] = collision_reflex.public_collision_behaviour()
    if getattr(sim, 'cameras', None):
        out['cameras'] = public_camera_calibration(sim, hand)
    return out


HAND_PATH = '/World/Franka/panda_hand'


def public_camera_calibration(sim, hand):
    """The fixed calibration of the three cameras: resolution and intrinsics of each, the world pose of the two
    fixed views and the pose of the wrist camera on the hand (ee_link). The same for every observation of the episode;
    the wrist camera moves with the hand, so its world pose is world_from_hand @ hand_from_camera."""
    world_from_hand = np.asarray(hand).T
    views = {}
    for camera in sim.cameras:
        path = str(camera.prim_path)
        width, height = camera.get_resolution()
        world_from_camera = np.linalg.inv(np.asarray(camera.get_view_matrix_ros(device='cpu'), float))
        entry = {'resolution': [int(width), int(height)],
                 'intrinsics': np.asarray(camera.get_intrinsics_matrix(device='cpu'), float).tolist()}
        if path.startswith(HAND_PATH + '/'):
            entry['mounted_on'] = 'panda_hand'
            entry['hand_from_camera'] = (np.linalg.inv(world_from_hand) @ world_from_camera).tolist()
        else:
            entry['mounted_on'] = 'world'
            entry['world_from_camera'] = world_from_camera.tolist()
        views[path.rsplit('/', 1)[-1]] = entry
    return {'frame_convention': ('ROS optical camera frame: x right, y down, z forward along the optical axis; '
                                 'pixel (u, v) = (fx x/z + cx, fy y/z + cy) with intrinsics [[fx,0,cx],[0,fy,cy],[0,0,1]], '
                                 'u to the right and v down from the top-left image corner. 4x4 transforms map column '
                                 'vectors, p_world = world_from_camera @ p_camera, metres; hand_from_camera is relative '
                                 'to the ee_link (panda_hand) frame, as last_link_to_hand. No depth image is provided.'),
            'views': views}

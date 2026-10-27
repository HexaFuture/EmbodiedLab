"""Collision geometry of a held part for cuRobo transit (move_hand, plan_return_to_start).

The bench's host geometry gives the corners of the part between the fingers; their world bounds are covered with
spheres and carried in the hand frame. A planned path is checked again, densely, against the bench's boxes.
Nothing here reaches the Agent but a refusal to move.
"""
import numpy as np
from .collision import sphere_box_collision, cover_box


def held_spheres(owner):
    """Hand-frame spheres covering the held part (at most 96, the planner's capacity)."""
    state, _, hand = owner.hand.pose()
    if not state.get('gripper_holding'):
        raise ValueError('hold a payload before placement')
    geometry = owner.geometry_snapshot()
    world = np.asarray(geometry['payload_corners_world'], float)
    if world.shape != (8, 3) or not np.isfinite(world).all():
        raise RuntimeError('held geometry unavailable; no motion')
    # Cover the world-aligned bounds and transform those spheres into the hand frame.
    spheres = cover_box(world.min(axis=0), world.max(axis=0))
    spheres[:, :3] = (spheres[:, :3]-hand[:3, 3])@hand[:3, :3]
    return spheres.tolist()


def payload_clearance(owner, path, spheres):
    """Independent dense check of the held part along a joint path, in addition to cuRobo."""
    spheres = np.asarray(spheres, float)
    kinematics = owner.hand.kinematics
    boxes = [{'lo': np.asarray(b['lo'])-.001, 'hi': np.asarray(b['hi'])+.001}
             for i, b in enumerate(owner.geometry_snapshot()['world_boxes'])]
    for start, end in zip(np.asarray(path)[:-1], np.asarray(path)[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(end-start))/.01)))
        for u in np.linspace(0, 1, count+1):
            hand = kinematics.base@kinematics.forward(start+u*(end-start))
            moved = spheres.copy(); moved[:, :3] = spheres[:, :3]@hand[:3, :3].T+hand[:3, 3]
            if sphere_box_collision(moved, boxes):
                raise RuntimeError('path of the held object is not clear; no motion')

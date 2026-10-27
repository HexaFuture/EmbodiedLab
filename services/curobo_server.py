"""cuRobo motion planning service for EmbodiedLab (Franka Panda, collision-aware, payload-aware).

    python curobo_server.py --host 127.0.0.1 --port 18768

Each request replaces the obstacle world and the attached payload geometry; requests are planned one at a time.
"""
from __future__ import annotations

import argparse
import asyncio
import threading
import time

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_PAYLOAD_SPHERES = 256   # attached-object sphere slots reserved in the robot model

app = FastAPI(title="EmbodiedLab payload-aware cuRobo")
lock = threading.Lock()
planner = tensor_args = None


class Cuboid(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    dims: list[float] = Field(min_length=3, max_length=3)
    pose: list[float] = Field(min_length=7, max_length=7)

    @model_validator(mode="after")
    def valid(self):
        if not np.isfinite(self.dims + self.pose).all() or min(self.dims) <= 0:
            raise ValueError("invalid obstacle geometry")
        if abs(np.linalg.norm(self.pose[3:]) - 1) > 1e-3:
            raise ValueError("obstacle quaternion must be normalized")
        return self


class MotionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_joint_positions: list[float] = Field(min_length=7, max_length=7)
    goal_pose_wxyz_xyz: list[float] = Field(min_length=7, max_length=7)
    obstacles: list[Cuboid] = Field(min_length=1, max_length=128)
    # [x,y,z,r] expressed in panda_hand, which owns attached_object at identity.
    payload_spheres_hand: list[list[float]] = Field(default_factory=list, max_length=MAX_PAYLOAD_SPHERES)
    max_attempts: int = Field(default=6, ge=1, le=20)
    timeout: float = Field(default=25, gt=0, le=60)
    enable_graph: bool = True

    @model_validator(mode="after")
    def valid(self):
        if not np.isfinite(self.start_joint_positions + self.goal_pose_wxyz_xyz).all():
            raise ValueError("nonfinite robot state")
        if abs(np.linalg.norm(self.goal_pose_wxyz_xyz[:4]) - 1) > 1e-3:
            raise ValueError("goal quaternion must be normalized")
        if len({o.name for o in self.obstacles}) != len(self.obstacles):
            raise ValueError("duplicate obstacle names")
        for sphere in self.payload_spheres_hand:
            if len(sphere) != 4 or not np.isfinite(sphere).all() or sphere[3] <= 0:
                raise ValueError("invalid payload sphere")
        return self


def initialize():
    global planner, tensor_args
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.geom.types import WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    tensor_args = TensorDeviceType()
    data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    data["kinematics"]["ee_link"] = "panda_hand"
    data["kinematics"]["extra_collision_spheres"]["attached_object"] = MAX_PAYLOAD_SPHERES
    robot = RobotConfig.from_dict(data, tensor_args=tensor_args)
    world = WorldConfig.from_dict({"cuboid": {"warmup_floor": {
        "dims": [2, 2, 0.1], "pose": [0, 0, -0.1, 1, 0, 0, 0]}}})
    cfg = MotionGenConfig.load_from_robot_config(
        robot, world, tensor_args=tensor_args,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        collision_cache={"obb": 128}, use_cuda_graph=True,
        position_threshold=0.005, rotation_threshold=0.05,
        num_ik_seeds=32, num_trajopt_seeds=4, num_graph_seeds=4,
        interpolation_dt=0.02, trajopt_dt=0.25, trajopt_tsteps=32,
        interpolation_steps=2000, collision_activation_distance=0.02)
    planner = MotionGen(cfg)
    planner.warmup(enable_graph=True, warmup_js_trajopt=False)


def plan(req):
    import torch
    from curobo.geom.types import WorldConfig
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    with lock:
        # Clear on every call (including empty gripper), and also on exceptions.
        planner.detach_spheres_from_robot()
        try:
            planner.world_coll_checker.clear_cache()
            planner.update_world(WorldConfig.from_dict({"cuboid": {
                o.name: {"dims": o.dims, "pose": o.pose} for o in req.obstacles}}))
            if req.payload_spheres_hand:
                spheres = tensor_args.to_device(np.zeros((MAX_PAYLOAD_SPHERES, 4), dtype=np.float32))
                spheres[:, 3] = -10
                spheres[:len(req.payload_spheres_hand)] = tensor_args.to_device(req.payload_spheres_hand)
                planner.attach_spheres_to_robot(sphere_tensor=spheres)
            start = JointState.from_position(tensor_args.to_device([req.start_joint_positions]))
            pose = req.goal_pose_wxyz_xyz
            goal = Pose(position=tensor_args.to_device([pose[4:]]), quaternion=tensor_args.to_device([pose[:4]]))
            cfg = MotionGenPlanConfig(max_attempts=req.max_attempts, timeout=req.timeout,
                enable_graph=req.enable_graph, enable_graph_attempt=1 if req.enable_graph else None,
                enable_finetune_trajopt=True)
            planner.reset(reset_seed=False)
            begin = time.monotonic()
            result = planner.plan_single(start, goal, cfg)
            torch.cuda.synchronize()
            success = bool(result.success.item())
            points = result.get_interpolated_plan().position.cpu().numpy() if success else np.empty((0, 7))
            return {"success": success, "waypoints": points.tolist(), "status": str(result.status),
                "num_waypoints": len(points), "plan_time_ms": (time.monotonic() - begin) * 1000,
                "payload_collision_checked": True, "payload_sphere_count": len(req.payload_spheres_hand),
                "obstacle_count": len(req.obstacles)}
        finally:
            planner.detach_spheres_from_robot()


@app.post("/motion_plan")
async def motion_plan(req: MotionRequest):
    if planner is None:
        from fastapi import HTTPException
        raise HTTPException(503, "planner initializing")
    return await asyncio.to_thread(plan, req)


@app.get("/health")
def health():
    return {"status": "ready" if planner is not None else "initializing",
            "capabilities": ["payload_spheres_hand", "replace_world_each_request"]}


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18768)
    args = parser.parse_args()
    initialize()
    uvicorn.run(app, host=args.host, port=args.port, ws="none")

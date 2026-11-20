"""Per-instance apparatus placement (layout jitter).

The instance record carries a short list of candidate placements
{dx_m, dy_m, yaw_deg}. The whole apparatus is moved rigidly: a translation in
the table plane and a yaw about a vertical axis through the apparatus'
footprint centre. Physical parameters are untouched. The first candidate for
which every key point stays inside the camera views that saw it without jitter
(with a pixel margin) and within robot reach is used; the choice is recorded
privately. Legacy task ids never call this module.

USD note: the placement is prepended to the root prim's xformOpOrder, so
world = T(pivot + d) · Rz(yaw) · T(-pivot) · (authored root transform).
Joints whose body0 is the world keep world-frame local poses and must be
moved explicitly (`retarget_world_joints`).
"""
from __future__ import annotations

import math

import numpy as np

CAMERA_MARGIN_PX = 8
REACH_LIMIT_M = 0.80          # Franka shoulder-to-wrist-flange reach is ~0.855 m
SHOULDER_HEIGHT_M = 0.333


class Placement:
    def __init__(self, dx_m=0.0, dy_m=0.0, yaw_deg=0.0, pivot_xy=(0.0, 0.0)):
        self.dx, self.dy, self.yaw_deg = float(dx_m), float(dy_m), float(yaw_deg)
        self.pivot = np.array([float(pivot_xy[0]), float(pivot_xy[1]), 0.0])
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        self.R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.d = np.array([self.dx, self.dy, 0.0])

    def point(self, p):
        """Apparatus-frame (unjittered world) point -> jittered world point."""
        return self.R @ (np.asarray(p, float) - self.pivot) + self.pivot + self.d

    def direction(self, v):
        return self.R @ np.asarray(v, float)

    def inverse_point(self, p):
        return self.R.T @ (np.asarray(p, float) - self.pivot - self.d) + self.pivot

    def inverse_direction(self, v):
        return self.R.T @ np.asarray(v, float)

    def quat_wxyz(self):
        half = math.radians(self.yaw_deg) / 2
        return (math.cos(half), 0.0, 0.0, math.sin(half))

    def record(self, index=None):
        return {"dx_m": self.dx, "dy_m": self.dy, "yaw_deg": self.yaw_deg,
                "pivot_xy_m": self.pivot[:2].tolist(), "candidate_index": index}


IDENTITY = Placement()


def footprint_pivot(stage, root_path):
    """Centre of the root's world-aligned bounding box in the table plane."""
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    box = cache.ComputeWorldBound(stage.GetPrimAtPath(root_path)).ComputeAlignedRange()
    if box.IsEmpty():
        raise RuntimeError(f"cannot place empty apparatus {root_path}")
    centre = (box.GetMin() + box.GetMax()) / 2
    return float(centre[0]), float(centre[1])


def apply_to_root(stage, root_path, placement):
    """Prepend the rigid placement to the root prim's transform stack."""
    from pxr import Gf, UsdGeom
    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(root_path))
    existing = xformable.GetOrderedXformOps()
    post = xformable.AddTranslateOp(opSuffix="layoutPost")
    yaw = xformable.AddRotateZOp(opSuffix="layoutYaw")
    pre = xformable.AddTranslateOp(opSuffix="layoutPre")
    post.Set(Gf.Vec3d(*(placement.pivot + placement.d)))
    yaw.Set(placement.yaw_deg)
    pre.Set(Gf.Vec3d(*(-placement.pivot)))
    xformable.SetXformOpOrder([post, yaw, pre, *existing])


def retarget_world_joints(stage, root_path, placement):
    """Move world-anchored joint frames (empty body0 or body1) with the apparatus."""
    from pxr import Gf, Usd, UsdPhysics
    q = placement.quat_wxyz()
    rot = Gf.Quatf(q[0], Gf.Vec3f(*q[1:]))
    moved = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        for rel, pos_attr, rot_attr in ((joint.GetBody0Rel(), joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
                                        (joint.GetBody1Rel(), joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr())):
            if rel.GetTargets():
                continue
            pos = pos_attr.Get() or Gf.Vec3f(0)
            pos_attr.Set(Gf.Vec3f(*placement.point(tuple(pos))))
            rot_attr.Set(rot * (rot_attr.Get() or Gf.Quatf(1)))
            moved.append(str(prim.GetPath()))
    return moved


def camera_views(stage, camera_paths, resolution=512):
    """Pinhole projectors for USD cameras (square pixels, square image), before physics start."""
    from pxr import UsdGeom
    cache = UsdGeom.XformCache()
    views = []
    for path in camera_paths:
        prim = stage.GetPrimAtPath(path)
        camera = UsdGeom.Camera(prim)
        focal = float(camera.GetFocalLengthAttr().Get())
        aperture = float(camera.GetHorizontalApertureAttr().Get())
        world = np.array(cache.GetLocalToWorldTransform(prim)).T      # column-vector convention
        views.append((np.linalg.inv(world), focal / aperture * resolution, resolution))
    return views


def project(views, point):
    """Pixel coordinates of a world point in each camera, or None when behind it."""
    out = []
    p = np.append(np.asarray(point, float), 1.0)
    for camera_from_world, f_px, size in views:
        x, y, z, _ = camera_from_world @ p
        out.append(None if z >= -1e-6 else (size / 2 + f_px * x / -z, size / 2 - f_px * y / -z))
    return out


def inside(pixel, size, margin=CAMERA_MARGIN_PX):
    return pixel is not None and margin <= pixel[0] <= size - margin and margin <= pixel[1] <= size - margin


def choose(stage, candidates, pivot_xy, key_points, camera_paths, shoulder_xyz, roi_checks=()):
    """First candidate keeping every key point as visible as without jitter and within reach.

    key_points: {name: (apparatus-frame world point, needs_reach)}
    roi_checks: [(key point name, camera index, image size px, (x0, y0, x1, y1), margin px)]
        e.g. a key point must stay inside a fixed region of one camera image.
    Returns (Placement, report). Raises when no candidate qualifies.
    """
    views = camera_views(stage, camera_paths)
    roi_views = {(i, size): camera_views(stage, [camera_paths[i]], size)[0] for _, i, size, _, _ in roi_checks}
    shoulder = np.asarray(shoulder_xyz, float)
    baseline = {name: [inside(px, v[2]) for px, v in zip(project(views, p), views)]
                for name, (p, _) in key_points.items()}
    rejected = []
    for index, candidate in enumerate(candidates):
        placement = Placement(candidate["dx_m"], candidate["dy_m"], candidate["yaw_deg"], pivot_xy)
        problems = []
        for name, (p, needs_reach) in key_points.items():
            q = placement.point(p)
            seen = [inside(px, v[2]) for px, v in zip(project(views, q), views)]
            lost = [i for i, (was, now) in enumerate(zip(baseline[name], seen)) if was and not now]
            if lost:
                problems.append(f"{name} leaves camera {lost}")
            for key, i, size, (x0, y0, x1, y1), margin in roi_checks:
                if key == name:
                    px = project([roi_views[(i, size)]], q)[0]
                    if px is None or not (x0 + margin <= px[0] <= x1 - margin and y0 + margin <= px[1] <= y1 - margin):
                        problems.append(f"{name} leaves its image region")
            if needs_reach and np.linalg.norm(q - shoulder) > max(REACH_LIMIT_M, np.linalg.norm(np.asarray(p) - shoulder) + 1e-3):
                problems.append(f"{name} out of reach")
        if not problems:
            return placement, {"chosen": index, "rejected": rejected}
        rejected.append({"index": index, "problems": problems})
    raise RuntimeError(f"no admissible apparatus placement: {rejected}")


def _bbox(stage, path):
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    box = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    if box.IsEmpty():
        raise RuntimeError(f"no geometry at {path}")
    return np.array(box.GetMin()), np.array(box.GetMax())


def _shoulder(stage, robot_path):
    from pxr import UsdGeom
    base = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(robot_path)).ExtractTranslation())
    return base + np.array([0.0, 0.0, SHOULDER_HEIGHT_M])


def _write(private_dir, record):
    import json
    from pathlib import Path
    (Path(private_dir) / "layout.json").write_text(json.dumps(record, indent=2))


def hooke_problem(stage, asset):
    """Key points, pivot and checks for the Hooke bench (computed before any placement)."""
    root = asset.APPARATUS_PATH
    lo, hi = _bbox(stage, root + "/Scale")
    mid = (lo + hi) / 2
    keys = {"ruler_top": ((mid[0], mid[1], hi[2]), False), "ruler_bottom": ((mid[0], mid[1], lo[2]), False)}
    lo, hi = _bbox(stage, asset.TRAY_PATH)
    keys["tray"] = ((lo + hi) / 2, True)
    lo, hi = _bbox(stage, root + "/Spring")
    keys["spring_top"] = (((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]), False)
    for grams, path in asset.WEIGHTS.items():
        lo, hi = _bbox(stage, path)
        keys[f"weight_{grams}"] = ((lo + hi) / 2, True)
    return dict(root=root, key_points=keys, pivot_xy=footprint_pivot(stage, root),
                camera_paths=list(asset.CAMERA_PATHS), shoulder_xyz=_shoulder(stage, asset.ROBOT_PATH),
                roi_checks=[])


def pendulum_problem(stage, descriptors, root, camera_paths, robot_path):
    """Pendulum bench: pendula, gates, timers and their buttons move together."""
    keys = {}
    for d in descriptors:
        keys[d["id"] + "_bob"] = (d["rest_bob_center"], True)
        keys[d["id"] + "_pivot"] = (d["pivot"], False)
        keys[d["id"] + "_gate"] = (d["gate_ray_origin"], False)
        keys[d["id"] + "_timer"] = (d["timer_center"], False)
    return dict(root=root, key_points=keys, pivot_xy=footprint_pivot(stage, root),
                camera_paths=list(camera_paths), shoulder_xyz=_shoulder(stage, robot_path), roi_checks=[])


def torsion_problem(stage, asset, lab):
    """Torsion bench: turntable, rim fixtures, photogate and both objects on the table move together."""
    keys = {}
    for name, path, reach in (("turntable", asset.TURNTABLE + "/Platter", True),
                              ("push_post", asset.TURNTABLE + "/PushPost", True),
                              ("photogate", asset.ROOT + "/GateUpperArm", False),
                              ("rec_key", asset.PANEL_PATH + "/Keypad/Key_rec", True),
                              ("recorder_display", asset.PANEL_PATH + "/RecorderDisplay", False),
                              ("damper_lever", asset.LEVER_PATH + "/Carriage", True),
                              ("storage_box", asset.BOX_PATH + "/Body", True)):
        lo, hi = _bbox(stage, path)
        keys[name] = ((lo + hi) / 2, reach)
    for key in asset.OBJECTS:
        path = asset.ROOT + "/" + "".join(part.title() for part in key.split("_"))
        lo, hi = _bbox(stage, path)
        keys[key] = ((lo + hi) / 2, True)
    return dict(root=asset.ROOT, key_points=keys, pivot_xy=footprint_pivot(stage, asset.ROOT),
                camera_paths=list(lab.CAMERA_PATHS), shoulder_xyz=_shoulder(stage, lab.ROBOT_PATH), roi_checks=[])


def circuit_problem(stage, asset, lab):
    """Circuit bench: panel, sliders, black box and logger move together."""
    keys = {}
    for name in asset.SLIDER_X:
        lo, hi = _bbox(stage, asset.carriage_path(name) + "/Post")
        keys[name.lower() + "_post"] = ((lo + hi) / 2, True)
    panel_board = getattr(asset, "PANEL", {}).get("board")      # led_planck: the LED board; battery_emf: the cell
    board = ("CellBoard" if panel_board.get("kind") == "cell" else "LedBoard") if panel_board else "BlackBox"
    for name, reach in ((board, False), ("Logger", False)):
        lo, hi = _bbox(stage, asset.ROOT + "/" + name)
        keys[name] = ((lo + hi) / 2, reach)
    return dict(root=asset.ROOT, key_points=keys, pivot_xy=footprint_pivot(stage, asset.ROOT),
                camera_paths=list(lab.CAMERA_PATHS), shoulder_xyz=_shoulder(stage, lab.ROBOT_PATH), roi_checks=[])


def chem_problem(stage, asset, lab):
    """Spectrophotometry bench: rack, dispensing station, spectrophotometer, buttons and cuvettes move together."""
    keys = {}
    for name, path, reach in [(sid, path, True) for sid, path in asset.seat_paths(stage).items()] + \
            [(bid, path, True) for bid, path in asset.button_paths(stage).items()] + [("display", asset.DISPLAY, False)]:
        lo, hi = _bbox(stage, path)
        keys[name] = ((lo + hi) / 2, reach)
    return dict(root=asset.ROOT, key_points=keys, pivot_xy=footprint_pivot(stage, asset.ROOT),
                camera_paths=list(lab.CAMERA_PATHS), shoulder_xyz=_shoulder(stage, lab.ROBOT_PATH), roi_checks=[])


def solve(stage, problem, candidates):
    return choose(stage, candidates, problem["pivot_xy"], problem["key_points"], problem["camera_paths"],
                  problem["shoulder_xyz"], problem["roi_checks"])


def _apply(stage, problem, candidates, private_dir):
    placement, report = solve(stage, problem, candidates)
    apply_to_root(stage, problem["root"], placement)
    moved = retarget_world_joints(stage, problem["root"], placement)
    _write(private_dir, placement.record(report["chosen"]) | {"rejected": report["rejected"], "world_joints_moved": moved})
    return placement


def place_torsion(stage, candidates, private_dir, asset, lab):
    return _apply(stage, torsion_problem(stage, asset, lab), candidates, private_dir)


def place_circuit(stage, candidates, private_dir, asset, lab):
    return _apply(stage, circuit_problem(stage, asset, lab), candidates, private_dir)


def place_chem(stage, candidates, private_dir, asset, lab):
    return _apply(stage, chem_problem(stage, asset, lab), candidates, private_dir)


def place_hooke(stage, candidates, private_dir, asset):
    return _apply(stage, hooke_problem(stage, asset), candidates, private_dir)


def place_pendulum(stage, descriptors, candidates, private_dir, root, camera_paths, robot_path):
    placement = _apply(stage, pendulum_problem(stage, descriptors, root, camera_paths, robot_path), candidates, private_dir)
    for d in descriptors:
        for key in ("pivot", "rest_bob_center", "gate_ray_origin", "timer_center"):
            d[key] = tuple(float(v) for v in placement.point(d[key]))
        d["gate_ray_direction"] = tuple(float(v) for v in placement.direction(d["gate_ray_direction"]))
        d["swing_axis_world"] = tuple(float(v) for v in placement.direction((1.0, 0.0, 0.0)))
    return placement

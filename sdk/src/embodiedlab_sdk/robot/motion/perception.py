"""Agent-selected RGB point -> SAM3 masks; same-frame RGB-D backprojection of masks and pixels."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import urllib.request
import urllib.error
import uuid

import cv2
import numpy as np


def post_json(url, value, timeout=90):
    request = urllib.request.Request(url, data=json.dumps(value, allow_nan=False).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read(16 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"service HTTP {exc.code}: {detail}") from exc
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("service response too large")
    return json.loads(data)


def png(rgb):
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoder failed")
    return encoded.tobytes()


def image_content(rgb):
    return {"type": "image", "mime_type": "image/png", "encoding": "base64",
            "data": base64.b64encode(png(rgb)).decode("ascii")}


def unproject(mask, depth, intrinsics, camera_to_world):
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth, dtype=float)
    k = np.asarray(intrinsics, dtype=float)
    pose = np.asarray(camera_to_world, dtype=float)
    if depth.shape != mask.shape or mask.ndim != 2:
        raise ValueError("mask/depth dimensions differ")
    if k.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(pose).all():
        raise ValueError("invalid camera calibration")
    valid = mask & np.isfinite(depth) & (depth > 0.02) & (depth < 5)
    ys, xs = np.where(valid)
    if len(xs) < 4:
        raise ValueError("insufficient valid masked depth")
    camera = (np.linalg.inv(k) @ np.vstack([xs, ys, np.ones_like(xs)])) * depth[ys, xs]
    world = pose @ np.vstack([camera, np.ones(len(xs))])
    return world[:3].T


class VisualTargeting:
    def __init__(self, frame, sam_url):
        self.frame = frame
        self.sam_url = sam_url.rstrip("/")
        self.frames = {}
        self.masks = {}

    def register(self, rgb, depth, intrinsics, camera_to_world, camera):
        ref = uuid.uuid4().hex
        rgb = np.asarray(rgb, dtype=np.uint8).copy()
        if depth is None or np.asarray(depth).shape != rgb.shape[:2]:
            raise RuntimeError("same-frame depth not ready")
        self.frames[ref] = {"rgb": rgb, "depth": np.asarray(depth).copy(),
                            "intrinsics": np.asarray(intrinsics).copy(),
                            "pose": np.asarray(camera_to_world).copy(),
                            "frame": self.frame(), "camera": camera,
                            "rgb_sha256": hashlib.sha256(png(rgb)).hexdigest()}
        # Only recent frames can generate new plans. Old images/evidence are
        # retained separately by the episode audit, not in this planning cache.
        while len(self.frames) > 12:
            del self.frames[next(iter(self.frames))]
        self.masks = {key: value for key, value in self.masks.items() if value["rgb_ref"] in self.frames}
        return {"rgb_ref": ref, "rgb_sha256": self.frames[ref]["rgb_sha256"], "camera": camera}

    def fresh(self, ref):
        if ref not in self.frames or self.frames[ref]["frame"] != self.frame():
            raise ValueError("stale RGB reference; inspect a fresh get_observation image")
        return self.frames[ref]

    def segment_sam3_point_prompt(self, rgb_ref, point_xy):
        frame = self.fresh(rgb_ref)
        if not isinstance(point_xy, (tuple, list)) or len(point_xy) != 2:
            raise ValueError("point_xy must be the Agent-selected [x,y] pixel")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point_xy):
            raise ValueError("point pixels must be finite numbers")
        x, y = point_xy
        h, w = frame["rgb"].shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("point outside RGB image")
        result = post_json(self.sam_url + "/segment_point", {
            "image_base64": base64.b64encode(png(frame["rgb"])).decode("ascii"),
            "point_coords": [x, y],
        })
        self.fresh(rgb_ref)
        shape = result["masks_shape"]
        dtype = result["masks_dtype"]
        if dtype not in ("bool", "uint8", "float32") or len(shape) != 3 or shape[1:] != [h, w] or not 1 <= shape[0] <= 8:
            raise ValueError("invalid SAM3 mask format")
        raw = base64.b64decode(result["masks_base64"], validate=True)
        if len(raw) != int(np.prod(shape)) * np.dtype(dtype).itemsize:
            raise ValueError("invalid SAM3 mask byte count")
        arrays = np.frombuffer(raw, dtype=dtype).reshape(shape)
        if not np.isfinite(arrays).all():
            raise ValueError("nonfinite SAM3 mask")
        scores = result["scores"]
        if len(scores) != len(arrays):
            raise ValueError("mask/score counts differ")
        candidates = []
        visuals = []
        source = frame["rgb"].copy()
        cv2.drawMarker(source, (int(x), int(y)), (255, 180, 0), cv2.MARKER_CROSS, 12, 2)
        visuals.append(image_content(source))
        for index in sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True):
            score = float(scores[index])
            mask = arrays[index].astype(bool)
            if not math.isfinite(score) or score <= 0.05 or not mask[int(y), int(x)] or mask.sum() < 4:
                continue
            mask_id = uuid.uuid4().hex
            self.masks[mask_id] = {"rgb_ref": rgb_ref, "mask": mask.copy(), "point_xy": [x, y]}
            candidates.append({"mask_id": mask_id, "score": score, "pixels": int(mask.sum())})
            view = frame["rgb"].copy()
            view[mask] = (0.5 * view[mask] + 0.5 * np.array([0, 255, 80])).astype(np.uint8)
            cv2.drawMarker(view, (int(x), int(y)), (255, 180, 0), cv2.MARKER_CROSS, 12, 2)
            visuals.append(image_content(view))
            if len(candidates) == 3:
                break
        if not candidates:
            raise RuntimeError("SAM3 returned no valid mask containing the selected pixel")
        return {"__embodiedlab_tool_result__": {
            "rgb_ref": rgb_ref, "rgb_sha256": frame["rgb_sha256"],
            "agent_selected_pixel": [x, y], "mask_candidates": candidates,
        }, "__embodiedlab_tool_images__": visuals}

    def inspect_mask_image(self, mask_id, padding_px=32, scale=4):
        from .mask_inspection import mask_closeup
        if mask_id not in self.masks:
            raise ValueError("unknown mask; use SAM3 point prompt first")
        record = self.masks[mask_id]
        frame = self.fresh(record["rgb_ref"])
        metadata, images = mask_closeup(frame["rgb"], record["mask"],
                                        record["point_xy"], padding_px, scale)
        return {"__embodiedlab_tool_result__": {
            "mask_id": mask_id, "rgb_ref": record["rgb_ref"],
            "rgb_sha256": frame["rgb_sha256"], "state_id": frame["frame"],
            "agent_selected_pixel": list(record["point_xy"]),
            "pixels": int(record["mask"].sum()), "executed": False,
            **metadata,
        }, "__embodiedlab_tool_images__": [image_content(im) for im in images]}

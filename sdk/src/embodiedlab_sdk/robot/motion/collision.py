"""Shared conservative payload geometry for planning and trajectory validation."""
from __future__ import annotations

import itertools
import numpy as np

MAX_PAYLOAD_SPHERES = 96
PAYLOAD_MARGIN_M = 0.004


def cover_box(lo, hi, cell_size=0.018):
    """Cover the entire box, not just its surface, with circumscribed cell spheres."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
        raise ValueError("invalid payload bounds")
    counts = np.maximum(1, np.ceil((hi - lo) / cell_size).astype(int))
    # Larger payloads get coarser (larger, still circumscribed) cells within the planner capacity.
    while np.prod(counts) > MAX_PAYLOAD_SPHERES and cell_size < 0.2:
        cell_size *= 1.1
        counts = np.maximum(1, np.ceil((hi - lo) / cell_size).astype(int))
    if np.prod(counts) > MAX_PAYLOAD_SPHERES:
        raise ValueError("payload exceeds collision sphere capacity")
    sizes = (hi - lo) / counts
    centers = np.asarray(list(itertools.product(*[
        lo[i] + (np.arange(counts[i]) + 0.5) * sizes[i] for i in range(3)
    ])))
    return np.column_stack([centers, np.full(len(centers), np.linalg.norm(sizes) / 2 + PAYLOAD_MARGIN_M)])


def sphere_box_collision(spheres, boxes):
    """True when any sphere intersects any box. Names no box, sphere or clearance (results may reach the Agent)."""
    if not len(spheres):
        return False
    for box in boxes:
        nearest = np.clip(spheres[:, :3], box["lo"], box["hi"])
        clearance = np.linalg.norm(spheres[:, :3] - nearest, axis=1) - spheres[:, 3]
        if np.min(clearance) <= 0:
            return True
    return False

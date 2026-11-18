"""Robot side of the EmbodiedLab SDK, run inside the lab environment image.

This package provides:
- the environment adapter, participant and tool catalog of the benchmark (``embodiedlab``)
- the kinematics, planning, perception and motion helpers of the T1 tools (``motion``)

The SDK control plane never imports this package.
"""

from __future__ import annotations

__all__ = [
    "embodiedlab",
    "motion",
]

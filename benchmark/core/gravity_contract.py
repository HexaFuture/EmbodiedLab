"""Private benchmark contract between episode gravity and the physics context."""
import math


def verify_gravity(context, expected):
    expected = float(expected)
    if not math.isfinite(expected) or expected <= 0:
        raise ValueError("Episode gravity must be finite and positive")
    direction, magnitude = context.get_gravity()
    if (len(direction) != 3 or any(not math.isfinite(float(x)) for x in direction)
            or any(abs(float(x)-y) > 1e-6 for x, y in zip(direction, (0, 0, -1)))
            or not math.isfinite(float(magnitude))
            or not math.isclose(float(magnitude), expected, rel_tol=1e-6, abs_tol=1e-6)):
        raise RuntimeError("Physics gravity differs from private episode configuration")
    return {"direction": list(direction), "magnitude_m_s2": float(magnitude),
            "configured_m_s2": expected}


def configure_gravity(context, expected):
    if not math.isfinite(float(expected)) or float(expected) <= 0:
        raise ValueError("Episode gravity must be finite and positive")
    context.set_gravity(-float(expected))
    return verify_gravity(context, expected)

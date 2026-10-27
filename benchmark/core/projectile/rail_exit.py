"""Exact sub-tick rail exit of the launched ball (an integrator correction, not a launch rule).

On the rails the ball's motion is supported; past the rail ends it is free. A fixed-step engine ends the support only
on a tick boundary, so without a correction the range would depend on which tick the ball leaves on: a sawtooth in
compression with a period of a few mm at 240 Hz. In the tick where the ball centre passes the rail-end plane the
backend replaces the end-of-tick state by the exact free flight from the moment it left.
"""
import numpy as np

GRAVITY = np.array([0., 0., -9.81])
RAIL_CLEAR_U = .035          # ball centre this far past the rail ends: clear of the rail caps, rail contact back on


def exact_exit(p0, p1, v0, v1, dt, muzzle, axis):
    """Free-flight state at the end of a tick in which the ball centre passed the rail-end plane (u = 0).

    The tick's supported motion is taken as constant acceleration a = (v1 - v0)/dt through the two sampled positions
    (start velocity u0 = (p1 - p0)/dt - a dt/2). Exit at tau_e (u(tau_e) = 0), then gravity only for dt - tau_e. The
    returned velocity is the engine's step velocity (the mean over the coming step, the exact velocity minus g dt/2):
    the engine's position update p += v dt then follows the exact parabola at every later tick.
    Returns (position, step velocity, exit fraction, exit position, exit velocity)."""
    a = (v1 - v0) / dt
    u0 = (p1 - p0) / dt - a * dt / 2
    s0, su, sa = float((p0 - muzzle) @ axis), float(u0 @ axis), float(a @ axis)
    if abs(sa) < 1e-9:
        tau = -s0 / su
    else:                                            # root of s0 + su tau + sa tau^2 / 2 in (0, dt]
        disc = su * su - 2 * sa * s0
        tau = 2 * -s0 / (su + np.sqrt(max(disc, 0.)))
    tau = float(np.clip(tau, 0., dt))
    pe = p0 + u0 * tau + a * tau * tau / 2
    ue = u0 + a * tau
    rest = dt - tau
    position = pe + ue * rest + GRAVITY * rest * rest / 2
    return position, ue + GRAVITY * rest - GRAVITY * dt / 2, tau / dt, pe, ue

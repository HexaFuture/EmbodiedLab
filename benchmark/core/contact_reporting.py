"""Interpret physical contact manifolds without a minimum-force requirement."""
import math


def contact_flags(total_count, net_force_norm, robot_count=0):
    if (type(total_count) is not int or type(robot_count) is not int
            or not 0 <= total_count < 512 or not 0 <= robot_count <= total_count):
        raise RuntimeError('contact detail buffer exhausted or invalid')
    if not math.isfinite(net_force_norm) or net_force_norm < 0:
        raise RuntimeError('contact sensor force invalid')
    # PhysX may report a touching pair with zero normal impulse. Opposing
    # contacts can also cancel in the aggregate. Pair counts remain evidence
    # of contact in both cases; force magnitude is only a fallback.
    # robot_contact: any part of the robot (fingers or other links) touches
    # the pendulum. Any of them can start a swing, as with a real arm.
    return dict(robot_contact=robot_count > 0,
                any_contact=total_count > 0 or net_force_norm > 1e-5)

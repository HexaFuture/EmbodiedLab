"""torsion_centroid: torsion_inertia dynamics with the locating pin, the holed sample bar and the bored disc.

The evaluator's private truth of a mounting is `mount`: which object sits on the pin and through which hole.
"""
import math
import numpy as np
from torsion_backend import TorsionBackend


class CentroidBackend(TorsionBackend):
    def apparatus_module(self):
        import centroid_apparatus
        return centroid_apparatus

    def build_apparatus(self):
        return self.asset.build(self.stage, turntable_inertia_kgm2=self.dynamics['turntable_inertia_kgm2'],
                                ballast_fraction=self.dynamics['ballast_fraction'], imbalance=self.dynamics['imbalance'],
                                ballast_halfspan_m=self.dynamics['ballast_halfspan_m'], mark_rgb=self.dynamics['mark_rgb'])

    def frame_extra(self):
        """Adds the mounting as `mount` returns it now: [load, hole]."""
        return super().frame_extra() | {'mount': list(self.mount())}

    def mount(self):
        """(load, hole) of the turntable now (private; geometric poses only, never mass properties).

        ('empty', None): nothing presses on the platter and the robot does not touch the turntable. (key, hole): exactly one object
        counted on the turntable (backend.loads), flat (tilt <= MOUNT_TILT_DEG), its underside within MOUNT_LIFT_M of
        the platter top and the axis inside one of its bores: the sample's hole number (1-7) or 0 for the disc's centre
        bore. (None, None) otherwise (held, touched, more than one object, on the pin tip, tilted)."""
        asset = self.asset
        loads = self.loads()
        latest = self.history[-1] if self.history else None
        if not loads:
            present = latest is not None and any(latest['turntable_force'][k] > .02 for k in self.bodies)
            # Six levels (private record only): an object held or touched away from the platter no longer
            # unmounts the empty turntable; only an object pressing on it or the robot touching the turntable does.
            touched = latest is not None and latest.get('turntable_contact', False) is not False
            return (None, None) if (present or touched) else ('empty', None)
        if len(loads) != 1:
            return None, None
        key = next(iter(loads))
        if loads[key]['tilt_deg'] > asset.MOUNT_TILT_DEG:
            return None, None
        p, q = self.bodies[key].get_world_pose()
        top = float(self.turntable.get_world_pose()[0][2]) + asset.base.PLATTER_H / 2
        if float(p[2]) - top > asset.MOUNT_LIFT_M:
            return None, None
        w, x, y, z = map(float, q)
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        delta = self.axis_xy - np.asarray(p[:2], float)
        d = math.cos(yaw) * delta[0] + math.sin(yaw) * delta[1]
        t = -math.sin(yaw) * delta[0] + math.cos(yaw) * delta[1]
        reach = asset.HOLE_HALF_M + asset.MOUNT_SLACK_M
        holes = asset.HOLES.items() if key == 'unknown_object' else ((0, 0.0),)
        for number, hx in holes:
            if abs(d - hx) <= reach and abs(t) <= reach:
                return key, number
        return None, None

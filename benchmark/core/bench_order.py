"""Bench order: the loose items on the bench, how they stand, and whether the gripper holds one.

A loose item is a dynamic rigid body outside the robot that no physics joint ties to the apparatus when the
episode starts (cuvettes, beakers, weights, holders, probes...); spring-loaded keys, slider carriages and other
jointed parts are apparatus. For each loose item a snapshot gives where it is, how far the axis that pointed up at
the start is now tilted from vertical, and whether a joint ties it to the robot (a held part, see
hooke_physics.attach_grasp_joint). The episode takes the first snapshot before the first tool call, records
every change after a call (tipped over, righted, off the bench, back, picked up, set down) for the process record,
and seals the snapshot at submission for the evaluator's bench_in_order gate. The first snapshot is kept as
start_items, so the scorer can also say which items are back where they started (bench_restore, descriptive only).
Reading never moves anything.
"""
from __future__ import annotations

import math

ROBOT_PREFIX = '/World/Franka'
TABLE_TOP = '/World/LabTable/Top'
TIPPED_DEG = 45.0          # for the change log only; the evaluator applies its own limits to the sealed values
OFF_BENCH_M = 0.10


def _stage():
    import omni.usd
    return omni.usd.get_context().get_stage()


def _table_top_z(stage):
    from pxr import Usd, UsdGeom
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render']).ComputeWorldBound(
        stage.GetPrimAtPath(TABLE_TOP)).ComputeAlignedRange()
    return float(box.GetMax()[2])


def _joints(stage):
    """{body path: set of paths it is jointed to ('' = the world)} over the enabled joints of the stage."""
    from pxr import UsdPhysics
    links = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        enabled = joint.GetJointEnabledAttr().Get()
        if enabled is False:
            continue
        bodies = [str(t) for t in joint.GetBody0Rel().GetTargets()[:1]] or ['']
        bodies += [str(t) for t in joint.GetBody1Rel().GetTargets()[:1]] or ['']
        if len(bodies) == 2:
            links.setdefault(bodies[0], set()).add(bodies[1])
            links.setdefault(bodies[1], set()).add(bodies[0])
    return links


def _robot(path):
    return path == ROBOT_PREFIX or path.startswith(ROBOT_PREFIX + '/')


class BenchOrder:
    def __init__(self, stage=None):
        from pxr import Gf, UsdGeom, UsdPhysics
        self.Gf, self.UsdGeom = Gf, UsdGeom
        self.stage = stage or _stage()
        links = _joints(self.stage)
        self.items = []
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if _robot(path) or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            body = UsdPhysics.RigidBodyAPI(prim)
            if body.GetRigidBodyEnabledAttr().Get() is False or body.GetKinematicEnabledAttr().Get():
                continue
            if any(not _robot(other) for other in links.get(path, ())):
                continue                       # tied to the apparatus (or the world) from the start
            self.items.append(path)
        self.table_z = _table_top_z(self.stage)
        rotations = self._rotations()
        # The axis of each item that pointed up at the start, in the item's own frame.
        self.up_local = {p: r.GetInverse().TransformDir(Gf.Vec3d(0, 0, 1)) for p, r in rotations.items()}
        self.last = None
        self.start = None          # the first snapshot, taken before the first tool call
        self.changes = []

    def _rotations(self, cache=None):
        cache = cache or self.UsdGeom.XformCache()
        out = {}
        for path in self.items:
            m = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))
            out[path] = self.Gf.Transform(m).GetRotation()
        return out

    def snapshot(self):
        cache = self.UsdGeom.XformCache()
        links = _joints(self.stage)
        rotations = self._rotations(cache)
        items = {}
        for path in self.items:
            xyz = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(path)).ExtractTranslation()
            up = rotations[path].TransformDir(self.up_local[path]).GetNormalized()
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))
            items[path] = {'xyz_m': [round(float(c), 4) for c in xyz], 'tilt_deg': round(tilt, 2),
                           'below_table_top_m': round(self.table_z - float(xyz[2]), 4),
                           'held': any(_robot(other) for other in links.get(path, ()))}
        return {'table_top_z_m': round(self.table_z, 4), 'items': items}

    @staticmethod
    def _flags(item):
        return {'tipped': item['tilt_deg'] > TIPPED_DEG, 'off_bench': item['below_table_top_m'] > OFF_BENCH_M,
                'held': item['held']}

    def observe(self, tick, call):
        """After a tool call: log each item whose tipped / off-bench / held state changed since the last call."""
        now = self.snapshot()
        if self.start is None:
            self.start = now
        if self.last is not None:
            for path, item in now['items'].items():
                before, after = self._flags(self.last['items'][path]), self._flags(item)
                for flag in before:
                    if before[flag] != after[flag]:
                        self.changes.append({'tick': tick, 'call': call, 'item': path, flag: after[flag],
                                             'tilt_deg': item['tilt_deg'], 'xyz_m': item['xyz_m']})
        self.last = now
        return now

    def record(self):
        return {'items': list(self.items), 'table_top_z_m': round(self.table_z, 4), 'changes': list(self.changes),
                'last': self.last, 'start_items': self.start_items()}

    def start_items(self):
        """The items as they stood before the first tool call (same fields as a snapshot's items)."""
        return self.start['items'] if self.start is not None else None

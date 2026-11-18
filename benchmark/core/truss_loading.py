"""Truss bench collapse (truss_bridge v2): after the bridge failed on the loading frame (truss_frame) and the episode
was sealed at that tick, the bridge gives way on camera. Nothing here reaches the Agent or its score: the sealed record
is fixed at the failure, the robot can no longer act and nothing is reachable (the same status as after the seal).

The governing bar fails as in the tester (tension: it breaks in two; compression: it kinks upward and folds), the rope
keeps pulling C with the failure load and the bridge folds until C's travel ends (DROP_M, where the platform lands on
its catch blocks; truss_frame.slacken lets the platform fall).

Physics (post-seal only): each eye of a bar in the bridge is pinned to its pin by a revolute joint about the pin's
axis, where the eye lies now, so the bridge is the pin-jointed truss the evaluator analysed: A is pinned to the board,
B rides on a roller (a slide along the span), C and the loose pins of the bridge are free pins. The pins of the bridge
give way to free pins, twins built with the board (drawn alike, no colliders, no weight; hidden and still until then):
B's and C's stand where those fixed pins stand (A stays, no longer colliding); a loose pin's twin is put where the pin
stands and the pin itself leaves the scene, out of view, through the physics view (never an authored change to a moving
body: that rebuilds it from its stale authored pose). The twins are let go one physics step later, once they stand
there. The bars are BAR_MASS_FACTOR heavier (the joints converge with the load on 5 g bars). The rope is a distance
joint from C to a body on a slide in the load direction pulled with the failure load; the slide ends after DROP_M.
"""
from __future__ import annotations

import math

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

import chem_apparatus as C
from chem_apparatus import ROOT, _text
from torsion_apparatus import _box, _cylinder
import truss_bench as B
import truss_model as M

G = 9.81
PINS = ROOT + '/LoadPins'
ROPE = ROOT + '/LoadRope'
JOINTS = C.PHYSICS_ROOT + '/Loading'
STEEL, RED = (0.62, 0.63, 0.66), (0.85, 0.08, 0.06)

D = B.LOAD_Y
CX, CY = B.pin_xy(M.LOAD_PIN)
CABLE_Z = B.PIN_FOOT + B.PIN_H - 0.003   # where the rope body's distance joint takes C's twin (on its post)
DROP_M = 0.035                           # C's travel after the failure (the platform's fall to its catch blocks)
ROPE_L = 0.060                           # the rope body's distance from C in the load direction (not drawn)
FAIL_AFTER_S = 1.0 / 60                  # the governing bar fails this long after the joints are made
AFTER_S = 2.5                            # the collapse is shown this long after the failure
BAR_MASS_FACTOR = 20.0                   # bars' mass during the collapse (the joints converge with the load on 5 g bars)
SOLVER_ITERATIONS = 64
AWAY_XYZ = (0.0, -6.0, -3.0)             # world point, out of every camera's view, where a replaced loose pin goes
FIXED_PARTS = ('Collar/Collider', 'Post/Collider', 'Tip')


# --- build (with the board; invisible and still until the loading) ---------------------------------------------
def _quiet_body(prim, mass, kinematic=True):
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(kinematic)
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    body.CreateDisableGravityAttr(True)
    body.CreateSolverPositionIterationCountAttr(SOLVER_ITERATIONS)
    body.CreateSolverVelocityIterationCountAttr(16)
    body.CreateSleepThresholdAttr(0.0)
    body.CreateAngularDampingAttr(1.0)
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(float(mass))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-5, 1e-5, 1e-5))
    api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))


def _free_pin(stage, path, xy, colour, shank=False):
    """A free pin like a board pin (collar, post, tip, a loose pin's shank; frame at the collar's underside, set on the
    board top), drawn only: a quiet kinematic body, invisible."""
    pin = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(pin).SetTranslate(Gf.Vec3d(xy[0], xy[1], B.BOARD_H))
    UsdGeom.XformCommonAPI(pin).SetRotate(Gf.Vec3f(0, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    if shank:
        _cylinder(stage, path + '/Shank', B.SHANK_R, B.SHANK_L, (0, 0, -B.SHANK_L / 2), colour, collide=False)
    _cylinder(stage, path + '/Collar', B.COLLAR_R, B.COLLAR_H, (0, 0, B.COLLAR_H / 2), colour, collide=False)
    _cylinder(stage, path + '/Post', B.PIN_R, B.PIN_H, (0, 0, B.COLLAR_H + B.PIN_H / 2), colour, collide=False)
    tip = UsdGeom.Cone.Define(stage, path + '/Tip')
    tip.CreateRadiusAttr(B.PIN_R)
    tip.CreateHeightAttr(B.PIN_TIP)
    tip.CreateAxisAttr('Z')
    tip.CreateExtentAttr([Gf.Vec3f(-B.PIN_R, -B.PIN_R, -B.PIN_TIP / 2), Gf.Vec3f(B.PIN_R, B.PIN_R, B.PIN_TIP / 2)])
    UsdGeom.XformCommonAPI(tip).SetTranslate(Gf.Vec3d(0, 0, B.COLLAR_H + B.PIN_H + B.PIN_TIP / 2))
    tip.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    _quiet_body(pin.GetPrim(), 0.02)
    UsdGeom.Imageable(pin).MakeInvisible()
    return pin


def loose_twin(k):
    return f'{PINS}/Loose{k}'


def _euler_xyz(quat_wxyz):
    """XformCommonAPI rotateXYZ angles (degrees) of a rotation."""
    R = B._rotation(quat_wxyz)
    y = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    x = math.atan2(R[2, 1], R[2, 2])
    z = math.atan2(R[1, 0], R[0, 0])
    return Gf.Vec3f(math.degrees(x), math.degrees(y), math.degrees(z))


def build(stage):
    """The free pins (twins of the fixed pins B and C, and of each loose pin) and the rope body, all hidden and still."""
    UsdGeom.Xform.Define(stage, PINS)
    for name in M.FIXED:
        if name != 'A':
            _free_pin(stage, f'{PINS}/Pin{name}', B.pin_xy(name), RED if name == M.LOAD_PIN else STEEL)
    for k, xy in enumerate(B.stand_holes().values()):
        _free_pin(stage, loose_twin(k), xy, STEEL, shank=True)
    rope = UsdGeom.Xform.Define(stage, ROPE)
    UsdGeom.XformCommonAPI(rope).SetTranslate(Gf.Vec3d(CX, CY + D * ROPE_L, CABLE_Z))
    _quiet_body(rope.GetPrim(), 1.0)


# --- the loading ---------------------------------------------------------------------------------------------------
def _quat_conj(q):
    w, x, y, z = (float(v) for v in q)
    return Gf.Quatf(w, -x, -y, -z)


class Loading:
    """The collapse of the failed bridge (plan: the frozen bridge record {'capacity_N', 'governing', 'members'} and the
    bars failed before, 'failed'; load_n: the rope's pull at the failure). The backend calls apply() before each
    physics step and tick() after it; done once the bridge has failed and AFTER_S passed."""

    def __init__(self, backend, plan, load_n):
        self.b, self.plan = backend, plan
        self.stage = backend.stage
        self.load = float(load_n)
        self.t, self.failed_t, self.kink = 0.0, None, None
        self.log = {'capacity_N': float(plan['capacity_N']), 'load_N': round(self.load, 3)}

    @property
    def done(self):
        return self.failed_t is not None and self.t >= self.failed_t + AFTER_S

    def force_n(self):
        return self.load

    # set up --------------------------------------------------------------------------------------------------------
    def start(self):
        """Phase 1: heavier bars, the twins of the bridge's loose pins put where those pins stand and the pins sent out
        of the scene. The joints follow after one physics step (_connect, from the first tick)."""
        b, stage = self.b, self.stage
        table_z = b._table_z()
        positions, orientations = b.cuvettes.get_world_poses()
        positions, orientations = np.asarray(positions, float), np.asarray(orientations, float)
        self.pose = pose = {key: (positions[i], orientations[i]) for i, key in enumerate(b.ids)}
        failed = self.plan.get('failed', {})
        self.members = [(bar, tuple(pins)) for bar, pins in self.plan['members'] if bar not in failed]
        self.nodes = sorted({n for _, pins in self.members for n in pins})
        # the loose pin standing in each hole of the bridge (as the bench saw it at the seal)
        seated = {w[1]: pin for pin, w in b.where.items() if pin in b.pin_ids and w[0] == 'hole'}
        missing = [n for n in self.nodes if n not in M.FIXED and n not in seated]
        if missing:
            raise RuntimeError(f'no loose pin seen in holes {missing}')
        self.loose = {n: seated[n] for n in self.nodes if n not in M.FIXED}
        self.bar_index = [i for i, key in enumerate(b.ids) if key in b.bar_ids]
        # heavier bars through the physics view only (an authored change to a bar would rebuild it from its stale
        # authored pose); their inertia scales with them
        idx = np.asarray(self.bar_index, np.int32)
        b.cuvettes.set_masses(np.asarray(b.cuvettes.get_masses(indices=idx), np.float32).reshape(-1) * BAR_MASS_FACTOR,
                              indices=idx)
        if hasattr(b.cuvettes, 'set_inertias'):
            inertia = np.asarray(b.cuvettes.get_inertias(indices=idx), np.float32)
            b.cuvettes.set_inertias(inertia * BAR_MASS_FACTOR, indices=idx)
        self.log['speed_m_s'] = [[0.0, self._speed()]]
        self.node_body = {}                     # node -> (body path or None for the world, world pose (p, q))
        index = []
        for k, name in enumerate(sorted(self.loose)):
            pin = self.loose[name]
            p, q = pose[pin]
            twin = stage.GetPrimAtPath(loose_twin(k))
            api = UsdGeom.XformCommonAPI(twin)
            api.SetTranslate(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]) - table_z))
            api.SetRotate(_euler_xyz(q), UsdGeom.XformCommonAPI.RotationOrderXYZ)
            UsdGeom.Imageable(twin).MakeVisible()
            self.node_body[name] = (loose_twin(k), (p, q))
            index.append(b.ids.index(pin))
        if index:                               # the pins themselves leave: weightless, still, out of view
            idx = np.asarray(index, np.int32)
            n = len(index)
            if hasattr(b.cuvettes, 'disable_gravities'):
                b.cuvettes.disable_gravities(indices=idx)
            away = np.asarray(AWAY_XYZ, float) + np.stack([np.arange(n) * 0.1, np.zeros(n), np.zeros(n)], axis=1)
            b.cuvettes.set_world_poses(positions=away, orientations=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)), indices=idx)
            b.cuvettes.set_velocities(np.zeros((n, 6), np.float32), indices=idx)
        self.connected = False
        self.c_view = self.rope_view = None
        self.rope_y0 = CY + D * ROPE_L
        self.log.update(nodes=self.nodes, loose=self.loose, members=[[bar, list(p)] for bar, p in self.members])

    def _connect(self):
        """Phase 2: the twins let go, the fixed pins swapped, the eyes jointed to their pins, the roller and the rope."""
        b, stage = self.b, self.stage
        table_z = b._table_z()
        pose = self.pose
        UsdGeom.Scope.Define(stage, JOINTS)
        for path, _ in self.node_body.values():
            UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).GetKinematicEnabledAttr().Set(False)
        for name in self.nodes:
            if name not in M.FIXED:
                continue
            for part in FIXED_PARTS:            # the board's pin gives way to its twin (A: it stays, not colliding)
                self._collide(f'{ROOT}/Pin{name}{part}', False)
            at = np.array([*B.pin_xy(name), B.BOARD_H + table_z])
            if name == 'A':
                self.node_body[name] = (None, (at, (1.0, 0.0, 0.0, 0.0)))
                continue
            for part in ('Collar', 'Post', 'Tip', 'Look'):
                prim = stage.GetPrimAtPath(f'{ROOT}/Pin{name}{part}')
                if prim.IsValid():
                    UsdGeom.Imageable(prim).MakeInvisible()
            pin = stage.GetPrimAtPath(f'{PINS}/Pin{name}')
            UsdGeom.Imageable(pin).MakeVisible()
            UsdPhysics.RigidBodyAPI(pin).GetKinematicEnabledAttr().Set(False)
            self.node_body[name] = (f'{PINS}/Pin{name}', (at, (1.0, 0.0, 0.0, 0.0)))
        if 'B' in self.nodes:                   # the roller: B slides along the span
            self._slide(f'{JOINTS}/RollerB', f'{PINS}/PinB', (*B.pin_xy('B'), B.BOARD_H), 'X', -0.040, 0.040)
        self.joints = []
        for bar, pins in self.members:
            label = b.labels[bar]
            tail = B.tail_id(bar)
            eyes = B.eye_centres(*pose[bar], label, table_z, tail=pose[tail])
            first = min(pins, key=lambda n: math.dist(B.pin_xy(n), eyes[0][:2]))
            for eye, body, name in ((eyes[0], tail, first), (eyes[1], bar, next(n for n in pins if n != first))):
                self._pin_eye(f"{JOINTS}/{body}_{name.replace('-', 'm')}", name, eye[2] + table_z, b.paths[body], pose[body])
        # the rope: from C's cable eye to the rope body, which slides in the load direction until the hanger lands
        rope = UsdPhysics.DistanceJoint.Define(stage, JOINTS + '/Cable')
        rope.CreateBody0Rel().SetTargets([Sdf.Path(f'{PINS}/Pin{M.LOAD_PIN}')])
        rope.CreateBody1Rel().SetTargets([Sdf.Path(ROPE)])
        rope.CreateLocalPos0Attr(Gf.Vec3f(0, 0, CABLE_Z - B.BOARD_H))
        rope.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        rope.CreateMaxDistanceAttr(ROPE_L)
        rope.CreateExcludeFromArticulationAttr(True)
        lower, upper = (-DROP_M, 0.002) if D < 0 else (-0.002, DROP_M)
        self._slide(JOINTS + '/RopeSlide', ROPE, (CX, CY + D * ROPE_L, CABLE_Z), 'Y', lower, upper)
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(ROPE)).GetKinematicEnabledAttr().Set(False)
        self.connected = True
        self.log['connected_tick'] = int(self.b.tick_index)

    def _collide(self, path, on):
        prim = self.stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(on)

    def _slide(self, path, body, at, axis, lower, upper):
        joint = UsdPhysics.PrismaticJoint.Define(self.stage, path)
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(at[0], at[1], at[2] + self.b._table_z()))
        joint.CreateLocalRot0Attr(Gf.Quatf(1))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1))
        joint.CreateAxisAttr(axis)
        joint.CreateLowerLimitAttr(lower)
        joint.CreateUpperLimitAttr(upper)
        joint.CreateExcludeFromArticulationAttr(True)

    def _pin_eye(self, path, name, eye_z, body_path, body_pose):
        """A revolute joint about the pin's axis through the eye where it lies (eye_z: world height of the eye's
        mid-plane): A's to the world, the others to their free pin. Both frames world-aligned at the anchor on the pin's
        axis."""
        joint = UsdPhysics.RevoluteJoint.Define(self.stage, path)
        pin_body, (p0, q0) = self.node_body[name]
        p0 = np.asarray(p0, float)
        R0 = B._rotation(q0)
        axis = R0[:, 2]
        world = p0 + axis * (eye_z - p0[2]) / axis[2]
        if pin_body is None:
            joint.CreateLocalPos0Attr(Gf.Vec3f(*world))
            joint.CreateLocalRot0Attr(Gf.Quatf(1))
        else:
            joint.CreateBody0Rel().SetTargets([Sdf.Path(pin_body)])
            joint.CreateLocalPos0Attr(Gf.Vec3f(*(R0.T @ (world - p0))))
            joint.CreateLocalRot0Attr(_quat_conj(q0))
        position, quat = body_pose
        local = B._rotation(quat).T @ (world - np.asarray(position, float))
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
        joint.CreateLocalPos1Attr(Gf.Vec3f(*local))
        joint.CreateLocalRot1Attr(_quat_conj(quat))
        joint.CreateAxisAttr('Z')
        joint.CreateExcludeFromArticulationAttr(True)
        self.joints.append(path)

    # running ------------------------------------------------------------------------------------------------------
    def _views(self):
        if self.rope_view is None:
            self.rope_view = self.b._view(ROPE, 'truss_load_rope')
            self.c_view = self.b._view(f'{PINS}/Pin{M.LOAD_PIN}', 'truss_load_pin')
        return self.rope_view

    def apply(self):
        """Before a physics step: the hanger's mass on the rope body and its weight in the load direction."""
        if not self.connected:
            return
        view = self._views()
        view.set_masses(np.array([max(1.0, self.load / G)], dtype=np.float32))
        view.apply_forces(np.array([[0.0, D * self.force_n(), 0.0]], dtype=np.float32), is_global=True)

    def _speed(self):
        v = np.asarray(self.b.cuvettes.get_linear_velocities(), float)[self.bar_index]
        return round(float(np.max(np.linalg.norm(v, axis=1))), 4)

    def tick(self, dt):
        if not self.connected:
            self._connect()
            return
        self.t += dt
        if len(self.log['speed_m_s']) < 40 and int(round(self.t / dt)) % 12 == 0:
            self.log['speed_m_s'].append([round(self.t, 3), self._speed()])
        if self.failed_t is None and self.t >= FAIL_AFTER_S:
            self._fail()
        if self.kink is not None:
            bar, t0 = self.kink
            u = min(1.0, (self.t - t0) / self.b.asset.KINK_S)
            self.b.asset.bend_bar(self.stage, bar, u * self.b.asset.kink_deg(self.b.labels[bar]))
            if u >= 1.0:
                self._hinge(bar)
                self.kink = None

    def _fail(self):
        self.failed_t = self.t
        gov = self.plan.get('governing') or {}
        bar, mode = gov.get('bar'), gov.get('mode')
        self.log.update(failed_at_s=round(self.t, 3), failed_tick=int(self.b.tick_index), bar=bar, mode=mode)
        if bar is None:
            return
        if mode == 'tension':
            self.b.asset.break_bar(self.stage, bar)
        else:
            self.kink = (bar, self.t)

    def _hinge(self, bar):
        """A buckled bar folds: its joint becomes a hinge about the bar's width at the cut (as it lies, kinked)."""
        stage, path = self.stage, self.b.asset.bar_joint_path(bar)
        rot1 = UsdPhysics.FixedJoint(stage.GetPrimAtPath(path)).GetLocalRot1Attr().Get()
        stage.RemovePrim(path)
        joint = UsdPhysics.RevoluteJoint.Define(stage, path + 'Hinge')
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self.b.paths[B.tail_id(bar)])])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(self.b.paths[bar])])
        at = Gf.Vec3f(B.CUT_M, 0.0, B.BAR_T / 2)
        joint.CreateLocalPos0Attr(at)
        joint.CreateLocalPos1Attr(at)
        joint.CreateLocalRot0Attr(Gf.Quatf(1))
        joint.CreateLocalRot1Attr(rot1)
        joint.CreateAxisAttr('Y')
        joint.CreateExcludeFromArticulationAttr(True)

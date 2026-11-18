"""Adjuster knobs of the fiber coupling bench (fiber_coupling): a knurled thumbscrew knob on a revolute joint
with friction, its angle read from the physics every tick.

A knob is a dynamic rigid body (knurled grip and the exposed neck of its screw, both colliders) on a revolute joint to
the world at its mount; the joint's drive is a saturated damper, i.e. the screw's turning friction: at rest the knob
stays where it was left (nothing loads it: its centre of mass is on the axis), turned it takes FRICTION_NM to turn
(above SATURATION_DEG_S), and it stops when the turning stops. There is no spring and no detent. The knob turns without
end stops (the screw's travel is the bench model's: many turns, see fiber model); its collider turns with it.

Local frame of a knob: z along its axis, pointing out of the mount face (z = 0 on the face); the grip spans
GRIP_Z0..GRIP_Z0 + GRIP_L. Angle: right-handed about the outward axis (counter-clockwise seen from outside, looking at
the knob's end), unwrapped, 0 where it was built.

No SDK, no simulator handles outside the stage; KnobReader reads poses through Isaac's rigid prim views.
"""
from __future__ import annotations

import math

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

KNOB_R = 0.006                 # knurled grip radius (12 mm knob)
GRIP_L = 0.010                 # knurled length
NECK_R, NECK_L = 0.0022, 0.0022   # the screw's exposed neck between the mount face and the grip
FACE_GAP = 0.0004              # neck end clear of the mount face (no contact with the static mount)
GRIP_Z0 = FACE_GAP + NECK_L    # grip starts here above the mount face
CHAMFER = 0.0006
KNOB_MASS = 0.020              # knob with its screw [kg]
FRICTION_NM = 0.03             # turning torque of the screw against its spring preload [N m]
SATURATION_DEG_S = 0.2         # the friction drive saturates above this speed
# Rotational inertia: the knob's own (izz = m r^2 / 2 = 3.6e-7 kg m^2, no solver scaling). With it the
# grasp hold transmits a turn with a systematic lost motion of 0.45 deg per stroke (0.44-0.47, scatter 0.005; 0.50 with
# a grip 2 mm off the axis or 4 deg tilted), which the calibration model carries as a systematic term
# (fiber_model.GRIP_LAG_DEG). Measured: x10 leaves 0.03-0.05 deg, x100 none but breaks the hold at ~0.5 mm misfit.
INERTIA_SCALE = 1.0
KNURL_PITCH = 24
BLACK = (0.035, 0.036, 0.04)


def axis_rotation(axis, x_hint=(1.0, 0.0, 0.0)):
    """World rotation (3x3) whose z is `axis` and whose x is the component of x_hint across it."""
    z = np.asarray(axis, float)
    z = z / np.linalg.norm(z)
    x = np.asarray(x_hint, float) - np.dot(x_hint, z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, (0.0, 1.0, 0.0))
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _quat(R):
    """Gf.Quatf (w, x, y, z) of a world rotation matrix (column vectors)."""
    m = np.asarray(R, float)
    t = float(np.trace(m))
    if t > 0:
        r = 2 * math.sqrt(1 + t)
        w, x, y, z = r / 4, (m[2, 1] - m[1, 2]) / r, (m[0, 2] - m[2, 0]) / r, (m[1, 0] - m[0, 1]) / r
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        r = 2 * math.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
        v = [0.0, 0.0, 0.0]
        w = (m[k, j] - m[j, k]) / r
        v[i], v[j], v[k] = r / 4, (m[j, i] + m[i, j]) / r, (m[k, i] + m[i, k]) / r
        x, y, z = v
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return Gf.Quatf(w / n, Gf.Vec3f(x / n, y / n, z / n))


def _materials(stage, root):
    import visual_kit as V
    folder = V.texture_dir()
    knurl = V.knurl(folder, 'knurl_fiber', pitch=KNURL_PITCH)
    looks = {'knob': V.material(stage, root + '/Looks/KnobAnodized', BLACK, metallic=0.6, roughness=0.42,
                                normal=knurl),
             'neck': V.material(stage, root + '/Looks/ScrewSteel', (0.62, 0.63, 0.66), metallic=1.0, roughness=0.3),
             'end': V.material(stage, root + '/Looks/KnobEnd', BLACK, metallic=0.6, roughness=0.35)}
    physics = UsdShadeMaterial(stage, root + '/PhysicsMaterials/KnobAluminium', 0.6, 0.5)
    return looks, physics


def UsdShadeMaterial(stage, path, static, dynamic):
    from pxr import UsdShade
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    return mat


def _collider_cylinder(stage, path, radius, z0, z1, material):
    from pxr import UsdShade
    geom = UsdGeom.Cylinder.Define(stage, path)
    geom.CreateRadiusAttr(radius)
    geom.CreateHeightAttr(z1 - z0)
    geom.CreateAxisAttr('Z')
    geom.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -(z1 - z0) / 2), Gf.Vec3f(radius, radius, (z1 - z0) / 2)])
    UsdGeom.XformCommonAPI(geom).SetTranslate(Gf.Vec3d(0, 0, (z0 + z1) / 2))
    geom.CreateVisibilityAttr('invisible')
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim()).CreateCollisionEnabledAttr(True)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(material, materialPurpose='physics')
    return geom


def build_knob(stage, path, face_point, axis, joint_path, *, x_hint=(1.0, 0.0, 0.0), looks=None, physics=None,
               root=None, mass_kg=KNOB_MASS, inertia_scale=INERTIA_SCALE, friction_nm=FRICTION_NM):
    """A knob at `face_point` (world, on the mount face) turning about `axis` (world, out of the face). Returns its
    body path. Author before the simulation starts."""
    import visual_kit as V
    if looks is None:
        looks, physics = _materials(stage, root or path.rsplit('/', 1)[0])
    R = axis_rotation(axis, x_hint)
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in face_point]))
    xform.AddOrientOp().Set(_quat(R))
    # looks: neck, knurled grip with chamfered ends, slightly domed end face with a centre dimple
    z0, z1 = GRIP_Z0, GRIP_Z0 + GRIP_L
    V.define(stage, path + '/NeckLook', V.lathe([(0.0, FACE_GAP), (NECK_R, FACE_GAP), (NECK_R, z0 + 0.0002),
                                                 (0.0, z0 + 0.0002)], segments=32), looks['neck'], tile=0.004)
    side = V.lathe([(KNOB_R - CHAMFER, z0), (KNOB_R, z0 + CHAMFER), (KNOB_R, z1 - CHAMFER), (KNOB_R - CHAMFER, z1)],
                   segments=72)
    # knurl texture: u around the circumference, v along the axis (one diamond about 0.8 mm)
    st = []
    for f in side.faces:
        for i in f:
            p = side.P[i]
            st.append((math.atan2(p[1], p[0]) / (2 * math.pi) * 2 * math.pi * KNOB_R / 0.012, p[2] / 0.012))
    st = np.asarray(st)
    # repair the seam (u jumps by a full turn inside a face)
    k = 0
    turn = 2 * math.pi * KNOB_R / 0.012
    for f in side.faces:
        n = len(f)
        u = st[k:k + n, 0]
        if u.max() - u.min() > turn / 2:
            u[u < turn / 2] += turn
        st[k:k + n, 0] = u
        k += n
    V.define(stage, path + '/GripLook', side, looks['knob'], uv=st, crease_deg=60.0)
    V.define(stage, path + '/EndLook', V.lathe([(KNOB_R - CHAMFER, z1), (0.0012, z1 + 0.0003), (0.0008, z1 + 0.0001),
                                                (0.0, z1 + 0.0001)], segments=72), looks['end'], tile=0.01)
    V.define(stage, path + '/BaseLook', V.lathe([(0.0, z0), (KNOB_R - CHAMFER, z0)], segments=72), looks['end'],
             tile=0.01)
    _collider_cylinder(stage, path + '/GripCollider', KNOB_R, z0, z1, physics)
    _collider_cylinder(stage, path + '/NeckCollider', NECK_R, FACE_GAP, z0, physics)
    # dynamic body, centre of mass on the axis
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim()).CreateKinematicEnabledAttr(False)
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(xform.GetPrim())
    body.CreateSolverPositionIterationCountAttr(32)
    body.CreateSolverVelocityIterationCountAttr(16)
    body.CreateSleepThresholdAttr(0.0)
    body.CreateLinearDampingAttr(0.0)
    body.CreateAngularDampingAttr(0.0)
    body.CreateMaxDepenetrationVelocityAttr(0.1)
    mass = UsdPhysics.MassAPI.Apply(xform.GetPrim())
    mass.CreateMassAttr(float(mass_kg))
    izz = 0.5 * mass_kg * KNOB_R ** 2 * inertia_scale
    ixx = mass_kg * (3 * KNOB_R ** 2 + GRIP_L ** 2) / 12 * inertia_scale
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(ixx, ixx, izz))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, z0 + GRIP_L / 2))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
    _revolute(stage, joint_path, path, face_point, R, friction_nm)
    return path


def _revolute(stage, joint_path, body_path, anchor, R, friction_nm=FRICTION_NM):
    """Revolute joint from the world to the knob about its axis (local z), friction drive (saturated damper)."""
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in anchor]))
    joint.CreateLocalRot0Attr(_quat(R))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateAxisAttr('Z')
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
    drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(friction_nm / SATURATION_DEG_S)        # USD angular drives: per degree
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateMaxForceAttr(friction_nm)
    return joint


def _rotation_wxyz(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class KnobReader:
    """Angles of the knobs from the physics (call update() every tick): unwrapped degrees about each knob's outward
    axis, 0 at build; also how far each knob's axis has tilted (deg) and its anchor moved (mm) since the start (both 0
    for a knob on its joint)."""

    def __init__(self, knobs):
        from isaacsim.core.prims import SingleRigidPrim
        self.bodies, self.R0, self.p0, self.angle, self._raw = {}, {}, {}, {}, {}
        for key, path in knobs.items():
            body = SingleRigidPrim(path, name=f'fiber_knob_{key}', reset_xform_properties=False)
            body.initialize()
            p, q = body.get_world_pose()
            self.bodies[key], self.R0[key], self.p0[key] = body, _rotation_wxyz(q), np.asarray(p, float)
            self.angle[key], self._raw[key] = 0.0, 0.0
        self.tilt_deg = {k: 0.0 for k in knobs}
        self.drift_mm = {k: 0.0 for k in knobs}

    def update(self):
        for key, body in self.bodies.items():
            p, q = body.get_world_pose()
            rel = self.R0[key].T @ _rotation_wxyz(q)
            raw = math.degrees(math.atan2(rel[1, 0] - rel[0, 1], rel[0, 0] + rel[1, 1]))
            step = (raw - self._raw[key] + 180.0) % 360.0 - 180.0
            self.angle[key] += step
            self._raw[key] = raw
            self.tilt_deg[key] = math.degrees(math.acos(max(-1.0, min(1.0, float(rel[2, 2])))))
            self.drift_mm[key] = float(np.linalg.norm(np.asarray(p, float) - self.p0[key])) * 1000.0
        return dict(self.angle)

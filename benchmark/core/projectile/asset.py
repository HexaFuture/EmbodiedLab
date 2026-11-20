"""Spring launcher and ball (native rigid physics; no launch impulse, velocity assignment or trajectory scripting).

The visible coil spring, ratchet rack, pawl, notch scale and pointer are render-only (visuals.py).

The spring is the plunger joint's linear drive: stiffness k (private) towards the free length, no preload. The
ratchet pawl and the trigger act on the same drive (backend.Launcher): while the pawl holds a notch the drive is a
stiff stop at that notch; the trigger hands the plunger back to the spring. The ball rides two round rails in a
10 deg cradle with low friction and leaves the plunger face at the free length.
"""
import math
from .geometry import *


def _quat():
    from pxr import Gf
    return Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), -TILT_DEG).GetQuat())


def material(stage, path, mu, restitution=0.):
    from pxr import UsdPhysics, UsdShade, PhysxSchema
    m = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
    api.CreateStaticFrictionAttr(float(mu)); api.CreateDynamicFrictionAttr(float(mu)); api.CreateRestitutionAttr(float(restitution))
    PhysxSchema.PhysxMaterialAPI.Apply(m.GetPrim()).CreateFrictionCombineModeAttr('min')
    return m


def collider(prim, mat, offset=.001):
    from pxr import UsdPhysics, UsdShade, PhysxSchema
    UsdPhysics.CollisionAPI.Apply(prim)
    c = PhysxSchema.PhysxCollisionAPI.Apply(prim); c.CreateContactOffsetAttr(offset); c.CreateRestOffsetAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose='physics')


def axis_box(stage, path, u, v, w, dims, color, mat=None, tilted=True):
    """Box centred at axis coordinates (u along the launch axis from the muzzle, v lateral, w normal to the axis)."""
    from pxr import Gf, UsdGeom
    a = AXIS; n = (-math.sin(TILT), 0., math.cos(TILT))
    centre = [m + u * ai + w * ni for m, ai, ni in zip(MUZZLE, a, n)]; centre[1] += v
    c = UsdGeom.Cube.Define(stage, path); c.CreateSizeAttr(1.)
    c.AddTranslateOp().Set(Gf.Vec3d(*centre))
    if tilted: c.AddOrientOp().Set(_quat())
    c.AddScaleOp().Set(Gf.Vec3f(*dims)); c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if mat is not None: collider(c.GetPrim(), mat)
    return c


def build_ball(stage, root, centre):
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
    mat = material(stage, root + '/BallContact', .30)
    ball = UsdGeom.Sphere.Define(stage, root + '/Ball'); ball.CreateRadiusAttr(BALL_RADIUS)
    ball.AddTranslateOp().Set(Gf.Vec3d(*centre)); ball.CreateDisplayColorAttr([Gf.Vec3f(.8, .22, .07)])
    prim = ball.GetPrim(); collider(prim, mat)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    body.CreateSleepThresholdAttr(0.); body.CreateLinearDampingAttr(0.); body.CreateAngularDampingAttr(0.)
    body.CreateSolverPositionIterationCountAttr(32); body.CreateSolverVelocityIterationCountAttr(16)
    body.CreateEnableCCDAttr(True); body.CreateStabilizationThresholdAttr(0.)
    # USD unit is deg/s; the default 5729.58 clips a small rolling ball at 100 rad/s. Only this sphere changes.
    body.CreateMaxAngularVelocityAttr(22918.312)
    mass = UsdPhysics.MassAPI.Apply(prim); mass.CreateMassAttr(BALL_MASS); mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.4 * BALL_MASS * BALL_RADIUS ** 2)); mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
    return root + '/Ball'


def build_launcher(stage, root, origin, k):
    """Cradle rails, plunger (face, side arm, cocking knob) on its prismatic joint, static housing. Returns the drive."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema
    rail_mat = material(stage, root + '/RailContact', RAIL_FRICTION)
    face_mat = material(stage, root + '/PlungerContact', 0.)
    frame_mat = material(stage, root + '/FrameContact', .5)
    metal, dark, brass = (.42, .46, .49), (.08, .10, .13), (.75, .62, .20)
    quat = _quat()
    n = (-math.sin(TILT), 0., math.cos(TILT))
    # Round rails along the axis; the ball rests on them 45 deg below its centre on each side.
    off = (BALL_RADIUS + RAIL_RADIUS) * .70711
    for side, name in ((-1, 'RailL'), (1, 'RailR')):
        centre = [m - RAIL_LENGTH / 2 * a - off * ni for m, a, ni in zip(MUZZLE, AXIS, n)]; centre[1] += side * off
        cap = UsdGeom.Capsule.Define(stage, f'{root}/{name}')
        cap.CreateAxisAttr('X'); cap.CreateHeightAttr(RAIL_LENGTH); cap.CreateRadiusAttr(RAIL_RADIUS)
        cap.AddTranslateOp().Set(Gf.Vec3d(*centre)); cap.AddOrientOp().Set(quat); cap.CreateDisplayColorAttr([Gf.Vec3f(.85, .86, .88)])
        collider(cap.GetPrim(), rail_mat)
        # Web under each rail down to the cradle beam (clear of the plunger head, which runs between the rails).
        axis_box(stage, f'{root}/RailWeb{name[-1]}', -RAIL_LENGTH / 2, side * off, -.035, (RAIL_LENGTH, .003, .030), metal, frame_mat)
    axis_box(stage, root + '/CradleBeam', -.085, 0., -.055, (.17, .06, .010), dark, frame_mat)
    # Guide under the knob arm on the robot's side, and the spring housing behind the deepest plunger travel.
    axis_box(stage, root + '/KnobGuide', -.11, KNOB_OFFSET_Y - .012, -.019, (.13, .036, .010), metal, frame_mat)
    # Rear end block of the coil spring (bracketed to the cradle beam in visuals.build).
    axis_box(stage, root + '/SpringHousing', -.175, 0., 0., (.050, .030, .030), metal, frame_mat)
    for i, u in enumerate((-.02, -.15)):
        p = [m + u * a - .06 * ni for m, a, ni in zip(MUZZLE, AXIS, n)]
        UsdGeom.Xform.Define(stage, f'{root}/Stand{i}')
        c = UsdGeom.Cube.Define(stage, f'{root}/Stand{i}/Leg'); c.CreateSizeAttr(1.)
        c.AddTranslateOp().Set(Gf.Vec3d(p[0], 0., (p[2] - .040) / 2)); c.AddScaleOp().Set(Gf.Vec3f(.020, .050, p[2] + .040))
        c.CreateDisplayColorAttr([Gf.Vec3f(*metal)]); collider(c.GetPrim(), frame_mat)
        f = UsdGeom.Cube.Define(stage, f'{root}/Stand{i}/Foot'); f.CreateSizeAttr(1.)
        f.AddTranslateOp().Set(Gf.Vec3d(p[0], 0., -.035)); f.AddScaleOp().Set(Gf.Vec3f(.06, .12, .010))
        f.CreateDisplayColorAttr([Gf.Vec3f(*dark)]); collider(f.GetPrim(), frame_mat)
    # Plunger: an unrotated rigid body at the head centre; its parts carry their own orientation.
    face = -(BALL_RADIUS + MUZZLE_GAP)
    head = [m + (face - .015) * a for m, a in zip(MUZZLE, AXIS)]
    path = root + '/Plunger'
    UsdGeom.Xform.Define(stage, path).AddTranslateOp().Set(Gf.Vec3d(*head))
    def part(name, translate, dims, color, tilted=True):
        c = UsdGeom.Cube.Define(stage, f'{path}/{name}'); c.CreateSizeAttr(1.)
        c.AddTranslateOp().Set(Gf.Vec3d(*translate))
        if tilted: c.AddOrientOp().Set(quat)
        c.AddScaleOp().Set(Gf.Vec3f(*dims)); c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        collider(c.GetPrim(), face_mat)
    def at(du, dv, dw):
        return (du * AXIS[0] + dw * n[0], dv, du * AXIS[2] + dw * n[2])
    part('Head', at(0., 0., 0.), (.030, .020, .020), (.80, .82, .84))
    part('Arm', at(-.0025, KNOB_OFFSET_Y / 2 + .004, .001), (.015, KNOB_OFFSET_Y - .002, .010), metal)
    # Cocking knob: a world-vertical square post beside the cradle that the fingertip pushes back along the axis.
    base = at(-.0025, KNOB_OFFSET_Y, 0.)
    part('Knob', (base[0], base[1], base[2] + KNOB_TOP_ABOVE_AXIS / 2), (KNOB_SIZE, KNOB_SIZE, KNOB_TOP_ABOVE_AXIS), brass, tilted=False)
    # Render-only: coil spring, rack, pawl, numbered scale and pointer (no collider, mass or joint).
    from . import visuals
    visuals.build(stage, root, path, head)
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(path))
    mass = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(path)); mass.CreateMassAttr(PLUNGER_MASS)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(path))
    rb.CreateSolverPositionIterationCountAttr(32); rb.CreateSolverVelocityIterationCountAttr(16)
    rb.CreateSleepThresholdAttr(0.); rb.CreateEnableCCDAttr(True)
    joint = UsdPhysics.PrismaticJoint.Define(stage, root + '/Spring')
    joint.CreateBody1Rel().SetTargets([Sdf.Path(path)]); joint.CreateAxisAttr('X')
    joint.CreateLocalPos0Attr(Gf.Vec3f(*(o + h for o, h in zip(origin, head)))); joint.CreateLocalRot0Attr(quat)
    joint.CreateLocalPos1Attr(Gf.Vec3f(0)); joint.CreateLocalRot1Attr(quat)
    joint.CreateLowerLimitAttr(-REAR_LIMIT); joint.CreateUpperLimitAttr(FORWARD_LIMIT)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
    drive.CreateTypeAttr('force'); drive.CreateStiffnessAttr(float(k)); drive.CreateDampingAttr(DASHPOT_DAMPING)
    drive.CreateTargetPositionAttr(0.); drive.CreateTargetVelocityAttr(0.); drive.CreateMaxForceAttr(1e6)
    return drive

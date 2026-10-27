"""Neutral physical materials, spring and constrained-gripper mechanism."""

import numpy as np
from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema

APPARATUS = '/World/HookeApparatus'

HANGER = APPARATUS + '/Hanger'

SPRING_JOINT = '/World/HookePhysics/SpringSlider'

HANGER_MASS_KG = 0.080

SPRING_DAMPING_NS_M = 2.2

def configure_arm(stage, robot_path):
    """Limit the physical joints as well as the motion commands."""
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            articulation.CreateSolverPositionIterationCountAttr(32)
            articulation.CreateSolverVelocityIterationCountAttr(32)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            # USD angular joint velocity uses degrees/s; tensor APIs use rad/s.
            PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(
                float(np.degrees(.80)))
            if prim.GetName() in ('panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4'):
                # The proximal joints carry the arm's gravity torque. Keep
                # their authored torque limits and use a stiffer position
                # servo so extended reaches do not sag by centimetres.
                drive = UsdPhysics.DriveAPI.Apply(prim, 'angular')
                drive.CreateStiffnessAttr(3200.0)
                drive.CreateDampingAttr(120.0)
            else:
                UsdPhysics.DriveAPI.Apply(prim, 'angular').CreateStiffnessAttr(800.0)
    import robot_physics
    robot_physics.configure_robot_contacts(stage, robot_path)

def configure_contacts(stage, weight_paths, tray_path, spring_k, *, body_mass_kg=HANGER_MASS_KG,
                       damping_n_s_m=SPRING_DAMPING_NS_M):
    """body_mass_kg: tray rigid-body mass (m_t + m_s/3 for the dynamic experiment).
    damping_n_s_m: spring drive damping; hooke_k keeps the strong settling damping.
    """
    if not np.isfinite(spring_k) or spring_k <= 0:
        raise ValueError('spring stiffness must be positive and finite')
    if not (np.isfinite(body_mass_kg) and body_mass_kg > 0 and np.isfinite(damping_n_s_m) and damping_n_s_m >= 0):
        raise ValueError('tray mass and spring damping must be finite and positive')
    material = UsdShade.Material.Define(stage, '/World/PhysicsMaterials/HookeContact')
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(0.65)
    api.CreateDynamicFrictionAttr(0.50)
    api.CreateRestitutionAttr(0.02)
    for path in [HANGER, *weight_paths]:
        prim = stage.GetPrimAtPath(path)
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
        body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        body.CreateSolverPositionIterationCountAttr(16)
        body.CreateSolverVelocityIterationCountAttr(8)
        body.CreateSleepThresholdAttr(0.00001)
        body.CreateEnableCCDAttr(True)
        body.CreateLinearDampingAttr(0.0)
        body.CreateAngularDampingAttr(0.1)
        UsdPhysics.RigidBodyAPI(prim).CreateVelocityAttr(Gf.Vec3f(0))
        UsdPhysics.RigidBodyAPI(prim).CreateAngularVelocityAttr(Gf.Vec3f(0))
        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(True)
                c = PhysxSchema.PhysxCollisionAPI.Apply(child)
                c.CreateContactOffsetAttr(0.001)
                c.CreateRestOffsetAttr(0.0)
                UsdShade.MaterialBindingAPI.Apply(child).Bind(material, materialPurpose='physics')
    # The authored Hanger mesh also contains the obsolete small tray. Keep
    # its stem/pointer visible; use the new deck as the support collider.
    old_mesh = stage.GetPrimAtPath(HANGER + '/HangerMesh')
    UsdPhysics.CollisionAPI(old_mesh).CreateCollisionEnabledAttr(False)
    stem = UsdGeom.Cylinder.Define(stage, HANGER + '/StemCollider')
    stem.CreateRadiusAttr(.008)
    stem.CreateHeightAttr(.17)
    stem.CreateAxisAttr('Z')
    UsdGeom.XformCommonAPI(stem).SetTranslate(Gf.Vec3d(-.0195, 0, .001877))
    stem.CreateVisibilityAttr('invisible')
    UsdPhysics.CollisionAPI.Apply(stem.GetPrim())
    stem_contact = PhysxSchema.PhysxCollisionAPI.Apply(stem.GetPrim())
    stem_contact.CreateContactOffsetAttr(.001)
    stem_contact.CreateRestOffsetAttr(0)
    align_reading_pointer(stage)
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(HANGER)).CreateMassAttr(body_mass_kg)

    # Remove the historical global arm-vs-support exclusions. Contacts with
    # the deck and already placed weights must be respected by the robot.
    for prim in stage.Traverse():
        rel = prim.GetRelationship('physics:filteredPairs')
        if rel and str(prim.GetPath()).startswith(APPARATUS):
            rel.ClearTargets(True)
        # Several flattened cubes contain a baked, scaled extent, which
        # BBoxCache scales a second time. Repair it before measuring support.
        if prim.IsA(UsdGeom.Cube):
            cube = UsdGeom.Cube(prim)
            half = float(cube.GetSizeAttr().Get()) / 2
            cube.CreateExtentAttr([Gf.Vec3f(-half), Gf.Vec3f(half)])
    table = stage.GetPrimAtPath('/World/LabTable/Top')
    contact = PhysxSchema.PhysxCollisionAPI.Apply(table)
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)

    # A vertical guide constrains lateral motion/tilt. The force drive is the
    # spring itself; preload balances the empty tray body at its authored
    # height (the spring's effective mass only shifts the unknown zero). Payload gravity reaches it only through physical contact.
    stand = stage.GetPrimAtPath(APPARATUS + '/Stand')
    hanger = stage.GetPrimAtPath(HANGER)
    cache = UsdGeom.XformCache()
    world0 = cache.GetLocalToWorldTransform(stand)
    world1 = cache.GetLocalToWorldTransform(hanger)
    anchor = world1.ExtractTranslation() + Gf.Vec3d(0, 0, body_mass_kg * 9.81 / spring_k)
    joint = UsdPhysics.PrismaticJoint.Define(stage, SPRING_JOINT)
    joint.CreateBody0Rel().SetTargets([stand.GetPath()])
    joint.CreateBody1Rel().SetTargets([hanger.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(world0.GetInverse().Transform(anchor)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateAxisAttr('Z')
    joint.CreateLowerLimitAttr(-0.28)
    joint.CreateUpperLimitAttr(0.01)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
    drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(spring_k)
    drive.CreateDampingAttr(damping_n_s_m)
    drive.CreateTargetPositionAttr(0)
    drive.CreateTargetVelocityAttr(0)
    drive.CreateMaxForceAttr(100)
    return float(world1.ExtractTranslation()[2])

def align_reading_pointer(stage):
    """Put the entire horizontal needle at the scale face, with a red end.

    The original needle is a disconnected component of the joined hanger
    mesh. Extending only its end forward hid the bend behind the ruler and
    made the red end appear below the brass shaft in oblique camera views.
    Move the shaft forward too, and attach it to the stem with a visible
    return near the stem. All three parts share one physical height.
    """
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(HANGER + '/HangerMesh'))
    points = np.array(mesh.GetPointsAttr().Get(), dtype=float)
    # Bounds of the authored 114 mm needle component, in hanger coordinates.
    selected = ((points[:, 0] >= -.01951) & (points[:, 0] <= .09451)
                & (points[:, 2] >= -.01901) & (points[:, 2] <= -.00899))
    if selected.sum() != 96:
        raise RuntimeError('authored reading pointer geometry has changed')
    needle = points[selected]
    lower, upper = needle.min(axis=0), needle.max(axis=0)
    center = (lower + upper) / 2
    face_y, thickness = .051, .003
    scale = np.array([1.0, thickness / (upper[1] - lower[1]),
                      thickness / (upper[2] - lower[2])])
    points[selected] = (needle - center) * scale + [center[0], face_y, center[2]]
    mesh.GetPointsAttr().Set(points.astype(np.float32))
    mesh.CreateExtentAttr([Gf.Vec3f(*points.min(axis=0)), Gf.Vec3f(*points.max(axis=0))])
    # Preserve the bevel shading when reducing the shaft's cross-section.
    normals = np.array(mesh.GetNormalsAttr().Get(), dtype=float)
    interpolation = mesh.GetNormalsInterpolation()
    if interpolation == 'faceVarying':
        normal_mask = selected[np.array(mesh.GetFaceVertexIndicesAttr().Get())]
    elif interpolation in ('vertex', 'varying'):
        normal_mask = selected
    else:
        raise RuntimeError(f'unsupported pointer normal interpolation: {interpolation}')
    normals[normal_mask] /= scale
    normals[normal_mask] /= np.linalg.norm(normals[normal_mask], axis=1, keepdims=True)
    mesh.GetNormalsAttr().Set(normals.astype(np.float32))
    for name, position, dimensions, color in [
        ('PointerBridge', (lower[0], face_y / 2, center[2]),
         (thickness, face_y + thickness, thickness), (0.52, 0.25, 0.045)),
        ('PointerTip', (upper[0] + .013, face_y, center[2]),
         (.026, thickness, thickness), (0.75, 0.015, 0.01)),
    ]:
        cube = UsdGeom.Cube.Define(stage, HANGER + '/' + name)
        cube.CreateSizeAttr(1)
        xform = UsdGeom.XformCommonAPI(cube)
        xform.SetTranslate(Gf.Vec3d(*position))
        xform.SetScale(Gf.Vec3f(*dimensions))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])

def attach_grasp_joint(stage, hand_path, weight_path, anchor_world):
    """Approximate a closed grasp with a constraint at the current pose.

    Both local frames are computed from one world anchor, including rotation.
    No payload pose is rewritten; the body remains dynamic and collidable.
    """
    path = '/World/HookePhysics/GraspConstraint'
    if stage.GetPrimAtPath(path):
        raise RuntimeError('a grasp constraint is already active')
    cache = UsdGeom.XformCache()
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(hand_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(weight_path)])
    for index, body_path in enumerate([hand_path, weight_path]):
        inverse = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body_path)).GetInverse()
        getattr(joint, f'CreateLocalPos{index}Attr')(Gf.Vec3f(inverse.Transform(Gf.Vec3d(*anchor_world))))
        getattr(joint, f'CreateLocalRot{index}Attr')(Gf.Quatf(inverse.ExtractRotationQuat()))
    joint.CreateExcludeFromArticulationAttr(True)
    joint.CreateBreakForceAttr(35.0)
    joint.CreateBreakTorqueAttr(3.0)
    # Suppress only grasp-internal finger contacts; all scene contacts stay on.
    pairs = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(weight_path)).CreateFilteredPairsRel()
    pairs.SetTargets([Sdf.Path('/World/Franka/' + name) for name in
                      ('panda_hand', 'panda_leftfinger', 'panda_rightfinger')])
    return path

def release_grasp_joint(stage, weight_path):
    stage.RemovePrim('/World/HookePhysics/GraspConstraint')

def finish_release(stage, weight_path):
    rel = stage.GetPrimAtPath(weight_path).GetRelationship('physics:filteredPairs')
    if rel:
        rel.ClearTargets(True)

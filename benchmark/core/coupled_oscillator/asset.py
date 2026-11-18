"""Build the coupled-slider apparatus as a reusable USD stage.

Run inside Isaac Sim after SimulationApp is available. The pure-Python model
remains usable without pxr. Hidden dynamic parameters are
written only to the private manifest when explicitly requested by a worker.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from .model import CoupledParameters, sample_parameters


def build_apparatus(stage, params: CoupledParameters, root="/World/CoupledOscillatorLab", seed=7):
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    def mat(path, color, metallic=0.0, roughness=.35):
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    # Physics material is authored in the standalone asset.  Binding it at
    # build time is stable when the asset is later referenced into the
    # imported RoboCasa stage (runtime MaterialAPI edits made Kit terminate).
    contact_material = UsdShade.Material.Define(stage, root + "/Materials/ContactMaterial")
    contact_api = UsdPhysics.MaterialAPI.Apply(contact_material.GetPrim())
    # High friction is for the handle contacts. Ideal prismatic bearings
    # support each carriage without rail rubbing or Coulomb damping.
    contact_api.CreateStaticFrictionAttr().Set(0.8)
    contact_api.CreateDynamicFrictionAttr().Set(0.6)
    contact_api.CreateRestitutionAttr().Set(0.0)

    def finish(geom, pos, material, collider=False):
        geom.AddTranslateOp().Set(Gf.Vec3d(*pos))
        UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(material)
        if collider:
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim()).CreateContactOffsetAttr().Set(.0004)
            UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(
                contact_material, materialPurpose="physics")
        return geom

    def box(path, pos, size, material, collider=False):
        g = UsdGeom.Cube.Define(stage, path)
        g.CreateSizeAttr(1.0)
        finish(g, pos, material, collider).AddScaleOp().Set(Gf.Vec3f(*size))
        return g

    def cylinder(path, pos, radius, height, material, axis="Z", collider=False):
        g = UsdGeom.Cylinder.Define(stage, path)
        g.CreateRadiusAttr(radius); g.CreateHeightAttr(height); g.CreateAxisAttr(axis)
        return finish(g, pos, material, collider)

    def spring(path, x0, x1, y, z, material):
        # A visible curve follows the current nominal geometry. The physical
        # restoring force comes from the prismatic joint drive below.
        points = []
        turns = 9
        for i in range(turns * 2 + 1):
            u = i / (turns * 2)
            points.append(Gf.Vec3f(x0 + (x1 - x0) * u,
                                   y + (.016 if i % 2 else -.016), z))
        c = UsdGeom.BasisCurves.Define(stage, path)
        c.CreateCurveVertexCountsAttr([len(points)])
        c.CreatePointsAttr(points); c.CreateTypeAttr("linear"); c.CreateWidthsAttr([.006])
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(material)
        return c

    import numpy as np
    color = np.random.default_rng(seed + 9000).uniform([.04, .18, .30], [.20, .48, .65])
    UsdGeom.Xform.Define(stage, root)
    materials = {
        "steel": mat(root + "/Materials/Steel", (.40, .49, .58), .7, .24),
        "blue": mat(root + "/Materials/Slider", tuple(color), .35, .3),
        "red": mat(root + "/Materials/Coupling", (.78, .16, .12), .15, .25),
        "orange": mat(root + "/Materials/Return", (.82, .42, .10), .4, .26),
        "yellow": mat(root + "/Materials/Release", (.88, .62, .08), .4, .25),
        "dark": mat(root + "/Materials/Dark", (.025, .04, .06), .2),
        "white": mat(root + "/Materials/Mark", (.85, .92, .98), 0.0),
    }
    print("COUPLED_MATERIALS", flush=True)
    # All collider shapes use ContactMaterial; visual bindings remain separate.

    # The table/rails are intentionally self-contained so the asset can be
    # checked in isolation. A consumer may hide the table and use the shared
    # RoboCasa lab table while preserving the three camera mounts.
    print("COUPLED_BASE_START", flush=True)
    box(root + "/Base", (0.70, 0.0, .72), (1.55, .72, .08), materials["steel"], True)
    print("COUPLED_BASE_DONE", flush=True)
    for name, y in (("Near", -.16), ("Far", .16)):
        print(f"COUPLED_RAIL_START_{name}", flush=True)
        box(root + f"/Rail{name}", (.70, y, .86), (1.42, .055, .055), materials["steel"], True)
        print(f"COUPLED_RAIL_DONE_{name}", flush=True)
    for name, x in (("Left", .02), ("Right", 1.38)):
        box(root + f"/EndStop{name}", (x, 0, .95), (.06, .48, .20), materials["dark"], True)
    print("COUPLED_STATIC_GEOMETRY", flush=True)

    slider_paths = []
    anchor_paths = []
    centers = (.43, .97)
    for i, x in enumerate(centers, 1):
        body_path = f"{root}/Slider{i}"
        body = UsdGeom.Xform.Define(stage, body_path)
        body.AddTranslateOp().Set(Gf.Vec3d(x, 0, .99))
        rb = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        rb.CreateKinematicEnabledAttr().Set(False)
        phys = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
        phys.CreateSleepThresholdAttr(0.0)
        phys.CreateSolverPositionIterationCountAttr(32)
        phys.CreateSolverVelocityIterationCountAttr(1)
        phys.CreateLinearDampingAttr().Set(0.0); phys.CreateAngularDampingAttr().Set(0.0)
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr().Set(params.mass_kg)
        inertia = params.mass_kg * (.12**2 + .06**2) / 12
        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(inertia, inertia, inertia))
        shape = box(body_path + "/Housing", (0, 0, 0), (.24, .28, .17), materials["blue"], True)
        # Upright yellow grip post: the Franka fingers close around this
        # visible handle from above, then translate it along the rail X axis.
        cylinder(body_path + "/Handle", (0, -.19, .13), .012, .14, materials["yellow"], "Z", True)
        box(body_path + "/HandleMount", (0, -.16, .07), (.045, .10, .025), materials["steel"], True)
        slider_paths.append(body_path)

        anchor_path = f"{root}/ReturnAnchor{i}"
        anchor = UsdGeom.Xform.Define(stage, anchor_path)
        anchor.AddTranslateOp().Set(Gf.Vec3d(.02 if i == 1 else 1.38, 0, .99))
        ar = UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim()); ar.CreateKinematicEnabledAttr().Set(True)
        am = UsdPhysics.MassAPI.Apply(anchor.GetPrim()); am.CreateMassAttr().Set(1.0)
        anchor_paths.append(anchor_path)

        joint = UsdPhysics.PrismaticJoint.Define(stage, f"{root}/ReturnJoint{i}")
        # One world-to-slider rail per carriage. The coupling spring must
        # not be another prismatic joint: that adds five redundant locks.
        joint.CreateBody0Rel().SetTargets([])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(x, 0, .99)); joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.CreateAxisAttr().Set("X"); joint.CreateLowerLimitAttr().Set(-.10); joint.CreateUpperLimitAttr().Set(.10)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        # The joint supplies only the geometric X rail constraint.  Its drive
        # is intentionally disabled: the worker applies the sampled spring,
        # damping and coupling forces after the release event, so no hidden
        # parameter is embedded in the public USD asset.
        drive.CreateTypeAttr().Set("force"); drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(0.0)
        drive.CreateTargetPositionAttr().Set(0.0)
        drive.CreateTargetVelocityAttr().Set(0.0)
        drive.CreateMaxForceAttr().Set(0.0)
        print(f"COUPLED_SLIDER_{i}", flush=True)

    # All three springs and viscous damping use SI forces in the worker.

    # Curves are authored after the rigid/joint graph by the render-time
    # decorator. Keeping the initial stage primitive-only avoids a Kit 6.0
    # BasisCurves shutdown race during standalone export.
    print("COUPLED_SPRINGS_DEFERRED", flush=True)
    print("COUPLED_RELEASE_START", flush=True)
    release = UsdGeom.Xform.Define(stage, root + "/ReleaseBar")
    release.AddTranslateOp().Set(Gf.Vec3d(.70, -.23, 1.12))
    UsdPhysics.RigidBodyAPI.Apply(release.GetPrim())
    UsdPhysics.MassAPI.Apply(release.GetPrim()).CreateMassAttr(.1)
    cylinder(root + "/ReleaseBar/Grip", (0, 0, 0), .012, .14, materials["yellow"], "Z", True)
    box(root + "/ReleaseBar/Link", (0, .06, -.055), (.07, .14, .02), materials["steel"])
    release_joint = UsdPhysics.PrismaticJoint.Define(stage, root + "/ReleaseJoint")
    release_joint.CreateBody1Rel().SetTargets([release.GetPath()])
    release_joint.CreateLocalPos0Attr(Gf.Vec3f(.70, -.23, 1.12))
    release_joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    release_joint.CreateAxisAttr("Y")
    release_joint.CreateLowerLimitAttr(-.002)
    release_joint.CreateUpperLimitAttr(.07)
    release_drive = UsdPhysics.DriveAPI.Apply(release_joint.GetPrim(), "linear")
    release_drive.CreateTypeAttr("force")
    release_drive.CreateStiffnessAttr(25.0)
    release_drive.CreateDampingAttr(2.0)
    release_drive.CreateTargetPositionAttr(0.0)
    box(root + "/ReleaseGuide", (.70, -.19, 1.025), (.06, .18, .07), materials["dark"])
    print("COUPLED_RELEASE_BAR", flush=True)
    from .instrument_visuals import build_visuals
    build_visuals(stage, root)
    # Hand-operated controls: catches with CLAMP buttons and hooks, recorder keypad and display, damper lever.
    from .controls_asset import build_controls
    build_controls(stage, root, contact_material)
    print("COUPLED_RELEASE_DONE", flush=True)
    # Persist before returning: Kit may terminate its task group while the
    # process is unwinding, so a worker must never lose the authored stage.
    stage.GetRootLayer().Save()
    print("COUPLED_STAGE_AUTHORED", flush=True)
    return {
        "asset_id": "coupled_slider_v1", "root": root,
        "slider_paths": slider_paths, "return_anchor_paths": anchor_paths,
        "coupling_model": "equal_opposite_physx_forces", "mass_kg": params.mass_kg,
        "rail_origin_m": .02, "rail_length_m": 1.36,
        "public_calibration": {"mass_kg": params.mass_kg, "rail_length_m": 1.36, "position_resolution_m": .0001, "instrument_sample_hz": 120},
    }


"""Construct the physical pendulum apparatus and its in-scene timer displays.

Import after SimulationApp has started. Distances are metres, masses kilograms,
and the body origin is its pivot. Positive rotation about Y moves the bob in -X.
"""

from __future__ import annotations

import math

from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade


BOB_RADIUS_M = 0.015
BOB_MASS_KG = 0.100
ROD_RADIUS_M = 0.001
ROD_MASS_KG = 0.001
LINEAR_DAMPING_PER_S = 0.012
ANGULAR_DAMPING_PER_S = 0.004
NOMINAL_RELEASE_DEG = 4.0
MAX_RELEASE_DEG = 5.0
LENGTHS_M = (0.25, 0.35, 0.45)
GATE_HEIGHT_M = 1.15
GATE_STEM_BOTTOM_M = 0.850

# Gate push buttons (START, RESET) on a deck on top of each pendulum's timer unit, as the chemistry-bench keys: a
# spring-returned cap on a vertical prismatic joint; the switch closes when the cap is pushed 4 mm down and opens
# again above 2 mm. The deck sits on the housing top, its front edge flush with the timer face, and overhangs the
# housing toward the pendulums; the caps stand side by side along X, 60 mm apart (closed fingertips 57 mm wide plus
# a 24 mm cap leave 19 mm), START on the robot's left (+X). Offsets are from the timer housing centre.
TIMER_HOUSING_M = (.22, .030, .132)
DECK_SIZE_M = (0.140, 0.060, 0.016)              # X x Y x height
DECK_DY_M = .0155 - DECK_SIZE_M[1] / 2           # deck centre behind the timer centre (front edge on the face)
BUTTON_DY_M = -0.020                             # cap centres behind the timer centre
LABEL_DY_M = 0.004                               # START / RESET print on the deck's front strip
# Every timer stands in its pendulum's column (P3's stood 120 mm to +X of P3 earlier). P3's sits 120 mm
# farther from the robot than P1's and P2's so that the arm at its start pose hides its face in neither fixed view.
TIMER_OFFSET_M = {1: (0.0, 0.0), 2: (0.0, 0.0), 3: (0.0, -0.120)}
BUTTON_DX_M = {"start": 0.030, "reset": -0.030}
BUTTON_COLOURS = {"start": (0.10, 0.62, 0.16), "reset": (0.80, 0.08, 0.06)}
CAP_R_M, CAP_H_M = 0.012, 0.010
CAP_LIFT_M = 0.010                               # cap bottom above the console top when released
CAP_TRAVEL_M = 0.009
SWITCH_DOWN_M = 0.004
SWITCH_UP_M = 0.002
CAP_MASS_KG = 0.010
CAP_SPRING_N_PER_M = 400.0
CAP_SPRING_PRELOAD_M = 0.001
CAP_SPRING_DAMPING = 4.0


_FONT = {
    "0": (14, 17, 19, 21, 25, 17, 14),
    "1": (4, 12, 4, 4, 4, 4, 14),
    "2": (14, 17, 1, 2, 4, 8, 31),
    "3": (30, 1, 1, 14, 1, 1, 30),
    "4": (2, 6, 10, 18, 31, 2, 2),
    "5": (31, 16, 16, 30, 1, 1, 30),
    "6": (14, 16, 16, 30, 17, 17, 14),
    "7": (31, 1, 2, 4, 8, 8, 8),
    "8": (14, 17, 17, 14, 17, 17, 14),
    "9": (14, 17, 17, 15, 1, 1, 14),
    "A": (14, 17, 17, 31, 17, 17, 17),
    "B": (30, 17, 17, 30, 17, 17, 30),
    "C": (14, 17, 16, 16, 16, 17, 14),
    "D": (30, 17, 17, 17, 17, 17, 30),
    "E": (31, 16, 16, 30, 16, 16, 31),
    "G": (14, 17, 16, 23, 17, 17, 14),
    "I": (14, 4, 4, 4, 4, 4, 14),
    "L": (16, 16, 16, 16, 16, 16, 31),
    "M": (17, 27, 21, 21, 17, 17, 17),
    "N": (17, 25, 25, 21, 19, 19, 17),
    "O": (14, 17, 17, 17, 17, 17, 14),
    "P": (30, 17, 17, 30, 16, 16, 16),
    "R": (30, 17, 17, 30, 20, 18, 17),
    "S": (15, 16, 16, 14, 1, 1, 30),
    "T": (31, 4, 4, 4, 4, 4, 4),
    "U": (17, 17, 17, 17, 17, 17, 14),
    "W": (17, 17, 17, 21, 21, 27, 17),
    "Y": (17, 17, 10, 4, 4, 4, 4),
    "<": (1, 2, 4, 8, 4, 2, 1),
    "=": (0, 0, 31, 0, 31, 0, 0),
    ".": (0, 0, 0, 0, 0, 6, 6),
    "-": (0, 0, 0, 31, 0, 0, 0),
    " ": (0, 0, 0, 0, 0, 0, 0),
}
_SEGMENTS = {
    "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg",
    "4": "bcfg", "5": "acdfg", "6": "acdefg", "7": "abc",
    "8": "abcdefg", "9": "abcdfg", "-": "g", " ": "",
}


def _material(stage, path, color, metallic=0.0, roughness=0.35, emission=0.0):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    if emission:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*(c * emission for c in color))
        )
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _finish(geom, position, material, collider=False, dynamic=False):
    geom.AddTranslateOp().Set(Gf.Vec3d(*position))
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(material)
    if collider:
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        collision = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
        collision.CreateContactOffsetAttr().Set(0.0003 if dynamic else 0.0005)
        collision.CreateRestOffsetAttr().Set(0.0)
    return geom


def _box(stage, path, position, size, material, collider=False):
    geom = UsdGeom.Cube.Define(stage, path)
    geom.CreateSizeAttr(1.0)
    _finish(geom, position, material, collider)
    geom.AddScaleOp().Set(Gf.Vec3f(*size))
    return geom


def _cylinder(stage, path, position, radius, height, material,
              axis="Z", collider=False, dynamic=False):
    geom = UsdGeom.Cylinder.Define(stage, path)
    geom.CreateRadiusAttr(radius)
    geom.CreateHeightAttr(height)
    geom.CreateAxisAttr(axis)
    return _finish(geom, position, material, collider, dynamic)


def _text(stage, path, text, center, height, material):
    """Flat 5x7 glyph mesh readable from +Y, with no external font dependency."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    pixel = height / 7.0
    width = (len(text) * 6 - 1) * pixel
    vertices, counts, indices = [], [], []
    for char_index, char in enumerate(text.upper()):
        for row, bits in enumerate(_FONT[char]):
            for column in range(5):
                if not bits & (1 << (4 - column)):
                    continue
                x = center[0] + width / 2 - (char_index * 6 + column) * pixel
                z = center[2] + height / 2 - row * pixel
                y = center[1]
                offset = len(vertices)
                vertices.extend([(x, y, z), (x - pixel * .87, y, z),
                                 (x - pixel * .87, y, z - pixel * .87),
                                 (x, y, z - pixel * .87)])
                counts.append(4)
                indices.extend(range(offset, offset + 4))
    mesh.CreatePointsAttr(vertices)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _flat_text(stage, path, text, center, height, material):
    """Flat 5x7 glyph mesh lying in the table plane at center[2], read from the robot's side (+Y):
    letter tops toward -Y, reading toward -X."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    pixel = height / 7.0
    width = (len(text) * 6 - 1) * pixel
    vertices, counts, indices = [], [], []
    for char_index, char in enumerate(text.upper()):
        for row, bits in enumerate(_FONT[char]):
            for column in range(5):
                if not bits & (1 << (4 - column)):
                    continue
                x = center[0] + width / 2 - (char_index * 6 + column) * pixel
                y = center[1] - height / 2 + row * pixel
                z = center[2]
                offset = len(vertices)
                vertices.extend([(x, y, z), (x - pixel * .87, y, z),
                                 (x - pixel * .87, y + pixel * .87, z),
                                 (x, y + pixel * .87, z)])
                counts.append(4)
                indices.extend(range(offset, offset + 4))
    mesh.CreatePointsAttr(vertices)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def build_gate_buttons(stage, assembly, timer_position, label, materials, physical_mat):
    """Deck with the START and RESET push buttons of one gate on top of its timer unit (apparatus frame of
    `assembly`; timer_position is the timer housing centre).

    Each cap is a dynamic body on a spring-returned vertical prismatic joint to a kinematic anchor inside the
    assembly, so the buttons follow any placement of the bench. Returns {button: paths and rest centre}."""
    tx, ty, tz = timer_position
    housing_top = tz + TIMER_HOUSING_M[2] / 2
    top = housing_top + DECK_SIZE_M[2]
    _box(stage, assembly + "/ButtonDeck", (tx, ty + DECK_DY_M, housing_top + DECK_SIZE_M[2] / 2),
         DECK_SIZE_M, materials["housing"], True)
    _flat_text(stage, assembly + "/ButtonDeckLabel", label, (tx, ty + BUTTON_DY_M, top + .0004), .010, materials["ink"])
    button_y = ty + BUTTON_DY_M
    buttons = {}
    for name, dx in BUTTON_DX_M.items():
        bx = tx + dx
        part = assembly + "/" + name.title() + "Button"
        centre = (bx, button_y, top + CAP_LIFT_M + CAP_H_M / 2)
        colour = materials["button_" + name]
        _cylinder(stage, part + "Collar", (bx, button_y, top + (CAP_LIFT_M - .001) / 2),
                  CAP_R_M + .003, CAP_LIFT_M - .001, materials["dark"])
        _flat_text(stage, part + "Label", name.upper(), (bx, ty + LABEL_DY_M, top + .0004), .008, materials["ink"])
        cap_path = part + "Cap"
        cap = UsdGeom.Xform.Define(stage, cap_path)
        cap.AddTranslateOp().Set(Gf.Vec3d(*centre))
        UsdPhysics.RigidBodyAPI.Apply(cap.GetPrim())
        rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(cap.GetPrim())
        rigid.CreateSolverPositionIterationCountAttr().Set(16)
        rigid.CreateSolverVelocityIterationCountAttr().Set(8)
        rigid.CreateSleepThresholdAttr().Set(0.0)
        UsdPhysics.MassAPI.Apply(cap.GetPrim()).CreateMassAttr().Set(CAP_MASS_KG)
        shape = _cylinder(stage, cap_path + "/Cap", (0, 0, 0), CAP_R_M, CAP_H_M, colour, collider=True, dynamic=True)
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(
            physical_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        anchor_path = part + "Anchor"
        anchor = UsdGeom.Xform.Define(stage, anchor_path)
        anchor.AddTranslateOp().Set(Gf.Vec3d(*centre))
        UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim()).CreateKinematicEnabledAttr().Set(True)
        anchor_mass = UsdPhysics.MassAPI.Apply(anchor.GetPrim())
        anchor_mass.CreateMassAttr().Set(1.0)
        anchor_mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1.0))
        joint_path = part + "Joint"
        joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(anchor_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(cap_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.CreateAxisAttr().Set("Z")
        joint.CreateLowerLimitAttr().Set(-CAP_TRAVEL_M)
        joint.CreateUpperLimitAttr().Set(0.0)
        joint.CreateCollisionEnabledAttr().Set(False)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(CAP_SPRING_N_PER_M)
        drive.CreateDampingAttr().Set(CAP_SPRING_DAMPING)
        drive.CreateTargetPositionAttr().Set(CAP_SPRING_PRELOAD_M)
        drive.CreateTargetVelocityAttr().Set(0.0)
        drive.CreateMaxForceAttr().Set(100.0)
        buttons[name] = {"cap_path": cap_path, "anchor_path": anchor_path, "joint_path": joint_path,
                         "rest_centre": centre}
    return buttons


TIMER_DIGIT_PITCH = .85  # pendulum timer face: digit pitch in digit heights


def _digits(stage, path, center, count, height, material, separator_after=None, separator_gap=0.0, pitch=.68):
    UsdGeom.Xform.Define(stage, path)
    width, thickness, spacing = height * .52, height * .105, height * pitch
    coordinates = {
        "a": (0, height / 2), "b": (width / 2, height / 4),
        "c": (width / 2, -height / 4), "d": (0, -height / 2),
        "e": (-width / 2, -height / 4), "f": (-width / 2, height / 4),
        "g": (0, 0),
    }
    for i in range(count):
        origin = center[0] + (count - 1) * spacing / 2 - i * spacing
        if separator_after is not None:
            origin += separator_gap / 2 if i < separator_after else -separator_gap / 2
        for name, (sx, sz) in coordinates.items():
            horizontal = name in "adg"
            size = ((width - thickness, .0005, thickness) if horizontal else
                    (thickness, .0005, height / 2 - thickness))
            _box(stage, f"{path}/D{i}_{name}",
                 (origin - sx, center[1], center[2] + sz), size, material)
    return path


def _set_digits(stage, path, value):
    for index, char in enumerate(value):
        enabled = _SEGMENTS[char]
        for name in "abcdefg":
            prim = stage.GetPrimAtPath(f"{path}/D{index}_{name}")
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(
                UsdGeom.Tokens.inherited if name in enabled else UsdGeom.Tokens.invisible
            )


def build_display(stage, path, position, length_m, materials=None, instrument_label=None):
    """Return a display descriptor; the face points toward world +Y."""
    if materials is None:
        materials = {
            "housing": _material(stage, path + "/Materials/Housing", (.055, .14, .17), .3),
            "face": _material(stage, path + "/Materials/Face", (.008, .015, .013)),
            "ink": _material(stage, path + "/Materials/Ink", (.85, .92, .89)),
            "digits": _material(stage, path + "/Materials/Digits", (.22, 1.0, .58), emission=.4),
        }
    UsdGeom.Xform.Define(stage, path)
    x, y, z = position
    _box(stage, path + "/Housing", position, (.22, .030, .132), materials["housing"], True)
    for side, offset in (("Left", -.080), ("Right", .080)):
        _box(stage, path + "/Foot" + side, (x + offset, y, z - .073),
             (.027, .048, .014), materials["housing"], True)
    face_y = y + .0155
    _box(stage, path + "/Face", (x, face_y, z - .004),
         (.204, .001, .109), materials["face"])
    length_label = f"L = {length_m:.3f} M"
    if instrument_label is not None:
        length_label = f"{instrument_label}  {length_label}"
    _text(stage, path + "/Length", length_label,
          (x, face_y + .001, z + .055), .0105, materials["ink"])
    _text(stage, path + "/TimeLabel", "TIME S", (x, face_y + .001, z + .035),
          .0075, materials["ink"])
    time_height = .032
    # Digit pitch 0.85 h: at 0.68 h a "1" (segments b, c only) sat ~0.06 h from the next digit and
    # "10.538" read as "0.538" in the camera images. The decimal point sits in the middle of the seconds/ms gap.
    time_path = _digits(stage, path + "/Time", (x, face_y + .001, z + .009),
                        5, time_height, materials["digits"], separator_after=2, separator_gap=.010,
                        pitch=TIMER_DIGIT_PITCH)
    _box(stage, path + "/Decimal", (x + time_height * TIMER_DIGIT_PITCH / 2, face_y + .001, z - .0045),
         (.004, .0005, .004), materials["digits"])
    _text(stage, path + "/CyclesLabel", "CYCLES", (x + .055, face_y + .001, z - .020),
          .007, materials["ink"])
    cycles_path = _digits(stage, path + "/Cycles", (x + .055, face_y + .001, z - .041),
                          2, .022, materials["digits"], pitch=TIMER_DIGIT_PITCH)
    # The state word, as tall as the CYCLES digits so that the fixed cameras can read it (9 mm was ~3 px there).
    for state in STATE_WORDS:
        _text(stage, path + "/State" + state, state,
              (x - .050, face_y + .001, z - .034), .022, materials["digits"])
    descriptor = {"path": path, "time_path": time_path, "cycles_path": cycles_path}
    update_display(stage, descriptor, 0, 0, "IDLE")
    return descriptor


# Timer state words: IDLE (no acquisition: never started or RESET, display at zero), ARM (START pressed: waiting
# for the bob's first crossing and through the warm-up cycle), RUN (timing), DONE (complete), ERR (invalid, the
# digits show dashes).
STATE_WORDS = ("IDLE", "ARM", "RUN", "DONE", "ERR")


def update_display(stage, display, elapsed_s, cycles, state):
    """Update actual scene geometry. Values must come from photogate events."""
    state = state.upper()
    if state not in STATE_WORDS:
        raise ValueError(f"Unknown timer state: {state}")
    if not math.isfinite(elapsed_s) or elapsed_s < 0 or elapsed_s >= 100:
        raise ValueError("Display range is 0 <= elapsed time < 100 seconds")
    if not isinstance(cycles, int) or not 0 <= cycles <= 99:
        raise ValueError("Display cycle count must be an integer between 0 and 99")
    milliseconds = min(99999, int(elapsed_s * 1000 + .5))
    _set_digits(stage, display["time_path"], "-----" if state == "ERR" else f"{milliseconds:05d}")
    _set_digits(stage, display["cycles_path"], "--" if state == "ERR" else f"{cycles:02d}")
    for candidate in STATE_WORDS:
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath(display["path"] + "/State" + candidate))
        imageable.GetVisibilityAttr().Set(
            UsdGeom.Tokens.inherited if state == candidate else UsdGeom.Tokens.invisible
        )


def build_apparatus(stage, root="/World/PendulumLab"):
    """Create three physical pendula and return a list of serializable descriptors.

    Each descriptor includes body_path, joint_path, pivot, length_m, mass_kg,
    center_of_mass_local, diagonal_inertia, gate ray geometry and display.
    No drive or scripted force acts on any pendulum.
    """
    UsdGeom.Xform.Define(stage, root)
    materials = {
        "housing": _material(stage, root + "/Materials/Housing", (.055, .14, .17), .3),
        "face": _material(stage, root + "/Materials/Face", (.008, .015, .013)),
        "ink": _material(stage, root + "/Materials/Ink", (.85, .92, .89)),
        "digits": _material(stage, root + "/Materials/Digits", (.22, 1.0, .58), emission=.4),
        "steel": _material(stage, root + "/Materials/Steel", (.53, .59, .63), .85, .23),
        "brass": _material(stage, root + "/Materials/Brass", (.66, .43, .12), .78, .26),
        "dark": _material(stage, root + "/Materials/Dark", (.07, .085, .09), .5),
        "plate": _material(stage, root + "/Materials/Plate", (.82, .85, .80), .2),
        "gate": _material(stage, root + "/Materials/Gate", (.13, .21, .28), .2),
        "lens": _material(stage, root + "/Materials/Lens", (.68, .04, .025), .1, emission=.1),
        "button_start": _material(stage, root + "/Materials/ButtonStart", BUTTON_COLOURS["start"], .0, .55),
        "button_reset": _material(stage, root + "/Materials/ButtonReset", BUTTON_COLOURS["reset"], .0, .55),
    }
    physical_mat = UsdShade.Material.Define(stage, root + "/Materials/Contact")
    surface = UsdPhysics.MaterialAPI.Apply(physical_mat.GetPrim())
    surface.CreateStaticFrictionAttr().Set(.35)
    surface.CreateDynamicFrictionAttr().Set(.25)
    surface.CreateRestitutionAttr().Set(0.0)

    result = []
    for index, (x, length) in enumerate(zip((.52, .76, 1.00), LENGTHS_M), start=1):
        assembly = f"{root}/Pendulum{index}"
        # Stagger the taller pendula toward the rear and keep the whole row
        # left of the robot's central approach lane. This preserves the
        # original RoboCasa camera mounts while exposing rods and displays.
        layout_offset = (-.08, -.06 * (index - 1), 0.0)
        UsdGeom.Xform.Define(stage, assembly).AddTranslateOp().Set(Gf.Vec3d(*layout_offset))
        pivot = (x, -.18, 1.03 + length)
        _box(stage, assembly + "/Base", (x, -.29, .834), (.195, .17, .028), materials["dark"], True)
        pole_height = pivot[2] - .850
        _cylinder(stage, assembly + "/Support", (x, -.29, .850 + pole_height / 2),
                  .009, pole_height, materials["steel"], collider=True)
        _box(stage, assembly + "/TopClamp", (x, -.251, pivot[2]),
             (.035, .110, .029), materials["dark"], True)
        _cylinder(stage, assembly + "/Bearing", (x, -.210, pivot[2]),
                  .014, .027, materials["brass"], "Y", True)
        _cylinder(stage, assembly + "/Axle", (x, -.193, pivot[2]),
                  .003, .029, materials["steel"], "Y")
        _box(stage, assembly + "/AnglePlate", (x, -.230, pivot[2] - .025),
             (.098, .002, .065), materials["plate"], True)
        _text(stage, assembly + "/InstrumentId", f"P{index}",
              (x + .029, -.227, pivot[2] - .012), .014, materials["dark"])
        _text(stage, assembly + "/SmallAngleLabel", "<5 DEG",
              (x, -.228, pivot[2] - .047), .008, materials["dark"])
        for angle in (-5, 0, 5):
            theta = math.radians(angle)
            tick = _box(stage, assembly + f"/Tick{angle + 5}",
                        (x + .033 * math.sin(theta), -.227, pivot[2] - .033 * math.cos(theta)),
                        (.00065, .0006, .009 if angle == 0 else .004), materials["dark"])
            tick.AddRotateYOp().Set(-angle)

        body_path = assembly + "/MovingBody"
        body = UsdGeom.Xform.Define(stage, body_path)
        body.AddTranslateOp().Set(Gf.Vec3d(*pivot))
        body.AddOrientOp().Set(Gf.Quatf(1.0))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        physics = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
        # The bob's orbital kinetic energy is mostly translation of the COM.
        # Angular drag alone dissipates only the very small COM spin inertia.
        # Real velocity-proportional linear drag gives visible gentle decay;
        # no joint motor or position target drives the free pendulum.
        physics.CreateAngularDampingAttr().Set(ANGULAR_DAMPING_PER_S)
        physics.CreateLinearDampingAttr().Set(LINEAR_DAMPING_PER_S)
        physics.CreateSolverPositionIterationCountAttr().Set(32)
        physics.CreateSolverVelocityIterationCountAttr().Set(8)
        physics.CreateEnableCCDAttr().Set(True)
        physics.CreateSleepThresholdAttr().Set(0.0)
        total_mass = BOB_MASS_KG + ROD_MASS_KG
        com_distance = (BOB_MASS_KG * length + ROD_MASS_KG * length / 2) / total_mass
        bob_inertia = .4 * BOB_MASS_KG * BOB_RADIUS_M ** 2
        rod_inertia = ROD_MASS_KG * (length ** 2 / 12 + ROD_RADIUS_M ** 2 / 4)
        transverse = (bob_inertia + BOB_MASS_KG * (length - com_distance) ** 2
                      + rod_inertia + ROD_MASS_KG * (length / 2 - com_distance) ** 2)
        axial = bob_inertia + ROD_MASS_KG * ROD_RADIUS_M ** 2 / 2
        diagonal = (transverse, transverse, axial)
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr().Set(total_mass)
        mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0, 0, -com_distance))
        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*diagonal))
        mass.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0))
        rod = _cylinder(stage, body_path + "/Rod", (0, 0, -length / 2),
                        ROD_RADIUS_M, length, materials["steel"], collider=True, dynamic=True)
        bob = UsdGeom.Sphere.Define(stage, body_path + "/Bob")
        bob.CreateRadiusAttr(BOB_RADIUS_M)
        _finish(bob, (0, 0, -length), materials["brass"], True, True)
        for shape in (rod, bob):
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(
                physical_mat, UsdShade.Tokens.weakerThanDescendants, "physics")

        # A kinematic local anchor follows placement of the reusable asset.
        # A world joint with no body0 would leave its absolute anchor behind
        # when the asset root is translated or rotated after referencing it.
        anchor_path = assembly + "/FixedAnchor"
        anchor = UsdGeom.Xform.Define(stage, anchor_path)
        anchor.AddTranslateOp().Set(Gf.Vec3d(*pivot))
        anchor_body = UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim())
        anchor_body.CreateKinematicEnabledAttr().Set(True)
        anchor_mass = UsdPhysics.MassAPI.Apply(anchor.GetPrim())
        anchor_mass.CreateMassAttr().Set(1.0)
        anchor_mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1.0))
        joint_path = assembly + "/PivotJoint"
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(anchor_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.CreateAxisAttr().Set("Y")
        joint.CreateLowerLimitAttr().Set(-20.0)
        joint.CreateUpperLimitAttr().Set(20.0)
        joint.CreateCollisionEnabledAttr().Set(False)

        # Detect the exposed rod higher up, leaving the bob's contact region
        # below the housings. The gate remains a physical optical instrument.
        gate_z = GATE_HEIGHT_M
        stem_top = gate_z - .004
        _box(stage, assembly + "/GateStem", (x, -.29, (GATE_STEM_BOTTOM_M + stem_top) / 2),
             (.012, .015, stem_top - GATE_STEM_BOTTOM_M), materials["steel"], True)
        for label, gate_y in (("Emitter", -.220), ("Receiver", -.140)):
            _box(stage, assembly + "/" + label, (x, gate_y, gate_z),
                 (.022, .014, .030), materials["gate"], True)
            lens_y = gate_y + (.0075 if label == "Emitter" else -.0075)
            _cylinder(stage, assembly + "/" + label + "Lens", (x, lens_y, gate_z),
                      .004, .001, materials["lens"], "Y")
        # The U-shaped gate frame routes around the side of the swing envelope.
        # Only the thin rod crosses the horizontal gate ray.
        _box(stage, assembly + "/GateRearArm", (x, -.251, gate_z),
             (.014, .048, .013), materials["gate"], True)
        _box(stage, assembly + "/GateFrontArm", (x + .045, -.19, gate_z),
             (.014, .12, .013), materials["gate"], True)
        _box(stage, assembly + "/GateFrontElbow", (x + .0225, -.14, gate_z),
             (.055, .014, .013), materials["gate"], True)
        _box(stage, assembly + "/GateRearElbow", (x + .0225, -.245, gate_z),
             (.055, .014, .013), materials["gate"], True)

        # The timer faces stand in front of the staggered pendulum stands, each in its own pendulum's column
        # (TIMER_OFFSET_M), and carry the gate's buttons on top. Their caps stay below the bobs' height.
        timer_dx, timer_dy = TIMER_OFFSET_M[index]
        timer_position = (x + timer_dx, .120 - layout_offset[1] + timer_dy, .899)
        display = build_display(stage, assembly + "/Timer", timer_position, length, materials,
                                instrument_label=f"P{index}")
        buttons = build_gate_buttons(stage, assembly, timer_position, f"P{index}", materials, physical_mat)
        result.append({
            "id": f"pendulum_{index}", "index": index, "body_path": body_path,
            "joint_path": joint_path, "anchor_path": anchor_path, "rod_path": str(rod.GetPath()),
            "bob_path": str(bob.GetPath()), "pivot": pivot, "length_m": length,
            "mass_kg": total_mass, "bob_mass_kg": BOB_MASS_KG,
            "rod_mass_kg": ROD_MASS_KG, "bob_radius_m": BOB_RADIUS_M,
            "rod_radius_m": ROD_RADIUS_M, "center_of_mass_local": (0, 0, -com_distance),
            "linear_damping_per_s": LINEAR_DAMPING_PER_S,
            "angular_damping_per_s": ANGULAR_DAMPING_PER_S,
            "diagonal_inertia": diagonal,
            "pivot_inertia_kg_m2": transverse + total_mass * com_distance ** 2,
            "effective_length_m": (transverse + total_mass * com_distance ** 2) / (total_mass * com_distance),
            "bob_local_point": (0, 0, -length), "rest_bob_center": (x, -.18, 1.03),
            "nominal_release_degrees": NOMINAL_RELEASE_DEG,
            "gate_ray_origin": (x, -.2115, gate_z), "gate_ray_direction": (0, 1, 0),
            "gate_ray_distance": .063, "display": display, "buttons": buttons,
            "timer_center": tuple(a + b for a, b in zip(timer_position, layout_offset)),
        })
        descriptor = result[-1]
        for key in ("pivot", "rest_bob_center", "gate_ray_origin"):
            descriptor[key] = tuple(a + b for a, b in zip(descriptor[key], layout_offset))
        descriptor["layout_offset_m"] = layout_offset
        for button in buttons.values():
            button["rest_centre"] = tuple(a + b for a, b in zip(button["rest_centre"], layout_offset))
    return result


def validate_apparatus(stage, descriptors):
    """Check authored physical structure without advancing the simulation."""
    if len(descriptors) != 3:
        raise ValueError("Expected three pendulum assemblies")
    checks = []
    for descriptor in descriptors:
        body = stage.GetPrimAtPath(descriptor["body_path"])
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Missing rigid body: {body.GetPath()}")
        rigid_body = UsdPhysics.RigidBodyAPI(body)
        if rigid_body.GetKinematicEnabledAttr().Get():
            raise ValueError(f"Pendulum is kinematic: {body.GetPath()}")
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(descriptor["joint_path"]))
        if joint.GetAxisAttr().Get() != "Y":
            raise ValueError("Pivot must rotate about the apparatus Y axis")
        anchor_targets = joint.GetBody0Rel().GetTargets()
        if anchor_targets != [Sdf.Path(descriptor["anchor_path"])]:
            raise ValueError("Pivot must reference its local kinematic anchor")
        anchor = stage.GetPrimAtPath(descriptor["anchor_path"])
        if not UsdPhysics.RigidBodyAPI(anchor).GetKinematicEnabledAttr().Get():
            raise ValueError("Pivot anchor must be fixed to the apparatus")
        if tuple(joint.GetLocalPos0Attr().Get()) != (0.0, 0.0, 0.0):
            raise ValueError("Anchor coordinates must remain local for asset placement")
        if body.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ValueError("Pendulum must remain separate from robot articulation")
        if "PhysicsDriveAPI:angular" in joint.GetPrim().GetAppliedSchemas():
            raise ValueError("Pendulum must not have an angular drive")
        physics = PhysxSchema.PhysxRigidBodyAPI(body)
        for attr, expected in ((physics.GetLinearDampingAttr(), LINEAR_DAMPING_PER_S),
                               (physics.GetAngularDampingAttr(), ANGULAR_DAMPING_PER_S)):
            if abs(float(attr.Get()) - expected) > 1e-8:
                raise ValueError("Pendulum drag does not match the validated passive model")
        for key in ("rod_path", "bob_path"):
            if not stage.GetPrimAtPath(descriptor[key]).HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError(f"Missing physical collider: {descriptor[key]}")
        mass = UsdPhysics.MassAPI(body)
        if abs(mass.GetMassAttr().Get() - descriptor["mass_kg"]) > 1e-7:
            raise ValueError("Mass does not match apparatus descriptor")
        if not all(i > 0 for i in mass.GetDiagonalInertiaAttr().Get()):
            raise ValueError("Nonpositive moment of inertia")
        length = descriptor["length_m"]
        gate_offset = descriptor["pivot"][2] - descriptor["gate_ray_origin"][2]
        allowed_envelope = gate_offset * math.tan(math.radians(MAX_RELEASE_DEG)) + ROD_RADIUS_M
        if allowed_envelope >= .038:
            raise ValueError("Gate frame intersects allowed swing envelope")
        # The ray crosses the rod at rest and its endpoints clear both housings.
        if not 0 < gate_offset < length - BOB_RADIUS_M:
            raise ValueError("Optical ray does not target the exposed rod")
        panel = descriptor["display"]
        for label in STATE_WORDS:
            if not stage.GetPrimAtPath(panel["path"] + "/State" + label).IsValid():
                raise ValueError(f"Missing timer state {label}")
        if set(descriptor.get("buttons", {})) != {"start", "reset"}:
            raise ValueError("Each gate needs a START and a RESET button")
        for button in descriptor["buttons"].values():
            joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(button["joint_path"]))
            if not joint or joint.GetBody1Rel().GetTargets() != [Sdf.Path(button["cap_path"])]:
                raise ValueError("Gate button cap must ride on its prismatic joint")
        checks.append({"id": descriptor["id"], "dynamic": True,
                       "length_m": length, "mass_kg": descriptor["mass_kg"],
                       "effective_length_m": descriptor["effective_length_m"],
                       "linear_damping_per_s": LINEAR_DAMPING_PER_S,
                       "angular_damping_per_s": ANGULAR_DAMPING_PER_S,
                       "gate_frame_clearance_at_5_deg_m": .038 - allowed_envelope})
    return checks


def export_apparatus(output_path):
    """Export the asset alone; a consuming environment supplies gravity and table."""
    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    descriptors = build_apparatus(stage, root="/PendulumLab")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/PendulumLab"))
    validate_apparatus(stage, descriptors)
    stage.GetRootLayer().Export(str(output_path))
    return descriptors


__all__ = [
    "BOB_RADIUS_M", "BOB_MASS_KG", "ROD_RADIUS_M", "ROD_MASS_KG",
    "NOMINAL_RELEASE_DEG", "MAX_RELEASE_DEG", "LENGTHS_M",
    "SWITCH_DOWN_M", "SWITCH_UP_M",
    "build_apparatus", "build_display", "build_gate_buttons", "update_display", "validate_apparatus", "export_apparatus",
]

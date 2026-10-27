"""Neutral Hooke apparatus construction; extracted from the validated lab asset.
No experiment policy, perception, planning or robot skills.
"""

import json
import numpy as np
from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,UsdLux

APPARATUS_PATH = "/World/HookeApparatus"

APPARATUS_WORLD_TRANSLATION = np.array([0.65, -0.28, 0.82], dtype=np.float64)

HANGER_PATH = f"{APPARATUS_PATH}/Hanger"

SPRING_PATH = f"{APPARATUS_PATH}/Spring"
REST_LENGTH_M = 0.305

TRAY_PATH = f"{HANGER_PATH}/CumulativeRuntimeTray"

ROBOT_PATH = "/World/Franka"

HAND_PATH = f"{ROBOT_PATH}/panda_hand"

CAMERA_PATHS = (
    "/World/Franka/RoboCasaCameras/robot0_agentview_left",
    "/World/Franka/RoboCasaCameras/robot0_agentview_right",
    f"{HAND_PATH}/robot0_eye_in_hand",
)

WEIGHT_SEQUENCE = (50, 100, 150, 200, 250)

WEIGHTS = {
    grams: f"{APPARATUS_PATH}/Weight50g" if index == 0
    else f"{APPARATUS_PATH}/Weight50g_{index + 1}"
    for index, grams in enumerate(WEIGHT_SEQUENCE)
}

WEIGHT_MASS_KG = 0.050

RUNTIME_TRAY_LOCAL_X = -0.060

RUNTIME_TRAY_LOCAL_Y = 0.110

RUNTIME_TRAY_LOCAL_Z = -0.0831

RUNTIME_TRAY_WIDTH_M = 0.280

RUNTIME_TRAY_DEPTH_M = 0.280

RUNTIME_TRAY_HEIGHT_M = 0.004

TRAY_SLOT_OFFSETS_M = (
    (0.100, 0.100),
    (0.000, 0.100),
    (-0.100, 0.100),
    (-0.050, 0.020),
    (0.050, 0.020),
)

# --- weight box ---------------------------------------------------------------------------------------------------
# The weights start in a weight box: a static block on the table top with six square seats in a grid of 3 along X
# by 2 along Y. Seat k holds the k-th weight (weight_0k) at the start; seats beyond the task's weight count start
# empty. Seats 1-3 form the row nearer the robot (+Y), seats 4-6 the row behind it, each row numbered towards +X;
# each seat's number is printed on the box top on its robot side. Before layout jitter seat 1 stands at world
# (1.05, -0.08): every seat lies 0.43-0.64 m from the robot base axis, where a top-down grasp keeps every joint at
# least 0.47 rad from its limit along the whole vertical approach (the former single row at 0.30-0.48 m left
# joint 4 within 0.05 rad of its limit at seats 1-2). Along every seat's vertical approach the arm stays at least
# 0.10 m from the stand base plate and the fully loaded deck and 0.05 m from the ruler, and the box lies inside the
# left, right and start-pose wrist views.
# Coordinates are in the apparatus frame (origin on the table top); the box moves with the apparatus.
# - Well: 80 mm square, floor 8 mm above the table. A weight is 62.4 mm across its red body (the collider's badge
#   reaches 34.8 mm from the axis on one side), so a weight set down within about 5 mm of the seat centre stands
#   on the floor; the chamfered mouth (92 mm, 53 degrees) guides a slightly wider landing in.
# - Height 22 mm: the red body (30 mm) stands 16 mm proud of the box top, and the fingers, which straddle the
#   handle post 50-90 mm above the seat floor, stay well above the box.
# - Pitch 110 mm along X: the open fingers (80 mm apart, closing along world X) keep the same clearance from the
#   neighbouring weights as the former row. 120 mm along Y: the 28 mm strip between the rows carries the back
#   row's numbers.
WEIGHT_BOX_PATH = f"{APPARATUS_PATH}/WeightBox"
WEIGHT_BOX_BODY_PATH = f"{WEIGHT_BOX_PATH}/Body"
WEIGHT_SEAT_ONE_LOCAL_XY = (0.40, 0.20)
WEIGHT_SEAT_COLUMNS = 3                       # along +X
WEIGHT_SEAT_ROWS = 2                          # along -Y, away from the robot
WEIGHT_SEAT_COUNT = WEIGHT_SEAT_COLUMNS * WEIGHT_SEAT_ROWS
WEIGHT_SEAT_PITCH_M = 0.110                   # along X
WEIGHT_SEAT_ROW_PITCH_M = 0.120               # along Y
WEIGHT_BOX_HEIGHT_M = 0.022
WEIGHT_SEAT_FLOOR_M = 0.008
WEIGHT_SEAT_WELL_HALF_M = 0.040
WEIGHT_SEAT_MOUTH_HALF_M = 0.046
WEIGHT_SEAT_CHAMFER_DEPTH_M = 0.008
WEIGHT_BOX_MARGIN_M = {"end": 0.055, "back": 0.055, "front": 0.083}   # beyond the outer columns; -Y; +Y (label strip)
WEIGHT_BOX_COLOUR = (0.16, 0.25, 0.22)        # dark green-grey: neither the red weights nor the orange-brown deck
WEIGHT_BOX_INK = (0.94, 0.94, 0.91)
WEIGHT_SEAT_LABEL = {"height": 0.020, "width": 0.012, "stroke": 0.0035}
WEIGHT_SEAT_DROP_M = 0.001                    # weights are authored this far above their seat floor and settle


# --- hooke_dynamic_k bench controls (world x, y before layout jitter; heights above the table top) ------------------
# Recorder panel of the tray encoder (the shared builder): keypad console on the robot's right front, below and in
# front of the tray (which hangs high above the table), clear of the stand base plate and the weight box; its display
# on a stand behind the keypad, face towards the robot and the fixed cameras. Damper lever on the keypad's -X side:
# OFF towards the robot, ON 40 mm towards -Y. Every key and the lever post are reached from above (IK checked).
CONTROLS_PANEL_PATH = f"{APPARATUS_PATH}/RecorderPanel"
PANEL_CENTRE_XY = (0.58, 0.11)
PANEL_TOP_H = 0.080                 # console top above the table (key tops 20 mm higher)
DISPLAY_XY = (0.58, -0.05)
DISPLAY_CENTRE_H = 0.160
DISPLAY_STAND = (0.030, 0.016)      # stand post section (x, y) under the display housing
# Live tray field of the recorder display (hooke_dynamic_k only): a strip of the same housing on top of the shared
# recorder display, labelled TRAY Z MM, showing the encoder's latest z_mm (sign, three digits, one decimal) on the
# physical display only; no tool returns it. The shared recorder display (coupled_modes, torsion_inertia, airtrack_collision, incline_friction, torsion_centroid) is not changed.
TRAY_READOUT_PATH = f"{CONTROLS_PANEL_PATH}/TrayReadout"
TRAY_READOUT_H = 0.062              # strip height (housing width and depth as the recorder display)
TRAY_DIGIT_H = 0.034
TRAY_DIGIT_Z = -0.007               # strip-local digit centre (label above it)
TRAY_DIGITS = 5
TRAY_POINT_GAP = 0.006              # extra gap before the decimal digit, for the decimal point
RECORD_S_RANGE = (10, 120)          # record lengths REC accepts (s), the tray encoder's former arm range
LEVER_PATH = f"{APPARATUS_PATH}/DamperLever"
# Light detent (1 N mid-way), as on coupled_modes: this bench's fingers press with at most 3 N each, so a firm parallel grip on
# the post must carry the carriage across by friction or by a finger's push.
LEVER = dict(name="Damper", detent0=(0.45, 0.17), direction=(0.0, -1.0), spacing=0.04, labels=("OFF", "ON"),
             stiffness=50.0, damping=5.0)
LEVER_TOP_H = 0.030                 # pedestal top above the table: the post top stays below a hand pressing keys


def weight_seat_local(index):
    """Centre (x, y) of seat `index` + 1 (0-based; the seat weight `index` starts in) in the apparatus frame."""
    row, column = divmod(index, WEIGHT_SEAT_COLUMNS)
    return (WEIGHT_SEAT_ONE_LOCAL_XY[0] + column * WEIGHT_SEAT_PITCH_M,
            WEIGHT_SEAT_ONE_LOCAL_XY[1] - row * WEIGHT_SEAT_ROW_PITCH_M)


def weight_seat_path(index):
    """Marker Xform at the centre of the floor of seat `index` + 1 (follows the layout jitter)."""
    return f"{WEIGHT_BOX_PATH}/Seat{index + 1}"


def weight_box_extent():
    """(x0, y0, x1, y1) of the weight box in the apparatus frame."""
    margin = WEIGHT_BOX_MARGIN_M
    xs = [weight_seat_local(i)[0] for i in range(WEIGHT_SEAT_COUNT)]
    ys = [weight_seat_local(i)[1] for i in range(WEIGHT_SEAT_COUNT)]
    return min(xs) - margin["end"], min(ys) - margin["back"], max(xs) + margin["end"], max(ys) + margin["front"]


def _oriented(points, hint):
    """The quad's vertices ordered so that its normal points along `hint`."""
    a, b, c = (np.asarray(p, dtype=np.float64) for p in points[:3])
    return list(points) if float(np.dot(np.cross(b - a, c - a), hint)) > 0 else list(points)[::-1]


def _quads_mesh(stage, path, quads, colour):
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = [p for quad in quads for p in quad]
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
    mesh.CreateFaceVertexCountsAttr([4] * len(quads))
    mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
    mesh.CreateSubdivisionSchemeAttr("none")
    array = np.asarray(points, dtype=np.float64)
    mesh.CreateExtentAttr([Gf.Vec3f(*array.min(axis=0).tolist()), Gf.Vec3f(*array.max(axis=0).tolist())])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    return mesh


# Seven-segment digits (u along the reading direction, v up the glyph), as (u0, v0, u1, v1) in units of the glyph.
_SEGMENTS = {"a": (0, 1, 1, 1), "b": (1, 0.5, 1, 1), "c": (1, 0, 1, 0.5), "d": (0, 0, 1, 0),
             "e": (0, 0, 0, 0.5), "f": (0, 0.5, 0, 1), "g": (0, 0.5, 1, 0.5)}
_DIGITS = {"2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc"}


def _digit_rectangles(text):
    """Stroke rectangles (u0, v0, u1, v1) in metres of a one-digit label centred on the origin."""
    h, w, t = WEIGHT_SEAT_LABEL["height"], WEIGHT_SEAT_LABEL["width"], WEIGHT_SEAT_LABEL["stroke"]
    if text == "1":
        return [(-t / 2, -h / 2, t / 2, h / 2)]
    rects = []
    for name in _DIGITS[text]:
        u0, v0, u1, v1 = _SEGMENTS[name]
        x0, x1 = -w / 2 + u0 * (w - t), -w / 2 + u1 * (w - t) + t
        y0, y1 = -h / 2 + v0 * (h - t), -h / 2 + v1 * (h - t) + t
        rects.append((x0, y0, x1, y1))
    return rects


def build_weight_box(stage):
    """Author the weight box (static collider, seat markers and printed numbers)."""
    from pxr import PhysxSchema, UsdShade
    x0, y0, x1, y1 = weight_box_extent()
    top, floor = WEIGHT_BOX_HEIGHT_M, WEIGHT_SEAT_FLOOR_M
    mouth, well = WEIGHT_SEAT_MOUTH_HALF_M, WEIGHT_SEAT_WELL_HALF_M
    throat = top - WEIGHT_SEAT_CHAMFER_DEPTH_M
    centres = [weight_seat_local(i) for i in range(WEIGHT_SEAT_COUNT)]
    columns = sorted({cx for cx, _ in centres})
    rows = sorted({sy for _, sy in centres})          # back row (-Y) first
    up = np.array([0.0, 0.0, 1.0])
    quads = []

    def quad(points, hint):
        quads.append(_oriented(points, np.asarray(hint, dtype=np.float64)))

    # Top: full-width strips behind, between and in front of the rows of mouths, and the bars between the mouths.
    bands = [y0] + [v for sy in rows for v in (sy - mouth, sy + mouth)] + [y1]
    for a, b in zip(bands[0::2], bands[1::2]):
        quad([(x0, a, top), (x1, a, top), (x1, b, top), (x0, b, top)], up)
    edges = [x0] + [v for cx in columns for v in (cx - mouth, cx + mouth)] + [x1]
    for sy in rows:
        for a, b in zip(edges[0::2], edges[1::2]):
            quad([(a, sy - mouth, top), (b, sy - mouth, top), (b, sy + mouth, top), (a, sy + mouth, top)], up)
    # Outer walls and bottom.
    quad([(x0, y0, 0), (x1, y0, 0), (x1, y0, top), (x0, y0, top)], (0, -1, 0))
    quad([(x0, y1, 0), (x1, y1, 0), (x1, y1, top), (x0, y1, top)], (0, 1, 0))
    quad([(x0, y0, 0), (x0, y1, 0), (x0, y1, top), (x0, y0, top)], (-1, 0, 0))
    quad([(x1, y0, 0), (x1, y1, 0), (x1, y1, top), (x1, y0, top)], (1, 0, 0))
    quad([(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)], -up)
    # Seats: chamfered mouth, well walls and floor, all facing into the well.
    for cx, sy in centres:
        def ring(half, z):
            return [(cx - half, sy - half, z), (cx + half, sy - half, z), (cx + half, sy + half, z), (cx - half, sy + half, z)]
        rings = [ring(mouth, top), ring(well, throat), ring(well, floor)]
        for upper, lower in zip(rings, rings[1:]):
            for i in range(4):
                j = (i + 1) % 4
                points = [upper[i], upper[j], lower[j], lower[i]]
                middle = np.mean(np.asarray(points, dtype=np.float64), axis=0)
                quad(points, (cx - middle[0], sy - middle[1], 0.01))
        quad(ring(well, floor), up)
    box = UsdGeom.Xform.Define(stage, WEIGHT_BOX_PATH)
    body = _quads_mesh(stage, WEIGHT_BOX_BODY_PATH, quads, WEIGHT_BOX_COLOUR)
    # Static triangle-mesh collider (no rigid body: it never moves).
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(body.GetPrim()).CreateApproximationAttr("none")
    contact = PhysxSchema.PhysxCollisionAPI.Apply(body.GetPrim())
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    material = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/HookeWeightBox")
    surface = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    surface.CreateStaticFrictionAttr(0.65)
    surface.CreateDynamicFrictionAttr(0.50)
    surface.CreateRestitutionAttr(0.02)
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(material, materialPurpose="physics")
    # Printed seat numbers (render only) on the strip on each seat's robot side (+Y): the front strip for the front
    # row, the strip between the rows for the back row. They read upright from the robot: the glyphs' up is -Y and
    # they read towards -X (the robot's right).
    for index, (cx, sy) in enumerate(centres):
        label_y = sy + (mouth + WEIGHT_BOX_MARGIN_M["front"]) / 2 if sy == rows[-1] else sy + WEIGHT_SEAT_ROW_PITCH_M / 2
        glyph = []
        for u0, v0, u1, v1 in _digit_rectangles(str(index + 1)):
            corners = [(cx - u, label_y - v, top + 0.0004) for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
            glyph.append(_oriented(corners, up))
        _quads_mesh(stage, f"{WEIGHT_BOX_PATH}/Label{index + 1}", glyph, WEIGHT_BOX_INK)
        seat = UsdGeom.Xform.Define(stage, weight_seat_path(index))
        UsdGeom.XformCommonAPI(seat).SetTranslate(Gf.Vec3d(cx, sy, floor))
    return box

def build_encoder_controls(stage):
    """hooke_dynamic_k: the recorder panel (keypad and display) and the damper lever under the apparatus root, in its
    frame, before the layout places the bench. Returns the lever info (its detent0 in the apparatus frame; the
    runtime measures the carriage after the warm-up)."""
    from pxr import PhysxSchema, UsdShade
    from coupled_oscillator.controls_asset import DISPLAY_HOUSING, build_recorder_panel
    from damper_lever import build_lever
    from pendulum_asset import _box, _material
    world = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(APPARATUS_PATH))
    origin = np.asarray(world.ExtractTranslation(), float)
    if not np.allclose(np.asarray(world.ExtractRotationMatrix(), float), np.eye(3), atol=1e-9):
        raise RuntimeError("the Hooke apparatus root must be a pure translation")
    material = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/HookeControls")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(0.8)
    api.CreateDynamicFrictionAttr(0.7)
    api.CreateRestitutionAttr(0.0)

    def local(xy):
        return (xy[0] - origin[0], xy[1] - origin[1])

    dx, dy = local(DISPLAY_XY)
    build_recorder_panel(stage, CONTROLS_PANEL_PATH, centre_xy=local(PANEL_CENTRE_XY), top_z=PANEL_TOP_H, base_z=0.0,
                         contact_material=material, display_centre=(dx, dy, DISPLAY_CENTRE_H),
                         spec_text=f"{RECORD_S_RANGE[0]}-{RECORD_S_RANGE[1]} S")
    stand_top = DISPLAY_CENTRE_H - DISPLAY_HOUSING[2] / 2
    dark = _material(stage, CONTROLS_PANEL_PATH + "/Materials/Stand", (.10, .11, .13), .3)
    stand = _box(stage, CONTROLS_PANEL_PATH + "/DisplayStand", (dx, dy, stand_top / 2), (*DISPLAY_STAND, stand_top), dark)
    UsdPhysics.CollisionAPI.Apply(stand.GetPrim())
    contact = PhysxSchema.PhysxCollisionAPI.Apply(stand.GetPrim())
    contact.CreateContactOffsetAttr().Set(.0005)
    contact.CreateRestOffsetAttr().Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(stand.GetPrim()).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    _box(stage, CONTROLS_PANEL_PATH + "/DisplayFoot", (dx, dy, 0.003), (0.10, 0.06, 0.006), dark)
    _build_tray_readout(stage, dx, dy)
    info = build_lever(stage, LEVER_PATH, contact_material=material, detent0=local(LEVER["detent0"]),
                       top_z=LEVER_TOP_H, base_z=0.0, **{k: v for k, v in LEVER.items() if k != "detent0"})
    # The rail joint's body0 is the world: its frame is a world pose, not an apparatus-frame one.
    joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(info["joint"]))
    pos = joint.GetLocalPos0Attr().Get()
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(pos[0] + origin[0], pos[1] + origin[1], pos[2] + origin[2]))
    return info


def _build_tray_readout(stage, dx, dy):
    """hooke_dynamic_k: the live tray field (TRAY Z MM) on top of the recorder display housing, in the apparatus
    frame; same housing, print and LED materials as the recorder display (read from the built panel, not changed)."""
    from pxr import UsdShade
    from coupled_oscillator.controls_asset import DISPLAY_HOUSING, _ensure_glyphs
    from pendulum_asset import _box, _digits, _text
    _ensure_glyphs()
    from coupled_oscillator.controls_asset import LED_ON
    from pendulum_asset import _material
    mats = {k: UsdShade.Material.Get(stage, f"{CONTROLS_PANEL_PATH}/Materials/{name}")
            for k, name in (("housing", "RecorderHousing"), ("white", "ControlsPrint"))}
    if not all(mats.values()):
        raise RuntimeError("recorder panel materials missing: build the recorder panel first")
    # lit segments, as the recorder's (LED_ON, full emission); an unlit segment is hidden (the face is dark)
    mats["led_on"] = _material(stage, TRAY_READOUT_PATH + "/Materials/LedOn", LED_ON, emission=1.0)
    strip = UsdGeom.Xform.Define(stage, TRAY_READOUT_PATH)
    UsdGeom.XformCommonAPI(strip).SetTranslate(
        Gf.Vec3d(dx, dy, DISPLAY_CENTRE_H + DISPLAY_HOUSING[2] / 2 + TRAY_READOUT_H / 2))
    _box(stage, TRAY_READOUT_PATH + "/Housing", (0.0, 0.0, 0.0),
         (DISPLAY_HOUSING[0], DISPLAY_HOUSING[1], TRAY_READOUT_H), mats["housing"])
    face_y = DISPLAY_HOUSING[1] / 2 + .002
    _text(stage, TRAY_READOUT_PATH + "/Label", "TRAY Z MM", (0.0, face_y, .021), .008, mats["white"])
    _digits(stage, TRAY_READOUT_PATH + "/Digits", (0.0, face_y, TRAY_DIGIT_Z), TRAY_DIGITS, TRAY_DIGIT_H,
            mats["led_on"], separator_after=TRAY_DIGITS - 1, separator_gap=TRAY_POINT_GAP)
    spacing, stroke = TRAY_DIGIT_H * .68, TRAY_DIGIT_H * .105
    # decimal point: midway between the last two cells (pendulum_asset._digits places cell i at
    # (count - 1) * spacing / 2 - i * spacing, readable from +Y), on the bottom stroke line
    point_x = (TRAY_DIGITS - 1) * spacing / 2 - (TRAY_DIGITS - 1.5) * spacing
    _box(stage, TRAY_READOUT_PATH + "/Digits/DP", (point_x, face_y, TRAY_DIGIT_Z - TRAY_DIGIT_H / 2),
         (stroke, .0005, stroke), mats["led_on"])
    _TRAY_SHOWN.pop(str(stage.GetRootLayer().identifier), None)
    update_tray_readout(stage, None)            # blank until the encoder's first sample


def tray_readout_text(latest):
    """Cells and decimal point of the live tray field. latest: the encoder's latest sample (None before the first);
    its z_mm is None while the tray is outside the encoder range (dashes)."""
    if latest is None:
        return " " * TRAY_DIGITS, False
    z = latest.get("z_mm")
    if z is None:
        return " " + "-" * (TRAY_DIGITS - 1), False
    z = 0.0 if abs(z) < 0.05 else float(z)
    return f"{z:.1f}".replace(".", "").rjust(TRAY_DIGITS)[-TRAY_DIGITS:], True


_TRAY_SHOWN = {}     # stage root layer -> {segment path: shown}; only changed segments are written


def update_tray_readout(stage, latest):
    """Show a reading on the live tray field (segment visibility only; materials never change). True if it changed."""
    from pendulum_asset import _SEGMENTS
    text, point = tray_readout_text(latest)
    want = {f"{TRAY_READOUT_PATH}/Digits/D{index}_{segment}": segment in _SEGMENTS[char]
            for index, char in enumerate(text) for segment in "abcdefg"}
    want[TRAY_READOUT_PATH + "/Digits/DP"] = point
    shown = _TRAY_SHOWN.setdefault(str(stage.GetRootLayer().identifier), {})
    changed = False
    for path, on in want.items():
        if shown.get(path) is not on:
            UsdGeom.Imageable(stage.GetPrimAtPath(path)).GetVisibilityAttr().Set(
                UsdGeom.Tokens.inherited if on else UsdGeom.Tokens.invisible)
            shown[path] = on
            changed = True
    return changed


def set_collision(stage, path: str, enabled: bool) -> None:
    prim = stage.GetPrimAtPath(path)
    for child in Usd.PrimRange(prim):
        coll = UsdPhysics.CollisionAPI.Get(stage, child.GetPath())
        if coll:
            coll.CreateCollisionEnabledAttr().Set(enabled)

def tray_deck_top_z(stage) -> float:
    """Composed deck top; physics writes its measured transforms to USD."""
    tray = stage.GetPrimAtPath(TRAY_PATH)
    if not tray.IsValid():
        return 0.0
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    world = cache.ComputeWorldBound(tray).ComputeAlignedRange()
    if not world.IsEmpty():
        return float(world.GetMax()[2])
    # Fallback: compose the hanger-frame model if the bound is unavailable so a
    # payload is never simply dropped onto the floor.
    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    parent_world = xf_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(HANGER_PATH))
    return float(parent_world[2][3]) + RUNTIME_TRAY_LOCAL_Z + RUNTIME_TRAY_HEIGHT_M / 2.0

def set_kinematic(stage, path: str, enabled: bool) -> None:
    prim = stage.GetPrimAtPath(path)
    body = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
    if not body:
        body = UsdPhysics.RigidBodyAPI.Apply(prim)
    if enabled:
        ccd = prim.GetAttribute("physxRigidBody:enableCCD")
        if ccd.IsValid():
            ccd.Set(False)
    body.CreateKinematicEnabledAttr().Set(enabled)

def prepare_cumulative_weights(stage) -> None:
    """Replace the authored 50/100/200 set with five identical 50 g bodies.

    The packaged scene deliberately remains unchanged so the pre-change Git
    commit is a complete rollback point.  At runtime the first 50 g body is
    reused and four internal USD references are created from it.  This keeps
    the visual geometry identical while giving each payload an independent
    rigid body and table pose.
    """
    source_path = WEIGHTS[50]
    source = stage.GetPrimAtPath(source_path)
    if not source.IsValid():
        raise RuntimeError(f"missing source weight prim: {source_path}")
    # Blender joined the text into Weight50gMesh: it is not a separate Label
    # prim in the packaged USD.  Its faces are exactly the white-enamel
    # material subset.  Mark them as holes without touching the body, brass
    # ring, topology numbering, or root-to-bottom calibration.  Clones below
    # inherit the same edit through their internal USD reference.
    for child in Usd.PrimRange(source):
        if child.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(child)
            hidden_faces = set(mesh.GetHoleIndicesAttr().Get() or [])
            for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh):
                if subset.GetPrim().GetName() == "White_scale_enamel":
                    hidden_faces.update(subset.GetIndicesAttr().Get())
            if hidden_faces:
                mesh.CreateHoleIndicesAttr().Set(sorted(hidden_faces))
    for old in (f"{APPARATUS_PATH}/Weight100g", f"{APPARATUS_PATH}/Weight200g"):
        prim = stage.GetPrimAtPath(old)
        if prim.IsValid():
            prim.SetActive(False)
    # Every weight starts in its seat of the weight box (seat k holds weight k; the other seats stay empty), its
    # bottom WEIGHT_SEAT_DROP_M above the seat floor; it settles there before the episode starts.
    build_weight_box(stage)
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    for index, grams in enumerate(WEIGHT_SEQUENCE):
        path = WEIGHTS[grams]
        prim = stage.GetPrimAtPath(path)
        if index > 0:
            if prim.IsValid():
                stage.RemovePrim(path)
            prim = stage.DefinePrim(path, "Xform")
            prim.GetReferences().AddInternalReference(Sdf.Path(source_path))
        if not prim.IsValid():
            raise RuntimeError(f"unable to create cumulative weight prim: {path}")
        seat_x, seat_y = weight_seat_local(index)
        bottom = float(bounds.ComputeUntransformedBound(prim).ComputeAlignedRange().GetMin()[2])
        UsdGeom.XformCommonAPI(prim).SetTranslate(
            Gf.Vec3d(seat_x, seat_y, WEIGHT_SEAT_FLOOR_M + WEIGHT_SEAT_DROP_M - bottom))
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(WEIGHT_MASS_KG)
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(child)
                UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr("convexDecomposition")
            # The authored weight mesh contains a "50 g" text badge.  The
            # demo already identifies loads through the event log; keep the
            # physical weight visually unlabelled in all three views.
            if "Label" in child.GetName() or "label" in child.GetName():
                child.SetActive(False)
        # Hold before World.reset and camera warm-up, not after them: otherwise
        # contact impulses can displace a payload before its source is recorded.
        set_kinematic(stage, path, True)
        set_collision(stage, path, False)

def prepare_runtime_tray(stage) -> None:
    """Add a generously sized, visible tray deck for the five-slot demo.

    The square 28 cm deck matches the retained review video. Its
    audited five-slot layout leave room for the payloads and the central
    stem.  It follows the hanger; the old extension and fence are disabled.
    """
    # The effective payload footprint includes the 1.2 mm authored mesh
    # offset.  Check both pairwise clearance and the actual hanger stem,
    # rather than only checking that each centre lies on the deck.
    radius = 0.0312
    stem_xy = np.array([-0.0195 - RUNTIME_TRAY_LOCAL_X, -RUNTIME_TRAY_LOCAL_Y])
    slots = np.asarray(TRAY_SLOT_OFFSETS_M)
    pair_gap = min(np.linalg.norm(a - b) - 2 * radius
                   for i, a in enumerate(slots) for b in slots[i + 1:])
    rim_gap = min(min(RUNTIME_TRAY_WIDTH_M / 2 - abs(s[0]),
                      RUNTIME_TRAY_DEPTH_M / 2 - abs(s[1])) - radius for s in slots)
    stem_gap = min(np.linalg.norm(s - stem_xy) - radius - 0.008 for s in slots)
    # The hand needs ~10 cm from the stem centre during a vertical descent.
    if pair_gap < 0.002 or rim_gap < 0.003 or stem_gap < 0.060:
        raise RuntimeError(f"unsafe tray layout: gaps={pair_gap, rim_gap, stem_gap}")
    print("APPARATUS_TRAY_CLEARANCES=" + json.dumps({
        "between_weights_m": float(pair_gap), "rim_m": float(rim_gap),
        "stem_m": float(stem_gap),
    }), flush=True)
    authored_tray = stage.GetPrimAtPath(f"{HANGER_PATH}/TrayExtension")
    if authored_tray.IsValid():
        authored_tray.SetActive(False)
    # The authored tray fence is made of twelve small posts.  They are useful
    # for the original single-load asset but visually read as blue pegs in
    # this cumulative demo and can occlude the retained weights.
    for index in range(12):
        fence = stage.GetPrimAtPath(f"{HANGER_PATH}/TrayFence{index:02d}")
        if fence.IsValid():
            fence.SetActive(False)
    path = f"{HANGER_PATH}/CumulativeRuntimeTray"
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)
    tray = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    tray.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(tray)
    xform.SetTranslate(Gf.Vec3d(RUNTIME_TRAY_LOCAL_X, RUNTIME_TRAY_LOCAL_Y, RUNTIME_TRAY_LOCAL_Z))
    xform.SetScale(Gf.Vec3f(RUNTIME_TRAY_WIDTH_M, RUNTIME_TRAY_DEPTH_M, RUNTIME_TRAY_HEIGHT_M))
    tray.CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.36, 0.045)])
    tray.CreateDisplayOpacityAttr([1.0])
    UsdPhysics.CollisionAPI.Apply(tray.GetPrim())

def setup_neutral_lab_lighting(stage) -> None:
    """Add neutral laboratory illumination without changing any camera pose."""
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DomeLight):
            # The flattened asset references an EXR on its author's machine.
            UsdLux.DomeLight(prim).GetTextureFileAttr().Set(Sdf.AssetPath(''))
    lights = stage.GetPrimAtPath("/World/Lights")
    if not lights.IsValid():
        stage.DefinePrim("/World/Lights", "Scope")
    # Tone down the old blue/orange stylization when those authored lights are
    # present in the flattened scene.
    blue = stage.GetPrimAtPath("/World/Lights/BlueFill")
    if blue.IsValid():
        blue_api = UsdLux.DomeLight.Get(stage, blue.GetPath())
        blue_api.CreateIntensityAttr(260.0)
        blue_api.CreateColorAttr(Gf.Vec3f(0.82, 0.86, 0.90))
    warm = stage.GetPrimAtPath("/World/Lights/WarmRim")
    if warm.IsValid():
        warm_api = UsdLux.SphereLight.Get(stage, warm.GetPath())
        warm_api.CreateIntensityAttr(260.0)
        warm_api.CreateColorAttr(Gf.Vec3f(1.0, 0.90, 0.78))
    key = UsdLux.RectLight.Define(stage, "/World/Lights/NeutralTopKey")
    key.CreateIntensityAttr(4200.0)
    key.CreateWidthAttr(2.8)
    key.CreateHeightAttr(2.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.94))
    UsdGeom.XformCommonAPI(key).SetTranslate(Gf.Vec3d(0.70, 0.10, 3.10))
    front = UsdLux.RectLight.Define(stage, "/World/Lights/NeutralFrontFill")
    front.CreateIntensityAttr(1500.0)
    front.CreateWidthAttr(2.0)
    front.CreateHeightAttr(1.4)
    front.CreateColorAttr(Gf.Vec3f(0.90, 0.94, 1.0))
    # A dome provides direction-independent fill and avoids a dark eye-in-hand
    # camera while the top key supplies the characteristic soft shadows.
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/NeutralDome")
    dome.CreateIntensityAttr(180.0)
    dome.CreateColorAttr(Gf.Vec3f(0.72, 0.76, 0.80))

def update_spring_visual(stage, hanger, initial_z, advance_dt=0.0):
    """Draw the coil from the measured body pose; never command the hanger."""
    hanger_z = float(hanger.get_world_pose()[0][2])
    extension = initial_z - hanger_z
    spring = stage.GetPrimAtPath(SPRING_PATH)
    top_z = float(APPARATUS_WORLD_TRANSLATION[2] + 0.955)
    eye_z = hanger_z + 0.086
    UsdGeom.XformCommonAPI(spring).SetScale(
        Gf.Vec3f(1.0, 1.0, max(0.1, (top_z - eye_z) / REST_LENGTH_M)))
    return extension

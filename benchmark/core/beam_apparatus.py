"""Beam bench apparatus (beam_compliance): a clamped metal strip with four knife-edge load saddles, one digital
indicator on a carriage that runs on a rail under the strip, and a weight rack. Built with the chemistry kit's helpers on
the same root, so the chemistry backend's grasp, seat and button machinery runs it unchanged.

World layout (the robot stands at (0.95, 0.35) facing -y): the strip runs along world x at y = BEAM_Y, its clamp at
x = CLAMP_X (the strip's root), the free end 0.415 m further. The strip collides: it is SEGMENTS kinematic collider
pieces that the backend moves to the computed deflected shape (position and slope) every physics tick, so a weight can
stand anywhere on it and the robot can press on it. Each load seat is a knife-edge saddle riding on the strip: a level
block with an orange collar, resting on a steel edge across the strip at the seat's distance (a kinematic body moved
with the strip there); whatever presses on the saddle loads the strip at its edge. A rubber stop under the free end
limits its travel.

The indicator: a rail on the table in front of the strip, parallel to it, with detent holes every 50 mm (50 to 400 mm
from the clamp edge), each hole's distance printed on a white strip in front of the rail. The carriage is a dynamic
body on a prismatic rail joint (CARRIAGE_JOINT; a saturated damper drive is the rail friction, the backend adds the ball
detent's force). It carries the indicator head (housing, seven-segment LCD tilted toward the robot), an arm reaching
under the strip with the plunger (render only: it touches nothing, the reading is computed), an index pointer over the
hole numbers, and a tongue toward the robot with an orange push handle and the green ZERO cap (a dynamic cap on a
spring-returned prismatic joint to the carriage). The weight rack (three teal collars) stands behind the strip; the
weights are labelled steel blocks with a white rim handle, with their real masses.
"""
from __future__ import annotations

import math

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

import chem_apparatus as C
from chem_apparatus import (CUVETTE, FUNNEL_POST_R, FUNNEL_RIM_H, FUNNEL_RIM_R, GLYPHS, PORT_H, PORT_R, ROOT,
                            SEAT_FLOOR, SEGMENT_EMISSION, SEGMENT_ON, SEGMENTS as SEVEN, _bind,
                            _render_material, _rigid, _text, collar)
from circuit_apparatus import label
from torsion_apparatus import TABLE_TOP, _box, _collider, _cylinder, _material, repair_cube_extents, \
    retire_hooke_apparatus, table_top_z

CLAMP_X, BEAM_Y = 0.70, -0.11
BEAM_TOP = 0.120                     # strip top above the table, unloaded
BEAM = (0.415, 0.030, 0.004)         # length past the clamp (incl. 15 mm beyond the last seat), width, thickness
SADDLE = (0.046, 0.046, 0.008)
KNIFE_H = 0.004                      # the saddle block stands this high on its knife edge
KNIFE = (0.0015, 0.036)              # edge thickness along the strip, length across it
SEGMENTS = 40                        # collider pieces of the strip (each follows the deflection and slope at its centre)
SEGMENT_OVERLAP = 0.0004
STRIP_CONTACT_OFFSET = 0.002
STOP_GAP = 0.030                     # rubber stop under the free end: its top this far under the strip's underside
STOP_X = 0.413                       # clear of the plunger at the 400 mm hole
RACK = {'xs': (0.82, 0.92, 1.02), 'y': -0.215, 'base': (0.30, 0.07, 0.012)}   # behind the strip: the robot side
# stays clear for the carriage's handle and the leaned-in LCD views (in front of the strip, 40 mm closer to the robot
# than before, the arm could not lower a weight into the middle collar)
WEIGHT_FOOT = (CUVETTE[0], CUVETTE[1], 0.020)
WEIGHT_BODY = (0.014, 0.024)         # steel cylinder: radius, height
WEIGHT_POST, WEIGHT_STEM = 0.012, 0.014

# The indicator rail and its carriage (world y, heights above the table; x along the strip).
HOLES_MM = (50, 100, 150, 200, 250, 300, 350, 400)
RAIL_Y = -0.048                      # rail centre line
RAIL = (0.015, 0.435, 0.024, 0.010)  # from, to (distance from the clamp edge), width, height
HOLE_R = 0.0035
NUMBER_Y, NUMBER_H = -0.0225, 0.0060  # hole numbers on the white strip in front of the rail
NUMBER_STRIP = (-0.032, -0.013)
CARRIAGE_GAP = 0.0005                # carriage underside above the rail top (the rail joint carries it)
CARRIAGE_Z = RAIL[3] + CARRIAGE_GAP  # carriage origin above the table
CARRIAGE_BLOCK = (0.050, 0.030, 0.016)
CARRIAGE_MASS = 0.50
TONGUE_DX = 0.025                    # tongue, handle and ZERO cap: this far toward the free end from the index
TONGUE = (0.016, 0.004, 0.010)       # width (x), from-to heights above the carriage origin
TONGUE_Y = (-0.033, 0.040)           # world y span; its outer end widens into the ZERO plate
ZERO_PLATE = (0.032, -0.010)         # plate width (x) and its inner end (world y)
HANDLE = (0.014, 0.014)
HANDLE_Y, HANDLE_TOP = -0.005, 0.085
ZERO_Y = 0.024
POINTER = (0.0016, 0.009)            # index pointer: width, reach in front of the block (render only)
TRAVEL_MARGIN_M = 0.003              # end stops this far beyond the first and last hole
# Rail friction: the joint drive is a saturated damper (as the optics riders): a force below RAIL_FRICTION_N moves the
# carriage only by a slow creep (F / damping), a push above it slides it. The ball detent (beam_backend) adds a force
# toward a hole within DETENT_WINDOW_M of it: DETENT_STIFFNESS * offset, at most DETENT_PEAK_N (the push it takes to
# leave a hole); released inside the window the carriage clicks into the hole.
RAIL_FRICTION_N = 2.0
RAIL_FRICTION_DAMPING = 5.0e3       # N s/m: saturates above 0.4 mm/s
DETENT_WINDOW_M = 0.005              # the ball catches the carriage within this of a hole (the end stops lie inside)
DETENT_STIFFNESS = 1.0e4             # N/m
DETENT_PEAK_N = 4.0                  # with the rail friction: about 6 N to leave a hole

# The indicator head on the carriage: a housing, its face (the LCD) tilted toward the robot.
IND_Y = BEAM_Y + 0.065               # face centre, world y
IND_Z = 0.068                        # face centre above the table
IND_TILT_DEG = 45.0                  # face normal from vertical, toward the robot (+y)
IND_FACE = (0.068, 0.040)            # face width (along the strip) x height
IND_DEPTH = 0.024
LCD = (0.060, 0.020)                 # LCD glass on the face, centred IND_LCD_V below the face centre
IND_LCD_V = -0.003
DIGIT_H = 0.0115                     # seven-segment cell height
LCD_CELLS = 6                        # sign, tens, units | three decimals (the point after the units cell)
LCD_POINT_AFTER = 2
LCD_GLASS = (0.004, 0.018, 0.010)
HIDE_M = 0.004                       # an unlit segment sinks into the housing
ARM_Z = 0.085                        # plunger arm height above the table
PLUNGER_LEN = 0.040
CARRIAGE = ROOT + '/Carriage'
HEAD = CARRIAGE + '/Head'
PLUNGER = CARRIAGE + '/Plunger'
CARRIAGE_JOINT = C.PHYSICS_ROOT + '/CarriageRail'
ZERO_JOINT = C.PHYSICS_ROOT + '/ZeroKey'
ZERO_ID = 'zero'


def hole_x(mm):
    """World x of a hole (its distance from the clamp edge in mm)."""
    return CLAMP_X + mm / 1000.0


def seat_path(k):
    return f'{ROOT}/Saddle{k}'


def weight_path(w):
    return f'{ROOT}/Weight{w}'


def segment_path(i):
    return f'{ROOT}/Strip/S{i:02d}'


def segment_x(i):
    """Centre of collider piece i, distance from the clamp."""
    return (i + 0.5) * BEAM[0] / SEGMENTS


def carriage_path():
    return CARRIAGE


def seat_paths(stage=None):
    return {}                         # beam seats move (the backend tracks them); rack seats are static


def button_layout():
    return {}                         # the ZERO cap rides on the carriage (author_joints)


def handle_top_local():
    return WEIGHT_FOOT[2] + WEIGHT_BODY[1] + WEIGHT_POST + WEIGHT_STEM + FUNNEL_RIM_H + PORT_H


def latch_local():
    return (0.0, 0.0, handle_top_local() + 0.020)


def saddle_floor_local():
    """Collar well floor above the saddle's origin (the strip top at the knife edge)."""
    return KNIFE_H + SADDLE[2] + SEAT_FLOOR


def _weight(stage, w, label_text, xy, bottom_z, mass_kg, materials):
    path = weight_path(w)
    xform = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(xy[0], xy[1], bottom_z))
    fz = WEIGHT_FOOT[2]
    _box(stage, path + '/Foot', WEIGHT_FOOT, (0, 0, fz / 2), (0.30, 0.31, 0.33), material=materials['cell'])
    r, h = WEIGHT_BODY
    _cylinder(stage, path + '/Body', r, h, (0, 0, fz + h / 2), (0.46, 0.48, 0.52), materials["pmma"])
    z = fz + h
    _cylinder(stage, path + '/Post', 0.004, WEIGHT_POST, (0, 0, z + WEIGHT_POST / 2), (0.25, 0.26, 0.28), materials['pmma'])
    z += WEIGHT_POST
    white = (0.94, 0.94, 0.92)
    _cylinder(stage, path + '/HandleStem', FUNNEL_POST_R, WEIGHT_STEM, (0, 0, z + WEIGHT_STEM / 2), white, materials['pmma'])
    z += WEIGHT_STEM
    _cylinder(stage, path + '/HandleRim', FUNNEL_RIM_R, FUNNEL_RIM_H, (0, 0, z + FUNNEL_RIM_H / 2), white, materials['pmma'])
    z += FUNNEL_RIM_H
    _cylinder(stage, path + '/HandlePort', PORT_R, PORT_H, (0, 0, z + PORT_H / 2), white, collide=False)
    _text(stage, path + '/Label', label_text, (0.0, 0.0), z + PORT_H + 0.0004, 0.007)
    _rigid(xform.GetPrim(), float(mass_kg), (0, 0, fz + h / 2), ccd=True)
    return path


def _kinematic(prim):
    """A kinematic body whose contacts are reported (every contact, no force threshold)."""
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)
    report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
    report.CreateThresholdAttr(0.0)


def _strip_collider(prim):
    contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    contact.CreateContactOffsetAttr(STRIP_CONTACT_OFFSET)
    contact.CreateRestOffsetAttr(0.0)


# --- digital indicator ------------------------------------------------------------------------------------------
def _cell_u(i):
    pitch = DIGIT_H * 0.72
    gap = DIGIT_H * 0.2                                 # the point stands in this extra gap
    width = LCD_CELLS * pitch + gap
    return -width / 2 + i * pitch + (gap if i > LCD_POINT_AFTER else 0.0)


def lcd_rects():
    """Face-local rectangles of the LCD: {'segments': {(cell, seg): ((u, v), (du, dv))}, 'point': ((u, v), size)}.
    u runs along the reading direction (world -x), v up the tilted face."""
    w, h, t = DIGIT_H * 0.5, DIGIT_H, DIGIT_H * 0.14
    v0 = IND_LCD_V - h / 2
    segments = {(i, s): ((_cell_u(i) + fu * w, v0 + fv * h), (w, t) if horizontal else (t, h / 2))
                for i in range(LCD_CELLS) for s, (fu, fv, horizontal) in SEVEN.items()}
    pitch = DIGIT_H * 0.72
    u = _cell_u(LCD_POINT_AFTER) + w + (pitch - w + DIGIT_H * 0.2) / 2
    return {'segments': segments, 'point': ((u, v0 + t / 2), (t * 1.3, t * 1.3))}


def _segment_prim(i, s):
    return f'{HEAD}/Seg{i}{s}'


def _carriage(stage, x, materials, looks):
    """The carriage at world x (its index, the plunger's distance from the clamp): a dynamic body (block, post, head,
    tongue, handle; arm, plunger, pointer and ZERO ring drawn) on the rail. Returns the ZERO cap's path."""
    xform = UsdGeom.Xform.Define(stage, CARRIAGE)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(x, RAIL_Y, CARRIAGE_Z))
    z0 = CARRIAGE_Z
    grey, dark = (0.30, 0.31, 0.33), (0.16, 0.17, 0.19)
    bx, by, bz = CARRIAGE_BLOCK
    _box(stage, CARRIAGE + '/Block', CARRIAGE_BLOCK, (0, 0, bz / 2), grey, material=materials['rubber'])
    # index pointer from the block's robot face down over the hole numbers, a red index line on it
    py = by / 2 + POINTER[1] / 2
    _box(stage, CARRIAGE + '/Pointer', (POINTER[0] * 3, POINTER[1], 0.0008), (0, py, 0.0012 - z0), (0.12, 0.12, 0.13),
         collide=False)
    _box(stage, CARRIAGE + '/PointerTab', (POINTER[0] * 3, 0.0012, z0 - 0.0012), (0, by / 2 + 0.0006, (0.0012 - z0) / 2),
         (0.12, 0.12, 0.13), collide=False)
    _box(stage, CARRIAGE + '/IndexLine', (0.0005, POINTER[1], 0.0002), (0, py, 0.0017 - z0), (0.85, 0.05, 0.05),
         collide=False)
    # post and head
    post_y = IND_Y - 0.012 - RAIL_Y
    post_top = IND_Z - 0.026 - z0
    _cylinder(stage, CARRIAGE + '/Post', 0.006, post_top - bz, (0, post_y, bz + (post_top - bz) / 2), (0.25, 0.26, 0.28),
              materials['rubber'])
    head = UsdGeom.Xform.Define(stage, HEAD)
    api = UsdGeom.XformCommonAPI(head)
    api.SetTranslate(Gf.Vec3d(0.0, IND_Y - RAIL_Y, IND_Z - z0))
    api.SetRotate(Gf.Vec3f(IND_TILT_DEG, 0.0, 180.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    fw, fh = IND_FACE
    _box(stage, HEAD + '/Housing', (fw, fh, IND_DEPTH), (0, 0, -IND_DEPTH / 2), dark, material=materials['rubber'])
    _bind(_box(stage, HEAD + '/Glass', (LCD[0], LCD[1], 0.0006), (0, IND_LCD_V, 0.0003), LCD_GLASS, collide=False),
          looks['glass'])
    rects = lcd_rects()
    for (i, s), ((u, v), (du, dv)) in rects['segments'].items():
        _bind(_box(stage, _segment_prim(i, s), (du, dv, 0.0006), (u, v, 0.0007), SEGMENT_ON, collide=False), looks['lit'])
    (u, v), (du, dv) = rects['point']
    _bind(_box(stage, HEAD + '/Point', (du, dv, 0.0006), (u, v, 0.0007), SEGMENT_ON, collide=False), looks['lit'])
    white = (0.95, 0.95, 0.93)
    top = IND_LCD_V + LCD[1] / 2 + (fh / 2 - IND_LCD_V - LCD[1] / 2) / 2
    label(stage, HEAD + '/Unit', 'mm', (fw / 2 - 0.005, top), 0.0004, height_m=0.0065, align='right', colour=white,
          up=(0.0, 1.0))
    # arm from the head's back under the strip, the plunger sleeve and rod (render only: the tip presses on nothing)
    arm_from = IND_Y - 0.018 - RAIL_Y
    arm_to = BEAM_Y - 0.004 - RAIL_Y
    _box(stage, CARRIAGE + '/Arm', (0.004, arm_from - arm_to, 0.005), (-0.004, (arm_from + arm_to) / 2, ARM_Z - z0),
         (0.25, 0.26, 0.28), collide=False)
    _box(stage, CARRIAGE + '/ArmFoot', (0.008, 0.006, 0.006), (-0.002, arm_to + 0.002, ARM_Z - z0), (0.25, 0.26, 0.28),
         collide=False)
    _cylinder(stage, CARRIAGE + '/PlungerSleeve', 0.004, 0.010, (0, BEAM_Y - RAIL_Y, ARM_Z + 0.0075 - z0),
              (0.55, 0.56, 0.58), collide=False)
    plunger = UsdGeom.Xform.Define(stage, PLUNGER)
    UsdGeom.XformCommonAPI(plunger).SetTranslate(Gf.Vec3d(0, BEAM_Y - RAIL_Y, plunger_z(0.0)))
    _cylinder(stage, PLUNGER + '/Rod', 0.002, PLUNGER_LEN, (0, 0, 0), (0.78, 0.78, 0.75), collide=False)
    # tongue toward the robot: the push handle, then the ZERO plate with the cap's housing ring
    t0, t1 = TONGUE_Y[0] - RAIL_Y, ZERO_PLATE[1] - RAIL_Y
    tz = (TONGUE[1] + TONGUE[2]) / 2
    th = TONGUE[2] - TONGUE[1]
    _box(stage, CARRIAGE + '/Tongue', (TONGUE[0], t1 - t0, th), (TONGUE_DX, (t0 + t1) / 2, tz), grey,
         material=materials['rubber'])
    p0, p1 = t1, TONGUE_Y[1] - RAIL_Y
    _box(stage, CARRIAGE + '/ZeroPlate', (ZERO_PLATE[0], p1 - p0, th), (TONGUE_DX, (p0 + p1) / 2, tz), grey,
         material=materials['rubber'])
    hy = HANDLE_Y - RAIL_Y
    hh = HANDLE_TOP - z0 - TONGUE[2]
    _box(stage, CARRIAGE + '/Handle', (HANDLE[0], HANDLE[1], hh), (TONGUE_DX, hy, TONGUE[2] + hh / 2), (0.92, 0.48, 0.10),
         material=materials['rubber'])
    _cylinder(stage, CARRIAGE + '/ZeroHousing', C.CAP_R + 0.003, C.CAP_LIFT - 0.001,
              (TONGUE_DX, ZERO_Y - RAIL_Y, TONGUE[2] + (C.CAP_LIFT - 0.001) / 2), (0.25, 0.25, 0.27), collide=False)
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim()).CreateKinematicEnabledAttr(False)
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(xform.GetPrim())
    body.CreateSolverPositionIterationCountAttr(16)
    body.CreateSolverVelocityIterationCountAttr(8)
    body.CreateSleepThresholdAttr(0.0)
    body.CreateLinearDampingAttr(0.0)
    body.CreateAngularDampingAttr(0.0)
    UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(CARRIAGE_MASS)
    # the ZERO cap: its own dynamic body above the plate (joint to the carriage in author_joints)
    path = C.button_path(ZERO_ID)
    UsdGeom.Xform.Define(stage, ROOT + '/Buttons')
    cap = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(cap).SetTranslate(Gf.Vec3d(x + TONGUE_DX, ZERO_Y, z0 + TONGUE[2] + C.CAP_LIFT + C.CAP_H / 2))
    green = (0.15, 0.62, 0.25)
    _cylinder(stage, path + '/Cap', C.CAP_R, C.CAP_H, (0, 0, 0), green, materials['rubber'])
    _cylinder(stage, path + '/Pad', C.PAD_R, C.PAD_H, (0, 0, C.CAP_H / 2 + C.PAD_H / 2), green, materials['rubber'])
    _text(stage, path + '/Text', 'ZERO', (0.0, 0.0), C.CAP_H / 2 + C.PAD_H + 0.0004, 0.006, colour=(0.97, 0.97, 0.95))
    _rigid(cap.GetPrim(), C.CAP_MASS_KG)
    return path


def plunger_z(w):
    """Local height of the plunger rod's centre on the carriage: its top on the strip's underside, w (m) below rest."""
    return BEAM_TOP - BEAM[2] - PLUNGER_LEN / 2 - w - CARRIAGE_Z


def _rail(stage, materials):
    """The rail (static collider), its detent holes (drawn) and the white strip with the hole numbers."""
    x0, x1 = CLAMP_X + RAIL[0], CLAMP_X + RAIL[1]
    _box(stage, ROOT + '/Rail', (x1 - x0, RAIL[2], RAIL[3]), ((x0 + x1) / 2, RAIL_Y, RAIL[3] / 2), (0.55, 0.56, 0.60),
         material=materials['metal'])
    y0, y1 = NUMBER_STRIP
    _box(stage, ROOT + '/NumberStrip', (x1 - x0, y1 - y0, 0.0006), ((x0 + x1) / 2, (y0 + y1) / 2, 0.0003), C.PAPER,
         collide=False)
    for mm in HOLES_MM:
        _cylinder(stage, f'{ROOT}/Hole{mm}', HOLE_R, 0.0006, (hole_x(mm), RAIL_Y, RAIL[3] + 0.0001), (0.05, 0.05, 0.06),
                  collide=False)
        _text(stage, f'{ROOT}/HoleNumber{mm}', str(mm), (hole_x(mm), NUMBER_Y), 0.0008, NUMBER_H)


def build(stage, config):
    """config: {'seats_m': {k: distance from the clamp}, 'carriage_hole_mm': the hole the carriage starts on,
    'weights': {w: label}, 'masses_kg': {w: mass}, 'weight_seats': {w: 'rack:<n>'}}. Returns objects (weights),
    buttons (the ZERO cap), seats (rack collars)."""
    retire_hooke_apparatus(stage)
    repair_cube_extents(stage)
    z0 = table_top_z(stage)
    root = UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(0, 0, z0))
    materials = {'rubber': _material(stage, '/World/PhysicsMaterials/BeamRubber', 0.8, 0.7),
                 'pmma': _material(stage, '/World/PhysicsMaterials/BeamPMMA', 0.25, 0.20),
                 'cell': _material(stage, '/World/PhysicsMaterials/BeamCell', 0.25, 0.20, compliant=C.CUVETTE_CONTACT),
                 'seat': _material(stage, '/World/PhysicsMaterials/BeamSeat', 0.20, 0.15),
                 'metal': _material(stage, '/World/PhysicsMaterials/BeamStrip', 0.40, 0.30)}
    looks = {'glass': _render_material(stage, ROOT + '/Looks/LcdGlass', LCD_GLASS, matte=True),
             'lit': _render_material(stage, ROOT + '/Looks/LcdSegment', SEGMENT_ON, emission=SEGMENT_EMISSION, matte=True)}
    # Clamp: a pillar with a top jaw; the strip leaves it at CLAMP_X.
    _box(stage, ROOT + '/ClampPillar', (0.06, 0.07, BEAM_TOP), (CLAMP_X - 0.03, BEAM_Y, BEAM_TOP / 2), (0.20, 0.22, 0.25),
         material=materials['rubber'])
    _box(stage, ROOT + '/ClampJaw', (0.06, 0.07, 0.018), (CLAMP_X - 0.03, BEAM_Y, BEAM_TOP + 0.009), (0.30, 0.32, 0.36),
         material=materials['rubber'])
    _text(stage, ROOT + '/ClampLabel', 'CLAMP', (CLAMP_X - 0.03, BEAM_Y + 0.05), 0.0004, 0.009)
    # The strip: kinematic collider pieces the backend moves to the deflected shape every physics tick.
    UsdGeom.Xform.Define(stage, ROOT + '/Strip')
    L, w, t = BEAM
    for i in range(SEGMENTS):
        piece = UsdGeom.Xform.Define(stage, segment_path(i))
        UsdGeom.XformCommonAPI(piece).SetTranslate(Gf.Vec3d(CLAMP_X + segment_x(i), BEAM_Y, BEAM_TOP - t / 2))
        UsdGeom.XformCommonAPI(piece).SetRotate(Gf.Vec3f(0, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        box = _box(stage, segment_path(i) + '/Box', (L / SEGMENTS + SEGMENT_OVERLAP, w, t), (0, 0, 0), (0.78, 0.80, 0.83),
                   material=materials['metal'])
        _strip_collider(box.GetPrim())
        _kinematic(piece.GetPrim())
    # Distance labels at the seats.
    for k, a in config['seats_m'].items():
        _text(stage, f'{ROOT}/SeatLabel{k}', f'{a * 1000:.0f} mm', (CLAMP_X + a, BEAM_Y - 0.040), 0.0004, 0.008)
    # Knife-edge load saddles (kinematic, riding on the strip) with collars.
    for k, a in config['seats_m'].items():
        path = seat_path(k)
        xform = UsdGeom.Xform.Define(stage, path)
        UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(CLAMP_X + a, BEAM_Y, BEAM_TOP))
        knife = _box(stage, path + '/Knife', (KNIFE[0], KNIFE[1], KNIFE_H), (0, 0, KNIFE_H / 2), (0.55, 0.56, 0.60),
                     material=materials['metal'])
        _strip_collider(knife.GetPrim())
        block = _box(stage, path + '/Block', SADDLE, (0, 0, KNIFE_H + SADDLE[2] / 2), (0.30, 0.31, 0.33),
                     material=materials['rubber'])
        _strip_collider(block.GetPrim())
        collar(stage, path + '/Seat', (0.0, 0.0), KNIFE_H + SADDLE[2], (0.95, 0.55, 0.10), materials['seat'])
        _kinematic(xform.GetPrim())
    # The indicator rail and its carriage.
    _rail(stage, materials)
    zero = _carriage(stage, hole_x(config['carriage_hole_mm']), materials, looks)
    # Travel stop under the free end.
    stop_top = BEAM_TOP - t - STOP_GAP
    _cylinder(stage, ROOT + '/StopPost', 0.006, stop_top - 0.006, (CLAMP_X + STOP_X, BEAM_Y, (stop_top - 0.006) / 2),
              (0.25, 0.26, 0.28), materials['rubber'])
    _cylinder(stage, ROOT + '/StopPad', 0.009, 0.006, (CLAMP_X + STOP_X, BEAM_Y, stop_top - 0.003), (0.08, 0.08, 0.08),
              materials['rubber'])
    # Weight rack.
    bx, by, bz = RACK['base']
    _box(stage, ROOT + '/Rack', RACK['base'], (sum(RACK['xs']) / 3, RACK['y'], bz / 2), (0.12, 0.13, 0.15),
         material=materials['rubber'])
    seats = {}
    for n, x in enumerate(RACK['xs'], start=1):
        seats[f'rack:{n}'] = str(collar(stage, f'{ROOT}/RackSeat{n}', (x, RACK['y']), bz, (0.03, 0.16, 0.20), materials['seat']).GetPath())
    objects = {}
    for w_id, sid in config['weight_seats'].items():
        n = int(sid.split(':')[1])
        objects[w_id] = _weight(stage, w_id, config['weights'][w_id], (RACK['xs'][n - 1], RACK['y']),
                                bz + SEAT_FLOOR + 0.0003, config['masses_kg'][w_id], materials)
    buttons = {ZERO_ID: zero}
    contact = PhysxSchema.PhysxCollisionAPI.Apply(stage.GetPrimAtPath(TABLE_TOP))
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    show_lcd(stage, ' ' * LCD_CELLS, None)
    return {'objects': objects, 'buttons': buttons, 'seats': seats}


# --- poses of the moving parts ----------------------------------------------------------------------------------
def strip_poses(table_z, seg_w, seg_slope, saddle_w):
    """World poses (x, y, z, qx, qy, qz, qw) of the strip pieces (deflection and slope at each centre) and the saddles
    (level, on the strip at their edges), in the order of strip_bodies."""
    out = []
    t = BEAM[2]
    for i, (w, s) in enumerate(zip(seg_w, seg_slope)):
        h = 0.5 * math.atan(s)                       # rotation about +y by the slope: the piece's far end lower
        out.append((CLAMP_X + segment_x(i), BEAM_Y, table_z + BEAM_TOP - t / 2 - w, 0.0, math.sin(h), 0.0, math.cos(h)))
    for a, w in saddle_w:
        out.append((CLAMP_X + a, BEAM_Y, table_z + BEAM_TOP - w, 0.0, 0.0, 0.0, 1.0))
    return out


def strip_bodies(seats):
    """Prim paths of the strip pieces and the saddles of `seats` (in order)."""
    return [segment_path(i) for i in range(SEGMENTS)] + [seat_path(k) for k in seats]


def set_render_poses(stage, paths, poses, table_z, plunger_w=None):
    """Author the moving parts' poses in USD (the renderer's copy; the physics bodies get the same kinematic targets).
    plunger_w: the strip's deflection at the carriage's index; the plunger rod follows the strip's underside."""
    for path, (x, y, z, qx, qy, qz, qw) in zip(paths, poses):
        api = UsdGeom.XformCommonAPI(UsdGeom.Xform(stage.GetPrimAtPath(path)))
        api.SetTranslate(Gf.Vec3d(x, y, z - table_z))
        angle = math.degrees(2 * math.atan2(qy, qw))
        api.SetRotate(Gf.Vec3f(0.0, angle, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    if plunger_w is not None:
        api = UsdGeom.XformCommonAPI(UsdGeom.Xform(stage.GetPrimAtPath(PLUNGER)))
        t = api.GetXformVectors(Usd.TimeCode.Default())[0]
        api.SetTranslate(Gf.Vec3d(t[0], t[1], plunger_z(plunger_w)))


def _light(stage, path, lit):
    prim = stage.GetPrimAtPath(path)
    api = UsdGeom.XformCommonAPI(UsdGeom.Xform(prim))
    t = api.GetXformVectors(Usd.TimeCode.Default())[0]
    api.SetTranslate(Gf.Vec3d(t[0], t[1], 0.0007 if lit else 0.0007 - HIDE_M))


def show_lcd(stage, text, previous):
    """Light the LCD cells that differ from the previously shown text (six characters, see beam_bench.lcd_text); the
    point is lit whenever a reading is shown."""
    if text == previous:
        return text
    for i, char in enumerate(text):
        if previous is not None and previous[i] == char:
            continue
        on = GLYPHS.get(char, '')
        for s in SEVEN:
            _light(stage, _segment_prim(i, s), s in on)
    blank = not text.strip()
    if previous is None or (not previous.strip()) != blank:
        _light(stage, HEAD + '/Point', not blank)
    return text


def author_joints(stage, buttons):
    """The carriage's rail joint (prismatic along world x from the world, end stops TRAVEL_MARGIN_M beyond the first
    and last hole, a saturated damper drive as the rail friction) and the ZERO cap's spring-returned vertical joint to
    the carriage (the chemistry kit's cap joint, on the carriage instead of the world)."""
    from pxr import Sdf
    UsdGeom.Scope.Define(stage, C.PHYSICS_ROOT)
    cache = UsdGeom.XformCache()
    carriage = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(CARRIAGE)).ExtractTranslation()
    x0 = float(carriage[0])
    joint = UsdPhysics.PrismaticJoint.Define(stage, CARRIAGE_JOINT)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(CARRIAGE)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(carriage))
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateAxisAttr('X')
    joint.CreateLowerLimitAttr(hole_x(HOLES_MM[0]) - TRAVEL_MARGIN_M - x0)
    joint.CreateUpperLimitAttr(hole_x(HOLES_MM[-1]) + TRAVEL_MARGIN_M - x0)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
    drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(RAIL_FRICTION_DAMPING)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateMaxForceAttr(RAIL_FRICTION_N)
    for bid, path in buttons.items():
        cap = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation()
        key = UsdPhysics.PrismaticJoint.Define(stage, ZERO_JOINT)
        key.CreateBody0Rel().SetTargets([Sdf.Path(CARRIAGE)])
        key.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        key.CreateLocalPos0Attr(Gf.Vec3f(cap - carriage))
        key.CreateLocalRot0Attr(Gf.Quatf(1))
        key.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        key.CreateLocalRot1Attr(Gf.Quatf(1))
        key.CreateAxisAttr('Z')
        key.CreateLowerLimitAttr(-C.TRAVEL_M)
        key.CreateUpperLimitAttr(0.0)
        spring = UsdPhysics.DriveAPI.Apply(key.GetPrim(), 'linear')
        spring.CreateTypeAttr('force')
        spring.CreateStiffnessAttr(C.SPRING_N_PER_M)
        spring.CreateDampingAttr(C.SPRING_DAMPING)
        spring.CreateTargetPositionAttr(C.SPRING_PRELOAD_M)
        spring.CreateTargetVelocityAttr(0.0)
        spring.CreateMaxForceAttr(100.0)


seat_frames = C.seat_frames

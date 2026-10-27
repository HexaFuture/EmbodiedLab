"""Prewired rectifier supply (rectifier_supply) with two detent controls and a continuous load slider.
The load rail has viscous damping and no positional servo.
"""
import math

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema

from .bench import DETENT_LABELS, DETENT_SPACING_M, SELECTORS
from torsion_apparatus import (TABLE_TOP, _box, _collider, _cylinder, _material, repair_cube_extents,
                               retire_hooke_apparatus, table_top_z)

ROOT = '/World/CircuitApparatus'
PHYSICS_ROOT = '/World/CircuitPhysics'

# Panel origin on the table top, about 0.45 m in front of the robot base (world (0.95, 0.35)).
ORIGIN_XY = (0.90, -0.10)

PLATE = (0.46, 0.24, 0.020)                 # x, y, z size; top at PLATE[2] above the table
DETENT0_Y = -0.030                          # panel-frame y of detent 0; detent k at DETENT0_Y + k * spacing
SLIDER_X = {'CAP': -0.15, 'LOAD': 0.0, 'MODE': 0.15}
CARRIAGE = (0.032, 0.022, 0.012)
CARRIAGE_GAP = 0.001                        # carriage bottom above the plate top (the joint carries it)
POST_R, POST_H = 0.012, 0.060
CARRIAGE_MASS_KG = 0.10
DETENT_STIFFNESS_N_PER_M = 300.0            # 4.5 N at mid-way between detents
DETENT_DAMPING_NS_PER_M = 15.0
TRAVEL_MARGIN_M = 0.003                     # end stops inside the end detents' contact zone (circuit_bench.DETENT_TOLERANCE_M)
TEXT_HEIGHT_M = 0.012
NAME_HEIGHT_M = 0.016
# Printed text reads upright in the wrist camera at the start pose, whose image up is world -X.
TEXT_UP = (-1.0, 0.0)
POST_COLOUR = (0.95, 0.78, 0.05)
INK = (0.02, 0.02, 0.02)

BOX = {'centre': (-0.29, 0.0), 'size': (0.08, 0.12, 0.05)}
TERMINALS = {'A': (-0.035, (0.80, 0.08, 0.06)), 'B': (0.0, (0.05, 0.05, 0.05)), 'C': (0.035, (0.92, 0.80, 0.08))}
TERMINAL_R, TERMINAL_H = 0.007, 0.018
LOGGER = {'centre': (0.32, 0.0), 'size': (0.08, 0.12, 0.045)}

SLIDER_LABELS = {'CAP': ('C1', 'C2', 'C3'), 'LOAD': ('150', '704', '3300'), 'MODE': ('ON', 'OFF', 'DISCHARGE')}


def carriage_path(name):
    return f'{ROOT}/{name.title()}Slider'


def joint_path(name):
    return f'{PHYSICS_ROOT}/{name.title()}Rail'


def detent_offset_m(index):
    """Carriage travel from detent 0 (positive toward the robot)."""
    return index * DETENT_SPACING_M


def _text_rectangles(text, height_m):
    """Pixel-font rectangles (x0, y0, x1, y1) in metres of a single-line label, origin at its left baseline centre."""
    import cv2
    scale, thickness = 1.0, 2
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = 2
    image = np.zeros((h + base + 2 * pad, w + 2 * pad), np.uint8)
    cv2.putText(image, text, (pad, pad + h), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thickness, cv2.LINE_8)
    pixel = height_m / h
    rows = image > 127
    rects = []
    for r in range(rows.shape[0]):
        c = 0
        while c < rows.shape[1]:
            if rows[r, c]:
                start = c
                while c < rows.shape[1] and rows[r, c]:
                    c += 1
                y_top = (pad + h - r) * pixel      # image rows grow downward; label y grows upward
                rects.append(((start - pad) * pixel, y_top - pixel, (c - pad) * pixel, y_top))
            else:
                c += 1
    return rects, w * pixel


def label(stage, path, text, anchor_xy, z, *, height_m=TEXT_HEIGHT_M, align='left', colour=INK, up=TEXT_UP):
    """Flat printed text (render only) at panel-frame anchor_xy, reading with its top toward `up`."""
    rects, width = _text_rectangles(text, height_m)
    shift = {'left': 0.0, 'centre': -width / 2, 'right': -width}[align]
    ux, uy = up
    rx, ry = uy, -ux                                   # reading direction: up rotated -90 degrees
    points, counts, indices = [], [], []
    for x0, y0, x1, y1 in rects:
        base = len(points)
        for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            u, v = px + shift, py - height_m / 2
            points.append(Gf.Vec3f(anchor_xy[0] + u * rx + v * ux, anchor_xy[1] + u * ry + v * uy, z))
        counts.append(4)
        indices += [base, base + 1, base + 2, base + 3]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    arr = np.asarray([[p[0], p[1], p[2]] for p in points])
    mesh.CreateExtentAttr([Gf.Vec3f(*arr.min(0).tolist()), Gf.Vec3f(*arr.max(0).tolist())])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    return mesh


def build(stage, *, initial):
    """Author the bench in its unjittered pose with the sliders at the initial detents {PAIR, RS, MODE: label}.

    Joints and detent drives are authored later (author_joints), after any layout jitter.
    """
    retire_hooke_apparatus(stage)
    repair_cube_extents(stage)
    origin = (*ORIGIN_XY, table_top_z(stage))
    rubber = _material(stage, '/World/PhysicsMaterials/CircuitRubber', 0.8, 0.7)
    root = UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(*origin))
    top = PLATE[2]
    _box(stage, ROOT + '/Plate', PLATE, (0, 0, top / 2), (0.16, 0.18, 0.22), material=rubber)
    ink_z = top + 0.0004
    for name in SELECTORS:
        x = SLIDER_X[name]
        travel = 2 * DETENT_SPACING_M
        _box(stage, f'{ROOT}/{name.title()}Slot', (0.010, travel + 0.030, 0.0006),
             (x, DETENT0_Y + travel / 2, top + 0.0003), (0.03, 0.03, 0.035), collide=False)
        # Name before detent 0; detent names beside the rail, centred on their detents, in stepped rows.
        label(stage, f'{ROOT}/{name.title()}Name', 'CAP SELECT' if name == 'CAP' else name, (x, DETENT0_Y - 0.030), ink_z, align='right',
              height_m=NAME_HEIGHT_M)
        for k, text in enumerate(SLIDER_LABELS[name]):
            y = DETENT0_Y + detent_offset_m(k)
            row = (0.024, 0.041, 0.058)[k]      # stepped rows: long names (CHARGE, DISCHARGE) never overlap
            _box(stage, f'{ROOT}/{name.title()}Tick{k}', (row - 0.012, 0.0015, 0.0006), (x + 0.006 + (row - 0.012) / 2, y, ink_z - 0.0001),
                 INK, collide=False)
            label(stage, f'{ROOT}/{name.title()}Detent{k}', text, (x + row + TEXT_HEIGHT_M / 2, y), ink_z, align='centre')
        index = SLIDER_LABELS[name].index(_label_text(name, initial[name]))
        carriage = UsdGeom.Xform.Define(stage, carriage_path(name))
        cz = top + CARRIAGE_GAP + CARRIAGE[2] / 2
        UsdGeom.XformCommonAPI(carriage).SetTranslate(Gf.Vec3d(x, DETENT0_Y + detent_offset_m(index), cz))
        _box(stage, carriage_path(name) + '/Block', CARRIAGE, (0, 0, 0), (0.30, 0.31, 0.33), material=rubber)
        _cylinder(stage, carriage_path(name) + '/Post', POST_R, POST_H, (0, 0, CARRIAGE[2] / 2 + POST_H / 2),
                  POST_COLOUR, rubber)
        _rigid_carriage(carriage.GetPrim())
    # Fixed prewired source board, beside the mechanical controls.
    # Black box above the panel in the wrist image (world -x), terminals in a row along world y (left to
    # right in the image) with their letters below them; the logger below the panel (world +x).
    bx, by = BOX['centre']
    sx, sy, sz = BOX['size']
    _box(stage, ROOT + '/RectifierPCB', (.11, .19, .012), (bx, by, .006), (.08,.30,.24), material=rubber)
    for k, nominal in enumerate(('100uF','220uF','470uF')):
        y=-.057+k*.057
        _cylinder(stage, ROOT+f'/Capacitor{k}', .016, .04, (bx+.017,y,.032), (.15,.35,.8),rubber)
        label(stage,ROOT+f'/CapLabel{k}',f'C{k+1} '+nominal,(bx-.025,y),.013,height_m=.008,align='centre',colour=(1,1,1))
    label(stage,ROOT+'/BridgeLabel','FILTER CAPS',(bx-.043,0),.013,height_m=.010,align='centre',colour=(1,1,1))
    # A separate physical constant-resistance electronic load. The slide is its
    # mechanical command encoder, not a second set of resistors.
    ex,ey=0.,.180
    _box(stage,ROOT+'/LoadModule',(.105,.075,.035),(ex,ey,.0175),(.18,.20,.24),material=rubber)
    for k in range(5):
        _box(stage,ROOT+f'/LoadFin{k}',(.006,.050,.012),(ex-.038+k*.010,ey,.041),(.38,.41,.44),material=rubber)
    _box(stage,ROOT+'/LoadResistorSymbol',(.024,.052,.010),(ex+.030,ey,.040),(.93,.91,.82),material=rubber)
    label(stage,ROOT+'/LoadModuleLabel','R LOAD',(ex+.030,ey),.0455,height_m=.008,align='centre')
    label(stage,ROOT+'/LoadModuleRange','150-3300 OHM',(ex+.064,ey),.0008,height_m=.008,align='centre')
    # Public source enclosure and its panel label, kept outside all slider travel.
    ax,ay=bx,.180
    _box(stage,ROOT+'/ACSource',(.085,.075,.035),(ax,ay,.0175),(.25,.34,.48),material=rubber)
    label(stage,ROOT+'/ACSourceLabel','AC SOURCE',(ax-.016,ay),.0355,height_m=.010,align='centre',colour=(1,1,1))
    label(stage,ROOT+'/ACSourceValue','12Vp 50Hz',(ax+.016,ay),.0355,height_m=.010,align='centre',colour=(1,1,1))
    # Printed bridge module indicates the prewired topology; electrical behavior
    # comes from the documented diode equations, not from decorative geometry.
    dx,dy=bx,-.125
    _box(stage,ROOT+'/BridgeModule',(.085,.055,.018),(dx,dy,.009),(.16,.18,.20),material=rubber)
    label(stage,ROOT+'/BridgeModuleLabel','BRIDGE',(dx,dy),.0185,height_m=.012,align='centre',colour=(1,1,1))
    for k,(u,v) in enumerate(((-.024,-.015),(-.024,.015),(.024,-.015),(.024,.015))):
        _box(stage,ROOT+f'/Diode{k}',(.008,.014,.004),(dx+u,dy+v,.021),(.03,.03,.03),material=rubber)
        _box(stage,ROOT+f'/DiodeBand{k}',(.008,.002,.0006),(dx+u,dy+v+.004,.0234),(.85,.85,.85),collide=False)
    def wire(name,a,b,colour,width=.003):
        if abs(a[0]-b[0])>1e-8 and abs(a[1]-b[1])>1e-8:raise ValueError('orthogonal wiring segment required')
        _box(stage,ROOT+'/'+name,(max(abs(a[0]-b[0]),width),max(abs(a[1]-b[1]),width),.002),
             ((a[0]+b[0])/2,(a[1]+b[1])/2,.002),colour,collide=False)
    wire('SourceLead',(ax+.046,ay),(ax+.046,dy),(.75,.15,.12))
    wire('BridgeLead',(ax+.046,dy),(dx+.042,dy),(.75,.15,.12))
    wire('OutputLead',(bx+.055,-.093),(-.215,-.093),(.75,.15,.12))
    wire('OutputBus',(-.215,-.093),(-.215,.180),(.75,.15,.12))
    wire('OutputToLoad',(-.215,.180),(-.055,.180),(.75,.15,.12))
    wire('LoadControlLink',(0.,.060),(0.,.140),(.15,.45,.85),.004)
    lx, ly = LOGGER['centre']
    lsx, lsy, lsz = LOGGER['size']
    _box(stage, ROOT + '/Logger', (lsx, lsy, lsz), (lx, ly, lsz / 2), (0.20, 0.24, 0.30), material=rubber)
    _box(stage, ROOT + '/LoggerScreen', (0.042, 0.092, 0.0008), (lx - 0.012, ly, lsz + 0.0004), (0.004, 0.018, 0.010),
         collide=False)
    build_display(stage, (lx - 0.012, ly), lsz + 0.0010)
    label(stage, ROOT + '/LoggerLabel', 'SCOPE', (lx + 0.024, ly), lsz + 0.0004, align='centre', height_m=0.012,
          colour=(0.95, 0.95, 0.92))
    # Fixed wiring (render only): black box and logger to the plate edges.
    for i, (x0, x1, y) in enumerate(((bx + sx / 2, -PLATE[0] / 2, by), (lx - lsx / 2, PLATE[0] / 2, ly))):
        _box(stage, f'{ROOT}/Cable{i}', (abs(x1 - x0), 0.006, 0.006), ((x0 + x1) / 2, y, 0.003),
             (0.10, 0.10, 0.10), collide=False)
    table_top = stage.GetPrimAtPath(TABLE_TOP)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(table_top)
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    return {name: carriage_path(name) for name in SELECTORS}


def _label_text(name, value):
    return SLIDER_LABELS[name][DETENT_LABELS[name].index(value)]


def _rigid_carriage(prim):
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    body.CreateSolverPositionIterationCountAttr(16)
    body.CreateSolverVelocityIterationCountAttr(8)
    body.CreateSleepThresholdAttr(0.0)
    body.CreateLinearDampingAttr(0.0)
    body.CreateAngularDampingAttr(0.0)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(CARRIAGE_MASS_KG)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))


def author_joints(stage, initial):
    """Prismatic rails from the world to each carriage at the plate's current (possibly jittered) pose.

    Joint coordinate q = carriage travel from detent 0 along the rail, positive toward the robot.
    The detent drive targets the initial detent; the runtime moves the target to the nearest detent.
    """
    plate = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(ROOT))
    rotation = plate.ExtractRotationQuat()
    # Joint frame: x axis along the rail direction (panel +y), so q grows toward the robot.
    along = Gf.Rotation(Gf.Vec3d(0, 0, 1), 90.0).GetQuat()
    frame = Gf.Quatf(rotation * along)
    drives = {}
    for name in SELECTORS:
        index = DETENT_LABELS[name].index(initial[name])
        detent0_world = plate.Transform(Gf.Vec3d(SLIDER_X[name], DETENT0_Y, PLATE[2] + CARRIAGE_GAP + CARRIAGE[2] / 2))
        joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path(name))
        joint.CreateBody1Rel().SetTargets([Sdf.Path(carriage_path(name))])
        joint.CreateLocalPos0Attr(Gf.Vec3f(detent0_world))
        joint.CreateLocalRot0Attr(frame)
        # Carriage frame: at detent `index` the carriage origin sits `index * spacing` along the rail,
        # so the initial joint coordinate equals that offset.
        carriage = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(carriage_path(name)))
        offset = detent_offset_m(index)
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot1Attr(Gf.Quatf(carriage.ExtractRotationQuat().GetInverse() * rotation * along))
        joint.CreateAxisAttr('X')
        joint.CreateLowerLimitAttr(-TRAVEL_MARGIN_M)
        joint.CreateUpperLimitAttr(detent_offset_m(2) + TRAVEL_MARGIN_M)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
        drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(0.0 if name == 'LOAD' else DETENT_STIFFNESS_N_PER_M)
        drive.CreateDampingAttr(DETENT_DAMPING_NS_PER_M)
        drive.CreateTargetPositionAttr(offset)
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateMaxForceAttr(50.0)
        drives[name] = drive
    return drives


def set_detent_target(stage, name, offset_m):
    UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(joint_path(name)), 'linear').GetTargetPositionAttr().Set(float(offset_m))


def rails(stage):
    """World xy of detent 0 and the unit rail direction (toward the robot) per slider, after any jitter."""
    plate = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(ROOT))
    direction = np.asarray(plate.TransformDir(Gf.Vec3d(0, 1, 0)))[:2]
    direction /= np.linalg.norm(direction)
    return {name: (np.asarray(plate.Transform(Gf.Vec3d(SLIDER_X[name], DETENT0_Y, 0.0)))[:2], direction)
            for name in SELECTORS}


# --- logger screen: live CH1 reading ------------------------------------------------------------
DISPLAY = ROOT + '/LoggerDisplay'
DIGIT_H, DIGIT_W, SEGMENT_T, DIGIT_PITCH = 0.018, 0.009, 0.0022, 0.0135
POINT_AFTER, POINT_GAP = 2, 0.004      # the decimal point follows cell POINT_AFTER in a gap of its own (in the
                                         # plain digit gap it touched both neighbouring digits' lower verticals)
SEGMENT_ON = (0.35, 1.0, 0.45)
SEGMENTS = {'a': (0.5, 1.0, 1), 'b': (1.0, 0.75, 0), 'c': (1.0, 0.25, 0), 'd': (0.5, 0.0, 1),
            'e': (0.0, 0.25, 0), 'f': (0.0, 0.75, 0), 'g': (0.5, 0.5, 1)}      # (u/w, v/h, horizontal)
DIGIT_SEGMENTS = {'0': 'abcdef', '1': 'bc', '2': 'abdeg', '3': 'abcdg', '4': 'bcfg', '5': 'acdfg', '6': 'acdefg',
                  '7': 'abc', '8': 'abcdefg', '9': 'abcdfg', '-': 'g', ' ': ''}


def _flat_box(stage, path, centre_uv, size_uv, origin, z, colour):
    """Render-only rectangle in the screen's reading frame (u along TEXT_UP rotated -90 deg, v along TEXT_UP)."""
    ux, uy = TEXT_UP
    rx, ry = uy, -ux
    (cu, cv), (su, sv) = centre_uv, size_uv
    x = origin[0] + cu * rx + cv * ux
    y = origin[1] + cu * ry + cv * uy
    sx, sy = abs(su * rx) + abs(sv * ux), abs(su * ry) + abs(sv * uy)
    return _box(stage, path, (sx, sy, 0.0006), (x, y, z), colour, collide=False)


def display_rects():
    """Lit rectangles of the readout in the reading frame ((u, v) centre, (du, dv) size), the row centred as built:
    {'segments': {(cell, segment): rect}, 'point': rect}."""
    width = 5 * DIGIT_PITCH + POINT_GAP
    u0, v0 = -width / 2 - 0.004, -0.013            # lower-left of the digit row in the reading frame
    cell = [u0 + i * DIGIT_PITCH + (POINT_GAP if i > POINT_AFTER else 0.0) for i in range(5)]
    segments = {(i, name): ((cell[i] + fu * DIGIT_W, v0 + fv * DIGIT_H),
                            (DIGIT_W, SEGMENT_T) if horizontal else (SEGMENT_T, DIGIT_H / 2))
                for i in range(5) for name, (fu, fv, horizontal) in SEGMENTS.items()}
    point = ((cell[POINT_AFTER] + DIGIT_W + (DIGIT_PITCH - DIGIT_W + POINT_GAP) / 2, v0 + 0.0016), (0.0032, 0.0032))
    return {'segments': segments, 'point': point, 'u0': u0, 'v0': v0, 'width': width}


def build_display(stage, centre_xy, z):
    """Sign, four digits with a decimal point after the first, a unit label and the REC lamp."""
    UsdGeom.Xform.Define(stage, DISPLAY)
    layout = display_rects()
    width, u0, v0 = layout['width'], layout['u0'], layout['v0']
    for (i, name), (centre, size) in layout['segments'].items():
        _flat_box(stage, f'{DISPLAY}/D{i}{name}', centre, size, centre_xy, z, SEGMENT_ON)
    _flat_box(stage, f'{DISPLAY}/Point', *layout['point'],
              centre_xy, z, SEGMENT_ON)
    ux, uy = TEXT_UP
    rx, ry = uy, -ux
    unit_u = u0 + width + 0.002
    label(stage, f'{DISPLAY}/Unit', 'V', (centre_xy[0] + unit_u * rx + (v0 + 0.005) * ux,
                                          centre_xy[1] + unit_u * ry + (v0 + 0.005) * uy), z, height_m=0.010,
          colour=SEGMENT_ON)
    label(stage, f'{DISPLAY}/Channel', 'CH1', (centre_xy[0] + u0 * rx + 0.012 * ux, centre_xy[1] + u0 * ry + 0.012 * uy),
          z, height_m=0.008, colour=SEGMENT_ON)
    _flat_box(stage, f'{DISPLAY}/RecLamp', (width / 2, 0.012), (0.006, 0.006), centre_xy, z, (1.0, 0.08, 0.05))
    label(stage, f'{DISPLAY}/RecLabel', 'REC', (centre_xy[0] + (width / 2 - 0.022) * rx + 0.012 * ux,
                                                centre_xy[1] + (width / 2 - 0.022) * ry + 0.012 * uy), z,
          height_m=0.008, colour=(0.9, 0.9, 0.85))
    show_reading(stage, None, False, previous=None)


def display_text(volts):
    """'+4.995'-style text: sign, then four digits (the first before the decimal point)."""
    if volts is None:
        return '     '
    v = max(-99.99, min(99.99, round(volts, 2)))
    digits = f'{abs(v):05.2f}'.replace('.', '')
    return ('-' if v < 0 else ' ') + digits



def show_reading(stage, volts, armed, previous):
    """Update the segments that differ from `previous` (text, armed); returns the new state."""
    text = display_text(volts)
    state = (text, bool(armed))
    if state == previous:
        return state
    for i, char in enumerate(text):
        old = previous[0][i] if previous else None
        if old == char:
            continue
        lit = DIGIT_SEGMENTS[char]
        for name in SEGMENTS:
            prim = UsdGeom.Imageable(stage.GetPrimAtPath(f'{DISPLAY}/D{i}{name}'))
            prim.MakeVisible() if name in lit else prim.MakeInvisible()
    if previous is None or previous[1] != state[1]:
        lamp = UsdGeom.Imageable(stage.GetPrimAtPath(f'{DISPLAY}/RecLamp'))
        lamp.MakeVisible() if armed else lamp.MakeInvisible()
    point = UsdGeom.Imageable(stage.GetPrimAtPath(f'{DISPLAY}/Point'))
    point.MakeVisible() if text.strip() else point.MakeInvisible()
    return state

"""Truss bench apparatus (truss_bridge v2): the pegboard with its fixed pins A, C, B, the pin stand with the four loose
pins, the eighty bars in their racks (four colours), the scrap tray and the bar tester (fixed pin, crosshead carriage on a detent rail,
console with the display, START and the MODE slider). Built with the chemistry kit's helpers on its root, so the
chemistry backend's grasp and button machinery runs it unchanged. The layout, the plates' mesh and the pin and bar
frames are in truss_bench (pxr-free).

Collision geometry is what is drawn:
- the pegboard and the pin stand are static triangle meshes (truss_bench.holed_plate): square through holes with
  45-degree chamfered mouths;
- a fixed pin is a collar and a post (cylinders) with a conical tip, static on the board; a loose pin is one dynamic
  body (steel, PIN_MASS_KG): the shank with its chamfered end (cylinder and cone), the collar, the post and its tip.
  It goes into a hole by contact only: the chamfer guides the shank, the hole's walls hold it upright and the collar
  rests on the board;
- the rack plates and their ribs, the tray, the pedestal, the rail and the console are boxes;
- a bar is two dynamic bodies held by a fixed joint at truss_bench.CUT_M, beside the grip knob: the main body (the strip
  from the cut to its eye, that eye, the knob) and the tail (the strip from its eye to the cut, that eye). Each eye is a
  ring of EYE_PIECES boxes around its hole (Ø10.75 mm: the pin passes with 1.375 mm to spare, nothing centres it). A
  bar that breaks in tension loses the joint and lies in two pieces; one that buckles has the joint turned by
  KINK_DEG(label) about the bar's width, so it lies kinked upward at the cut (the backend does both at the failure);
- the crosshead carriage and the MODE slider are dynamic bodies on prismatic rails (a saturated damper drive is the
  rail friction; the backend adds the ball detent's force, as the beam bench's indicator carriage, and the tester's
  drive of the crosshead during and after a run, DRIVE_*); START is the kit's spring-returned cap.
"""
from __future__ import annotations

import math

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

import chem_apparatus as C
from chem_apparatus import (GLYPHS, ROOT, SEGMENT_EMISSION, SEGMENT_ON, SEGMENTS as SEVEN, _bind, _remember_lit,
                            _render_material, _rigid, _text, _xy)
from circuit_apparatus import label
from torsion_apparatus import TABLE_TOP, _box, _cylinder, _material, repair_cube_extents, retire_hooke_apparatus, \
    table_top_z
import truss_bench as B
import truss_frame
import truss_loading
import truss_model as M

KIND_COLOUR = {'blue': (0.10, 0.32, 0.85), 'orange': (0.90, 0.24, 0.02), 'green': (0.08, 0.55, 0.20),
               'white': (0.90, 0.90, 0.88)}
STEEL, DARK, BOARD_COLOUR = (0.62, 0.63, 0.66), (0.16, 0.17, 0.19), (0.10, 0.11, 0.12)
YELLOW, RED, WHITE = (0.95, 0.80, 0.08), (0.85, 0.08, 0.06), (0.95, 0.95, 0.93)
EYE_PIECES = 12
BODY_INSET = 0.007                   # the strip box ends this far inside each eye centre (inside the eye boss)
KINK_RISE_M = 0.006                  # a buckled bar's rise at the cut
BAR_MASS_SCALE = 1.0

# Tester (world, metres; heights above the table).
# The bed: one level surface at B.TESTER_TOP that a bar under test and the pieces of a broken one lie on flat -- the
# pedestal under the fixed pin, then two plates either side of a slot along the rail's axis; the crosshead runs under
# the plates, its web up through the slot to its pin (the web's top just below the bed's), its tongue out under the
# right plate to the push handle beside the bed. The plates rest on the pedestal and on an end block.
PEDESTAL_Y = (-0.215, -0.176)        # world y; the slot begins at its end (the carriage pin's nearest approach, at the
                                     # 50 mm hole with the drive's buckle stroke, keeps its web clear of it)
BED_HALF_W = 0.015                   # the bed's half width (x)
SLOT_HALF = 0.0045                   # the slot's half width: the web (the pin's diameter) with 0.5 mm play
BED_END_Y = (-0.062, -0.054)         # the end block (world y)
PLATE_T = 0.006                      # the plates' thickness (the tongue runs below them)
WEB = (0.008, 0.024)                 # the crosshead's web in the slot (x, y)
WEB_DROP = 0.0005                    # its top below the bed's top
CARRIAGE_GAP = 0.0005
CARRIAGE_MASS = 0.40
TONGUE_DX, HANDLE, HANDLE_TOP = 0.030, (0.012, 0.012), 0.070
TONGUE_Z = 0.0135                    # the tongue's middle above the table (on the lead screw's nut, under the plate)
TRAVEL_MARGIN_M = 0.012              # the rail's end stops beyond the first and last hole (the drive's strokes fit)
RAIL_FRICTION_N, RAIL_FRICTION_DAMPING = 2.0, 5.0e3
DETENT_WINDOW_M, DETENT_STIFFNESS, DETENT_PEAK_N = 0.005, 1.0e4, 4.0
HOLE_X = B.TESTER_X - 0.024          # detent marks and their numbers on the robot's right of the rail
# The tester's crosshead drive (backend): a capped spring-damper force on the carriage toward a target along the rail.
# During a run it moves the carriage from its hole by up to DRIVE_RUN_STROKE_M (away from the fixed pin in tension,
# toward it in compression) at DRIVE_TAKEUP_M_S: it takes up the eyes' play and then loads the mounted bar. When the
# bar breaks the carriage springs on to DRIVE_SNAP_M; when it buckles it follows the kink in to DRIVE_BUCKLE_M (as far
# as the kinked bar lets it) and then backs off DRIVE_UNLOAD_M. Once it has stopped the drive stops too (parked: no force, the rail's friction keeps it,
# so nothing presses on what lies on the pins). When no eye is on the tester's pins it returns the carriage to its hole
# at DRIVE_RETURN_M_S and lets it go.
DRIVE_STIFFNESS, DRIVE_DAMPING, DRIVE_N = 5.0e3, 80.0, 12.0
DRIVE_RUN_STROKE_M, DRIVE_TAKEUP_M_S = 0.003, 0.005
DRIVE_SNAP_M, DRIVE_BUCKLE_M, DRIVE_RETURN_M_S = 0.008, 0.005, 0.010
DRIVE_PARK_M_S = 0.002
DRIVE_SETTLE_S = 0.6                 # ... or once this long past the move (a kinked bar on the pins may never quite rest)
DRIVE_UNLOAD_M = 0.0015              # after a buckle the drive backs off this far before it stops (unloads the kinked bar:
                                     # its eyes come free of the pins' sides, nothing stays squeezed between the pins)
KINK_S = 0.4                         # a buckle bends the bar over this long
CARRIAGE = ROOT + '/Crosshead'
CARRIAGE_JOINT = C.PHYSICS_ROOT + '/CrossheadRail'

# Console: the display at the back (away from the robot, where the cameras see it past the hand), START and the MODE
# slider at the front (robot frame: u to the robot's right = world -x, v away from it = world -y).
CONSOLE = {'size': (0.150, 0.120, 0.025)}
DISPLAY_V = 0.030                    # glass centre (v, from the console centre)
GLASS = (0.130, 0.058)
LINE_V = {'status': 0.019, 'force': 0.0, 'peak': -0.019}    # v from the glass centre: line 1 the farthest (its top)
DIGIT_H = 0.012
CELLS = 4                            # hundreds, tens, units | one decimal
START_UV = (-0.045, -0.032)
SLIDER_V = -0.034
SLIDER_U = {'tension': 0.004, 'compression': 0.050}         # detents along u (world -x)
SLIDER = (0.022, 0.016, 0.012)       # carriage block (u, v, height)
SLIDER_POST = (0.005, 0.016)
SLIDER_MASS = 0.10
SLIDER_DETENT_N_PER_M, SLIDER_DETENT_PEAK_N = 300.0, 5.0
SLIDER_FRICTION_N, SLIDER_FRICTION_DAMPING = 1.0, 2.0e3
SLIDER_PATH = ROOT + '/ModeSlider'
SLIDER_JOINT = C.PHYSICS_ROOT + '/ModeRail'
DISPLAY = ROOT + '/TesterDisplay'


def bar_path(bar):
    return f'{ROOT}/Bars/{bar}'


def bar_joint_path(bar):
    return f'{C.PHYSICS_ROOT}/BarJoints/{bar}'


def kink_deg(label_mm):
    """The angle between a buckled bar's tail and main body (the two rise KINK_RISE_M to the cut from their eyes)."""
    half = M.BAR_LENGTHS_MM[label_mm] / 2000.0
    a, b = B.CUT_M + half, half - B.CUT_M
    return math.degrees(math.atan2(KINK_RISE_M, a) + math.atan2(KINK_RISE_M, b))


def console_xy():
    return B.CONSOLE_XY


def slider_x(mode):
    """World x of the slider at a detent."""
    return _xy(B.CONSOLE_XY, SLIDER_U[mode], 0.0)[0]


def slider_y():
    return _xy(B.CONSOLE_XY, 0.0, SLIDER_V)[1]


def button_layout():
    return {B.START_ID: (_xy(B.CONSOLE_XY, *START_UV), CONSOLE['size'][2], 'round')}


def seat_paths(stage=None):
    return {}


# --- small geometry helpers ------------------------------------------------------------------------------------
def _tube_mesh(stage, path, inner, outer, height, centre, colour, segments=48):
    """Render mesh of a flat ring (an eye boss): outer and inner walls, top and bottom faces."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    cx, cy, cz = centre
    points, counts, indices = [], [], []
    for z in (cz - height / 2, cz + height / 2):
        for r in (outer, inner):
            points += [Gf.Vec3f(cx + r * math.cos(2 * math.pi * i / segments), cy + r * math.sin(2 * math.pi * i / segments), z)
                       for i in range(segments)]
    ob, ib, ot, it = 0, segments, 2 * segments, 3 * segments
    for i in range(segments):
        j = (i + 1) % segments
        counts += [4, 4, 4, 4]
        indices += [ob + i, ob + j, ot + j, ot + i,          # outer wall
                    ib + j, ib + i, it + i, it + j,          # inner wall
                    ot + i, ot + j, it + j, it + i,          # top
                    ob + j, ob + i, ib + i, ib + j]          # bottom
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateExtentAttr([Gf.Vec3f(cx - outer, cy - outer, cz - height / 2), Gf.Vec3f(cx + outer, cy + outer, cz + height / 2)])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    return mesh


def _cone(stage, path, radius, height, centre, colour, material, down=False):
    """A colliding cone along z (apex up, or down)."""
    from torsion_apparatus import _collider
    tip = UsdGeom.Cone.Define(stage, path)
    tip.CreateRadiusAttr(radius)
    tip.CreateHeightAttr(height)
    tip.CreateAxisAttr('Z')
    tip.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2), Gf.Vec3f(radius, radius, height / 2)])
    api = UsdGeom.XformCommonAPI(tip)
    api.SetTranslate(Gf.Vec3d(*centre))
    if down:
        api.SetRotate(Gf.Vec3f(180.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    tip.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    _collider(tip.GetPrim(), material)
    return tip


def _pin(stage, path, xy, foot_z, height, colour, material):
    """A tester pin: straight cylinder from foot_z, then a conical tip (both collide)."""
    _cylinder(stage, path, B.PIN_R, height, (xy[0], xy[1], foot_z + height / 2), colour, material)
    _cone(stage, path + 'Tip', B.PIN_R, B.PIN_TIP, (xy[0], xy[1], foot_z + height + B.PIN_TIP / 2), colour, material)


def _collared_pin(stage, path, xy, base_z, colour, material, shank=False):
    """A board pin from base_z (the collar's underside): collar, post and tip; with its shank below for a loose pin.
    xy, base_z in the parent's frame."""
    x, y = xy
    if shank:
        straight = B.SHANK_L - B.SHANK_TIP
        _cylinder(stage, path + 'Shank', B.SHANK_R, straight, (x, y, base_z - straight / 2), colour, material)
        _cone(stage, path + 'ShankEnd', B.SHANK_R, 2 * B.SHANK_TIP, (x, y, base_z - straight - B.SHANK_TIP), colour,
              material, down=True)
    _cylinder(stage, path + 'Collar', B.COLLAR_R, B.COLLAR_H, (x, y, base_z + B.COLLAR_H / 2), colour, material)
    foot = base_z + B.COLLAR_H
    _cylinder(stage, path + 'Post', B.PIN_R, B.PIN_H, (x, y, foot + B.PIN_H / 2), colour, material)
    _cone(stage, path + 'Tip', B.PIN_R, B.PIN_TIP, (x, y, foot + B.PIN_H + B.PIN_TIP / 2), colour, material)


def _flat_poly(stage, path, points_xy, z, colour):
    """Render-only flat polygon (printed marks)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(x, y, z) for x, y in points_xy])
    mesh.CreateFaceVertexCountsAttr([len(points_xy)])
    mesh.CreateFaceVertexIndicesAttr(list(range(len(points_xy))))
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    xs, ys = [p[0] for p in points_xy], [p[1] for p in points_xy]
    mesh.CreateExtentAttr([Gf.Vec3f(min(xs), min(ys), z), Gf.Vec3f(max(xs), max(ys), z)])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    return mesh


def _disc(stage, path, xy, r, z, colour, n=24):
    return _flat_poly(stage, path, [(xy[0] + r * math.cos(2 * math.pi * i / n), xy[1] + r * math.sin(2 * math.pi * i / n))
                                    for i in range(n)], z, colour)


# --- the pegboard, the pin stand and the loose pins -----------------------------------------------------------
def pin_path(pin):
    return f'{ROOT}/Pins/{pin}'


def _plate(stage, path, rect, holes, materials):
    points, faces = B.holed_plate(rect, holes, B.BOARD_H)
    return C._mesh(stage, path, points, faces, BOARD_COLOUR, collide=True, material=materials['board'])


def _board(stage, materials):
    open_holes = [B.pin_xy(n) for n in M.LOOSE_HOLES]             # A, C, B are pressed into the board
    _plate(stage, ROOT + '/Board', B.board_rect(), open_holes, materials)
    ink = B.BOARD_H + 0.0004
    for name in M.FIXED:
        _collared_pin(stage, f'{ROOT}/Pin{name}', B.pin_xy(name), B.BOARD_H, RED if name == M.LOAD_PIN else STEEL,
                      materials['steel'])
    # Supports: a yellow triangle under A and B on the load side, B's on two rollers; letters beside the pins.
    d = B.LOAD_Y
    for name, roller in (('A', False), ('B', True)):
        px, py = B.pin_xy(name)
        y = py + 0.011 * d
        _flat_poly(stage, f'{ROOT}/Support{name}', [(px, y), (px - 0.009, y + 0.014 * d), (px + 0.009, y + 0.014 * d)],
                   ink, YELLOW)
        if roller:
            for k, dx in enumerate((-0.005, 0.005)):
                _disc(stage, f'{ROOT}/Roller{k}', (px + dx, y + 0.0175 * d), 0.0027, ink, YELLOW)
        else:
            _box(stage, f'{ROOT}/Ground{name}', (0.022, 0.0025, 0.0006), (px, y + 0.0165 * d, ink), YELLOW, collide=False)
        _text(stage, f'{ROOT}/Letter{name}', name, (px - 0.018, py + 0.004), ink, 0.010, colour=WHITE)
    cx, cy = B.pin_xy(M.LOAD_PIN)
    # the load arrow between C and the next hole in the load direction (the holes are 50 mm apart)
    _box(stage, ROOT + '/ArrowShaft', (0.0025, 0.012, 0.0006), (cx, cy + 0.017 * d, ink), RED, collide=False)
    _flat_poly(stage, ROOT + '/ArrowHead', [(cx, cy + 0.030 * d), (cx - 0.005, cy + 0.022 * d), (cx + 0.005, cy + 0.022 * d)],
               ink, RED)
    _text(stage, ROOT + '/LetterC', 'C', (cx - 0.016, cy + 0.004), ink, 0.010, colour=WHITE)


def _stand(stage, materials):
    _plate(stage, ROOT + '/PinStand', B.stand_rect(), list(B.stand_holes().values()), materials)


def _loose_pin(stage, pin, xy, materials):
    """A loose pin: one dynamic body, its frame on its axis at the collar's underside (truss_bench). Returns its path."""
    path = pin_path(pin)
    xform = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(xy[0], xy[1], B.BOARD_H + 0.0002))
    _collared_pin(stage, path + '/', (0.0, 0.0), 0.0, STEEL, materials['steel'], shank=True)
    com_z = (B.COLLAR_H + B.PIN_H) * 0.35
    _rigid(xform.GetPrim(), B.PIN_MASS_KG, (0.0, 0.0, com_z), ccd=True)
    return path


# --- bars --------------------------------------------------------------------------------------------------------
def _bar(stage, bar, kind, label_mm, xy, yaw_deg, materials):
    """The bar's two bodies (main <bar>, tail <bar>t), both in the bar's frame at xy, yaw_deg. Returns {id: path}."""
    import truss_bench
    s = M.BAR_LENGTHS_MM[label_mm] / 1000.0
    colour = KIND_COLOUR[kind]
    t = B.BAR_T
    a, d, cut = -s / 2 + BODY_INSET, s / 2 - BODY_INSET, B.CUT_M
    radial = B.EYE_R - B.EYE_HOLE_R
    tangential = 2 * B.EYE_R * math.tan(math.pi / EYE_PIECES)
    r_mid = B.EYE_HOLE_R + radial / 2
    mass = M.bar_mass_g(kind, label_mm) / 1000.0 * BAR_MASS_SCALE
    tail_share = (cut + s / 2) / s
    paths = {}
    for body, (x0, x1), e, sign, share in ((bar, (cut, d), 1, 1, 1 - tail_share),
                                            (truss_bench.tail_id(bar), (a, cut), 0, -1, tail_share)):
        path = bar_path(body)
        xform = UsdGeom.Xform.Define(stage, path)
        api = UsdGeom.XformCommonAPI(xform)
        api.SetTranslate(Gf.Vec3d(xy[0], xy[1], B.RACK_BASE_H + 0.0003))
        api.SetRotate(Gf.Vec3f(0.0, 0.0, yaw_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        _box(stage, path + '/Strip', (x1 - x0, B.BAR_W, t), ((x0 + x1) / 2, 0, t / 2), colour, material=materials['bar'])
        ex = sign * s / 2
        for k in range(EYE_PIECES):                      # ring colliders (invisible) around the hole
            th = 2 * math.pi * (k + 0.5) / EYE_PIECES
            piece = _box(stage, f'{path}/EyeRing{k:02d}', (radial, tangential, t),
                         (ex + r_mid * math.cos(th), r_mid * math.sin(th), t / 2), colour, rotate_xyz=(0, 0, math.degrees(th)),
                         material=materials['bar'])
            piece.CreateVisibilityAttr('invisible')
        _tube_mesh(stage, f'{path}/Eye{e}', B.EYE_HOLE_R, B.EYE_R, t, (ex, 0.0, t / 2), colour)
        com = ((x0 + x1) / 2 + (ex - (x0 + x1) / 2) * 0.3, 0.0, t / 2)
        if body == bar:
            kx, ky, kz = B.KNOB
            _box(stage, path + '/Knob', (kx, ky, kz), (0, 0, t + kz / 2), tuple(c * 0.8 for c in colour),
                 material=materials['bar'])
            xt = (B.KNOB[0] / 2 + s / 2 - B.EYE_R) / 2
            label(stage, path + '/Length', str(label_mm), (xt, 0.0), t + 0.0004, height_m=0.0055, align='centre',
                  colour=WHITE, up=(0.0, -1.0))
            com = (com[0], 0.0, 0.0035)
        _rigid(xform.GetPrim(), mass * share, com, ccd=True)
        paths[body] = path
    return paths


def _bar_joint(stage, bar):
    """The fixed joint of a bar's tail to its main body at the cut (both frames there, the bar's frame axes)."""
    import truss_bench
    joint = UsdPhysics.FixedJoint.Define(stage, bar_joint_path(bar))
    joint.CreateBody0Rel().SetTargets([Sdf.Path(bar_path(truss_bench.tail_id(bar)))])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(bar_path(bar))])
    at = Gf.Vec3f(B.CUT_M, 0.0, B.BAR_T / 2)
    joint.CreateLocalPos0Attr(at)
    joint.CreateLocalPos1Attr(at)
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateExcludeFromArticulationAttr(True)
    return joint


def break_bar(stage, bar):
    """A bar broken in tension: its two bodies come apart."""
    stage.RemovePrim(bar_joint_path(bar))


def bend_bar(stage, bar, angle_deg):
    """A buckling bar: its main body turned angle_deg about the bar's width from its tail, the cut rising."""
    # joint frames agree: main = tail . rot0 . rot1^-1, so rot1 = R_y(-angle) turns the main body by +angle about
    # the bar's y, which takes its far end down while the tail's rises to the cut (a tent, not a valley)
    half = math.radians(angle_deg) / 2
    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(bar_joint_path(bar)))
    joint.GetLocalRot1Attr().Set(Gf.Quatf(math.cos(half), 0.0, -math.sin(half), 0.0))


def rack_rect(name):
    """World (x0, y0, x1, y1) of a rack plate (truss_bench.RACKS): the rects of its rows."""
    rects = [B.row_rect(row) for row in B.RACKS[name]]
    return (min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects), max(r[3] for r in rects))


def _rack(stage, materials):
    ink = B.RACK_BASE_H + 0.0004
    for name in B.RACKS:
        x0, y0, x1, y1 = rack_rect(name)
        _box(stage, f'{ROOT}/Rack{name.title()}', (x1 - x0, y1 - y0, B.RACK_BASE_H),
             ((x0 + x1) / 2, (y0 + y1) / 2, B.RACK_BASE_H / 2), (0.22, 0.23, 0.26), material=materials['rack'])
    for (label, group), (axis, sx0, sy0) in B.RACK_ROWS.items():
        half = B.bar_length_m(label) / 2
        slots = B.row_slots((label, group))
        for k in range(B.RACK_SLOTS + 1):                 # ribs between the slots
            if axis == 'x':
                size, xy = (0.002, 2 * half, 0.003), (sx0 + (k - 0.5) * B.RACK_PITCH, sy0)
            else:
                size, xy = (2 * half, 0.002, 0.003), (sx0, sy0 + (k - 0.5) * B.RACK_PITCH)
            _box(stage, f'{ROOT}/RackRib{label}_{group * (B.RACK_SLOTS + 1) + k}', size,
                 (xy[0], xy[1], B.RACK_BASE_H + 0.0015), (0.30, 0.31, 0.34), material=materials['rack'])
        for k in range(len(slots)):
            _text(stage, f'{ROOT}/RackLabel{label}_{group * B.RACK_SLOTS + k + 1}', str(label),
                  B.label_xy((label, group), k), ink, 0.007, colour=WHITE)


def _tray(stage, materials):
    (cx, cy), (ix, iy), w, h = B.TRAY_XY, B.TRAY_INNER, B.TRAY_WALL, B.TRAY_H
    grey = (0.35, 0.36, 0.38)
    _box(stage, ROOT + '/TrayFloor', (ix + 2 * w, iy + 2 * w, 0.003), (cx, cy, 0.0015), grey, material=materials['rack'])
    for name, size, xy in (('N', (ix + 2 * w, w, h), (cx, cy - iy / 2 - w / 2)), ('S', (ix + 2 * w, w, h), (cx, cy + iy / 2 + w / 2)),
                           ('W', (w, iy, h), (cx - ix / 2 - w / 2, cy)), ('E', (w, iy, h), (cx + ix / 2 + w / 2, cy))):
        _box(stage, f'{ROOT}/TrayWall{name}', size, (xy[0], xy[1], h / 2), grey, material=materials['rack'])
    _text(stage, ROOT + '/TrayLabel', 'SCRAP', (cx, cy - iy / 2 - w - 0.008), 0.0004, 0.008, colour=WHITE)  # in front


# --- the tester --------------------------------------------------------------------------------------------------
def bed_parts():
    """{name: (size, centre)} of the bed's static boxes (world x, y; z above the table)."""
    x, top = B.TESTER_X, B.TESTER_TOP
    p0, p1 = PEDESTAL_Y
    e0, e1 = BED_END_Y
    w = BED_HALF_W - SLOT_HALF
    out = {'Pedestal': ((2 * BED_HALF_W, p1 - p0, top), (x, (p0 + p1) / 2, top / 2)),
           'TesterBedEnd': ((2 * BED_HALF_W, e1 - e0, top), (x, (e0 + e1) / 2, top / 2))}
    for side, sx in (('L', -1), ('R', 1)):
        out[f'TesterBed{side}'] = ((w, e0 - p1, PLATE_T), (x + sx * (SLOT_HALF + w / 2), (p1 + e0) / 2, top - PLATE_T / 2))
    return out


def _tester(stage, carriage_label, materials):
    x = B.TESTER_X
    fx, fy = B.tester_pins(0.0)['fixed']
    for name, (size, centre) in bed_parts().items():
        _box(stage, f'{ROOT}/{name}', size, centre, DARK if name == 'Pedestal' else (0.55, 0.56, 0.60),
             material=materials['steel'])
    _pin(stage, ROOT + '/TesterPinFixed', (fx, fy), B.TESTER_TOP, B.TESTER_PIN_H, STEEL, materials['steel'])
    ink = 0.0004
    for mm in B.TESTER_SPACINGS_MM:
        y = B.detent_y(mm)
        _box(stage, f'{ROOT}/TesterMark{mm}', (0.010, 0.0012, 0.0006), (HOLE_X, y, ink), WHITE, collide=False)
        _text(stage, f'{ROOT}/TesterNumber{mm}', str(mm), (HOLE_X - 0.014, y), ink, 0.007, colour=WHITE)
    # the crosshead carriage (dynamic, on its rail joint): the web in the slot with its pin, the pointer, the tongue
    # under the right plate with the push handle beside the bed
    y = B.detent_y(carriage_label)
    xform = UsdGeom.Xform.Define(stage, CARRIAGE)
    z0 = TONGUE_Z - 0.003                                          # the carriage's origin: the tongue's underside
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(x, y, z0))
    wh = B.TESTER_TOP - WEB_DROP - z0
    _box(stage, CARRIAGE + '/Web', (*WEB, wh), (0, 0, wh / 2), DARK, material=materials['rubber'])
    _pin(stage, CARRIAGE + '/Pin', (0.0, 0.0), wh, B.TESTER_PIN_H + WEB_DROP, STEEL, materials['steel'])
    reach = (x - HOLE_X) - WEB[0] / 2 + 0.002                       # from the web's face over the detent marks
    _box(stage, CARRIAGE + '/Pointer', (reach, 0.0012, 0.0006), (-WEB[0] / 2 - reach / 2, 0.0, 0.0012 - z0), RED,
         collide=False)
    tongue_x0, tongue_x1 = 0.0, BED_HALF_W + TONGUE_DX - 0.002
    _box(stage, CARRIAGE + '/Tongue', (tongue_x1 - tongue_x0, 0.012, 0.006), ((tongue_x0 + tongue_x1) / 2, 0, 0.003),
         DARK, material=materials['rubber'])
    hh = HANDLE_TOP - TONGUE_Z - 0.003                             # the handle stands on the tongue, beside the bed
    _box(stage, CARRIAGE + '/Handle', (*HANDLE, hh), (BED_HALF_W + TONGUE_DX - 0.006, 0, 0.006 + hh / 2),
         (0.90, 0.24, 0.02), material=materials['rubber'])
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim()).CreateKinematicEnabledAttr(False)
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(xform.GetPrim())
    body.CreateSolverPositionIterationCountAttr(16)
    body.CreateSolverVelocityIterationCountAttr(8)
    body.CreateSleepThresholdAttr(0.0)
    UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(CARRIAGE_MASS)


def _console(stage, mode, materials, looks):
    cx, cy = B.CONSOLE_XY
    sx, sy, sz = CONSOLE['size']
    _box(stage, ROOT + '/Console', (sx, sy, sz), (cx, cy, sz / 2), (0.14, 0.24, 0.42), material=materials['rubber'])
    top = sz + 0.0004
    # display
    gx, gy = _xy(B.CONSOLE_XY, 0.0, DISPLAY_V)
    UsdGeom.Xform.Define(stage, DISPLAY)
    _bind(_box(stage, DISPLAY + '/Glass', (GLASS[0], GLASS[1], 0.0006), (gx, gy, sz + 0.0003), (0.004, 0.018, 0.010),
               collide=False), looks['glass'])
    z = sz + 0.0008
    centre = (gx, gy)
    words = {'READY': -0.044, 'RUN': -0.044, 'DONE': -0.044, 'TENSION': 0.030, 'COMPRESSION': 0.030}
    for word, u in words.items():
        mesh = _text(stage, f'{DISPLAY}/Word{word}', word, _xy(centre, u, LINE_V['status']), z, 0.0075, colour=SEGMENT_ON)
        _bind(mesh, looks['lit'])
    for line, name in (('force', 'FORCE'), ('peak', 'PEAK')):
        _text(stage, f'{DISPLAY}/{name}Label', name, _xy(centre, -0.040, LINE_V[line]), z, 0.0065, colour=(0.55, 0.75, 0.6))
        _text(stage, f'{DISPLAY}/{name}Unit', 'N', _xy(centre, 0.052, LINE_V[line]), z, 0.0065, colour=(0.55, 0.75, 0.6))
        for (i, seg), ((u, v), (du, dv)) in digit_rects(line).items():
            _bind(C._flat(stage, f'{DISPLAY}/{line}{i}{seg}', _xy(centre, u, v), (du, dv), z, SEGMENT_ON), looks['lit'])
        (u, v), (du, dv) = point_rect(line)
        _bind(C._flat(stage, f'{DISPLAY}/{line}Point', _xy(centre, u, v), (du, dv), z, SEGMENT_ON), looks['lit'])
    for path in display_paths():
        _remember_lit(stage.GetPrimAtPath(path))
    # START
    (bx, by), bz, _ = button_layout()[B.START_ID]
    C._button(stage, B.START_ID, (bx, by), bz, 'round', (0.15, 0.62, 0.25), 'START', materials, text_colour=WHITE)
    # MODE slider: a slot, the detent names, the carriage with its post
    u0, u1 = SLIDER_U['tension'], SLIDER_U['compression']
    sxy = _xy(B.CONSOLE_XY, (u0 + u1) / 2, SLIDER_V)
    _box(stage, ROOT + '/ModeSlot', (u1 - u0 + 0.024, 0.006, 0.0006), (sxy[0], sxy[1], top), (0.03, 0.03, 0.035), collide=False)
    for k, (mode_name, text) in enumerate((('tension', 'TENSION'), ('compression', 'COMPRESSION'))):
        _text(stage, f'{ROOT}/Mode{mode_name.title()}', text, _xy(B.CONSOLE_XY, SLIDER_U[mode_name], SLIDER_V + 0.013 + 0.008 * k),
              top, 0.0055, colour=WHITE)
    xform = UsdGeom.Xform.Define(stage, SLIDER_PATH)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(slider_x(mode), slider_y(), sz + CARRIAGE_GAP + SLIDER[2] / 2))
    _box(stage, SLIDER_PATH + '/Block', (SLIDER[0], SLIDER[1], SLIDER[2]), (0, 0, 0), (0.30, 0.31, 0.33), material=materials['rubber'])
    _cylinder(stage, SLIDER_PATH + '/Post', SLIDER_POST[0], SLIDER_POST[1], (0, 0, SLIDER[2] / 2 + SLIDER_POST[1] / 2),
              (0.95, 0.78, 0.05), materials['rubber'])
    _rigid(xform.GetPrim(), SLIDER_MASS)


def digit_rects(line):
    """Glass-centre rectangles {(cell, seg): ((u, v), (du, dv))} of a force line's four cells (centres; v away from the
    robot, the digits' tops away from it)."""
    h = DIGIT_H
    w, t = h * 0.5, h * 0.15
    pitch = h * 0.78
    u0 = -0.004
    out = {}
    for i in range(CELLS):
        u = u0 + i * pitch + (h * 0.25 if i == CELLS - 1 else 0.0)
        for seg, (fu, fv, horizontal) in SEVEN.items():
            out[(i, seg)] = ((u + fu * w, LINE_V[line] - h / 2 + fv * h), (w, t) if horizontal else (t, h / 2))
    return out


def point_rect(line):
    h = DIGIT_H
    t = h * 0.15
    u = -0.004 + (CELLS - 2) * h * 0.78 + h * 0.5 + (h * 0.78 - h * 0.5 + h * 0.25) / 2   # between units and decimal
    return (u, LINE_V[line] - h / 2 + t / 2), (t * 1.4, t * 1.4)


def load_display(stage, looks):
    """The loading frame's display (truss_frame.DISPLAY_XY): a glass with one force line of four seven-segment cells
    as the tester's (the rope's pull, 0.1 N), LOAD and N beside it."""
    F = truss_frame
    centre = F.DISPLAY_XY
    z = F.DISPLAY_SIZE[2] + 0.0008
    UsdGeom.Xform.Define(stage, F.DISPLAY_PATH)
    _text(stage, F.DISPLAY_PATH + '/Label', 'LOAD', _xy(centre, -0.040, 0.0), z, 0.0065, colour=(0.55, 0.75, 0.6))
    _text(stage, F.DISPLAY_PATH + '/Unit', 'N', _xy(centre, 0.052, 0.0), z, 0.0065, colour=(0.55, 0.75, 0.6))
    for (i, seg), ((u, v), (du, dv)) in digit_rects('force').items():
        _bind(C._flat(stage, f'{F.DISPLAY_PATH}/load{i}{seg}', _xy(centre, u, v), (du, dv), z, SEGMENT_ON), looks['lit'])
    (u, v), (du, dv) = point_rect('force')
    _bind(C._flat(stage, F.DISPLAY_PATH + '/loadPoint', _xy(centre, u, v), (du, dv), z, SEGMENT_ON), looks['lit'])
    for path in [f'{F.DISPLAY_PATH}/load{i}{seg}' for i in range(CELLS) for seg in SEVEN] + [F.DISPLAY_PATH + '/loadPoint']:
        _remember_lit(stage.GetPrimAtPath(path))
    show_load(stage, B.force_text(F.HANGER_KG * F.G), None)


def show_load(stage, text, previous):
    """Light the load display's four cells (B.force_text) where they differ from the previous text."""
    if text == previous:
        return text
    P = truss_frame.DISPLAY_PATH
    for i, char in enumerate(text):
        if previous is not None and previous[i] == char:
            continue
        for seg in SEVEN:
            C._light(stage, f'{P}/load{i}{seg}', seg in GLYPHS.get(char, ''))
    if previous is None or bool(previous.strip()) != bool(text.strip()):
        C._light(stage, P + '/loadPoint', bool(text.strip()))
    return text


def display_paths():
    words = [f'{DISPLAY}/Word{w}' for w in ('READY', 'RUN', 'DONE', 'TENSION', 'COMPRESSION')]
    return words + [f'{DISPLAY}/{line}{i}{seg}' for line in ('force', 'peak') for i in range(CELLS) for seg in SEVEN] + \
        [f'{DISPLAY}/{line}Point' for line in ('force', 'peak')]


def show(stage, screen, previous):
    """Light what the tester shows ({'state', 'mode', 'force', 'peak'} from TrussBench.screen) where it differs from
    the previously shown screen."""
    if screen == previous:
        return screen
    old = previous or {}
    if old.get('state') != screen['state'] or previous is None:
        for word in ('READY', 'RUN', 'DONE'):
            C._light(stage, f'{DISPLAY}/Word{word}', word == screen['state'])
    if old.get('mode') != screen['mode'] or previous is None:
        C._light(stage, f'{DISPLAY}/WordTENSION', screen['mode'] == 'tension')
        C._light(stage, f'{DISPLAY}/WordCOMPRESSION', screen['mode'] == 'compression')
    for line in ('force', 'peak'):
        text = screen[line]
        before = old.get(line)
        for i, char in enumerate(text):
            if before is not None and before[i] == char:
                continue
            for seg in SEVEN:
                C._light(stage, f'{DISPLAY}/{line}{i}{seg}', seg in GLYPHS.get(char, ''))
        if before is None or bool(before.strip()) != bool(text.strip()):
            C._light(stage, f'{DISPLAY}/{line}Point', bool(text.strip()))
    return screen


# --- build -------------------------------------------------------------------------------------------------------
def build(stage, config):
    """config: {'bars': {bar: {'kind', 'label', 'slot'}}, 'pins': {pin: stand slot}, 'carriage_label': the detent the
    crosshead starts on, 'mode': the slider's starting detent}. Returns objects (bars, loose pins and the weights of
    the loading frame, truss_frame), buttons (START), seats (none: the backend classifies)."""
    retire_hooke_apparatus(stage)
    repair_cube_extents(stage)
    z0 = table_top_z(stage)
    root = UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(0, 0, z0))
    materials = {'rubber': _material(stage, '/World/PhysicsMaterials/TrussRubber', 0.8, 0.7),
                 'steel': _material(stage, '/World/PhysicsMaterials/TrussSteel', 0.30, 0.25),
                 'board': _material(stage, '/World/PhysicsMaterials/TrussBoard', 0.45, 0.40),
                 'rack': _material(stage, '/World/PhysicsMaterials/TrussRack', 0.45, 0.40),
                 'bar': _material(stage, '/World/PhysicsMaterials/TrussBar', 0.35, 0.30)}
    looks = {'glass': _render_material(stage, ROOT + '/Looks/TesterGlass', (0.004, 0.018, 0.010), matte=True),
             'lit': _render_material(stage, ROOT + '/Looks/TesterLit', SEGMENT_ON, emission=SEGMENT_EMISSION, matte=True)}
    _board(stage, materials)
    _stand(stage, materials)
    truss_loading.build(stage)
    weights = truss_frame.build(stage, materials)
    load_display(stage, looks)
    _rack(stage, materials)
    _tray(stage, materials)
    _tester(stage, config.get('carriage_label', 100), materials)
    _console(stage, config.get('mode', 'tension'), materials, looks)
    UsdGeom.Xform.Define(stage, ROOT + '/Bars')
    slots = B.rack_slots()
    objects = {}
    for bar, spec in config['bars'].items():
        (x, y), _, _ = slots[spec['slot']]
        objects.update(_bar(stage, bar, spec['kind'], spec['label'], (x, y), B.slot_yaw_deg(spec['slot']), materials))
    UsdGeom.Xform.Define(stage, ROOT + '/Pins')
    stand = B.stand_holes()
    for pin, slot in config['pins'].items():
        objects[pin] = _loose_pin(stage, pin, stand[slot], materials)
    objects.update(weights)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(stage.GetPrimAtPath(TABLE_TOP))
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    show(stage, {'state': 'READY', 'mode': config.get('mode', 'tension'), 'force': '    ', 'peak': '    '}, None)
    C.finish_looks(stage)
    import truss_looks                                 # render-only looks over the drawn colliders (build time only)
    truss_looks.apply(stage, config)
    return {'objects': objects, 'buttons': {B.START_ID: C.button_path(B.START_ID)}, 'seats': {}}


def _rail_joint(stage, path, body, axis, lower, upper, friction_n, damping):
    joint = UsdPhysics.PrismaticJoint.Define(stage, path)
    world = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(body)).ExtractTranslation()
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(world))
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateAxisAttr(axis)
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
    drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(damping)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateMaxForceAttr(friction_n)
    return float(world[0]), float(world[1])


def author_joints(stage, buttons):
    """The crosshead's rail (prismatic along world y, end stops TRAVEL_MARGIN_M beyond the first and last detent), the
    MODE slider's rail (along world x, end stops at its detents) and START's spring-returned cap (the kit's)."""
    UsdGeom.Scope.Define(stage, C.PHYSICS_ROOT)
    _, y = _rail_joint(stage, CARRIAGE_JOINT, CARRIAGE, 'Y', 0.0, 0.0, RAIL_FRICTION_N, RAIL_FRICTION_DAMPING)
    ys = [B.detent_y(mm) for mm in B.TESTER_SPACINGS_MM]
    joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(CARRIAGE_JOINT))
    joint.GetLowerLimitAttr().Set(min(ys) - TRAVEL_MARGIN_M - y)
    joint.GetUpperLimitAttr().Set(max(ys) + TRAVEL_MARGIN_M - y)
    x, _ = _rail_joint(stage, SLIDER_JOINT, SLIDER_PATH, 'X', 0.0, 0.0, SLIDER_FRICTION_N, SLIDER_FRICTION_DAMPING)
    xs = [slider_x(m) for m in B.MODES]
    slider = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(SLIDER_JOINT))
    slider.GetLowerLimitAttr().Set(min(xs) - x)
    slider.GetUpperLimitAttr().Set(max(xs) - x)
    truss_frame.author_joints(stage)
    UsdGeom.Scope.Define(stage, C.PHYSICS_ROOT + '/BarJoints')
    for prim in stage.GetPrimAtPath(ROOT + '/Bars').GetChildren():
        name = prim.GetName()
        if not name.endswith('t'):
            _bar_joint(stage, name)
    C.author_joints(stage, buttons)


seat_frames = C.seat_frames

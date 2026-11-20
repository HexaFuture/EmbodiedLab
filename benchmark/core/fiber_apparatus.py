"""Fiber coupling bench apparatus (fiber_coupling): an optical breadboard with a HeNe laser and its
power supply, two rear-adjust kinematic mirror mounts, the removable alignment target after the second mirror, the
fiber coupler (lens cartridge seat, alignment cap slot, fiber port on a focus stage), the lens box and the optical
power meter on a yellow patch cord. Built on the chemistry kit's root and helpers (collar-type seats, a POWER push
button with its spring joint, printed text, LCD segments), so the chemistry backend's grasp, seat and button
machinery runs it; the knobs are fiber_knob knobs (dynamic bodies on revolute joints with friction).

World layout (the robot stands at (0.95, 0.35) facing -y; z above the lab table top unless noted): the breadboard's
top is at BOARD_TOP; the beam runs BEAM_H above it. The beam path (the model's coordinate p, metres from the laser's
exit, fiber_model L1, L2, L_T, L3): the laser points +y toward M1; M1 turns the beam +x to M2; M2 turns it -y, away
from the robot, through the target's position (L_T) to the coupler (L3). The mirrors face away from the robot; their
adjuster knobs are on the mounts' backs, 45 deg either side of the robot's line of sight, axes horizontal.

Mirror mount (1 inch, rear adjust): mount frame X along the mount's back (into the mount from the mirror face), Y to
the right seen from behind, Z up, origin on the mirror face centre (on the beam). Front plate (mirror) X 0.001-0.011,
back plate X 0.014-0.024, 38 mm square; the pivot ball and two adjuster screws at three corners, 15 mm from the centre
(knobs 42 mm apart). The screws are right-hand threads in the back plate; a knob turned counter-clockwise seen from its
end (+ in fiber_knob) backs its screw out and the springs pull the front plate's corner back. Corners: M1 pivot
bottom-left, H (horizontal steering) bottom-right, V top-left; M2 mirrored (pivot bottom-right, H bottom-left, V
top-right): with these a + turn of an H knob moves the spot at the coupler toward world +x and a + turn of a V knob
moves it up, for both mirrors (the model's + x and + y). The front plate is a kinematic body the backend tilts about
the pivot by the model's mirror tilts (set_front_plate).

Focus stage: a lead screw along the beam beside the coupler, its knob at the stage's front facing the robot; the nut
rides the carriage, which carries the fiber port. A + turn (counter-clockwise from the front) moves the port away
from the lens (the model's + dz). The carriage is a kinematic body the backend moves (set_carriage).

Holders (lens cartridges, alignment cap, target): dynamic bodies with their origin at the bottom centre, a white rim
handle on top (the optics kit's), standing in keyed rectangular wells (a cartridge cannot turn in its well; turned
around it seats but its lens then sits on the wrong side, as on a real bench). Cartridge: base 18 x 12 mm, the lens
on a post AXIS_LOCAL above the bottom, its disc where that cartridge's focus lands on the fiber face (the lens
position differs by focal length). Cap and target: a plate across the beam on a stem and a tongue; the cap black with
a white crosshair and rings at 1 mm steps, the target a white card with a black crosshair, rings and a 2 mm hole.

Render only: the spots where the beam meets a surface (set_spot: the target card, the cap, a lens cell it clips, the
fiber port's face, anything else in its way, a faint scatter on each mirror; the beam itself is not seen in the clean
air of the lab), the meter's LCD (show_meter), the pilot lamp and the laser's
exit (set_power), texts, the patch cord and the power cable.
"""
from __future__ import annotations

import math

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import chem_apparatus as C
from chem_apparatus import ROOT, _button, _remember_lit, _rigid, _text
from torsion_apparatus import TABLE_TOP, _box, _collider, _material, repair_cube_extents, retire_hooke_apparatus, \
    table_top_z
import fiber_knob as KN
import fiber_model as M
import visual_kit as V

# --- layout --------------------------------------------------------------------------------------------------------
BOARD = {'x': (0.55, 1.45), 'y': (-0.595, 0.005), 'h': 0.0127, 'pitch': 0.025, 'hole_r': 0.0033}
BOARD_TOP = BOARD['h']
BEAM_H = 0.140                       # the beam about 0.14 m above the bench top
BEAM_Z = BOARD_TOP + BEAM_H
MIRRORS = {'M1': {'xy': (0.85, -0.02), 'yaw': 135.0, 'sigma': -1.0, 'corners': {'pivot': (-1, -1), 'H': (1, -1),
                                                                                 'V': (-1, 1)}},
           'M2': {'xy': (1.10, -0.02), 'yaw': 45.0, 'sigma': 1.0, 'corners': {'pivot': (1, -1), 'H': (-1, -1),
                                                                                'V': (1, 1)}}}
LASER_X = MIRRORS['M1']['xy'][0]
LASER_EXIT_Y = MIRRORS['M1']['xy'][1] - M.L1
BEAM_X = MIRRORS['M2']['xy'][0]      # the final leg (M2 to the coupler) runs along world -y at this x
PLATE = 0.038                        # mount plates (square)
CORNER = 0.015                       # pivot and screws from the plate centre
FRONT_X = (0.001, 0.011)
BACK_X = (0.014, 0.024)
MIRROR_R, MIRROR_T = 0.0127, 0.006
POST_R = 0.00635
HOLDER_R = 0.0125                    # post holder
HOLDER_TOP = BOARD_TOP + 0.070
PIVOT_LOCAL_X = 0.0125

AXIS_LOCAL = 0.060                   # lens centre / plate centre above a holder's bottom
SEAT_FLOOR = C.SEAT_FLOOR
SEAT_BASE_Z = BEAM_Z - AXIS_LOCAL - SEAT_FLOOR      # collar base of the coupler and target seats
FIBER_Y = -0.278                     # fiber face (the focus stage at its start)
LENS_SEAT_Y = FIBER_Y + (M.LENSES_MM[0] + M.LENSES_MM[-1]) / 2000.0
# keyed wells: (well half x, half y); mouth and outer walls 3 mm and 6 mm beyond the well
CART_BASE = (0.018, 0.012, 0.030)
CART_WELL = (0.0105, 0.0075)
TONGUE = (0.020, 0.006, 0.015)
SLOT_WELL = (0.0115, 0.0035)
COLLAR_H, WELL_DEPTH, CHAMFER = C.COLLAR_H, C.WELL_DEPTH, C.CHAMFER_DEPTH
CAP_SEAT_Y = LENS_SEAT_Y + (CART_WELL[1] + 0.006) + (SLOT_WELL[1] + 0.006)
TARGET_SEAT_Y = MIRRORS['M2']['xy'][1] - (M.L_T - M.L2) - 0.001      # the target card's face at L_T
PLATE_SIZE = (0.030, 0.002, 0.030)   # cap / target plate (x, y, z)
HOLE_R = 0.001                       # the target's hole
RING_R, RING_T = 0.008, 0.006        # lens cell
LENS_CA_R = 0.003                    # lens clear aperture radius (render)
FOCUS_KNOB = {'x': BEAM_X + 0.120, 'z': BEAM_Z - 0.015, 'face_y': FIBER_Y - 0.004}
STAGE = {'x': (BEAM_X - 0.022, FOCUS_KNOB['x'] + 0.012), 'y': (FIBER_Y - 0.092, FIBER_Y - 0.004)}
PORT = {'size': (0.024, 0.020, 0.024)}           # fiber port block on the carriage (x, y, z), front face at FIBER_Y
LENS_BOX = {'centre': (0.47, 0.45), 'base': (0.350, 0.050, 0.012), 'lens_pitch': 0.050, 'slot_pitch': 0.050}
PSU = {'centre': (0.66, -0.165), 'size': (0.120, 0.090, 0.065)}
POWER_ID = 'power'
METER = {'centre': (1.36, -0.300), 'body': (0.170, 0.130, 0.040), 'panel': (0.170, 0.075, 0.050), 'tilt_deg': 42.0}
METER_DIGIT_H = 0.024
METER_CELLS = 3

LENS_IDS = tuple(f'lens_{k + 1}' for k in range(len(M.LENSES_MM)))
LENS_TEXT = ('4.51', '6.24', '8.00', '11.00', '15.29')
CAP_ID, TARGET_ID = 'cap', 'target'
KNOB_KEYS = ('M1H', 'M1V', 'M2H', 'M2V', 'F')      # model order (fiber_model.KNOBS: M1x, M1y, M2x, M2y, focus)
KNOB_IDS = {k: f'knob_{k}' for k in KNOB_KEYS}
KNOB_ROOT = '/World/FiberKnobs'                  # knob bodies and joints in world coordinates (untransformed)
LOOKS = ROOT + '/FiberLooks'
FRONT_PATH = ROOT + '/{}Front'
CARRIAGE_PATH = ROOT + '/FocusCarriage'
SPOT_PATH = ROOT + '/Spot_{}'
LAMP_MAT = LOOKS + '/PilotLamp'
EXIT_MAT = LOOKS + '/LaserExit'
LAMP_ON, LAMP_OFF = (1.0, 0.25, 0.08), (0.20, 0.06, 0.05)
RED = (1.0, 0.06, 0.03)
SPOT_LAYERS = ((1.6, 0.12), (1.0, 0.45), (0.55, 1.0))      # radius (x beam radius w), glow
SPOT_GLOW = 60.0
SPOT_NAMES = ('target', 'cap', 'lens', 'fiber', 'block', 'm1', 'm2')   # the surfaces a spot can be drawn on
SEATS = ('coupler:lens', 'coupler:cap', 'bench:target') + tuple(f'box:{k + 1}' for k in range(len(LENS_IDS))) + \
    ('box:cap', 'box:target')


def seat_path(sid):
    return f"{ROOT}/Seat_{sid.replace(':', '_')}"


def knob_path(key):
    return f'{KNOB_ROOT}/{KNOB_IDS[key]}'


def holder_path(oid):
    return f'{ROOT}/{oid}'


def front_path(mirror):
    return FRONT_PATH.format(mirror)


# --- frames ----------------------------------------------------------------------------------------------------------
def mount_rotation(mirror):
    a = math.radians(MIRRORS[mirror]['yaw'])
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def mount_origin(mirror):
    x, y = MIRRORS[mirror]['xy']
    return np.array([x, y, BEAM_Z])


def mount_point(mirror, local):
    return mount_origin(mirror) + mount_rotation(mirror) @ np.asarray(local, float)


def corner_local(mirror, name, x):
    su, sv = MIRRORS[mirror]['corners'][name]
    return np.array([x, su * CORNER, sv * CORNER])


def knob_layout(z0):
    """{key: (face point (world), outward axis (world))} of the five knobs (z0: the lab table top, world)."""
    out = {}
    for mirror in ('M1', 'M2'):
        axis = mount_rotation(mirror) @ np.array([1.0, 0.0, 0.0])
        for name in ('H', 'V'):
            p = mount_point(mirror, corner_local(mirror, name, BACK_X[1]))
            out[f'{mirror}{name}'] = (p + np.array([0.0, 0.0, z0]), axis)
    out['F'] = (np.array([FOCUS_KNOB['x'], FOCUS_KNOB['face_y'], FOCUS_KNOB['z'] + z0]), np.array([0.0, 1.0, 0.0]))
    return out


def path_point(p):
    """Nominal beam point (ROOT frame) at path coordinate p (m from the laser's exit)."""
    m1x, m1y = MIRRORS['M1']['xy']
    m2x, m2y = MIRRORS['M2']['xy']
    if p <= M.L1:
        return np.array([m1x, m1y - (M.L1 - p), BEAM_Z])
    if p <= M.L2:
        return np.array([m1x + (p - M.L1), m1y, BEAM_Z])
    return np.array([m2x, m2y - (p - M.L2), BEAM_Z])


def path_of_y(y):
    """Path coordinate of a point on the final leg at world y."""
    return M.L2 + (MIRRORS['M2']['xy'][1] - y)


CAP_FACE_Y = CAP_SEAT_Y + PLATE_SIZE[1] / 2
TARGET_FACE_Y = TARGET_SEAT_Y + PLATE_SIZE[1] / 2
P_CAP = path_of_y(CAP_FACE_Y)
P_TARGET = path_of_y(TARGET_FACE_Y)
P_FIBER = path_of_y(FIBER_Y)


def lens_disc_y(k):
    """World y of cartridge k's lens disc when seated the right way round."""
    return FIBER_Y + M.LENSES_MM[k] / 1000.0


# --- primitives --------------------------------------------------------------------------------------------------------
def _hide(prim):
    UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)


def _coll_box(stage, path, size, centre, physics, rotate_xyz=(0, 0, 0)):
    cube = _box(stage, path, size, centre, (0.1, 0.1, 0.1), rotate_xyz=rotate_xyz, material=physics)
    _hide(cube.GetPrim())
    return cube


def _coll_cyl(stage, path, r, h, centre, physics, axis='Z'):
    geom = UsdGeom.Cylinder.Define(stage, path)
    geom.CreateRadiusAttr(r)
    geom.CreateHeightAttr(h)
    geom.CreateAxisAttr(axis)
    half = {'X': (h / 2, r, r), 'Y': (r, h / 2, r), 'Z': (r, r, h / 2)}[axis]
    geom.CreateExtentAttr([Gf.Vec3f(*[-v for v in half]), Gf.Vec3f(*half)])
    UsdGeom.XformCommonAPI(geom).SetTranslate(Gf.Vec3d(*[float(v) for v in centre]))
    geom.CreateVisibilityAttr('invisible')
    _collider(geom.GetPrim(), physics)
    return geom


def _rbox(stage, path, size, centre, mat, radius=0.0008, rotate_xyz=None, tile=0.05):
    """Render-only rounded box."""
    r = min(radius, 0.25 * min(size))
    geo = V.rounded_box(size, r) if r > 0 else V.prism([(-size[0] / 2, -size[1] / 2), (size[0] / 2, -size[1] / 2),
                                                        (size[0] / 2, size[1] / 2), (-size[0] / 2, size[1] / 2)],
                                                       -size[2] / 2, size[2] / 2)
    return V.define(stage, path, geo, mat, tile=tile, translate=tuple(float(v) for v in centre),
                    rotate_xyz=rotate_xyz)


def solid_box(stage, path, size, centre, mat, physics, radius=0.0008, rotate_xyz=(0, 0, 0), tile=0.05):
    """A box collider (invisible) with its rounded look beside it."""
    _coll_box(stage, path, size, centre, physics, rotate_xyz)
    _rbox(stage, path + 'Look', size, centre, mat, radius, rotate_xyz if any(rotate_xyz) else None, tile)
    return path


def span_box(stage, path, xs, ys, zs, mat, physics, radius=0.0008):
    size = (xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0])
    centre = ((xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2, (zs[0] + zs[1]) / 2)
    return solid_box(stage, path, size, centre, mat, physics, radius)


_AXIS_ROT = {'Z': None, 'Y': (-90.0, 0.0, 0.0), 'X': (0.0, 90.0, 0.0)}


def _cyl_geo(r, h, chamfer=0.0005, segments=48):
    c = min(chamfer, r * 0.3, h * 0.3)
    return V.lathe([(0.0, -h / 2), (r - c, -h / 2), (r, -h / 2 + c), (r, h / 2 - c), (r - c, h / 2), (0.0, h / 2)],
                   segments)


def solid_cyl(stage, path, r, h, centre, mat, physics, axis='Z', chamfer=0.0005, segments=48, collide=True):
    """A cylinder collider (invisible) along axis with its chamfered look beside it."""
    if collide:
        _coll_cyl(stage, path, r, h, centre, physics, axis)
    V.define(stage, path + 'Look', _cyl_geo(r, h, chamfer, segments), mat, tile=0.02,
             translate=tuple(float(v) for v in centre), rotate_xyz=_AXIS_ROT[axis])
    return path


def _rect_collar(stage, path, centre, base_z, well, colour, physics, yaw_deg=0.0):
    """A keyed seat: raised rectangular collar, chamfered mouth and a rectangular well (static triangle mesh), the
    chemistry kit's collar topology (its floor ring is points 16-19, as chem_apparatus.seat_frames reads)."""
    wx, wy = well
    rings = [((wx + 0.006, wy + 0.006), 0.0), ((wx + 0.006, wy + 0.006), COLLAR_H), ((wx + 0.003, wy + 0.003), COLLAR_H),
             ((wx, wy), COLLAR_H - CHAMFER), ((wx, wy), SEAT_FLOOR)]
    a = math.radians(yaw_deg)
    ca, sa = math.cos(a), math.sin(a)
    points = []
    for (hx, hy), z in rings:
        for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
            points.append((centre[0] + ca * x - sa * y, centre[1] + sa * x + ca * y, base_z + z))
    faces = []

    def band(i0, i1):
        for i in range(4):
            j = (i + 1) % 4
            faces.append([4 * i0 + i, 4 * i0 + j, 4 * i1 + j, 4 * i1 + i])
    band(0, 1)
    band(1, 2)
    band(2, 3)
    band(3, 4)
    faces.append([16, 17, 18, 19])
    faces.append([3, 2, 1, 0])
    return C._mesh(stage, path, points, faces, colour, collide=True, material=physics)


def _emissive(stage, path, colour, emission, opacity=None):
    mat = V.material(stage, path, colour, roughness=0.9, emission=emission)
    if opacity is not None:
        UsdShade.Shader(stage.GetPrimAtPath(path + '/Surface')).CreateInput(
            'opacity', Sdf.ValueTypeNames.Float).Set(float(opacity))
    return mat


def set_emission(stage, material_path, colour, emission):
    shader = UsdShade.Shader(stage.GetPrimAtPath(material_path + '/Surface'))
    if not shader:
        return
    attr = shader.GetInput('emissiveColor') or shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f)
    attr.Set(Gf.Vec3f(*[float(c) * emission for c in colour]))


def _decal(stage, path, image, size, centre, normal_axis, mat_name=None, emission=0.0, colour=(1, 1, 1)):
    """A printed decal (render only): image (path to an RGBA png) on a quad of size (w, h) at centre, facing
    normal_axis ('+y', '+z', '-x' ...), upright (its top toward +z, or toward -y on a horizontal face)."""
    w, h = size[0] / 2, size[1] / 2
    if normal_axis == '+y':           # read from +y (the robot side): right toward -x, up +z
        pts = [(w, 0, -h), (-w, 0, -h), (-w, 0, h), (w, 0, h)]
    elif normal_axis == '-y':
        pts = [(-w, 0, -h), (w, 0, -h), (w, 0, h), (-w, 0, h)]
    elif normal_axis == '+x':
        pts = [(0, -w, -h), (0, w, -h), (0, w, h), (0, -w, h)]
    elif normal_axis == '-x':
        pts = [(0, w, -h), (0, -w, -h), (0, -w, h), (0, w, h)]
    else:                             # '+z': read from the robot side, top toward -y
        pts = [(w, h, 0), (-w, h, 0), (-w, -h, 0), (w, -h, 0)]
    geo = V.Geo(pts, [(0, 1, 2, 3)])
    mat = V.material(stage, f'{LOOKS}/Decal_{mat_name or path.rsplit("/", 1)[-1]}', colour, roughness=0.6,
                     cutout=image, emission=emission)
    return V.define(stage, path, geo, mat, uv=[(0, 0), (1, 0), (1, 1), (0, 1)],
                    translate=tuple(float(v) for v in centre))


# --- textures (PIL) ----------------------------------------------------------------------------------------------------
def _board_texture(folder):
    """One 25 mm cell of the breadboard top: black anodizing with an M6 tapped hole (dark bore, bright chamfer)."""
    import os
    from PIL import Image, ImageDraw
    path = os.path.join(folder, 'breadboard_cell_v1.png')
    if os.path.exists(path):
        return path
    px = 128
    img = Image.new('L', (px, px), 220)
    d = ImageDraw.Draw(img)
    c = px / 2
    r_ch = BOARD['hole_r'] / BOARD['pitch'] * px * 1.25
    r_bore = BOARD['hole_r'] / BOARD['pitch'] * px
    d.ellipse([c - r_ch, c - r_ch, c + r_ch, c + r_ch], fill=255)
    d.ellipse([c - r_bore, c - r_bore, c + r_bore, c + r_bore], fill=25)
    d.ellipse([c - r_bore * 0.7, c - r_bore * 0.7, c + r_bore * 0.7, c + r_bore * 0.7], fill=8)
    tmp = path + f'.{os.getpid()}.tmp.png'
    img.convert('RGB').save(tmp)
    os.replace(tmp, path)
    return path


def _reticle(folder, name, ink, paper, hole=False, rings_mm=(1, 2, 3, 4, 5), size_mm=30.0, px=600):
    """Crosshair with rings every mm (the cap's or the target's face), labelled rings; RGBA png."""
    import os
    from PIL import Image, ImageDraw, ImageFont
    path = os.path.join(folder, f'reticle_{name}_v1.png')
    if os.path.exists(path):
        return path
    s = px / size_mm
    img = Image.new('RGBA', (px, px), (*paper, 255))
    d = ImageDraw.Draw(img)
    c = px / 2
    lw = max(1, int(0.12 * s))
    for r in rings_mm:
        d.ellipse([c - r * s, c - r * s, c + r * s, c + r * s], outline=(*ink, 255), width=lw)
    d.line([(c, c - 13 * s), (c, c + 13 * s)], fill=(*ink, 255), width=lw)
    d.line([(c - 13 * s, c), (c + 13 * s, c)], fill=(*ink, 255), width=lw)
    for k in range(-12, 13):
        t = (0.8 if k % 5 == 0 else 0.4) * s
        d.line([(c + k * s, c - t), (c + k * s, c + t)], fill=(*ink, 255), width=lw)
        d.line([(c - t, c + k * s), (c + t, c + k * s)], fill=(*ink, 255), width=lw)
    font = ImageFont.load_default(size=int(1.4 * s))
    for r in (2, 4):
        d.text((c + r * s * 0.72 + 0.3 * s, c - r * s * 0.72 - 0.3 * s), str(r), fill=(*ink, 255), font=font,
               anchor='lb')
    if hole:
        d.ellipse([c - HOLE_R * 1000 * s, c - HOLE_R * 1000 * s, c + HOLE_R * 1000 * s, c + HOLE_R * 1000 * s],
                  fill=(10, 10, 10, 255))
    tmp = path + f'.{os.getpid()}.tmp.png'
    img.save(tmp)
    os.replace(tmp, path)
    return path


# --- materials -----------------------------------------------------------------------------------------------------------
def _materials(stage, folder):
    UsdGeom.Scope.Define(stage, LOOKS)
    anod = V.grain(folder, 'anodized', 0.10, 0.40, 0.08, seed=7, blur=1)
    brushed = V.brushed(folder)
    peel = V.orange_peel(folder)

    def mat(name, colour, **kw):
        return V.material(stage, f'{LOOKS}/{name}', colour, **kw)

    board = mat('BreadboardTop', (0.05, 0.052, 0.056), metallic=0.4, roughness=0.45, diffuse=_board_texture(folder))
    return {
        'board': board,
        'anod': mat('BlackAnodized', (0.055, 0.057, 0.06), metallic=0.35, diffuse=anod[0], rough_map=anod[1]),
        'steel': mat('Stainless', (0.80, 0.80, 0.82), metallic=1.0, diffuse=brushed[0], rough_map=brushed[1]),
        'chrome': mat('Chrome', (0.92, 0.92, 0.94), metallic=1.0, roughness=0.08),
        'mirror': mat('ProtectedSilver', (0.97, 0.97, 0.98), metallic=1.0, roughness=0.02),
        'glass': mat('LensGlass', (0.80, 0.88, 0.95), metallic=0.0, roughness=0.03, clearcoat=1.0),
        'white': mat('WhitePlastic', (0.93, 0.93, 0.91), roughness=0.35),
        'grey_case': mat('CaseGrey', (0.70, 0.71, 0.72), roughness=0.5, normal=peel),
        'dark_case': mat('CaseDark', (0.16, 0.17, 0.18), roughness=0.5, normal=peel),
        'laser': mat('LaserHead', (0.12, 0.13, 0.16), metallic=0.5, roughness=0.35, clearcoat=0.4),
        'yellow': mat('FiberJacket', (0.95, 0.75, 0.05), roughness=0.45),
        'cable': mat('PowerCable', (0.03, 0.03, 0.03), roughness=0.6),
        'brass': mat('Brass', (0.86, 0.66, 0.33), metallic=1.0, roughness=0.25),
        'box': mat('LensBoxPlastic', (0.22, 0.23, 0.25), roughness=0.55),
        'seat': mat('SeatBlack', (0.04, 0.04, 0.045), roughness=0.5),
        'lcd': V.material(stage, f'{LOOKS}/LcdGlass', (0.05, 0.07, 0.06), roughness=0.15),
        'lit': V.material(stage, f'{LOOKS}/LcdSegment', (0.35, 1.0, 0.45), roughness=0.9, emission=0.55),
        'ink': mat('Ink', (0.02, 0.02, 0.02), roughness=0.8),
        'label': mat('LabelYellow', (0.98, 0.80, 0.05), roughness=0.5),
    }


# --- breadboard ------------------------------------------------------------------------------------------------------------
def _breadboard(stage, m, physics):
    (x0, x1), (y0, y1) = BOARD['x'], BOARD['y']
    size = (x1 - x0, y1 - y0, BOARD['h'])
    centre = ((x0 + x1) / 2, (y0 + y1) / 2, BOARD['h'] / 2)
    _coll_box(stage, ROOT + '/Breadboard', size, centre, physics)
    geo = V.rounded_box(size, 0.0015)
    # the hole grid: a 25 mm tile on the top face, a hole in the middle of each cell, holes 12.5 mm from the edges
    V.define(stage, ROOT + '/BreadboardLook', geo, m['board'], tile=BOARD['pitch'], translate=centre)
    prim = stage.GetPrimAtPath(TABLE_TOP)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)


# --- mirror mounts ---------------------------------------------------------------------------------------------------------
def _mount(stage, mirror, m, physics):
    """Static parts of a mount (back plate, post, post holder, base, screws' and pivot's visible parts) and its
    kinematic front plate with the mirror."""
    R = mount_rotation(mirror)
    yaw = MIRRORS[mirror]['yaw']
    base = f'{ROOT}/{mirror}'
    UsdGeom.Xform.Define(stage, base)

    def world(local):
        return mount_point(mirror, local)

    rot = (0.0, 0.0, yaw)
    bx = BACK_X[1] - BACK_X[0]
    solid_box(stage, base + '/BackPlate', (bx, PLATE, PLATE), world(((BACK_X[0] + BACK_X[1]) / 2, 0, 0)), m['anod'],
              physics, radius=0.0012, rotate_xyz=rot)
    # the bracket's foot under the back plate, the post, the post holder and its base on the breadboard
    foot_top = BEAM_Z - PLATE / 2
    post_xy = world((0.019, 0.0, 0.0))[:2]
    solid_cyl(stage, base + '/Post', POST_R, foot_top - (HOLDER_TOP - 0.004), (post_xy[0], post_xy[1],
              (foot_top + HOLDER_TOP - 0.004) / 2), m['steel'], physics)
    solid_cyl(stage, base + '/PostHolder', HOLDER_R, HOLDER_TOP - BOARD_TOP - 0.008,
              (post_xy[0], post_xy[1], BOARD_TOP + 0.008 + (HOLDER_TOP - BOARD_TOP - 0.008) / 2), m['anod'], physics,
              chamfer=0.001)
    solid_cyl(stage, base + '/HolderBase', 0.016, 0.008, (post_xy[0], post_xy[1], BOARD_TOP + 0.004), m['anod'],
              physics, chamfer=0.001)
    # clamping fork across the base, two cap screws into the breadboard
    fork_c = (post_xy[0], post_xy[1], BOARD_TOP + 0.003)
    solid_box(stage, base + '/Fork', (0.060, 0.026, 0.006), fork_c, m['anod'], physics, radius=0.002,
              rotate_xyz=(0.0, 0.0, yaw + 90.0))
    a = math.radians(yaw + 90.0)
    for k, s in enumerate((-1, 1)):
        p = (fork_c[0] + s * 0.024 * math.cos(a), fork_c[1] + s * 0.024 * math.sin(a))
        V.define(stage, f'{base}/ForkScrew{k}', V.cap_screw(0.0045, 0.004), m['steel'], tile=0.01,
                 translate=(p[0], p[1], BOARD_TOP + 0.006))
    # thumbscrew of the post holder (render: inside the clearance round the holder)
    tx = world((0.019 + HOLDER_R + 0.003, 0.0, 0.0))
    V.define(stage, base + '/HolderScrew', _cyl_geo(0.004, 0.006, 0.0008, 24), m['anod'],
             translate=(tx[0], tx[1], HOLDER_TOP - 0.012), rotate_xyz=(0.0, 90.0, yaw))
    # pivot ball and the screws in the gap (inside the plates' outline: render only), springs
    for name in ('pivot', 'H', 'V'):
        p = world(corner_local(mirror, name, (FRONT_X[1] + BACK_X[0]) / 2))
        if name == 'pivot':
            V.define(stage, f'{base}/PivotBall', V.lathe([(0.0, -0.002), (0.0018, -0.0012), (0.002, 0.0),
                                                          (0.0018, 0.0012), (0.0, 0.002)], 24), m['chrome'],
                     translate=tuple(p))
        else:
            V.define(stage, f'{base}/Screw{name}', _cyl_geo(0.0016, BACK_X[0] - FRONT_X[1] + 0.002), m['steel'],
                     translate=tuple(p), rotate_xyz=(0.0, 90.0, yaw))
    for k, (su, sv) in enumerate(((0, 1), (1, 0))):
        loc = np.array([(FRONT_X[1] + BACK_X[0]) / 2, 0.006 * su * (1 if MIRRORS[mirror]['sigma'] > 0 else -1),
                        0.006 * sv])
        p = world(loc)
        V.define(stage, f'{base}/Spring{k}', V.torus(0.0018, 0.0004, 16, 6), m['steel'], translate=tuple(p),
                 rotate_xyz=(0.0, 90.0, yaw))
    # engraved marks on the back plate: H / V beside the knobs (render only, a decal on the back face)
    _front_plate(stage, mirror, m, physics)


def _front_plate(stage, mirror, m, physics):
    """Kinematic body: origin at the pivot (mount frame (PIVOT_LOCAL_X, pivot y, pivot z)), the mount's rotation."""
    path = front_path(mirror)
    xform = UsdGeom.Xform.Define(stage, path)
    pivot = corner_local(mirror, 'pivot', PIVOT_LOCAL_X)
    xform.AddTranslateOp().Set(Gf.Vec3d(*mount_point(mirror, pivot).tolist()))
    xform.AddOrientOp().Set(KN._quat(mount_rotation(mirror)))
    fx = FRONT_X[1] - FRONT_X[0]
    c = np.array([(FRONT_X[0] + FRONT_X[1]) / 2, 0.0, 0.0]) - pivot
    _coll_box(stage, path + '/Plate', (fx + FRONT_X[0], PLATE, PLATE), c - np.array([FRONT_X[0] / 2, 0, 0]), physics)
    _rbox(stage, path + '/PlateLook', (fx, PLATE, PLATE), c, m['anod'], 0.0012)
    # the mirror in its bore, 1 mm proud of the plate
    mc = np.array([MIRROR_T / 2, 0.0, 0.0]) - pivot
    V.define(stage, path + '/MirrorLook', _cyl_geo(MIRROR_R, MIRROR_T, 0.0004, 72), m['mirror'],
             translate=tuple(mc.tolist()), rotate_xyz=(0.0, 90.0, 0.0))
    # retaining ring
    V.define(stage, path + '/RingLook', V.lathe([(MIRROR_R - 0.0004, 0.0), (MIRROR_R + 0.0025, 0.0),
                                                 (MIRROR_R + 0.0025, 0.0006), (MIRROR_R - 0.0004, 0.0006)], 72),
             m['anod'], translate=tuple((np.array([FRONT_X[0] - 0.0006, 0, 0]) - pivot).tolist()),
             rotate_xyz=(0.0, -90.0, 0.0))
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim()).CreateKinematicEnabledAttr(True)
    return path


def front_plate_pose(mirror, phi_x, phi_y):
    """(position (ROOT frame), rotation 3x3) of a front plate tilted by the model's mirror tilts (rad)."""
    s = MIRRORS[mirror]['sigma']
    az, ay = s * phi_x, phi_y
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    R = mount_rotation(mirror) @ Rz @ Ry
    return mount_point(mirror, corner_local(mirror, 'pivot', PIVOT_LOCAL_X)), R


def set_front_plate(stage, mirror, phi_x, phi_y, body=None, z0=0.0):
    p, R = front_plate_pose(mirror, phi_x, phi_y)
    prim = stage.GetPrimAtPath(front_path(mirror))
    ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
    ops[0].Set(Gf.Vec3d(*p.tolist()))
    q = KN._quat(R)
    ops[1].Set(q)
    if body is not None:
        im = q.GetImaginary()
        body.set_world_pose(position=np.array([p[0], p[1], p[2] + z0]),
                            orientation=np.array([q.GetReal(), im[0], im[1], im[2]]))


# --- laser and power supply -----------------------------------------------------------------------------------------------
def _laser(stage, m, physics, materials):
    x, ye = LASER_X, LASER_EXIT_Y
    r, length = 0.0222, 0.300
    y0 = ye - length
    _coll_cyl(stage, ROOT + '/LaserHead', r, length, (x, (y0 + ye) / 2, BEAM_Z), physics, axis='Y')
    body = V.lathe([(0.0, 0.0), (0.020, 0.0), (r, 0.002), (r, length - 0.016), (0.019, length - 0.014),
                    (0.019, length - 0.004), (0.017, length - 0.002), (0.0045, length - 0.002), (0.0045, length - 0.0008),
                    (0.0, length - 0.0008)], 72)
    V.define(stage, ROOT + '/LaserHeadLook', body, m['laser'], tile=0.05, translate=(x, y0, BEAM_Z),
             rotate_xyz=(-90.0, 0.0, 0.0))
    # exit aperture (glows red when on)
    _emissive(stage, EXIT_MAT, RED, 0.0)
    V.define(stage, ROOT + '/LaserExit', V.lathe([(0.0, 0.0), (0.0016, 0.0), (0.0016, 0.0003), (0.0, 0.0003)], 32),
             UsdShade.Material(stage.GetPrimAtPath(EXIT_MAT)), translate=(x, ye - 0.0011, BEAM_Z),
             rotate_xyz=(-90.0, 0.0, 0.0))
    # warning label on the tube's side toward +x (a decal reading along the tube)
    folder = materials['folder']
    lab, aspect = V.decal(folder, 'laser_label_v1', ['CAUTION  LASER RADIATION', 'HeNe 632.8 nm  1 mW  CLASS 2'],
                          px_h=96, ink=(15, 15, 15), background=(250, 205, 20, 255), bold=True)
    _decal(stage, ROOT + '/LaserLabel', lab, (0.014 * aspect, 0.014), (x + r + 0.0012, y0 + 0.15, BEAM_Z), '+x')
    # two ring mounts on posts
    for k, yc in enumerate((ye - 0.075, ye - 0.235)):
        base = f'{ROOT}/LaserRing{k}'
        ring_t = 0.010
        V.define(stage, base + 'Look', V.lathe([(r + 0.0002, -ring_t / 2), (r + 0.006, -ring_t / 2),
                                                (r + 0.006, ring_t / 2), (r + 0.0002, ring_t / 2)], 64, closed=True),
                 m['anod'], translate=(x, yc, BEAM_Z), rotate_xyz=(-90.0, 0.0, 0.0))
        for name, size, centre in (('Top', (2 * r + 0.012, ring_t, 0.0058), (x, yc, BEAM_Z + r + 0.0031)),
                                   ('Left', (0.0058, ring_t, 2 * r), (x - r - 0.0031, yc, BEAM_Z)),
                                   ('Right', (0.0058, ring_t, 2 * r), (x + r + 0.0031, yc, BEAM_Z)),
                                   ('Bottom', (2 * r + 0.012, ring_t, 0.0058), (x, yc, BEAM_Z - r - 0.0031))):
            _coll_box(stage, f'{base}{name}', size, centre, physics)
        ring_bottom = BEAM_Z - r - 0.006
        solid_cyl(stage, base + 'Post', POST_R, ring_bottom - (HOLDER_TOP - 0.004),
                  (x, yc, (ring_bottom + HOLDER_TOP - 0.004) / 2), m['steel'], physics)
        solid_cyl(stage, base + 'Holder', HOLDER_R, HOLDER_TOP - BOARD_TOP - 0.008,
                  (x, yc, BOARD_TOP + 0.008 + (HOLDER_TOP - BOARD_TOP - 0.008) / 2), m['anod'], physics, chamfer=0.001)
        solid_cyl(stage, base + 'Base', 0.016, 0.008, (x, yc, BOARD_TOP + 0.004), m['anod'], physics, chamfer=0.001)
    # power supply with POWER push button and pilot lamp
    (px, py), (sx, sy, sz) = PSU['centre'], PSU['size']
    solid_box(stage, ROOT + '/PSU', (sx, sy, sz), (px, py, BOARD_TOP + sz / 2), m['dark_case'], physics, radius=0.003)
    top = BOARD_TOP + sz
    bxy = (px - 0.030, py + 0.018)
    button = _button(stage, POWER_ID, bxy, top, 'round', (0.80, 0.10, 0.08), '', materials['button'])
    _text(stage, ROOT + '/PowerLabel', 'POWER', (bxy[0], bxy[1] + 0.022), top + 0.0004, 0.007, C.PAPER)
    _emissive(stage, LAMP_MAT, LAMP_OFF, 0.0)
    lxy = (px + 0.030, py + 0.018)
    V.define(stage, ROOT + '/PilotLamp', V.lathe([(0.0, 0.0), (0.0045, 0.0), (0.0045, 0.002), (0.0035, 0.0035),
                                                  (0.0, 0.004)], 32),
             UsdShade.Material(stage.GetPrimAtPath(LAMP_MAT)), translate=(lxy[0], lxy[1], top))
    _text(stage, ROOT + '/LampLabel', 'EMISSION', (lxy[0], lxy[1] + 0.022), top + 0.0004, 0.006, C.PAPER)
    _text(stage, ROOT + '/PsuLabel', 'HeNe LASER POWER SUPPLY', (px, py - 0.030), top + 0.0004, 0.0065, C.PAPER)
    # power cable from the supply's back to the laser's tail
    cable = [(px + 0.02, py - sy / 2, BOARD_TOP + 0.025), (px + 0.03, py - sy / 2 - 0.03, BOARD_TOP + 0.006),
             (px + 0.10, y0 - 0.02, BOARD_TOP + 0.005), (x, y0 - 0.035, BOARD_TOP + 0.010),
             (x, y0 - 0.025, BEAM_Z - 0.015), (x, y0 - 0.004, BEAM_Z)]
    V.define(stage, ROOT + '/PowerCable', V.tube(_smooth_path(cable), 0.0032, 10), m['cable'])
    return {POWER_ID: button}


def _smooth_path(points, n=8):
    """Catmull-Rom through the points."""
    P = np.asarray(points, float)
    out = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, len(P) - 1)]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1])
    return out


def set_power(stage, on):
    set_emission(stage, LAMP_MAT, LAMP_ON if on else LAMP_OFF, 4.0 if on else 0.0)
    set_emission(stage, EXIT_MAT, RED, 300.0 if on else 0.0)


# --- target, coupler and focus stage ------------------------------------------------------------------------------------------
def _target_stand(stage, m, physics, seat_colour, seat_physics):
    cy = TARGET_SEAT_Y
    ox, oy = SLOT_WELL[0] + 0.006, SLOT_WELL[1] + 0.006
    span_box(stage, ROOT + '/TargetStand', (BEAM_X - ox, BEAM_X + ox), (cy - oy, cy + oy), (BOARD_TOP + 0.006,
             SEAT_BASE_Z), m['anod'], physics)
    span_box(stage, ROOT + '/TargetStandFoot', (BEAM_X - 0.020, BEAM_X + 0.020), (cy - 0.012, cy + 0.012),
             (BOARD_TOP, BOARD_TOP + 0.006), m['anod'], physics, radius=0.0015)
    return {'bench:target': _rect_collar(stage, seat_path('bench:target'), (BEAM_X, cy), SEAT_BASE_Z, SLOT_WELL,
                                         seat_colour, seat_physics)}


def _coupler(stage, m, physics, seat_colour, seat_physics):
    seats = {}
    lo = LENS_SEAT_Y - (CART_WELL[1] + 0.006)
    hi = CAP_SEAT_Y + SLOT_WELL[1] + 0.006
    ox = max(CART_WELL[0], SLOT_WELL[0]) + 0.006
    span_box(stage, ROOT + '/CouplerBody', (BEAM_X - ox, BEAM_X + ox), (lo, hi), (BOARD_TOP, SEAT_BASE_Z), m['anod'],
             physics)
    seats['coupler:lens'] = _rect_collar(stage, seat_path('coupler:lens'), (BEAM_X, LENS_SEAT_Y), SEAT_BASE_Z,
                                         CART_WELL, seat_colour, seat_physics)
    seats['coupler:cap'] = _rect_collar(stage, seat_path('coupler:cap'), (BEAM_X, CAP_SEAT_Y), SEAT_BASE_Z, SLOT_WELL,
                                        seat_colour, seat_physics)
    # focus stage: base and lead screw housing (static), carriage with the fiber port (kinematic)
    (x0, x1), (y0, y1) = STAGE['x'], STAGE['y']
    stage_top = BEAM_Z - PORT['size'][2] / 2 - 0.010
    span_box(stage, ROOT + '/StageBase', (x0, x1), (y0, y1), (BOARD_TOP, stage_top - 0.008), m['anod'], physics,
             radius=0.0015)
    kx = FOCUS_KNOB['x']
    solid_cyl(stage, ROOT + '/LeadScrewHousing', 0.009, y1 - y0, (kx, (y0 + y1) / 2, FOCUS_KNOB['z']), m['anod'],
              physics, axis='Y', chamfer=0.001)
    span_box(stage, ROOT + '/LeadScrewWeb', (kx - 0.006, kx + 0.006), (y0, y1), (stage_top - 0.008, FOCUS_KNOB['z']),
             m['anod'], physics, radius=0.0005)
    carriage = UsdGeom.Xform.Define(stage, CARRIAGE_PATH)
    carriage.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    span = (x0 + 0.002, kx - 0.010)
    cz = (stage_top - 0.008, stage_top)
    cy = (y0 + 0.004, FIBER_Y - PORT['size'][1] - 0.001)
    span_box(stage, CARRIAGE_PATH + '/Plate', span, cy, cz, m['anod'], physics, radius=0.001)
    px, py, pz = PORT['size']
    span_box(stage, CARRIAGE_PATH + '/PortBracket', (BEAM_X - 0.012, BEAM_X + 0.012), (FIBER_Y - py - 0.001, FIBER_Y - py + 0.007),
             (stage_top, BEAM_Z - pz / 2), m['anod'], physics, radius=0.0008)
    span_box(stage, CARRIAGE_PATH + '/Port', (BEAM_X - px / 2, BEAM_X + px / 2), (FIBER_Y - py, FIBER_Y),
             (BEAM_Z - pz / 2, BEAM_Z + pz / 2), m['anod'], physics, radius=0.0015)
    # FC/PC connector on the port's back (render), the ferrule face at the port's front (render)
    solid_cyl(stage, CARRIAGE_PATH + '/FCNut', 0.0045, 0.008, (BEAM_X, FIBER_Y - py - 0.004, BEAM_Z), m['brass'],
              physics, axis='Y', chamfer=0.0005)
    solid_cyl(stage, CARRIAGE_PATH + '/Boot', 0.0025, 0.020, (BEAM_X, FIBER_Y - py - 0.018, BEAM_Z), m['yellow'],
              physics, axis='Y', chamfer=0.0008)
    V.define(stage, CARRIAGE_PATH + '/Ferrule', _cyl_geo(0.00125, 0.0008, 0.0002, 24), m['white'],
             translate=(BEAM_X, FIBER_Y + 0.0001, BEAM_Z), rotate_xyz=(-90.0, 0.0, 0.0))
    UsdPhysics.RigidBodyAPI.Apply(carriage.GetPrim()).CreateKinematicEnabledAttr(True)
    return seats


def set_carriage(stage, travel_m, body=None, z0=0.0):
    """The focus carriage moved travel_m away from the lens (world -y) from its start."""
    prim = stage.GetPrimAtPath(CARRIAGE_PATH)
    UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(Gf.Vec3d(0.0, -float(travel_m), 0.0))
    if body is not None:
        body.set_world_pose(position=np.array([0.0, -float(travel_m), z0]), orientation=np.array([1.0, 0.0, 0.0, 0.0]))


# --- holders -----------------------------------------------------------------------------------------------------------------
def _handle(stage, path, z, m, physics):
    stem_h = 0.018
    solid_cyl(stage, path + '/HandleStem', C.FUNNEL_POST_R, stem_h, (0, 0, z + stem_h / 2), m['white'], physics)
    rim_z = z + stem_h
    solid_cyl(stage, path + '/HandleRim', C.FUNNEL_RIM_R, C.FUNNEL_RIM_H, (0, 0, rim_z + C.FUNNEL_RIM_H / 2),
              m['white'], physics, chamfer=0.0008)
    solid_cyl(stage, path + '/HandlePort', C.PORT_R, C.PORT_H, (0, 0, rim_z + C.FUNNEL_RIM_H + C.PORT_H / 2),
              m['white'], physics, collide=False)
    return rim_z + C.FUNNEL_RIM_H


def _body(stage, path, xy, bottom_z, mass, com):
    xform = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(xy[0], xy[1], bottom_z))
    return xform


def _finish_body(xform, mass, com):
    _rigid(xform.GetPrim(), mass, com, ccd=True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(xform.GetPrim()).CreateMaxDepenetrationVelocityAttr(0.1)


def _cartridge(stage, oid, k, xy, bottom_z, m, physics):
    path = holder_path(oid)
    xform = _body(stage, path, xy, bottom_z, 0.05, (0, 0, 0.02))
    bx, by, bz = CART_BASE
    solid_box(stage, path + '/Base', CART_BASE, (0, 0, bz / 2), m['anod'], physics, radius=0.0012)
    d = lens_disc_y(k) - LENS_SEAT_Y                   # the lens disc from the base centre (+ upstream)
    post_r = 0.0035
    post_top = AXIS_LOCAL - RING_R
    solid_cyl(stage, path + '/Post', post_r, post_top - bz, (0, d, (bz + post_top) / 2), m['anod'], physics)
    solid_cyl(stage, path + '/Cell', RING_R, RING_T, (0, d, AXIS_LOCAL), m['anod'], physics, axis='Y',
              chamfer=0.0006, segments=64)
    for s in (1, -1):
        V.define(stage, f'{path}/Lens{"Front" if s > 0 else "Back"}', V.lathe(
            [(0.0, 0.0), (LENS_CA_R, 0.0), (LENS_CA_R, 0.0002), (0.0, 0.0007)], 48), m['glass'],
            translate=(0, d + s * RING_T / 2, AXIS_LOCAL), rotate_xyz=(-90.0 * s, 0.0, 0.0))
    top = _handle(stage, path, AXIS_LOCAL + RING_R, m, physics)
    _text(stage, path + '/Name', LENS_TEXT[k], (0.0, 0.0090), top + 0.0004, 0.0055)
    # the focal length engraved on the base's front (toward +y), read from the robot side
    folder = m['folder']
    img, aspect = V.decal(folder, f'cart_{k}_v1', [f'f {LENS_TEXT[k]}'], px_h=64, ink=(240, 240, 235), bold=True)
    _decal(stage, path + '/Engraving', img, (0.0062 * aspect, 0.0062), (0, by / 2 + 0.0002, bz * 0.55), '+y',
           mat_name=f'engraving_{k}')        # one material per cartridge (a shared one showed the last focal length on all)
    _finish_body(xform, 0.05, (0, 0, 0.022))
    return path


def _plate_holder(stage, oid, kind, xy, bottom_z, m, physics):
    path = holder_path(oid)
    xform = _body(stage, path, xy, bottom_z, 0.03, (0, 0, 0.04))
    tx, ty, tz = TONGUE
    solid_box(stage, path + '/Tongue', TONGUE, (0, 0, tz / 2), m['anod'], physics, radius=0.0006)
    px, py, pz = PLATE_SIZE
    plate_bottom = AXIS_LOCAL - pz / 2
    solid_cyl(stage, path + '/Stem', 0.003, plate_bottom - tz, (0, 0, (tz + plate_bottom) / 2), m['anod'], physics)
    face_mat = m['anod'] if kind == 'cap' else m['white']
    solid_box(stage, path + '/Plate', PLATE_SIZE, (0, 0, AXIS_LOCAL), face_mat, physics, radius=0.0006)
    folder = m['folder']
    if kind == 'cap':
        img = _reticle(folder, 'cap', ink=(235, 235, 230), paper=(14, 14, 16))
    else:
        img = _reticle(folder, 'target', ink=(15, 15, 15), paper=(238, 238, 228), hole=True)
    _decal(stage, path + '/Reticle', img, (px - 0.001, pz - 0.001), (0, py / 2 + 0.0002, AXIS_LOCAL), '+y',
           mat_name=f'reticle_{kind}')
    top = _handle(stage, path, AXIS_LOCAL + pz / 2, m, physics)
    _text(stage, path + '/Name', 'CAP' if kind == 'cap' else 'TARGET', (0.0, 0.0090), top + 0.0004, 0.0045)
    _finish_body(xform, 0.03, (0, 0, 0.045))
    return path


# --- lens box ----------------------------------------------------------------------------------------------------------------
def box_wells():
    """{seat id: (x, y)} of the lens box's wells (five cartridges, then the cap's and the target's slots)."""
    (cx, cy), (bx, by, bz) = LENS_BOX['centre'], LENS_BOX['base']
    out = {}
    x = cx - bx / 2 + 0.006 + CART_WELL[0] + 0.006
    for k in range(len(LENS_IDS)):
        out[f'box:{k + 1}'] = (x, cy)
        x += LENS_BOX['lens_pitch']
    x += (SLOT_WELL[0] + 0.006) - (CART_WELL[0] + 0.006) + 0.004
    for name in ('cap', 'target'):
        out[f'box:{name}'] = (x, cy)
        x += LENS_BOX['slot_pitch']
    return out


def _lens_box(stage, m, physics, seat_colour, seat_physics):
    (cx, cy), (bx, by, bz) = LENS_BOX['centre'], LENS_BOX['base']
    solid_box(stage, ROOT + '/LensBox', (bx, by, bz), (cx, cy, bz / 2), m['box'], physics, radius=0.003)
    seats = {}
    for sid, (x, y) in box_wells().items():
        well = SLOT_WELL if sid in ('box:cap', 'box:target') else CART_WELL
        seats[sid] = _rect_collar(stage, seat_path(sid), (x, y), bz, well, seat_colour, seat_physics)
        text = LENS_TEXT[int(sid.split(':')[1]) - 1] if sid[4:].isdigit() else sid[4:].upper()
        _text(stage, f"{ROOT}/LensBoxLabel_{sid[4:]}", text, (x, cy + by / 2 - 0.0045), bz + 0.0004, 0.0048, C.PAPER)
    return seats


# --- power meter -------------------------------------------------------------------------------------------------------------
SEVEN = C.SEGMENTS
METER_PANEL = ROOT + '/MeterPanel'


def _unit_decal(folder, name, unit, px_h=96, ink=(90, 255, 115)):
    """A unit legend decal. The kit's default font has no micro sign: a µ is drawn as a u with its left stem carried
    down below the baseline. Returns (path, aspect) as visual_kit.decal."""
    import os
    from PIL import Image, ImageDraw, ImageFont
    path = os.path.join(folder, name + '.png')
    micro = unit.startswith('µ')
    text = ('u' + unit[1:]) if micro else unit
    pad = 0.12
    font = ImageFont.load_default(size=int(px_h * (1 - 2 * pad) * 0.92))
    probe = ImageDraw.Draw(Image.new('L', (8, 8)))
    w = int(probe.textlength(text, font=font) + 2 * pad * px_h) + 1
    if not os.path.exists(path):
        img = Image.new('RGBA', (w, px_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((w / 2, px_h / 2), text, font=font, fill=(*ink, 255), anchor='mm')
        if micro:
            x0, y0, x1, y1 = draw.textbbox((w / 2, px_h / 2), 'u', font=font, anchor='mm')
            a = np.asarray(img)[..., 3]
            cols = np.where(a[int(y0):int(y1)].max(axis=0) > 128)[0]
            left = int(cols.min())
            stem = max(3, int(0.10 * px_h))
            draw.rectangle([left, (y0 + y1) / 2, left + stem - 1, min(px_h - 2, y1 + 0.22 * px_h)], fill=(*ink, 255))
        tmp = path + f'.{os.getpid()}.tmp.png'
        img.save(tmp)
        os.replace(tmp, path)
    return path, w / px_h


def meter_frame():
    """(centre of the panel's face (ROOT frame), tilt (deg about +x)) of the meter's display panel."""
    (mx, my), (bx, by, bz) = METER['centre'], METER['body']
    px, py, pz = METER['panel']
    t = math.radians(METER['tilt_deg'])
    # the panel's lower front edge on the body's top, tilted back (its normal toward +y and up)
    edge = np.array([mx, my + by / 2 - 0.012, BOARD_TOP + bz])
    up_slope = np.array([0.0, -math.cos(t), math.sin(t)])
    normal = np.array([0.0, math.sin(t), math.cos(t)])
    centre = edge + up_slope * (py / 2) + normal * 0.022
    return centre, METER['tilt_deg']


def cell_rects(cells, digit_h, centre):
    """{(cell, seg): ((u, v), (du, dv))} of a seven-segment row read from the robot side (reading toward -u, digit
    tops toward -v), centred on `centre`, and the decimal points' ((u, v), size) after each cell."""
    w, t = digit_h * 0.52, digit_h * 0.14
    pitch = digit_h * 0.80
    width = cells * pitch - (pitch - w)
    out, points = {}, {}
    for i in range(cells):
        u0 = -width / 2 + i * pitch
        for seg, (fu, fv, horizontal) in SEVEN.items():
            u, v = u0 + fu * w, -digit_h / 2 + fv * digit_h
            out[(i, seg)] = ((centre[0] - u, centre[1] - v), (w, t) if horizontal else (t, digit_h / 2))
        u = u0 + w + (pitch - w) / 2
        points[i] = ((centre[0] - u, centre[1] + digit_h / 2 - t / 2), t * 1.3)
    return out, points


def _meter(stage, m, physics):
    (mx, my), (bx, by, bz) = METER['centre'], METER['body']
    solid_box(stage, ROOT + '/MeterBody', (bx, by, bz), (mx, my, BOARD_TOP + bz / 2), m['grey_case'], physics,
              radius=0.004)
    centre, tilt = meter_frame()
    px, py, pz = METER['panel']
    pc = centre - np.array([0.0, math.sin(math.radians(tilt)), math.cos(math.radians(tilt))]) * (pz / 2)
    solid_box(stage, ROOT + '/MeterPanelBox', (px, py, pz), tuple(pc.tolist()), m['grey_case'], physics,
              radius=0.003, rotate_xyz=(-tilt, 0.0, 0.0))
    # feet
    for k, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        V.define(stage, f'{ROOT}/MeterFoot{k}', _cyl_geo(0.006, 0.003, 0.0008, 24), m['seat'],
                 translate=(mx + sx * (bx / 2 - 0.015), my + sy * (by / 2 - 0.015), BOARD_TOP + 0.0015))
    panel = UsdGeom.Xform.Define(stage, METER_PANEL)
    api = UsdGeom.XformCommonAPI(panel)
    api.SetTranslate(Gf.Vec3d(*centre.tolist()))
    api.SetRotate(Gf.Vec3f(-tilt, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    # bezel, glass, digits and annunciators in the panel frame (z = 0 the panel face, read from the robot side)
    gw, gh = 0.110, 0.042
    gc = (0.008, -0.004)
    V.define(stage, METER_PANEL + '/Bezel', V.frame_ring((gw + 0.010, gh + 0.010), (gw + 0.001, gh + 0.001), 0.0,
                                                         0.0012), m['dark_case'], translate=(gc[0], gc[1], 0.0))
    _box(stage, METER_PANEL + '/Glass', (gw, gh, 0.0006), (gc[0], gc[1], 0.0003), (0.05, 0.07, 0.06), collide=False)
    V.bind(stage.GetPrimAtPath(METER_PANEL + '/Glass'), m['lcd'])
    z = 0.0008
    dc = (gc[0] + 0.012, gc[1])
    rects, points = cell_rects(METER_CELLS, METER_DIGIT_H, dc)
    for (i, seg), ((u, v), (du, dv)) in rects.items():
        prim = _box(stage, f'{METER_PANEL}/Digit{i}{seg}', (du, dv, 0.0006), (u, v, z), (0.35, 1.0, 0.45),
                    collide=False)
        V.bind(prim.GetPrim(), m['lit'])
        _remember_lit(prim.GetPrim())
    for i, ((u, v), s) in points.items():
        if i == METER_CELLS - 1:
            continue
        prim = _box(stage, f'{METER_PANEL}/Point{i}', (s, s, 0.0006), (u, v, z), (0.35, 1.0, 0.45), collide=False)
        V.bind(prim.GetPrim(), m['lit'])
        _remember_lit(prim.GetPrim())
    folder = m['folder']
    for k, unit in enumerate(('µW', 'nW')):
        img, aspect = _unit_decal(folder, f'meter_unit_{k}_v2', unit, px_h=96, ink=(90, 255, 115))
        h = 0.010
        mesh = _decal(stage, f'{METER_PANEL}/Unit{k}', img, (h * aspect, h),
                      (gc[0] - gw / 2 + 0.014, gc[1] - 0.009 + k * 0.016, z), '+z', emission=0.55,
                      colour=(0.35, 1.0, 0.45))
        _remember_lit(mesh.GetPrim())
    for name, text, xy, h in (('Brand', 'OPTICAL POWER METER', (0.0, -0.031), 0.0055),
                              ('Lambda', '633 nm  Si', (-0.052, 0.024), 0.0042),
                              ('Range', 'AUTO RANGE', (0.040, 0.024), 0.0042)):
        _text(stage, f'{METER_PANEL}/{name}', text, xy, 0.0004, h, C.PAPER)
    # the fiber input on the body's front, the patch cord's end
    fx, fy, fz = mx + 0.050, my + by / 2, BOARD_TOP + bz / 2
    solid_cyl(stage, ROOT + '/MeterInput', 0.0055, 0.008, (fx, fy + 0.004, fz), m['brass'], physics, axis='Y')
    img, aspect = V.decal(folder, 'meter_input_v1', ['FIBER IN'], px_h=64, ink=(20, 20, 20))
    _decal(stage, ROOT + '/MeterInputLabel', img, (0.0055 * aspect, 0.0055), (fx, fy + 0.0003, fz + 0.012), '+y')
    return (fx, fy + 0.008, fz)


def meter_text(reading_uW):
    """(three digits, the decimal point after cell 0 or 1 or None, unit) as the meter shows a reading: auto range,
    three significant digits, microwatts down to 1 uW, then nanowatts; below 1 nW it shows 0.00 nW."""
    p, unit = float(reading_uW), 'uW'
    if p < 1.0:
        p, unit = p * 1000.0, 'nW'
    v = float(f'{p:.3g}') if p > 0 else 0.0
    if unit == 'nW' and v >= 1000.0:
        v, unit = v / 1000.0, 'uW'
    if unit == 'nW' and v < 1.0:
        return ('000', 0, 'nW')
    v = min(v, 999.0)
    if v >= 100.0:
        return (f'{int(round(v)):3d}', None, unit)
    if v >= 10.0:
        return (f'{v:.1f}'.replace('.', ''), 1, unit)
    return (f'{v:.2f}'.replace('.', ''), 0, unit)


def show_meter(stage, shown, previous):
    """Light the meter's cells, point and unit for shown = meter_text(...) where they differ from previous."""
    if shown == previous:
        return shown
    digits, point, unit = shown
    old = previous or (None, -1, None)
    for i, ch in enumerate(digits):
        if old[0] is not None and old[0][i] == ch:
            continue
        on = C.GLYPHS.get(ch, '')
        for seg in SEVEN:
            C._light(stage, f'{METER_PANEL}/Digit{i}{seg}', seg in on)
    if old[1] != point:
        for i in range(METER_CELLS - 1):
            C._light(stage, f'{METER_PANEL}/Point{i}', point == i)
    if old[2] != unit:
        C._light(stage, f'{METER_PANEL}/Unit0', unit == 'uW')
        C._light(stage, f'{METER_PANEL}/Unit1', unit == 'nW')
    return shown


def meter_value(shown):
    """The reading as displayed (uW)."""
    digits, point, unit = shown
    v = float(digits if point is None else digits[:point + 1] + '.' + digits[point + 1:])
    return v if unit == 'uW' else v / 1000.0


# --- spots (render) --------------------------------------------------------------------------------------------------------
def _spots(stage):
    """One spot slot per surface the beam can land on: concentric emissive discs (SPOT_LAYERS) facing local +y, set
    a little in front of the surface, turned about z (rot_z) to face the beam on its leg."""
    for name in SPOT_NAMES:
        slot = UsdGeom.Xform.Define(stage, SPOT_PATH.format(name))
        slot.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        slot.AddRotateZOp().Set(0.0)
        slot.AddScaleOp().Set(Gf.Vec3f(0.0008, 1.0, 0.0008))
        for layer, (r, _) in enumerate(SPOT_LAYERS):
            mpath = f'{LOOKS}/Spot_{name}_{layer}'
            _emissive(stage, mpath, RED, 0.0)
            disc = V.lathe([(0.0, 0.0), (r, 0.0), (r, 1e-5), (0.0, 1e-5)], 40)
            V.define(stage, f'{SPOT_PATH.format(name)}/Disc{layer}', disc,
                     UsdShade.Material(stage.GetPrimAtPath(mpath)),
                     translate=(0.0, 0.0004 + 0.00005 * layer, 0.0), rotate_xyz=(-90.0, 0.0, 0.0))
        UsdGeom.Imageable(slot).MakeInvisible()


def set_spot(stage, name, spot):
    """spot: None (hidden) or {'xyz': point on the surface (ROOT frame), 'w': beam radius m, 'level': 0..1, 'rot_z':
    degrees about z (0: facing +y)}."""
    prim = stage.GetPrimAtPath(SPOT_PATH.format(name))
    if spot is None or spot['level'] <= 1e-4:
        UsdGeom.Imageable(prim).MakeInvisible()
        return
    ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
    ops[0].Set(Gf.Vec3d(*[float(v) for v in spot['xyz']]))
    ops[1].Set(float(spot.get('rot_z', 0.0)))
    ops[2].Set(Gf.Vec3f(float(spot['w']), 1.0, float(spot['w'])))
    for layer, (_, glow) in enumerate(SPOT_LAYERS):
        tint = tuple((1.0 - 0.5 * glow) * c + 0.5 * glow for c in RED) if layer == len(SPOT_LAYERS) - 1 else RED
        set_emission(stage, f'{LOOKS}/Spot_{name}_{layer}', tint, SPOT_GLOW * glow * float(spot['level']))
    UsdGeom.Imageable(prim).MakeVisible()


# --- build ---------------------------------------------------------------------------------------------------------------------
def build(stage, config):
    """config: {'lens_seats': {lens id: seat id}, 'cap_seat', 'target_seat'} (initial seats). Returns objects (holders
    and knobs: id -> body path), buttons ({'power': cap path}), seats (id -> collar path), knobs (key -> body path),
    kinematic (front plates and the carriage: name -> body path), z0 (the lab table top, world)."""
    retire_hooke_apparatus(stage)
    repair_cube_extents(stage)
    z0 = table_top_z(stage)
    root = UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(0, 0, z0))
    UsdGeom.Xform.Define(stage, ROOT + '/Buttons')
    physics = _material(stage, '/World/PhysicsMaterials/FiberAnodized', 0.40, 0.35)
    seat_physics = _material(stage, '/World/PhysicsMaterials/FiberSeat', 0.20, 0.15)
    cell = _material(stage, '/World/PhysicsMaterials/FiberHolder', 0.25, 0.20, compliant=C.CUVETTE_CONTACT)
    button_materials = {'rubber': _material(stage, '/World/PhysicsMaterials/FiberRubber', 0.8, 0.7)}
    folder = V.texture_dir(config)
    m = _materials(stage, folder)
    m['folder'] = folder
    m['button'] = button_materials
    seat_colour = (0.04, 0.04, 0.045)
    _breadboard(stage, m, physics)
    for mirror in ('M1', 'M2'):
        _mount(stage, mirror, m, physics)
    buttons = _laser(stage, m, physics, m)
    seats = {}
    seats.update(_target_stand(stage, m, physics, seat_colour, seat_physics))
    seats.update(_coupler(stage, m, physics, seat_colour, seat_physics))
    seats.update(_lens_box(stage, m, physics, seat_colour, seat_physics))
    for sid in seats:
        V.bind(stage.GetPrimAtPath(str(seats[sid].GetPath())), m['seat'])
    meter_in = _meter(stage, m, physics)
    # patch cord: fiber port's boot to the meter's input
    boot = (BEAM_X, FIBER_Y - PORT['size'][1] - 0.028, BEAM_Z)
    cord = [boot, (BEAM_X, boot[1] - 0.030, BEAM_Z - 0.03), (BEAM_X + 0.03, boot[1] - 0.06, BOARD_TOP + 0.004),
            (BEAM_X + 0.12, boot[1] - 0.05, BOARD_TOP + 0.003), (meter_in[0] - 0.02, meter_in[1] + 0.05, BOARD_TOP + 0.003),
            (meter_in[0], meter_in[1] + 0.025, meter_in[2] - 0.004), (meter_in[0], meter_in[1] + 0.008, meter_in[2])]
    V.define(stage, ROOT + '/PatchCord', V.tube(_smooth_path(cord), 0.0015, 10), m['yellow'])
    # holders on their initial seats
    frames = seat_floor_points(seats)
    objects = {}
    for k, oid in enumerate(LENS_IDS):
        (x, y), floor = frames[config['lens_seats'][oid]]
        objects[oid] = _cartridge(stage, oid, k, (x, y), floor + 0.0003, m, cell)
    for oid, kind in ((CAP_ID, 'cap'), (TARGET_ID, 'target')):
        (x, y), floor = frames[config[f'{kind}_seat']]
        objects[oid] = _plate_holder(stage, oid, kind, (x, y), floor + 0.0003, m, cell)
    # knobs (world frame): fiber_knob bodies on revolute joints
    UsdGeom.Xform.Define(stage, KNOB_ROOT)
    UsdGeom.Scope.Define(stage, KNOB_ROOT + '/Joints')
    looks, kphys = KN._materials(stage, KNOB_ROOT)
    knobs = {}
    for key, (face, axis) in knob_layout(z0).items():
        hint = (0.0, 0.0, 1.0)
        knobs[key] = KN.build_knob(stage, knob_path(key), face, axis, f'{KNOB_ROOT}/Joints/{key}', x_hint=hint,
                                   looks=looks, physics=kphys)
        objects[KNOB_IDS[key]] = knobs[key]
    _spots(stage)
    set_power(stage, False)
    table = stage.GetPrimAtPath(TABLE_TOP)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(table)
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    return {'objects': objects, 'buttons': buttons, 'seats': {k: str(v.GetPath()) for k, v in seats.items()},
            'knobs': knobs, 'kinematic': {'M1': front_path('M1'), 'M2': front_path('M2'), 'carriage': CARRIAGE_PATH},
            'z0': z0}


def seat_floor_points(seats):
    """{seat id: ((x, y), floor z)} in the ROOT frame from the collar meshes as authored."""
    out = {}
    for sid, mesh in seats.items():
        pts = np.asarray(UsdGeom.Mesh(mesh.GetPrim()).GetPointsAttr().Get(), float)
        floor = pts[16:20].mean(axis=0)
        out[sid] = ((float(floor[0]), float(floor[1])), float(floor[2]))
    return out


def support_paths():
    """Static prims a holder may stand on or lean against (the backend's support sensors)."""
    return [ROOT + n for n in ('/Breadboard', '/LensBox', '/CouplerBody', '/TargetStand')]


author_joints = C.author_joints
seat_frames = C.seat_frames
SWITCH_DOWN_M, SWITCH_UP_M = C.SWITCH_DOWN_M, C.SWITCH_UP_M

"""Spectrophotometry bench (kit chemistry, recipe chem_spectro_bench; spectro_basic and the other spectrophotometry experiments).

Built on the shared lab stage (robot, table, cameras): the Hooke apparatus is deactivated and three stations
are placed on the table in front of the robot. Pure USD construction; no experiment policy, perception or
planning.

Every place where a cuvette stands is the same kind of seat: a coloured raised collar with a chamfered square
well (26 mm mouth narrowing to 18 mm, 15 mm deep): a cuvette lowered on the collar centre stands on the well
floor, held upright by the walls. Seats: a rack of nine (initially holding the nine empty cuvettes), the dispensing station and
the spectrophotometer's cell well.

Dispensing station: three reagent bottles (stock, water, unknown) behind the station, each with its own
dosing pump; their lines end in one manifold whose three spouts point into the funnel of the cuvette on the
station seat from the side, clear of the gripper. Each line has a coarse (1.00 mL) and a fine (0.10 mL) push
button; the seat sits on a stirrer with a MIX button (each cuvette carries a stir bar). A drip tray under the
seat collects what is dispensed with no cuvette in place.

Spectrophotometer: cell well, BLANK and READ buttons, a 3 x 4 keypad (digits, CLR, ENT) to set the
wavelength, and a seven-segment display (wavelength, absorbance, the BLANK prompt) sized to be read from the robot's
own cameras; it has no camera of its own. Buttons and keys are
spring-returned caps on vertical prismatic joints with 9 mm travel; the runtime reads their positions.

Cuvettes: standard 10 mm cells (12.5 x 12.5 x 45 mm) with a filling funnel on top whose rim is the gripper
handle (manipulation concession: real cuvettes have no handle). The number 1-9 is printed on
each rim. The liquid is drawn as coloured blocks inside an open frame (level from volume, colour from the
absorption spectrum, layers until mixed).

Every instrument faces the robot, which stands on the +Y side of the table looking toward -Y (world_from_robot_base):
text reads upright from the robot, controls sit at the front edge, bottles and the display at the back. Layout
coordinates below are in that robot frame: u to the robot's right (world -X), v away from the robot (world -Y).
Hand clearance (Franka hand about 210 mm along X, the finger axis, and 70 mm along Y): a tall object (bottle,
seated cuvette) stands at least 55 mm in v from every button whose u range the hand covers when pressing it.
"""
from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade

from circuit_apparatus import label
import scene_colliders
from torsion_apparatus import (TABLE_TOP, _box, _collider, _cylinder, _material, repair_cube_extents,
                               retire_hooke_apparatus, table_top_z)

ROOT = '/World/ChemApparatus'
PHYSICS_ROOT = '/World/ChemPhysics'
ORIGIN_XY = (0.95, -0.10)          # apparatus origin on the table top (world); stations are placed around it
TEXT_UP = (0.0, -1.0)                # away from the robot: text reads upright from the robot's side
INK = (0.02, 0.02, 0.02)
PAPER = (0.93, 0.93, 0.90)
# The lab lamps render mid greys near white (albedo 0.3 comes out at 0.7): bases and the housing are dark, so the
# white funnel rims, the caps and the printed text stand out; prints on them are light.
BASE_GREY = (0.12, 0.13, 0.15)
HOUSING_BLUE = (0.12, 0.26, 0.50)

SOURCES = ('stock', 'water', 'unknown')
SOURCE_LABELS = {'stock': 'STOCK', 'water': 'WATER', 'unknown': 'UNKNOWN'}
N_CUVETTES = 9

# --- cuvette ------------------------------------------------------------------------------------
CUVETTE = (0.0125, 0.0125, 0.045)
CUVETTE_INNER = 0.010                        # 10 mm light path, 1 cm^2 inner section: 1 mL = 10 mm of liquid
CUVETTE_FLOOR = 0.00125
FUNNEL_POST_R, FUNNEL_POST_H = 0.0055, 0.050
FUNNEL_RIM_R, FUNNEL_RIM_H = 0.016, 0.005
CUVETTE_TOP = CUVETTE[2] + FUNNEL_POST_H + FUNNEL_RIM_H          # rim top above the cuvette bottom
# Filling port: a short neck in the middle of the rim. The starter grasp aims at the median of the highest tenth of
# the mask's points; on a flat 32 mm rim that tenth is picked by millimetre depth noise and sits 12 mm off centre.
# The neck (10 % of the rim area, 3 mm proud) is that highest tenth, so the grasp is centred.
PORT_R, PORT_H = 0.005, 0.003
LATCH_LOCAL = (0.0, 0.0, CUVETTE_TOP + PORT_H + 0.020)           # as the Hooke weights: 20 mm above the handle top
CUVETTE_MASS_KG = 0.030
# Soft contact layer of the cell body, as the feet of the torsion objects: a position-controlled arm setting a
# held cuvette down on a rigid well floor does not spike the grasp force. PhysX applies the values per contact
# point and the flat bottom meets a mesh floor in about 16 points (measured 3.2e5 N/m with 2e4 per point, the
# 35 N grasp constraint broke within two control frames): 1e3 per point gives about 4 N per control frame of
# descent and 0.02 mm sag under the cuvette's weight.
CUVETTE_CONTACT = (1.0e3, 5.0)               # per contact point: stiffness N/m, damping N s/m
MAX_LAYERS = 3
LIQUID_HIDDEN_SIZE = (1e-4, 1e-4, 1e-4)

# --- seats --------------------------------------------------------------------------------------
COLLAR_HALF = 0.022
COLLAR_H = 0.020
MOUTH_HALF = 0.013
WELL_HALF = 0.0090                           # 2.75 mm around the cell: a cuvette lowered within the starter place skill's
                                             # accuracy lands flat on the floor (a held cuvette cannot slide down a chamfer)
CHAMFER_DEPTH = 0.006
WELL_DEPTH = 0.015
SEAT_FLOOR = COLLAR_H - WELL_DEPTH           # well floor above the collar base

# --- buttons ------------------------------------------------------------------------------------
CAP_R, CAP_H = 0.012, 0.010
KEY = (0.024, 0.024, 0.010)
CAP_LIFT = 0.010                             # cap bottom above the panel when released
# Raised centre of every cap, carrying its print: the starter press on an object's top aims at the median of the
# highest tenth of the mask's points, which on a flat cap sits up to 10 mm off centre; the pad keeps it within 4 mm.
PAD_R, PAD_KEY, PAD_H = 0.005, 0.012, 0.002
TRAVEL_M = 0.009
SWITCH_DOWN_M = 0.004                        # contact closes 4 mm down, opens again above 2 mm
SWITCH_UP_M = 0.002
CAP_MASS_KG = 0.010
SPRING_N_PER_M = 400.0
SPRING_PRELOAD_M = 0.001
SPRING_DAMPING = 4.0

# --- stations (world centre, robot-frame layout; sizes u x v x height) ------------------------------
# From the robot: rack in front, dispensing station behind it on the right, spectrophotometer on the left; all
# within the wrist view at the start pose (about 0.53-1.37 m in X).
# Hand geometry that sets the spacing (the fingers open and close along world X):
# - grasp: the SDK puts the jaw 18 mm above the rim, the open fingers (80 mm apart, 28 mm thick) reach 36 mm below
#   the rim top: rack neighbours along X stand 90 mm apart, along Y (the fingers' thin side) 50 mm;
# - press: closed fingers are 57 mm wide along X and 21 mm along Y: neighbouring caps stand 50 mm apart along u
#   and at least 36 mm along v, so a press 7 mm off centre touches one cap only;
# - the hand body (205 mm along X, 63 mm along Y) passes 46 mm above a pressed cap, below the rim of a cuvette
#   standing on the rack: a cap is either 134 mm in X from every rack column (STOCK: 140 mm) or 62 mm in Y from
#   every rack row (BLANK, READ: 85 mm), and 55 mm in Y from a cuvette on its own station's seat.
# Every graspable seat stands at least 0.31 m (horizontal) from the robot base: a top-down grasp at 0.26-0.28 m
# finds no IK; the rack is 3 deep, its near row at 0.31 m.
RACK = {'centre': (0.93, -0.01), 'base': (0.23, 0.15, 0.012), 'pitch_u': 0.090, 'pitch_v': 0.050,
        'colour': (0.03, 0.16, 0.20)}          # dark collars: the white funnel rims stand out for segmentation
# Dispensing station: three bottle columns at the back, the seat in the middle with the manifold behind it, a
# coarse and a fine row in front, labels and MIX at the front edge. The seated cuvette's rim is 12 mm clear of the
# hand when the coarse row is pressed, the bottles 45 mm behind the hand when it holds a cuvette on the seat.
STATION = {'centre': (0.70, -0.15), 'base': (0.17, 0.315, 0.015), 'base_centre': (0.05, -0.0325), 'seat': (0.05, -0.005),
           'colour': (0.95, 0.55, 0.10),
           'bottles_v': 0.098, 'columns_u': {'stock': 0.0, 'water': 0.05, 'unknown': 0.10},
           'button_v': {'coarse': -0.065, 'fine': -0.115}, 'labels_v': -0.142, 'mix': (0.05, -0.170)}
SPECTRO = {'centre': (1.185, -0.15), 'body': (0.26, 0.29, 0.050), 'well': (0.075, 0.075), 'colour': (0.10, 0.10, 0.11),
           'blank': (0.055, -0.005), 'read': (0.105, -0.005), 'scan': (0.005, 0.016), 'keypad_origin': (-0.105, -0.020),
           'key_pitch_u': 0.050, 'key_pitch_v': 0.036, 'display': (-0.068, 0.080), 'brand': (0.080, -0.100)}
# SCAN key (spectro_mixture bench only): between the keypad's far right key and BLANK, 50 mm from BLANK along u and 36 mm from
# key 3 along v (the cap spacing above); u 0.005 keeps the hand clear of the rack (134 mm in X from its columns)
# and v 0.016 keeps a cuvette standing in the cell well 59 mm away in v. Lime cap, SCAN printed on it.
SCAN_KEY = {'enabled': False, 'id': 'scan', 'text': 'SCAN'}     # the kinetics bench puts KIN (timed reads) here
KEYPAD = (('1', '2', '3'), ('4', '5', '6'), ('7', '8', '9'), ('clear', '0', 'enter'))
KEY_TEXT = {'clear': 'CLR', 'enter': 'ENT'}
SPOUT_R = 0.026                              # spout tips around the funnel axis, clear of the rim (16 mm) and fingers
SPOUT_ABOVE_RIM = 0.015


def _xy(centre, u, v):
    """Robot-frame offset (u to the robot's right, v away from the robot) from a world centre to world xy."""
    return (centre[0] - u, centre[1] - v)


def _local(xy, origin):
    return (xy[0] - origin[0], xy[1] - origin[1])


def cuvette_path(i):
    return f'{ROOT}/Cuvette{i:02d}'


def button_path(bid):
    return f'{ROOT}/Buttons/' + bid.replace(':', '_')


def joint_path(bid):
    return f'{PHYSICS_ROOT}/' + bid.replace(':', '_')


def rack_slots():
    """World xy of the nine rack wells (3 across, 3 deep), numbered 1-9 row by row from the far left as seen
    from the robot."""
    pu, pv = RACK['pitch_u'], RACK['pitch_v']
    return {3 * r + c + 1: _xy(RACK['centre'], (c - 1) * pu, (1 - r) * pv) for r in range(3) for c in range(3)}


def seat_positions():
    """World xy of every seat before layout jitter: rack:1..9, station, spectro."""
    seats = {f'rack:{k}': xy for k, xy in rack_slots().items()}
    seats['station'] = _xy(STATION['centre'], *STATION['seat'])
    seats['spectro'] = _xy(SPECTRO['centre'], *SPECTRO['well'])
    return seats


def seat_base_z():
    """Collar base height above the table top per seat kind."""
    return {'rack': RACK['base'][2], 'station': STATION['base'][2], 'spectro': SPECTRO['body'][2]}


def button_layout():
    """World xy (before jitter), base height above the table and shape of every button: {id: (xy, z, kind)}."""
    out = {}
    zs = STATION['base'][2]
    for name in SOURCES:
        for level, v in STATION['button_v'].items():
            out[f'disp:{name}:{level}'] = (_xy(STATION['centre'], STATION['columns_u'][name], v), zs, 'round')
    out['mix'] = (_xy(STATION['centre'], *STATION['mix']), zs, 'round')
    zb = SPECTRO['body'][2]
    out['spectro:blank'] = (_xy(SPECTRO['centre'], *SPECTRO['blank']), zb, 'round')
    out['spectro:read'] = (_xy(SPECTRO['centre'], *SPECTRO['read']), zb, 'round')
    if SCAN_KEY['enabled']:
        out['spectro:' + SCAN_KEY['id']] = (_xy(SPECTRO['centre'], *SPECTRO['scan']), zb, 'round')
    u0, v0 = SPECTRO['keypad_origin']
    pu, pv = SPECTRO['key_pitch_u'], SPECTRO['key_pitch_v']
    for r, row in enumerate(KEYPAD):
        for c, key in enumerate(row):
            out[f'key:{key}'] = (_xy(SPECTRO['centre'], u0 + c * pu, v0 - r * pv), zb, 'key')
    return out


# --- geometry helpers -----------------------------------------------------------------------------
def _mesh(stage, path, points, faces, colour, *, collide=False, material=None):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateSubdivisionSchemeAttr('none')
    arr = np.asarray(points, float)
    mesh.CreateExtentAttr([Gf.Vec3f(*arr.min(0).tolist()), Gf.Vec3f(*arr.max(0).tolist())])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    if collide:
        _collider(mesh.GetPrim(), material)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')   # static triangle mesh
    return mesh


def _square(half, z):
    return [(-half, -half, z), (half, -half, z), (half, half, z), (-half, half, z)]


def collar(stage, path, centre, base_z, colour, material):
    """A seat: raised square collar, chamfered mouth and a square well (static triangle mesh), at local centre."""
    cx, cy = centre
    rings = [_square(COLLAR_HALF, 0.0), _square(COLLAR_HALF, COLLAR_H), _square(MOUTH_HALF, COLLAR_H),
             _square(WELL_HALF, COLLAR_H - CHAMFER_DEPTH), _square(WELL_HALF, SEAT_FLOOR)]
    points = [(x + cx, y + cy, z + base_z) for ring in rings for x, y, z in ring]
    faces = []

    def band(a, b, outward=True):
        for i in range(4):
            j = (i + 1) % 4
            faces.append([4 * a + i, 4 * a + j, 4 * b + j, 4 * b + i] if outward else
                         [4 * a + i, 4 * b + i, 4 * b + j, 4 * a + j])
    band(0, 1)                    # outer walls
    band(1, 2)                    # top ring
    band(2, 3)                    # chamfer
    band(3, 4)                    # well walls
    faces.append([16, 17, 18, 19])  # floor, facing up
    faces.append([3, 2, 1, 0])      # bottom, facing down
    return _mesh(stage, path, points, faces, colour, collide=True, material=material)


def _flat(stage, path, centre_xy, size_uv, z, colour):
    """Render-only rectangle, size in the robot frame (u across, v away; |u| = world x, |v| = world y)."""
    su, sv = size_uv
    return _box(stage, path, (su, sv, 0.0006), (centre_xy[0], centre_xy[1], z), colour, collide=False)


def _text(stage, path, text, xy, z, height, colour=INK, align='centre'):
    return label(stage, path, text, xy, z, height_m=height, align=align, colour=colour, up=TEXT_UP)


# --- seven-segment display -------------------------------------------------------------------------
SEGMENTS = {'a': (0.5, 1.0, 1), 'b': (1.0, 0.75, 0), 'c': (1.0, 0.25, 0), 'd': (0.5, 0.0, 1),
            'e': (0.0, 0.25, 0), 'f': (0.0, 0.75, 0), 'g': (0.5, 0.5, 1)}      # (u/w, v/h, horizontal)
GLYPHS = {'0': 'abcdef', '1': 'bc', '2': 'abdeg', '3': 'abcdg', '4': 'bcfg', '5': 'acdfg', '6': 'acdefg', '7': 'abc',
          '8': 'abcdefg', '9': 'abcdfg', '-': 'g', ' ': '', 'E': 'adefg', 'r': 'eg'}
SEGMENT_ON = (0.35, 1.0, 0.45)
DISPLAY = ROOT + '/Display'
# Sized to be read from the robot's own cameras (with 20 mm digits and 2.4 mm strokes the decimal point of
# the absorbance was below a pixel in the wrist view from the start pose): 26 mm digits, 3.9 mm strokes, a 5.5 mm
# point, 14 mm prompt letters.
DISPLAY_SIZE = (0.120, 0.120)                # u x v
ROWS = {'wavelength': {'v': 0.030, 'digits': 3, 'point': None, 'height': 0.026, 'unit': 'nm'},
        'absorbance': {'v': -0.012, 'digits': 5, 'point': 2, 'height': 0.026, 'unit': 'A'}}    # sign, then d.ddd
STROKE = 0.15                                # segment thickness / digit height
POINT_SIZE = 1.4                             # decimal point side / stroke
UNIT_HEIGHT, LABEL_HEIGHT = 0.012, 0.014
LABELS_V = -0.047
LABEL_U = {'prompt': -0.016, 'entry': 0.036}  # BLANK prompt, SET mark (centres)
# Dual-wavelength screen (spectro_mixture): both wavelengths with their absorbances, the earlier entered
# on top (far side), the one entered last below it (that row also shows the digits being typed and Err). Four rows of
# 20 mm digits with 3.4 mm strokes and a wider gap for the point, on a 120 x 138 mm glass (the housing between the
# keypad and its back edge): about 5 mm between a wavelength and its absorbance, 8 mm between the pairs; the BLANK and
# SET marks below.
DUAL_DISPLAY = {'centre': (-0.070, 0.074), 'size': (0.120, 0.138), 'labels_v': -0.058}
_DUAL = {'height': 0.020, 'stroke': 0.17, 'gap': 0.35}
DUAL_ROWS = {'wavelength1': dict(_DUAL, v=0.054, digits=3, point=None, unit='nm'),
             'absorbance1': dict(_DUAL, v=0.026, digits=5, point=2, unit='A'),
             'wavelength2': dict(_DUAL, v=-0.005, digits=3, point=None, unit='nm'),
             'absorbance2': dict(_DUAL, v=-0.033, digits=5, point=2, unit='A')}
ALL_ROWS = {**ROWS, **DUAL_ROWS}


def _point_gap(row):
    """Extra space after the digit before the decimal point, which stands in it clear of both neighbours (in the
    plain digit gap it touched their lower verticals: 0.523 read as 0523)."""
    return row['height'] * row.get('gap', 0.25) if row['point'] is not None else 0.0


def _row_origin(name):
    row = ALL_ROWS[name]
    pitch = row['height'] * 0.72
    width = row['digits'] * pitch + _point_gap(row)
    return -width / 2 - 0.008, row['v'] - row['height'] / 2, pitch


def _digit_u(name, i):
    row = ALL_ROWS[name]
    u0, _, pitch = _row_origin(name)
    return u0 + i * pitch + (_point_gap(row) if row['point'] is not None and i >= row['point'] else 0.0)


def _point_u(name):
    row = ALL_ROWS[name]
    _, _, pitch = _row_origin(name)
    w = row['height'] * 0.5
    return _digit_u(name, row['point'] - 1) + w + (pitch - w + _point_gap(row)) / 2


def display_rects(name):
    """Lit rectangles of a row ((u, v) centre, (du, dv) size): {'segments': [...], 'point': rect or None}."""
    row = ALL_ROWS[name]
    _, v0, _ = _row_origin(name)
    w, h, t = row['height'] * 0.5, row['height'], row['height'] * row.get('stroke', STROKE)
    segments = [((_digit_u(name, i) + fu * w, v0 + fv * h), (w, t) if horizontal else (t, h / 2))
                for i in range(row['digits']) for fu, fv, horizontal in SEGMENTS.values()]
    point = ((_point_u(name), v0 + t * POINT_SIZE / 2), (t * POINT_SIZE, t * POINT_SIZE)) if row['point'] is not None else None
    return {'segments': segments, 'point': point}


SEGMENT_EMISSION = 0.55                      # emissive colour = colour x this (a lit screen, readable under the lab lamps)
AMBER = (1.0, 0.75, 0.2)


def front_text(stage, path, text, centre, height, colour=INK):
    """Printed text (render only) on a vertical face toward the robot (+Y), reading left to right as seen from the
    robot (world -X), its top up; centre is the text centre in the parent's frame."""
    from circuit_apparatus import _text_rectangles
    rects, width = _text_rectangles(text, height)
    cx, cy, cz = centre
    points, counts, indices = [], [], []
    for x0, y0, x1, y1 in rects:
        base = len(points)
        for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            points.append(Gf.Vec3f(cx - (px - width / 2), cy, cz + py - height / 2))
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


def bottle_label(stage, bottle, lines, outer, top):
    """A paper label (render only) on the bottle's face toward the robot with the printed lines, as a reagent bottle
    carries its name: the text fills the label's width up to 9 mm high."""
    from circuit_apparatus import _text_rectangles
    width = outer - 0.004
    heights = [min(0.009, 0.9 * width / max(_text_rectangles(line, 1.0)[1], 1e-9)) for line in lines]
    gap = 0.003
    tall = sum(heights) + gap * (len(lines) + 1)
    cz = min(0.45 * top, top - tall / 2 - 0.004)
    y = outer / 2
    _box(stage, bottle + '/LabelPaper', (width, 0.0008, tall), (0.0, y + 0.0005, cz), PAPER, collide=False)
    z = cz + tall / 2 - gap
    for k, (line, h) in enumerate(zip(lines, heights)):
        front_text(stage, f'{bottle}/LabelLine{k}', line, (0.0, y + 0.0011, z - h / 2), h)
        z -= h + gap


def _render_material(stage, path, colour, *, emission=0.0, matte=False):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + '/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(1.0 if matte else 0.5)
    if matte:                                 # no highlight: the lamps do not wash the screen out
        shader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput('specularColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    if emission:
        shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[c * emission for c in colour]))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return material


def _bind(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material)


LOOKS = ROOT + '/Looks'


def liquid_look(prim_path):
    return LOOKS + '/Liquid_' + prim_path[len(ROOT) + 1:].replace('/', '_')


def _set_liquid_colour(stage, prim_path, rgb):
    colour = Gf.Vec3f(*[float(c) for c in rgb])
    UsdGeom.Gprim(stage.GetPrimAtPath(prim_path)).GetDisplayColorAttr().Set([colour])
    shader = UsdShade.Shader(stage.GetPrimAtPath(liquid_look(prim_path) + '/Shader'))
    if shader:
        shader.GetInput('diffuseColor').Set(colour)


def finish_looks(stage):
    """Matte material of its own colour on every drawn prim of the bench (the default surface mirrors the lab lamps:
    a dark blue cap and a light blue one both come out near white in the cameras). A liquid block gets its own
    material, recoloured with the liquid."""
    shared = {}
    for prim in list(Usd.PrimRange(stage.GetPrimAtPath(ROOT))):       # materials are defined while binding
        if not prim.IsA(UsdGeom.Gprim) or UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterial():
            continue
        colour = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
        if not colour:
            continue
        key = tuple(round(float(c), 3) for c in colour[0])
        if prim.GetName().startswith('Liquid'):
            material = _render_material(stage, liquid_look(str(prim.GetPath())), key, matte=True)
        else:
            if key not in shared:
                shared[key] = _render_material(stage, f'{LOOKS}/Matte{len(shared):03d}', key, matte=True)
            material = shared[key]
        _bind(prim, material)


def build_display(stage, centre_xy, z, dual=False):
    """The screen; dual: the four-row dual-wavelength screen (DUAL_ROWS)."""
    rows = DUAL_ROWS if dual else ROWS
    size, labels_v = (DUAL_DISPLAY['size'], DUAL_DISPLAY['labels_v']) if dual else (DISPLAY_SIZE, LABELS_V)
    UsdGeom.Xform.Define(stage, DISPLAY)
    glass = _render_material(stage, DISPLAY + '/Materials/Glass', (0.004, 0.018, 0.010), matte=True)
    lit = _render_material(stage, DISPLAY + '/Materials/Segment', SEGMENT_ON, emission=SEGMENT_EMISSION, matte=True)
    amber = _render_material(stage, DISPLAY + '/Materials/Amber', AMBER, emission=SEGMENT_EMISSION, matte=True)
    _bind(_flat(stage, DISPLAY + '/Glass', centre_xy, size, z - 0.0004, (0.004, 0.018, 0.010)), glass)
    for name, row in rows.items():
        u0, v0, pitch = _row_origin(name)
        w, h, t = row['height'] * 0.5, row['height'], row['height'] * row.get('stroke', STROKE)
        for i in range(row['digits']):
            for seg, (fu, fv, horizontal) in SEGMENTS.items():
                size = (w, t) if horizontal else (t, h / 2)
                xy = _xy(centre_xy, _digit_u(name, i) + fu * w, v0 + fv * h)
                _bind(_flat(stage, f'{DISPLAY}/{name}{i}{seg}', xy, size, z, SEGMENT_ON), lit)
        if row['point'] is not None:
            xy = _xy(centre_xy, _point_u(name), v0 + t * POINT_SIZE / 2)
            _bind(_flat(stage, f'{DISPLAY}/{name}Point', xy, (t * POINT_SIZE, t * POINT_SIZE), z, SEGMENT_ON), lit)
        _bind(_text(stage, f'{DISPLAY}/{name}Unit', row['unit'], _xy(centre_xy, u0 + row['digits'] * pitch + _point_gap(row) + 0.004, v0 + 0.004),
                    z, UNIT_HEIGHT, SEGMENT_ON, align='left'), lit)
    _bind(_text(stage, f'{DISPLAY}/Prompt', 'BLANK', _xy(centre_xy, LABEL_U['prompt'], labels_v), z, LABEL_HEIGHT, AMBER), amber)
    _bind(_text(stage, f'{DISPLAY}/Entry', 'SET', _xy(centre_xy, LABEL_U['entry'], labels_v), z, LABEL_HEIGHT, AMBER), amber)
    for path in _toggled_paths(rows):
        _remember_lit(stage.GetPrimAtPath(path))
    blank = {name: ' ' * row['digits'] for name, row in rows.items()}
    points = {f"point{name[-1]}": False for name, row in rows.items() if dual and row['point'] is not None}
    show(stage, blank | {'prompt': False, 'entry': False} | (points if dual else {'point': False}), None)


# An unlit segment or label sinks under the glass. The renderer's denoiser keeps drawing what a still camera saw
# before; the runtime resets that history after a change and checks the screen image (chem_backend).
HIDE_M = 0.004
LIT_ATTR = 'embodiedlab:litTranslate'


def _toggled_paths(rows=None):
    rows = ROWS if rows is None else rows
    paths = [f'{DISPLAY}/{name}{i}{seg}' for name, row in rows.items() for i in range(row['digits']) for seg in SEGMENTS]
    return paths + [f'{DISPLAY}/Prompt', f'{DISPLAY}/Entry'] + [f'{DISPLAY}/{name}Point' for name, row in rows.items()
                                                                if row['point'] is not None]


def _remember_lit(prim):
    api = UsdGeom.XformCommonAPI(prim)
    translate = Gf.Vec3d(api.GetXformVectors(Usd.TimeCode.Default())[0])
    api.SetTranslate(translate)                      # label meshes get their translate op here
    prim.CreateAttribute(LIT_ATTR, Sdf.ValueTypeNames.Double3, custom=True).Set(translate)


def _light(stage, path, lit):
    prim = stage.GetPrimAtPath(path)
    t = prim.GetAttribute(LIT_ATTR).Get()
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(t) if lit else Gf.Vec3d(t[0], t[1], t[2] - HIDE_M))


# The screen's text (pxr-free, shared with the evaluator's tests): chem_spectro.screen_text.
from chem_spectro import screen_text as display_text  # noqa: E402


def show(stage, text, previous):
    """Update the display prims that differ from the previously shown text dict."""
    if text == previous:
        return text
    for name in [n for n in ALL_ROWS if n in text]:
        for i, char in enumerate(text[name]):
            if previous is not None and previous[name][i] == char:
                continue
            lit = GLYPHS[char]
            for seg in SEGMENTS:
                _light(stage, f'{DISPLAY}/{name}{i}{seg}', seg in lit)
    for key, path in (('prompt', 'Prompt'), ('entry', 'Entry'), ('point', 'absorbancePoint'),
                      ('point1', 'absorbance1Point'), ('point2', 'absorbance2Point')):
        if key in text and (previous is None or previous[key] != text[key]):
            _light(stage, f'{DISPLAY}/{path}', text[key])
    return text


# --- build ------------------------------------------------------------------------------------------
def _rigid(prim, mass, com=(0, 0, 0), ccd=False):
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    body.CreateSolverPositionIterationCountAttr(16)
    body.CreateSolverVelocityIterationCountAttr(8)
    body.CreateSleepThresholdAttr(0.0)
    body.CreateLinearDampingAttr(0.0)
    body.CreateAngularDampingAttr(0.0)
    if ccd:
        body.CreateEnableCCDAttr(True)
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(float(mass))
    api.CreateCenterOfMassAttr(Gf.Vec3f(*com))


def _cuvette(stage, index, xy, bottom_z, materials):
    path = cuvette_path(index)
    xform = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(xy[0], xy[1], bottom_z))
    sx, sy, sz = CUVETTE
    body = _box(stage, path + '/Body', CUVETTE, (0, 0, sz / 2), (0.9, 0.9, 0.9), material=materials['cell'])
    UsdGeom.Imageable(body).MakeInvisible()          # the collider; the visible cell is an open frame
    frame = (0.80, 0.86, 0.90)
    edge = 0.0012
    for k, (dx, dy) in enumerate(((1, 1), (1, -1), (-1, -1), (-1, 1))):
        _box(stage, f'{path}/Edge{k}', (edge, edge, sz), (dx * (sx - edge) / 2, dy * (sy - edge) / 2, sz / 2), frame,
             collide=False)
    _box(stage, path + '/Floor', (sx, sy, CUVETTE_FLOOR), (0, 0, CUVETTE_FLOOR / 2), frame, collide=False)
    _box(stage, path + '/Collar', (sx, sy, 0.002), (0, 0, sz - 0.001), frame, collide=False)
    for k in range(MAX_LAYERS):
        _box(stage, f'{path}/Liquid{k}', LIQUID_HIDDEN_SIZE, (0, 0, CUVETTE_FLOOR / 2), (1, 1, 1), collide=False)
    white = (0.94, 0.94, 0.92)
    _cylinder(stage, path + '/FunnelStem', FUNNEL_POST_R, FUNNEL_POST_H, (0, 0, sz + FUNNEL_POST_H / 2), white,
              materials['pmma'])
    _cylinder(stage, path + '/FunnelRim', FUNNEL_RIM_R, FUNNEL_RIM_H, (0, 0, sz + FUNNEL_POST_H + FUNNEL_RIM_H / 2),
              white, materials['pmma'])
    _cylinder(stage, path + '/FunnelPort', PORT_R, PORT_H, (0, 0, CUVETTE_TOP + PORT_H / 2), white, collide=False)
    _cylinder(stage, path + '/FunnelMouth', 0.0025, 0.0004, (0, 0, CUVETTE_TOP + PORT_H + 0.0002), (0.05, 0.05, 0.05),
              collide=False)
    _text(stage, path + '/Number', str(index), (0.0, 0.0105), CUVETTE_TOP + 0.0004, 0.008)
    _rigid(xform.GetPrim(), CUVETTE_MASS_KG, (0, 0, 0.025), ccd=True)
    return path


def set_liquid(stage, index, layers):
    """layers: [(height m, linear rgb)] bottom to top (at most MAX_LAYERS; empty list hides the liquid). An unused
    layer shrinks inside the floor (transforms, not visibility tokens, reach the renderer reliably)."""
    path = cuvette_path(index)
    z = CUVETTE_FLOOR
    for k in range(MAX_LAYERS):
        cube = UsdGeom.Cube(stage.GetPrimAtPath(f'{path}/Liquid{k}'))
        api = UsdGeom.XformCommonAPI(cube)
        if k < len(layers) and layers[k][0] > 1e-5:
            height, rgb = layers[k]
            api.SetTranslate(Gf.Vec3d(0, 0, z + height / 2))
            api.SetScale(Gf.Vec3f(CUVETTE_INNER, CUVETTE_INNER, height))
            _set_liquid_colour(stage, f'{path}/Liquid{k}', rgb)
            z += height
        else:
            api.SetTranslate(Gf.Vec3d(0, 0, CUVETTE_FLOOR / 2))
            api.SetScale(Gf.Vec3f(*LIQUID_HIDDEN_SIZE))


# Reagent bottles are drawn to the scale of their contents, like the cuvettes (1 cm2 x 4.5 cm): a square bottle
# whose inner volume is the initial volume plus 15 % headspace, inner height 1.4 x its side; the liquid block has
# the remaining volume.
BOTTLE_HEADSPACE, BOTTLE_ASPECT, BOTTLE_WALL = 1.15, 1.4, 0.002
BOTTLE_COLUMN_H = 0.135                      # collider above the plinth: bottle, pump and the riser of its dosing line
_BOTTLE_INNER = {}                           # source -> (side m, height m), set by build


def bottle_inner(volume_mL):
    """Inner side and height (m) of the bottle that holds volume_mL."""
    side_cm = (float(volume_mL) * BOTTLE_HEADSPACE / BOTTLE_ASPECT) ** (1.0 / 3.0)
    return side_cm / 100.0, BOTTLE_ASPECT * side_cm / 100.0


def bottle_liquid_path(name):
    return f'{ROOT}/Station/{name.title()}Bottle/Liquid'


def set_bottle(stage, name, fraction, rgb):
    side, inner_h = _BOTTLE_INNER[name]
    cube = UsdGeom.Cube(stage.GetPrimAtPath(bottle_liquid_path(name)))
    height = max(1e-4, inner_h / BOTTLE_HEADSPACE * min(1.0, max(0.0, fraction)))
    api = UsdGeom.XformCommonAPI(cube)
    api.SetTranslate(Gf.Vec3d(0, 0, BOTTLE_WALL + height / 2))
    api.SetScale(Gf.Vec3f(side, side, height))
    _set_liquid_colour(stage, bottle_liquid_path(name), rgb)


def _button(stage, bid, xy, base_z, kind, colour, text, materials, text_colour=(0.97, 0.97, 0.95)):
    """Cap (dynamic) above a housing ring (render). Joint authored later (author_joints)."""
    path = button_path(bid)
    cap = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(cap).SetTranslate(Gf.Vec3d(xy[0], xy[1], base_z + CAP_LIFT + CAP_H / 2))
    if kind == 'round':
        _cylinder(stage, path + '/Cap', CAP_R, CAP_H, (0, 0, 0), colour, materials['rubber'])
        _cylinder(stage, path + '_Housing', CAP_R + 0.003, CAP_LIFT - 0.001, (xy[0], xy[1], base_z + (CAP_LIFT - 0.001) / 2),
                  (0.25, 0.25, 0.27), collide=False)
    else:
        _box(stage, path + '/Cap', KEY, (0, 0, 0), colour, material=materials['rubber'])
        _box(stage, path + '_Housing', (KEY[0] + 0.004, KEY[1] + 0.004, CAP_LIFT - 0.001),
             (xy[0], xy[1], base_z + (CAP_LIFT - 0.001) / 2), (0.25, 0.25, 0.27), collide=False)
    if kind == 'round':
        _cylinder(stage, path + '/Pad', PAD_R, PAD_H, (0, 0, CAP_H / 2 + PAD_H / 2), colour, materials['rubber'])
    else:
        _box(stage, path + '/Pad', (PAD_KEY, PAD_KEY, PAD_H), (0, 0, CAP_H / 2 + PAD_H / 2), colour, material=materials['rubber'])
    if text:
        _text(stage, path + '/Text', text, (0.0, 0.0), CAP_H / 2 + PAD_H + 0.0004,
              0.009 if len(text) < 3 else 0.007, colour=text_colour)
    _rigid(cap.GetPrim(), CAP_MASS_KG)
    return path


def build(stage, config):
    """Author the bench in its unjittered pose. config: {'bottle_mL': {source: initial volume}, 'bottle_rgb':
    {source: linear rgb}, 'bottle_fill': {source: fraction}, optional 'labels': {source: printed name},
    'scan_key': bool (spectro_mixture) and 'dual_display': bool (the dual-wavelength screen, spectro_mixture)}. Button joints are authored later (author_joints), after any layout jitter."""
    SCAN_KEY['enabled'] = bool(config.get('scan_key', False) or config.get('extra_key'))
    SCAN_KEY['id'] = config.get('extra_key') or 'scan'
    SCAN_KEY['text'] = SCAN_KEY['id'].upper()
    labels = dict(SOURCE_LABELS, **(config.get('labels') or {}))
    retire_hooke_apparatus(stage)
    repair_cube_extents(stage)
    z0 = table_top_z(stage)
    origin = ORIGIN_XY
    materials = {'rubber': _material(stage, '/World/PhysicsMaterials/ChemRubber', 0.8, 0.7),
                 'pmma': _material(stage, '/World/PhysicsMaterials/ChemPMMA', 0.25, 0.20),
                 'cell': _material(stage, '/World/PhysicsMaterials/ChemCell', 0.25, 0.20, compliant=CUVETTE_CONTACT),
                 'seat': _material(stage, '/World/PhysicsMaterials/ChemSeat', 0.20, 0.15)}
    root = UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(origin[0], origin[1], z0))
    L = lambda xy: _local(xy, origin)
    seats = {}
    # Rack.
    rc = L(RACK['centre'])
    bu, bv, bh = RACK['base']
    _box(stage, ROOT + '/Rack', (bu, bv, bh), (rc[0], rc[1], bh / 2), BASE_GREY, material=materials['rubber'])
    for k, xy in rack_slots().items():
        seats[f'rack:{k}'] = collar(stage, f'{ROOT}/RackSeat{k:02d}', L(xy), bh, RACK['colour'], materials['seat'])
        # the well number, printed on a paper tag on the collar top in its front-left corner (as seen from the
        # robot), outside the funnel rim
        corner = (MOUTH_HALF + COLLAR_HALF) / 2
        nx, ny = L(_xy(xy, -corner, -corner))
        _flat(stage, f'{ROOT}/RackNumberTag{k}', (nx, ny), (0.0085, 0.0085), bh + COLLAR_H + 0.0003, PAPER)
        _text(stage, f'{ROOT}/RackNumber{k}', str(k), (nx, ny), bh + COLLAR_H + 0.0008, 0.0062, INK)
    # Dispensing station: bottles at the back, the seat front left with the manifold behind it, a coarse and a
    # fine button in front of each bottle, MIX in front of the seat.
    UsdGeom.Xform.Define(stage, ROOT + '/Station')
    UsdGeom.Xform.Define(stage, ROOT + '/Buttons')
    C = STATION['centre']
    bu, bv, bh = STATION['base']
    sc = L(_xy(C, *STATION['base_centre']))
    _box(stage, ROOT + '/StationBase', (bu, bv, bh), (sc[0], sc[1], bh / 2), BASE_GREY,
         material=materials['rubber'])
    seat_u, seat_v = STATION['seat']
    seat_xy = L(_xy(C, seat_u, seat_v))
    _box(stage, ROOT + '/StationDripTray', (0.060, 0.060, 0.004), (seat_xy[0], seat_xy[1], bh + 0.002),
         (0.12, 0.12, 0.13), collide=False)
    seats['station'] = collar(stage, ROOT + '/StationSeat', seat_xy, bh + 0.004, STATION['colour'], materials['seat'])
    rim_z = bh + 0.004 + SEAT_FLOOR + CUVETTE_TOP
    spout_z = rim_z + SPOUT_ABOVE_RIM
    bottles_v = STATION['bottles_v']
    columns = STATION['columns_u']
    shelf = L(_xy(C, (min(columns.values()) + max(columns.values())) / 2, bottles_v))
    plinth = 0.020
    _box(stage, ROOT + '/StationShelf', (max(columns.values()) - min(columns.values()) + 0.050, 0.054, plinth),
         (shelf[0], shelf[1], bh + plinth / 2), (0.20, 0.21, 0.23), material=materials['rubber'])
    for k, name in enumerate(SOURCES):
        u = columns[name]
        bottle = f'{ROOT}/Station/{name.title()}Bottle'
        bxy = L(_xy(C, u, bottles_v))
        xform = UsdGeom.Xform.Define(stage, bottle)
        UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(bxy[0], bxy[1], bh + plinth))
        side, inner_h = _BOTTLE_INNER[name] = bottle_inner(config['bottle_mL'][name])
        outer, top = side + 2 * BOTTLE_WALL, BOTTLE_WALL + inner_h
        glass = (0.75, 0.80, 0.82)
        # The bottle's glass panes are not drawn (only its edges, floor and shoulder are): the collider is the glass
        # box itself, exactly the drawn outline. Pump, riser and line collide as drawn (scene_colliders).
        size, cz = ((outer, outer, top + BOTTLE_WALL), (top + BOTTLE_WALL) / 2) if scene_colliders.ENABLED else \
            ((outer + 0.002, outer + 0.002, BOTTLE_COLUMN_H), BOTTLE_COLUMN_H / 2)
        _box(stage, bottle + '/Collider', size, (0, 0, cz),
             (1, 1, 1), material=materials['rubber'])
        UsdGeom.Imageable(stage.GetPrimAtPath(bottle + '/Collider')).MakeInvisible()
        _box(stage, bottle + '/Glass', (outer, outer, BOTTLE_WALL), (0, 0, BOTTLE_WALL / 2), glass, collide=False)
        _box(stage, bottle + '/Liquid', (side, side, 0.01), (0, 0, BOTTLE_WALL + 0.005), (1, 1, 1), collide=False)
        set_bottle(stage, name, config['bottle_fill'][name], config['bottle_rgb'][name])
        for e, (dx, dy) in enumerate(((1, 1), (1, -1), (-1, -1), (-1, 1))):
            _box(stage, f'{bottle}/Edge{e}', (BOTTLE_WALL, BOTTLE_WALL, top), (dx * (side + BOTTLE_WALL) / 2,
                 dy * (side + BOTTLE_WALL) / 2, top / 2), glass, collide=False)
        _box(stage, bottle + '/Shoulder', (outer, outer, BOTTLE_WALL), (0, 0, top + BOTTLE_WALL / 2), glass, collide=False)
        bottle_label(stage, bottle, [labels[name]], outer, top)
        pump_r, pump_top = min(0.012, side / 2 - 0.002), top + BOTTLE_WALL + 0.020
        _cylinder(stage, bottle + '/Pump', pump_r, 0.020, (0, 0, pump_top - 0.010), (0.85, 0.85, 0.85), collide=False)
        _text(stage, f'{ROOT}/Station/{name.title()}Label', labels[name],
              L(_xy(C, u, STATION['labels_v'])), bh + 0.0004, 0.008, PAPER)
        # Line from the pump forward to the manifold behind the seat, then the spout toward the funnel from
        # behind (+v, world -Y): the fingers close along world X and stay clear of it.
        spread = (k - 1) * 0.0125
        hose_z = spout_z + 0.012
        start = _xy(C, u, bottles_v)
        riser = hose_z + 0.003 - (bh + plinth + pump_top)
        _cylinder(stage, bottle + '/Riser', 0.003, riser, (0, 0, pump_top + riser / 2), (0.85, 0.85, 0.85), collide=False)
        end = _xy(C, seat_u + spread, seat_v + SPOUT_R + 0.012)
        _pipe(stage, f'{ROOT}/Station/{name.title()}Line', L(start), L(end), hose_z, 0.003, (0.85, 0.85, 0.85))
        tip = _xy(C, seat_u + spread, seat_v + SPOUT_R)
        _pipe(stage, f'{ROOT}/Station/{name.title()}Spout', L(end), L(tip), spout_z + 0.006, 0.0025,
              (0.70, 0.72, 0.75))
    # Manifold block holding the three spouts (collider: the planner keeps the arm above it).
    block = L(_xy(C, seat_u, seat_v + SPOUT_R + 0.020))
    _box(stage, ROOT + '/StationManifold', (0.040, 0.016, 0.014), (block[0], block[1], spout_z + 0.010), (0.55, 0.57, 0.60),
         material=materials['rubber'])
    _text(stage, ROOT + '/Station/MixLabel', 'MIX', L(_xy(C, STATION['mix'][0] + 0.032, STATION['mix'][1])),
          bh + 0.0004, 0.010, PAPER)
    # Spectrophotometer: display and cell well at the back, BLANK / READ and the keypad at the front.
    P = SPECTRO['centre']
    pc = L(P)
    bu, bv, bh = SPECTRO['body']
    _box(stage, ROOT + '/Spectro', (bu, bv, bh), (pc[0], pc[1], bh / 2), HOUSING_BLUE, material=materials['rubber'])
    seats['spectro'] = collar(stage, ROOT + '/SpectroWell', L(_xy(P, *SPECTRO['well'])), bh,
                              SPECTRO['colour'], materials['seat'])
    dual = bool(config.get('dual_display'))
    display_xy = L(_xy(P, *(DUAL_DISPLAY['centre'] if dual else SPECTRO['display'])))
    build_display(stage, display_xy, bh + 0.0010, dual=dual)
    for bid, text in (('blank', 'BLANK'), ('read', 'READ')):
        u, v = SPECTRO[bid]
        _text(stage, f'{ROOT}/Spectro{bid.title()}Label', text, L(_xy(P, u, v + 0.021)), bh + 0.0004, 0.008, PAPER)
    _text(stage, ROOT + '/SpectroBrand', 'UV-VIS  1 cm', L(_xy(P, *SPECTRO['brand'])), bh + 0.0004, 0.010, PAPER)
    # Buttons.
    colours = {'coarse': (0.10, 0.32, 0.75), 'fine': (0.45, 0.68, 0.95)}
    buttons = {}
    for bid, (xy, base, kind) in button_layout().items():
        if bid.startswith('disp:'):
            colour, text = colours[bid.rsplit(':', 1)[1]], {'coarse': '1.0', 'fine': '0.1'}[bid.rsplit(':', 1)[1]]
        elif bid == 'mix':
            colour, text = (0.15, 0.62, 0.25), None
        elif bid == 'spectro:blank':
            colour, text = (0.50, 0.25, 0.75), None
        elif bid == 'spectro:read':
            colour, text = (0.80, 0.10, 0.08), None
        elif bid in ('spectro:scan', 'spectro:kin'):
            colour, text = (0.55, 0.85, 0.10), SCAN_KEY['text']   # lime: no other cap, collar or liquid has this hue
        else:
            key = bid.split(':', 1)[1]
            colour, text = (0.20, 0.21, 0.23), KEY_TEXT.get(key, key)
        ink = INK if bid.endswith(':fine') or bid in ('spectro:scan', 'spectro:kin') else (0.97, 0.97, 0.95)
        buttons[bid] = _button(stage, bid, L(xy), base, kind, colour, text, materials, ink)
    # Cuvettes, empty, one per rack well.
    cuvettes = {}
    rack_base = RACK['base'][2]
    for k, xy in rack_slots().items():
        cuvettes[k] = _cuvette(stage, k, L(xy), rack_base + SEAT_FLOOR + 0.0003, materials)
    table_top = stage.GetPrimAtPath(TABLE_TOP)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(table_top)
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    finish_looks(stage)
    for name in SOURCES:                              # the liquid materials exist now: colour them
        set_bottle(stage, name, config['bottle_fill'][name], config['bottle_rgb'][name])
    return {'cuvettes': cuvettes, 'buttons': buttons, 'seats': {k: str(v.GetPath()) for k, v in seats.items()}}


def _pipe(stage, path, a, b, z, radius, colour):
    """Render-only horizontal tube between two local xy points at height z."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    yaw = math.degrees(math.atan2(dy, dx))
    return _box(stage, path, (length, 2 * radius, 2 * radius), ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, z), colour,
                rotate_xyz=(0, 0, yaw), collide=False)


def author_joints(stage, buttons):
    """Spring-returned vertical prismatic joint from the world to each cap at its current (possibly jittered) pose.
    Joint coordinate q: cap travel, negative when pressed; the spring aims slightly above the upper stop."""
    UsdGeom.Scope.Define(stage, PHYSICS_ROOT)
    for bid, path in buttons.items():
        world = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path(bid))
        joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(world.ExtractTranslation()))
        joint.CreateLocalRot0Attr(Gf.Quatf(world.ExtractRotationQuat()))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1))
        joint.CreateAxisAttr('Z')
        joint.CreateLowerLimitAttr(-TRAVEL_M)
        joint.CreateUpperLimitAttr(0.0)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
        drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(SPRING_N_PER_M)
        drive.CreateDampingAttr(SPRING_DAMPING)
        drive.CreateTargetPositionAttr(SPRING_PRELOAD_M)
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateMaxForceAttr(100.0)


def seat_frames(stage, seats):
    """World xy of each well centre and the world z of its floor, after any jitter (from the collar prims)."""
    cache = UsdGeom.XformCache()
    out = {}
    for sid, path in seats.items():
        prim = stage.GetPrimAtPath(path)
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), float)
        floor = points[16:20].mean(axis=0)
        world = cache.GetLocalToWorldTransform(prim)
        p = np.asarray(world.Transform(Gf.Vec3d(*floor.tolist())))
        out[sid] = (p[:2].copy(), float(p[2]))
    return out


def seat_paths(stage=None):
    out = {f'rack:{k}': f'{ROOT}/RackSeat{k:02d}' for k in rack_slots()}
    out['station'] = ROOT + '/StationSeat'
    out['spectro'] = ROOT + '/SpectroWell'
    return out


def button_paths(stage=None):
    return {bid: button_path(bid) for bid in button_layout()}

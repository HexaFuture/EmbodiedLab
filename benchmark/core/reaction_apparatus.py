"""Reaction bench apparatus (reaction_yield): the kinetics bench (chem_apparatus: rack, dispensing station, spectro-
photometer with the KIN key) and a dry block heater in front of the spectrophotometer.

The dispensing station gets a fourth line, QUENCH, to the right of the three (as seen from the robot): the kit's
station is built from chem_apparatus's module tables (SOURCES, SOURCE_LABELS, STATION), which four_lines() swaps for
the length of each call into chem_apparatus and puts back, so the kit's other benches never see the fourth line.

The block (from the robot): a grey powder-coated housing with an anodized aluminium top holding four cuvette wells in
a 2 x 2 grid (numbered 1-4 row by row from the far left, as the rack), five keys in front of them (LOG, the data
logger on / off; -10, -1, +1, +10 C on the set point) and the controller screen on its front face (SET and BLOCK,
thermal_lcd, with a LOG mark lit while the logger records). The grid follows the kit's
seat spacing (the fingers close along world X): columns 90 mm apart along X, rows 50 mm along Y, so every well can be
grasped with the other three occupied (in a 48 mm row the open fingers came down on the neighbours'
rims). The keys stand 55 mm in front of the near row: a hand pressing one passes clear of a cuvette standing there. They
keep the kit's 34 mm pitch, centred on the block: with LOG at the left end the set-point keys moved 17 mm right.
Wells are the kit's seats (collar mesh: chamfered mouth, square well), drawn under the aluminium top; behind the
collars the block is solid: a base box whose top lies 0.5 mm under the well floors and a top plate around the well
mouths (box colliders), so a cuvette pushed into a well cannot pass through its floor or walls. The keys are the kit's
spring-returned caps. Everything else is chem_apparatus unchanged, so the backend uses this module as its asset.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

from pxr import Gf, UsdGeom

import chem_apparatus as C
import thermal_lcd as L
import visual_kit as V
from chem_apparatus import *  # noqa: F401,F403  (the backend's asset interface)
from chem_apparatus import (COLLAR_H, COLLAR_HALF, MOUTH_HALF, ROOT, SEAT_FLOOR, WELL_HALF, _bind, _box, _button,
                            _material, _render_material, _text, _xy, collar)

# Robot-frame offsets (u to the robot's right, v away from it) from the block centre. World: block x 1.135-1.315,
# y 0.005-0.165 (the spectrophotometer's front face is at y -0.005); wells at x 1.270 / 1.180, y 0.035 / 0.085; keys
# y 0.140. Graspable wells 0.35-0.45 m and keys 0.31 m or more (horizontal) from the robot base.
BLOCK = {'centre': (1.225, 0.085), 'size': (0.180, 0.160, 0.040), 'cols_u': (-0.045, 0.045), 'rows_v': (0.050, 0.000),
         'keys_v': -0.055}
FLOOR_GAP = 0.0005                  # solid base top under the well floor: the kit's seat alone carries a seated cuvette
BLOCK_KEYS = (('block:log', 'LOG'), ('block:down10', '-10'), ('block:down1', '-1'), ('block:up1', '+1'),
              ('block:up10', '+10'))
LOG_COLOUR = (0.22, 0.24, 0.27)     # dark grey, matte (the station has green MIX and lime KIN keys)
KEY_PITCH = 0.034
BLOCK_PATH = ROOT + '/Block'
CONTROLLER_PATH = ROOT + '/BlockController'
CELLS, DECIMALS = 3, 1
QUENCH_U = 0.15                     # the QUENCH column, 50 mm right of UNKNOWN's (the kit's column pitch)


@contextmanager
def four_lines():
    """chem_apparatus with the QUENCH line: a fourth bottle column, its buttons and a station base wide enough."""
    saved = C.SOURCES, C.SOURCE_LABELS, C.STATION
    station = deepcopy(C.STATION)
    station['columns_u']['quench'] = QUENCH_U
    lo, hi = min(station['columns_u'].values()), max(station['columns_u'].values())
    margin = (station['base'][0] - (max(saved[2]['columns_u'].values()) - min(saved[2]['columns_u'].values()))) / 2
    station['base'] = (hi - lo + 2 * margin,) + tuple(station['base'][1:])
    station['base_centre'] = ((lo + hi) / 2, station['base_centre'][1])
    C.SOURCES, C.SOURCE_LABELS, C.STATION = saved[0] + ('quench',), dict(saved[1], quench='QUENCH'), station
    try:
        yield
    finally:
        C.SOURCES, C.SOURCE_LABELS, C.STATION = saved


def well_xy(w):
    """World xy of block well w (1-4: far row left, right, then near row left, right, as seen from the robot)."""
    row, col = divmod(w - 1, 2)
    return _xy(BLOCK['centre'], BLOCK['cols_u'][col], BLOCK['rows_v'][row])


def key_xy(k):
    return _xy(BLOCK['centre'], (k - (len(BLOCK_KEYS) - 1) / 2) * KEY_PITCH, BLOCK['keys_v'])


def seat_positions():
    with four_lines():
        return C.seat_positions() | {f'block:{w}': well_xy(w) for w in range(1, 5)}


def seat_base_z():
    with four_lines():
        return C.seat_base_z() | {'block': BLOCK['size'][2] - COLLAR_H}


def button_layout():
    zb = BLOCK['size'][2]
    with four_lines():
        return C.button_layout() | {bid: (key_xy(k), zb, 'round') for k, (bid, _) in enumerate(BLOCK_KEYS)}


def seat_paths(stage=None):
    with four_lines():
        return C.seat_paths() | {f'block:{w}': f'{BLOCK_PATH}/Well{w}' for w in range(1, 5)}


def button_paths(stage=None):
    return {bid: C.button_path(bid) for bid in button_layout()}


def _looks(stage):
    folder = V.texture_dir()
    dif, rough = V.grain(folder, 'block_anodized', 0.05, 0.38, 0.06)
    top = V.material(stage, BLOCK_PATH + '/Looks/Aluminium', (0.78, 0.79, 0.80), metallic=0.85, roughness=0.38,
                     diffuse=dif, rough_map=rough)
    pdif, prough = V.grain(folder, 'block_powder', 0.04, 0.70, 0.04)
    housing = V.material(stage, BLOCK_PATH + '/Looks/Housing', (0.80, 0.81, 0.80), roughness=0.72, diffuse=pdif,
                         rough_map=prough, normal=V.orange_peel(folder))
    trim = V.material(stage, BLOCK_PATH + '/Looks/Trim', (0.13, 0.14, 0.15), roughness=0.55)
    return top, housing, trim


def _top_pieces(x_lo, x_hi, y_lo, y_hi, hole):
    """Rectangles (x0, x1, y0, y1, local) covering the block top between its edges, around the square well mouths
    (half-side `hole`): full-width bands between the rows, pieces between the mouths within a row's band."""
    o = C.ORIGIN_XY
    xs = sorted({round(well_xy(w)[0] - o[0], 6) for w in range(1, 5)})
    ys = sorted({round(well_xy(w)[1] - o[1], 6) for w in range(1, 5)})
    y_edges = [y_lo] + [v for y in ys for v in (y - hole, y + hole)] + [y_hi]
    x_edges = [x_lo] + [v for x in xs for v in (x - hole, x + hole)] + [x_hi]
    pieces = []
    for i in range(len(y_edges) - 1):
        c, d = y_edges[i], y_edges[i + 1]
        if i % 2 == 0:                                   # between rows (or at the edges): one full-width band
            pieces.append((x_lo, x_hi, c, d))
        else:                                            # a row of mouths: the pieces between them
            pieces += [(x_edges[2 * j], x_edges[2 * j + 1], c, d) for j in range(len(xs) + 1)]
    return pieces


def block_support_paths():
    """The block's own colliders besides the well collars (contact with them counts as support)."""
    (cx, cy), (sx, sy, _) = BLOCK['centre'], BLOCK['size']
    o = C.ORIGIN_XY
    lc = (cx - o[0], cy - o[1])
    n = len(_top_pieces(lc[0] - sx / 2, lc[0] + sx / 2, lc[1] - sy / 2, lc[1] + sy / 2, MOUTH_HALF))
    return [BLOCK_PATH + '/Base'] + [f'{BLOCK_PATH}/Top{i}' for i in range(n)]


def _block(stage, z0_local, materials):
    """Housing and top (render: housing below the collars, the aluminium top around the well mouths) over invisible
    colliders: the kit's collars, a solid base box up to just under the well floors and the top plate around the
    mouths (boxes), so the block is solid everywhere but in the wells."""
    (cx, cy), (sx, sy, sz) = BLOCK['centre'], BLOCK['size']
    base_h = sz - COLLAR_H
    o = C.ORIGIN_XY
    lc = (cx - o[0], cy - o[1])
    solid_h = base_h + SEAT_FLOOR - FLOOR_GAP
    _box(stage, BLOCK_PATH + '/Base', (sx, sy, solid_h), (lc[0], lc[1], solid_h / 2), (0.8, 0.8, 0.8),
         material=materials['seat'])
    UsdGeom.Imageable(stage.GetPrimAtPath(BLOCK_PATH + '/Base')).MakeInvisible()
    seats = {}
    for w in range(1, 5):
        xy = well_xy(w)
        seats[f'block:{w}'] = collar(stage, f'{BLOCK_PATH}/Well{w}', (xy[0] - o[0], xy[1] - o[1]), base_h,
                                     (0.6, 0.6, 0.62), materials['seat'])
        UsdGeom.Imageable(seats[f'block:{w}'].GetPrim()).MakeInvisible()
    # The top plate: boxes from the base up to the top around the mouths (the collars' chamfers and wells inside).
    plate_h = sz - solid_h
    for i, (a, b, c, d) in enumerate(_top_pieces(lc[0] - sx / 2, lc[0] + sx / 2, lc[1] - sy / 2, lc[1] + sy / 2,
                                                 MOUTH_HALF)):
        _box(stage, f'{BLOCK_PATH}/Top{i}', (b - a, d - c, plate_h), ((a + b) / 2, (c + d) / 2, solid_h + plate_h / 2),
             (0.8, 0.8, 0.8), material=materials['seat'])
        UsdGeom.Imageable(stage.GetPrimAtPath(f'{BLOCK_PATH}/Top{i}')).MakeInvisible()
    # Looks: a rounded housing, the aluminium top with four chamfered square wells, a dark trim line.
    top, housing, trim = _looks(stage)
    V.define(stage, BLOCK_PATH + '/HousingLook', V.rounded_box((sx, sy, base_h + 0.004), 0.004), housing,
             translate=(lc[0], lc[1], (base_h + 0.004) / 2 - 0.0005))
    # The top: pieces around four square openings (2 MOUTH_HALF), each opening lined dark down to the well floor.
    t0, t1 = base_h + 0.0005, sz
    hole = MOUTH_HALF
    pieces = _top_pieces(lc[0] - sx / 2 + 0.002, lc[0] + sx / 2 - 0.002, lc[1] - sy / 2 + 0.002, lc[1] + sy / 2 - 0.002,
                         hole)
    for i, (a, b, c, d) in enumerate(pieces):
        V.define(stage, f'{BLOCK_PATH}/TopLook{i}', V.prism(V.round_rect(b - a, d - c, 0.0008), t0, t1), top,
                 translate=((a + b) / 2, (c + d) / 2, 0.0))
    for w in range(1, 5):
        xy = well_xy(w)
        lx, ly = xy[0] - o[0], xy[1] - o[1]
        liner = V.frame_ring((2 * hole, 2 * hole), (2 * WELL_HALF, 2 * WELL_HALF), -(COLLAR_H - SEAT_FLOOR), 0.0,
                             r_out=0.0005, r_in=0.0005)
        V.define(stage, f'{BLOCK_PATH}/LinerLook{w}', liner, trim, translate=(lx, ly, sz - 0.0002))
        V.define(stage, f'{BLOCK_PATH}/FloorLook{w}', V.prism(V.round_rect(2 * WELL_HALF, 2 * WELL_HALF, 0.0005), 0.0, 0.0004),
                 trim, translate=(lx, ly, base_h + SEAT_FLOOR - 0.0004))
        # The well number in front of its mouth (robot side): between the rows, between the near row and the keys.
        _text(stage, f'{BLOCK_PATH}/WellNumber{w}', str(w), (lx, ly + hole + 0.007), sz + 0.0004, 0.007, C.INK)
    V.define(stage, BLOCK_PATH + '/TrimLook', V.prism(V.round_rect(sx + 0.001, 0.004, 0.001), 0.0, 0.002), trim,
             translate=(lc[0], lc[1] + sy / 2 - 0.002, base_h - 0.001))
    # Brand between the two columns, at the level of the far row (the numbers 3 and 4 stand between the rows).
    far_y = well_xy(1)[1] - o[1]
    _text(stage, BLOCK_PATH + '/Brand', 'DRY BLOCK', (lc[0], far_y - 0.004), sz + 0.0004, 0.006, C.INK)
    _text(stage, BLOCK_PATH + '/Range', '20-80 C', (lc[0], far_y + 0.006), sz + 0.0004, 0.005, C.INK)
    return seats


LOG_MARK = ('LOG', (0.006, 0.0062), 0.0045, 'left')     # in the window right of the SET digits, lit while logging


def build_controller(stage, path, *, x, front_y, looks_):
    """thermal_lcd.build_controller's screen (same face, rows, labels and window) with the LOG mark, a switchable
    label (thermal_lcd's toggles, as the recording thermometer's REC); thermal_lcd itself is shared and unchanged."""
    rows = L.controller_rows(CELLS, DECIMALS)
    right = -0.008 + max(L.row_width(r) for r in rows.values())
    labels = [('SET', (-0.041, 0.0090), 0.0050, 'left'), ('BLOCK', (-0.041, -0.0095), 0.0050, 'left'),
              ('C', (right + 0.007, 0.0090), 0.0050, 'left'), ('C', (right + 0.007, -0.0095), 0.0050, 'left')]
    left, margin = -0.012, 0.004
    L.build_panel(stage, path, centre=(x, front_y + L.CONTROLLER_DEPTH, L.CONTROLLER_Z), tilt_deg=90.0,
                  face=L.CONTROLLER_FACE, depth=L.CONTROLLER_DEPTH, rows=rows, labels=labels, toggles={'log': LOG_MARK},
                  glass=(((left + right + margin) / 2, -0.0010), (right + margin - left, 0.033)),
                  housing_colour=(0.10, 0.10, 0.11), looks_=looks_, collide=False)
    return rows


def controller_state(set_c, block_c, logging):
    return L.controller_state(set_c, block_c, CELLS, DECIMALS) | {'toggles': {'log': bool(logging)}}


def build(stage, config):
    """chem_apparatus.build (with the QUENCH line) plus the dry block; returns cuvettes, buttons, seats and panels."""
    with four_lines():
        built = C.build(stage, config)
    materials = {'rubber': _material(stage, '/World/PhysicsMaterials/ReactionRubber', 0.8, 0.7),
                 'seat': _material(stage, '/World/PhysicsMaterials/ReactionSeat', 0.20, 0.15)}
    UsdGeom.Xform.Define(stage, BLOCK_PATH)
    seats = _block(stage, 0.0, materials)
    o = C.ORIGIN_XY
    zb = BLOCK['size'][2]
    for k, (bid, text) in enumerate(BLOCK_KEYS):
        xy = key_xy(k)
        colour = (LOG_COLOUR if bid == 'block:log' else (0.80, 0.30, 0.10) if bid.startswith('block:up')
                  else (0.15, 0.35, 0.80))
        built['buttons'][bid] = _button(stage, bid, (xy[0] - o[0], xy[1] - o[1]), zb, 'round', colour, text,
                                        {'rubber': materials['rubber']})
    looks = L.looks(stage, BLOCK_PATH)
    (cx, cy), (sx, sy, sz) = BLOCK['centre'], BLOCK['size']
    rows = build_controller(stage, CONTROLLER_PATH, x=cx - o[0], front_y=cy - o[1] + sy / 2, looks_=looks)
    # C.build already ran finish_looks: a second run under the kit's Looks would number its materials from Matte000
    # again and recolour the rack, station and spectrophotometer (a dark rack would be drawn light blue).
    saved, C.LOOKS = C.LOOKS, BLOCK_PATH + '/Looks'
    try:
        C.finish_looks(stage)
    finally:
        C.LOOKS = saved
    built['seats'].update({k: str(v.GetPath()) for k, v in seats.items()})
    built['panels'] = {'block': (CONTROLLER_PATH, rows)}
    return built

"""Truss bench looks (render only): what the bench looks like in the cameras, after real workshop equipment. Built at
the end of truss_apparatus.build and never touched afterwards.

Each collider keeps its geometry, its collision and its place; it is only made invisible, and a look is authored beside
it (a sibling under the same parent, so the looks of a bar's main body and tail move with those bodies and part with
them when the bar breaks or kinks). A look stays within the collider it dresses (bounds of a body come from its looks:
BBoxCache skips invisible prims). Parts without a collider (render marks, the loading frame) are dressed the same way.

- bars: 3D-printed (the public masses, 0.92 / 1.08 / 1.25 / 1.42 g/cm3 on the 12 x 5 mm section, are printed
  plastics): matte blue, orange, green and white filament with 0.2 mm layer lines on the sides and a diagonal raster on the top, lightening slots in the
  web (render only: the collider is the solid strip; none across the cut beside the knob or under the length label),
  plain printed eyes with a glossier brim round the bore, a black printed grip knob with ribbed faces;
- pegboard and pin stand: black anodized tooling plate with cap screws, bright machined chamfers round the holes, dark
  bores; the pins polished steel (C red) with their collars (and the loose pins' shanks);
- racks: beech with beech dividers, dark printed labels; scrap tray: galvanized steel;
- tester: ground steel guide rail with a lead screw and its bearing blocks, the fixed head (load cell under a black
  oxide clamp) and the drive housing behind it, black oxide crosshead with an orange handle and a rubber grip; the
  console a powder-coated housing with a dark membrane panel, a bezel round the screen and a nameplate;
- loading frame (truss_frame): T-slot aluminium columns on black oxide foot pads with a top beam, machined sheaves on
  bearings, steel wire rope (including its wraps over the sheaves), a black oxide carriage with an aluminium load
  platform, rubber catch blocks, the load display in a housing with a bezel; the weight case (beech, rubber lined) and
  the stackable knob weights (polished steel 2 / 0.5 kg, brass 0.1 kg, their masses marked on top).
"""
from __future__ import annotations

import math
import os

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdShade

import chem_apparatus as C
from chem_apparatus import ROOT
import truss_apparatus as A
import truss_bench as B
import truss_frame as F
import truss_loading as L
import truss_model as M
import visual_kit as K

LOOKS = ROOT + '/TrussLooks'
BAR_CHAMFER = 0.0004
PRINT_TILE = 0.0128                  # UV tile of the printed textures: 64 layers of 0.2 mm
SLOT_WALL, SLOT_GAP, SLOT_R = 0.0025, 0.0025, 0.0015   # lightening slots: wall to each edge, between slots, corners
SLOT_LEN = (0.004, 0.010)            # a slot's length range; at most 3 in a stretch of web
BRIM_W = 0.0011
ROPE_R = 0.0008
ROD_R, SLOT_HALF, BORE_R = 0.0025, 0.0031, 0.0036    # hanger rod; the masses' slot half width and centre bore
SCREW = (1.258, 0.003, 0.005)        # lead screw: world x, radius, axis height
SCREW_Y = (-0.2215, -0.066)
DRIVE = ((1.213, -0.262), (1.278, -0.2205), 0.022)   # drive housing (x0, y0), (x1, y1), height


def enabled(config):
    return os.environ.get('EMBODIEDLAB_TRUSS_LOOKS', '1') != '0' and (config or {}).get('looks', True)


# --- materials ---------------------------------------------------------------------------------------------------
def _materials(stage, folder):
    UsdGeom.Scope.Define(stage, LOOKS)
    brushed = K.brushed(folder)
    oxide = K.grain(folder, 'oxide', 0.25, 0.42, 0.12, seed=6)
    anodized = K.grain(folder, 'anodized', 0.10, 0.40, 0.08, seed=7, blur=1)
    peel = K.orange_peel(folder)
    galv = K.spangle(folder)
    beech = K.wood(folder)
    knurl = K.knurl(folder)
    cast = K.orange_peel(folder, 'cast', strength=14.0, seed=8)

    def mat(name, colour, **kw):
        return K.material(stage, f'{LOOKS}/{name}', colour, **kw)

    layers, top = K.layer_lines(folder), K.raster(folder)
    m = {}
    for kind, colour in (('blue', (0.10, 0.33, 0.88)), ('orange', (0.98, 0.40, 0.06)), ('green', (0.08, 0.58, 0.22)),
                         ('white', (0.93, 0.93, 0.91))):
        name = kind.title()
        m[kind] = mat(f'Printed{name}', colour, roughness=0.66, diffuse=layers[0], normal=layers[1])
        m[kind + '_top'] = mat(f'Printed{name}Top', colour, roughness=0.62, diffuse=top[0], normal=top[1])
        m[kind + '_brim'] = mat(f'Printed{name}Brim', colour, roughness=0.35)
    m.update({
         'knurl': mat('PrintedBlackKnob', (0.045, 0.045, 0.05), roughness=0.62, normal=knurl),
         'plate': mat('BlackAnodizedPlate', (0.07, 0.075, 0.08), metallic=0.35, diffuse=anodized[0], rough_map=anodized[1]),
         'steel': mat('PolishedSteel', (0.82, 0.82, 0.84), metallic=1.0, roughness=0.14),
         'red_pin': mat('RedPin', (0.80, 0.06, 0.04), metallic=0.2, roughness=0.30, clearcoat=0.6),
         'chamfer': mat('ChamferBright', (0.62, 0.62, 0.64), metallic=1.0, roughness=0.22),
         'socket': mat('Socket', (0.02, 0.02, 0.02), roughness=0.8),
         'wood': mat('Beech', (1.0, 1.0, 1.0), diffuse=beech[0], rough_map=beech[1]),
         'wood_dark': mat('BeechDivider', (0.82, 0.74, 0.66), diffuse=beech[0], rough_map=beech[1]),
         'ink': mat('PrintInk', (0.03, 0.03, 0.03), roughness=0.85),
         'galv': mat('Galvanized', (0.84, 0.86, 0.88), metallic=0.85, diffuse=galv[0], rough_map=galv[1]),
         'alu': mat('Aluminium', (0.86, 0.87, 0.89), metallic=1.0, diffuse=brushed[0], rough_map=brushed[1]),
         'oxide': mat('BlackOxide', (0.10, 0.10, 0.11), metallic=0.8, diffuse=oxide[0], rough_map=oxide[1]),
         'ground': mat('GroundSteel', (0.70, 0.71, 0.73), metallic=1.0, diffuse=brushed[0], rough_map=brushed[1],
                       rough_scale=0.8),
         'powder': mat('PowderCoatGrey', (0.78, 0.78, 0.76), roughness=0.55, normal=peel),
         'membrane': mat('MembranePanel', (0.11, 0.12, 0.13), roughness=0.55),
         'handle': mat('OrangePowderCoat', (0.92, 0.33, 0.05), roughness=0.40, clearcoat=0.3, normal=peel),
         'rubber': mat('Rubber', (0.035, 0.035, 0.035), roughness=0.92),
         'plastic': mat('SatinBlack', (0.05, 0.05, 0.055), roughness=0.35),
         'grey_plastic': mat('GreyPlastic', (0.21, 0.22, 0.23), roughness=0.50),
         'yellow': mat('YellowPlastic', (0.95, 0.75, 0.05), roughness=0.30, clearcoat=0.4),
         'green': mat('GreenCap', (0.15, 0.62, 0.25), roughness=0.25, clearcoat=0.5),
         'chrome': mat('Chrome', (0.92, 0.92, 0.94), metallic=1.0, roughness=0.06),
         'cast_iron': mat('CastIron', (0.045, 0.045, 0.05), roughness=0.60, normal=cast),
         'brass': mat('Brass', (0.86, 0.66, 0.33), metallic=1.0, roughness=0.22),
         'wire': mat('WireRope', (0.74, 0.75, 0.77), metallic=1.0, roughness=0.35),
         'lcd': mat('LcdGlass', (0.01, 0.03, 0.02), specular=(0.0, 0.0, 0.0), roughness=1.0),
         'white_print': mat('WhitePrint', (0.93, 0.93, 0.90), roughness=0.7)})
    for name, lines, ink, bg in (('Nameplate', ['BAR TESTER'], (25, 25, 28), (205, 207, 210, 255)),
                                 ('LoadCellLabel', ['LOAD CELL'], (240, 240, 240), None),
                                 ('Mass2kg', ['2 kg'], (150, 150, 152), None),
                                 ('Mass05kg', ['0.5 kg'], (40, 30, 10), None),
                                 ('Mass01kg', ['0.1 kg'], (40, 30, 10), None)):
        path, aspect = K.decal(folder, name, lines, ink=ink, background=bg, bold=True)
        m[name] = (mat(name, (1, 1, 1), roughness=0.5, cutout=path), aspect)
    return m


def _material(stage, name):
    return UsdShade.Material.Get(stage, f'{LOOKS}/{name}')


# --- dressing helpers ------------------------------------------------------------------------------------------------
def _hide(stage, path):
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    return prim


def _pose(stage, path):
    t, r, s, _, _ = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(path)).GetXformVectors(Usd.TimeCode.Default())
    return tuple(t), tuple(r), tuple(s)


def _dress_box(stage, path, mat, radius, *, tile=0.05, shrink=0.0):
    """Hide a drawn box (Cube with translate, rotate, scale) and put a rounded box of its size beside it."""
    t, r, s = _pose(stage, path)
    _hide(stage, path)
    size = [max(v - 2 * shrink, 1e-5) for v in s]
    return K.define(stage, path + 'Look', K.rounded_box(size, radius), mat, tile=tile, translate=t,
                    rotate_xyz=r if any(r) else None)


def _bind(stage, path, mat):
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        K.bind(prim, mat)


def _pin_geo(h, tip, z0):
    """Dowel pin from z0: shaft (B.PIN_R) and the cone along the collider's to a small round end."""
    r = B.PIN_R
    end = 0.0006
    return K.lathe([(0.0, z0), (r, z0), (r, z0 + h), (end, z0 + h + tip * (1 - end / r)),
                    (end * 0.5, z0 + h + tip * (1 - end / r) + 0.0002), (0.0, z0 + h + tip * (1 - end / r) + 0.0003)], 32)


def _collared_geo(shank):
    """A board pin from its collar's underside (z = 0): shank (loose pins) with its pointed end, collar, post, tip;
    the outline of truss_apparatus._collared_pin's colliders."""
    end = 0.0006
    prof = [(0.0, 0.0)]
    if shank:
        straight = B.SHANK_L - B.SHANK_TIP
        prof = [(0.0, -straight - 2 * B.SHANK_TIP), (end, -straight - 2 * B.SHANK_TIP + end * 2 * B.SHANK_TIP / B.SHANK_R),
                (B.SHANK_R, -straight), (B.SHANK_R, 0.0)]
    foot = B.COLLAR_H
    tip = foot + B.PIN_H + B.PIN_TIP * (1 - end / B.PIN_R)
    prof += [(B.COLLAR_R, 0.0), (B.COLLAR_R, B.COLLAR_H), (B.PIN_R, foot), (B.PIN_R, foot + B.PIN_H), (end, tip),
             (end * 0.5, tip + 0.0002), (0.0, tip + 0.0003)]
    return K.lathe(prof, 40)


def _dress_collared(stage, prefix, look_path, mat, shank, translate=None):
    """Hide a collared pin's drawn parts (prefix + Collar/Post/Shank/Tip/ShankEnd) and put its lathe look at look_path."""
    for part in ('Collar', 'Post', 'Shank'):
        _hide(stage, f'{prefix}{part}/Visual')
    for part in ('Tip', 'ShankEnd'):
        _hide(stage, prefix + part)
    K.define(stage, look_path, _collared_geo(shank), mat, crease_deg=40, translate=translate)


def _dress_plate(stage, path, rect, holes, m):
    """Hide a holed plate's collider mesh and draw it in three looks: plate, bright chamfers, dark bores."""
    groups = []
    points, faces = B.holed_plate(rect, holes, B.BOARD_H, groups=groups)
    _hide(stage, path)
    for part, mat in (('plate', m['plate']), ('chamfer', m['chamfer']), ('bore', m['socket'])):
        K.define(stage, f'{path}Look{part.title()}', K.Geo(points, [f for f, g in zip(faces, groups) if g == part]), mat,
                 tile=0.08, crease_deg=30)


def _dress_pin(stage, path, h, mat):
    """A pin from truss_apparatus._pin (Xform with Visual and Collider at the shaft's centre, the tip a cone beside)."""
    _hide(stage, path + '/Visual')
    _hide(stage, path + 'Tip')
    K.define(stage, path + '/Look', _pin_geo(h, B.PIN_TIP, -h / 2), mat, crease_deg=40)


def _ring(r0, r1, z0, z1):
    return K.lathe([(r0, z0), (r1, z0), (r1, z1), (r0, z1)], 48, closed=True)


def _cylinder_y(radius, y0, y1, ridges=0):
    """Round bar along +y from y0 to y1 (ridges: a thread's crests, one per 2 mm)."""
    prof = [(0.0, 0.0), (radius, 0.0)]
    length = y1 - y0
    if ridges:
        n = int(length / 0.002)
        for k in range(n):
            prof += [(radius, (k + 0.25) * length / n), (radius * 0.85, (k + 0.5) * length / n),
                     (radius, (k + 0.75) * length / n)]
    prof += [(radius, length), (0.0, length)]
    return K.lathe(prof, 24).moved(K.rot_x(-90), (0.0, y0, 0.0))


def _decal(stage, path, m, name, height, centre, rotation=None):
    mat, aspect = m[name]
    geo, uv = K.quad(height * aspect, height)
    K.define(stage, path, geo.moved(rotation, centre), mat, uv=uv)


# --- the parts ---------------------------------------------------------------------------------------------------------
def _board(stage, m, config):
    _dress_plate(stage, ROOT + '/Board', B.board_rect(), [B.pin_xy(n) for n in M.LOOSE_HOLES], m)
    _dress_plate(stage, ROOT + '/PinStand', B.stand_rect(), list(B.stand_holes().values()), m)
    x0, y0, x1, y1 = B.board_rect()
    top = B.BOARD_H
    for k, (x, y) in enumerate(((x0 + 0.010, y0 + 0.010), (x1 - 0.010, y0 + 0.010), (x0 + 0.010, y1 - 0.010),
                                (x1 - 0.010, y1 - 0.010))):
        K.define(stage, f'{ROOT}/BoardScrew{k}', K.cap_screw(0.0040, 0.0004, top).moved(None, (x, y, 0)), m['steel'])
        K.define(stage, f'{ROOT}/BoardScrewSocket{k}', K.hex_prism(0.0030, 0.0001, top + 0.00035).moved(None, (x, y, 0)),
                 m['socket'])
    for name in M.FIXED:
        x, y = B.pin_xy(name)
        _dress_collared(stage, f'{ROOT}/Pin{name}', f'{ROOT}/Pin{name}Look',
                        m['red_pin'] if name == M.LOAD_PIN else m['steel'], False, translate=(x, y, top))
    for pin in config['pins']:
        path = A.pin_path(pin)
        _dress_collared(stage, path + '/', path + '/Look', m['steel'], True)
    for prim in list(stage.GetPrimAtPath(ROOT).GetChildren()):   # printed marks: paint, not plastic
        name = prim.GetName()
        if name.startswith(('Letter', 'Support', 'Roller', 'Ground', 'Arrow')) and prim.IsA(UsdGeom.Gprim):
            colour = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()[0]
            if colour[0] > 0.9 and colour[2] > 0.9:
                K.bind(prim, m['white_print'])


def _slots(lo, hi):
    """Lightening slots along a stretch of web from x = lo to hi: as many as fit (at most 3), evenly spaced."""
    span = hi - lo
    for n in (3, 2, 1):
        length = (span - (n - 1) * SLOT_GAP) / n
        if length >= SLOT_LEN[0]:
            length = min(length, SLOT_LEN[1])
            pitch = (span - n * length) / max(n - 1, 1) + length if n > 1 else 0.0
            x0 = lo + (span - (n * length + (n - 1) * (pitch - length))) / 2
            return [(x0 + k * pitch, x0 + k * pitch + length) for k in range(n)]
    return []


def bar_slots(stage, bar, label):
    """The slots of a bar in its frame: {'main': [(a, b)], 'tail': [...]}: in the web between the knob or the cut
    and the eye bosses, clear of the cut and the length label by SLOT_WALL."""
    half = M.BAR_LENGTHS_MM[label] / 2000.0
    end = half - B.EYE_R - 0.0015                      # the web ends short of the eye boss
    ext = UsdGeom.Mesh(stage.GetPrimAtPath(A.bar_path(bar) + '/Length')).GetExtentAttr().Get()
    pad = (float(ext[0][0]) - SLOT_WALL, float(ext[1][0]) + SLOT_WALL)
    main = _slots(B.KNOB[0] / 2 + SLOT_WALL, pad[0]) + _slots(pad[1], end)
    tail = _slots(-end, B.CUT_M - SLOT_WALL)
    return {'main': main, 'tail': tail}


def _bar_shape(label, e, slots):
    """A bar body's printed shape (its frame): the web with its slots, the eye piece beyond, split into the top
    faces and the rest; and the brim round the bore."""
    t, w, c = B.BAR_T, B.BAR_W / 2, BAR_CHAMFER
    s = M.BAR_LENGTHS_MM[label] / 1000.0
    ex = s / 2 if e else -s / 2
    hh = w - SLOT_WALL
    holes = [(a, b, hh, SLOT_R) for a, b in slots]
    if not holes:
        whole = K.baked(K.eye_link(ex, B.CUT_M, w, B.EYE_R, B.EYE_HOLE_R, t, c))
    elif e:                                            # main: cut .. web .. xm | eye piece
        xm = holes[-1][1] + SLOT_WALL
        whole = K.merge(K.baked(K.web(B.CUT_M, xm, w, t, c, holes)),
                        K.baked(K.eye_link(ex, xm, w, B.EYE_R, B.EYE_HOLE_R, t, c)))
    else:                                              # tail: eye piece | xm .. web .. cut
        xm = holes[0][0] - SLOT_WALL
        whole = K.merge(K.baked(K.eye_link(ex, xm, w, B.EYE_R, B.EYE_HOLE_R, t, c)),
                        K.baked(K.web(xm, B.CUT_M, w, t, c, holes)))
    top, rest = K.split(whole, lambda n: n[2] > 0.9)
    brim = K.baked(_ring(B.EYE_HOLE_R, B.EYE_HOLE_R + BRIM_W, t, t + 0.00004).moved(None, (ex, 0, 0)))
    return top, rest, brim


def _bars(stage, config, m):
    t = B.BAR_T
    shapes = {}                                        # (label, eye, slots) -> the body's shape
    for bar, spec in config['bars'].items():
        slots = bar_slots(stage, bar, spec['label'])
        kind = spec['kind']
        for body, e, side in ((bar, 1, 'main'), (B.tail_id(bar), 0, 'tail')):
            path = A.bar_path(body)
            for child in ('Strip', f'Eye{e}', 'Knob'):
                _hide(stage, f'{path}/{child}')
            key = (spec['label'], e, tuple(slots[side]))
            if key not in shapes:
                shapes[key] = _bar_shape(spec['label'], e, slots[side])
            top, rest, brim = shapes[key]
            K.define(stage, path + '/Look', rest, m[kind], tile=PRINT_TILE)
            K.define(stage, path + '/TopLook', top, m[kind + '_top'], tile=PRINT_TILE)
            K.define(stage, path + '/Brim', brim, m[kind + '_brim'], tile=PRINT_TILE)
            if body == bar:
                kx, ky, kz = B.KNOB
                K.define(stage, path + '/KnobLook', K.rounded_box((kx, ky, kz), 0.0015).moved(None, (0, 0, t + kz / 2)),
                         m['knurl'], tile=0.01)
                _bind(stage, path + '/Length', m['ink'] if kind == 'white' else m['white_print'])   # dark on white


def _rack(stage, m):
    for name in B.RACKS:
        _dress_box(stage, f'{ROOT}/Rack{name.title()}', m['wood'], 0.0015, tile=0.12)
    for prim in list(stage.GetPrimAtPath(ROOT).GetChildren()):
        name = prim.GetName()
        if name.startswith('RackRib'):
            _dress_box(stage, str(prim.GetPath()), m['wood_dark'], 0.0006, tile=0.12)
        elif name.startswith('RackLabel'):
            K.bind(prim, m['ink'])


def _tray(stage, m):
    for name in ('TrayFloor', 'TrayWallN', 'TrayWallS', 'TrayWallW', 'TrayWallE'):
        _dress_box(stage, f'{ROOT}/{name}', m['galv'], 0.0008, tile=0.10)
    _bind(stage, ROOT + '/TrayLabel', m['white_print'])


def _tester(stage, m):
    fx, fy = B.tester_pins(0.0)['fixed']
    (px, py, _), (cx, cy, _) = A.bed_parts()['Pedestal']
    top = B.TESTER_TOP
    _hide(stage, ROOT + '/Pedestal')
    split = top * 0.68
    K.define(stage, ROOT + '/PedestalLoadCell', K.rounded_box((px - 0.001, py - 0.001, split - 0.0003), 0.0012)
             .moved(None, (cx, cy, split / 2)), m['alu'], tile=0.03)
    K.define(stage, ROOT + '/PedestalClamp', K.rounded_box((px, py, top - split), 0.0012)
             .moved(None, (cx, cy, split + (top - split) / 2)), m['oxide'], tile=0.03)
    _decal(stage, ROOT + '/PedestalLabel', m, 'LoadCellLabel', 0.0045, (cx, cy + (py - 0.001) / 2 + 0.0002, split / 2),
           np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]))
    _dress_pin(stage, ROOT + '/TesterPinFixed', B.TESTER_PIN_H, m['steel'])
    for name in ('TesterBedL', 'TesterBedR', 'TesterBedEnd'):
        _dress_box(stage, f'{ROOT}/{name}', m['ground'], 0.0006, tile=0.06)
    # the lead screw beside the rail (under the crosshead's tongue), its bearing blocks, the drive behind the head
    x, r, z = SCREW
    K.define(stage, ROOT + '/LeadScrew', _cylinder_y(r, *SCREW_Y, ridges=True).moved(None, (x, 0, z)), m['ground'])
    for k, y in enumerate((-0.181, -0.066)):
        K.define(stage, f'{ROOT}/ScrewBearing{k}', K.rounded_box((0.012, 0.008, 0.009), 0.001)
                 .moved(None, (x, y, 0.0045)), m['oxide'], tile=0.03)
    (x0, y0), (x1, y1), h = DRIVE
    K.define(stage, ROOT + '/DriveHousing', K.rounded_box((x1 - x0, y1 - y0, h), 0.003)
             .moved(None, ((x0 + x1) / 2, (y0 + y1) / 2, h / 2)), m['powder'], tile=0.05)
    K.define(stage, ROOT + '/LoadCellCable', K.tube([(fx + px / 2 - 0.001, fy - 0.004, 0.006), (fx + px / 2 + 0.006, fy - 0.006, 0.004),
                                                     (fx + px / 2 + 0.008, y1 + 0.004, 0.003), (fx + px / 2 + 0.008, y1 - 0.002, 0.006)],
                                                    0.0012, 10), m['rubber'])
    # the crosshead (dynamic: its looks are its children)
    for part in ('Web', 'Tongue'):
        _dress_box(stage, f'{A.CARRIAGE}/{part}', m['oxide'], 0.0010, tile=0.03)
    _dress_pin(stage, A.CARRIAGE + '/Pin', B.TESTER_PIN_H, m['steel'])
    t, _, s = _pose(stage, A.CARRIAGE + '/Handle')
    _hide(stage, A.CARRIAGE + '/Handle')
    grip = 0.025
    z0 = t[2] - s[2] / 2                                           # the handle's foot (on the tongue)
    K.define(stage, A.CARRIAGE + '/HandleLook', K.rounded_box((s[0], s[1], s[2] - grip), 0.0015)
             .moved(None, (t[0], t[1], z0 + (s[2] - grip) / 2)), m['handle'], tile=0.03)
    K.define(stage, A.CARRIAGE + '/HandleGrip', K.rounded_box((s[0], s[1], grip), 0.0025)
             .moved(None, (t[0], t[1], z0 + s[2] - grip / 2)), m['rubber'], tile=0.02)


def _console(stage, m):
    cx, cy = B.CONSOLE_XY
    sx, sy, sz = A.CONSOLE['size']
    _dress_box(stage, ROOT + '/Console', m['powder'], 0.004, tile=0.08)
    K.define(stage, ROOT + '/ConsolePanel', K.prism(K.round_rect(sx - 0.010, sy - 0.010, 0.003), sz, sz + 0.0002)
             .moved(None, (cx, cy, 0)), m['membrane'])
    gx, gy = C._xy(B.CONSOLE_XY, 0.0, A.DISPLAY_V)
    gw, gh = A.GLASS
    K.define(stage, ROOT + '/ConsoleBezel', K.frame_ring((gw + 0.008, gh + 0.008), (gw + 0.0006, gh + 0.0006), sz, sz + 0.0009)
             .moved(None, (gx, gy, 0)), m['plastic'])
    _decal(stage, ROOT + '/Nameplate', m, 'Nameplate', 0.010, (cx, cy + sy / 2 + 0.0003, sz / 2),
           np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]))
    _dress_box(stage, A.SLIDER_PATH + '/Block', m['plastic'], 0.0015, tile=0.03)
    _hide(stage, A.SLIDER_PATH + '/Post/Visual')
    r, h = A.SLIDER_POST
    K.define(stage, A.SLIDER_PATH + '/Post/Look', K.lathe([(0, -h / 2), (r, -h / 2), (r, h / 2 - 0.001), (r - 0.001, h / 2),
                                                          (0, h / 2)], 32), m['yellow'])
    start = C.button_path(B.START_ID)
    for part in ('/Cap/Visual', '/Pad/Visual'):
        _bind(stage, start + part, m['green'])
    _bind(stage, start + '_Housing/Visual', m['chrome'])


def _sheave(radius, thickness, bore):
    """A grooved sheave about z (rope centre on radius): flanges 1.5 mm beyond, the groove 0.8 mm inside."""
    T = thickness / 2
    R = radius + 0.0015
    return K.lathe([(bore, -T), (R, -T), (R, -T * 0.45), (radius - ROPE_R, 0.0), (R, T * 0.45), (R, T), (bore, T)], 64,
                   closed=True)


def _arc(centre_yz, radius, a0, a1, n=16):
    yc, zc = centre_yz
    return [(F.CX, yc + radius * math.cos(a), zc + radius * math.sin(a)) for a in np.linspace(a0, a1, n)]


def weight_geo(r, h):
    """A stackable knob weight about its bottom centre: the body with chamfered edges, the neck and the head of the
    knob (the recess underneath is not seen)."""
    c = min(0.002, r * 0.08)
    n, k = F.NECK_R, F.HEAD_R
    z1, z2, z3 = h + F.NECK_H, h + F.NECK_H + F.HEAD_H, h + F.KNOB_H
    return K.lathe([(0.0, 0.0), (r - c, 0.0), (r, c), (r, h - c), (r - c, h), (n + 0.0006, h), (n, h + 0.0006),
                    (n, z1), (k - 0.0006, z1), (k, z1 + 0.0006), (k, z2), (0.0012, z3 - 0.0002), (0.0, z3)], 64)


def _loading(stage, m):
    """The loading frame of truss_frame: T-slot columns on foot pads, top beam, sheaves, wire rope, the carriage with
    its platform, the catch blocks, the load display, the weight case and the weights."""
    P = F.FRAME
    for k in range(2):
        _dress_box(stage, f'{P}/Foot{k}', m['oxide'], 0.001, tile=0.04)
        t, _, s = _pose(stage, f'{P}/Column{k}')
        _hide(stage, f'{P}/Column{k}')
        K.define(stage, f'{P}/Column{k}Look', K.tslot(s[2]).moved(None, (t[0], t[1], t[2] - s[2] / 2)), m['alu'],
                 tile=0.05)
    _dress_box(stage, P + '/Beam', m['alu'], 0.001, tile=0.05)
    for k in range(2):
        _dress_box(stage, f'{P}/FootCheek{k}', m['oxide'], 0.0005, tile=0.03)
    _hide(stage, P + '/TopAxle')
    t, _, s = _pose(stage, P + '/TopAxle')
    K.define(stage, P + '/TopAxleLook', K.lathe([(0, -s[0] / 2), (0.004, -s[0] / 2), (0.004, s[0] / 2), (0, s[0] / 2)], 24)
             .moved(K.rot_y(90), t), m['chrome'])
    for wheel, radius, thick in (('FootSheave', F.R_FOOT, 0.008), ('TopSheave', F.R_TOP, 0.010)):
        path = f'{P}/{wheel}'
        _hide(stage, path + '/Disc/Visual')
        _hide(stage, path + '/Spoke')
        to_x = K.rot_y(90)
        K.define(stage, path + '/Sheave', _sheave(radius, thick, radius * 0.35).moved(to_x), m['alu'], tile=0.03)
        K.define(stage, path + '/Bearing', _ring(radius * 0.12, radius * 0.35, -thick / 2 - 0.0005, thick / 2 + 0.0005)
                 .moved(to_x), m['chrome'])
    K.define(stage, P + '/RopeWrapFoot', K.tube(_arc(F.FOOT_SHEAVE_YZ, F.R_FOOT, -math.pi / 2, -math.pi), ROPE_R, 10),
             m['wire'])
    K.define(stage, P + '/RopeWrapTop', K.tube(_arc((F.TOP_SHEAVE_Y, F.TOP_Z), F.R_TOP, math.pi, 0.0, 24), ROPE_R, 10),
             m['wire'])
    for k in range(3):
        _bind(stage, f'{P}/Rope{k}', m['wire'])
    # the carriage: black oxide crossbar and shoes, the rope's eye, the aluminium platform and rim
    C_ = F.CARRIAGE
    _dress_box(stage, C_ + '/Crossbar', m['oxide'], 0.0015, tile=0.04)
    for k in range(2):
        for j in range(2):
            if stage.GetPrimAtPath(f'{C_}/Shoe{k}{j}'):
                _dress_box(stage, f'{C_}/Shoe{k}{j}', m['oxide'], 0.0008, tile=0.03)
    _dress_box(stage, C_ + '/Eye', m['steel'], 0.001, tile=0.02)
    _dress_box(stage, C_ + '/Plate', m['alu'], 0.0015, tile=0.08)
    for part in ('RimFront', 'RimSide0', 'RimSide1'):
        _dress_box(stage, f'{C_}/{part}', m['alu'], 0.001, tile=0.05)
    for k in range(2):
        _dress_box(stage, f'{P}/Catch{k}', m['rubber'], 0.002, tile=0.04)
    # the load display
    t, _, s = _pose(stage, P + '/DisplayPanel')
    _hide(stage, P + '/DisplayPanel')
    K.define(stage, P + '/DisplayHousing', K.rounded_box((s[0], s[1], s[2] - 0.0004), 0.003)
             .moved(None, (t[0], t[1], (s[2] - 0.0004) / 2)), m['grey_plastic'])
    K.define(stage, P + '/DisplayGlass', K.prism(K.round_rect(s[0] - 0.018, s[1] - 0.014, 0.001), s[2] - 0.0004, s[2] - 0.0002)
             .moved(None, (t[0], t[1], 0)), m['lcd'])
    K.define(stage, P + '/DisplayBezel', K.frame_ring((s[0] - 0.008, s[1] - 0.006), (s[0] - 0.018, s[1] - 0.014), s[2] - 0.0004,
                                                     s[2] + 0.0002).moved(None, (t[0], t[1], 0)), m['plastic'])
    # the weight case: a beech tray lined with black rubber
    _dress_box(stage, P + '/CaseFloor', m['rubber'], 0.001, tile=0.06)
    for k in range(4):
        _dress_box(stage, f'{P}/CaseWall{k}', m['wood'], 0.0015, tile=0.08)
    # the weights
    labels = {2.0: 'Mass2kg', 0.5: 'Mass05kg', 0.1: 'Mass01kg'}
    for key in F.weight_ids():
        kg = F.weight_kg(key)
        r, h, _, _ = F.WEIGHT_KINDS[kg]
        path = F.weight_path(key)
        for part in ('Body/Visual', 'Neck/Visual', 'Head/Visual', 'Cap'):
            _hide(stage, f'{path}/{part}')
        for k in range(F.RECESS_PIECES):
            for piece in ('Skirt', 'Foot', 'Chamfer'):
                _hide(stage, f'{path}/{piece}{k:02d}')
        K.define(stage, path + '/Look', weight_geo(r, h), m['brass'] if kg < 0.3 else m['steel'], tile=0.05,
                 crease_deg=40)
        label, aspect = m[labels[kg]]
        height = 0.3 * (r - F.HEAD_R)
        geo, uv = K.quad(height * float(aspect), height)
        K.define(stage, path + '/Label', geo.moved(None, (0, (r + F.HEAD_R) / 2, h + 0.0002)), label, uv=uv)


def slotted_mass_geo(r, h):
    """A slotted mass about the origin (thickness h): disc of radius r, centre bore, slot to it toward -y."""
    w = SLOT_HALF
    ro, rc = math.sqrt(r * r - w * w), math.sqrt(BORE_R * BORE_R - w * w)
    a0, a1 = math.atan2(-ro, w), math.atan2(-ro, -w) + 2 * math.pi
    outer = [(r * math.cos(a), r * math.sin(a)) for a in np.linspace(a0, a1, 72)]
    b0, b1 = math.atan2(-rc, -w) + 2 * math.pi, math.atan2(-rc, w)
    bore = [(BORE_R * math.cos(a), BORE_R * math.sin(a)) for a in np.linspace(b0, b1, 24)]
    return K.prism(outer + bore, -h / 2, h / 2)


def slotted_mass(stage, path, kg, r, h, z):
    """A slotted mass the loading puts on the hanger (an Xform at path, z its centre on the hanger)."""
    xform = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(0, 0, z))
    body = 'CastIron' if kg >= 1.0 else 'Brass'
    K.define(stage, path + '/Look', slotted_mass_geo(r, h), _material(stage, body), tile=0.05, crease_deg=40)
    name = {2.0: 'Mass2kg', 0.5: 'Mass05kg', 0.1: 'Mass01kg'}[kg]
    label = _material(stage, name)
    height = 0.22 * r
    geo, uv = K.quad(height * float(label.GetPrim().GetCustomDataByKey('aspect') or 2.5), height)
    K.define(stage, path + '/Label', geo.moved(None, (0, r * 0.55, h / 2 + 0.0001)), label, uv=uv)
    return xform


# --- apply -------------------------------------------------------------------------------------------------------------
def apply(stage, config):
    """Dress the built bench (config as truss_apparatus.build's). No-op when the looks are off."""
    if not enabled(config):
        return False
    m = _materials(stage, K.texture_dir(config))
    for name in ('Mass2kg', 'Mass05kg', 'Mass01kg'):
        m[name][0].GetPrim().SetCustomDataByKey('aspect', float(m[name][1]))
    _board(stage, m, config)
    _bars(stage, config, m)
    _rack(stage, m)
    _tray(stage, m)
    _tester(stage, m)
    _console(stage, m)
    _loading(stage, m)
    _loading_pins(stage, m)
    return True


def _loading_pins(stage, m):
    """The loading's free pins for B and C (hidden until the loading; they stand in for the board's pins)."""
    for name in M.FIXED:
        if name == 'A':
            continue
        path = f'{L.PINS}/Pin{name}'
        _dress_collared(stage, path + '/', path + '/Look', m['red_pin'] if name == M.LOAD_PIN else m['steel'], False)
    for k in range(M.N_LOOSE_PINS):
        path = L.loose_twin(k)
        _dress_collared(stage, path + '/', path + '/Look', m['steel'], True)

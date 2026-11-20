"""Vibration absorber bench looks (render only), after real lab equipment. Built at the end of absorber_apparatus.build
and never touched afterwards (moving parts carry their looks: a look is a sibling of the collider it dresses, under the
same body).

Each drawn box keeps its geometry, its collision and its place; it is made invisible and a rounded box of its size is
authored beside it with the part's material (a look stays within the collider it dresses). Cylinders keep their smooth
render mesh with the part's material, or get a shaped look within their collider (the exciter's finned body, the
accelerometer's hexagon). Flat render-only prims (LCD segments, screen glass, printed marks) keep their own materials.

- vise: cast iron, the strip's clamped root and the spindle bright steel;
- strip: ground steel; blade: blued spring steel; stinger: steel rod on an aluminium armature;
- absorber: black-oxide mount, cheeks and yoke, black rubber stop linings, chrome piston rod, red anodized damper
  cylinder under a black valve head; carrier and tray natural (off-white) POM, steel bumper pins, the orange push
  handle; the hold-downs' slides clear-anodized aluminium with red push tabs;
- exciter: cast base and trunnions, blue powder-coated finned body; amplifier and analyzer grey powder coat, a bezel
  round the analyzer's screen;
- weights: polished brass with the kit's white grip; rack black rubber, its seats dark plastic.
"""
from __future__ import annotations

import math
import os

from pxr import Usd, UsdGeom

import absorber_bench as B
from chem_apparatus import ROOT
import visual_kit as K

LOOKS = ROOT + '/AbsorberLooks'
FLAT_M = 0.0008                     # boxes thinner than this are marks, glass or segments: not dressed


def enabled(config):
    return os.environ.get('EMBODIEDLAB_ABSORBER_LOOKS', '1') != '0' and (config or {}).get('looks', True)


def _materials(stage, folder):
    UsdGeom.Scope.Define(stage, LOOKS)
    brushed = K.brushed(folder)
    anodized = K.grain(folder, 'anodized', 0.10, 0.40, 0.08, seed=7, blur=1)
    oxide = K.grain(folder, 'oxide', 0.25, 0.42, 0.12, seed=6)
    peel = K.orange_peel(folder)
    cast = K.orange_peel(folder, 'cast', strength=14.0, seed=8)

    def mat(name, colour, **kw):
        return K.material(stage, f'{LOOKS}/{name}', colour, **kw)

    return {
        'cast_iron': mat('CastIron', (0.16, 0.20, 0.24), roughness=0.62, normal=cast),
        'ground': mat('GroundSteel', (0.30, 0.31, 0.33), metallic=0.9, diffuse=brushed[0], rough_map=brushed[1],
                      rough_scale=2.0),
        'steel': mat('PolishedSteel', (0.82, 0.82, 0.84), metallic=1.0, roughness=0.15),
        'chrome': mat('Chrome', (0.92, 0.92, 0.94), metallic=1.0, roughness=0.06),
        'blued': mat('BluedSpringSteel', (0.20, 0.24, 0.34), metallic=1.0, roughness=0.28),
        'oxide': mat('BlackOxide', (0.10, 0.10, 0.11), metallic=0.8, diffuse=oxide[0], rough_map=oxide[1]),
        'alu': mat('BlackAnodizedAluminium', (0.07, 0.075, 0.08), metallic=0.35, diffuse=anodized[0],
                   rough_map=anodized[1]),
        'brushed_alu': mat('BrushedAluminium', (0.86, 0.87, 0.89), metallic=1.0, diffuse=brushed[0],
                           rough_map=brushed[1]),
        'red': mat('RedAnodized', (0.62, 0.06, 0.05), metallic=0.7, roughness=0.30),
        'rubber': mat('Rubber', (0.035, 0.035, 0.035), roughness=0.92),
        'handle': mat('OrangePowderCoat', (0.92, 0.33, 0.05), roughness=0.40, clearcoat=0.3, normal=peel),
        'blue_powder': mat('BluePowderCoat', (0.08, 0.20, 0.42), roughness=0.45, normal=peel),
        'powder': mat('PowderCoatGrey', (0.78, 0.78, 0.76), roughness=0.55, normal=peel),
        'dark_powder': mat('PowderCoatDark', (0.22, 0.23, 0.25), roughness=0.55, normal=peel),
        'brass': mat('Brass', (0.86, 0.66, 0.33), metallic=1.0, roughness=0.22),
        'plastic': mat('SatinBlack', (0.05, 0.05, 0.055), roughness=0.35),
        'pom': mat('NaturalPOM', (0.86, 0.85, 0.80), roughness=0.45),
        'seat': mat('SeatPlastic', (0.05, 0.12, 0.14), roughness=0.55),
    }


def _hide(prim):
    UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)


def _dress_box(stage, prim, mat, radius, tile=0.05):
    t, r, s, _, _ = UsdGeom.XformCommonAPI(prim).GetXformVectors(Usd.TimeCode.Default())
    size = [max(float(v), 1e-5) for v in s]
    _hide(prim)
    rad = min(radius, 0.25 * min(size))
    if rad > 0:
        geo = K.rounded_box(size, rad)
    else:                               # sharp edges (strip and blade pieces butt against their neighbours)
        hx, hy, hz = (v / 2 for v in size)
        geo = K.prism([(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)], -hz, hz)
    return K.define(stage, str(prim.GetPath()) + 'Look', geo, mat, tile=tile,
                    translate=tuple(t), rotate_xyz=tuple(r) if any(r) else None)


def _box_style(path, name):
    """(material key, corner radius) of a drawn box, or None to leave it."""
    rel = path[len(ROOT) + 1:]
    if rel.startswith('Strip/'):
        return 'ground', 0.0
    if rel.startswith('Blade/'):
        return 'blued', 0.0
    if rel.startswith('HoldDown'):
        return ('red', 0.0008) if name == 'Tab' else ('brushed_alu', 0.0 if name.startswith(('Ramp', 'Nose')) else 0.0005)
    if rel.startswith('Weight'):
        return ('brass', 0.0008) if name == 'Foot' else None
    if rel.startswith('Carrier/'):
        if name == 'Handle':
            return 'handle', 0.002
        if name.startswith('Chamfer'):
            return 'pom', 0.0
        return 'pom', 0.0006
    if rel.startswith('Mount/'):
        if name.startswith('Stop'):
            return 'rubber', 0.0003
        if name == 'BladeRoot':
            return 'blued', 0.0
        return 'oxide', 0.0008
    if rel.startswith('Yoke/'):
        return 'oxide', 0.0006
    if rel.startswith('Stinger/'):
        return 'oxide', 0.0005
    if name.startswith('Vise'):
        return 'cast_iron', 0.003 if name == 'ViseBase' else 0.002
    if name == 'StripRoot':
        return 'ground', 0.0
    if name.startswith('Shaker'):
        return 'cast_iron', 0.002
    if name == 'Amplifier':
        return 'dark_powder', 0.004
    if name == 'Analyzer':
        return 'powder', 0.004
    if name == 'Rack':
        return 'rubber', 0.003
    return None


def _cylinder_style(path, name):
    rel = path[len(ROOT) + 1:]
    if rel.startswith('Weight'):
        return None if name.startswith('Handle') else 'brass'
    if name in ('ViseSpindle', 'ViseHandle') or name.startswith('ViseBolt'):
        return 'steel'
    if name == 'DamperCylinder':
        return 'red'
    if rel.startswith('Yoke/') and name == 'Rod':
        return 'chrome'
    if rel.startswith('Stinger/'):
        return 'brushed_alu' if name == 'Armature' else 'steel'
    if rel.startswith('Carrier/') and name.startswith('Pin'):
        return 'steel'
    if name.endswith('Housing'):
        return 'plastic'
    return None


def _finned_body(stage, path, r, h, mat, fins=7):
    """The exciter's body: a drum with cooling fins (grooves cut into it), end caps, within its cylinder collider."""
    groove, cap = r - 0.004, 0.008
    zs = [-h / 2 + cap + k * (h - 2 * cap) / (2 * fins - 1) for k in range(2 * fins)]
    prof = [(0.0, -h / 2), (r - 0.002, -h / 2), (r, -h / 2 + 0.002)]
    for k, z in enumerate(zs):
        rr = r if k % 2 == 0 else groove
        prof += [(rr, z), (rr, z + (h - 2 * cap) / (2 * fins - 1))]
    prof += [(r, h / 2 - 0.002), (r - 0.002, h / 2), (0.0, h / 2)]
    clean = [prof[0]]
    for p in prof[1:]:
        if abs(p[0] - clean[-1][0]) > 1e-9 or abs(p[1] - clean[-1][1]) > 1e-9:
            clean.append(p)
    return K.define(stage, path + '/Look', K.lathe(clean, 64), mat, crease_deg=30.0)


def apply(stage, config=None):
    if not enabled(config):
        return
    m = _materials(stage, K.texture_dir(config))
    prims = list(Usd.PrimRange(stage.GetPrimAtPath(ROOT)))
    for prim in prims:
        path, name = str(prim.GetPath()), prim.GetName()
        if path.startswith(LOOKS) or '/Looks' in path:
            continue
        if prim.IsA(UsdGeom.Cube):
            _, _, s, _, _ = UsdGeom.XformCommonAPI(prim).GetXformVectors(Usd.TimeCode.Default())
            if min(float(v) for v in s) < FLAT_M:
                continue
            style = _box_style(path, name)
            if style is not None:
                _dress_box(stage, prim, m[style[0]], style[1])
        elif prim.IsA(UsdGeom.Xform) and prim.GetChild('Visual'):
            visual = prim.GetChild('Visual')
            if name == 'ShakerBody':
                _hide(visual)
                _finned_body(stage, path, B.SHAKER['r'], B.SHAKER['z'][1] - B.SHAKER['z'][0], m['blue_powder'])
            elif name == 'Accel':
                _hide(visual)
                af, ah, _ = B.ACCEL
                K.define(stage, path + '/Look', K.hex_prism(af * math.cos(math.pi / 6), ah, -ah / 2), m['brushed_alu'])
            else:
                style = _cylinder_style(path, name)
                if style is not None:
                    K.bind(visual, m[style])
        elif prim.IsA(UsdGeom.Mesh) and name.startswith('RackSeat'):
            K.bind(prim, m['seat'])
    # the analyzer's bezel round its screen (0.8 mm proud of the case top)
    (ax, ay), (sx, sy, sz) = B.ANALYZER['xy'], B.ANALYZER['size']
    gx, gy = ax - B.SCREEN_UV[0], ay - B.SCREEN_UV[1]
    gw, gh = B.SCREEN
    K.define(stage, ROOT + '/AnalyzerBezel', K.frame_ring((gw + 0.010, gh + 0.010), (gw + 0.001, gh + 0.001), sz,
                                                          sz + 0.0008), m['plastic'], translate=(gx, gy, 0.0))

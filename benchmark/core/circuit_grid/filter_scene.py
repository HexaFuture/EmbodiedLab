"""active_filter scene (Isaac; render looks from visual_kit): the 7 x 6 board with its ground rail, the
frequency-response analyzer (spring-cap keypad, screen with fields, lamps and the gain trace, output leads to the IN
pad and the rail), two trays (A: resistor and capacitor pieces; B: op-amp modules in long wells, wires, the probe) and
every block in its well.

Colliders are the platform's (asset.build_block, spring caps, housings, tray and board meshes); looks dress them
within their bounds (rounded handles of the resistor, capacitor and probe pieces: the handle box stays as the
invisible collider). Render-only: printed text and bands, the rail copper print, the screen, the leads (cables) and
their lugs on the board margin.
"""
from __future__ import annotations

import math

import numpy as np

from . import filter as F
from . import layout as L

TRACE = '/Analyzer/Trace'
SCREEN_DB = (10.0, -90.0)                 # gain at the plot's top and bottom edge
SCREEN_DECADES = (1, 5)                   # log10 f at the plot's left and right edge (10 Hz .. 100 kHz)
PLOT = (-0.058, 0.070, -0.062, 0.022)     # u0, u1, v0, v1 of the plot in the screen frame
FIELD_DIGIT = dict(digit_w=0.0062, digit_h=0.011, stroke=0.0013, pitch=0.0086)
FIELDS = {'Start': (-0.040, 0.068, 6), 'Stop': (-0.040, 0.050, 6), 'Pts': (-0.040, 0.032, 3),
          'No': (0.020, 0.032, 3), 'Entry': (-0.040, -0.087, 6)}       # u0, v0, digits
LAMPS = {'Run': ((0.050, 0.0735), (0.20, 1.0, 0.30)), 'Err': ((0.050, 0.0555), (1.0, 0.60, 0.05))}
ANALYZER_COLOUR = (0.60, 0.62, 0.64)
BEZEL = (0.10, 0.10, 0.11)
GRID = (0.09, 0.26, 0.11)
RESISTOR_BANDS = {0: (0.02, 0.02, 0.02), 1: (0.45, 0.22, 0.06), 2: (0.85, 0.08, 0.06), 3: (0.95, 0.45, 0.05),
                  4: (0.95, 0.85, 0.10), 5: (0.10, 0.60, 0.15), 6: (0.10, 0.25, 0.85), 7: (0.55, 0.15, 0.70),
                  8: (0.55, 0.55, 0.55), 9: (0.95, 0.95, 0.95)}


def screen_frame():
    from .display import Frame
    from . import asset as A
    (x0, x1), (y0, y1) = F.ANALYZER_SCREEN
    return Frame(((x0 + x1) / 2, (y0 + y1) / 2), A.READ, A.UP, F.analyzer_top_z())


def screen_half():
    (x0, x1), (y0, y1) = F.ANALYZER_SCREEN
    return (y1 - y0) / 2, (x1 - x0) / 2


def plot_uv(f_hz, g_db):
    u0, u1, v0, v1 = PLOT
    lf = (math.log10(max(f_hz, 1e-9)) - SCREEN_DECADES[0]) / (SCREEN_DECADES[1] - SCREEN_DECADES[0])
    g = min(SCREEN_DB[0], max(SCREEN_DB[1], g_db))
    return u0 + (u1 - u0) * min(1.0, max(0.0, lf)), v0 + (v1 - v0) * (g - SCREEN_DB[1]) / (SCREEN_DB[0] - SCREEN_DB[1])


def resistor_bands(value_ohm):
    """Four colour bands: two digits, multiplier, tolerance (brown, 1 %)."""
    digits = f'{value_ohm:.2e}'
    mant, exp = digits.split('e')
    d1, d2 = int(mant[0]), int(mant[2])
    mult = int(exp) - 1
    return [d1, d2, mult, 1]


# --- board -----------------------------------------------------------------------------------------------------------
def dress_board(stage):
    """The ground rail's printed copper between the far row's pads, GND and IN printed on the margin."""
    from . import asset as A
    from .display import Frame, quad_mesh, text
    world = Frame((0, 0), (1, 0), (0, 1), L.BOARD_TOP_Z)
    rects = []
    for (a, b) in F.rail_edges():
        (xa, ya), (xb, _) = L.node_xy(a), L.node_xy(b)
        rects.append((xa + L.PAD_HALF - 0.0005, ya - 0.0045, xb - L.PAD_HALF + 0.0005, ya + 0.0045))
    quad_mesh(stage, A.ROOT + '/Board/RailCopper', world, rects, A.BRASS, 0.0002)
    (x0, _), _ = L.board_extent()
    for name, node, colour in (('In', F.IN_NODE, A.RED), ('Gnd', F.GROUND_NODE, A.WHITE)):
        _, y = L.node_xy(node)
        text(stage, f'{A.ROOT}/Board/Print{name}', name.upper(), Frame((x0 + 0.006, y - 0.016), A.READ, A.UP,
                                                                         L.BOARD_TOP_Z), 0, 0, 0.0065, colour,
             align='centre')


# --- analyzer ----------------------------------------------------------------------------------------------------------
def _looks(stage):
    import visual_kit as V
    from pxr import UsdShade
    from . import asset as A
    root = A.ROOT + '/Looks/Filter'
    names = ('Analyzer', 'Resistor', 'Capacitor', 'Probe', 'LeadRed', 'LeadBlack', 'OpAmp')
    if stage.GetPrimAtPath(root + '/Analyzer'):
        return {n: UsdShade.Material(stage.GetPrimAtPath(f'{root}/{n}')) for n in names}
    return {'Analyzer': V.material(stage, root + '/Analyzer', ANALYZER_COLOUR, roughness=0.55),
            'Resistor': V.material(stage, root + '/Resistor', F.R_COLOUR, roughness=0.6),
            'Capacitor': V.material(stage, root + '/Capacitor', F.C_COLOUR, roughness=0.25, clearcoat=0.6),
            'Probe': V.material(stage, root + '/Probe', F.PROBE_COLOUR, roughness=0.4),
            'LeadRed': V.material(stage, root + '/LeadRed', (0.78, 0.06, 0.05), roughness=0.45),
            'LeadBlack': V.material(stage, root + '/LeadBlack', (0.03, 0.03, 0.03), roughness=0.45),
            'OpAmp': V.material(stage, root + '/OpAmp', F.OPAMP_COLOUR, roughness=0.5)}


def _dress(stage, collider_path, look_path, size, centre, radius, material):
    """A rounded look over a box collider of the same size (the collider made invisible, colliding as before)."""
    import visual_kit as V
    from pxr import UsdGeom
    UsdGeom.Imageable(stage.GetPrimAtPath(collider_path)).MakeInvisible()
    V.define(stage, look_path, V.rounded_box(size, radius), material, translate=centre)


def build_analyzer(stage, material, key_material):
    import visual_kit as V
    from pxr import UsdGeom
    from . import asset as A
    from .display import INK, WHITE, Frame, Readout, quad_mesh, text
    root = A.ROOT + '/Analyzer'
    UsdGeom.Xform.Define(stage, root)
    (hx0, hx1), (hy0, hy1), h = F.ANALYZER_BOX
    A._housing(stage, root + '/Housing', F.ANALYZER_BOX, ANALYZER_COLOUR, material)
    looks = _looks(stage)
    _dress(stage, root + '/Housing', root + '/HousingLook', (hx1 - hx0, hy1 - hy0, h),
           ((hx0 + hx1) / 2, (hy0 + hy1) / 2, L.TABLE_Z + h / 2), 0.006, looks['Analyzer'])
    top = F.analyzer_top_z()
    keys = {}
    for key, (x, y) in F.analyzer_key_centres().items():
        path = f'{root}/Key_{key}'
        sweep = key == 'sweep'
        colour = (0.14, 0.55, 0.24) if sweep else (0.24, 0.26, 0.30) if key in F.ANALYZER_KEY_PRINT else \
            (0.84, 0.85, 0.87)
        A.spring_cap(stage, path, (x, y, top + L.CAP_LIFT + L.CAP_H / 2), (L.KEY_SIZE, L.KEY_SIZE, L.CAP_H), colour,
                     key_material)
        label = F.ANALYZER_KEY_PRINT.get(key, key)
        dark = key in F.ANALYZER_KEY_PRINT
        text(stage, path + '/Print', label, Frame((0, 0), A.READ, A.UP, L.CAP_H / 2), 0, 0,
             0.010 if len(label) == 1 else 0.0045, WHITE if dark else INK, align='centre')
        A._box(stage, f'{root}/Key_{key}Collar', (L.KEY_SIZE + 0.004, L.KEY_SIZE + 0.004, L.CAP_LIFT - 0.001),
               (x, y, top + (L.CAP_LIFT - 0.001) / 2), (0.05, 0.05, 0.06), collide=False)
        keys[key] = path
    # screen: bezel, glass, fields, lamps, plot grid, trace
    frame = screen_frame()
    hu, hv = screen_half()
    disp = A.display_looks(stage)
    quad_mesh(stage, root + '/Bezel', frame, [(-hu - 0.005, -hv - 0.005, hu + 0.005, hv + 0.005)], BEZEL, 0.0002)
    quad_mesh(stage, root + '/Window', frame, [(-hu, -hv, hu, hv)], A.WINDOW, 0.0004, material=disp['DisplayWindow'])
    labels = {'Start': 'START', 'Stop': 'STOP', 'Pts': 'PTS', 'No': 'NO', 'Entry': 'ENTRY'}
    for name, (u0, v0, count) in FIELDS.items():
        lu = -0.072 if u0 < 0 else u0 - 0.017
        text(stage, f'{root}/{name}Label', labels[name], frame, lu, v0 + FIELD_DIGIT['digit_h'] / 2, 0.0045, WHITE,
             dz=0.0008)
        Readout(stage, f'{root}/{name}', frame, u0=u0, v0=v0, sign=False, count=count, material=disp['DisplayLit'],
                **FIELD_DIGIT)
    for name in ('Start', 'Stop'):
        u0, v0, count = FIELDS[name]
        text(stage, f'{root}/{name}Unit', 'HZ', frame, u0 + count * FIELD_DIGIT['pitch'] + 0.002,
             v0 + 0.003, 0.0045, WHITE, dz=0.0008)
    for name, ((u, v), colour) in LAMPS.items():
        p = frame.at(u, v)
        A._disc(stage, f'{root}/{name}LampDim', 0.0040, (p[0], p[1], top + 0.0006), (0.10, 0.10, 0.09))
        A._disc(stage, f'{root}/{name}Lamp', 0.0041, (p[0], p[1], top + 0.0009), colour, visible=False,
                material=A._render_material(stage, f'{root}/Looks/Lamp{name}', colour, emission=4.0))
        text(stage, f'{root}/{name}LampLabel', name.upper(), frame, u + 0.007, v, 0.0045, WHITE, dz=0.0008)
    u0, u1, v0, v1 = PLOT
    t = 0.0004
    grid = [(u0, v0, u1, v0 + t), (u0, v1 - t, u1, v1), (u0, v0, u0 + t, v1), (u1 - t, v0, u1, v1)]
    for k in range(SCREEN_DECADES[0] + 1, SCREEN_DECADES[1]):
        u, _ = plot_uv(10.0 ** k, 0.0)
        grid.append((u - t / 2, v0, u + t / 2, v1))
    for g in range(0, -90, -20):
        _, v = plot_uv(10.0, g)
        grid.append((u0, v - t / 2, u1, v + t / 2))
        text(stage, f'{root}/Db{-g}', str(g), frame, u0 - 0.002, v, 0.0034, WHITE, align='right', dz=0.0008)
    quad_mesh(stage, root + '/Grid', frame, grid, GRID, 0.0006)
    for k, name in zip(range(SCREEN_DECADES[0], SCREEN_DECADES[1] + 1), ('10', '100', '1K', '10K', '100K')):
        u, _ = plot_uv(10.0 ** k, 0.0)
        text(stage, f'{root}/Hz{name}', name, frame, u, v0 - 0.006, 0.0034, WHITE, align='centre', dz=0.0008)
    text(stage, root + '/AxisDb', 'DB', frame, u0 - 0.002, v1 + 0.004, 0.0034, WHITE, align='right', dz=0.0008)
    text(stage, root + '/AxisHz', 'HZ', frame, u1, v0 - 0.012, 0.0034, WHITE, align='right', dz=0.0008)
    mesh = UsdGeom.Mesh.Define(stage, A.ROOT + TRACE)
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([(0.25, 0.92, 0.30)])
    from pxr import UsdShade
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(disp['DisplayLit'])
    _set_trace(stage, [])
    # printing on the housing top
    text(stage, root + '/Title', 'FREQUENCY RESPONSE ANALYZER', Frame((hx0 + 0.013, (hy0 + hy1) / 2), A.READ, A.UP,
                                                                       top), 0, 0, 0.0060, INK, align='centre')
    # output jacks on the +x face, leads (render-only cables) to the IN pad and the ground rail, lugs on the pads
    for name, node, lead in (('Out', F.IN_NODE, 'LeadRed'), ('Gnd', F.GROUND_NODE, 'LeadBlack')):
        nx, ny = L.node_xy(node)
        zj = L.TABLE_Z + 0.022
        V.define(stage, f'{root}/Jack{name}', V.tube([(hx1, ny, zj), (hx1 + 0.007, ny, zj)], 0.0045, sides=20),
                 looks[lead])
        bx0 = L.board_extent()[0][0]
        r = 0.0018
        path = [(hx1 + 0.007, ny, zj), (hx1 + 0.014, ny, zj - 0.004), (hx1 + 0.019, ny, L.TABLE_Z + r + 0.003),
                (hx1 + 0.022, ny, L.TABLE_Z + r), (bx0 - r - 0.004, ny, L.TABLE_Z + r),
                (bx0 - r - 0.0005, ny, L.TABLE_Z + r + 0.004), (bx0 - r - 0.0005, ny, L.BOARD_TOP_Z),
                (bx0 + 0.002, ny, L.BOARD_TOP_Z + r)]
        V.define(stage, f'{root}/Lead{name}', V.tube(path, r, sides=12), looks[lead])
        A._strap(stage, f'{root}/Lug{name}', bx0 + 0.001, nx - L.PAD_HALF + 0.004, ny,
                 (0.78, 0.06, 0.05) if name == 'Out' else (0.03, 0.03, 0.03))
        text(stage, f'{root}/Jack{name}Label', 'OUTPUT' if name == 'Out' else 'GND', Frame((hx1 - 0.007, ny),
                                                                                        A.READ, A.UP, top),
             0, 0, 0.0045, INK, align='centre')
    return keys


def _set_trace(stage, uv):
    """The trace mesh from screen points [(u, v)]: a 1.2 mm wide band along the polyline (nothing: invisible)."""
    from pxr import Gf, UsdGeom
    from . import asset as A
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(A.ROOT + TRACE))
    if not uv:
        mesh.MakeInvisible()
        return
    frame = screen_frame()
    w = 0.0006
    quads = []
    pts = uv if len(uv) > 1 else [uv[0], (uv[0][0] + 0.001, uv[0][1])]
    for (ua, va), (ub, vb) in zip(pts, pts[1:]):
        d = np.array([ub - ua, vb - va])
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        nu, nv = -d[1] / n * w, d[0] / n * w
        ea, eb = d / n * w * 0.5, d / n * w * 0.5        # small overlap at the joints
        quads.append([(ua - ea[0] - nu, va - ea[1] - nv), (ub + eb[0] - nu, vb + eb[1] - nv),
                      (ub + eb[0] + nu, vb + eb[1] + nv), (ua - ea[0] + nu, va - ea[1] + nv)])
    points = [Gf.Vec3f(*frame.at(u, v, 0.0010)) for q in quads for u, v in q]
    if not points:
        mesh.MakeInvisible()
        return
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set([4] * len(quads))
    mesh.GetFaceVertexIndicesAttr().Set(list(range(len(points))))
    arr = np.asarray([[p[0], p[1], p[2]] for p in points])
    mesh.GetExtentAttr().Set([Gf.Vec3f(*arr.min(0).tolist()), Gf.Vec3f(*arr.max(0).tolist())])
    mesh.MakeVisible()


def show_analyzer(backend, d):
    """The screen from Analyzer.display(): fields, entry, sweep number, RUN and ERR lamps, the trace. Returns the
    shown state (its change resets the denoiser history)."""
    from . import asset as A
    from .display import set_visible
    root = A.ROOT + '/Analyzer'
    trace = d['trace']
    fields = {'Start': str(d['start_hz']), 'Stop': str(d['stop_hz']), 'Pts': str(d['pts']),
              'No': '' if trace is None else str(trace['n']), 'Entry': d['entry']}
    for name, value in fields.items():
        backend.displays._readout(f'{root}/{name}', FIELDS[name][2]).show(' ', value, None)
    set_visible(backend.stage, root + '/RunLamp', bool(d['running']))
    set_visible(backend.stage, root + '/ErrLamp', bool(d['error']))
    shown = {'fields': fields, 'run': bool(d['running']), 'err': bool(d['error']),
             'trace': None if trace is None else [trace['n'], len(trace['f'])]}
    if shown['trace'] != getattr(backend, '_trace_shown', 'none'):
        backend._trace_shown = shown['trace']
        _set_trace(backend.stage, [] if trace is None else [plot_uv(f, g) for f, g in zip(trace['f'], trace['g'])])
    return shown


# --- trays -------------------------------------------------------------------------------------------------------------
def build_trays(stage, material):
    """Both trays as one static mesh (the platform's support sensors watch Tray/Body): platform wells (mouth, chamfer,
    floor either side of the pin groove) and long wells for the op-amp modules in tray B, labelled with the print."""
    from pxr import Gf, UsdGeom
    from . import asset as A
    from .display import Frame, text
    top, floor, groove = L.TABLE_Z + L.TRAY_H, L.TABLE_Z + L.TRAY_FLOOR, L.TABLE_Z + L.TRAY_GROOVE_FLOOR
    throat = top - L.TRAY_CHAMFER_DEPTH
    gh = L.TRAY_GROOVE_HALF
    grow = F.opamp_well_grow()
    wells = [(label, s['xy'], L.TRAY_WELL_HALF, L.TRAY_MOUTH_HALF, s['printed'], s['tray'])
             for label, s in F.tray_seats().items()]
    wells += [(b, xy, (L.TRAY_WELL_HALF[0], L.TRAY_WELL_HALF[1] + grow),
               (L.TRAY_MOUTH_HALF[0], L.TRAY_MOUTH_HALF[1] + grow), F.THREE[b][1], 'B')
              for b, (xy, _) in F.OPAMP_WELLS.items()]
    quads = []
    for tray, ((x0, x1), (y0, y1)) in F.tray_extents().items():
        mine = [w for w in wells if w[5] == tray]
        for _, (cx, cy), _, (mx, my), _, _ in mine:
            if cx - mx < x0 or cx + mx > x1 or cy - my < y0 or cy + my > y1:
                raise ValueError(f'tray well at {(cx, cy)} outside tray {tray}')
        holes = [(cx - mx, cx + mx, cy - my, cy + my) for _, (cx, cy), _, (mx, my), _, _ in mine]
        quads += [q for q, _ in A._grid_cells((x0, x1), (y0, y1), holes, top)]
        for _, (cx, cy), (wx, wy), (mx, my), _, _ in mine:
            quads += A._pocket(cx, cy, [(mx, my, top), (wx, wy, throat), (wx, wy, floor)])
            quads.append(((cx - wx, cy - wy, floor), (cx - gh, cy - wy, floor), (cx - gh, cy + wy, floor),
                          (cx - wx, cy + wy, floor)))
            quads.append(((cx + gh, cy - wy, floor), (cx + wx, cy - wy, floor), (cx + wx, cy + wy, floor),
                          (cx + gh, cy + wy, floor)))
            for sx in (-gh, gh):
                quads.append(((cx + sx, cy - wy, floor), (cx + sx, cy + wy, floor), (cx + sx, cy + wy, groove),
                              (cx + sx, cy - wy, groove)))
            for sy in (-wy, wy):
                quads.append(((cx - gh, cy + sy, floor), (cx + gh, cy + sy, floor), (cx + gh, cy + sy, groove),
                              (cx - gh, cy + sy, groove)))
            quads.append(((cx - gh, cy - wy, groove), (cx + gh, cy - wy, groove), (cx + gh, cy + wy, groove),
                          (cx - gh, cy + wy, groove)))
        bottom = L.TABLE_Z
        quads += [((x0, y0, bottom), (x1, y0, bottom), (x1, y0, top), (x0, y0, top)),
                  ((x0, y1, bottom), (x1, y1, bottom), (x1, y1, top), (x0, y1, top)),
                  ((x0, y0, bottom), (x0, y1, bottom), (x0, y1, top), (x0, y0, top)),
                  ((x1, y0, bottom), (x1, y1, bottom), (x1, y1, top), (x1, y0, top))]
    UsdGeom.Xform.Define(stage, A.ROOT + '/Tray')
    A._quads(stage, A.ROOT + '/Tray/Body', quads, A.TRAY_COLOUR, collide=True, material=material)
    for label, (cx, cy), _, _, printed, _ in wells:
        text(stage, f'{A.ROOT}/Tray/Label{label}', printed, Frame((cx + L.TRAY_LABEL_DX, cy), A.READ, A.UP, top), 0,
             0, 0.0075 if len(printed) < 6 else 0.0060, A.WHITE, align='centre')
        seat = UsdGeom.Xform.Define(stage, f'{A.ROOT}/Tray/Seat{label}')
        UsdGeom.XformCommonAPI(seat).SetTranslate(Gf.Vec3d(cx, cy, floor))


# --- blocks -----------------------------------------------------------------------------------------------------------
def dress_block(stage, block, path, looks):
    """Looks of a two-pin piece built by asset.build_block."""
    from . import asset as A
    from .display import WHITE, Frame, text
    kind, printed = L.BLOCKS[block][:2]
    bz = L.BASE[2]
    hx, hy, hz = L.HANDLE
    centre = (0.0, 0.0, bz + hz / 2)
    top = Frame((0, 0), (1, 0), (0, 1), L.BLOCK_TOP)
    if kind == 'resistor':
        _dress(stage, path + '/Handle', path + '/HandleLook', L.HANDLE, centre, 0.0025, looks['Resistor'])
        value = _r_nominal()[F.R_PIECES.index(block)]
        for k, (code, u) in enumerate(zip(resistor_bands(value), (-0.0065, -0.0030, 0.0005, 0.0065))):
            w = 0.0011
            z0, z1 = bz + 0.0015, bz + hz - 0.0015
            quads = []
            for s in (-1, 1):
                y = s * (hy / 2 + 0.0002)
                quads.append(((u - w, y, z0), (u + w, y, z0), (u + w, y, z1), (u - w, y, z1)))
            A._quads(stage, f'{path}/Band{k}', quads, RESISTOR_BANDS[code])
    elif kind == 'capacitor':
        _dress(stage, path + '/Handle', path + '/HandleLook', L.HANDLE, centre, 0.0040, looks['Capacitor'])
        text(stage, path + '/Value', printed, top, 0, 0, 0.0045, WHITE, align='centre')
    elif kind == 'probe':
        from pxr import Gf, UsdGeom
        _dress(stage, path + '/Handle', path + '/HandleLook', L.HANDLE, centre, 0.0030, looks['Probe'])
        UsdGeom.Gprim(stage.GetPrimAtPath(path + '/Base')).GetDisplayColorAttr().Set([Gf.Vec3f(0.06, 0.06, 0.07)])
        from .display import quad_mesh
        band = Frame((0, 0), (1, 0), (0, 1), bz)
        quad_mesh(stage, path + '/PlusBand', band, [(L.BASE_PLUS_BAND[0], -L.BASE[1] / 2, L.BASE_PLUS_BAND[1],
                                                     L.BASE[1] / 2)], A.RED, 0.0004)
        text(stage, path + '/Label', 'PROBE', top, 0, 0, 0.0036, (0.02, 0.02, 0.02), align='centre')
        text(stage, path + '/PlusMark', '+', band, 0.0140, 0, 0.0050, WHITE, align='centre', dz=0.0008)


def _r_nominal():
    import filter_model as M
    return [float(M.R_NOM[k // M.COPIES]) for k in range(len(M.R_NOM) * M.COPIES)]


def build_opamp(stage, block_id, pose_xyz, yaw_deg, material, looks):
    """An op-amp module: a three-node block (layout.BASE3; pins IN+, IN-, OUT where the changeover has T1, C, T2)
    with its handle over the middle of the IN- / OUT edge (no lever)."""
    from pxr import Gf, UsdGeom
    from . import asset as A
    from .display import WHITE, Frame, text
    kind, printed, _, mass = F.THREE[block_id]
    path = L.block_path(A.ROOT, block_id)
    body = UsdGeom.Xform.Define(stage, path)
    api = UsdGeom.XformCommonAPI(body)
    api.SetTranslate(Gf.Vec3d(*pose_xyz))
    api.SetRotate(Gf.Vec3f(0, 0, yaw_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    A._rigid(body.GetPrim(), mass)
    bz = L.BASE3[2]
    A._box(stage, path + '/Base', L.BASE3, (0, 0, bz / 2), (0.12, 0.13, 0.14), material)
    for name, local in L.changeover_pins().items():
        sphere = UsdGeom.Sphere.Define(stage, f'{path}/Pin{name}')
        sphere.CreateRadiusAttr(L.PIN_R)
        sphere.CreateExtentAttr([Gf.Vec3f(-L.PIN_R), Gf.Vec3f(L.PIN_R)])
        UsdGeom.XformCommonAPI(sphere).SetTranslate(Gf.Vec3d(*local))
        sphere.CreateDisplayColorAttr([Gf.Vec3f(*A.BRASS)])
        A._collider(sphere.GetPrim(), material)
    hx, hy, hz = L.HANDLE
    centre = (F.OPAMP_HANDLE_X, 0, bz + hz / 2)
    A._box(stage, path + '/Handle', L.HANDLE, centre, F.OPAMP_COLOUR, material)
    _dress(stage, path + '/Handle', path + '/HandleLook', L.HANDLE, centre, 0.0025, looks['OpAmp'])
    local = Frame((0, 0), (1, 0), (0, 1), bz)
    for name, x in zip(('IN+', 'IN-', 'OUT'), (L.CHANGEOVER_PINS['T1'], L.CHANGEOVER_PINS['C'],
                                               L.CHANGEOVER_PINS['T2'])):
        text(stage, f'{path}/Mark{name.replace("+", "P").replace("-", "M")}', name, local, x, 0, 0.0045, WHITE,
             align='centre', dz=0.0004)
    text(stage, f'{path}/Label', printed, local, 0.002, 0.0, 0.0036, WHITE, align='centre', dz=0.0004)
    text(stage, f'{path}/HandleLabel', 'OA' + block_id[-1], Frame((F.OPAMP_HANDLE_X, 0), (1, 0), (0, 1), L.BLOCK_TOP),
         0, 0, 0.0050, WHITE, align='centre')
    return path


# --- whole bench ------------------------------------------------------------------------------------------------------
def build(stage, profile):
    from pxr import PhysxSchema, UsdGeom
    from . import asset as A
    A.BLOCK_COLOURS.update(profile.colours)
    A.retire_hooke_apparatus(stage)
    A.repair_cube_extents(stage)
    z = A.table_top_z(stage)
    if abs(z - L.TABLE_Z) > 0.002:
        raise RuntimeError(f'table top at {z:.4f} m, the circuit board layout expects {L.TABLE_Z} m')
    UsdGeom.Xform.Define(stage, A.ROOT)
    UsdGeom.Xform.Define(stage, A.PHYSICS_ROOT)
    fixed = A._physics_material(stage, '/World/PhysicsMaterials/CircuitGridFixed', 0.5, 0.4)
    blocks_mat = A._physics_material(stage, '/World/PhysicsMaterials/CircuitGridBlock', 0.5, 0.4)
    keys_mat = A._physics_material(stage, '/World/PhysicsMaterials/CircuitGridKey', 0.8, 0.7)
    A.build_board(stage, fixed)
    dress_board(stage)
    keys = build_analyzer(stage, fixed, keys_mat)
    build_trays(stage, fixed)
    looks = _looks(stage)
    paths = {}
    UsdGeom.Xform.Define(stage, A.ROOT + '/Blocks')
    for label, seat in F.tray_seats().items():
        pose = (seat['xy'][0], seat['xy'][1], L.TABLE_Z + L.TRAY_FLOOR + 0.0005)
        paths[label] = A.build_block(stage, label, pose, L.TRAY_YAW_DEG, blocks_mat)
        dress_block(stage, label, paths[label], looks)
    for block, (xy, yaw) in F.OPAMP_WELLS.items():
        paths[block] = build_opamp(stage, block, (xy[0], xy[1], L.TABLE_Z + L.TRAY_FLOOR + 0.0005), yaw, blocks_mat,
                                   looks)
    table = stage.GetPrimAtPath(A.TABLE_TOP)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(table)
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    # no switch rocker on this bench: the backend's rocker view goes on the first block, never read
    return {'blocks': paths, 'keys': keys, 'button': None, 'rocker': next(iter(paths.values()))}

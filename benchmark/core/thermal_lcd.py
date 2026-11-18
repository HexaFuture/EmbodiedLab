"""Seven-segment LCD panels of the thermal benches (heater / dry-block controller screen, digital thermometer,
ohmmeter, digital indicator), in the look of the beam bench's digital indicators: dark glass, green lit segments.

A panel is an Xform at `centre` turned by (tilt from vertical toward the robot, 0, 180): its local x reads from the
robot's left to its right (world -x), local y runs up the face, local z is the face normal. The housing lies behind
the face (local z < 0). Every segment, decimal point and switchable label (annunciator) is its own render-only prim;
an unlit one sinks HIDE_M into the housing (the renderer's denoiser history is reset by the backend after a change,
as for the chemistry screens).
"""
from __future__ import annotations

from pxr import Gf, Usd, UsdGeom

from chem_apparatus import SEGMENT_EMISSION, SEGMENT_ON, SEGMENTS as SEVEN, _bind, _render_material
from circuit_apparatus import label
from torsion_apparatus import _box

GLYPHS = {'0': 'abcdef', '1': 'bc', '2': 'abdeg', '3': 'abcdg', '4': 'bcfg', '5': 'acdfg', '6': 'acdefg', '7': 'abc',
          '8': 'abcdefg', '9': 'abcdfg', '-': 'g', ' ': '', 'O': 'abcdef', 'L': 'def', 'E': 'adefg', 'r': 'eg'}
GLASS = (0.004, 0.018, 0.010)
WHITE = (0.95, 0.95, 0.93)
HIDE_M = 0.004
SEG_Z = 0.0007


def looks(stage, root):
    return {'glass': _render_material(stage, root + '/LcdLooks/Glass', GLASS, matte=True),
            'lit': _render_material(stage, root + '/LcdLooks/Segment', SEGMENT_ON, emission=SEGMENT_EMISSION, matte=True)}


def _cell_u(row, i):
    h = row['digit_h']
    pitch, gap = h * 0.72, h * 0.2
    return row['u'] + i * pitch + gap * sum(1 for p in row.get('points', ()) if p < i)


def row_width(row):
    h = row['digit_h']
    return _cell_u(row, row['cells'] - 1) + h * 0.5 - row['u']


def row_rects(row):
    """{'segments': {(cell, seg): ((u, v), (du, dv))}, 'points': {after_cell: ((u, v), (du, dv))}} in face coordinates.
    row: {'cells', 'digit_h', 'u' (left edge), 'v' (bottom edge), 'points' (cells a point may follow)}."""
    h = row['digit_h']
    w, t = h * 0.5, h * 0.14
    v0 = row['v']
    segments = {(i, s): ((_cell_u(row, i) + fu * w, v0 + fv * h), (w, t) if horizontal else (t, h / 2))
                for i in range(row['cells']) for s, (fu, fv, horizontal) in SEVEN.items()}
    points = {}
    for p in row.get('points', ()):
        u = _cell_u(row, p) + w + (_cell_u(row, p + 1) - _cell_u(row, p) - w) / 2
        points[p] = ((u, v0 + t / 2), (t * 1.3, t * 1.3))
    return {'segments': segments, 'points': points}


def build_panel(stage, path, *, centre, tilt_deg, face, depth, rows, labels=(), toggles=None, glass=None,
                housing_colour=(0.16, 0.17, 0.19), material=None, looks_=None, collide=True):
    """Author one panel. rows {name: row}; labels [(text, (u, v), height, align)] printed white on the face;
    toggles {name: (text, (u, v), height, align)} lit-colour labels that can be switched off; glass ((u, v), (w, h))
    of the LCD window (default: around the rows)."""
    xform = UsdGeom.Xform.Define(stage, path)
    api = UsdGeom.XformCommonAPI(xform)
    api.SetTranslate(Gf.Vec3d(*centre))
    api.SetRotate(Gf.Vec3f(float(tilt_deg), 0.0, 180.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    fw, fh = face
    _box(stage, path + '/Housing', (fw, fh, depth), (0, 0, -depth / 2), housing_colour, material=material, collide=collide)
    if glass is not None:
        (gu, gv), (gw, gh) = glass
        _bind(_box(stage, path + '/Glass', (gw, gh, 0.0006), (gu, gv, 0.0003), GLASS, collide=False), looks_['glass'])
    for name, row in rows.items():
        rects = row_rects(row)
        for (i, s), ((u, v), (du, dv)) in rects['segments'].items():
            _bind(_box(stage, f'{path}/{name}{i}{s}', (du, dv, 0.0006), (u, v, SEG_Z), SEGMENT_ON, collide=False),
                  looks_['lit'])
        for p, ((u, v), (du, dv)) in rects['points'].items():
            _bind(_box(stage, f'{path}/{name}Point{p}', (du, dv, 0.0006), (u, v, SEG_Z), SEGMENT_ON, collide=False),
                  looks_['lit'])
    for k, (text, (u, v), height, align) in enumerate(labels):
        label(stage, f'{path}/Label{k}', text, (u, v), 0.0004, height_m=height, align=align, colour=WHITE, up=(0.0, 1.0))
    for name, (text, (u, v), height, align) in (toggles or {}).items():
        mesh = label(stage, f'{path}/Toggle_{name}', text, (u, v), SEG_Z, height_m=height, align=align,
                     colour=SEGMENT_ON, up=(0.0, 1.0))
        _bind(mesh, looks_['lit'])
        UsdGeom.XformCommonAPI(mesh).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.0))
    return xform


def _sink(stage, path, lit, z=SEG_Z):
    prim = stage.GetPrimAtPath(path)
    api = UsdGeom.XformCommonAPI(UsdGeom.Xformable(prim))
    t = api.GetXformVectors(Usd.TimeCode.Default())[0]
    api.SetTranslate(Gf.Vec3d(t[0], t[1], z if lit else z - HIDE_M))


def show(stage, path, rows, state, previous):
    """Light what `state` asks for, changing only what differs from `previous` (None: everything).
    state: {'text': {row: str of cells}, 'point': {row: cell or None}, 'toggles': {name: bool}}."""
    if state == previous:
        return state
    for name, text in state.get('text', {}).items():
        old = previous['text'].get(name) if previous else None
        for i, char in enumerate(text):
            if old is not None and old[i] == char:
                continue
            on = GLYPHS.get(char, '')
            for s in SEVEN:
                _sink(stage, f'{path}/{name}{i}{s}', s in on)
    for name, point in state.get('point', {}).items():
        old = previous['point'].get(name, -1) if previous else -1
        if old == point:
            continue
        for p in rows[name].get('points', ()):
            _sink(stage, f'{path}/{name}Point{p}', p == point)
    for name, lit in state.get('toggles', {}).items():
        if previous is None or previous['toggles'].get(name) != lit:
            _sink(stage, f'{path}/Toggle_{name}', lit, z=0.0)
    return state


def fixed(value, cells, decimals):
    """Right-aligned digits of a value for `cells` cells with the point before the last `decimals` cells; leading
    zeros before the units cell blank, a minus sign in the cell left of the first digit. None shows dashes."""
    if value is None:
        return '-' * cells
    counts = int(round(abs(float(value)) * 10 ** decimals))
    digits = str(counts).rjust(decimals + 1, '0')
    text = ('-' if value < 0 and counts else '') + digits
    text = text[-cells:].rjust(cells)
    return text


# --- the thermal kit's instruments -----------------------------------------------------------------------------
CONTROLLER_FACE = (0.090, 0.036)       # bezel on the block's front face: SET (top) and the block reading (bottom)
CONTROLLER_DEPTH = 0.004
CONTROLLER_Z = 0.020                   # face centre above the table (the blocks are 40 mm tall)


def controller_rows(cells, decimals):
    return {'set': {'cells': 2, 'digit_h': 0.009, 'u': -0.008, 'v': 0.0045, 'points': ()},
            'pv': {'cells': cells, 'digit_h': 0.014, 'u': -0.008, 'v': -0.0155, 'points': (cells - decimals - 1,)}}


def build_controller(stage, path, *, x, front_y, looks_, cells, decimals, material=None):
    """The heater / dry-block controller screen on the block's front face (normal toward the robot)."""
    rows = controller_rows(cells, decimals)
    right = -0.008 + max(row_width(r) for r in rows.values())
    labels = [('SET', (-0.041, 0.0090), 0.0050, 'left'), ('BLOCK', (-0.041, -0.0095), 0.0050, 'left'),
              ('C', (right + 0.007, 0.0090), 0.0050, 'left'), ('C', (right + 0.007, -0.0095), 0.0050, 'left')]
    left, margin = -0.012, 0.004                       # the window: 4 mm clear of the last digit (it ended flush)
    build_panel(stage, path, centre=(x, front_y + CONTROLLER_DEPTH, CONTROLLER_Z), tilt_deg=90.0, face=CONTROLLER_FACE,
                depth=CONTROLLER_DEPTH, rows=rows, labels=labels,
                glass=(((left + right + margin) / 2, -0.0010), (right + margin - left, 0.033)),
                housing_colour=(0.10, 0.10, 0.11), material=material, looks_=looks_, collide=False)
    return rows


def controller_state(set_c, block_c, cells, decimals):
    return {'text': {'set': fixed(set_c, 2, 0), 'pv': fixed(block_c, cells, decimals)},
            'point': {'pv': None if block_c is None else cells - decimals - 1}, 'toggles': {}}


THERMOMETER_FACE = (0.064, 0.040)
THERMOMETER_ROWS = {'T': {'cells': 4, 'digit_h': 0.013, 'u': -0.022, 'v': -0.011, 'points': (1,)}}


def build_stand_panel(stage, path, *, xy, top_z, tilt_deg, face, depth, rows, labels, toggles=None, looks_, material,
                      foot_colour=(0.20, 0.21, 0.23), glass=None):
    """A panel on a short post and a round foot, its face centre top_z above the table, tilted toward the robot.
    glass ((u, v), (w, h)) of the LCD window (default: around the rows, which stand side by side)."""
    import math
    from torsion_apparatus import _cylinder
    x, y = xy
    fw, fh = face
    if glass is None:
        widths = [row_width(r) for r in rows.values()]
        first = min(r['u'] for r in rows.values())
        glass = ((first + max(widths) / 2, min(r['v'] for r in rows.values()) + max(r['digit_h'] for r in rows.values()) / 2),
                 (max(widths) + 0.008, max(r['digit_h'] for r in rows.values()) + 0.008))
    build_panel(stage, path, centre=(x, y, top_z), tilt_deg=tilt_deg, face=face, depth=depth, rows=rows, labels=labels,
                toggles=toggles, glass=glass, material=material, looks_=looks_)
    t = math.radians(tilt_deg)
    low = top_z - (fh / 2) * math.sin(t) - depth * math.cos(t)          # the housing's lowest edge
    post_y = y - (depth / 2) * math.sin(t)
    _cylinder(stage, path + 'Post', 0.006, max(low, 0.004), (x, post_y, max(low, 0.004) / 2), (0.25, 0.26, 0.28), material)
    _cylinder(stage, path + 'Foot', 0.020, 0.006, (x, post_y, 0.003), foot_colour, material)


def build_thermometer(stage, path, *, xy, title, looks_, material, top_z=0.050):
    rows = THERMOMETER_ROWS
    labels = [(title, (-0.029, 0.0150), 0.0042, 'left'), ('C', (0.024, -0.005), 0.0060, 'left')]
    build_stand_panel(stage, path, xy=xy, top_z=top_z, tilt_deg=45.0, face=THERMOMETER_FACE, depth=0.024, rows=rows,
                      labels=labels, looks_=looks_, material=material)
    return rows


def thermometer_state(t_c):
    return {'text': {'T': fixed(t_c, 4, 2)}, 'point': {'T': None if t_c is None else 1}, 'toggles': {}}


# The calorimetry bench's water thermometer with its recorder: the temperature row at the bottom of the LCD, above it
# the REC mark (lit while a record runs) and the running record's number.
RECORDER_FACE = (0.070, 0.054)
RECORDER_ROWS = {'T': {'cells': 4, 'digit_h': 0.013, 'u': -0.022, 'v': -0.020, 'points': (1,)},
                 'N': {'cells': 2, 'digit_h': 0.009, 'u': 0.003, 'v': -0.002, 'points': ()}}
RECORDER_GLASS = ((-0.0025, -0.0070), (0.049, 0.034))


def build_recording_thermometer(stage, path, *, xy, title, looks_, material, top_z=0.052):
    labels = [(title, (-0.032, 0.0205), 0.0042, 'left'), ('C', (0.025, -0.0135), 0.0060, 'left'),
              ('NO', (0.0245, 0.0025), 0.0040, 'left')]
    toggles = {'rec': ('REC', (-0.022, 0.0025), 0.0070, 'left')}
    build_stand_panel(stage, path, xy=xy, top_z=top_z, tilt_deg=45.0, face=RECORDER_FACE, depth=0.024,
                      rows=RECORDER_ROWS, labels=labels, toggles=toggles, looks_=looks_, material=material,
                      glass=RECORDER_GLASS)
    return RECORDER_ROWS


def recording_thermometer_state(t_c, recording, record):
    """The water temperature; while a record runs the REC mark and the record's number (1-99, blank otherwise)."""
    number = f'{int(record) % 100:2d}' if recording and record is not None else '  '
    return {'text': {'T': fixed(t_c, 4, 2), 'N': number}, 'point': {'T': None if t_c is None else 1, 'N': None},
            'toggles': {'rec': bool(recording)}}


OHMMETER_FACE = (0.086, 0.048)
OHMMETER_ROWS = {'R': {'cells': 5, 'digit_h': 0.013, 'u': -0.032, 'v': -0.013, 'points': (0, 1, 2)}}


def build_ohmmeter(stage, path, *, xy, looks_, material, top_z=0.055):
    labels = [('OHMMETER', (0.039, 0.0170), 0.0042, 'right')]
    toggles = {'auto': ('AUTO', (-0.039, 0.0100), 0.0050, 'left'),
               'kohm': ('kOhm', (0.040, 0.0085), 0.0055, 'right'),      # the unit above the window's right end
               'ohm': ('Ohm', (0.040, 0.0085), 0.0055, 'right')}
    build_stand_panel(stage, path, xy=xy, top_z=top_z, tilt_deg=45.0, face=OHMMETER_FACE, depth=0.030,
                      rows=OHMMETER_ROWS, labels=labels, toggles=toggles, looks_=looks_, material=material)
    return OHMMETER_ROWS


def ohmmeter_state(shown):
    """shown: thermo_model.Ohmmeter.read's display dict, or None before the first reading."""
    if not shown:
        return {'text': {'R': '     '}, 'point': {'R': None}, 'toggles': {'auto': True, 'kohm': False, 'ohm': False}}
    return {'text': {'R': shown['digits']}, 'point': {'R': shown['point']},
            'toggles': {'auto': bool(shown['auto']), 'kohm': shown['unit'] == 'kOhm', 'ohm': shown['unit'] == 'Ohm'}}


INDICATOR_FACE = (0.068, 0.040)
INDICATOR_ROWS = {'G': {'cells': 6, 'digit_h': 0.0115, 'u': -0.027, 'v': -0.012, 'points': (2,)}}


def indicator_state(mm):
    return {'text': {'G': fixed(mm, 6, 3)}, 'point': {'G': None if mm is None else 2}, 'toggles': {}}

"""Optics bench apparatus (optics kit): rail, light box, motorised carriages, lens holders, screen, stage
controller keypad. Built with the chemistry kit's helpers on the same root (collar seats, button caps and their
joints, text, materials), so the chemistry backend's grasp, seat and button machinery runs it unchanged.

World layout (the robot stands at (0.95, 0.35) facing -y): the rail runs along world x at y = RAIL_Y; rail
coordinate s (mm) is world x = OBJECT_X + s / 1000, the object plane (the light box's illuminated F) at s = 0.
Lens carriages carry a collar seat; a lens holder is a cuvette-sized block (it stands in a collar well like a
cuvette) with a post, the lens disc on the optical axis and a white rim handle on top. The screen carriage carries
a vertical screen facing the light box; its image is a texture the backend replaces after each move. The lens
stand (three collars) and the controller keypad stand in front of the rail.

build_hand (config drive 'hand': lens_focal, grating_wavelength) builds the hand-slid variant instead: no motors and
no keypad; each rider is a dynamic body on a prismatic rail joint (end stops as its limits, a saturated damper drive
as the rail friction), slid by the robot's hand at its orange handle; a lens rider's seat collar is a kinematic body
the backend keeps on the rider; a millimetre scale strip runs along the rail; the graduated screen shows the printed
card plus the light on it (two textures); the source has a POWER push button and pilot lamp.
"""
from __future__ import annotations

import math

import numpy as np
from pxr import Gf, UsdGeom, UsdShade, Sdf

import chem_apparatus as C
from chem_apparatus import (CUVETTE, PORT_H, PORT_R, FUNNEL_RIM_H, FUNNEL_RIM_R, FUNNEL_POST_R, ROOT, SEAT_FLOOR,
                            _bind, _button, _render_material, _rigid, _text, collar)
from torsion_apparatus import TABLE_TOP, _box, _cylinder, _material, repair_cube_extents, retire_hooke_apparatus, table_top_z

OBJECT_X = 0.66
RAIL_Y = -0.17
AXIS_Z = 0.115                    # optical axis above the table top
RAIL = (0.66, 0.030, 0.020)       # length (x), width, height
CARRIAGE = (0.050, 0.060, 0.020)
CARRIAGE_TOP = RAIL[2] + CARRIAGE[2]
HOLDER_BASE = (CUVETTE[0], CUVETTE[1], 0.030)
LENS_R, LENS_T = 0.0125, 0.004
LENS_R_MAX = 0.020                 # a larger lens (plate_index) still clears the holder's post top
PLATE_L = 0.060                    # glass block length along the axis (plate_index)
SCREEN_SIZE = 0.120
STAND = {'xs': (0.78, 0.86, 0.94), 'y': -0.02, 'base': (0.26, 0.06, 0.012)}
KEYPAD = {'xs': (1.09, 1.14, 1.19, 1.24), 'y0': -0.095, 'pitch_y': 0.036,
          'rows': (('axis:A', 'axis:B', 'axis:S', 'key:clear'), ('key:1', 'key:2', 'key:3', 'key:m10'),
                   ('key:4', 'key:5', 'key:6', 'key:m1'), ('key:7', 'key:8', 'key:9', 'key:p1'),
                   ('key:point', 'key:0', 'key:go', 'key:p10'))}
KEY_TEXT = {'point': '.', 'clear': 'CLR', 'go': 'GO', 'm10': '-10', 'm1': '-1', 'p1': '+1', 'p10': '+10'}
BODY_Z = 0.030
LENS_PATH = ROOT + '/Lens{}'
CARRIAGE_PATH = ROOT + '/Carriage{}'
SCREEN_PATH = ROOT + '/Carriage{}/Screen'
TEXTURE = {'path': None}


def rail_x(s_mm):
    return OBJECT_X + s_mm / 1000.0


def carriage_path(axis):
    return CARRIAGE_PATH.format(axis)


def lens_path(lens):
    return LENS_PATH.format(lens)


def seat_path(sid):
    kind, _, key = sid.partition(':')
    return f'{ROOT}/StandSeat{key}' if kind == 'stand' else f'{carriage_path(key)}/Seat'


def button_paths(stage=None):
    return {bid: C.button_path(bid) for row in KEYPAD['rows'] for bid in row if bid}


def button_layout():
    out = {}
    for r, row in enumerate(KEYPAD['rows']):
        for c, bid in enumerate(row):
            if bid:
                out[bid] = ((KEYPAD['xs'][c], KEYPAD['y0'] + r * KEYPAD['pitch_y']), BODY_Z, 'key')
    return out


def _holder(stage, lens, xy, bottom_z, materials, colour=(0.62, 0.80, 0.92), kind='lens', label_=None, radius=LENS_R):
    path = lens_path(lens)
    xform = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(xy[0], xy[1], bottom_z))
    sx, sy, sz = HOLDER_BASE
    _box(stage, path + '/Base', HOLDER_BASE, (0, 0, sz / 2), (0.20, 0.21, 0.23), material=materials['cell'])
    axis_local = AXIS_Z - (CARRIAGE_TOP + SEAT_FLOOR)                 # lens centre above the holder bottom when seated
    post_top = axis_local + LENS_R + 0.010
    _cylinder(stage, path + '/Post', 0.004, post_top - sz, (0, 0, sz + (post_top - sz) / 2), (0.25, 0.26, 0.28),
              materials['pmma'])
    if kind == 'plate':             # a glass block along the axis (plate_index), wider than the beam; deep amber so it
        # is not mistaken for the pale blue lens under the lamp
        _box(stage, path + '/Glass', (PLATE_L, 0.034, 0.034), (0, 0, axis_local), (0.82, 0.42, 0.08), collide=False)
    elif kind == 'slit':            # a blackened plate across the axis with a vertical slit (drawn wider than it is)
        _box(stage, path + '/Plate', (0.0015, 2 * LENS_R, 2 * LENS_R), (0, 0, axis_local), (0.08, 0.08, 0.09), collide=False)
        _box(stage, path + '/Slit', (0.0017, 0.0006, 1.6 * LENS_R), (0, 0, axis_local), (0.85, 0.85, 0.82), collide=False)
    elif kind == 'grating':         # a thin square plate across the axis
        _box(stage, path + '/Plate', (0.0015, 2 * LENS_R, 2 * LENS_R), (0, 0, axis_local), (0.55, 0.58, 0.62), collide=False)
    else:
        _cylinder(stage, path + '/Disc', radius, LENS_T, (0, 0, axis_local), colour, collide=False)
        UsdGeom.XformCommonAPI(UsdGeom.Xform(stage.GetPrimAtPath(path + '/Disc'))).SetRotate(Gf.Vec3f(0, 90, 0))
    _box(stage, path + '/Ring', (LENS_T + 0.002, 2 * radius + 0.004, 0.003), (0, 0, axis_local + radius + 0.0015),
         (0.25, 0.26, 0.28), collide=False)
    white = (0.94, 0.94, 0.92)
    stem_h = 0.018
    _cylinder(stage, path + '/HandleStem', FUNNEL_POST_R, stem_h, (0, 0, post_top + stem_h / 2), white, materials['pmma'])
    rim_z = post_top + stem_h
    _cylinder(stage, path + '/HandleRim', FUNNEL_RIM_R, FUNNEL_RIM_H, (0, 0, rim_z + FUNNEL_RIM_H / 2), white,
              materials['pmma'])
    _cylinder(stage, path + '/HandlePort', PORT_R, PORT_H, (0, 0, rim_z + FUNNEL_RIM_H + PORT_H / 2), white, collide=False)
    _text(stage, path + '/Name', label_ or lens, (0.0, 0.0105), rim_z + FUNNEL_RIM_H + 0.0004, 0.008)
    _rigid(xform.GetPrim(), 0.050, (0, 0, 0.02), ccd=True)
    from pxr import PhysxSchema
    PhysxSchema.PhysxRigidBodyAPI.Apply(xform.GetPrim()).CreateMaxDepenetrationVelocityAttr(MAX_DEPEN_MPS)
    return path


def holder_top_local():
    axis_local = AXIS_Z - (CARRIAGE_TOP + SEAT_FLOOR)
    return axis_local + LENS_R + 0.010 + 0.018 + FUNNEL_RIM_H + PORT_H


def latch_local():
    return (0.0, 0.0, holder_top_local() + 0.020)


def build(stage, config):
    """config: {'axes': {axis: initial mm}, 'lens_axes': [...], 'screen_axis': axis, 'lenses': {lens: seat id},
    'lens_colours': {lens: rgb}}. Returns objects (lens holders), buttons, seats (stand seats; carriage seats move).
    config['drive'] == 'hand': the hand-slid bench (build_hand)."""
    if config.get('drive') == 'hand':
        return build_hand(stage, config)
    retire_hooke_apparatus(stage)
    repair_cube_extents(stage)
    z0 = table_top_z(stage)
    root = UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(0, 0, z0))
    materials = {'rubber': _material(stage, '/World/PhysicsMaterials/OpticsRubber', 0.8, 0.7),
                 'pmma': _material(stage, '/World/PhysicsMaterials/OpticsPMMA', 0.25, 0.20),
                 'cell': _material(stage, '/World/PhysicsMaterials/OpticsCell', 0.25, 0.20, compliant=C.CUVETTE_CONTACT),
                 'seat': _material(stage, '/World/PhysicsMaterials/OpticsSeat', 0.20, 0.15)}
    UsdGeom.Xform.Define(stage, ROOT + '/Buttons')
    # Rail with a printed scale every 50 mm.
    _box(stage, ROOT + '/Rail', RAIL, (OBJECT_X + RAIL[0] / 2 - 0.02, RAIL_Y, RAIL[2] / 2), (0.15, 0.16, 0.18),
         material=materials['rubber'])
    for s in range(0, 651, 50):
        x = rail_x(s)
        _box(stage, f'{ROOT}/Tick{s}', (0.0015, 0.010, 0.0006), (x, RAIL_Y + RAIL[1] / 2 + 0.004, 0.0003), C.INK, collide=False)
        if s % 100 == 0:
            _text(stage, f'{ROOT}/TickLabel{s}', str(s), (x, RAIL_Y + RAIL[1] / 2 + 0.016), 0.0004, 0.007)
    # Light box with the illuminated F on its face at the object plane.
    _box(stage, ROOT + '/LightBox', (0.06, 0.10, AXIS_Z + 0.05), (OBJECT_X - 0.03, RAIL_Y, (AXIS_Z + 0.05) / 2),
         (0.10, 0.11, 0.13), material=materials['rubber'])
    import optics_model as M
    if config.get('source') == 'laser':     # a laser module: a bright dot of the laser's colour on the axis
        from circuit_apparatus import wavelength_rgb
        rgb = wavelength_rgb(config['laser_nm'])
        dot = _render_material(stage, ROOT + '/Materials/LaserGlow', rgb, emission=200.0)
        _bind(_cylinder(stage, ROOT + '/LaserAperture', 0.003, 0.0012, (OBJECT_X + 0.0006, RAIL_Y, AXIS_Z), rgb,
                        collide=False), dot)
        UsdGeom.XformCommonAPI(UsdGeom.Xform(stage.GetPrimAtPath(ROOT + '/LaserAperture'))).SetRotate(Gf.Vec3f(0, 90, 0))
        _text(stage, ROOT + '/LaserLabel', 'LASER', (OBJECT_X - 0.03, RAIL_Y + 0.065), 0.0004, 0.010)
    glow = _render_material(stage, ROOT + '/Materials/ObjectGlow', (1.0, 0.95, 0.80), emission=30.0)
    for r, line in ([] if config.get('source') == 'laser' else enumerate(M.F_CELLS)):
        for c, ch in enumerate(line):
            if ch == '1':
                y = RAIL_Y + (c - (len(line) - 1) / 2) * M.F_CELL_M
                z = AXIS_Z + ((len(M.F_CELLS) - 1) / 2 - r) * M.F_CELL_M
                _bind(_box(stage, f'{ROOT}/ObjectF{r}{c}', (0.0008, M.F_CELL_M, M.F_CELL_M), (OBJECT_X + 0.0004, y, z),
                           (1.0, 0.95, 0.80), collide=False), glow)
    # Carriages (kinematic: the backend moves them along the rail).
    seats = {}
    for axis, s in config['axes'].items():
        path = carriage_path(axis)
        xform = UsdGeom.Xform.Define(stage, path)
        UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(rail_x(s), RAIL_Y, RAIL[2]))
        _box(stage, path + '/Block', CARRIAGE, (0, 0, CARRIAGE[2] / 2), (0.30, 0.31, 0.33), material=materials['rubber'])
        _text(stage, path + '/Name', axis, (0.0, CARRIAGE[1] / 2 + 0.010 - 0.0), CARRIAGE[2] + 0.0004, 0.010)
        if axis in config['lens_axes']:
            collar(stage, path + '/Seat', (0.0, 0.0), CARRIAGE[2], (0.95, 0.55, 0.10), materials['seat'])
        if axis == config['screen_axis']:
            _box(stage, path + '/ScreenPost', (0.006, 0.006, AXIS_Z - CARRIAGE_TOP - SCREEN_SIZE / 2 + 0.002),
                 (0, 0, CARRIAGE[2] + (AXIS_Z - CARRIAGE_TOP - SCREEN_SIZE / 2) / 2), (0.25, 0.26, 0.28), collide=False)
            _screen(stage, SCREEN_PATH.format(axis), AXIS_Z - RAIL[2])
        prim = xform.GetPrim()
        from pxr import UsdPhysics
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)
    # Lens stand.
    bx, by, bz = STAND['base']
    _box(stage, ROOT + '/Stand', STAND['base'], (sum(STAND['xs']) / 3, STAND['y'], bz / 2), (0.12, 0.13, 0.15),
         material=materials['rubber'])
    for k, x in enumerate(STAND['xs'], start=1):
        seats[f'stand:{k}'] = collar(stage, f'{ROOT}/StandSeat{k}', (x, STAND['y']), bz, (0.03, 0.16, 0.20), materials['seat'])
    # Lens holders on their initial seats.
    objects = {}
    for lens, sid in config['lenses'].items():
        kind, _, key = sid.partition(':')
        if kind == 'stand':
            xy, base = (STAND['xs'][int(key) - 1], STAND['y']), bz
        else:
            xy, base = (rail_x(config['axes'][key]), RAIL_Y), CARRIAGE_TOP
        objects[lens] = _holder(stage, lens, xy, base + SEAT_FLOOR + 0.0003, materials,
                                tuple(config.get('lens_colours', {}).get(lens, (0.62, 0.80, 0.92))),
                                kind=config.get('kinds', {}).get(lens, 'lens'),
                                radius=min(LENS_R_MAX, config.get('radii', {}).get(lens, LENS_R)))
    # Stage controller keypad.
    xs = KEYPAD['xs']
    rows = len(KEYPAD['rows'])
    _box(stage, ROOT + '/Controller', (0.22, rows * KEYPAD['pitch_y'] + 0.03, BODY_Z),
         (sum(xs) / len(xs), KEYPAD['y0'] + (rows - 1) * KEYPAD['pitch_y'] / 2, BODY_Z / 2), (0.12, 0.26, 0.50),
         material=materials['rubber'])
    buttons = {}
    for bid, (xy, base, kind) in button_layout().items():
        key = bid.split(':', 1)[1]
        colour = (0.95, 0.78, 0.05) if bid.startswith('axis:') else (0.15, 0.62, 0.25) if key == 'go' else (0.20, 0.21, 0.23)
        text = key if bid.startswith('axis:') else KEY_TEXT.get(key, key)
        ink = C.INK if bid.startswith('axis:') else (0.97, 0.97, 0.95)
        buttons[bid] = _button(stage, bid, xy, base, kind, colour, text, materials, ink)
    _text(stage, ROOT + '/ControllerLabel', 'STAGE', (sum(xs) / len(xs), KEYPAD['y0'] - 0.030), BODY_Z + 0.0004, 0.010, C.PAPER)
    table_top = stage.GetPrimAtPath(TABLE_TOP)
    from pxr import PhysxSchema
    contact = PhysxSchema.PhysxCollisionAPI.Apply(table_top)
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    return {'objects': objects, 'buttons': buttons, 'seats': {k: str(v.GetPath()) for k, v in seats.items()}}


def _screen(stage, path, centre_z_local):
    """White screen facing the light box (-x), textured with the computed image."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    h = SCREEN_SIZE / 2
    x = -0.002
    # Seen from the light box (looking +x), image column 0 is at world +y (the viewer's left) and row 0 at the top.
    mesh.CreatePointsAttr([(x, h, centre_z_local - h), (x, -h, centre_z_local - h), (x, -h, centre_z_local + h),
                           (x, h, centre_z_local + h)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([(-1, 0, 0)] * 4)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying)
    st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    mesh.CreateDoubleSidedAttr(True)
    import omni.kit.commands
    omni.kit.commands.execute('CreateMdlMaterialPrimCommand', mtl_url='OmniPBR.mdl', mtl_name='OmniPBR',
                              mtl_path=ROOT + '/Materials/ScreenImage')
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath(ROOT + '/Materials/ScreenImage')))
    _box(stage, path + 'Frame', (0.004, SCREEN_SIZE + 0.008, SCREEN_SIZE + 0.008), (0.001, 0, centre_z_local),
         (0.85, 0.85, 0.83), collide=False)


def set_screen_image(stage, png_path):
    """Point the screen material at a new image file (a new path forces the renderer to load it)."""
    shader = UsdShade.Shader(stage.GetPrimAtPath(ROOT + '/Materials/ScreenImage/Shader'))
    for name in ('diffuse_texture', 'emissive_color_texture'):
        attr = shader.GetInput(name) or shader.CreateInput(name, Sdf.ValueTypeNames.Asset)
        attr.Set(Sdf.AssetPath(str(png_path)))
    (shader.GetInput('enable_emission') or shader.CreateInput('enable_emission', Sdf.ValueTypeNames.Bool)).Set(True)
    (shader.GetInput('emissive_intensity') or shader.CreateInput('emissive_intensity', Sdf.ValueTypeNames.Float)).Set(800.0)
    TEXTURE['path'] = str(png_path)


def set_carriage(stage, axis, s_mm, body=None):
    """Move a carriage to rail coordinate s (kinematic body pose; the seated lens rides on it)."""
    x = rail_x(s_mm)
    if body is not None:                 # the physics body (the seated lens rides on it) ...
        position, orientation = body.get_world_pose()
        body.set_world_pose(position=np.array([x, position[1], position[2]]), orientation=orientation)
    # ... and the USD pose the renderer draws (a kinematic body moved through physics is not written back)
    UsdGeom.XformCommonAPI(UsdGeom.Xform(stage.GetPrimAtPath(carriage_path(axis)))).SetTranslate(
        Gf.Vec3d(x, RAIL_Y, RAIL[2]))


author_joints = C.author_joints
seat_frames = C.seat_frames


# --- the hand-slid bench (recipe optics_hand_bench: lens_focal, grating_wavelength) --------------------------------
# Riders slide on the rail by hand: each is a dynamic body on a prismatic rail joint (world x) whose drive is a
# saturated damper, i.e. rail friction of FRICTION_N: at rest a rider holds wherever it was released (a force below
# FRICTION_N does not move it), pushed it follows the push, and released at v it coasts v^2 m / (2 FRICTION_N)
# (0.07 mm at 50 mm/s). The joint limits are the end stops; the riders' rubber bumpers keep their index marks at
# least HAND_MIN_GAP_MM apart. A rider carries a push handle on an arm toward the robot (on the side away from the
# other rider), an index pointer with a red index line over the scale, and (lens riders) a seat collar, a
# kinematic body that follows the rider every tick. The screen rider carries the graduated screen, its surface at the
# index mark. A white scale bar runs along the rail's robot side: 1 mm ticks, longer every 5 and 10 mm, numbers every
# 10 mm. The source (light box or laser module) has a POWER push button and a green pilot lamp on its top.
HAND_RAIL = (0.70, 0.030, 0.020)            # rail coordinate -20 to 680 mm
RIDER_GAP = 0.0005                          # rider underside above the rail top (the rail joint carries it)
RIDER_BLOCK = (0.050, 0.044, 0.019)         # collar base lands at CARRIAGE_TOP: the element on the axis
BUMPER = (0.005, 0.030, 0.014)
HAND_MIN_GAP_MM = 60.0                      # block 50 mm + two 5 mm bumpers
HANDLE_SIZE = (0.016, 0.016, 0.064)
HANDLE_DX = 0.032                           # handle centre from the index mark, on the side away from the other rider
HANDLE_SIDE = {'A': -1.0, 'B': -1.0, 'S': 1.0}
HANDLE_Y = 0.105                            # handle centre toward the robot from the rail axis
HANDLE_Z0 = 0.006                           # arm and handle bottom above the rider underside
ARM_THICK = 0.008
POINTER = (0.0016, 0.012, 0.0010)           # index pointer from the block face over the tick roots (render only)
SCALE_Y = (0.015, 0.045)                    # scale bar across (from the rail axis toward the robot)
SCALE_TOP = RAIL[2] - 0.0005
TICK_ROOT_Y = 0.0235
TICKS = {1: (0.0030, 0.00025), 5: (0.0050, 0.00030), 10: (0.0070, 0.00040)}     # length, width
NUMBER_H, NUMBER_Y = 0.0040, 0.0350
SCALE_RANGE_MM = (-15, 680)
RIDER_MASS = {'lens': 0.40, 'screen': 0.55}
FRICTION_N = 3.0
FRICTION_DAMPING = 2.0e4                    # N s/m: saturates at FRICTION_N above 0.15 mm/s
# Riders and lens holders: an overlap with the fingers (a forced descent onto a handle) is resolved at most this fast,
# not fired out (a 9 mm stroke had ejected a rider 27-56 mm along the rail; normal pushes unchanged).
MAX_DEPEN_MPS = 0.1
SOURCE_BOX = (0.06, 0.10, AXIS_Z + 0.05)
POWER_LOCAL = (-0.030, 0.026)               # POWER button on the source's top, from (OBJECT_X, RAIL_Y)
LAMP_LOCAL = (-0.030, -0.022)
LAMP_R, LAMP_H = 0.0045, 0.003
LAMP_OFF, LAMP_ON = (0.04, 0.10, 0.05), (0.20, 1.0, 0.30)
F_GLOW = (1.0, 0.95, 0.80)
F_GLOW_EMISSION = 30.0
SCREEN_MATERIAL = ROOT + '/Materials/ScreenImage'
SEAT_BODY = ROOT + '/Seat{}'
RAIL_JOINT = C.PHYSICS_ROOT + '/Rail{}'
POWER_ID = 'power'


def rider_path(axis):
    return carriage_path(axis)


def seat_body_path(axis):
    return SEAT_BODY.format(axis)


def hand_seat_path(axis):
    return seat_body_path(axis) + '/Collar'


def power_xy():
    return (OBJECT_X + POWER_LOCAL[0], RAIL_Y + POWER_LOCAL[1])


def _scale(stage, z_top):
    """The white scale bar along the rail's robot side with its ticks and numbers (one mesh each, render only)."""
    x0, x1 = rail_x(SCALE_RANGE_MM[0]), rail_x(SCALE_RANGE_MM[1])
    y0, y1 = RAIL_Y + SCALE_Y[0], RAIL_Y + SCALE_Y[1]
    _box(stage, ROOT + '/ScaleBar', (x1 - x0, y1 - y0, z_top), ((x0 + x1) / 2, (y0 + y1) / 2, z_top / 2),
         (0.20, 0.21, 0.23), material=None)
    _box(stage, ROOT + '/ScaleFace', (x1 - x0, y1 - y0, 0.0004), ((x0 + x1) / 2, (y0 + y1) / 2, z_top + 0.0002),
         C.PAPER, collide=False)
    from circuit_apparatus import _text_rectangles
    z = z_top + 0.0005
    points, counts, indices = [], [], []

    def quad(xa, xb, ya, yb):
        base = len(points)
        points.extend([Gf.Vec3f(xa, ya, z), Gf.Vec3f(xb, ya, z), Gf.Vec3f(xb, yb, z), Gf.Vec3f(xa, yb, z)])
        counts.append(4)
        indices.extend([base, base + 1, base + 2, base + 3])

    root = RAIL_Y + TICK_ROOT_Y
    for s in range(0, SCALE_RANGE_MM[1] + 1):
        length, width = TICKS[10 if s % 10 == 0 else 5 if s % 5 == 0 else 1]
        x = rail_x(s)
        quad(x - width / 2, x + width / 2, root, root + length)
    # numbers every 10 mm, upright from the robot's side (reading toward -x), centred on their ticks
    for s in range(0, SCALE_RANGE_MM[1] + 1, 10):
        rects, width = _text_rectangles(str(s), NUMBER_H)
        cx, cy = rail_x(s), RAIL_Y + NUMBER_Y
        for a, b, c, d in rects:                 # text frame: u along the reading direction (-x), v up (-y)
            u0, u1 = a - width / 2, c - width / 2
            v0, v1 = b - NUMBER_H / 2, d - NUMBER_H / 2
            quad(cx - u1, cx - u0, cy - v1, cy - v0)
    for name, sub in (('ScaleMarks', None),):
        mesh = UsdGeom.Mesh.Define(stage, ROOT + '/' + name)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDoubleSidedAttr(True)
        arr = np.asarray([[p[0], p[1], p[2]] for p in points])
        mesh.CreateExtentAttr([Gf.Vec3f(*arr.min(0).tolist()), Gf.Vec3f(*arr.max(0).tolist())])
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*C.INK)])


def _rider(stage, axis, s, kind, materials, label_text):
    """A rider at rail coordinate s: dynamic body (block, bumpers, arm, handle; pointer and index line drawn)."""
    from pxr import PhysxSchema, UsdPhysics
    path = rider_path(axis)
    xform = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(rail_x(s), RAIL_Y, RAIL[2] + RIDER_GAP))
    bx, by, bz = RIDER_BLOCK
    grey = (0.30, 0.31, 0.33)
    _box(stage, path + '/Block', RIDER_BLOCK, (0, 0, bz / 2), grey, material=materials['rubber'])
    for k, sx in enumerate((-1, 1)):
        _box(stage, f'{path}/Bumper{k}', BUMPER, (sx * (bx / 2 + BUMPER[0] / 2), 0, bz / 2), (0.08, 0.08, 0.09),
             material=materials['rubber'])
    side = HANDLE_SIDE.get(axis, -1.0)
    hx = side * HANDLE_DX
    arm_x = (side * 0.012, side * (HANDLE_DX + HANDLE_SIZE[0] / 2))
    arm_y = (0.010, HANDLE_Y + HANDLE_SIZE[1] / 2)
    _box(stage, path + '/Arm', (abs(arm_x[1] - arm_x[0]), arm_y[1] - arm_y[0], ARM_THICK),
         (sum(arm_x) / 2, sum(arm_y) / 2, HANDLE_Z0 + ARM_THICK / 2), grey, material=materials['rubber'])
    _box(stage, path + '/Handle', HANDLE_SIZE, (hx, HANDLE_Y, HANDLE_Z0 + HANDLE_SIZE[2] / 2), (0.92, 0.48, 0.10),
         material=materials['rubber'])
    _text(stage, path + '/Name', label_text, (hx, HANDLE_Y), HANDLE_Z0 + HANDLE_SIZE[2] + 0.0004, 0.009)
    # index pointer from the block's robot face over the tick roots, its red index line on the rider's index mark
    py = by / 2 + POINTER[1] / 2
    _box(stage, path + '/Pointer', POINTER, (0, py, -0.0001), (0.12, 0.12, 0.13), collide=False)
    _box(stage, path + '/IndexLine', (0.0005, POINTER[1], 0.0002), (0, py, -0.0001 + POINTER[2] / 2 + 0.0001),
         (0.85, 0.05, 0.05), collide=False)
    _box(stage, path + '/IndexLineTop', (0.0005, 0.010, 0.0002), (0, by / 2 - 0.005, bz + 0.0001), (0.85, 0.05, 0.05),
         collide=False)
    if kind == 'screen':
        centre_z = AXIS_Z - (RAIL[2] + RIDER_GAP)
        bottom = centre_z - SCREEN_SIZE / 2
        _box(stage, path + '/ScreenPost', (0.006, 0.010, bottom - bz + 0.002), (0.0035, 0, bz + (bottom - bz) / 2),
             (0.25, 0.26, 0.28), material=materials['rubber'])
        _screen_hand(stage, path + '/Screen', centre_z, materials)
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim()).CreateKinematicEnabledAttr(False)
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(xform.GetPrim())
    body.CreateSolverPositionIterationCountAttr(16)
    body.CreateSolverVelocityIterationCountAttr(8)
    body.CreateSleepThresholdAttr(0.0)
    body.CreateLinearDampingAttr(0.0)
    body.CreateAngularDampingAttr(0.0)
    body.CreateMaxDepenetrationVelocityAttr(MAX_DEPEN_MPS)
    mass = UsdPhysics.MassAPI.Apply(xform.GetPrim())
    mass.CreateMassAttr(RIDER_MASS[kind])
    return path


def _rail_joint(stage, axis, s, travel):
    """Prismatic joint along world x from the world to the rider, limits at the travel ends, friction drive."""
    from pxr import UsdPhysics
    UsdGeom.Scope.Define(stage, C.PHYSICS_ROOT)
    world = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(rider_path(axis)))
    joint = UsdPhysics.PrismaticJoint.Define(stage, RAIL_JOINT.format(axis))
    joint.CreateBody1Rel().SetTargets([Sdf.Path(rider_path(axis))])
    joint.CreateLocalPos0Attr(Gf.Vec3f(world.ExtractTranslation()))
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateAxisAttr('X')
    joint.CreateLowerLimitAttr((travel[0] - s) / 1000.0)
    joint.CreateUpperLimitAttr((travel[1] - s) / 1000.0)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
    drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(FRICTION_DAMPING)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateMaxForceAttr(FRICTION_N)
    return str(joint.GetPath())


def _seat_body(stage, axis, s, materials):
    """The lens rider's seat collar: a kinematic body the backend keeps on the rider (set_seat)."""
    from pxr import UsdPhysics
    path = seat_body_path(axis)
    xform = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(rail_x(s), RAIL_Y, CARRIAGE_TOP))
    collar(stage, path + '/Collar', (0.0, 0.0), 0.0, (0.95, 0.55, 0.10), materials['seat'])
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim()).CreateKinematicEnabledAttr(True)
    return path


def _screen_hand(stage, path, centre_z_local, materials):
    """The graduated screen, its surface at the rider's index mark facing the source (-x); a frame behind it."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    h = SCREEN_SIZE / 2
    x = 0.0
    mesh.CreatePointsAttr([(x, h, centre_z_local - h), (x, -h, centre_z_local - h), (x, -h, centre_z_local + h),
                           (x, h, centre_z_local + h)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([(-1, 0, 0)] * 4)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying)
    st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    mesh.CreateDoubleSidedAttr(True)
    import omni.kit.commands
    omni.kit.commands.execute('CreateMdlMaterialPrimCommand', mtl_url='OmniPBR.mdl', mtl_name='OmniPBR',
                              mtl_path=SCREEN_MATERIAL)
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath(SCREEN_MATERIAL)))
    _box(stage, path + 'Frame', (0.004, SCREEN_SIZE + 0.008, SCREEN_SIZE + 0.008), (0.0025, 0, centre_z_local),
         (0.85, 0.85, 0.83), material=materials['rubber'])


def _source(stage, config, materials):
    """Light box (the illuminated F) or laser module, with its POWER button and pilot lamp; everything off."""
    import optics_model as M
    _box(stage, ROOT + '/LightBox', SOURCE_BOX, (OBJECT_X - 0.03, RAIL_Y, SOURCE_BOX[2] / 2), (0.10, 0.11, 0.13),
         material=materials['rubber'])
    top = SOURCE_BOX[2]
    if config.get('source') == 'laser':
        from circuit_apparatus import wavelength_rgb
        rgb = wavelength_rgb(config['laser_nm'])
        dot = _render_material(stage, ROOT + '/Materials/LaserGlow', (0.12, 0.12, 0.12))
        _bind(_cylinder(stage, ROOT + '/LaserAperture', 0.003, 0.0012, (OBJECT_X + 0.0006, RAIL_Y, AXIS_Z), rgb,
                        collide=False), dot)
        UsdGeom.XformCommonAPI(UsdGeom.Xform(stage.GetPrimAtPath(ROOT + '/LaserAperture'))).SetRotate(Gf.Vec3f(0, 90, 0))
        _text(stage, ROOT + '/LaserLabel', 'LASER', (OBJECT_X - 0.03, RAIL_Y - 0.004), top + 0.0004, 0.010, C.PAPER)
    else:
        glow = _render_material(stage, ROOT + '/Materials/ObjectGlow', F_GLOW)
        for r, line in enumerate(M.F_CELLS):
            for c, ch in enumerate(line):
                if ch == '1':
                    y = RAIL_Y + (c - (len(line) - 1) / 2) * M.F_CELL_M
                    z = AXIS_Z + ((len(M.F_CELLS) - 1) / 2 - r) * M.F_CELL_M
                    _bind(_box(stage, f'{ROOT}/ObjectF{r}{c}', (0.0008, M.F_CELL_M, M.F_CELL_M), (OBJECT_X + 0.0004, y, z),
                               F_GLOW, collide=False), glow)
    px, py = power_xy()
    button = _button(stage, POWER_ID, (px, py), top, 'round', (0.80, 0.10, 0.08), '', materials)
    _text(stage, ROOT + '/PowerLabel', 'POWER', (px + 0.021, py), top + 0.0004, 0.006, C.PAPER)
    lamp = _render_material(stage, ROOT + '/Materials/PowerLamp', LAMP_OFF)
    _bind(_cylinder(stage, ROOT + '/PowerLamp', LAMP_R, LAMP_H, (OBJECT_X + LAMP_LOCAL[0], RAIL_Y + LAMP_LOCAL[1],
                                                                 top + LAMP_H / 2), LAMP_OFF, collide=False), lamp)
    return {POWER_ID: button}


def _set_emission(stage, material_path, colour, emission):
    shader = UsdShade.Shader(stage.GetPrimAtPath(material_path + '/Shader'))
    if not shader:
        return
    attr = shader.GetInput('emissiveColor') or shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f)
    attr.Set(Gf.Vec3f(*[c * emission for c in colour]))


def set_power(stage, on, source='object', laser_nm=0.0):
    """The source switched on or off: pilot lamp, the illuminated F or the laser's exit aperture."""
    _set_emission(stage, ROOT + '/Materials/PowerLamp', LAMP_ON if on else LAMP_OFF, 3.0 if on else 0.0)
    if source == 'laser':
        from circuit_apparatus import wavelength_rgb
        _set_emission(stage, ROOT + '/Materials/LaserGlow', wavelength_rgb(laser_nm), 200.0 if on else 0.0)
    else:
        _set_emission(stage, ROOT + '/Materials/ObjectGlow', F_GLOW, F_GLOW_EMISSION if on else 0.0)


def build_hand(stage, config):
    """config: {'drive': 'hand', 'axes': {axis: initial mm}, 'travel_mm': {axis: (lo, hi)}, 'lens_axes', 'screen_axis',
    'lenses': {element: seat id}, 'lens_colours', 'kinds', 'radii', 'source', 'laser_nm'}. Returns objects (holders),
    buttons ({'power': cap path}), seats (stand seats), joints ({axis: rail joint path})."""
    retire_hooke_apparatus(stage)
    repair_cube_extents(stage)
    z0 = table_top_z(stage)
    root = UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(0, 0, z0))
    materials = {'rubber': _material(stage, '/World/PhysicsMaterials/OpticsRubber', 0.8, 0.7),
                 'pmma': _material(stage, '/World/PhysicsMaterials/OpticsPMMA', 0.25, 0.20),
                 'cell': _material(stage, '/World/PhysicsMaterials/OpticsCell', 0.25, 0.20, compliant=C.CUVETTE_CONTACT),
                 'seat': _material(stage, '/World/PhysicsMaterials/OpticsSeat', 0.20, 0.15)}
    UsdGeom.Xform.Define(stage, ROOT + '/Buttons')
    _box(stage, ROOT + '/Rail', HAND_RAIL, (OBJECT_X + HAND_RAIL[0] / 2 - 0.02, RAIL_Y, HAND_RAIL[2] / 2),
         (0.15, 0.16, 0.18), material=materials['rubber'])
    _scale(stage, SCALE_TOP)
    buttons = _source(stage, config, materials)
    joints, seats = {}, {}
    for axis, s in config['axes'].items():
        kind = 'screen' if axis == config['screen_axis'] else 'lens'
        _rider(stage, axis, s, kind, materials, axis)
        if kind == 'screen' and config.get('source') == 'laser':
            _laser_spots(stage, axis)
        joints[axis] = _rail_joint(stage, axis, s, config['travel_mm'][axis])
        if axis in config['lens_axes']:
            _seat_body(stage, axis, s, materials)
    # Holder stand.
    bx, by, bz = STAND['base']
    _box(stage, ROOT + '/Stand', STAND['base'], (sum(STAND['xs']) / 3, STAND['y'], bz / 2), (0.12, 0.13, 0.15),
         material=materials['rubber'])
    for k, x in enumerate(STAND['xs'], start=1):
        seats[f'stand:{k}'] = collar(stage, f'{ROOT}/StandSeat{k}', (x, STAND['y']), bz, (0.03, 0.16, 0.20), materials['seat'])
        # The seat's number: a white label on the collar top on the robot's side, the digit in black (render only).
        ly = STAND['y'] + (C.MOUTH_HALF + C.COLLAR_HALF) / 2
        _box(stage, f'{ROOT}/StandSeatLabel{k}', (0.018, 0.0085, 0.0004), (x, ly, bz + C.COLLAR_H + 0.0002), C.PAPER,
             collide=False)
        _text(stage, f'{ROOT}/StandSeatNumber{k}', str(k), (x, ly), bz + C.COLLAR_H + 0.0006, 0.0075, C.INK)
    objects = {}
    for lens, sid in config['lenses'].items():
        kind, _, key = sid.partition(':')
        if kind == 'stand':
            xy, base = (STAND['xs'][int(key) - 1], STAND['y']), bz
        else:
            xy, base = (rail_x(config['axes'][key]), RAIL_Y), CARRIAGE_TOP
        objects[lens] = _holder(stage, lens, xy, base + SEAT_FLOOR + 0.0003, materials,
                                tuple(config.get('lens_colours', {}).get(lens, (0.62, 0.80, 0.92))),
                                kind=config.get('kinds', {}).get(lens, 'lens'),
                                radius=min(LENS_R_MAX, config.get('radii', {}).get(lens, LENS_R)))
    table_top = stage.GetPrimAtPath(TABLE_TOP)
    from pxr import PhysxSchema
    contact = PhysxSchema.PhysxCollisionAPI.Apply(table_top)
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    return {'objects': objects, 'buttons': buttons, 'seats': {k: str(v.GetPath()) for k, v in seats.items()},
            'joints': joints}


# Laser spots (render only): the card texture alone caps a spot at white paper, dimmer than a laser spot
# is. Each spot also gets a glowing core (about 1.7 mm across) and a soft two-step halo, discs lying on the card at the
# spot's centre on the beam line; they ride with the screen. A disc that would overhang the card is not drawn. The glow
# has the laser's hue (as its exit aperture), the core whitened as a camera saturates on a laser spot; the discs' own
# colour is that of the lit card, so they only add light.
SPOT_ORDERS = range(-3, 4)
SPOT_LAYERS = ((0.0018, 0.08, -0.00002, 0.0), (0.0013, 0.30, -0.00004, 0.20),
               (0.00085, 1.0, -0.00006, 0.55))       # radius, glow, off the card (tens of um: no parallax), whitening
SPOT_GLOW = 40.0                    # core emission of a spot of level 1
SPOT_BASE = (0.33, 0.33, 0.33)      # the lit card as it renders (wrist view: card ~100, rim of albedo 0.85 ~160)


def _spot_path(axis, m, layer):
    return f'{rider_path(axis)}/LaserSpot{m + 3}/Disc{layer}'


def _spot_material(m, layer):
    return f'{ROOT}/Materials/LaserSpot{m + 3}_{layer}'


def _laser_spots(stage, axis):
    """The glow discs of the laser spots on the screen rider (hidden until a spot shows)."""
    centre_z = AXIS_Z - (RAIL[2] + RIDER_GAP)
    for m in SPOT_ORDERS:
        slot = UsdGeom.Xform.Define(stage, f'{rider_path(axis)}/LaserSpot{m + 3}')
        UsdGeom.XformCommonAPI(slot).SetTranslate(Gf.Vec3d(0.0, 0.0, centre_z))
        for layer, (radius, _, dx, _) in enumerate(SPOT_LAYERS):
            disc = _cylinder(stage, _spot_path(axis, m, layer), radius, 0.00001, (dx, 0.0, 0.0), SPOT_BASE, collide=False)
            UsdGeom.XformCommonAPI(disc).SetRotate(Gf.Vec3f(0, 90, 0))
            _bind(disc, _render_material(stage, _spot_material(m, layer), SPOT_BASE, matte=True))
        UsdGeom.Imageable(slot).MakeInvisible()


def set_laser_spots(stage, axis, spots, laser_nm):
    """Show the glow of the laser spots [(order, screen mm, level)] (screen mm + to the viewer's right as seen from
    the source, i.e. world -y) in the laser's hue; every other order's glow hidden."""
    from circuit_apparatus import wavelength_rgb
    hue = wavelength_rgb(laser_nm)
    shown = {m: (x_mm, level) for m, x_mm, level in spots
             if abs(x_mm) / 1000.0 + SPOT_LAYERS[0][0] <= SCREEN_SIZE / 2}
    centre_z = AXIS_Z - (RAIL[2] + RIDER_GAP)
    for m in SPOT_ORDERS:
        prim = stage.GetPrimAtPath(f'{rider_path(axis)}/LaserSpot{m + 3}')
        if not prim:
            continue
        slot = UsdGeom.Xform(prim)
        if m not in shown:
            UsdGeom.Imageable(slot).MakeInvisible()
            continue
        x_mm, level = shown[m]
        UsdGeom.XformCommonAPI(slot).SetTranslate(Gf.Vec3d(0.0, -x_mm / 1000.0, centre_z))
        for layer, (_, glow, _, white) in enumerate(SPOT_LAYERS):
            tint = tuple((1.0 - white) * c + white for c in hue)
            _set_emission(stage, _spot_material(m, layer), tint, SPOT_GLOW * level * glow)
        UsdGeom.Imageable(slot).MakeVisible()


def set_screen_textures(stage, lit_png):
    """The screen card as lit (printed card under the room light plus the projected light) as its diffuse texture;
    a new path forces the renderer to load it. (A textured OmniPBR emission does not show in these renders.)"""
    shader = UsdShade.Shader(stage.GetPrimAtPath(SCREEN_MATERIAL + '/Shader'))
    attr = shader.GetInput('diffuse_texture') or shader.CreateInput('diffuse_texture', Sdf.ValueTypeNames.Asset)
    attr.Set(Sdf.AssetPath(str(lit_png)))
    TEXTURE['path'] = str(lit_png)


SCREEN_EMISSIVE_INTENSITY = 800.0


def set_seat(stage, axis, s_mm, body=None):
    """Keep a lens rider's seat collar on the rider (kinematic body pose and the USD pose the renderer draws)."""
    x = rail_x(s_mm)
    if body is not None:
        position, orientation = body.get_world_pose()
        body.set_world_pose(position=np.array([x, position[1], position[2]]), orientation=orientation)
    UsdGeom.XformCommonAPI(UsdGeom.Xform(stage.GetPrimAtPath(seat_body_path(axis)))).SetTranslate(
        Gf.Vec3d(x, RAIL_Y, CARRIAGE_TOP))

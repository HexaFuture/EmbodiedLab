"""Dry-block bench apparatus (thermistor_beta, rod_expansion): a dry-block heater with a well (orange collar) and
its controller screen (SET and the block's reference thermometer) on the front face, the probe on its stand (teal
collar), the set-point caps; thermistor_beta: the bench ohmmeter (4.5-digit LCD) with its RANGE and AUTO keys;
rod_expansion: the probe's digital thermometer, the heating jacket with the rod, a steel rule beside it (zero at the
rod's fixed stop) and a digital indicator (LCD, 0.001 mm) on the rod's free end. Built with the chemistry kit's
helpers on the same root, so the chemistry backend's grasp, seat and button machinery runs it unchanged (as the
calorimetry bench).

From the robot (at (0.95, 0.35), facing -y): the dry block on its right, the probe stand on its left with the probe
standing in its collar (dark stainless body, white rim handle on top, as the calorimetry sample), the meter on a post
left of the stand, the red UP and the purple DN cap in front of the block (the meter's keys in front of the meter).
rod_expansion: the jacket lies behind them along x (render only), the stop at its right end, the indicator at its left.
"""
from __future__ import annotations

import numpy as np
from pxr import Gf, UsdGeom

import chem_apparatus as C
from calor_apparatus import SAMPLE_BODY, SAMPLE_FOOT, SAMPLE_POST, SAMPLE_STEM
from chem_apparatus import FUNNEL_POST_R, FUNNEL_RIM_H, FUNNEL_RIM_R, PORT_H, PORT_R, ROOT, SEAT_FLOOR, _button, _rigid, _text, collar
from circuit_apparatus import label
from torsion_apparatus import TABLE_TOP, _box, _cylinder, _material, repair_cube_extents, retire_hooke_apparatus, table_top_z
import thermal_lcd as L

PROBE_MASS_KG = 0.030                          # rigid-body mass of the probe

BLOCK = {'centre': (0.84, -0.06), 'size': (0.10, 0.10, 0.040)}
STAND = {'centre': (1.06, -0.06), 'size': (0.08, 0.08, 0.020)}
BUTTONS = {'down': ((0.80, 0.035), (0.50, 0.25, 0.75), 'DN', 'SET -5 C'), 'up': ((0.88, 0.035), (0.80, 0.10, 0.08), 'UP', 'SET +5 C')}
METER_KEYS = {'range': ((1.140, 0.035), (0.35, 0.65, 0.90), 'RNG', 'RANGE'), 'auto': ((1.190, 0.035), (0.15, 0.62, 0.25), 'AUTO', 'AUTO')}
METER_XY = (1.165, -0.030)               # the ohmmeter (thermistor_beta) or the probe's thermometer (rod_expansion)
JACKET = {'centre': (0.95, -0.20), 'length': 0.17, 'radius': 0.018}   # shorter than the rod: both rod ends show
ROD_R = 0.005
STOP_FACE_X = 0.85                       # the fixed stop's face: the rod's right end (as the robot sees it), the rule's zero
ROD_Z = 0.021                            # rod axis above the table (it lies on the jacket floor)
RULE = {'length_mm': 215, 'width': 0.016, 'thickness': 0.001, 'gap': 0.001}   # steel rule, its top level with the rod top
INDICATOR = {'stem': 0.014, 'base': (0.068, 0.030, 0.025)}
PROBE_PATH = ROOT + '/Probe'
CONTROLLER_PATH = ROOT + '/BlockController'
METER_PATH = ROOT + '/Meter'
INDICATOR_PATH = ROOT + '/Indicator'


def seat_paths(stage=None):
    return {'well': ROOT + '/BlockSeat', 'stand': ROOT + '/StandSeat'}


def _buttons(kind):
    return BUTTONS | (METER_KEYS if kind == 'thermistor' else {})


def button_paths(stage=None, kind='thermistor'):
    return {b: C.button_path(b) for b in _buttons(kind)}


def button_layout(kind='thermistor'):
    return {b: (spec[0], 0.0, 'round') for b, spec in _buttons(kind).items()}


def body_centre_local():
    return SAMPLE_FOOT[2] + SAMPLE_BODY[1] / 2


def handle_top_local():
    return SAMPLE_FOOT[2] + SAMPLE_BODY[1] + SAMPLE_POST + SAMPLE_STEM + FUNNEL_RIM_H + PORT_H


def latch_local():
    return (0.0, 0.0, handle_top_local() + 0.020)


def _probe(stage, xy, bottom_z, materials):
    """The calorimetry sample's shape (cuvette-sized foot, the same handle), stainless with a dark sensor tip."""
    xform = UsdGeom.Xform.Define(stage, PROBE_PATH)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(xy[0], xy[1], bottom_z))
    fz = SAMPLE_FOOT[2]
    steel = (0.35, 0.36, 0.39)                     # dark enough not to read as a white rim
    _box(stage, PROBE_PATH + '/Foot', SAMPLE_FOOT, (0, 0, fz / 2), (0.10, 0.10, 0.11), material=materials['cell'])
    r, h = SAMPLE_BODY
    _cylinder(stage, PROBE_PATH + '/Body', r, h, (0, 0, fz + h / 2), steel, materials['pmma'])
    z = fz + h
    _cylinder(stage, PROBE_PATH + '/Post', 0.004, SAMPLE_POST, (0, 0, z + SAMPLE_POST / 2), (0.25, 0.26, 0.28), materials['pmma'])
    z += SAMPLE_POST
    white = (0.94, 0.94, 0.92)
    _cylinder(stage, PROBE_PATH + '/HandleStem', FUNNEL_POST_R, SAMPLE_STEM, (0, 0, z + SAMPLE_STEM / 2), white, materials['pmma'])
    z += SAMPLE_STEM
    _cylinder(stage, PROBE_PATH + '/HandleRim', FUNNEL_RIM_R, FUNNEL_RIM_H, (0, 0, z + FUNNEL_RIM_H / 2), white, materials['pmma'])
    z += FUNNEL_RIM_H
    _cylinder(stage, PROBE_PATH + '/HandlePort', PORT_R, PORT_H, (0, 0, z + PORT_H / 2), white, collide=False)
    _rigid(xform.GetPrim(), PROBE_MASS_KG, (0, 0, fz + h / 2), ccd=True)
    return PROBE_PATH


def rod_top_z():
    return ROD_Z + ROD_R


def rule_y():
    """World y of the rule's graduated edge (the one toward the rod) and of its far edge (toward the robot)."""
    jy = JACKET['centre'][1]
    near = jy + 0.012 + 0.006 + RULE['gap']          # outside the jacket's front wall
    return near, near + RULE['width']


def _jacket(stage, config, looks):
    """rod_expansion, render only (the indicator is read from its LCD and the log). Colours stay clear of the
    reference's cues (the probe's white rim: S <= 12, V >= 228; the red and purple caps; the orange and teal collars):
    a matte black heating jacket, open on top, its walls level with the rod's top, along x behind the block and fed
    from it by two hoses; the aluminium rod (L0 to scale) lies in it with both ends out, the right one (as the robot
    sees it) against a clamped stop, the left one against the plunger of a digital indicator. A steel rule lies
    beside the rod in front of the jacket, its top level with the rod's top, its zero end against the stop's face,
    graduated in millimetres along the edge toward the rod and numbered every 10 mm."""
    (jx, jy), length = JACKET['centre'], JACKET['length']
    l0 = config['rod_l0_mm'] / 1000
    base_h, floor_h, wall_t, inner_w = 0.010, 0.006, 0.006, 0.024
    wall_h = rod_top_z() - base_h
    black, alu, chrome, ink = (0.07, 0.07, 0.08), (0.70, 0.71, 0.73), (0.60, 0.61, 0.64), (0.03, 0.03, 0.03)
    _box(stage, ROOT + '/JacketFoot', (0.36, 0.07, base_h), (jx + 0.02, jy, base_h / 2), (0.20, 0.21, 0.23), collide=False)
    _box(stage, ROOT + '/Jacket', (length, inner_w + 2 * wall_t, floor_h), (jx, jy, base_h + floor_h / 2), black, collide=False)
    for side in (-1, 1):
        _box(stage, f'{ROOT}/JacketWall{side + 1}', (length, wall_t, wall_h),
             (jx, jy + side * (inner_w + wall_t) / 2, base_h + wall_h / 2), black, collide=False)
    x0, x1 = STOP_FACE_X, STOP_FACE_X + l0
    rod = _cylinder(stage, ROOT + '/Rod', ROD_R, l0, ((x0 + x1) / 2, jy, ROD_Z), alu, collide=False)
    UsdGeom.XformCommonAPI(UsdGeom.Xform(rod)).SetRotate(Gf.Vec3f(0, 90, 0))
    for k, hx in enumerate((0.872, 0.884)):          # hoses from the block's back face to the jacket's front wall
        y0, y1 = BLOCK['centre'][1] - BLOCK['size'][1] / 2, jy + inner_w / 2 + wall_t
        hose = _cylinder(stage, f'{ROOT}/JacketHose{k}', 0.0035, y0 - y1, (hx, (y0 + y1) / 2, 0.016), black, collide=False)
        UsdGeom.XformCommonAPI(UsdGeom.Xform(hose)).SetRotate(Gf.Vec3f(90, 0, 0))
    _box(stage, ROOT + '/RodStop', (0.016, 0.05, 0.045), (x0 - 0.008, jy, base_h + 0.0225), (0.16, 0.19, 0.26), collide=False)
    _cylinder(stage, ROOT + '/RodStopScrew', 0.006, 0.004, (x0 - 0.008, jy, base_h + 0.047), chrome, collide=False)
    # The steel rule.
    near, far = rule_y()
    top = rod_top_z()
    n = RULE['length_mm']
    _box(stage, ROOT + '/Rule', (n / 1000, far - near, RULE['thickness']), (x0 + n / 2000, (near + far) / 2,
         top - RULE['thickness'] / 2), (0.80, 0.81, 0.82), collide=False)
    for k, x in enumerate((x0 + 0.006, x0 + n / 1000 - 0.006)):
        _box(stage, f'{ROOT}/RuleSupport{k}', (0.008, far - near - 0.004, top - RULE['thickness'] - base_h),
             (x, (near + far) / 2, (top - RULE['thickness'] + base_h) / 2), (0.25, 0.26, 0.28), collide=False)
    for mm in range(n + 1):
        tick = 0.006 if mm % 10 == 0 else 0.0045 if mm % 5 == 0 else 0.003
        _box(stage, f'{ROOT}/RuleTick{mm:03d}', (0.00035, tick, 0.0002), (x0 + mm / 1000, near + tick / 2, top + 0.0001), ink,
             collide=False)
        if mm % 10 == 0:
            label(stage, f'{ROOT}/RuleNumber{mm:03d}', str(mm), (x0 + mm / 1000, near + 0.0105), top + 0.0002,
                  height_m=0.0035, align='centre', colour=ink, up=C.TEXT_UP)
    # The digital indicator: its plunger on the rod's free end, the head on a base block, the LCD toward the robot.
    stem = _cylinder(stage, ROOT + '/IndicatorStem', 0.0018, INDICATOR['stem'], (x1 + INDICATOR['stem'] / 2, jy, ROD_Z), chrome,
                     collide=False)
    UsdGeom.XformCommonAPI(UsdGeom.Xform(stem)).SetRotate(Gf.Vec3f(0, 90, 0))
    bx, by, bz = INDICATOR['base']
    cx = x1 + INDICATOR['stem'] + bx / 2
    _box(stage, ROOT + '/IndicatorBase', (bx, by, bz), (cx, jy, base_h + bz / 2), (0.16, 0.17, 0.19), collide=False)
    labels = [('INDICATOR', (-0.031, 0.0150), 0.0042, 'left'), ('mm', (0.031, 0.0150), 0.0050, 'right')]
    rows = L.INDICATOR_ROWS
    width = L.row_width(rows['G'])
    L.build_panel(stage, INDICATOR_PATH, centre=(cx, jy + 0.004, base_h + bz + 0.020), tilt_deg=45.0, face=L.INDICATOR_FACE,
                  depth=0.020, rows=rows, labels=labels, glass=((rows['G']['u'] + width / 2, rows['G']['v'] + 0.0058),
                                                                 (width + 0.008, 0.0195)), looks_=looks, collide=False)
    _text(stage, ROOT + '/JacketLabel', 'HEATING JACKET', (jx, jy + 0.045), 0.0004, 0.009)
    return rows


def build(stage, config):
    """config: {'kind': 'thermistor' | 'rod', 'rod_l0_mm' (rod), 'probe_label'}. Returns objects, buttons, seats,
    panels {name: (path, rows)}."""
    retire_hooke_apparatus(stage)
    repair_cube_extents(stage)
    z0 = table_top_z(stage)
    root = UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(0, 0, z0))
    materials = {'rubber': _material(stage, '/World/PhysicsMaterials/ThermoRubber', 0.8, 0.7),
                 'pmma': _material(stage, '/World/PhysicsMaterials/ThermoPMMA', 0.25, 0.20),
                 'cell': _material(stage, '/World/PhysicsMaterials/ThermoCell', 0.25, 0.20, compliant=C.CUVETTE_CONTACT),
                 'seat': _material(stage, '/World/PhysicsMaterials/ThermoSeat', 0.20, 0.15)}
    looks = L.looks(stage, ROOT)
    UsdGeom.Xform.Define(stage, ROOT + '/Buttons')
    (bx, by), (sx, sy, sz) = BLOCK['centre'], BLOCK['size']
    _box(stage, ROOT + '/Block', (sx, sy, sz), (bx, by, sz / 2), (0.32, 0.33, 0.36), material=materials['rubber'])
    seat_w = collar(stage, ROOT + '/BlockSeat', (bx, by), sz, (0.95, 0.55, 0.10), materials['seat'])
    _text(stage, ROOT + '/BlockLabel', 'DRY BLOCK 25-80 C', (bx, by + sy / 2 + 0.014), 0.0004, 0.010)
    panels = {'controller': (CONTROLLER_PATH, L.build_controller(stage, CONTROLLER_PATH, x=bx, front_y=by + sy / 2,
                                                                 looks_=looks, cells=4, decimals=2))}
    (tx, ty), (ux, uy, uz) = STAND['centre'], STAND['size']
    _box(stage, ROOT + '/Stand', (ux, uy, uz), (tx, ty, uz / 2), (0.20, 0.21, 0.23), material=materials['rubber'])
    seat_s = collar(stage, ROOT + '/StandSeat', (tx, ty), uz, (0.03, 0.16, 0.20), materials['seat'])
    _text(stage, ROOT + '/ProbeLabel', config['probe_label'], (tx, ty + uy / 2 + 0.014), 0.0004, 0.009)
    buttons = {}
    for bid, (xy, colour, cap, text) in _buttons(config['kind']).items():
        buttons[bid] = _button(stage, bid, xy, 0.0, 'round', colour, cap, materials)
        _text(stage, f'{ROOT}/{bid.title()}Label', text, (xy[0], xy[1] + 0.030), 0.0004, 0.008)
    if config['kind'] == 'thermistor':
        panels['ohmmeter'] = (METER_PATH, L.build_ohmmeter(stage, METER_PATH, xy=METER_XY, looks_=looks,
                                                          material=materials['rubber']))
    else:
        panels['thermometer'] = (METER_PATH, L.build_thermometer(stage, METER_PATH, xy=METER_XY, title='PROBE',
                                                                looks_=looks, material=materials['rubber']))
        panels['indicator'] = (INDICATOR_PATH, _jacket(stage, config, looks))
    # The probe's cable from its handle side to the meter (render only, lying on the table).
    _box(stage, ROOT + '/ProbeCable', (METER_XY[0] - tx - uy / 2, 0.003, 0.003),
         ((METER_XY[0] + tx + uy / 2) / 2, ty + 0.025, 0.0015), (0.10, 0.10, 0.10), collide=False)
    objects = {'probe': _probe(stage, (tx, ty), uz + SEAT_FLOOR + 0.0003, materials)}
    from pxr import PhysxSchema
    contact = PhysxSchema.PhysxCollisionAPI.Apply(stage.GetPrimAtPath(TABLE_TOP))
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    blank = {'controller': L.controller_state(25, None, 4, 2), 'ohmmeter': L.ohmmeter_state(None),
             'thermometer': L.thermometer_state(None), 'indicator': L.indicator_state(None)}
    for name, (path, rows) in panels.items():
        L.show(stage, path, rows, blank[name], None)
    return {'objects': objects, 'buttons': buttons, 'seats': {'well': str(seat_w.GetPath()), 'stand': str(seat_s.GetPath())},
            'panels': panels}


author_joints = C.author_joints
seat_frames = C.seat_frames

"""Calorimetry bench apparatus (calorimetry_specific_heat): heater block with its controller, calorimeter cup with
stirrer and digital thermometer, the sample. Built with the chemistry kit's helpers on the same root, so the chemistry
backend's grasp, seat and button machinery runs it unchanged.

From the robot (at (0.95, 0.35), facing -y): the heater block on its right with the sample standing in its well
(orange collar), the controller screen on the block's front face (SET and the block temperature) and the purple DN and
red UP caps in front of it; the calorimeter cup on its left (white insulated wall, render only so the hand works
inside it; a teal seat collar on its floor under clear water, the water line drawn as a ring at the wall; the
thermometer probe and the stirrer shaft at the back), the green STIR cap with the stirrer lamp in front of the cup and
the recording digital thermometer (LCD, 0.01 C, with the REC mark and the record number) on its left and its blue REC
cap in front of it. The sample is a black-painted alloy block (cuvette-sized foot,
round body, white rim handle on top, as a cuvette's) with its real mass; its foot's compliant contact is scaled with
the mass so it rests as the kit's 30 g parts do.
"""
from __future__ import annotations

import math

from pxr import Gf, Usd, UsdGeom

import chem_apparatus as C
from chem_apparatus import (CUVETTE, FUNNEL_POST_R, FUNNEL_RIM_H, FUNNEL_RIM_R, PORT_H, PORT_R, ROOT, SEAT_FLOOR,
                            _bind, _button, _render_material, _rigid, _text, collar)
from torsion_apparatus import TABLE_TOP, _box, _cylinder, _material, repair_cube_extents, retire_hooke_apparatus, table_top_z
import thermal_lcd as L

HEATER = {'centre': (0.84, -0.06), 'size': (0.10, 0.10, 0.040)}
CUP = {'centre': (1.06, -0.06), 'inner_r': 0.032, 'wall': 0.006, 'height': 0.085, 'floor': 0.008}
STIR = {'xy': (1.06, 0.035), 'lamp': (1.095, 0.035)}
REC = {'xy': (1.150, 0.035), 'colour': (0.35, 0.65, 0.90)}      # the recorder's REC cap, in front of the thermometer
HEATER_BUTTONS = {'down': ((0.80, 0.035), (0.50, 0.25, 0.75), 'DN', 'SET -5 C'),
                  'up': ((0.88, 0.035), (0.80, 0.10, 0.08), 'UP', 'SET +5 C')}
THERMOMETER_XY = (1.150, -0.030)
CONTROLLER_PATH = ROOT + '/HeaterController'
THERMOMETER_PATH = ROOT + '/Thermometer'
STIR_LAMP_PATH = ROOT + '/StirLamp'
SAMPLE_FOOT = (CUVETTE[0], CUVETTE[1], 0.020)
SAMPLE_BODY = (0.011, 0.030)                  # radius, height
SAMPLE_POST = 0.020
SAMPLE_STEM = 0.014
SAMPLE_CONTACT_KG = 0.030                      # the kit's compliant contact (chem_apparatus.CUVETTE_CONTACT) is tuned for 30 g
SAMPLE_PATH = ROOT + '/Sample'
WATER_DENSITY_G_M3 = 1.0e6


def seat_paths(stage=None):
    return {'heater': ROOT + '/HeaterSeat', 'cup': ROOT + '/CupSeat'}


def button_paths(stage=None):
    return {b: C.button_path(b) for b in ('stir', 'up', 'down', 'rec')}


def button_layout():
    return ({'stir': (STIR['xy'], 0.0, 'round'), 'rec': (REC['xy'], 0.0, 'round')} |
            {b: (spec[0], 0.0, 'round') for b, spec in HEATER_BUTTONS.items()})


def body_centre_local():
    return SAMPLE_FOOT[2] + SAMPLE_BODY[1] / 2


def handle_top_local():
    return SAMPLE_FOOT[2] + SAMPLE_BODY[1] + SAMPLE_POST + SAMPLE_STEM + FUNNEL_RIM_H + PORT_H


def latch_local():
    return (0.0, 0.0, handle_top_local() + 0.020)


def water_top_local(water_g):
    """Water line above the table top."""
    return CUP['floor'] + water_g / WATER_DENSITY_G_M3 / (math.pi * CUP['inner_r'] ** 2)


def _sample(stage, xy, bottom_z, materials, mass_kg):
    xform = UsdGeom.Xform.Define(stage, SAMPLE_PATH)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(xy[0], xy[1], bottom_z))
    fz = SAMPLE_FOOT[2]
    black = (0.05, 0.05, 0.06)
    _box(stage, SAMPLE_PATH + '/Foot', SAMPLE_FOOT, (0, 0, fz / 2), black, material=materials['sample'])
    r, h = SAMPLE_BODY
    _cylinder(stage, SAMPLE_PATH + '/Body', r, h, (0, 0, fz + h / 2), black, materials['pmma'])
    z = fz + h
    _cylinder(stage, SAMPLE_PATH + '/Post', 0.004, SAMPLE_POST, (0, 0, z + SAMPLE_POST / 2), (0.25, 0.26, 0.28), materials['pmma'])
    z += SAMPLE_POST
    white = (0.94, 0.94, 0.92)
    _cylinder(stage, SAMPLE_PATH + '/HandleStem', FUNNEL_POST_R, SAMPLE_STEM, (0, 0, z + SAMPLE_STEM / 2), white, materials['pmma'])
    z += SAMPLE_STEM
    _cylinder(stage, SAMPLE_PATH + '/HandleRim', FUNNEL_RIM_R, FUNNEL_RIM_H, (0, 0, z + FUNNEL_RIM_H / 2), white, materials['pmma'])
    z += FUNNEL_RIM_H
    _cylinder(stage, SAMPLE_PATH + '/HandlePort', PORT_R, PORT_H, (0, 0, z + PORT_H / 2), white, collide=False)
    _rigid(xform.GetPrim(), mass_kg, (0, 0, fz + h / 2), ccd=True)
    return SAMPLE_PATH


def build(stage, config):
    """config: {'water_g', 'sample_g', 'cup_j_k', 'sample_kg'}. Returns objects, buttons, seats, panels."""
    retire_hooke_apparatus(stage)
    repair_cube_extents(stage)
    z0 = table_top_z(stage)
    root = UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(0, 0, z0))
    materials = {'rubber': _material(stage, '/World/PhysicsMaterials/CalorRubber', 0.8, 0.7),
                 'pmma': _material(stage, '/World/PhysicsMaterials/CalorPMMA', 0.25, 0.20),
                 'cell': _material(stage, '/World/PhysicsMaterials/CalorCell', 0.25, 0.20, compliant=C.CUVETTE_CONTACT),
                 'seat': _material(stage, '/World/PhysicsMaterials/CalorSeat', 0.20, 0.15)}
    scale = float(config['sample_kg']) / SAMPLE_CONTACT_KG
    materials['sample'] = _material(stage, '/World/PhysicsMaterials/CalorSample', 0.25, 0.20,
                                    compliant=(C.CUVETTE_CONTACT[0] * scale, C.CUVETTE_CONTACT[1] * scale))
    looks = L.looks(stage, ROOT)
    UsdGeom.Xform.Define(stage, ROOT + '/Buttons')
    # Heater block: a dark red block with the orange seat collar on top and a red lamp; set temperature in front.
    (hx, hy), (sx, sy, sz) = HEATER['centre'], HEATER['size']
    _box(stage, ROOT + '/Heater', (sx, sy, sz), (hx, hy, sz / 2), (0.30, 0.08, 0.06), material=materials['rubber'])
    seat_h = collar(stage, ROOT + '/HeaterSeat', (hx, hy), sz, (0.95, 0.55, 0.10), materials['seat'])
    lamp = _render_material(stage, ROOT + '/Materials/HeaterLamp', (1.0, 0.15, 0.05), emission=60.0)
    _bind(_cylinder(stage, ROOT + '/HeaterLamp', 0.005, 0.003, (hx + 0.035, hy + 0.035, sz + 0.0015), (1.0, 0.15, 0.05),
                    collide=False), lamp)
    _text(stage, ROOT + '/HeaterLabel', 'HEATER 25-95 C', (hx, hy + sy / 2 + 0.014), 0.0004, 0.010)
    controller_rows = L.build_controller(stage, CONTROLLER_PATH, x=hx, front_y=hy + sy / 2, looks_=looks, cells=3,
                                         decimals=1)
    # Calorimeter cup.
    (cx, cy), r, w, height, floor = CUP['centre'], CUP['inner_r'], CUP['wall'], CUP['height'], CUP['floor']
    _cylinder(stage, ROOT + '/CupFloor', r + w, floor, (cx, cy, floor / 2), (0.92, 0.92, 0.90), materials['rubber'])
    n = 24
    for k in range(n):
        a = 2 * math.pi * k / n
        path = f'{ROOT}/CupWall{k:02d}'
        _box(stage, path, (w, 2 * (r + w) * math.sin(math.pi / n) + 0.001, height - floor),
             (cx + (r + w / 2) * math.cos(a), cy + (r + w / 2) * math.sin(a), floor + (height - floor) / 2),
             (0.92, 0.92, 0.90), rotate_xyz=(0, 0, math.degrees(a)), collide=False)
    seat_c = collar(stage, ROOT + '/CupSeat', (cx, cy), floor, (0.03, 0.16, 0.20), materials['seat'])
    water = _render_material(stage, ROOT + '/Materials/Water', (0.45, 0.70, 0.90), matte=True)
    top = water_top_local(config['water_g'])
    for k in range(n):                                     # the water line: a ring of tiles at the wall
        a = 2 * math.pi * k / n
        path = f'{ROOT}/WaterLine{k:02d}'
        _bind(_box(stage, path, (0.008, 2 * r * math.sin(math.pi / n) + 0.0005, 0.0015),
                   (cx + (r - 0.004) * math.cos(a), cy + (r - 0.004) * math.sin(a), top - 0.00075), (0.45, 0.70, 0.90),
                   rotate_xyz=(0, 0, math.degrees(a)), collide=False), water)
    _cylinder(stage, ROOT + '/Probe', 0.002, height + 0.03 - floor, (cx + 0.020, cy - 0.020, floor + (height + 0.03 - floor) / 2),
              (0.75, 0.75, 0.72), collide=False)
    _cylinder(stage, ROOT + '/StirShaft', 0.0025, height + 0.03 - floor, (cx - 0.020, cy - 0.020, floor + (height + 0.03 - floor) / 2),
              (0.20, 0.20, 0.22), collide=False)
    # The probe's cable over the cup rim to the digital thermometer on the cup's left.
    tx, ty = THERMOMETER_XY
    _box(stage, ROOT + '/ProbeCable', (tx - (cx + 0.020), 0.003, 0.003), ((tx + cx + 0.020) / 2, cy - 0.020, height + 0.030),
         (0.10, 0.10, 0.10), collide=False)
    _box(stage, ROOT + '/ProbeCableDrop', (0.003, ty - (cy - 0.020), 0.003), (tx, (ty + cy - 0.020) / 2, height + 0.030),
         (0.10, 0.10, 0.10), collide=False)
    thermometer_rows = L.build_recording_thermometer(stage, THERMOMETER_PATH, xy=THERMOMETER_XY, title='WATER',
                                                     looks_=looks, material=materials['rubber'])
    _text(stage, ROOT + '/CupLabel', f"WATER {config['water_g']:.1f} g   CUP {config['cup_j_k']:.1f} J/K",
          (cx, cy + r + w + 0.014), 0.0004, 0.009)
    buttons = {'stir': _button(stage, 'stir', STIR['xy'], 0.0, 'round', (0.15, 0.62, 0.25), 'STIR', materials, C.INK),
               'rec': _button(stage, 'rec', REC['xy'], 0.0, 'round', REC['colour'], 'REC', materials)}
    _text(stage, ROOT + '/RecLabel', 'RECORD', (REC['xy'][0], REC['xy'][1] + 0.030), 0.0004, 0.008)
    for bid, (xy, colour, cap, text) in HEATER_BUTTONS.items():
        buttons[bid] = _button(stage, bid, xy, 0.0, 'round', colour, cap, materials)
        _text(stage, f'{ROOT}/{bid.title()}Label', text, (xy[0], xy[1] + 0.030), 0.0004, 0.008)
    # The stirrer lamp: a green lens beside the STIR cap, lit while the stirrer runs (the dark lens shows otherwise).
    lx, ly = STIR['lamp']
    _cylinder(stage, ROOT + '/StirLampBase', 0.009, 0.006, (lx, ly, 0.003), (0.20, 0.21, 0.23), collide=False)
    _cylinder(stage, ROOT + '/StirLampDark', 0.006, 0.002, (lx, ly, 0.007), (0.05, 0.12, 0.06), collide=False)
    lit = _render_material(stage, ROOT + '/Materials/StirLamp', (0.30, 1.0, 0.35), emission=1.5)
    lamp = _cylinder(stage, STIR_LAMP_PATH, 0.006, 0.002, (lx, ly, 0.0085), (0.30, 1.0, 0.35), collide=False)
    _bind(lamp, lit)
    _text(stage, ROOT + '/StirLampLabel', 'STIR ON', (lx, ly + 0.020), 0.0004, 0.006)
    _text(stage, ROOT + '/SampleLabel', f"SAMPLE {config['sample_g']:.2f} g", (hx, hy - sy / 2 - 0.014), 0.0004, 0.009)
    objects = {'sample': _sample(stage, (hx, hy), sz + SEAT_FLOOR + 0.0003, materials, float(config['sample_kg']))}
    from pxr import PhysxSchema
    contact = PhysxSchema.PhysxCollisionAPI.Apply(stage.GetPrimAtPath(TABLE_TOP))
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    L.show(stage, CONTROLLER_PATH, controller_rows, L.controller_state(25, None, 3, 1), None)
    L.show(stage, THERMOMETER_PATH, thermometer_rows, L.recording_thermometer_state(None, False, None), None)
    return {'objects': objects, 'buttons': buttons, 'seats': {'heater': str(seat_h.GetPath()), 'cup': str(seat_c.GetPath())},
            'panels': {'controller': (CONTROLLER_PATH, controller_rows), 'thermometer': (THERMOMETER_PATH, thermometer_rows)}}


def set_lamp(stage, lit):
    """The stirrer lamp: the lit lens stands over the dark one, or sinks into the base."""
    prim = stage.GetPrimAtPath(STIR_LAMP_PATH)
    t = UsdGeom.XformCommonAPI(prim).GetXformVectors(Usd.TimeCode.Default())[0]
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(t[0], t[1], 0.0085 if lit else 0.0085 - 0.006))


author_joints = C.author_joints
seat_frames = C.seat_frames

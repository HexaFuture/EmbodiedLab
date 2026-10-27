"""Titration bench (kit chemistry, recipe chem_titration_bench; titration_basic, titration_weak_acid).

Built on the shared lab stage like the spectrophotometry bench (chem_apparatus, whose primitives it uses) and
with what that bench taught: every instrument faces the robot, bases are dark and matte, every top the fingers
aim at has a raised centre, caps stand 50 mm apart along the finger axis, wells are a little wider than
what stands in them and what is set down has a compliant contact layer.

Sample vessels: three square beakers (40 x 40 mm inside, 22 mm deep: 1 mL is 0.625 mm, 35 mL to the brim), each
on a carrier: a base plate, two posts with gripper handles (rim and port neck as on the cuvettes) 54 mm in
front of and behind the beaker, and a bridge over the beaker whose raised green collar holds the pH electrode. The
carrier is symmetric, so the starter place skill (which centres the held object's footprint on the surface)
sets the beaker in the middle of a seat whichever handle is held. They start in a rack of three wells. Behind them
(the meter is not calibrated) a back row of two wells, behind S1 and S3, holds the two pH standard buffers
on the same carriers (labels B4 and B7; their handle rims show 4.01 and 7.00).

Stirrer station: a seat for a carrier on the stirrer; the titrant bottle (drawn to the scale of its volume)
behind it with its spout over the beaker; in front three buttons (1.00, 0.10 and 0.02 mL), the counter screen and
its ZERO key. Meter: a housing with the pH screen (pH, STABLE, REC and the record number), its REC key and the
electrode's park seat.

pH electrode: a rod below a seat disc, a stem and a gripper handle above it. It is set down with the starter
place skill on a bridge (or on the park seat): the rod goes through the hole and the disc rests on the surface,
the tip 2 mm above the beaker floor. Handles on the vessels and on the electrode are a manipulation concession. A cable
(drawn only: no mass, no collision) connects the electrode to the meter and follows it (build_cable, set_cable).

Layout coordinates are in the robot frame of chem_apparatus: u to the robot's right (world -X), v away from
the robot (world -Y).
"""
from __future__ import annotations

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema

import chem_apparatus as C
from chem_apparatus import (BASE_GREY, HOUSING_BLUE, INK, PAPER, ROOT, _bind, _button, _flat, _light, _local, _mesh,
                            _remember_lit, _render_material, _rigid, _text, _xy)
from torsion_apparatus import TABLE_TOP, _box, _cylinder, _material, repair_cube_extents, retire_hooke_apparatus, table_top_z

BEAKERS = ('S1', 'S2', 'S3')
BUFFERS = ('B4', 'B7')                              # pH standard buffers (chem_titration.BUFFER_PH), back row of the rack
CARRIERS = BEAKERS + BUFFERS
CARRIER_TEXT = {'B4': '4.01', 'B7': '7.00'}         # printed on the handle rims (the samples show their label)
TITRANT_LEVELS = ('large', 'coarse', 'fine')
TITRANT_TEXT = {'large': '1.0', 'coarse': '0.1', 'fine': '.02'}

# --- carrier (local z = 0 at the base plate's underside; x across, y along the handles) -------------------
CARRIER_BASE = (0.048, 0.119, 0.004)
BEAKER_INNER, BEAKER_DEPTH, BEAKER_GLASS = 0.040, 0.022, 0.002
BEAKER_FLOOR_Z = CARRIER_BASE[2] + BEAKER_GLASS                      # liquid starts here
BRIDGE_Z = BEAKER_FLOOR_Z + BEAKER_DEPTH                             # bridge underside = beaker brim
BRIDGE = (0.046, 0.004)                                              # width across, thickness
RING_H, RING_COLOUR = 0.002, (0.15, 0.62, 0.25)                        # raised green collar round the hole: the electrode's seat
BRIDGE_TOP_Z = BRIDGE_Z + BRIDGE[1] + RING_H
HOLE = 0.020                                                         # square hole for the 10 mm rod: 5 mm to spare each way
POST_Y = 0.054
POST_R, POST_TOP_Z = 0.0055, 0.090
RIM_R, RIM_H = 0.016, 0.005
PORT_R, PORT_H = C.PORT_R, C.PORT_H
CARRIER_TOP_Z = POST_TOP_Z + RIM_H + PORT_H
CARRIER_MASS_KG = 0.060
MM_PER_ML = 10.0 / (BEAKER_INNER * 100) ** 2                          # liquid height per mL (mm)

# --- electrode (local z = 0 at the underside of its seat disc) ----------------------------------------------
ROD_R, ROD_LEN = 0.005, 0.026
DISC_R, DISC_H = 0.018, 0.003                                        # still carried all round when 5 mm off centre
STEM_R, STEM_TOP_Z = 0.0055, 0.063
ELECTRODE_TOP_Z = STEM_TOP_Z + RIM_H + PORT_H
ELECTRODE_MASS_KG = 0.020
TIP_ABOVE_FLOOR_M = BRIDGE_TOP_Z - ROD_LEN - BEAKER_FLOOR_Z

LATCH_ABOVE_TOP = 0.020                                              # as the Hooke weights and the cuvettes
SWITCH_DOWN_M, SWITCH_UP_M = C.SWITCH_DOWN_M, C.SWITCH_UP_M
author_joints, seat_frames = C.author_joints, C.seat_frames
OBJECT_CONTACT = C.CUVETTE_CONTACT

# --- seats ----------------------------------------------------------------------------------------------
SEAT_CLEARANCE = 0.00275
SEAT_WELL = (CARRIER_BASE[0] / 2 + SEAT_CLEARANCE, CARRIER_BASE[1] / 2 + SEAT_CLEARANCE)     # half sizes
SEAT_MOUTH = (SEAT_WELL[0] + 0.004, SEAT_WELL[1] + 0.004)
SEAT_OUTER = (SEAT_MOUTH[0] + 0.006, SEAT_MOUTH[1] + 0.006)
SEAT_H, SEAT_DEPTH, SEAT_CHAMFER = 0.014, 0.009, 0.004
SEAT_FLOOR = SEAT_H - SEAT_DEPTH
# Floor backstop: a well floor is one face of a static triangle mesh and the carrier's base plate (4 mm)
# meets it with the soft contact layer (OBJECT_CONTACT, 1e3 N/m per contact point): a carrier pressed down hard (a
# held carrier pushed onto its handle, 15-60 N) sank more than half the plate into the floor, PhysX stopped colliding it
# with the face it had passed and it stayed 3.9 mm low. Under each carrier well's floor an invisible static block (the
# well's solid floor, down to the table) whose top lies FLOOR_BACKSTOP_GAP below the floor face catches such a plate
# and pushes it back up once the load is gone; it is deep, so a plate pressed far into it is still pushed up, not
# down. With its small contact offset it is not touched while a carrier stands or is set down normally.
FLOOR_BACKSTOP_GAP, FLOOR_BACKSTOP_CONTACT_OFFSET = 0.0015, 0.0002
# Park seat: a collar on the meter housing whose well takes the rod; the disc rests on its rim.
PARK_OUTER, PARK_MOUTH, PARK_HOLE, PARK_H, PARK_DEPTH = 0.025, HOLE / 2 + 0.003, HOLE / 2, 0.030, 0.026
ELECTRODE_XY_TOL_M = 0.006

# --- stations (world centres; robot-frame layout) -------------------------------------------------------
# Every handle stands at least 0.31 m (horizontal) from the robot base (chem_apparatus.RACK): the front handles of
# the rack carriers are 0.316 m away.
# The labels are printed 54 mm in front of the front handles: closer, a handle rim and its label merge in the wrist view.
# The back row: the buffer wells 0.16 m behind the front row, behind S1 and S3 (16 mm between the
# collars); the base reaches 0.16 m further back. The rack cannot widen: the station and the meter flank it.
RACK = {'centre': (0.95, -0.02), 'pitch_u': 0.090, 'base': (0.262, 0.356, 0.012), 'base_v': 0.057, 'labels_v': -0.108,
        'back_v': 0.16, 'back_u': {'B4': -0.090, 'B7': 0.090}, 'colour': (0.03, 0.16, 0.20)}
# Stirrer station: bottle at the back, the seat, then (clear of the hand over the front handle: 115 mm) the three
# buttons, their prints and the counter screen at the front edge.
STATION = {'seat': (0.665, -0.13), 'base': (0.17, 0.345, 0.015), 'base_v': -0.0325, 'colour': (0.95, 0.55, 0.10),
           'bottle_v': 0.105, 'buttons_v': -0.115, 'button_u': {'large': -0.05, 'coarse': 0.0, 'fine': 0.05},
           'labels_v': -0.142, 'counter': (0.0, -0.178), 'zero': (-0.068, -0.165)}
METER = {'centre': (1.215, -0.10), 'body': (0.20, 0.24, 0.050), 'park': (0.045, 0.055), 'screen': (-0.035, -0.045),
         'rec': (0.060, -0.040),
         'colour': (0.10, 0.10, 0.11)}
SPOUT_ABOVE = 0.012


def carrier_path(label):
    return f'{ROOT}/Carrier{label}'


ELECTRODE = ROOT + '/Electrode'


def rack_slots():
    """World xy of the rack wells: the three sample wells (S1 on the robot's left) and the two buffer wells behind
    S1 and S3."""
    slots = {label: _xy(RACK['centre'], (k - 1) * RACK['pitch_u'], 0.0) for k, label in enumerate(BEAKERS)}
    slots.update({label: _xy(RACK['centre'], RACK['back_u'][label], RACK['back_v']) for label in BUFFERS})
    return slots


def seat_positions():
    seats = {f'rack:{label}': xy for label, xy in rack_slots().items()}
    seats['stirrer'] = STATION['seat']
    return seats


def park_xy():
    return _xy(METER['centre'], *METER['park'])


def button_layout():
    """World xy (before jitter), base height above the table and shape of every button: the three titrant buttons,
    the counter's ZERO key beside the counter screen and the meter's REC key beside the pH screen."""
    zs = STATION['base'][2]
    out = {f'titrant:{level}': (_xy(STATION['seat'], STATION['button_u'][level], STATION['buttons_v']), zs, 'round')
           for level in TITRANT_LEVELS}
    out['counter:zero'] = (_xy(STATION['seat'], *STATION['zero']), zs, 'round')
    out['meter:rec'] = (_xy(METER['centre'], *METER['rec']), METER['body'][2], 'round')
    return out


def button_paths(stage=None):
    return {bid: C.button_path(bid) for bid in button_layout()}


def seat_paths(stage=None):
    out = {f'rack:{label}': f'{ROOT}/RackSeat{label}' for label in CARRIERS}
    out['stirrer'] = ROOT + '/StirrerSeat'
    return out


PARK_SEAT = ROOT + '/ParkSeat'


# --- geometry ---------------------------------------------------------------------------------------------
def _rect(hx, hy, z):
    return [(-hx, -hy, z), (hx, -hy, z), (hx, hy, z), (-hx, hy, z)]


def well(stage, path, centre, base_z, outer, mouth, inner, height, depth, chamfer, colour, material):
    """A seat: raised rectangular collar with a chamfered mouth and a well (static triangle mesh). Half sizes.
    Vertex rings as chem_apparatus.collar (the floor is ring 4), so chem_apparatus.seat_frames reads it."""
    cx, cy = centre
    rings = [_rect(*outer, 0.0), _rect(*outer, height), _rect(*mouth, height), _rect(*inner, height - chamfer),
             _rect(*inner, height - depth)]
    points = [(x + cx, y + cy, z + base_z) for ring in rings for x, y, z in ring]
    faces = []

    def band(a, b):
        for i in range(4):
            j = (i + 1) % 4
            faces.append([4 * a + i, 4 * a + j, 4 * b + j, 4 * b + i])
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4)):
        band(a, b)
    faces.append([16, 17, 18, 19])
    faces.append([3, 2, 1, 0])
    mesh = _mesh(stage, path, points, faces, colour, collide=True, material=material)
    if path != PARK_SEAT:                 # carrier wells: the floor backstop (the park seat carries the electrode's disc)
        h = base_z + height - depth - FLOOR_BACKSTOP_GAP        # from the table top (z 0) up to just below the floor
        box = _box(stage, path + 'Backstop', (2 * inner[0], 2 * inner[1], h), (cx, cy, h / 2), colour, material=material)
        UsdGeom.Imageable(box).MakeInvisible()
        PhysxSchema.PhysxCollisionAPI(box.GetPrim()).GetContactOffsetAttr().Set(FLOOR_BACKSTOP_CONTACT_OFFSET)
    return mesh


def _handle(stage, path, name, xy, base_z, top_z, colour, materials):
    """Gripper handle as on the cuvettes: stem, rim and port neck; top_z is the stem's top."""
    _cylinder(stage, f'{path}/{name}Stem', POST_R, top_z - base_z, (xy[0], xy[1], (top_z + base_z) / 2), colour, materials['pmma'])
    _cylinder(stage, f'{path}/{name}Rim', RIM_R, RIM_H, (xy[0], xy[1], top_z + RIM_H / 2), colour, materials['pmma'])
    _cylinder(stage, f'{path}/{name}Port', PORT_R, PORT_H, (xy[0], xy[1], top_z + RIM_H + PORT_H / 2), colour, collide=False)


def _bridge(stage, path, colour, material):
    """Plate over the beaker, as boxes (a dynamic body takes no mesh hole): a raised square collar round the
    electrode hole, as wide as the plate, and a lower span to each handle post."""
    w, t = BRIDGE
    side = (w - HOLE) / 2
    z = BRIDGE_Z + (t + RING_H) / 2
    for k, sx in enumerate((-1, 1)):
        _box(stage, f'{path}/RingSide{k}', (side, w, t + RING_H), (sx * (HOLE + side) / 2, 0, z), RING_COLOUR, material=material)
    for k, sy in enumerate((-1, 1)):
        _box(stage, f'{path}/RingEnd{k}', (HOLE, side, t + RING_H), (0, sy * (HOLE + side) / 2, z), RING_COLOUR, material=material)
    span = POST_Y - w / 2
    for k, sy in enumerate((-1, 1)):
        _box(stage, f'{path}/BridgeSpan{k}', (w, span, t), (0, sy * (w / 2 + span / 2), BRIDGE_Z + t / 2), colour, material=material)


def latch_local(key):
    return (0.0, 0.0, ELECTRODE_TOP_Z + LATCH_ABOVE_TOP) if key == 'electrode' else None


def carrier_latches():
    """Latch points of a carrier (either handle), carrier frame."""
    return [(0.0, sy * POST_Y, CARRIER_TOP_Z + LATCH_ABOVE_TOP) for sy in (-1, 1)]


def _carrier(stage, label, xy, bottom_z, materials):
    path = carrier_path(label)
    text = CARRIER_TEXT.get(label, label)
    xform = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(xy[0], xy[1], bottom_z))
    bx, by, bz = CARRIER_BASE
    frame, glass, white = (0.20, 0.21, 0.23), (0.80, 0.86, 0.90), (0.94, 0.94, 0.92)
    _box(stage, path + '/Base', CARRIER_BASE, (0, 0, bz / 2), frame, material=materials['cell'])
    half = BEAKER_INNER / 2
    edge = BEAKER_GLASS
    _box(stage, path + '/BeakerFloor', (BEAKER_INNER + 2 * edge, BEAKER_INNER + 2 * edge, BEAKER_GLASS),
         (0, 0, bz + BEAKER_GLASS / 2), glass, collide=False)
    for k, (dx, dy) in enumerate(((1, 1), (1, -1), (-1, -1), (-1, 1))):
        _box(stage, f'{path}/BeakerEdge{k}', (edge, edge, BEAKER_DEPTH), (dx * (half + edge / 2), dy * (half + edge / 2),
             BEAKER_FLOOR_Z + BEAKER_DEPTH / 2), glass, collide=False)
    _box(stage, path + '/Liquid', C.LIQUID_HIDDEN_SIZE, (0, 0, BEAKER_FLOOR_Z), (1, 1, 1), collide=False)
    _bridge(stage, path, frame, materials['pmma'])
    for name, sy in (('Front', 1), ('Back', -1)):
        _handle(stage, path, name, (0, sy * POST_Y), bz, POST_TOP_Z, white, materials)
    for name, sy in (('Front', 1), ('Back', -1)):                      # the beaker's name on both handle rims
        _text(stage, f'{path}/Label{name}', text, (0.0, sy * POST_Y + 0.0105), POST_TOP_Z + RIM_H + 0.0004,
              0.007 if len(text) <= 2 else 0.0055)
    _rigid(xform.GetPrim(), CARRIER_MASS_KG, (0, 0, 0.012), ccd=True)
    return path


def set_beaker_liquid(stage, label, volume_mL, rgb=(0.72, 0.84, 0.92)):
    cube = UsdGeom.Cube(stage.GetPrimAtPath(carrier_path(label) + '/Liquid'))
    height = min(volume_mL * MM_PER_ML / 1000.0, BEAKER_DEPTH)
    api = UsdGeom.XformCommonAPI(cube)
    if height > 1e-5:
        api.SetTranslate(Gf.Vec3d(0, 0, BEAKER_FLOOR_Z + height / 2))
        api.SetScale(Gf.Vec3f(BEAKER_INNER, BEAKER_INNER, height))
        C._set_liquid_colour(stage, carrier_path(label) + '/Liquid', rgb)
    else:
        api.SetTranslate(Gf.Vec3d(0, 0, BEAKER_FLOOR_Z))
        api.SetScale(Gf.Vec3f(*C.LIQUID_HIDDEN_SIZE))


def _electrode(stage, xy, seat_z, materials):
    xform = UsdGeom.Xform.Define(stage, ELECTRODE)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(xy[0], xy[1], seat_z + 0.0003))
    body, white = (0.15, 0.16, 0.18), (0.94, 0.94, 0.92)
    _cylinder(stage, ELECTRODE + '/Rod', ROD_R, ROD_LEN, (0, 0, -ROD_LEN / 2), (0.75, 0.80, 0.85), materials['cell'])
    _cylinder(stage, ELECTRODE + '/Disc', DISC_R, DISC_H, (0, 0, DISC_H / 2), body, materials['cell'])
    _cylinder(stage, ELECTRODE + '/Stem', STEM_R, STEM_TOP_Z - DISC_H, (0, 0, (STEM_TOP_Z + DISC_H) / 2), body, materials['pmma'])
    _cylinder(stage, ELECTRODE + '/Rim', RIM_R, RIM_H, (0, 0, STEM_TOP_Z + RIM_H / 2), white, materials['pmma'])
    _cylinder(stage, ELECTRODE + '/Port', PORT_R, PORT_H, (0, 0, STEM_TOP_Z + RIM_H + PORT_H / 2), white, collide=False)
    _rigid(xform.GetPrim(), ELECTRODE_MASS_KG, (0, 0, 0.005), ccd=True)
    return ELECTRODE


# --- electrode cable (drawn only) ---------------------------------------------------
# The electrode is cabled to the meter as a real one is. The cable is drawn only: a tube mesh with no mass, no rigid
# body and its collision disabled (so the scene collider pass leaves it alone); nothing reads it. The backend redraws
# it from the electrode's pose before every image (TitrationBackend._draw_cable). It leaves the back face of the meter,
# lies on the table behind the rack and rises from behind to the back of the electrode's stem. All three cameras look
# from the robot's side, so everything it passes lies behind the meter screen, the counter screen and the carriers'
# green collars as the cameras see them (a held electrode drags it over the bench while carried).
CABLE = ROOT + '/ElectrodeCable'
CABLE_R, CABLE_SIDES, CABLE_SAMPLES = 0.002, 8, 72
CABLE_COLOUR = (0.05, 0.05, 0.06)
CABLE_ATTACH_Z = STEM_TOP_Z - 0.008                 # on the stem's axis below the handle rim (electrode frame)
CABLE_SOCKET = (0.075, 0.12, 0.022)                 # meter back face: u, v from the meter centre, height
CABLE_LANE_V = 0.21                                 # behind the meter centre: the cable lies on the table along this line,
                                                    # behind the rack's back row
CABLE_LANE_END_U = 0.10                             # the lane's end by the meter (u from the meter centre)
CABLE_DROP_U = -0.125                               # from an electrode over the station (or to its right) the cable
                                                    # comes down beside the station on the meter's side (u from the
                                                    # seat), clear of the bottle


def _cubic(p0, p1, p2, p3, n=24):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


def _chaikin(points, rounds=2):
    p = np.asarray(points, float)
    for _ in range(rounds):
        q = [p[0]]
        for a, b in zip(p[:-1], p[1:]):
            q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        q.append(p[-1])
        p = np.asarray(q)
    return p


def _resample(points, n):
    p = np.asarray(points, float)
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-7]
    p = p[keep]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, k]) for k in range(3)], axis=1)


def _lead(e1, far):
    """Second control point of the cable's first span: from an electrode standing low (on a seat) the cable leaves the
    stem backward, clear of the carrier's back handle; from a raised (held) one it hangs down, so that in the wrist view
    it stays close to the held electrode and thin."""
    h = min(max((e1[2] - 0.15) / 0.10, 0.0), 1.0)
    return e1 + (1.0 - h) * 0.04 * far + h * np.array([0.0, 0.0, -0.5 * e1[2]])


def cable_route(attach_local):
    """Centre line of the cable (ROOT-local, table top at z = 0) from the electrode's attachment point (ROOT-local) to
    the meter's socket: CABLE_SAMPLES points."""
    origin = C.ORIGIN_XY
    lp = lambda xy, z: np.array([*_local(xy, origin), z], float)
    M, S = METER['centre'], STATION['seat']
    su, sv, sz = CABLE_SOCKET
    inner = lp(_xy(M, su, sv - 0.015), sz)                      # starts inside the housing
    socket = lp(_xy(M, su, sv), sz)
    out = lp(_xy(M, su, sv + 0.015), 0.6 * sz)
    lane_end = lp(_xy(M, CABLE_LANE_END_U, CABLE_LANE_V), CABLE_R)
    far = np.array([0.0, -1.0, 0.0])                            # away from the robot (v), ROOT-local
    e0 = np.asarray(attach_local, float)
    e1 = e0 + 0.02 * far + np.array([0.0, 0.0, -0.004])         # out of the back of the stem
    x_drop = lp(_xy(S, CABLE_DROP_U, 0.0), 0.0)[0]
    if e0[0] >= lane_end[0]:                                    # over the meter (on its park seat): straight back down
        drop = out + 0.04 * far
        drop[2] = CABLE_R
        path = [e0[None], _cubic(e1, _lead(e1, far), drop, out), socket[None], inner[None]]
    else:
        q = np.array([min(max(e0[0], x_drop), lane_end[0]), lane_end[1], CABLE_R])
        towards = np.r_[e1[:2] - q[:2], 0.0]
        towards /= max(np.linalg.norm(towards), 1e-9)
        path = [e0[None], _cubic(e1, _lead(e1, far), q + 0.06 * towards, q)]
        if lane_end[0] - q[0] > 0.01:
            path.append(lane_end[None])
        path += [out[None], socket[None], inner[None]]
    points = _resample(_chaikin(np.vstack(path)), CABLE_SAMPLES)
    points[:, 2] = np.maximum(points[:, 2], CABLE_R)
    return points


def _tube(points):
    """Vertices and normals of a tube of CABLE_R round the centre line (parallel-transported frames)."""
    p = np.asarray(points, float)
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    normal = np.cross(t[0], [0.0, 0.0, 1.0])
    if np.linalg.norm(normal) < 1e-6:
        normal = np.cross(t[0], [1.0, 0.0, 0.0])
    normal /= np.linalg.norm(normal)
    angles = 2 * np.pi * np.arange(CABLE_SIDES) / CABLE_SIDES
    verts, norms = [], []
    for k in range(len(p)):
        normal = normal - np.dot(normal, t[k]) * t[k]
        normal /= max(np.linalg.norm(normal), 1e-12)
        binormal = np.cross(t[k], normal)
        ring = np.cos(angles)[:, None] * normal + np.sin(angles)[:, None] * binormal
        norms.append(ring)
        verts.append(p[k] + CABLE_R * ring)
    return np.vstack(verts), np.vstack(norms)


def build_cable(stage, attach_local):
    """Author the cable (drawn only) for the electrode's attachment point attach_local (ROOT-local)."""
    mesh = UsdGeom.Mesh.Define(stage, CABLE)
    counts, indices = [], []
    for i in range(CABLE_SAMPLES - 1):
        for j in range(CABLE_SIDES):
            k = (j + 1) % CABLE_SIDES
            counts.append(4)
            indices += [i * CABLE_SIDES + j, (i + 1) * CABLE_SIDES + j, (i + 1) * CABLE_SIDES + k, i * CABLE_SIDES + k]
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*CABLE_COLOUR)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(False)
    set_cable(stage, attach_local)
    return CABLE


def set_cable(stage, attach_local):
    """Redraw the cable for the electrode's attachment point (ROOT-local)."""
    verts, norms = _tube(cable_route(attach_local))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(CABLE))
    mesh.CreatePointsAttr().Set([Gf.Vec3f(*v) for v in verts.tolist()])
    mesh.CreateNormalsAttr().Set([Gf.Vec3f(*n) for n in norms.tolist()])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    mesh.CreateExtentAttr().Set([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())])


# --- seven-segment screens ------------------------------------------------------------------------------------
DISPLAY = ROOT + '/PhScreen'            # layout key point (benchmark/core/layout.py)
# read from the robot's own cameras (no close-up tool): 15 % strokes and a decimal point 1.4 strokes wide;
# the meter screen also shows the REC mark and the running record's number (two small digits).
SCREENS = {
    'ph': {'path': ROOT + '/PhScreen', 'size': (0.100, 0.056), 'digits': 4, 'point': 2, 'height': 0.020, 'unit': 'pH',
           'mark': 'STABLE', 'rec': True},
    'counter': {'path': ROOT + '/CounterScreen', 'size': (0.086, 0.040), 'digits': 4, 'point': 2, 'height': 0.020,
                'unit': 'mL', 'mark': None, 'rec': False},
}
STROKE, POINT_SIZE = 0.15, 1.4
SEGMENT_ON = C.SEGMENT_ON
REC_UV, REC_TEXT_H = (0.016, -0.019), 0.009            # REC mark centre (meter screen)
REC_DIGITS_U, REC_DIGIT_H = 0.028, 0.010                # record number: two digits from this u (left edge), centred on REC_UV v


def _point_gap(screen):
    """Extra space after the digit before the decimal point, which stands in it clear of both neighbours (in the
    plain digit gap it touched their lower verticals: 7.00 read as 700)."""
    return screen['height'] * 0.25       # 0.25 since the 1.4-stroke point: >= 1 mm clear of both digits


def _digit_origin(screen):
    pitch = screen['height'] * 0.72
    width = screen['digits'] * pitch + _point_gap(screen)
    return -width / 2 - 0.007, (0.006 if screen['mark'] else 0.0) - screen['height'] / 2, pitch


def _digit_u(screen, i):
    u0, _, pitch = _digit_origin(screen)
    return u0 + i * pitch + (_point_gap(screen) if i >= screen['point'] else 0.0)


def _segment_points(screen):
    """{(digit, segment): (u, v)} segment centres relative to the screen centre."""
    _, v0, _ = _digit_origin(screen)
    w, h = screen['height'] * 0.5, screen['height']
    return {(i, seg): (_digit_u(screen, i) + fu * w, v0 + fv * h) for i in range(screen['digits'])
            for seg, (fu, fv, _) in C.SEGMENTS.items()}


def _point_uv(screen):
    _, v0, pitch = _digit_origin(screen)
    w, t = screen['height'] * 0.5, screen['height'] * STROKE
    return _digit_u(screen, screen['point'] - 1) + w + (pitch - w + _point_gap(screen)) / 2, v0 + t * POINT_SIZE / 2


def display_rects(name):
    """Lit rectangles of a screen ((u, v) centre, (du, dv) size): {'segments': [...], 'point': rect}."""
    screen = SCREENS[name]
    w, h, t = screen['height'] * 0.5, screen['height'], screen['height'] * STROKE
    segments = [((u, v), (w, t) if C.SEGMENTS[seg][2] else (t, h / 2)) for (_, seg), (u, v) in _segment_points(screen).items()]
    return {'segments': segments, 'point': (_point_uv(screen), (t * POINT_SIZE, t * POINT_SIZE))}


# STABLE at the screen's left edge, 8 mm letters, about 19 mm clear of REC (9 mm letters centred at -20 mm
# rendered as one word, STABLEREC)
MARK_UV, MARK_HALF, MARK_TEXT_H = (-0.029, -0.019), (0.0165, 0.005), 0.008


def _rec_digit_rects():
    """{(digit, segment): ((u, v) centre, (du, dv) size)} of the record number on the meter screen."""
    h = REC_DIGIT_H
    w, t, pitch, v0 = h * 0.5, h * STROKE, h * 0.72, REC_UV[1] - h / 2
    return {(i, seg): ((REC_DIGITS_U + i * pitch + fu * w, v0 + fv * h), (w, t) if horizontal else (t, h / 2))
            for i in range(2) for seg, (fu, fv, horizontal) in C.SEGMENTS.items()}


def build_screen(stage, name, centre_xy, z):
    screen = SCREENS[name]
    path = screen['path']
    UsdGeom.Xform.Define(stage, path)
    glass = _render_material(stage, path + '/Materials/Glass', (0.004, 0.018, 0.010), matte=True)
    lit = _render_material(stage, path + '/Materials/Segment', SEGMENT_ON, emission=C.SEGMENT_EMISSION, matte=True)
    amber = _render_material(stage, path + '/Materials/Amber', C.AMBER, emission=C.SEGMENT_EMISSION, matte=True)
    _bind(_flat(stage, path + '/Glass', centre_xy, screen['size'], z - 0.0004, (0.004, 0.018, 0.010)), glass)
    w, h, t = screen['height'] * 0.5, screen['height'], screen['height'] * STROKE
    for (i, seg), (u, v) in _segment_points(screen).items():
        horizontal = C.SEGMENTS[seg][2]
        _bind(_flat(stage, f'{path}/d{i}{seg}', _xy(centre_xy, u, v), (w, t) if horizontal else (t, h / 2), z, SEGMENT_ON), lit)
    _bind(_flat(stage, path + '/Point', _xy(centre_xy, *_point_uv(screen)), (t * POINT_SIZE, t * POINT_SIZE), z, SEGMENT_ON), lit)
    if screen['rec']:
        _bind(_text(stage, path + '/Rec', 'REC', _xy(centre_xy, *REC_UV), z, REC_TEXT_H, C.AMBER), amber)
        for (i, seg), ((u, v), size) in _rec_digit_rects().items():
            _bind(_flat(stage, f'{path}/r{i}{seg}', _xy(centre_xy, u, v), size, z, SEGMENT_ON), lit)
    u0, v0, pitch = _digit_origin(screen)
    _bind(_text(stage, path + '/Unit', screen['unit'], _xy(centre_xy, _digit_u(screen, screen['digits'] - 1) + pitch + 0.004, v0 + 0.004),
                z, 0.008, SEGMENT_ON, align='left'), lit)
    if screen['mark']:
        _bind(_text(stage, path + '/Mark', screen['mark'], _xy(centre_xy, *MARK_UV), z, MARK_TEXT_H, C.AMBER), amber)
    for prim_path in _screen_toggles(name):
        _remember_lit(stage.GetPrimAtPath(prim_path))
    show(stage, name, blank_text(name), None)


def _screen_toggles(name):
    screen = SCREENS[name]
    paths = [f"{screen['path']}/d{i}{seg}" for i in range(screen['digits']) for seg in C.SEGMENTS] + [screen['path'] + '/Point']
    if screen['rec']:
        paths += [screen['path'] + '/Rec'] + [f"{screen['path']}/r{i}{seg}" for i in range(2) for seg in C.SEGMENTS]
    return paths + ([screen['path'] + '/Mark'] if screen['mark'] else [])


def blank_text(name):
    return {'digits': ' ' * SCREENS[name]['digits'], 'point': False, 'mark': False, 'rec': False, 'record': '  '}


def screen_text(name, shown):
    """Seven-segment text of a screen from TitrationBench.display: two digits before the point, two after."""
    value = shown['pH'] if name == 'ph' else shown['counter_mL']
    whole, _, frac = f'{float(value):.2f}'.partition('.')
    meter = name == 'ph'
    record = f"{shown['record'] % 100:2d}" if meter and shown.get('record') else '  '
    return {'digits': whole.rjust(2)[-2:] + frac, 'point': True, 'mark': bool(shown['stable']) if meter else False,
            'rec': bool(shown.get('recording')) if meter else False, 'record': record}


def show(stage, name, text, previous):
    if text == previous:
        return text
    path = SCREENS[name]['path']
    for i, char in enumerate(text['digits']):
        if previous is not None and previous['digits'][i] == char:
            continue
        for seg in C.SEGMENTS:
            _light(stage, f'{path}/d{i}{seg}', seg in C.GLYPHS[char])
    if previous is None or previous['point'] != text['point']:
        _light(stage, path + '/Point', text['point'])
    if SCREENS[name]['mark'] and (previous is None or previous['mark'] != text['mark']):
        _light(stage, path + '/Mark', text['mark'])
    if SCREENS[name]['rec']:
        if previous is None or previous['rec'] != text['rec']:
            _light(stage, path + '/Rec', text['rec'])
        for i, char in enumerate(text['record']):
            if previous is None or previous['record'][i] != char:
                for seg in C.SEGMENTS:
                    _light(stage, f'{path}/r{i}{seg}', seg in C.GLYPHS[char])
    return text


# --- build ------------------------------------------------------------------------------------------------
def build(stage, config):
    """Author the bench in its unjittered pose. config: {'titrant_mL': volume, 'beaker_mL': {label: volume}}.
    Button joints are authored later (chem_apparatus.author_joints), after any layout jitter."""
    retire_hooke_apparatus(stage)
    repair_cube_extents(stage)
    z0 = table_top_z(stage)
    origin = C.ORIGIN_XY
    materials = {'rubber': _material(stage, '/World/PhysicsMaterials/ChemRubber', 0.8, 0.7),
                 'pmma': _material(stage, '/World/PhysicsMaterials/ChemPMMA', 0.25, 0.20),
                 'cell': _material(stage, '/World/PhysicsMaterials/ChemCell', 0.25, 0.20, compliant=OBJECT_CONTACT),
                 'seat': _material(stage, '/World/PhysicsMaterials/ChemSeat', 0.20, 0.15)}
    root = UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(origin[0], origin[1], z0))
    L = lambda xy: _local(xy, origin)
    seats = {}
    # Rack: three sample carriers in front, the two buffer carriers behind (labels printed by the sample wells only).
    bu, bv, bh = RACK['base']
    rb = L(_xy(RACK['centre'], 0.0, RACK['base_v']))
    _box(stage, ROOT + '/Rack', (bu, bv, bh), (rb[0], rb[1], bh / 2), BASE_GREY, material=materials['rubber'])
    for label, xy in rack_slots().items():
        if label in BEAKERS:
            _text(stage, f'{ROOT}/RackLabel{label}', label, L(_xy(xy, 0.0, RACK['labels_v'])), bh + 0.0004, 0.012, PAPER)
        seats[f'rack:{label}'] = well(stage, f'{ROOT}/RackSeat{label}', L(xy), bh, SEAT_OUTER, SEAT_MOUTH, SEAT_WELL, SEAT_H,
                                      SEAT_DEPTH, SEAT_CHAMFER, RACK['colour'], materials['seat'])
    # Stirrer station.
    UsdGeom.Xform.Define(stage, ROOT + '/Station')
    UsdGeom.Xform.Define(stage, ROOT + '/Buttons')
    S = STATION['seat']
    bu, bv, bh = STATION['base']
    sc = L(_xy(S, 0.0, STATION['base_v']))
    _box(stage, ROOT + '/StationBase', (bu, bv, bh), (sc[0], sc[1], bh / 2), BASE_GREY, material=materials['rubber'])
    seat_xy = L(S)
    _box(stage, ROOT + '/StirrerPlate', (2 * SEAT_OUTER[0] + 0.012, 2 * SEAT_OUTER[1] + 0.012, 0.004),
         (seat_xy[0], seat_xy[1], bh + 0.002), (0.20, 0.21, 0.23), collide=False)
    seats['stirrer'] = well(stage, ROOT + '/StirrerSeat', seat_xy, bh + 0.004, SEAT_OUTER, SEAT_MOUTH, SEAT_WELL, SEAT_H,
                            SEAT_DEPTH, SEAT_CHAMFER, STATION['colour'], materials['seat'])
    carrier_z = bh + 0.004 + SEAT_FLOOR
    top_z = carrier_z + CARRIER_TOP_Z
    plinth = 0.020
    side, inner_h = C._BOTTLE_INNER['titrant'] = C.bottle_inner(config['titrant_mL'])
    wall = C.BOTTLE_WALL
    outer, top = side + 2 * wall, wall + inner_h
    bxy = L(_xy(S, 0.0, STATION['bottle_v']))
    _box(stage, ROOT + '/StationShelf', (outer + 0.02, outer + 0.02, plinth), (bxy[0], bxy[1], bh + plinth / 2),
         (0.20, 0.21, 0.23), material=materials['rubber'])
    bottle = ROOT + '/Station/TitrantBottle'
    xform = UsdGeom.Xform.Define(stage, bottle)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(bxy[0], bxy[1], bh + plinth))
    glass, dark = (0.30, 0.36, 0.42), (0.20, 0.21, 0.23)   # nothing white near the carrier's handle rims
    # Collider: the undrawn glass panes, exactly the drawn outline (pump, riser and line collide as drawn).
    size, cz = ((outer, outer, top + wall), (top + wall) / 2) if C.scene_colliders.ENABLED else \
        ((outer + 0.002, outer + 0.002, C.BOTTLE_COLUMN_H), C.BOTTLE_COLUMN_H / 2)
    _box(stage, bottle + '/Collider', size, (0, 0, cz),
         (1, 1, 1), material=materials['rubber'])
    UsdGeom.Imageable(stage.GetPrimAtPath(bottle + '/Collider')).MakeInvisible()
    _box(stage, bottle + '/Glass', (outer, outer, wall), (0, 0, wall / 2), glass, collide=False)
    _box(stage, bottle + '/Liquid', (side, side, 0.01), (0, 0, wall + 0.005), (0.72, 0.84, 0.92), collide=False)
    for e, (dx, dy) in enumerate(((1, 1), (1, -1), (-1, -1), (-1, 1))):
        _box(stage, f'{bottle}/Edge{e}', (wall, wall, top), (dx * (side + wall) / 2, dy * (side + wall) / 2, top / 2), glass,
             collide=False)
    _box(stage, bottle + '/Shoulder', (outer, outer, wall), (0, 0, top + wall / 2), glass, collide=False)
    # three lines, each sized to the label width: the concentration is legible on the narrow bottle
    C.bottle_label(stage, bottle, ['NaOH', str(config.get('titrant_label', '0.1')), 'mol/L'], outer, top)
    pump_top = top + wall + 0.020
    _cylinder(stage, bottle + '/Pump', 0.012, 0.020, (0, 0, pump_top - 0.010), dark, collide=False)
    hose_z = top_z + SPOUT_ABOVE + 0.012
    riser = hose_z + 0.003 - (bh + plinth + pump_top)
    _cylinder(stage, bottle + '/Riser', 0.003, riser, (0, 0, pump_top + riser / 2), dark, collide=False)
    # The line runs forward above the carrier's back handle and ends over the beaker, beside the electrode.
    end = L(_xy(S, 0.014, 0.014))
    C._pipe(stage, ROOT + '/Station/TitrantLine', bxy, end, hose_z, 0.003, dark)
    _text(stage, ROOT + '/Station/TitrantLabel', 'NaOH', L(_xy(S, 0.055, STATION['bottle_v'])), bh + 0.0004, 0.010, PAPER)
    for level in TITRANT_LEVELS:
        _text(stage, f'{ROOT}/Station/{level.title()}Label', TITRANT_TEXT[level] + ' mL',
              L(_xy(S, STATION['button_u'][level], STATION['labels_v'])), bh + 0.0004, 0.007, PAPER)
    build_screen(stage, 'counter', L(_xy(S, *STATION['counter'])), bh + 0.0010)
    # Meter: screen and the electrode's park seat on one housing.
    M = METER['centre']
    mc = L(M)
    bu, bv, bh_m = METER['body']
    _box(stage, ROOT + '/Meter', (bu, bv, bh_m), (mc[0], mc[1], bh_m / 2), HOUSING_BLUE, material=materials['rubber'])
    park = L(park_xy())
    seats_park = well(stage, PARK_SEAT, park, bh_m, (PARK_OUTER, PARK_OUTER), (PARK_MOUTH, PARK_MOUTH),
                      (PARK_HOLE, PARK_HOLE), PARK_H, PARK_DEPTH, 0.003, METER['colour'], materials['seat'])
    build_screen(stage, 'ph', L(_xy(M, *METER['screen'])), bh_m + 0.0010)
    _text(stage, ROOT + '/MeterLabel', 'pH METER', L(_xy(M, 0.045, -0.085)), bh_m + 0.0004, 0.010, PAPER)
    # Buttons.
    colours = {'large': (0.10, 0.32, 0.75), 'coarse': (0.20, 0.50, 0.85), 'fine': (0.45, 0.68, 0.95),
               'zero': (0.55, 0.35, 0.80), 'rec': (0.80, 0.15, 0.12)}
    texts = dict(TITRANT_TEXT, zero='ZERO', rec='REC')
    buttons = {}
    for bid, (xy, base, kind) in button_layout().items():
        level = bid.split(':')[1]
        buttons[bid] = _button(stage, bid, L(xy), base, kind, colours[level], texts[level], materials,
                               INK if level == 'fine' else (0.97, 0.97, 0.95))
    # Carriers in the rack, the electrode on its park seat.
    carriers = {}
    for label, xy in rack_slots().items():
        carriers[label] = _carrier(stage, label, L(xy), RACK['base'][2] + SEAT_FLOOR + 0.0003, materials)
        set_beaker_liquid(stage, label, config['beaker_mL'][label])
    electrode = _electrode(stage, park, bh_m + PARK_H, materials)
    build_cable(stage, (park[0], park[1], bh_m + PARK_H + 0.0003 + CABLE_ATTACH_Z))
    table_top = stage.GetPrimAtPath(TABLE_TOP)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(table_top)
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    C.finish_looks(stage)
    set_titrant(stage, 1.0)
    return {'carriers': carriers, 'electrode': electrode, 'buttons': buttons,
            'seats': {k: str(v.GetPath()) for k, v in seats.items()}, 'park': str(seats_park.GetPath())}


def set_titrant(stage, fraction, rgb=(0.72, 0.84, 0.92)):
    side, inner_h = C._BOTTLE_INNER['titrant']
    path = ROOT + '/Station/TitrantBottle/Liquid'
    cube = UsdGeom.Cube(stage.GetPrimAtPath(path))
    height = max(1e-4, inner_h / C.BOTTLE_HEADSPACE * min(1.0, max(0.0, fraction)))
    api = UsdGeom.XformCommonAPI(cube)
    api.SetTranslate(Gf.Vec3d(0, 0, C.BOTTLE_WALL + height / 2))
    api.SetScale(Gf.Vec3f(side, side, height))
    C._set_liquid_colour(stage, path, rgb)


def park_frame(stage):
    """World xy of the park seat's hole and the world z of its rim (the electrode's disc rests there)."""
    prim = stage.GetPrimAtPath(PARK_SEAT)
    points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
    world = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    p = np.asarray(world.Transform(Gf.Vec3d(*points[8:12].mean(axis=0).tolist())))
    return p[:2].copy(), float(p[2])

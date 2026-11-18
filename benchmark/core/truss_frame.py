"""Truss bench loading frame (truss_bridge): the Agent loads its own bridge with stackable knob weights and its
held load is its score. Built with the bench (truss_apparatus.build / author_joints); every part keeps its collider.

The frame. A steel wire rope leaves the lower end of the load pin C (in a groove under the board) in the load direction
(truss_bench.LOAD_Y, away from the robot), runs along the table under the load platform to a sheave at the foot of a
two-column guide frame past the board's far edge, up behind the columns, over a sheave at their top and down onto the
crossbar of a carriage that runs on the columns. The carriage carries the load platform (a plate with a low rim,
cantilevered toward the robot, open above). The carriage and the platform are one body of HANGER_KG; the platform
rests on two catch blocks DROP_M below its unloaded height once the rope goes slack (after a failure). The display
beside the platform shows the rope's pull (the force on C) to 0.1 N, DISPLAY_HZ times a second.

The weights. Stackable knob weights of polished steel (2 kg, 0.5 kg) and brass (0.1 kg): a cylinder with a knob on top
(neck and head) and a round recess underneath that takes the knob of the weight below, so they stack centred by
contact alone (no snapping). They stand stacked by size in the weight case at the robot's right. The gripper takes a
weight by its knob, as it takes a bar by its grip knob.

Physics. C is the board's fixed pin, so the rope's pull goes into the board; the bridge on the board is the built
bridge with every collider (pins in holes, bars on pins by contact), and its response is the evaluator's analysis of
the bars seated when the loading starts (truss_bench.bridge_capacity; its deflection at failure, 0.1-0.4 mm, is below
the eyes' play on the pins). The rope is the carriage's guide joint (prismatic, vertical) with a spring-damper drive:
stiffness ROPE_K (the wire rope in series with the bridge at C) and damping ROPE_C; the drive's force is the rope's
pull. The pull follows the physics every tick: a weight let go onto the platform, the robot pressing on the platform or
a weight, a weight dropped from above, all pull as they do. LoadMonitor (in the backend, every tick) reads the pull,
starts the loading when the first weight touches the platform (it freezes the bridge as it stands then: its capacity,
its mass, the bench), fails the bridge at the first tick the pull reaches the capacity, keeps the held load (the
largest settled pull while the bridge is intact: the platform still for SETTLE_S and nothing of the robot touching the
platform or a weight on it) and records every touch of the robot (or anything it holds, or a weight) on the bridge from
the loading's start (the truss_hands_off gate).
"""
from __future__ import annotations

import json
import math
import os
from collections import deque

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

import chem_apparatus as C
from chem_apparatus import ROOT, _rigid, _text
from torsion_apparatus import _box, _collider, _cylinder
import truss_bench as B
import truss_model as M

G = 9.81
FRAME = ROOT + '/LoadFrame'
CARRIAGE = FRAME + '/Carriage'
GUIDE_JOINT = C.PHYSICS_ROOT + '/LoadGuide'
WEIGHTS = ROOT + '/Weights'
STEEL, DARK, ALU = (0.62, 0.63, 0.66), (0.16, 0.17, 0.19), (0.72, 0.73, 0.75)
BRASS = (0.78, 0.62, 0.22)
CABLE_COLOUR = (0.92, 0.88, 0.70)

# --- geometry (world metres on the table; z above it) -----------------------------------------------------------
D = B.LOAD_Y                               # -1: the load pulls C toward world -y
CX, CY = B.pin_xy(M.LOAD_PIN)
ROPE_Z = 0.006                             # the rope's run along the table (under the board in its groove)
EDGE_Y = B.board_rect()[1]                 # the board's far edge (the rope comes out of the groove there)
COLUMN_X = (CX - 0.075, CX + 0.075)        # the guide frame's two columns (square tube)
COLUMN_Y, COLUMN_W, COLUMN_H = -0.435, 0.020, 0.680     # 20-series T-slot extrusion
FOOT = (0.050, 0.060, 0.008)               # the frame's foot pads under the two columns (the rope runs between)
R_FOOT, R_TOP = 0.015, 0.0275              # sheaves: at the frame's foot (rope under it, then up) and at its top
FOOT_SHEAVE_YZ = (COLUMN_Y - 0.010, ROPE_Z + R_FOOT)
UP_Y = FOOT_SHEAVE_YZ[0] - R_FOOT          # the rising run, behind the columns
TOP_Z = 0.640                              # the top sheave's axle
TOP_SHEAVE_Y = UP_Y + R_TOP
DOWN_Y = UP_Y + 2 * R_TOP                  # the falling run, onto the carriage's crossbar
BEAM = (0.170, 0.020, 0.016)               # the top beam across the columns (above the top sheave)
# the carriage: crossbar between the columns, the platform cantilevered toward the robot, one body
CROSSBAR = (0.126, 0.030, 0.060)
SHOE = (0.005, 0.026)                      # guide shoes beside each column (x, y), 1 mm clear of it
CROSSBAR_Y = COLUMN_Y + 0.030              # its centre (its back face clear of the columns' front faces)
PLATFORM_X = (CX - 0.210, CX + 0.210)        # four stacks of 2 kg weights in a row along x (105 mm pitch)
PLATFORM_Y = (CROSSBAR_Y + CROSSBAR[1] / 2, EDGE_Y - 0.035)
PLATFORM_T = 0.010
PLATFORM_TOP = 0.145                       # the platform's top (the weights' floor) unloaded
RIM_H, RIM_T = 0.008, 0.004
HOOK_Z = PLATFORM_TOP - PLATFORM_T / 2 + CROSSBAR[2] / 2 + 0.010      # the rope's end on the crossbar
# frame probe only (never in an evaluated episode): the bridge never fails; the EMBODIEDLAB_TRUSS_* overrides below
PROBE = os.environ.get('EMBODIEDLAB_TRUSS_FRAME_PROBE', '0') == '1'
HANGER_KG = 1.0                            # carriage + platform
if PROBE and os.environ.get('EMBODIEDLAB_TRUSS_HANGER_KG'):
    HANGER_KG = float(os.environ['EMBODIEDLAB_TRUSS_HANGER_KG'])
DROP_M = 0.035                             # the platform's fall to the catch blocks once the rope goes slack
CATCH = (0.030, 0.100)                     # catch blocks (x, y) under the platform's two ends
DISPLAY_XY, DISPLAY_SIZE = (PLATFORM_X[1] + 0.080, -0.300), (0.130, 0.060, 0.025)   # right of the platform (in
                                           # front of the tester), the scrap tray left of it
DISPLAY_PATH = FRAME + '/LoadDisplay'          # the seven-segment line (truss_apparatus.load_display)
DISPLAY_HZ = 10
RESOLUTION_N = 0.1

# the rope as a spring-damper: wire rope 1.5 mm (EA ~ 1.1e5 N, ~0.75 m) in series with the bridge at C (median 1.3e6
# N/m, truss_model unit forces over the 160 study designs; +-10 % across bridges): one stiffness for every bridge. The
# damping is an ASSUMPTION (rope over two sheaves): zeta 0.03 at 20 kg on the platform.
ROPE_K = 1.3e5                             # N/m
ROPE_ZETA, ROPE_ZETA_KG = 0.03, 20.0
ROPE_C = 2.0 * ROPE_ZETA * math.sqrt(ROPE_K * ROPE_ZETA_KG)           # ~97 N s/m
GUIDE_LIMITS = (-0.045, 0.010)             # the guide's travel (m, the catch stops the fall at -DROP_M)

# --- the weights --------------------------------------------------------------------------------------------------
# kg -> (body radius, body height, how many, material colour). Stackable: knob (neck + head) on top, a recess under
# the body that takes a knob with play. The nominal mass is exact (calibrated weights).
WEIGHT_KINDS = {2.0: (0.040, 0.050, 20, STEEL), 0.5: (0.025, 0.032, 6, STEEL), 0.1: (0.015, 0.018, 10, BRASS)}
NECK_R, NECK_H = 0.0045, 0.003
HEAD_R, HEAD_H = 0.0070, 0.003            # the head's straight band (the fingers close on it) ...
CAP_H = 0.0055                            # ... under a conical cap: the lead-in that centres the weight set on it
KNOB_H = NECK_H + HEAD_H + CAP_H          # 11.5 mm, inside the 12 mm recess
RECESS_R, RECESS_H = 0.0080, 0.012          # 1.0 mm radial play on the knob head (with 0.2 mm a weight
                                            # held by its knob jams on the knob below); the cap's lead-in
                                            # centres it
RECESS_PIECES = 16
# the weight case at the robot's right, on the table (its top ends at world x 1.55): a tray with a low wall; weights
# stacked by size in three rows along x, 180 / 165 mm apart: the 2 kg stacks two per row (105 mm pitch) in the two far
# rows, the small ones (45-60 mm pitch) in the near row. The hand (about 0.2 m along its closing axis, world y) never
# reaches over another row; within a row it clears its neighbours while they are no taller than one weight more
CASE_INNER = (1.300, 0.190, 1.525, 0.630)  # x0, y0, x1, y1
CASE_FLOOR, CASE_WALL, CASE_WALL_H = 0.008, 0.005, 0.015
# a sprung seat under each stack: a steel seat plate on a rubber pad, the locating peg on the plate. It gives a little
# under a press, as the rubber or felt in a real weight case does. The rubber's compression is the seat body's vertical
# joint with a spring-damper drive (stiffness PAD_K: a full 2 kg stack sinks it about 0.5 mm); the body clears the
# floor by PAD_GAP. The steel plate gives the seat its mass: a light body between the spring and a heavy stack
# jitters (a 25 g rubber disc kept the case from settling, 0.06 m/s)
RUBBER_H, SEAT_H, PAD_GAP = 0.003, 0.010, 0.0003
PAD_H = RUBBER_H + SEAT_H
PAD_K, PAD_C = 2.0e5, 800.0
PAD_LIMITS = (-0.0025, 0.0010)
CASE_SEAT = CASE_FLOOR + PAD_GAP + PAD_H     # a pad's top, unloaded
CASE_STACKS = {2.0: [(1.360, 0.240), (1.465, 0.240), (1.360, 0.420), (1.465, 0.420)],
               0.5: [(1.335, 0.585), (1.395, 0.585)],
               0.1: [(1.450, 0.585), (1.495, 0.585)]}

# --- the monitor -------------------------------------------------------------------------------------------------
SETTLE_S = 0.5                             # the platform still this long (and untouched) for a held reading
STILL_M_S = 0.001
TRACE_TICKS, TRACE_STEP_N = 8, 0.5         # private trace while loading: every 8 ticks and on a 0.5 N change
ROBOT_LINKS = tuple(f'panda_link{k}' for k in range(8)) + ('panda_hand', 'panda_leftfinger', 'panda_rightfinger')


def weight_id(kg, k):
    return {2.0: 'w2kg', 0.5: 'w05kg', 0.1: 'w01kg'}[kg] + f'{k + 1:02d}'


def weight_kg(key):
    return {'w2kg': 2.0, 'w05kg': 0.5, 'w01kg': 0.1}[key[:-2]]


def is_weight(key):
    return key.startswith(('w2kg', 'w05kg', 'w01kg'))


def weight_path(key):
    return f'{WEIGHTS}/{key}'


def weight_ids():
    return [weight_id(kg, k) for kg, (_, _, n, _) in WEIGHT_KINDS.items() for k in range(n)]


def stack_spots(kg):
    """Initial (x, y, bottom z) of each weight of a size in the case: stacked, the first stacks filled first."""
    r, h, n, _ = WEIGHT_KINDS[kg]
    spots = CASE_STACKS[kg]
    per = math.ceil(n / len(spots))
    return [(spots[k // per][0], spots[k // per][1], CASE_SEAT + (k % per) * h) for k in range(n)]


def case_peg_xy():
    return [xy for kg in WEIGHT_KINDS for xy in CASE_STACKS[kg]]


def case_pad_radius(k):
    """The k-th pad's radius (under a stack of case_peg_xy()[k]): a little inside its weights' outline."""
    sizes = [kg for kg in WEIGHT_KINDS for _ in CASE_STACKS[kg]]
    return 0.9 * WEIGHT_KINDS[sizes[k]][0]


def case_pad_path(k):
    return f'{FRAME}/CasePad{k:02d}'


def case_pad_k(k):
    """The pads' stiffness (frame probe only: EMBODIEDLAB_TRUSS_PAD_K, a comma list per pad, overrides it)."""
    raw = os.environ.get('EMBODIEDLAB_TRUSS_PAD_K', '') if PROBE else ''
    values = [float(v) for v in raw.split(',') if v.strip()]
    return values[k] if k < len(values) else PAD_K


def case_peg_paths():
    """The pads' and pegs' collider prims."""
    return [f'{case_pad_path(k)}/{part}' for k in range(len(case_peg_xy()))
            for part in ('Pad/Collider', 'Seat/Collider', 'Neck/Collider', 'Head/Collider', 'Cap')]


PROBE_SPOTS = [(CX - 0.1575, -0.275), (CX - 0.0525, -0.275), (CX + 0.0525, -0.275), (CX + 0.1575, -0.275)]


def preload():
    """Frame probe only (EMBODIEDLAB_TRUSS_PRELOAD, JSON [[kg, spot], ...]): weights that start on the platform,
    stacked in that order on PROBE_SPOTS, taken from the tops of the case stacks."""
    raw = os.environ.get('EMBODIEDLAB_TRUSS_PRELOAD', '') if PROBE else ''
    return [(float(kg), int(spot)) for kg, spot in json.loads(raw)] if raw.strip() else []


def initial_weights():
    out = {weight_id(kg, k): xyz for kg in WEIGHT_KINDS for k, xyz in enumerate(stack_spots(kg))}
    tops = {kg: sorted(range(WEIGHT_KINDS[kg][2]), key=lambda k: (-out[weight_id(kg, k)][2], k)) for kg in WEIGHT_KINDS}
    height = {}
    for kg, spot in preload():
        key = weight_id(kg, tops[kg].pop(0))
        x, y = PROBE_SPOTS[spot]
        out[key] = (x, y, PLATFORM_TOP + height.get(spot, 0.0))
        height[spot] = height.get(spot, 0.0) + WEIGHT_KINDS[kg][1]
    return out


# the solver iterations of the carriage and the weights (one island once a weight is on the platform). Measured
# (60 s idle, four 2 kg weights on the 1 kg carriage): at the default 16/8 one weight crept 13 mm and the others
# up to 0.8 mm, enough to break a hold on the level above; at 64/32 at most 0.6 mm (+9 % wall time)
SOLVER_ITERS = (64, 32)


def _solver_iterations(prim):
    """The carriage's and the weights' solver iterations: SOLVER_ITERS; frame probe only, EMBODIEDLAB_TRUSS_ITERS
    'position,velocity' instead."""
    raw = os.environ.get('EMBODIEDLAB_TRUSS_ITERS', '') if PROBE else ''
    pos, vel = (int(v) for v in raw.split(',')) if raw.strip() else SOLVER_ITERS
    body = PhysxSchema.PhysxRigidBodyAPI(prim)
    body.GetSolverPositionIterationCountAttr().Set(pos)
    body.GetSolverVelocityIterationCountAttr().Set(vel)


def platform_rect():
    return (PLATFORM_X[0], min(PLATFORM_Y), PLATFORM_X[1], max(PLATFORM_Y))


def in_case(xyz):
    x0, y0, x1, y1 = CASE_INNER
    return x0 - 0.005 <= xyz[0] <= x1 + 0.005 and y0 - 0.005 <= xyz[1] <= y1 + 0.005 and xyz[2] < 0.40


def on_platform(xyz):
    """A weight (frame at its bottom centre) over the platform, at or above its floor (the floor may sink under load or
    drop to the catch)."""
    x0, y0, x1, y1 = platform_rect()
    return x0 <= xyz[0] <= x1 and y0 <= xyz[1] <= y1 and PLATFORM_TOP - DROP_M - 0.006 <= xyz[2] <= 1.2


def classify_weights(poses, held):
    """{weight: 'held' | 'case' | 'load' | 'loose'} (poses: {id: (position, quat)}, table frame)."""
    out = {}
    for key, (p, _) in poses.items():
        if not is_weight(key):
            continue
        if key in held:
            out[key] = 'held'
        elif on_platform(p):
            out[key] = 'load'
        elif in_case(p):
            out[key] = 'case'
        else:
            out[key] = 'loose'
    return out


# --- build ---------------------------------------------------------------------------------------------------------
def _sheave(stage, path, centre, radius, thickness):
    """A sheave turning about world x (drawn; its axle on a static bracket). Static collider: the disc."""
    wheel = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(wheel).SetTranslate(Gf.Vec3d(*centre))
    disc = _cylinder(stage, path + '/Disc', radius, thickness, (0, 0, 0), STEEL)
    UsdGeom.XformCommonAPI(disc).SetRotate(Gf.Vec3f(0, 90, 0))
    _box(stage, path + '/Spoke', (thickness * 1.15, radius * 0.35, radius * 1.8), (0, 0, 0), DARK, collide=False)
    return wheel


def _run(stage, path, p, q, collide):
    """A straight rope run from p to q (a thin box along it)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    length = float(np.linalg.norm(d))
    yaw = math.degrees(math.atan2(-d[0], d[1]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    return _box(stage, path, (0.0016, max(length, 1e-4), 0.0016), tuple((p + q) / 2), CABLE_COLOUR,
                rotate_xyz=(pitch, 0.0, yaw), collide=collide)


def rope_points(drop=0.0):
    """The rope's straight runs (start, end) with the platform `drop` m below its unloaded height."""
    return [((CX, EDGE_Y, ROPE_Z), (CX, FOOT_SHEAVE_YZ[0], ROPE_Z)),
            ((CX, UP_Y, FOOT_SHEAVE_YZ[1]), (CX, UP_Y, TOP_Z)),
            ((CX, DOWN_Y, TOP_Z), (CX, DOWN_Y, HOOK_Z - drop))]


def _cap(stage, path, radius, height, centre, colour, material):
    """The knob's conical cap (apex up, colliding; the looks render it as part of the weight's lathe)."""
    cone = UsdGeom.Cone.Define(stage, path)
    cone.CreateRadiusAttr(radius)
    cone.CreateHeightAttr(height)
    cone.CreateAxisAttr('Z')
    cone.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2), Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.XformCommonAPI(cone).SetTranslate(Gf.Vec3d(*centre))
    cone.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    _collider(cone.GetPrim(), material)
    return cone


def _weight(stage, key, xyz, materials):
    """One weight: a dynamic body, its frame at the bottom centre: the body above the recess (a cylinder), the skirt
    round the recess (RECESS_PIECES boxes), the knob's neck and head (cylinders), its conical cap."""
    kg = weight_kg(key)
    r, h, _, colour = WEIGHT_KINDS[kg]
    path = weight_path(key)
    xform = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(xyz[0], xyz[1], xyz[2] + 0.0002))
    material = materials['steel']
    upper = h - RECESS_H
    _cylinder(stage, path + '/Body', r, upper, (0, 0, RECESS_H + upper / 2), colour, material)
    radial = r - RECESS_R
    tangential = 2 * r * math.tan(math.pi / RECESS_PIECES)
    r_mid = RECESS_R + radial / 2
    for k in range(RECESS_PIECES):
        th = 2 * math.pi * (k + 0.5) / RECESS_PIECES
        _box(stage, f'{path}/Skirt{k:02d}', (radial, tangential, RECESS_H),
             (r_mid * math.cos(th), r_mid * math.sin(th), RECESS_H / 2), colour,
             rotate_xyz=(0, 0, math.degrees(th)), material=material)
    _cylinder(stage, path + '/Neck', NECK_R, NECK_H, (0, 0, h + NECK_H / 2), colour, material)
    _cylinder(stage, path + '/Head', HEAD_R, HEAD_H, (0, 0, h + NECK_H + HEAD_H / 2), colour, material)
    _cap(stage, path + '/Cap', HEAD_R, CAP_H, (0, 0, h + NECK_H + HEAD_H + CAP_H / 2), colour, material)
    _rigid(xform.GetPrim(), kg, (0.0, 0.0, h * 0.55), ccd=True)
    _solver_iterations(xform.GetPrim())
    # The grasp's filter API is authored here with no targets, so a later grasp only sets
    # targets and never adds an API schema mid-run (that resync jolted the loaded platform).
    UsdPhysics.FilteredPairsAPI.Apply(xform.GetPrim()).CreateFilteredPairsRel()
    inertia = UsdPhysics.MassAPI(xform.GetPrim())
    inertia.CreateDiagonalInertiaAttr(Gf.Vec3f(kg * (3 * r * r + h * h) / 12, kg * (3 * r * r + h * h) / 12, kg * r * r / 2))
    return path


def build(stage, materials):
    """The frame (static parts, the carriage body), the case and the weights. Returns {weight id: path}."""
    UsdGeom.Xform.Define(stage, FRAME)
    m = materials['steel']
    # guide frame: two columns on foot pads, the top beam; the foot sheave on its bracket, the top sheave on its axle
    for k, x in enumerate(COLUMN_X):
        _box(stage, f'{FRAME}/Foot{k}', FOOT, (x, COLUMN_Y, FOOT[2] / 2), DARK, material=m)
        _box(stage, f'{FRAME}/Column{k}', (COLUMN_W, COLUMN_W, COLUMN_H - FOOT[2]),
             (x, COLUMN_Y, FOOT[2] + (COLUMN_H - FOOT[2]) / 2), ALU, material=m)
    _box(stage, FRAME + '/Beam', BEAM, (CX, COLUMN_Y, COLUMN_H + BEAM[2] / 2), ALU, material=m)
    for k, dx in enumerate((-0.007, 0.007)):
        _box(stage, f'{FRAME}/FootCheek{k}', (0.003, 0.026, FOOT_SHEAVE_YZ[1] + 0.010),
             (CX + dx, FOOT_SHEAVE_YZ[0], (FOOT_SHEAVE_YZ[1] + 0.010) / 2), DARK, material=m)
    _sheave(stage, FRAME + '/FootSheave', (CX, *FOOT_SHEAVE_YZ), R_FOOT, 0.008)
    _box(stage, FRAME + '/TopAxle', (COLUMN_X[1] - COLUMN_X[0] - COLUMN_W, 0.008, 0.008), (CX, TOP_SHEAVE_Y, TOP_Z),
         STEEL, material=m)
    _sheave(stage, FRAME + '/TopSheave', (CX, TOP_SHEAVE_Y, TOP_Z), R_TOP, 0.010)
    # the rope: fixed runs collide (static), the falling run is drawn to the carriage each frame (its collider fixed at
    # the unloaded length)
    for k, (p, q) in enumerate(rope_points()):
        _run(stage, f'{FRAME}/RopeCollider{k}', p, q, True)
        UsdGeom.Imageable(stage.GetPrimAtPath(f'{FRAME}/RopeCollider{k}')).MakeInvisible()
        _run(stage, f'{FRAME}/Rope{k}', p, q, False)
    # catch blocks under the platform's two ends
    catch_top = PLATFORM_TOP - PLATFORM_T - DROP_M
    cy = (PLATFORM_Y[0] + PLATFORM_Y[1]) / 2
    for k, x in enumerate((PLATFORM_X[0] + CATCH[0] / 2 + 0.005, PLATFORM_X[1] - CATCH[0] / 2 - 0.005)):
        _box(stage, f'{FRAME}/Catch{k}', (CATCH[0], CATCH[1], catch_top), (x, cy, catch_top / 2), DARK, material=m)
    # the carriage (one body): crossbar, the rope's eye on it, the platform plate and its rim
    car = UsdGeom.Xform.Define(stage, CARRIAGE)
    z0 = PLATFORM_TOP - PLATFORM_T / 2
    UsdGeom.XformCommonAPI(car).SetTranslate(Gf.Vec3d(CX, CROSSBAR_Y, z0))
    _box(stage, CARRIAGE + '/Crossbar', CROSSBAR, (0, 0, 0), DARK, material=m)
    for k, x in enumerate(COLUMN_X):            # a shoe each side of each column (the guide joint holds them clear)
        for j, side in enumerate((-1, 1)):
            sx = x - CX + side * (COLUMN_W / 2 + 0.001 + SHOE[0] / 2)
            if abs(sx) < CROSSBAR[0] / 2:
                continue                        # the crossbar's end is the inner shoe
            _box(stage, f'{CARRIAGE}/Shoe{k}{j}', (SHOE[0], SHOE[1] + CROSSBAR[1], CROSSBAR[2]),
                 (sx, -(SHOE[1]) / 2, 0), DARK, material=m)
    _box(stage, CARRIAGE + '/Eye', (0.010, 0.008, 0.010), (0, DOWN_Y - CROSSBAR_Y, CROSSBAR[2] / 2 + 0.005), STEEL,
         material=m)
    y0, y1 = min(PLATFORM_Y) - CROSSBAR_Y, max(PLATFORM_Y) - CROSSBAR_Y
    w = PLATFORM_X[1] - PLATFORM_X[0]
    _box(stage, CARRIAGE + '/Plate', (w, y1 - y0, PLATFORM_T), (0, (y0 + y1) / 2, 0), ALU, material=m)
    rim_z = PLATFORM_T / 2 + RIM_H / 2
    _box(stage, CARRIAGE + '/RimFront', (w, RIM_T, RIM_H), (0, y1 - RIM_T / 2, rim_z), ALU, material=m)
    for k, sx in enumerate((-1, 1)):
        _box(stage, f'{CARRIAGE}/RimSide{k}', (RIM_T, y1 - y0, RIM_H), (sx * (w / 2 - RIM_T / 2), (y0 + y1) / 2, rim_z),
             ALU, material=m)
    _rigid(car.GetPrim(), HANGER_KG, (0.0, (y0 + y1) / 2 * 0.6, 0.0))
    # the falling run's static collider ends on the eye it is fixed to and does not collide with it (in
    # contact there the empty carriage chattered 29 um every tick and read 11.7 N on average instead of 9.8 N)
    UsdPhysics.FilteredPairsAPI.Apply(car.GetPrim()).CreateFilteredPairsRel().SetTargets(
        [Sdf.Path(f'{FRAME}/RopeCollider{len(rope_points()) - 1}')])
    _solver_iterations(car.GetPrim())
    # the display beside the platform
    dx, dy = DISPLAY_XY
    _box(stage, FRAME + '/DisplayPanel', DISPLAY_SIZE, (dx, dy, DISPLAY_SIZE[2] / 2), (0.02, 0.05, 0.03), material=m)
    # the weight case and the weights
    x0, y0c, x1, y1c = CASE_INNER
    _box(stage, FRAME + '/CaseFloor', (x1 - x0 + 2 * CASE_WALL, y1c - y0c + 2 * CASE_WALL, CASE_FLOOR),
         ((x0 + x1) / 2, (y0c + y1c) / 2, CASE_FLOOR / 2), (0.10, 0.10, 0.11), material=materials['rack'])
    hz = CASE_FLOOR + CASE_WALL_H / 2
    for k, (size, centre) in enumerate((((x1 - x0 + 2 * CASE_WALL, CASE_WALL, CASE_WALL_H), ((x0 + x1) / 2, y0c - CASE_WALL / 2, hz)),
                                        ((x1 - x0 + 2 * CASE_WALL, CASE_WALL, CASE_WALL_H), ((x0 + x1) / 2, y1c + CASE_WALL / 2, hz)),
                                        ((CASE_WALL, y1c - y0c, CASE_WALL_H), (x0 - CASE_WALL / 2, (y0c + y1c) / 2, hz)),
                                        ((CASE_WALL, y1c - y0c, CASE_WALL_H), (x1 + CASE_WALL / 2, (y0c + y1c) / 2, hz)))):
        _box(stage, f'{FRAME}/CaseWall{k}', size, centre, (0.10, 0.10, 0.11), material=materials['rack'])
    # a rubber pad under each stack with a locating peg on it (a knob like the weights' own): the bottom weight's
    # recess sits over the peg, so a stack cannot slide when the weight above it is lifted off. Pad and peg are one
    # body (frame at the pad's underside) on its vertical joint (author_joints)
    for k, (x, y) in enumerate(case_peg_xy()):
        pad = case_pad_path(k)
        xform = UsdGeom.Xform.Define(stage, pad)
        UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(x, y, CASE_FLOOR + PAD_GAP))
        r = case_pad_radius(k)
        _cylinder(stage, pad + '/Pad', r, RUBBER_H, (0, 0, RUBBER_H / 2), (0.03, 0.03, 0.03), materials['rubber'])
        _cylinder(stage, pad + '/Seat', r, SEAT_H, (0, 0, RUBBER_H + SEAT_H / 2), STEEL, m)
        _cylinder(stage, pad + '/Neck', NECK_R, NECK_H, (0, 0, PAD_H + NECK_H / 2), STEEL, m)
        _cylinder(stage, pad + '/Head', HEAD_R, HEAD_H, (0, 0, PAD_H + NECK_H + HEAD_H / 2), STEEL, m)
        _cap(stage, pad + '/Cap', HEAD_R, CAP_H, (0, 0, PAD_H + NECK_H + HEAD_H + CAP_H / 2), STEEL, m)
        mass = (1100.0 * math.pi * r * r * RUBBER_H + 7850.0 * math.pi * r * r * SEAT_H
                + 7850.0 * math.pi * (NECK_R ** 2 * NECK_H + HEAD_R ** 2 * (HEAD_H + CAP_H / 3)))
        _rigid(xform.GetPrim(), mass, (0.0, 0.0, PAD_H / 2))
        UsdPhysics.FilteredPairsAPI.Apply(xform.GetPrim()).CreateFilteredPairsRel().SetTargets([Sdf.Path(FRAME + '/CaseFloor')])
    UsdGeom.Xform.Define(stage, WEIGHTS)
    return {key: _weight(stage, key, xyz, materials) for key, xyz in initial_weights().items()}


def author_joints(stage):
    """The carriage's guide (prismatic along world z, its travel GUIDE_LIMITS) with the rope's spring-damper drive,
    its target set so the unloaded carriage hangs where it was built."""
    joint = UsdPhysics.PrismaticJoint.Define(stage, GUIDE_JOINT)
    world = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(CARRIAGE)).ExtractTranslation()
    joint.CreateBody1Rel().SetTargets([Sdf.Path(CARRIAGE)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(world))
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateAxisAttr('Z')
    joint.CreateLowerLimitAttr(GUIDE_LIMITS[0])
    joint.CreateUpperLimitAttr(GUIDE_LIMITS[1])
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
    drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(ROPE_K)
    drive.CreateDampingAttr(ROPE_C)
    drive.CreateTargetPositionAttr(rest_offset())
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateMaxForceAttr(1.0e6)
    PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
    cache = UsdGeom.XformCache()
    for k in range(len(case_peg_xy())):          # the case pads: the rubber's give under the stack on it
        pad = case_pad_path(k)
        pj = UsdPhysics.PrismaticJoint.Define(stage, pad + 'Joint')
        at = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(pad)).ExtractTranslation()
        pj.CreateBody1Rel().SetTargets([Sdf.Path(pad)])
        pj.CreateLocalPos0Attr(Gf.Vec3f(at))
        pj.CreateLocalRot0Attr(Gf.Quatf(1))
        pj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        pj.CreateLocalRot1Attr(Gf.Quatf(1))
        pj.CreateAxisAttr('Z')
        pj.CreateLowerLimitAttr(PAD_LIMITS[0])
        pj.CreateUpperLimitAttr(PAD_LIMITS[1])
        d = UsdPhysics.DriveAPI.Apply(pj.GetPrim(), 'linear')
        d.CreateTypeAttr('force')
        d.CreateStiffnessAttr(case_pad_k(k))
        d.CreateDampingAttr(PAD_C)
        d.CreateTargetPositionAttr(0.0)
        d.CreateTargetVelocityAttr(0.0)
        d.CreateMaxForceAttr(1.0e5)
        PhysxSchema.PhysxJointAPI.Apply(pj.GetPrim())
    return joint


def rest_offset():
    """The drive's target (m above the built height): the unloaded carriage's weight on the spring."""
    return HANGER_KG * G / ROPE_K


def slacken(stage):
    """After a failure (post-seal): the rope goes slack; the platform falls onto its catch blocks."""
    drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath(GUIDE_JOINT), 'linear')
    drive.GetStiffnessAttr().Set(0.0)
    drive.GetDampingAttr().Set(0.0)


# --- the monitor -------------------------------------------------------------------------------------------------
class LoadMonitor:
    """Every physics tick (TrussBackend.step, after the bench): the rope's pull, the loading's start, the failure, the
    held load and the touches on the bridge. Private state; record() goes into the sealed device record."""

    def __init__(self, backend):
        self.b = backend
        self.state = 'idle'                     # idle -> loading -> failed
        self.pull = HANGER_KG * G
        self.quiet = 0
        self.held_n = 0.0
        self.peak_n = self.pull
        self.start_tick = self.fail_tick = None
        self.frozen = None                      # {'audit', 'bridge', 'structure'} at the loading's start
        self.touches = []                       # first touch per (body, by) from the loading's start
        self._touched = set()
        self.readings = []                      # settled readings (tick, N) while loading, when they change
        self.peaks = []                         # (tick, N) the largest pull of each excursion above the settled load
        self.placements = 0
        self.on_load = set()
        self.shown = None
        self.trace = deque(maxlen=480)          # last 2 s of (tick, pull, z, vz)
        self.z0 = None
        self.slack = False
        self.vz = 0.0                           # the carriage's vertical speed from its positions (tick())
        self._z_last = None

    # views (created at build time by the backend; initialised on first use)
    def _carriage(self):
        """The carriage's height and vertical speed. The speed is the change of its height over the last tick (kept by
        tick()): the rigid-body velocity read back after a step carries about g dt / 2 (~20 mm/s at 240 Hz) on every
        body at rest (19 mm/s on the still platform), which would keep the platform from ever counting
        as still and bias the damping term by ~2 N."""
        view = self.b._view(CARRIAGE, 'truss_load_carriage')
        positions, orientations = view.get_world_poses()
        z = float(np.asarray(positions, float)[0, 2])
        self.quat = np.asarray(orientations, float)[0]          # wxyz (the private trace's tilt)
        if self.z0 is None:                     # built height (world)
            self.z0 = self.b._table_z() + PLATFORM_TOP - PLATFORM_T / 2
        return z, self.vz

    def rope_pull(self):
        z, vz = self._carriage()
        if self.slack:                          # after the failure (truss_frame.slacken): the rope pulls nothing
            return 0.0, z, vz
        stretch = rest_offset() - (z - self.z0)
        return float(ROPE_K * stretch - ROPE_C * vz), z, vz

    def _advance(self):
        """Once per physics tick, before the pull is read: the carriage's speed from its height now and a tick ago."""
        z, _ = self._carriage()
        self.vz = 0.0 if self._z_last is None else (z - self._z_last) / self.b.dt
        self._z_last = z

    def drop(self):
        z, _ = self._carriage()
        return max(0.0, self.z0 - z)

    def tick(self, tick, poses):
        b = self.b
        if not getattr(self, '_table_logged', False) and getattr(b, 'audit_file', None) is not None:
            self._table_logged = True               # the table top's extent, once (layout checks)
            try:
                from pxr import Usd
                from torsion_apparatus import TABLE_TOP
                box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render']).ComputeWorldBound(
                    b.stage.GetPrimAtPath(TABLE_TOP)).ComputeAlignedRange()
                row = {'event': 'table_extent', 'min': [round(float(v), 4) for v in box.GetMin()],
                       'max': [round(float(v), 4) for v in box.GetMax()]}
                b.audit_file.write(json.dumps(row) + '\n')
                print('truss_frame:', json.dumps(row), flush=True)
            except Exception as exc:
                b.audit_file.write(json.dumps({'event': 'table_extent', 'error': repr(exc)}) + '\n')
        self._advance()
        self.pull, z, vz = self.rope_pull()
        self.trace.append((tick, round(self.pull, 3), round(z, 6), round(vz, 5)))
        if self.state == 'failed':
            return
        rows, cols, matrix = b.contact_rows, b.contact_cols, b.contact_matrix()
        touching = matrix is not None and np.linalg.norm(matrix, axis=2) > 1e-6
        where = classify_weights(poses, {b.held, b.releasing} - {None})
        load = {k for k, w in where.items() if w == 'load'}
        if load - self.on_load:
            self.placements += len(load - self.on_load)
        self.on_load = load
        if PROBE:
            self._probe_log(tick, poses, load)
        if self.state == 'idle':
            # the loading starts only in the episode (not while the bench settles before it: a probe's preloaded
            # weights touch the platform from the first tick)
            if (getattr(b, 'bench', None) is not None and touching is not None
                    and touching[rows['platform'], [cols[k] for k in cols if is_weight(k)]].any()):
                self._start(tick)
            else:
                if PROBE:
                    self._trace(tick, z, vz, False, len(load))
                return
        # touches on the bridge (robot links, anything held, weights) from the loading's start
        if touching is not None:
            for body in self.frozen['structure']:
                hit = touching[rows[body]]
                for name, c in cols.items():
                    if hit[c] and (body, name) not in self._touched:
                        self._touched.add((body, name))
                        self.touches.append({'tick': tick, 'body': body, 'by': name})
        if self.pull >= self.frozen['bridge']['capacity_N']:
            self._fail(tick)
            return
        self.peak_n = max(self.peak_n, self.pull)
        # settled: the platform still and nothing of the robot (or what it holds) on the platform or a weight on it
        robot = [cols[n] for n in ROBOT_LINKS if n in cols]
        holding = [cols[k] for k in ({b.held, b.releasing} - {None}) if k in cols]
        load_rows = [rows['platform']] + [rows[k] for k in load]
        touched = touching is not None and bool(touching[np.ix_(load_rows, robot + holding)].any())
        self._trace(tick, z, vz, touched, len(load))
        if abs(vz) < STILL_M_S and not touched:
            self.quiet += 1
        else:
            self.quiet = 0
        if self.quiet >= round(SETTLE_S / b.dt):
            reading = round(self.pull, 2)
            if not self.readings or abs(self.readings[-1][1] - reading) >= 0.05:
                self.readings.append((tick, reading))
            self.held_n = max(self.held_n, reading)

    def _trace(self, tick, z, vz, touched, n_load):
        """Private trace of the loading (the audit log): every TRACE_TICKS ticks, and whenever the pull moved
        TRACE_STEP_N since the last row."""
        last = getattr(self, '_last_trace', None)
        if last is not None and tick - last[0] < TRACE_TICKS and abs(self.pull - last[1]) < TRACE_STEP_N:
            return
        self._last_trace = (tick, self.pull)
        audit = getattr(self.b, 'audit_file', None)
        if audit is not None:
            audit.write(json.dumps({'event': 'load_trace', 'tick': int(tick), 'pull_N': round(self.pull, 3),
                                    'z_mm': round(z * 1000, 4), 'vz_mm_s': round(vz * 1000, 3),
                                    'robot_on_load': bool(touched), 'on_load': n_load, 'held_N': round(self.held_n, 3),
                                    'quiet': self.quiet, 'held': self.b.held, 'tilt_urad': self._tilt()}) + '\n')

    def _probe_log(self, tick, poses, load):
        """Frame probe only: every second, the platform weights' positions (um, table frame) and tilts (urad)."""
        audit = getattr(self.b, 'audit_file', None)
        if audit is None or tick % 240:
            return
        rows = {}
        for key in sorted(load):
            p, q = poses[key]
            w, x, y, z = (float(v) for v in q)
            rows[key] = [round(float(v) * 1e6) for v in p] + [round(2e6 * (x * z + w * y)), round(2e6 * (y * z - w * x))]
        audit.write(json.dumps({'event': 'probe_weights', 'tick': int(tick), 'pull_N': round(self.pull, 3),
                                'tilt_urad': self._tilt(), 'w': rows}) + '\n')

    def _tilt(self):
        """The carriage's tilt (private trace): its z axis's x and y components in micro-radians."""
        q = getattr(self, 'quat', None)
        if q is None:
            return None
        w, x, y, z = (float(v) for v in q)
        return [round(2e6 * (x * z + w * y), 1), round(2e6 * (y * z - w * x), 1)]

    def _start(self, tick):
        """The first weight touches the platform: the loading starts; the bridge is frozen as it stands."""
        b = self.b
        audit = b.seat_audit()
        where = {k: (w[0], tuple(w[1]) if w[0] == 'bridge' else w[1]) for k, w in audit['where'].items()}
        bridge = B.bridge_capacity(where, b.bench.bars, b.bench.failed)
        if PROBE:
            bridge = dict(bridge, capacity_N=1.0e9, probe=True)
        nodes = {n for _, pins in bridge['members'] for n in pins}
        pins = [p for p, w in audit['pins'].items() if w[0] == 'hole' and w[1] in nodes]
        structure = sorted({body for bar, _ in bridge['members'] for body in (bar, B.tail_id(bar))} | set(pins))
        self.frozen = {'audit': audit, 'bridge': bridge, 'structure': [s for s in structure if s in b.contact_rows]}
        self.start_tick = tick
        self.state = 'loading'
        # the bridge held the unloaded platform if it can carry it (else it fails at once, holding nothing)
        if bridge['capacity_N'] > HANGER_KG * G:
            self.held_n = HANGER_KG * G
        self.quiet = 0

    def _fail(self, tick):
        self.state = 'failed'
        self.fail_tick = tick
        self.fail_pull = self.pull

    @property
    def started(self):
        return self.state != 'idle'

    @property
    def failed(self):
        return self.state == 'failed'

    def record(self):
        if self.frozen is None:
            return {'started': False, 'held_N': 0.0}
        bridge = self.frozen['bridge']
        return {'started': True, 'start_tick': self.start_tick, 'failed': self.failed, 'fail_tick': self.fail_tick,
                'fail_pull_N': round(getattr(self, 'fail_pull', 0.0), 3) if self.failed else None,
                'held_N': round(self.held_n, 3), 'peak_N': round(self.peak_n, 3),
                'capacity_N': bridge['capacity_N'], 'governing': bridge['governing'], 'mass_g': bridge['mass_g'],
                'members': bridge['members'], 'structure': self.frozen['structure'], 'touches': list(self.touches),
                'readings': [list(r) for r in self.readings], 'placements': self.placements,
                'trace_tail': [list(t) for t in self.trace][-120:],
                'rope': {'k_N_m': ROPE_K, 'c_N_s_m': round(ROPE_C, 3), 'hanger_kg': HANGER_KG},
                'probe': bool(bridge.get('probe'))}

    # drawing (render only) --------------------------------------------------------------------------------------
    def draw(self, stage, t_s):
        drop = self.drop()
        (p, q) = rope_points(drop)[2]
        api = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(f'{FRAME}/Rope2'))
        api.SetTranslate(Gf.Vec3d(*((np.asarray(p) + np.asarray(q)) / 2)))
        api.SetScale(Gf.Vec3f(0.0016, max(abs(p[2] - q[2]), 1e-4), 0.0016))
        tick = int(round(t_s * DISPLAY_HZ))
        if self.shown is not None and self.shown[0] == tick:
            return
        import truss_apparatus as A                 # (imports this module)
        text = B.force_text(max(0.0, self.pull))
        self.shown = (tick, A.show_load(stage, text, None if self.shown is None else self.shown[1]))


def contact_bodies(paths, pin_ids):
    """Rows and columns of the backend's contact view: rows the platform, every weight, every bar body and loose pin
    (the bridge's bodies are among them); columns every robot link and every weight. Returns (row paths, rows,
    column paths, cols) with rows/cols {name: index}."""
    import hooke_apparatus as lab
    row_names = ['platform'] + sorted(k for k in paths if is_weight(k)) + sorted(k for k in paths if not is_weight(k))
    row_paths = [CARRIAGE] + [paths[k] for k in row_names[1:]]
    col_names = list(ROBOT_LINKS) + sorted(k for k in paths if is_weight(k))
    col_paths = [f'{lab.ROBOT_PATH}/{n}' for n in ROBOT_LINKS] + [paths[k] for k in col_names[len(ROBOT_LINKS):]]
    return row_paths, {n: i for i, n in enumerate(row_names)}, col_paths, {n: i for i, n in enumerate(col_names)}

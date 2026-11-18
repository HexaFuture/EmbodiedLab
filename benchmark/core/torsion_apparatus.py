"""Torsion-pendulum bench (kit mechanics, recipe torsion_bench; torsion_inertia).

Built on the shared lab stage (robot, table, cameras): the Hooke apparatus is
deactivated and a torsion turntable is placed on the table where the Hooke tray
hung. Pure USD construction; no experiment policy, perception or planning.

Turntable: a platter on a vertical revolute joint to the world; an angular force
drive is the torsion spring (USD drives are per degree). A painted circle marks
the centre of the flat platter; a yellow push post and a
black flag sit on the rim; a fixed photogate fork straddles the flag at the rest
angle. Objects: a standard solid disc and an unknown object with the same outer
shape and a hidden internal mass distribution; both carry the weights' handle.

Bench controls (shared with torsion_centroid): the photogate's recorder panel (the shared keypad and
display, coupled_oscillator.controls_asset.build_recorder_panel) on the robot's left front, and the eddy-current
damper lever (damper_lever.build_lever) beside the turntable on the robot's right; the damper brakes the turntable
in proportion to the magnet's overlap with the copper vane (torsion_backend). torsion_inertia only: the two objects
start in a storage box with two numbered seats (the Hooke weight-box style) in front of the robot.
All of it sits under ROOT and moves with the layout jitter.
"""
import math

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

ROOT = '/World/TorsionApparatus'
TURNTABLE = ROOT + '/Turntable'
JOINT = '/World/TorsionPhysics/TorsionJoint'
TABLE_TOP = '/World/LabTable/Top'
HOOKE_ROOT = '/World/HookeApparatus'

# Apparatus origin on the table top, under the former Hooke tray centre (z from the repaired table bound).
ORIGIN_XY = (0.6695, -0.17)

BASE_R, BASE_H = 0.075, 0.012
COLUMN_R, COLUMN_H = 0.024, 0.090
PLATTER_R, PLATTER_H = 0.260, 0.020
PLATTER_Z = BASE_H + COLUMN_H + 0.002 + PLATTER_H / 2      # platter centre above the table top
PLATTER_TOP_Z = PLATTER_Z + PLATTER_H / 2

FOOT_CONTACT = (2.0e4, 150.0)     # object feet: compliant contact stiffness N/m, damping N s/m
OBJECT_R, OBJECT_H = 0.040, 0.020          # 80 mm diameter
# Flat turntable top (no centring pocket). A painted magenta circle around the axis (radius about half an
# object's) marks the centre; placing an object on the axis is the Agent's task (audit: eccentricity).
MARK_R, MARK_W = 0.020, 0.004

# Handle identical to the Hooke weights: post r 8 mm x 50 mm, knob r 16 mm x 5 mm.
POST_R, POST_H, KNOB_R, KNOB_H = 0.008, 0.050, 0.016, 0.005
HANDLE_TOP_LOCAL_Z = OBJECT_H + POST_H + KNOB_H
LATCH_LOCAL = (0.0, 0.0, HANDLE_TOP_LOCAL_Z + 0.020)          # weights: latch 20 mm above the knob top

# Rim fixtures, angles in the apparatus frame (x toward +world x before jitter).
PUSH_POST_ANGLE_DEG, PUSH_POST_RADIUS, PUSH_POST_R, PUSH_POST_H = 62.0, 0.220, 0.020, 0.100
FLAG_ANGLE_DEG = -28.0
FLAG_R0, FLAG_R1, FLAG_T, FLAG_H = 0.260, 0.280, 0.002, 0.012
GATE_BEAM_RADIUS = 0.272
GATE_BEAM_HALF_WIDTH_M = 0.0005
FLAG_HALF_ANGLE_RAD = (FLAG_T / 2 + GATE_BEAM_HALF_WIDTH_M) / GATE_BEAM_RADIUS

OBJECTS = {
    # id: (table position in world before jitter, colour, public mass kg)
    'standard_disc': ((0.98, 0.06), (0.16, 0.30, 0.62), 0.40),
    'unknown_object': ((1.17, 0.06), (0.08, 0.38, 0.16), 0.35),
}
STANDARD_INERTIA_KGM2 = 0.5 * 0.40 * OBJECT_R ** 2

# --- bench controls (world x, y before layout jitter; heights above the table top) -----------------------------------
# Recorder panel of the photogate: keypad console 160 x 160 mm on the robot's left front, clear of the storage box
# (torsion_inertia) and of the torsion_centroid sample's start spot (x <= 1.25, y >= 0.03); the display stands on a
# post behind it, its face towards the robot and the fixed cameras.
PANEL_PATH = ROOT + '/TimerPanel'
PANEL_CENTRE_XY = (1.32, -0.09)
PANEL_TOP_H = 0.080                     # console top above the table (keys' tops 20 mm higher)
DISPLAY_XY = (1.32, -0.255)
DISPLAY_CENTRE_H = 0.200
DISPLAY_STAND = (0.030, 0.016)          # stand post section (x, y) under the display housing
RECORD_S_RANGE = (10, 120)              # record lengths REC accepts (s): >= 5 periods of the slowest load (~3.7 s)
# Eddy-current damper lever beside the turntable on the robot's right: OFF towards the robot, ON 40 mm towards the
# turntable (-Y). The pedestal (80 x 110 mm) stands 60 mm high; the post top is 133 mm above the table.
LEVER_PATH = ROOT + '/DamperLever'
# Light detent (1 N mid-way), as on coupled_modes: this bench's fingers press with at most 3 N each, so a firm parallel grip on
# the post must carry the carriage across by friction or by a finger's push.
LEVER = dict(name='Damper', detent0=(0.47, 0.15), direction=(0.0, -1.0), spacing=0.04, labels=('OFF', 'ON'),
             stiffness=50.0, damping=5.0)
LEVER_TOP_H = 0.060
# Storage box: seats in a row along X in front of the robot, seat 1 (standard disc) at the disc's former table spot.
# {object: (x, y, well half-length along X, well half-width along Y)} in the world before jitter. torsion_inertia:
# two 92 mm square wells for the 80 mm objects (a seated object stands within 6 mm of the centre), 150 mm apart.
# torsion_centroid (centroid_apparatus.BOX_SEATS): the disc's square well and a 168 x 68 mm well for the
# 160 x 60 mm sample. Floor 8 mm above the table, mouths chamfered 8 mm wider, box 22 mm high: a body stands 6 mm
# proud and the fingers on its handle stay well above the box. Numbers 1 and 2 on the strip on the robot's side.
BOX_PATH = ROOT + '/StorageBox'
BOX_SEATS = {'standard_disc': (0.98, 0.06, 0.046, 0.046), 'unknown_object': (1.13, 0.06, 0.046, 0.046)}
BOX_HEIGHT, BOX_FLOOR = 0.022, 0.008
SEAT_CHAMFER = 0.008
BOX_MARGIN = {'end': 0.012, 'back': 0.012, 'front': 0.041}   # box wall beyond the mouths (front: label strip)
BOX_COLOUR = (0.16, 0.25, 0.22)
BOX_INK = (0.94, 0.94, 0.91)
# In-seat rule (as weights_in_box): within 20 mm horizontally and vertically of an object standing on the seat floor,
# tilted at most 10 degrees, not held.
SEAT_DXY_M, SEAT_DZ_M, SEAT_TILT_DEG = 0.02, 0.02, 10.0

# Locating pin (torsion_centroid only): on the turntable axis, 10 mm in diameter, 12 mm above the platter top; its top
# 4 mm are chamfered to a 3 mm flat, so a hole up to 3.5 mm off the axis still finds the pin when an object is set
# down. Polished steel: a low friction lets a hole slide down the chamfer.
PIN_R, PIN_H, PIN_TIP_R, PIN_TIP_H = 0.005, 0.012, 0.0015, 0.004
PIN_COLOUR = (0.62, 0.64, 0.68)
PIN_FRICTION = 0.10

DEG = math.pi / 180


def _material(stage, path, static, dynamic, compliant=None):
    """compliant: (stiffness N/m, damping N s/m) of a soft contact layer, else rigid contact."""
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    if compliant is not None:
        physx = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        physx.CreateCompliantContactStiffnessAttr(float(compliant[0]))
        physx.CreateCompliantContactDampingAttr(float(compliant[1]))
    return mat


def _collider(prim, material=None):
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')


def _smooth_cylinder_mesh(stage, path, radius, height, colour, segments=64):
    """Render mesh for a cylinder (Hydra draws UsdGeom.Cylinder with few facets)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    points, counts, indices = [], [], []
    for z in (-height / 2, height / 2):
        points += [Gf.Vec3f(radius * math.cos(2 * math.pi * i / segments), radius * math.sin(2 * math.pi * i / segments), z)
                   for i in range(segments)]
    points += [Gf.Vec3f(0, 0, -height / 2), Gf.Vec3f(0, 0, height / 2)]
    bottom, top = 2 * segments, 2 * segments + 1
    for i in range(segments):
        j = (i + 1) % segments
        counts += [4, 3, 3]
        indices += [i, j, segments + j, segments + i, bottom, j, i, top, segments + i, segments + j]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2), Gf.Vec3f(radius, radius, height / 2)])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    return mesh


def _annulus_mesh(stage, path, inner, outer, z, colour, segments=128):
    """Flat painted ring (render only)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    points, counts, indices = [], [], []
    for r in (inner, outer):
        points += [Gf.Vec3f(r * math.cos(2 * math.pi * i / segments), r * math.sin(2 * math.pi * i / segments), z)
                   for i in range(segments)]
    for i in range(segments):
        j = (i + 1) % segments
        counts.append(4)
        indices += [i, segments + i, segments + j, j]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr([Gf.Vec3f(-outer, -outer, z), Gf.Vec3f(outer, outer, z)])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    return mesh


def _cylinder(stage, path, radius, height, centre, colour, material=None, collide=True):
    """A cylinder: exact collider (invisible) plus a smooth render mesh under one transform."""
    xform = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(*centre))
    _smooth_cylinder_mesh(stage, path + '/Visual', radius, height, colour)
    if collide:
        geom = UsdGeom.Cylinder.Define(stage, path + '/Collider')
        geom.CreateRadiusAttr(radius)
        geom.CreateHeightAttr(height)
        geom.CreateAxisAttr('Z')
        geom.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2), Gf.Vec3f(radius, radius, height / 2)])
        geom.CreateVisibilityAttr('invisible')   # bounds stay visible to BBoxCache (default purpose)
        _collider(geom.GetPrim(), material)
    return xform


def _box(stage, path, size, centre, colour, rotate_xyz=(0, 0, 0), material=None, collide=True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([Gf.Vec3f(-0.5), Gf.Vec3f(0.5)])
    api = UsdGeom.XformCommonAPI(cube)
    api.SetTranslate(Gf.Vec3d(*centre))
    api.SetRotate(Gf.Vec3f(*rotate_xyz), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    api.SetScale(Gf.Vec3f(*size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    if collide:
        _collider(cube.GetPrim(), material)
    return cube


def _polar(radius, angle_deg, z):
    a = math.radians(angle_deg)
    return (radius * math.cos(a), radius * math.sin(a), z)


def _rigid(prim, mass, inertia_diag, com):
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    body.CreateSolverPositionIterationCountAttr(16)
    body.CreateSolverVelocityIterationCountAttr(8)
    body.CreateSleepThresholdAttr(0.0)
    body.CreateLinearDampingAttr(0.0)
    body.CreateAngularDampingAttr(0.0)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia_diag]))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))


def retire_hooke_apparatus(stage):
    """The shared stage carries the Hooke bench; torsion_inertia uses the table, robot and cameras only."""
    root = stage.GetPrimAtPath(HOOKE_ROOT)
    if not root.IsValid():
        raise RuntimeError('shared lab stage lacks the Hooke apparatus root')
    for child in root.GetChildren():
        if child.GetName() not in ('_materials', 'env_light'):
            child.SetActive(False)
    physics = stage.GetPrimAtPath('/World/HookePhysics')
    for child in physics.GetChildren() if physics.IsValid() else ():
        child.SetActive(False)      # the Hooke spring joint; the grasp constraint is authored here at run time


def repair_cube_extents(stage):
    """Flattened cubes carry a baked, scaled extent that BBoxCache scales again (as hooke_physics does).

    The table top is one: its stale bound puts the surface at 0.783 m, the collider at 0.820 m.
    """
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Cube):
            cube = UsdGeom.Cube(prim)
            half = float(cube.GetSizeAttr().Get()) / 2
            cube.CreateExtentAttr([Gf.Vec3f(-half), Gf.Vec3f(half)])


def table_top_z(stage):
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render']).ComputeWorldBound(
        stage.GetPrimAtPath(TABLE_TOP)).ComputeAlignedRange()
    return float(box.GetMax()[2])


def seat_path(index):
    """Marker Xform at the centre of seat `index` (0-based)'s floor (follows the layout jitter)."""
    return f'{BOX_PATH}/Seat{index + 1}'


def build_storage_box(stage, origin, seats=None):
    """Static box with one seat per object (one triangle-mesh collider, seat markers, printed numbers) in the
    apparatus frame; seats {object: (x, y, well half-length along X, half-width along Y)} in the world before jitter
    (BOX_SEATS), in one row along X."""
    import hooke_apparatus as hooke
    seats = BOX_SEATS if seats is None else seats
    centres = [(x - origin[0], y - origin[1]) for x, y, _, _ in seats.values()]
    halves = [(hx, hy) for _, _, hx, hy in seats.values()]
    cy = centres[0][1]
    if any(abs(sy - cy) > 1e-9 for _, sy in centres):
        raise ValueError('storage box seats must stand in one row along X')
    m = BOX_MARGIN
    mouth = max(hy for _, hy in halves) + SEAT_CHAMFER
    x0 = centres[0][0] - halves[0][0] - SEAT_CHAMFER - m['end']
    x1 = centres[-1][0] + halves[-1][0] + SEAT_CHAMFER + m['end']
    y0, y1 = cy - mouth - m['back'], cy + mouth + m['front']
    top, floor = BOX_HEIGHT, BOX_FLOOR
    throat = top - SEAT_CHAMFER
    up = np.array([0.0, 0.0, 1.0])
    quads = []

    def quad(points, hint):
        quads.append(hooke._oriented(points, np.asarray(hint, dtype=np.float64)))

    quad([(x0, y0, top), (x1, y0, top), (x1, cy - mouth, top), (x0, cy - mouth, top)], up)
    quad([(x0, cy + mouth, top), (x1, cy + mouth, top), (x1, y1, top), (x0, y1, top)], up)
    edges = [x0] + [v for (cx, _), (hx, _) in zip(centres, halves) for v in (cx - hx - SEAT_CHAMFER, cx + hx + SEAT_CHAMFER)] + [x1]
    for a, b in zip(edges[0::2], edges[1::2]):          # top between the mouths, full width
        quad([(a, cy - mouth, top), (b, cy - mouth, top), (b, cy + mouth, top), (a, cy + mouth, top)], up)
    for (cx, _), (hx, hy) in zip(centres, halves):      # top beside a mouth narrower than the widest
        a, b, inner = cx - hx - SEAT_CHAMFER, cx + hx + SEAT_CHAMFER, hy + SEAT_CHAMFER
        if inner < mouth - 1e-9:
            quad([(a, cy - mouth, top), (b, cy - mouth, top), (b, cy - inner, top), (a, cy - inner, top)], up)
            quad([(a, cy + inner, top), (b, cy + inner, top), (b, cy + mouth, top), (a, cy + mouth, top)], up)
    quad([(x0, y0, 0), (x1, y0, 0), (x1, y0, top), (x0, y0, top)], (0, -1, 0))
    quad([(x0, y1, 0), (x1, y1, 0), (x1, y1, top), (x0, y1, top)], (0, 1, 0))
    quad([(x0, y0, 0), (x0, y1, 0), (x0, y1, top), (x0, y0, top)], (-1, 0, 0))
    quad([(x1, y0, 0), (x1, y1, 0), (x1, y1, top), (x1, y0, top)], (1, 0, 0))
    quad([(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)], -up)
    for (cx, sy), (hx, hy) in zip(centres, halves):
        def ring(half_x, half_y, z):
            return [(cx - half_x, sy - half_y, z), (cx + half_x, sy - half_y, z), (cx + half_x, sy + half_y, z),
                    (cx - half_x, sy + half_y, z)]
        well = (hx, hy)
        rings = [ring(hx + SEAT_CHAMFER, hy + SEAT_CHAMFER, top), ring(*well, throat), ring(*well, floor)]
        for upper, lower in zip(rings, rings[1:]):
            for i in range(4):
                j = (i + 1) % 4
                points = [upper[i], upper[j], lower[j], lower[i]]
                middle = np.mean(np.asarray(points, dtype=np.float64), axis=0)
                quad(points, (cx - middle[0], sy - middle[1], 0.01))
        quad(ring(*well, floor), up)
    UsdGeom.Xform.Define(stage, BOX_PATH)
    body = hooke._quads_mesh(stage, BOX_PATH + '/Body', quads, BOX_COLOUR)
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(body.GetPrim()).CreateApproximationAttr('none')
    contact = PhysxSchema.PhysxCollisionAPI.Apply(body.GetPrim())
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    material = _material(stage, '/World/PhysicsMaterials/TorsionStorageBox', 0.65, 0.50)
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(material, materialPurpose='physics')
    # Seat numbers on the strip on the robot's side, upright from the robot (glyph up -Y, reading towards -X).
    label_y = cy + mouth + m['front'] / 2
    for index, (cx, sy) in enumerate(centres):
        glyph = []
        for u0, v0, u1, v1 in hooke._digit_rectangles(str(index + 1)):
            corners = [(cx - u, label_y - v, top + 0.0004) for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
            glyph.append(hooke._oriented(corners, up))
        hooke._quads_mesh(stage, f'{BOX_PATH}/Label{index + 1}', glyph, BOX_INK)
        seat = UsdGeom.Xform.Define(stage, seat_path(index))
        UsdGeom.XformCommonAPI(seat).SetTranslate(Gf.Vec3d(cx, sy, floor))
    return BOX_PATH + '/Body'


def build_pin(stage):
    """Locating pin on the turntable axis (part of the turntable body): a steel cylinder PIN_R standing PIN_H on the
    platter top, its top chamfered down to PIN_TIP_R over PIN_TIP_H. One convex collider (the shape is convex)."""
    path = TURNTABLE + '/LocatingPin'
    mesh = UsdGeom.Mesh.Define(stage, path)
    segments, z0 = 48, PLATTER_H / 2
    rings = ((PIN_R, z0), (PIN_R, z0 + PIN_H - PIN_TIP_H), (PIN_TIP_R, z0 + PIN_H))
    points = [Gf.Vec3f(r * math.cos(2 * math.pi * i / segments), r * math.sin(2 * math.pi * i / segments), z)
              for r, z in rings for i in range(segments)]
    points += [Gf.Vec3f(0, 0, z0), Gf.Vec3f(0, 0, z0 + PIN_H)]
    counts, indices = [], []
    for ring in range(len(rings) - 1):
        for i in range(segments):
            j = (i + 1) % segments
            counts.append(4)
            indices += [ring * segments + i, ring * segments + j, (ring + 1) * segments + j, (ring + 1) * segments + i]
    bottom, top = len(points) - 2, len(points) - 1
    last = (len(rings) - 1) * segments
    for i in range(segments):
        j = (i + 1) % segments
        counts += [3, 3]
        indices += [bottom, j, i, top, last + i, last + j]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateExtentAttr([Gf.Vec3f(-PIN_R, -PIN_R, z0), Gf.Vec3f(PIN_R, PIN_R, z0 + PIN_H)])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*PIN_COLOUR)])
    _collider(mesh.GetPrim(), _material(stage, '/World/PhysicsMaterials/TorsionPinSteel', PIN_FRICTION, PIN_FRICTION))
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
    return path


def build_controls(stage, origin):
    """Recorder panel (keypad and display, the shared builder) and damper lever, in the apparatus frame.

    Returns the lever info of damper_lever (its detent0 in the apparatus frame; the runtime measures the carriage)."""
    from coupled_oscillator.controls_asset import build_recorder_panel
    from damper_lever import build_lever
    from pendulum_asset import _box as panel_box, _material as visual_material
    contact = _material(stage, '/World/PhysicsMaterials/TorsionControls', 0.8, 0.7)
    local = lambda xy: (xy[0] - origin[0], xy[1] - origin[1])
    dx, dy = local(DISPLAY_XY)
    build_recorder_panel(stage, PANEL_PATH, centre_xy=local(PANEL_CENTRE_XY), top_z=PANEL_TOP_H, base_z=0.0,
                         contact_material=contact, display_centre=(dx, dy, DISPLAY_CENTRE_H),
                         spec_text=f'{RECORD_S_RANGE[0]}-{RECORD_S_RANGE[1]} S')
    from coupled_oscillator.controls_asset import DISPLAY_HOUSING
    stand_top = DISPLAY_CENTRE_H - DISPLAY_HOUSING[2] / 2
    stand = panel_box(stage, PANEL_PATH + '/DisplayStand', (dx, dy, stand_top / 2), (*DISPLAY_STAND, stand_top),
                      visual_material(stage, PANEL_PATH + '/Materials/Stand', (.10, .11, .13), .3))
    _collider(stand.GetPrim(), contact)
    panel_box(stage, PANEL_PATH + '/DisplayFoot', (dx, dy, 0.003), (0.10, 0.06, 0.006),
              visual_material(stage, PANEL_PATH + '/Materials/Stand', (.10, .11, .13), .3))
    info = build_lever(stage, LEVER_PATH, contact_material=contact, detent0=local(LEVER['detent0']),
                       top_z=LEVER_TOP_H, base_z=0.0, **{k: v for k, v in LEVER.items() if k != 'detent0'})
    # The rail joint's body0 is the world: its frame is a world pose, not an apparatus-frame one.
    joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(info['joint']))
    pos = joint.GetLocalPos0Attr().Get()
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(pos[0] + origin[0], pos[1] + origin[1], pos[2] + origin[2]))
    return info


def build(stage, *, turntable_inertia_kgm2, unknown_inertia_kgm2, mark_rgb=(0.03, 0.03, 0.03), storage_box=None,
          pin=False):
    """Author the bench in its unjittered pose. Physical joint is authored later (author_joint).

    storage_box: the seats {object: (x, y, well half-length, half-width)} of the storage box the objects start in
    (BOX_SEATS for torsion_inertia), or None: the objects start on the table at OBJECTS.
    pin (torsion_centroid): a vertical locating pin with a chamfered tip on the turntable axis (build_pin)."""
    if not (0 < turntable_inertia_kgm2 < 0.05 and 0 < unknown_inertia_kgm2 < 0.05):
        raise ValueError('inertias must be positive and small')
    retire_hooke_apparatus(stage)
    repair_cube_extents(stage)
    origin = (*ORIGIN_XY, table_top_z(stage))
    rubber = _material(stage, '/World/PhysicsMaterials/TorsionRubber', 0.8, 0.7)
    # Rubber feet under the objects: a soft contact layer (about 0.2 mm sag under load), so a
    # position-controlled arm setting an object down on a rigid surface does not spike the grasp force.
    foot = _material(stage, '/World/PhysicsMaterials/TorsionFoot', 0.8, 0.7, compliant=FOOT_CONTACT)
    root = UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(*origin))
    # Fixed frame: base, column (housing the torsion spring), photogate fork.
    _cylinder(stage, ROOT + '/Base', BASE_R, BASE_H, (0, 0, BASE_H / 2), (0.18, 0.19, 0.21), rubber)
    _cylinder(stage, ROOT + '/Column', COLUMN_R, COLUMN_H, (0, 0, BASE_H + COLUMN_H / 2), (0.30, 0.31, 0.33), rubber)
    for i in range(6):   # visible coil rings of the torsion spring inside the column window
        _cylinder(stage, ROOT + f'/SpringCoil{i}', COLUMN_R + 0.002, 0.003,
                  (0, 0, BASE_H + 0.02 + i * 0.011), (0.72, 0.55, 0.20), collide=False)
    gate_a = FLAG_ANGLE_DEG
    fork_z0 = PLATTER_Z - FLAG_H / 2 - 0.012
    fork_z1 = PLATTER_Z + FLAG_H / 2 + 0.006
    tangent = (-math.sin(math.radians(gate_a)), math.cos(math.radians(gate_a)))
    for name, z, h in (('GateLowerArm', fork_z0 - 0.004, 0.008), ('GateUpperArm', fork_z1 + 0.004, 0.008)):
        _box(stage, ROOT + '/' + name, (0.024, 0.016, h), _polar(GATE_BEAM_RADIUS, gate_a, z),
             (0.70, 0.08, 0.06), rotate_xyz=(0, 0, gate_a), material=rubber)
    spine_r = GATE_BEAM_RADIUS + 0.016
    _box(stage, ROOT + '/GateSpine', (0.008, 0.016, fork_z1 + 0.008 - 0.0), _polar(spine_r, gate_a, (fork_z1 + 0.008) / 2),
         (0.70, 0.08, 0.06), rotate_xyz=(0, 0, gate_a), material=rubber)
    _cylinder(stage, ROOT + '/GateLed', 0.002, 0.002, _polar(GATE_BEAM_RADIUS, gate_a, fork_z1 + 0.0085),
              (0.1, 0.9, 0.2), collide=False)
    del tangent
    # Turntable: one dynamic body; explicit mass properties (hidden I0) override its colliders.
    table = UsdGeom.Xform.Define(stage, TURNTABLE)
    UsdGeom.XformCommonAPI(table).SetTranslate(Gf.Vec3d(0, 0, PLATTER_Z))
    _cylinder(stage, TURNTABLE + '/Platter', PLATTER_R, PLATTER_H, (0, 0, 0), (0.22, 0.23, 0.25), rubber)
    _annulus_mesh(stage, TURNTABLE + '/CentreMark', MARK_R - MARK_W / 2, MARK_R + MARK_W / 2,
                  PLATTER_H / 2 + 0.0003, tuple(float(c) for c in mark_rgb))
    _cylinder(stage, TURNTABLE + '/PushPost', PUSH_POST_R, PUSH_POST_H,
              _polar(PUSH_POST_RADIUS, PUSH_POST_ANGLE_DEG, PLATTER_H / 2 + PUSH_POST_H / 2), (0.95, 0.78, 0.05), rubber)
    if pin:
        build_pin(stage)
    flag_mid = (FLAG_R0 + FLAG_R1) / 2
    _box(stage, TURNTABLE + '/Flag', (FLAG_R1 - FLAG_R0, FLAG_T, FLAG_H), _polar(flag_mid, FLAG_ANGLE_DEG, 0),
         (0.04, 0.04, 0.04), rotate_xyz=(0, 0, FLAG_ANGLE_DEG), collide=False)
    side = turntable_inertia_kgm2 * 0.55
    _rigid(table.GetPrim(), 0.60, (side, side, turntable_inertia_kgm2), (0, 0, 0))
    # Objects on the table: same outer shape and handle; internal distribution sets Izz.
    inertias = {'standard_disc': STANDARD_INERTIA_KGM2, 'unknown_object': float(unknown_inertia_kgm2)}
    paths = {}
    build_controls(stage, origin)
    if storage_box:
        build_storage_box(stage, origin, storage_box)
    for key, ((x, y), colour, mass) in OBJECTS.items():
        path = ROOT + '/' + ''.join(part.title() for part in key.split('_'))
        obj = UsdGeom.Xform.Define(stage, path)
        z = 0.0005
        if storage_box:
            x, y = storage_box[key][:2]
            z = BOX_FLOOR + 0.0005
        UsdGeom.XformCommonAPI(obj).SetTranslate(Gf.Vec3d(x - origin[0], y - origin[1], z))
        _cylinder(stage, path + '/Body', OBJECT_R, OBJECT_H, (0, 0, OBJECT_H / 2), colour, foot)
        _cylinder(stage, path + '/HandlePost', POST_R, POST_H, (0, 0, OBJECT_H + POST_H / 2), (0.62, 0.48, 0.18), rubber)
        _cylinder(stage, path + '/HandleKnob', KNOB_R, KNOB_H, (0, 0, OBJECT_H + POST_H + KNOB_H / 2), (0.62, 0.48, 0.18), rubber)
        side = mass * (3 * OBJECT_R ** 2 + OBJECT_H ** 2) / 12
        _rigid(obj.GetPrim(), mass, (side, side, inertias[key]), (0, 0, OBJECT_H / 2))
        body = PhysxSchema.PhysxRigidBodyAPI(obj.GetPrim())
        body.CreateEnableCCDAttr(True)
        paths[key] = path
    table_top = stage.GetPrimAtPath(TABLE_TOP)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(table_top)
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(table_top).Bind(rubber, materialPurpose='physics')
    return paths


def author_joint(stage, kappa_nm_per_rad, damping_nms_per_rad):
    """Torsion spring: revolute joint from the world to the turntable at its current (possibly jittered) pose."""
    if not (kappa_nm_per_rad > 0 and damping_nms_per_rad >= 0):
        raise ValueError('torsion constants must be positive')
    world = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(TURNTABLE))
    joint = UsdPhysics.RevoluteJoint.Define(stage, JOINT)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(TURNTABLE)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(world.ExtractTranslation()))
    joint.CreateLocalRot0Attr(Gf.Quatf(world.ExtractRotationQuat()))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateAxisAttr('Z')
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
    drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(float(kappa_nm_per_rad) * DEG)
    drive.CreateDampingAttr(float(damping_nms_per_rad) * DEG)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateMaxForceAttr(1e3)
    return drive


def set_joint_damping(stage, damping_nms_per_rad):
    drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(JOINT), 'angular')
    value = float(damping_nms_per_rad) * DEG
    drive.GetDampingAttr().Set(value)
    if abs(float(drive.GetDampingAttr().Get()) - value) > 1e-9 + 1e-6 * value:
        raise RuntimeError('torsion damping not applied')


def flag_blocked(angle_rad):
    """Beam state from the trusted turntable angle relative to rest (flag centred in the beam at rest)."""
    return abs(angle_rad) <= FLAG_HALF_ANGLE_RAD

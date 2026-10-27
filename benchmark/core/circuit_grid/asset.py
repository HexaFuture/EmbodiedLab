"""USD authoring of the circuit board bench (pure construction; no experiment policy, perception or planning).

Built on the shared lab stage (robot, table, cameras): the Hooke apparatus is deactivated, then the board, the DC
supply, the black box, the component tray with every block in its seat and the logger unit are placed on the table
(geometry in layout.py). The board, tray and housings are static colliders; the blocks are dynamic rigid bodies
(the switch block carries its rocker on a revolute joint); the supply button and the logger keys are spring-returned
caps on prismatic joints. Readouts and lamps are render-only and change by visibility.
"""
from __future__ import annotations

import math

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from . import layout as L
from .display import Frame, Readout, quad_mesh, text, INK, WHITE, LIT

ROOT = '/World/CircuitGrid'
PHYSICS_ROOT = '/World/CircuitGridPhysics'
HOOKE_ROOT = '/World/HookeApparatus'
TABLE_TOP = '/World/LabTable/Top'
SUPPLY_BUTTON = ROOT + '/Supply/ButtonCap'
ROCKER = ROOT + '/Blocks/SW_Rocker'
ROCKER_JOINT = PHYSICS_ROOT + '/RockerJoint'
LAMP_PATHS = {'ON': ROOT + '/Supply/LampOnLit', 'CC': ROOT + '/Supply/LampCcLit',
              'REC': ROOT + '/Logger/RecLampLit', 'ERR': ROOT + '/Logger/ErrLampLit'}

BOARD_COLOUR = (0.08, 0.24, 0.15)
BRASS = (0.80, 0.63, 0.30)
TRAY_COLOUR = (0.16, 0.25, 0.22)
SUPPLY_COLOUR = (0.62, 0.65, 0.69)
BOX_COLOUR = (0.05, 0.05, 0.06)
LOGGER_COLOUR = (0.14, 0.18, 0.26)
WINDOW = (0.012, 0.014, 0.016)
RED = (0.85, 0.08, 0.06)
BLOCK_COLOURS = {'wire': (0.72, 0.72, 0.74), 'resistor': (0.82, 0.70, 0.46), 'switch': (0.16, 0.26, 0.56),
                 'voltmeter': (0.18, 0.22, 0.32), 'ammeter': (0.42, 0.10, 0.10)}
CHANNEL_COLOURS = {'CH1': (0.95, 0.80, 0.10), 'CH2': (0.10, 0.74, 0.86)}
LAMP_ON = {'ON': (0.20, 1.0, 0.30), 'CC': (1.0, 0.60, 0.05), 'REC': (1.0, 0.10, 0.06), 'ERR': (1.0, 0.60, 0.05)}
UP = (-1.0, 0.0)          # printed text on fixed parts reads upright in the wrist view: up = world -x
READ = (0.0, 1.0)         # reading direction world +y


# --- small helpers ---------------------------------------------------------------------------------------------------
def _physics_material(stage, path, static, dynamic):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    return mat


def _render_material(stage, path, colour, emission=0.0):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + '/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.6)
    if emission:
        shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[c * emission for c in colour]))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return mat


_DISPLAY_LOOKS = {}


def display_looks(stage):
    """Matte materials of the instrument displays: a dark window that reflects no lamp light, and bright green lit
    segments (no specular), so the digits stay distinct in the camera images. Nothing on the bench emits light."""
    if stage.GetPrimAtPath(ROOT + '/Looks/DisplayWindow') and _DISPLAY_LOOKS:
        return _DISPLAY_LOOKS
    for name, colour, emission in (('DisplayWindow', WINDOW, None), ('DisplayLit', (0.25, 0.92, 0.30), None)):
        path = ROOT + '/Looks/' + name
        mat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + '/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput('specularColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        if emission:
            shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emission))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        _DISPLAY_LOOKS[name] = mat
    return _DISPLAY_LOOKS


def _collider(prim, material=None, offset=0.001):
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    contact.CreateContactOffsetAttr(offset)
    contact.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, 'physics')


def _box(stage, path, size, centre, colour, material=None, collide=True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([Gf.Vec3f(-0.5), Gf.Vec3f(0.5)])
    api = UsdGeom.XformCommonAPI(cube)
    api.SetTranslate(Gf.Vec3d(*centre))
    api.SetScale(Gf.Vec3f(*size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    if collide:
        _collider(cube.GetPrim(), material)
    return cube


def _disc(stage, path, radius, centre, colour, segments=32, visible=True, material=None):
    """Flat render-only disc (lamps, terminal dots)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    cx, cy, cz = centre
    points = [Gf.Vec3f(cx, cy, cz)] + [Gf.Vec3f(cx + radius * math.cos(2 * math.pi * k / segments),
                                                cy + radius * math.sin(2 * math.pi * k / segments), cz)
                                       for k in range(segments)]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * segments)
    mesh.CreateFaceVertexIndicesAttr([i for k in range(segments) for i in (0, 1 + k, 1 + (k + 1) % segments)])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr([Gf.Vec3f(cx - radius, cy - radius, cz), Gf.Vec3f(cx + radius, cy + radius, cz)])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    if not visible:
        mesh.MakeInvisible()
    return mesh


def _quads(stage, path, quads, colour, collide=False, material=None):
    """A mesh of planar quads (each four (x, y, z) corners, any winding); static triangle-mesh collider if collide."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = [Gf.Vec3f(*map(float, p)) for q in quads for p in q]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * len(quads))
    mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    arr = np.asarray([[p[0], p[1], p[2]] for p in points])
    mesh.CreateExtentAttr([Gf.Vec3f(*arr.min(0).tolist()), Gf.Vec3f(*arr.max(0).tolist())])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
        contact = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        contact.CreateContactOffsetAttr(0.001)
        contact.CreateRestOffsetAttr(0.0)
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, UsdShade.Tokens.weakerThanDescendants,
                                                                   'physics')
    return mesh


def _rigid(prim, mass, iterations=(16, 8)):
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    body.CreateSolverPositionIterationCountAttr(iterations[0])
    body.CreateSolverVelocityIterationCountAttr(iterations[1])
    body.CreateSleepThresholdAttr(0.0)
    body.CreateLinearDampingAttr(0.0)
    body.CreateAngularDampingAttr(0.05)
    body.CreateMaxDepenetrationVelocityAttr(0.5)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))


def retire_hooke_apparatus(stage):
    """The shared stage carries the Hooke bench; this bench uses the table, the robot and the cameras only."""
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
    """Flattened cubes carry a baked, scaled extent that BBoxCache scales again; the table top is one."""
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Cube):
            cube = UsdGeom.Cube(prim)
            half = float(cube.GetSizeAttr().Get()) / 2
            cube.CreateExtentAttr([Gf.Vec3f(-half), Gf.Vec3f(half)])


def table_top_z(stage):
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render']).ComputeWorldBound(
        stage.GetPrimAtPath(TABLE_TOP)).ComputeAlignedRange()
    return float(box.GetMax()[2])


def _grid_cells(x_bounds, y_bounds, holes, z):
    """Quads of the plane z over x_bounds x y_bounds minus axis-aligned rectangular holes [(x0, x1, y0, y1)]:
    [(quad, centre)] on the grid of all hole edges."""
    xs = sorted({x_bounds[0], x_bounds[1], *[v for h in holes for v in h[:2]]})
    ys = sorted({y_bounds[0], y_bounds[1], *[v for h in holes for v in h[2:]]})
    xs = [x for x in xs if x_bounds[0] <= x <= x_bounds[1]]
    ys = [y for y in ys if y_bounds[0] <= y <= y_bounds[1]]
    out = []
    for xa, xb in zip(xs, xs[1:]):
        for ya, yb in zip(ys, ys[1:]):
            cx, cy = (xa + xb) / 2, (ya + yb) / 2
            if any(h[0] < cx < h[1] and h[2] < cy < h[3] for h in holes):
                continue
            out.append((((xa, ya, z), (xb, ya, z), (xb, yb, z), (xa, yb, z)), (cx, cy)))
    return out


def _pocket(cx, cy, rings, floor_holes=()):
    """Wall quads between consecutive rings [(half_x, half_y, z)] of a rectangular pocket centred at (cx, cy)."""
    quads = []

    def ring(hx, hy, z):
        return [(cx - hx, cy - hy, z), (cx + hx, cy - hy, z), (cx + hx, cy + hy, z), (cx - hx, cy + hy, z)]
    for upper, lower in zip(rings, rings[1:]):
        a, b = ring(*upper), ring(*lower)
        for i in range(4):
            j = (i + 1) % 4
            quads.append((a[i], a[j], b[j], b[i]))
    return quads


# --- board ---------------------------------------------------------------------------------------------------------
def build_board(stage, material):
    (x0, x1), (y0, y1) = L.board_extent()
    top, bottom = L.BOARD_TOP_Z, L.TABLE_Z
    m = L.SOCKET_MOUTH_HALF
    holes, pads = [], []
    for sx, sy, _, _ in L.sockets():
        holes.append((sx - m, sx + m, sy - m, sy + m))
    for node in L.all_nodes():
        nx, ny = L.node_xy(node)
        pads.append((nx - L.PAD_HALF, nx + L.PAD_HALF, ny - L.PAD_HALF, ny + L.PAD_HALF))
    # The pad edges split the grid too (no hole there: they only change the colour).
    pad_lines = [(p[0], p[0], p[2], p[2]) for p in pads] + [(p[1], p[1], p[3], p[3]) for p in pads]
    cells = _grid_cells((x0, x1), (y0, y1), holes + pad_lines, top)
    plate, brass = [], []
    for quad, (cx, cy) in cells:
        (brass if any(p[0] < cx < p[1] and p[2] < cy < p[3] for p in pads) else plate).append(quad)
    floor = top - L.SOCKET_DEPTH
    fh = m - L.SOCKET_DEPTH
    for sx, sy, _, _ in L.sockets():
        brass += _pocket(sx, sy, [(m, m, top), (fh, fh, floor)])
        brass.append(((sx - fh, sy - fh, floor), (sx + fh, sy - fh, floor), (sx + fh, sy + fh, floor),
                      (sx - fh, sy + fh, floor)))
    plate += [((x0, y0, bottom), (x1, y0, bottom), (x1, y0, top), (x0, y0, top)),
              ((x0, y1, bottom), (x1, y1, bottom), (x1, y1, top), (x0, y1, top)),
              ((x0, y0, bottom), (x0, y1, bottom), (x0, y1, top), (x0, y0, top)),
              ((x1, y0, bottom), (x1, y1, bottom), (x1, y1, top), (x1, y0, top))]
    UsdGeom.Xform.Define(stage, ROOT + '/Board')
    _quads(stage, ROOT + '/Board/Plate', plate, BOARD_COLOUR, collide=True, material=material)
    _quads(stage, ROOT + '/Board/Brass', brass, BRASS, collide=True, material=material)
    # Node coordinates printed on the margin: column numbers on the far edge, row numbers on the supply side.
    frame = Frame((0, 0), READ, UP, top)
    for c in range(1, L.COLS + 1):
        x, _ = L.node_xy((c, 1))
        text(stage, f'{ROOT}/Board/Col{c}', str(c), Frame((x, y0 + 0.006), READ, UP, top), 0, 0, 0.007, WHITE,
             align='centre')
    for r in range(1, L.ROWS + 1):
        _, y = L.node_xy((1, r))
        text(stage, f'{ROOT}/Board/Row{r}', str(r), Frame((x0 + 0.006, y + 0.014), READ, UP, top), 0, 0, 0.007, WHITE,
             align='centre')
    del frame


# --- supply and black box ---------------------------------------------------------------------------------------------
def _housing(stage, path, box, colour, material):
    (x0, x1), (y0, y1), h = box
    _box(stage, path, (x1 - x0, y1 - y0, h), ((x0 + x1) / 2, (y0 + y1) / 2, L.TABLE_Z + h / 2), colour, material)


def _strap(stage, path, xa, xb, y, colour):
    """A terminal strap from a housing to its board pad (render only, on the board margin)."""
    _box(stage, path, (abs(xb - xa), 0.006, 0.0012), ((xa + xb) / 2, y, L.BOARD_TOP_Z + 0.0006), colour,
         collide=False)


def spring_cap(stage, cap_path, world_centre, size, colour, material, mass=L.CAP_MASS_KG):
    """A key cap on a spring-returned vertical prismatic joint to a kinematic anchor at its rest pose."""
    cap = UsdGeom.Xform.Define(stage, cap_path)
    cap.AddTranslateOp().Set(Gf.Vec3d(*world_centre))
    _rigid(cap.GetPrim(), mass)
    _box(stage, cap_path + '/Cap', size, (0, 0, 0), colour, material)
    anchor_path = cap_path + 'Anchor'
    anchor = UsdGeom.Xform.Define(stage, anchor_path)
    anchor.AddTranslateOp().Set(Gf.Vec3d(*world_centre))
    UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim()).CreateKinematicEnabledAttr().Set(True)
    mass_api = UsdPhysics.MassAPI.Apply(anchor.GetPrim())
    mass_api.CreateMassAttr().Set(1.0)
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1.0))
    joint = UsdPhysics.PrismaticJoint.Define(stage, cap_path + 'Joint')
    joint.CreateBody0Rel().SetTargets([Sdf.Path(anchor_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(cap_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateAxisAttr().Set('Z')
    joint.CreateLowerLimitAttr().Set(-L.TRAVEL_M)
    joint.CreateUpperLimitAttr().Set(0.0)
    joint.CreateCollisionEnabledAttr().Set(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
    drive.CreateTypeAttr().Set('force')
    drive.CreateStiffnessAttr().Set(L.SPRING_N_PER_M)
    drive.CreateDampingAttr().Set(L.SPRING_DAMPING)
    drive.CreateTargetPositionAttr().Set(L.SPRING_PRELOAD_M)
    drive.CreateTargetVelocityAttr().Set(0.0)
    drive.CreateMaxForceAttr().Set(100.0)
    return cap_path


def build_supply(stage, material, key_material):
    _housing(stage, ROOT + '/Supply/Housing', L.SUPPLY_BOX, SUPPLY_COLOUR, material)
    top = L.supply_top_z()
    bx, by = L.SUPPLY_BUTTON_XY
    spring_cap(stage, SUPPLY_BUTTON, (bx, by, top + L.CAP_LIFT + L.CAP_H / 2), (L.KEY_SIZE, L.KEY_SIZE, L.CAP_H),
               (0.92, 0.92, 0.90), key_material)
    _box(stage, ROOT + '/Supply/ButtonCollar', (L.KEY_SIZE + 0.004, L.KEY_SIZE + 0.004, L.CAP_LIFT - 0.001),
         (bx, by, top + (L.CAP_LIFT - 0.001) / 2), (0.10, 0.10, 0.11), collide=False)
    frame = lambda x, y: Frame((x, y), READ, UP, top)       # noqa: E731
    text(stage, SUPPLY_BUTTON + '/Print', 'I/O', Frame((0, 0), READ, UP, L.CAP_H / 2), 0, 0, 0.007, INK,
         align='centre')
    text(stage, ROOT + '/Supply/Title', 'DC SUPPLY', frame(0.628, -0.070), 0, 0, 0.0065, INK, align='centre')
    text(stage, ROOT + '/Supply/ButtonLabel', 'ON/OFF', frame(bx + 0.021, by), 0, 0, 0.005, INK, align='centre')
    for name, (lx, ly) in L.SUPPLY_LAMPS.items():
        _disc(stage, ROOT + f'/Supply/Lamp{name.title()}Dim', 0.0045, (lx, ly, top + 0.0004), (0.10, 0.10, 0.09))
        _disc(stage, LAMP_PATHS[name], 0.0046, (lx, ly, top + 0.0007), LAMP_ON[name], visible=False,
              material=_render_material(stage, ROOT + f'/Supply/Looks/Lamp{name}', LAMP_ON[name], emission=4.0))
        text(stage, ROOT + f'/Supply/Lamp{name}Label', name, frame(lx, ly + 0.010), 0, 0, 0.006, INK)
    text(stage, ROOT + '/Supply/Spec1', '5.00 V', frame(0.712, -0.070), 0, 0, 0.0065, INK, align='centre')
    text(stage, ROOT + '/Supply/Spec2', '0.10 A MAX', frame(0.724, -0.070), 0, 0, 0.005, INK, align='centre')
    x_edge = L.SUPPLY_BOX[0][1]
    for node, colour, mark in ((L.SUPPLY_PLUS, RED, '+'), (L.SUPPLY_MINUS, (0.03, 0.03, 0.03), '-')):
        nx, ny = L.node_xy(node)
        _strap(stage, ROOT + f'/Supply/Strap{"Plus" if mark == "+" else "Minus"}', x_edge, nx - L.PAD_HALF + 0.004,
               ny, colour)
        text(stage, ROOT + f'/Supply/Mark{"Plus" if mark == "+" else "Minus"}', mark, frame(x_edge - 0.008, ny), 0, 0,
             0.009, colour, align='centre')


def build_box(stage, material):
    _housing(stage, ROOT + '/BlackBox/Housing', L.BOX_BOX, BOX_COLOUR, material)
    top = L.TABLE_Z + L.BOX_BOX[2]
    x_edge = L.BOX_BOX[0][0]
    for name, node in L.BOX_TERMINALS.items():
        nx, ny = L.node_xy(node)
        colour = L.BOX_TERMINAL_COLOURS[name]
        _strap(stage, ROOT + f'/BlackBox/Strap{name}', nx + L.PAD_HALF - 0.004, x_edge, ny, colour)
        _disc(stage, ROOT + f'/BlackBox/Dot{name}', 0.006, (x_edge + 0.012, ny, top + 0.0004), colour)
        text(stage, ROOT + f'/BlackBox/Letter{name}', name, Frame((x_edge + 0.030, ny), READ, UP, top), 0, 0, 0.011,
             WHITE, align='centre')
    text(stage, ROOT + '/BlackBox/Title', 'BOX', Frame((L.BOX_BOX[0][1] - 0.016, -0.090), READ, UP, top), 0, 0, 0.010,
         WHITE, align='centre')


# --- tray ------------------------------------------------------------------------------------------------------------
def build_tray(stage, material):
    (x0, x1), (y0, y1) = L.tray_extent()
    top, floor, groove = L.TABLE_Z + L.TRAY_H, L.TABLE_Z + L.TRAY_FLOOR, L.TABLE_Z + L.TRAY_GROOVE_FLOOR
    mx, my = L.TRAY_MOUTH_HALF
    wx, wy = L.TRAY_WELL_HALF
    gh = L.TRAY_GROOVE_HALF
    throat = top - L.TRAY_CHAMFER_DEPTH
    seats = L.tray_seats()
    holes = [(s['xy'][0] - mx, s['xy'][0] + mx, s['xy'][1] - my, s['xy'][1] + my) for s in seats.values()]
    quads = [q for q, _ in _grid_cells((x0, x1), (y0, y1), holes, top)]
    for s in seats.values():
        cx, cy = s['xy']
        quads += _pocket(cx, cy, [(mx, my, top), (wx, wy, throat), (wx, wy, floor)])
        # Floor either side of the pin groove (the groove runs along the well, local x = world y).
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
    UsdGeom.Xform.Define(stage, ROOT + '/Tray')
    _quads(stage, ROOT + '/Tray/Body', quads, TRAY_COLOUR, collide=True, material=material)
    for label, s in seats.items():
        cx, cy = s['xy']
        text(stage, f'{ROOT}/Tray/Label{label}', s['printed'], Frame((cx + L.TRAY_LABEL_DX, cy), READ, UP, top), 0, 0,
             0.0075, WHITE, align='centre')
        seat = UsdGeom.Xform.Define(stage, f'{ROOT}/Tray/Seat{label}')
        UsdGeom.XformCommonAPI(seat).SetTranslate(Gf.Vec3d(cx, cy, floor))


# --- blocks ----------------------------------------------------------------------------------------------------------
def build_block(stage, block_id, pose_xyz, yaw_deg, material):
    kind, printed, _, mass = L.BLOCKS[block_id]
    path = L.block_path(ROOT, block_id)
    body = UsdGeom.Xform.Define(stage, path)
    api = UsdGeom.XformCommonAPI(body)
    api.SetTranslate(Gf.Vec3d(*pose_xyz))
    api.SetRotate(Gf.Vec3f(0, 0, yaw_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    _rigid(body.GetPrim(), mass)
    colour = CHANNEL_COLOURS.get(block_id, BLOCK_COLOURS.get(kind, (0.5, 0.5, 0.5)))
    bx, by, bz = L.BASE
    base_colour = (0.12, 0.12, 0.13) if kind == 'channel' else colour
    _box(stage, path + '/Base', L.BASE, (0, 0, bz / 2), base_colour, material)
    for sign, name in ((1, 'PinPlus'), (-1, 'PinMinus')):
        sphere = UsdGeom.Sphere.Define(stage, f'{path}/{name}')
        sphere.CreateRadiusAttr(L.PIN_R)
        sphere.CreateExtentAttr([Gf.Vec3f(-L.PIN_R), Gf.Vec3f(L.PIN_R)])
        UsdGeom.XformCommonAPI(sphere).SetTranslate(Gf.Vec3d(*L.pin_local(sign)))
        sphere.CreateDisplayColorAttr([Gf.Vec3f(*BRASS)])
        _collider(sphere.GetPrim(), material)
    meter = kind in ('voltmeter', 'ammeter')
    upper = L.HEAD if meter else L.HANDLE
    _box(stage, path + ('/Head' if meter else '/Handle'), upper, (0, 0, bz + upper[2] / 2), colour, material)
    top = L.BLOCK_TOP
    local = Frame((0, 0), (1, 0), (0, 1), top)
    if kind == 'wire':
        quad_mesh(stage, path + '/WireMark', local, [(-0.008, -0.0011, 0.008, 0.0011)], INK, 0.0004)
    elif kind == 'resistor':
        text(stage, path + '/Value', printed, local, 0, 0, 0.0055, INK, align='centre')
    elif kind == 'channel':
        text(stage, path + '/Label', printed, local, 0, 0, 0.0050, INK, align='centre')
        band = Frame((0, 0), (1, 0), (0, 1), bz)
        quad_mesh(stage, path + '/PlusBand', band, [(L.BASE_PLUS_BAND[0], -by / 2, L.BASE_PLUS_BAND[1], by / 2)], RED,
                  0.0004)
    elif meter:
        u0, v0, u1, v1 = L.METER_WINDOW
        looks = display_looks(stage)
        quad_mesh(stage, path + '/Window', local, [(u0, v0, u1, v1)], WINDOW, 0.0004, material=looks['DisplayWindow'])
        quad_mesh(stage, path + '/PlusBand', local, [(L.PLUS_BAND[0], -L.HEAD[1] / 2, L.PLUS_BAND[1], L.HEAD[1] / 2)],
                  RED, 0.0004)
        lu, lv, lh = L.METER_LABEL
        text(stage, path + '/Label', printed, local, lu, lv, lh, WHITE, dz=0.0008)
        uu, uv, uh = L.METER_UNIT
        units = ('mV', 'V') if kind == 'voltmeter' else ('uA', 'mA')
        for unit in units:
            text(stage, f'{path}/Unit_{unit}', unit, local, uu, uv, uh, LIT, align='right', dz=0.0008,
                 visible=False, material=looks['DisplayLit'])
        d = L.METER_DIGITS
        Readout(stage, path + '/Readout', local, u0=d['u0'], v0=d['v0'], digit_w=d['digit_w'], digit_h=d['digit_h'],
                stroke=d['stroke'], pitch=d['pitch'], count=d['count'], points=d['points'], material=looks['DisplayLit'])
    return path


def build_rocker(stage, block_pose, yaw_deg, material):
    """The switch's rocker: a body on a revolute joint across the block at the handle top (local y axis)."""
    block = L.block_path(ROOT, 'SW')
    rocker = UsdGeom.Xform.Define(stage, ROCKER)
    centre = (block_pose[0], block_pose[1], block_pose[2] + L.ROCKER_PIVOT_Z + L.ROCKER[2] / 2)
    api = UsdGeom.XformCommonAPI(rocker)
    api.SetTranslate(Gf.Vec3d(*centre))
    api.SetRotate(Gf.Vec3f(0, 0, yaw_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    _rigid(rocker.GetPrim(), 0.003)
    _box(stage, ROCKER + '/Plate', L.ROCKER, (0, 0, 0), (0.93, 0.93, 0.91), material)
    local = Frame((0, 0), (1, 0), (0, 1), L.ROCKER[2] / 2)
    text(stage, ROCKER + '/MarkOn', 'I', local, 0.0085, 0, 0.0055, INK, align='centre')
    text(stage, ROCKER + '/MarkOff', 'O', local, -0.0085, 0, 0.0055, INK, align='centre')
    joint = UsdPhysics.RevoluteJoint.Define(stage, ROCKER_JOINT)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(block)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(ROCKER)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, L.ROCKER_PIVOT_Z))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, -L.ROCKER[2] / 2))
    joint.CreateAxisAttr().Set('Y')
    joint.CreateLowerLimitAttr().Set(-L.ROCKER_ANGLE_DEG)
    joint.CreateUpperLimitAttr().Set(L.ROCKER_ANGLE_DEG)
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr(True)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
    drive.CreateTypeAttr().Set('force')
    drive.CreateStiffnessAttr().Set(0.0035)          # N m per degree: about 3 N at the rocker end to flip it
    drive.CreateDampingAttr().Set(0.0002)
    drive.CreateTargetPositionAttr().Set(-L.ROCKER_ANGLE_DEG - 2.0)    # starts OFF: the O end (-x) down
    drive.CreateMaxForceAttr().Set(1.0)
    return ROCKER


def set_rocker_target(stage, closed):
    target = (L.ROCKER_ANGLE_DEG + 2.0) * (1 if closed else -1)
    UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(ROCKER_JOINT), 'angular').GetTargetPositionAttr().Set(float(target))


# --- three-node block (changeover switch; for rc_capacitance / circuit_blackbox3, not built in resistor_blackbox) --
def build_changeover(stage, block_id, pose_xyz, yaw_deg, material):
    """A straight three-node block (layout.BASE3, pins T1, C, T2 at layout.changeover_pins) with a lever on a
    revolute joint across the block over the common pin. The lever rests against one end stop (+-LEVER_ANGLE_DEG):
    pushing its raised end down past the middle flips it (the runtime flips the drive target, as the switch rocker);
    layout.lever_position maps its angle to the throw, open between the contact angles (break before make).
    Returns (block path, lever path, joint path)."""
    kind, printed, _, mass = L.CHANGEOVER_BLOCKS[block_id]
    path = L.block_path(ROOT, block_id)
    body = UsdGeom.Xform.Define(stage, path)
    api = UsdGeom.XformCommonAPI(body)
    api.SetTranslate(Gf.Vec3d(*pose_xyz))
    api.SetRotate(Gf.Vec3f(0, 0, yaw_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    _rigid(body.GetPrim(), mass)
    bz = L.BASE3[2]
    _box(stage, path + '/Base', L.BASE3, (0, 0, bz / 2), (0.30, 0.30, 0.34), material)
    for name, local in L.changeover_pins().items():
        sphere = UsdGeom.Sphere.Define(stage, f'{path}/Pin{name}')
        sphere.CreateRadiusAttr(L.PIN_R)
        sphere.CreateExtentAttr([Gf.Vec3f(-L.PIN_R), Gf.Vec3f(L.PIN_R)])
        UsdGeom.XformCommonAPI(sphere).SetTranslate(Gf.Vec3d(*local))
        sphere.CreateDisplayColorAttr([Gf.Vec3f(*BRASS)])
        _collider(sphere.GetPrim(), material)
    _box(stage, path + '/Handle', (0.020, L.HANDLE[1], L.HANDLE[2]), (L.CHANGEOVER_PINS['C'], 0, bz + L.HANDLE[2] / 2),
         (0.30, 0.30, 0.34), material)
    local = Frame((0, 0), (1, 0), (0, 1), bz)
    for name, x in L.CHANGEOVER_PINS.items():
        if name == 'C':      # the handle stands over the C pin: C is printed on the base beside it, both sides
            for side, v in (('L', 0.0085), ('R', -0.0085)):
                text(stage, f'{path}/Mark{name}{side}', name, local, x, v, 0.004, WHITE, align='centre',
                     dz=0.0004)
        else:
            text(stage, f'{path}/Mark{name}', name, local, x, 0, 0.006, WHITE, align='centre', dz=0.0004)
    lever_path = path + '_Lever'
    pivot_z = L.BLOCK_TOP + 0.003
    lever = UsdGeom.Xform.Define(stage, lever_path)
    c, s_ = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    lx = L.CHANGEOVER_PINS['C']
    centre = (pose_xyz[0] + c * lx, pose_xyz[1] + s_ * lx, pose_xyz[2] + pivot_z + L.LEVER[2] / 2)
    lapi = UsdGeom.XformCommonAPI(lever)
    lapi.SetTranslate(Gf.Vec3d(*centre))
    lapi.SetRotate(Gf.Vec3f(0, 0, yaw_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    _rigid(lever.GetPrim(), 0.004)
    _box(stage, lever_path + '/Plate', L.LEVER, (0, 0, 0), (0.93, 0.80, 0.10), material)
    joint_path = PHYSICS_ROOT + f'/{block_id}LeverJoint'
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(lever_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(lx, 0, pivot_z))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, -L.LEVER[2] / 2))
    joint.CreateAxisAttr().Set('Y')
    joint.CreateLowerLimitAttr().Set(-L.LEVER_ANGLE_DEG)
    joint.CreateUpperLimitAttr().Set(L.LEVER_ANGLE_DEG)
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr(True)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
    drive.CreateTypeAttr().Set('force')
    drive.CreateStiffnessAttr().Set(0.003)
    drive.CreateDampingAttr().Set(0.0002)
    drive.CreateTargetPositionAttr().Set(-L.LEVER_ANGLE_DEG - 2.0)    # starts on T1
    drive.CreateMaxForceAttr().Set(1.0)
    return path, lever_path, joint_path


# --- logger unit -----------------------------------------------------------------------------------------------------
def build_logger(stage, material, key_material):
    _housing(stage, ROOT + '/Logger/Housing', L.LOGGER_BOX, LOGGER_COLOUR, material)
    top = L.logger_top_z()
    keys = {}
    for key, (x, y) in L.logger_key_centres().items():
        path = f'{ROOT}/Logger/Key_{key}'
        colour = (0.80, 0.10, 0.08) if key == 'rec' else (0.82, 0.83, 0.85)
        spring_cap(stage, path, (x, y, top + L.CAP_LIFT + L.CAP_H / 2), (L.KEY_SIZE, L.KEY_SIZE, L.CAP_H), colour,
                   key_material)
        label = L.LOGGER_KEY_PRINT.get(key, key)
        text(stage, path + '/Print', label, Frame((0, 0), READ, UP, L.CAP_H / 2), 0, 0,
             0.010 if len(label) == 1 else 0.0050, WHITE if key == 'rec' else INK, align='centre')
        _box(stage, f'{ROOT}/Logger/Key_{key}Collar', (L.KEY_SIZE + 0.004, L.KEY_SIZE + 0.004, L.CAP_LIFT - 0.001),
             (x, y, top + (L.CAP_LIFT - 0.001) / 2), (0.05, 0.05, 0.06), collide=False)
        keys[key] = path
    (wx0, wx1), (wy0, wy1) = L.LOGGER_DISPLAY
    centre = ((wx0 + wx1) / 2, (wy0 + wy1) / 2)
    frame = Frame(centre, READ, UP, top)
    hu, hv = (wy1 - wy0) / 2, (wx1 - wx0) / 2
    looks = display_looks(stage)
    quad_mesh(stage, ROOT + '/Logger/Window', frame, [(-hu, -hv, hu, hv)], WINDOW, 0.0004, material=looks['DisplayWindow'])
    d = L.LOGGER_CH1
    text(stage, ROOT + '/Logger/Ch1Label', 'CH1', frame, -0.062, d['v0'] + d['digit_h'] / 2, 0.007, WHITE, dz=0.0008)
    Readout(stage, ROOT + '/Logger/Ch1', frame, u0=d['u0'], v0=d['v0'], digit_w=d['digit_w'], digit_h=d['digit_h'],
            stroke=d['stroke'], pitch=d['pitch'], count=d['count'], points=d['points'], material=looks['DisplayLit'])
    text(stage, ROOT + '/Logger/Ch1Unit', 'V', frame, 0.016, d['v0'] + 0.004, 0.008, LIT, dz=0.0008,
         material=looks['DisplayLit'])
    f = L.LOGGER_FIELD_DIGIT
    labels = {'set': ('SET S', -0.062), 'rate': ('RATE', -0.005), 'no': ('NO', -0.062), 'left': ('LEFT', -0.005)}
    for name, spec in L.LOGGER_FIELDS.items():
        label, lu = labels[name]
        text(stage, f'{ROOT}/Logger/{name.title()}Label', label, frame, lu, spec['v0'] + f['digit_h'] / 2, 0.0055,
             WHITE, dz=0.0008)
        Readout(stage, f'{ROOT}/Logger/{name.title()}', frame, u0=spec['u0'], v0=spec['v0'], sign=False,
                count=spec['count'], material=looks['DisplayLit'], **f)
    text(stage, ROOT + '/Logger/RateUnit', 'HZ', frame, 0.049, -0.006 + 0.003, 0.005, WHITE, dz=0.0008)
    for name, (u, v) in (('REC', (0.048, 0.042)), ('ERR', (0.050, -0.036))):
        p = frame.at(u, v)
        _disc(stage, ROOT + f'/Logger/{name.title()}LampDim', 0.0045, (p[0], p[1], top + 0.0006), (0.10, 0.10, 0.09))
        _disc(stage, LAMP_PATHS[name], 0.0046, (p[0], p[1], top + 0.0009), LAMP_ON[name], visible=False,
              material=_render_material(stage, ROOT + f'/Logger/Looks/Lamp{name}', LAMP_ON[name], emission=4.0))
        text(stage, ROOT + f'/Logger/{name.title()}LampLabel', name, frame, u, v - 0.011, 0.005, WHITE,
             align='centre', dz=0.0008)
    text(stage, ROOT + '/Logger/Title', 'LOGGER', Frame((L.LOGGER_BOX[0][1] - 0.010, -0.133), READ, UP, top), 0, 0,
         0.007, WHITE, align='centre')
    return keys


# --- whole bench -----------------------------------------------------------------------------------------------------
def build(stage):
    """Author everything; returns {'blocks': {id: path}, 'keys': {key: cap path}, 'button': cap path}."""
    retire_hooke_apparatus(stage)
    repair_cube_extents(stage)
    z = table_top_z(stage)
    if abs(z - L.TABLE_Z) > 0.002:
        raise RuntimeError(f'table top at {z:.4f} m, the circuit board layout expects {L.TABLE_Z} m')
    UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.Xform.Define(stage, PHYSICS_ROOT)
    fixed = _physics_material(stage, '/World/PhysicsMaterials/CircuitGridFixed', 0.5, 0.4)
    blocks_mat = _physics_material(stage, '/World/PhysicsMaterials/CircuitGridBlock', 0.5, 0.4)
    keys_mat = _physics_material(stage, '/World/PhysicsMaterials/CircuitGridKey', 0.8, 0.7)
    build_board(stage, fixed)
    build_supply(stage, fixed, keys_mat)
    build_box(stage, fixed)
    build_tray(stage, fixed)
    keys = build_logger(stage, fixed, keys_mat)
    paths = {}
    UsdGeom.Xform.Define(stage, ROOT + '/Blocks')
    for label, seat in L.tray_seats().items():
        pose = (seat['xy'][0], seat['xy'][1], L.TABLE_Z + L.TRAY_FLOOR + 0.0005)
        paths[label] = build_block(stage, label, pose, L.TRAY_YAW_DEG, blocks_mat)
        if label == 'SW':
            build_rocker(stage, pose, L.TRAY_YAW_DEG, blocks_mat)
    table = stage.GetPrimAtPath(TABLE_TOP)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(table)
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    return {'blocks': paths, 'keys': keys, 'button': SUPPLY_BUTTON, 'rocker': ROCKER}


# --- runtime visuals -----------------------------------------------------------------------------------------------
class Displays:
    """Readouts and lamps of the bench, updated from the bench's shown state (only what changed)."""

    def __init__(self, stage):
        self.stage = stage
        self.readouts = {}
        self.units = {}
        self.lamps = {}

    def _readout(self, path, digits):
        """A Readout object bound to an already authored readout (prims looked up by name)."""
        if path not in self.readouts:
            r = object.__new__(Readout)
            r.stage, r.path, r.count, r.state = self.stage, path, digits, None
            r.prims = {}
            prim = self.stage.GetPrimAtPath(path)
            for child in prim.GetChildren():
                name = child.GetName()
                if name == 'Sign':
                    r.prims['sign'] = child.GetPath()
                elif name.startswith('D'):
                    r.prims[(int(name[1:-1]), name[-1])] = child.GetPath()
                elif name.startswith('P'):
                    r.prims[('dp', int(name[1:]))] = child.GetPath()
            self.readouts[path] = r
        return self.readouts[path]

    def meter(self, block_id, shown):
        path = L.block_path(ROOT, block_id)
        self._readout(path + '/Readout', 5).show(shown['sign'], shown['digits'], shown['point'])
        if self.units.get(block_id) != shown['unit']:
            kind = L.BLOCKS[block_id][0]
            for unit in (('mV', 'V') if kind == 'voltmeter' else ('uA', 'mA')):
                prim = UsdGeom.Imageable(self.stage.GetPrimAtPath(f'{path}/Unit_{unit}'))
                prim.MakeVisible() if unit == shown['unit'] else prim.MakeInvisible()
            self.units[block_id] = shown['unit']

    def lamp(self, name, on):
        if self.lamps.get(name) != bool(on):
            prim = UsdGeom.Imageable(self.stage.GetPrimAtPath(LAMP_PATHS[name]))
            prim.MakeVisible() if on else prim.MakeInvisible()
            self.lamps[name] = bool(on)

    def logger(self, shown):
        v = max(-9.999, min(9.999, shown['ch1_V']))
        digits = f'{abs(v):.3f}'.replace('.', '')
        self._readout(ROOT + '/Logger/Ch1', 4).show('-' if v < 0 else ' ', digits, 0)
        for name, value, count in (('Set', shown['set_s'], 2), ('Rate', shown['rate_hz'], 3),
                                   ('No', shown['number'] or None, 2), ('Left', shown['left'], 2)):
            text_ = ' ' * count if value is None else str(int(value))[-count:].rjust(count)
            self._readout(f'{ROOT}/Logger/{name}', count).show(' ', text_, None)
        self.lamp('REC', shown['rec'])
        self.lamp('ERR', shown['error'])

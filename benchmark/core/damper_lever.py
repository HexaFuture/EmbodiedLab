"""Eddy-current damper lever: a reusable two-detent bench control (USD authoring and the runtime detent logic).

A carriage with a vertical yellow post slides on a straight rail (a world prismatic joint) on top of a pedestal. A
spring detent (the circuit-panel selector pattern: a linear joint drive whose target the runtime moves to the detent
nearest the carriage) holds it at OFF (detent 0) or ON (detent 1). A magnet block rides on the carriage beside the
rail; a copper vane is fixed on the pedestal under the magnet's ON position, so the state is visible from the lever
position: OFF and ON are printed at the detents and the magnet stands over the copper vane when ON.

The physics of the damper is the magnet's overlap with the vane: `overlap(position)` is 0 at OFF (and anywhere
within a vane's length of it), 1 at ON, linear in between. The bench applies its viscous force scaled by it. The
lever can be moved at any time; nothing locks it.

No dependence on any particular bench: `build_lever` takes the pedestal placement, `Lever` the carriage position.
"""
from __future__ import annotations

STIFFNESS_N_PER_M = 200.0          # 4 N mid-way between detents 40 mm apart
DAMPING_NS_PER_M = 10.0
MAX_FORCE_N = 50.0
TOLERANCE_M = 0.004                # within this of a detent the lever is at that detent
TRAVEL_MARGIN_M = 0.003            # end stops just past the end detents
CARRIAGE = (0.034, 0.024, 0.012)   # along X, along the rail, height (rail along Y)
CARRIAGE_GAP = 0.001
CARRIAGE_MASS_KG = 0.10
POST_R, POST_H = 0.009, 0.060
MAGNET = (0.018, 0.020, 0.008)     # beside the carriage, MAGNET_DX from the rail
MAGNET_DX = 0.026
VANE = (0.020, 0.030, 0.002)
PEDESTAL_W = 0.080                 # across the rail
PEDESTAL_END = 0.035               # beyond the end detents
POST_COLOUR = (0.88, 0.62, 0.08)


def overlap(position_m, spacing_m):
    """Fraction of the magnet's length over the copper vane at carriage travel `position_m` from OFF."""
    lo = max(position_m - MAGNET[1] / 2, spacing_m - VANE[1] / 2)
    hi = min(position_m + MAGNET[1] / 2, spacing_m + VANE[1] / 2)
    return max(0.0, min(1.0, (hi - lo) / MAGNET[1]))


def build_lever(stage, path, *, name='Damper', detent0=(0.0, 0.0), direction=(0.0, 1.0), spacing=0.04, top_z=0.90,
                base_z=0.76, labels=('OFF', 'ON'), title='EDDY DAMPER', contact_material=None,
                stiffness=STIFFNESS_N_PER_M, damping=DAMPING_NS_PER_M):
    """Pedestal, printed scale, copper vane, carriage with post and magnet, and the rail joint with its detent drive.

    The rail runs along world +Y (direction (0, 1)) or -Y ((0, -1)) from detent 0 at world (x, y) = detent0; the
    pedestal top is at top_z. Returns the paths and the rail geometry for `Lever`."""
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade
    from pendulum_asset import _FONT, _box, _cylinder, _flat_text, _material, _text
    _FONT.setdefault('F', (31, 16, 16, 30, 16, 16, 16))
    _FONT.setdefault('H', (17, 17, 17, 31, 17, 17, 17))
    _FONT.setdefault('Y', (17, 17, 10, 4, 4, 4, 4))
    if tuple(direction) not in ((0.0, 1.0), (0.0, -1.0)):
        raise ValueError('the damper lever rail runs along world Y')
    sign = direction[1]
    x, y0 = detent0
    y1 = y0 + sign * spacing
    mats = {
        'pedestal': _material(stage, path + '/Materials/Pedestal', (.16, .18, .21), .3),
        'ink': _material(stage, path + '/Materials/Ink', (.93, .95, .97)),
        'slot': _material(stage, path + '/Materials/Slot', (.01, .01, .012)),
        'copper': _material(stage, path + '/Materials/Copper', (.80, .42, .20), .8, .3),
        'magnet': _material(stage, path + '/Materials/Magnet', (.45, .04, .05), .4, .4),
        'post': _material(stage, path + '/Materials/Post', POST_COLOUR, .4, .25),
        'carriage': _material(stage, path + '/Materials/Carriage', (.30, .32, .35), .5, .3),
    }
    UsdGeom.Xform.Define(stage, path)

    def physical(geom):
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        api = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
        api.CreateContactOffsetAttr().Set(.0005)
        api.CreateRestOffsetAttr().Set(0.0)
        if contact_material is not None:
            UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(
                contact_material, UsdShade.Tokens.weakerThanDescendants, 'physics')
        return geom

    ymid = (y0 + y1) / 2
    length = spacing + 2 * PEDESTAL_END
    physical(_box(stage, path + '/Pedestal', (x, ymid, (base_z + top_z) / 2), (PEDESTAL_W, length, top_z - base_z),
                  mats['pedestal']))
    _box(stage, path + '/Slot', (x, ymid, top_z + .0003), (.010, spacing + .030, .0006), mats['slot'])
    vane_x = x + MAGNET_DX
    _box(stage, path + '/Vane', (vane_x, y1, top_z + VANE[2] / 2), VANE, mats['copper'])
    # Detent names on the other side of the slot, read from the robot side (+Y): letter tops toward -Y.
    for k, (text, y) in enumerate(zip(labels, (y0, y1))):
        _box(stage, path + f'/Tick{k}', (x - .016, y, top_z + .0004), (.012, .0015, .0006), mats['ink'])
        _flat_text(stage, path + f'/Label{k}', text, (x - .030, y, top_z + .0004), .010, mats['ink'])
    near_y = max(y0, y1) + PEDESTAL_END
    _text(stage, path + '/Title', title, (x, near_y + .0008, top_z - .020), .009, mats['ink'])
    # Carriage at OFF.
    cz = top_z + CARRIAGE_GAP + CARRIAGE[2] / 2
    carriage_path = path + '/Carriage'
    carriage = UsdGeom.Xform.Define(stage, carriage_path)
    carriage.AddTranslateOp().Set(Gf.Vec3d(x, y0, cz))
    UsdPhysics.RigidBodyAPI.Apply(carriage.GetPrim()).CreateKinematicEnabledAttr(False)
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(carriage.GetPrim())
    body.CreateSolverPositionIterationCountAttr(16)
    body.CreateSolverVelocityIterationCountAttr(8)
    body.CreateSleepThresholdAttr(0.0)
    body.CreateLinearDampingAttr(0.0)
    body.CreateAngularDampingAttr(0.0)
    mass = UsdPhysics.MassAPI.Apply(carriage.GetPrim())
    mass.CreateMassAttr(CARRIAGE_MASS_KG)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
    physical(_box(stage, carriage_path + '/Block', (0, 0, 0), CARRIAGE, mats['carriage']))
    physical(_cylinder(stage, carriage_path + '/Post', (0, 0, CARRIAGE[2] / 2 + POST_H / 2), POST_R, POST_H, mats['post']))
    _box(stage, carriage_path + '/MagnetArm', (MAGNET_DX / 2, 0, 0), (MAGNET_DX, .008, .006), mats['carriage'])
    _box(stage, carriage_path + '/Magnet', (MAGNET_DX, 0, CARRIAGE[2] / 2 - MAGNET[2] / 2 + .001), MAGNET, mats['magnet'])
    joint = UsdPhysics.PrismaticJoint.Define(stage, path + '/RailJoint')
    joint.CreateBody1Rel().SetTargets([Sdf.Path(carriage_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(x, y0, cz))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateAxisAttr('Y')
    low, high = sorted((-sign * TRAVEL_MARGIN_M, sign * (spacing + TRAVEL_MARGIN_M)))
    joint.CreateLowerLimitAttr(low)
    joint.CreateUpperLimitAttr(high)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
    drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(float(stiffness))
    drive.CreateDampingAttr(float(damping))
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateMaxForceAttr(MAX_FORCE_N)
    return lever_info(path, name=name, detent0=detent0, direction=direction, spacing=spacing, labels=labels)


def lever_info(path, *, name='Damper', detent0=(0.0, 0.0), direction=(0.0, 1.0), spacing=0.04, labels=('OFF', 'ON'),
               **_):
    """Paths and rail geometry of a lever built by build_lever with the same arguments (for `Lever`)."""
    return {'name': name, 'path': path, 'carriage': path + '/Carriage', 'joint': path + '/RailJoint',
            'detent0': tuple(detent0), 'sign': direction[1], 'spacing': spacing, 'labels': tuple(labels)}


class Lever:
    """Runtime of one lever: carriage travel from detent 0, the detent spring target, overlap and change events."""

    def __init__(self, info, stage=None):
        self.info, self.stage = info, stage
        self.index = 0                 # detent the spring aims at
        self.state = info['labels'][0]  # label of the detent the carriage stands at, or 'between'
        self.position_m = 0.0
        self.fraction = 0.0

    def travel(self, carriage_xy):
        return float((carriage_xy[1] - self.info['detent0'][1]) * self.info['sign'])

    def update(self, tick, carriage_xy):
        """Read the carriage; aim the detent spring at the nearest detent. Returns change events {tick, old, new}."""
        spacing, labels = self.info['spacing'], self.info['labels']
        p = self.position_m = self.travel(carriage_xy)
        self.fraction = overlap(p, spacing)
        index = min(len(labels) - 1, max(0, int(round(p / spacing))))
        if index != self.index and self.stage is not None:
            from pxr import UsdPhysics
            UsdPhysics.DriveAPI.Get(self.stage.GetPrimAtPath(self.info['joint']), 'linear').GetTargetPositionAttr().Set(
                float(self.info['sign'] * index * spacing))
        self.index = index
        state = labels[index] if abs(p - index * spacing) <= TOLERANCE_M else 'between'
        events = []
        if state != self.state:
            events.append({'tick': tick, 'old': self.state, 'new': state, 'position_mm': round(p * 1000, 1)})
            self.state = state
        return events

    def record(self):
        return {'state': self.state, 'position_mm': round(self.position_m * 1000, 1), 'overlap': round(self.fraction, 3)}

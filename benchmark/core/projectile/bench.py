"""projectile_landing spring-launcher bench: launcher, optical gates, receiving deck with recovery pocket, cup, ball pad and trigger.

Placed for the public standard views (robot0_agentview_left/right and the wrist camera): the launcher, knob, trigger
and ball pad stand in the right agent view's free table area, the receiving deck and recovery pocket in the
left one's, the cup on its pad beyond the deck's far side (0.68 m ahead of the robot: comfortably reachable). The
launch axis runs along world +x at y -0.12, 0.47 m in front of the robot base line.
"""
import math
from .geometry import *
from .asset import build_ball, build_launcher, material, collider

ORIGIN = (.86, -.12, .86)
ROOT = '/World/ProjectileLab'
BALL_PAD_TOP_Z = .104 + .004          # lowest point of the seated ball (the ball starts seated on the ball pad)
# Ball pad: its top is a shallow seat, an inverted square pyramid that is steeper in the middle. A ball set down
# anywhere on it rolls to the middle and stays there, as on a real ball seat (PhysX has no rolling resistance, so a
# ball on a flat top keeps any small speed it is set down with). The seated ball centre is the old flat-top rest point.
BALL_PAD_HALF = .0275
BALL_SEAT_INNER_HALF = .009                                      # the seated ball touches the four inner faces
BALL_SEAT_INNER_SLOPE = math.tan(math.radians(15.))
BALL_SEAT_OUTER_SLOPE = math.tan(math.radians(5.))


def build_bench(stage, spring_k):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
    UsdGeom.Xform.Define(stage, ROOT).AddTranslateOp().Set(Gf.Vec3d(*ORIGIN))
    support = material(stage, ROOT + '/SupportContact', .5)

    def box(path, pos, dims, color, collision=True):
        obj = UsdGeom.Cube.Define(stage, path); obj.CreateSizeAttr(1.)
        obj.AddTranslateOp().Set(Gf.Vec3d(*pos)); obj.AddScaleOp().Set(Gf.Vec3f(*dims))
        obj.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        if collision: collider(obj.GetPrim(), support, .0001)
        return obj
    dark = (.08, .10, .13); metal = (.42, .46, .49); white = (.8, .83, .84)
    drive = build_launcher(stage, ROOT, ORIGIN, spring_k)
    ball = build_ball(stage, ROOT, (*BALL_PAD_LOCAL, BALL_PAD_TOP_Z + BALL_RADIUS))
    # Optical gates: U frames standing on the table beside the flight path, beams through the ball centre line.
    for i, x in enumerate(GATE_X):
        z = MUZZLE[2] + (x - MUZZLE[0]) * math.tan(TILT) if x <= MUZZLE[0] else MUZZLE[2]
        for sign in (-1, 1):
            box(ROOT + f'/Gate{i}Leg{"L" if sign < 0 else "R"}', (x, sign * .046, (z + .03 - .040) / 2), (.008, .008, z + .03 + .040), dark)
            box(ROOT + f'/Gate{i}Sensor{"L" if sign < 0 else "R"}', (x, sign * .038, z), (.008, .008, .009), (.65, .08, .05))
        box(ROOT + f'/Gate{i}Top', (x, 0., z + .034), (.008, .100, .008), dark)
    # Receiving deck: flat cup station, then a shallow V pocket that gathers the ball at the far end.
    box(ROOT + '/ReceivingFoundation', ((.03 + POCKET_END) / 2, 0., -.038), (POCKET_END - .03, 2 * DECK_HALF_WIDTH, .004), dark)
    xs = [.03, DECK_FLAT_END] + [DECK_FLAT_END + (POCKET_END - DECK_FLAT_END) * f for f in (.25, .5, .75, 1.)]
    ys = [-DECK_HALF_WIDTH, 0., DECK_HALF_WIDTH]
    top = [(x, y, recovery_floor_z(x, y)) for x in xs for y in ys]
    points = top + [(x, y, -.036) for x, y, _ in top]; count = len(top); faces = []
    for i in range(len(xs) - 1):
        for j in range(2):
            a = 3 * i + j; b = a + 3
            faces.extend([(a, b, a + 1), (b, b + 1, a + 1), (a + count, a + 1 + count, b + count), (b + count, a + 1 + count, b + 1 + count)])
    boundary = list(range(0, count, 3)) + [count - 2, count - 1] + list(range(count - 4, 1, -3)) + [1]
    for a, b in zip(boundary, boundary[1:] + boundary[:1]):
        faces.extend([(a, a + count, b), (b, a + count, b + count)])
    deck = UsdGeom.Mesh.Define(stage, ROOT + '/ReceivingDeck')
    deck.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    deck.CreateFaceVertexCountsAttr([3] * len(faces)); deck.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
    deck.CreateSubdivisionSchemeAttr('none'); deck.CreateDisplayColorAttr([Gf.Vec3f(*white)])
    UsdPhysics.CollisionAPI.Apply(deck.GetPrim()); UsdPhysics.MeshCollisionAPI.Apply(deck.GetPrim()).CreateApproximationAttr('none')
    contact = PhysxSchema.PhysxCollisionAPI.Apply(deck.GetPrim()); contact.CreateContactOffsetAttr(.0001); contact.CreateRestOffsetAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(deck.GetPrim()).Bind(support, materialPurpose='physics')
    # Backstop 45 mm above the deck: every shot lands on the flat deck and arrives rolling (ball centre 20 mm up).
    # A taller wall reaches the bottom of the hand when it grasps the ball resting against it.
    box(ROOT + '/Backstop', (POCKET_END + .004, 0., -.0075), (.008, 2 * DECK_HALF_WIDTH + .012, .065), white)
    for side in (-1, 1):
        box(ROOT + f'/DeckSide{"L" if side < 0 else "R"}', ((.03 + POCKET_END) / 2, side * (DECK_HALF_WIDTH + .003), -.0075),
            (POCKET_END - .03, .006, .057), white)
    for i in range(int((DECK_FLAT_END - .05) / .01) + 1):
        x = .05 + i * .01
        box(ROOT + f'/RulerTick{i}', (x, -.093, recovery_floor_z(x, -.093) + .0002), (.001, .012 if i % 5 else .022, .0003), dark, False)
    # Receiving light curtain (horizontal optical plane at the public ball-centre height) on rails beside the deck.
    for sign in (-1, 1):
        name = 'L' if sign < 0 else 'R'
        box(ROOT + f'/ReceivingSensorRail{name}', ((.03 + POCKET_END) / 2, sign * .135, RECEIVER_CENTER_Z), (POCKET_END - .03, .009, .009), dark)
        for j, x in enumerate((.045, POCKET_END - .015)):
            box(ROOT + f'/ReceivingRailStand{name}{j}', (x, sign * .135, (RECEIVER_CENTER_Z - .04) / 2), (.01, .01, RECEIVER_CENTER_Z + .04), metal)
            box(ROOT + f'/ReceivingRailFoot{name}{j}', (x, sign * .135, -.036), (.024, .024, .008), dark)
    # Ball pad: a column with a shallow ball seat on top, in which the ball starts.
    bx, by = BALL_PAD_LOCAL
    box(ROOT + '/BallLoadingFoot', (bx, by, -.035), (.070, .070, .010), dark)
    box(ROOT + '/BallLoadingColumn', (bx, by, .035), (.026, .026, .130), (.15, .25, .18))
    ball_seat(stage, ROOT + '/BallLoadingPad', bx, by, (.85, .88, .82), support)
    # Passive receiving cup on its pad beside the deck. The mouth is below the optical plane by a ball radius plus
    # 14 mm, so the landing is measured before the ball can touch the cup.
    cx, cy = CUP_PAD_LOCAL
    box(ROOT + '/CupLoadingPad', (cx, cy, -.032), (.14, .14, .016), dark)
    cup = ROOT + '/Cup'; UsdGeom.Xform.Define(stage, cup).AddTranslateOp().Set(Gf.Vec3d(cx, cy, -.004))
    prim = stage.GetPrimAtPath(cup); UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(.12)
    blue = (.06, .32, .58)
    box(cup + '/Floor', (0, 0, -.018), (.108, .108, .004), blue)
    for axis in (0, 1):
        for sign in (-1, 1):
            pos = [0., 0., (-.016 + CUP_RIM_Z) / 2]; pos[axis] = sign * .052
            dims = [.108, .108, CUP_RIM_Z + .016]; dims[axis] = .004
            box(cup + f'/Wall{axis}{"L" if sign < 0 else "R"}', pos, dims, blue)
    box(cup + '/GripPost', (0, CUP_GRIP_Y, (CUP_POST_TOP_Z - .020) / 2), (.012, .012, CUP_POST_TOP_Z + .020), (.88, .68, .09))
    box(cup + '/GripBridge', (0, .058, -.012), (.025, .025, .008), blue)
    # Trigger: a spring-return push button; the latch releases after real travel under robot contact.
    button = ROOT + '/TriggerButton'
    pos = TRIGGER_LOCAL
    UsdGeom.Xform.Define(stage, button).AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(button))
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(button)).CreateMassAttr(.03)
    from . import visuals
    box(button + '/Cap', (0, 0, 0), (.056, .056, .008), visuals.TRIGGER_COLOUR)
    visuals.trigger_label(stage, ROOT, pos)       # render-only label plate on the table, robot side
    box(ROOT + '/TriggerFoot', (pos[0], pos[1], -.0325), (.080, .080, .015), dark)
    box(ROOT + '/TriggerBase', (pos[0], pos[1], -.001), (.080, .080, .048), dark)
    for side in (-1, 1):
        # Fixed guides below the fully depressed cap and outside its footprint, clear of the fingertip shells.
        box(ROOT + f'/TriggerCollar{"L" if side < 0 else "R"}', (pos[0] + side * .034, pos[1], .023), (.006, .056, .006), metal)
    joint = UsdPhysics.PrismaticJoint.Define(stage, ROOT + '/TriggerGuide')
    joint.CreateBody1Rel().SetTargets([Sdf.Path(button)])
    joint.CreateAxisAttr('Z'); joint.CreateLocalPos0Attr(Gf.Vec3f(*(a + b for a, b in zip(ORIGIN, pos))))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0)); joint.CreateLocalRot0Attr(Gf.Quatf(1)); joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateLowerLimitAttr(-.012); joint.CreateUpperLimitAttr(0.)
    spring = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
    spring.CreateStiffnessAttr(800.); spring.CreateDampingAttr(4.); spring.CreateMaxForceAttr(15.)
    spring.CreateTargetPositionAttr(0.)
    return dict(ball_path=ball, spring_drive=drive, cup_path=cup, trigger_path=button, plunger_path=ROOT + '/Plunger')


def ball_seat(stage, path, bx, by, color, mat):
    """The ball pad: flat bottom on the column top (z .100), seat faces on top. Apex, inner ring, outer rim, bottom."""
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
    zc = BALL_PAD_TOP_Z + BALL_RADIUS - BALL_RADIUS * math.sqrt(1 + BALL_SEAT_INNER_SLOPE ** 2)
    zb = zc + BALL_SEAT_INNER_SLOPE * BALL_SEAT_INNER_HALF
    zh = zb + BALL_SEAT_OUTER_SLOPE * (BALL_PAD_HALF - BALL_SEAT_INNER_HALF)
    ring = ((1, 1), (-1, 1), (-1, -1), (1, -1))                  # counter-clockwise seen from above
    points = [(bx, by, zc)] + [(bx + sx * h, by + sy * h, z) for h, z in
              ((BALL_SEAT_INNER_HALF, zb), (BALL_PAD_HALF, zh), (BALL_PAD_HALF, .100)) for sx, sy in ring]
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        faces += [(0, 1 + i, 1 + j)]                                         # inner faces
        faces += [(1 + i, 5 + i, 5 + j), (1 + i, 5 + j, 1 + j)]              # outer faces
        faces += [(5 + i, 9 + i, 9 + j), (5 + i, 9 + j, 5 + j)]              # sides
    faces += [(9, 12, 11), (9, 11, 10)]                                      # bottom
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces)); mesh.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
    mesh.CreateSubdivisionSchemeAttr('none'); mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()); UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
    contact = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim()); contact.CreateContactOffsetAttr(.0001); contact.CreateRestOffsetAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat, materialPurpose='physics')
    return mesh

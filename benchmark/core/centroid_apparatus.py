"""torsion_centroid specimen on the torsion bench: a locating pin on the turntable axis, a holed sample bar and a
reference disc with a centre bore (the torsion axle with a bar holed at known distances of a parallel-axis kit).

The shared torsion builder (torsion_apparatus.build with pin=True) makes the turntable with the pin, the photogate,
the recorder panel, the damper lever and the storage box. Here the unknown object becomes the 160 x 60 x 20 mm
sample with HOLES on its centre line and the standard disc gets its centre bore. An object is mounted when one of
its holes sits over the pin and it lies flat on the platter: nothing snaps, centres or clamps it; friction on its
rubber underside turns it with the platter. Which hole sits on the pin is the evaluator's private truth
(centroid_backend.CentroidBackend.mount).

Collision: the bar is a set of boxes around square bores HOLE_HALF_M from the hole centre (0.05 mm radial clearance
on the 10 mm pin; drawn round); the disc is a thin full-radius top plate over a square frame around its bore. Their
mass properties stay explicit (centroid_model; the disc's uniform I).
"""
import torsion_apparatus as base
from torsion_apparatus import (ROOT, TURNTABLE, JOINT, TABLE_TOP, OBJECTS, LATCH_LOCAL, PIN_R, PIN_H,
                               FLAG_HALF_ANGLE_RAD, STANDARD_INERTIA_KGM2, OBJECT_R, OBJECT_H, author_joint,
                               set_joint_damping)
from torsion_apparatus import (PANEL_PATH, LEVER_PATH, LEVER, RECORD_S_RANGE, BOX_PATH, BOX_HEIGHT, BOX_FLOOR,
                               SEAT_DXY_M, SEAT_DZ_M, SEAT_TILT_DEG, seat_path, PLATTER_TOP_Z)
from centroid_model import mass_properties, LENGTH_M, WIDTH_M, HEIGHT_M
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

# Holes on the sample's centre line (sample frame, red end +x): number -> x (m). Public.
HOLE_PITCH_M = 0.020
HOLES = {n: round(-0.060 + (n - 1) * HOLE_PITCH_M, 3) for n in range(1, 8)}
HOLE_HALF_M = PIN_R + 0.00005           # bore half-width (square in collision, drawn as a circle of this radius)
DISC_BORE_DEPTH_M = 0.013               # the disc's bore from below (the pin stands 12 mm)
# Storage box: seat 1 the disc's square well, seat 2 a 168 x 68 mm well for the bar (length along X, red end +X).
BOX_SEATS = {'standard_disc': (0.98, 0.06, 0.046, 0.046), 'unknown_object': (1.14, 0.06, 0.084, 0.034)}
# Seated on the pin (private): the object flat (tilt <= 1 deg) with its underside within 2 mm of the platter top and the
# axis inside a bore (the pin cannot stand under the bar anywhere else while it lies flat).
MOUNT_TILT_DEG, MOUNT_LIFT_M, MOUNT_SLACK_M = 1.0, 0.002, 0.001
HOLE_INK = (0.02, 0.02, 0.02)
LABEL_INK = (0.93, 0.93, 0.90)
DIGIT = dict(height=0.010, width=0.006, stroke=0.0018)


# Seven-segment glyphs (own table: the hooke weight-box table holds only the digits it prints).
_SEGMENTS = {"a": (0, 1, 1, 1), "b": (1, 0.5, 1, 1), "c": (1, 0, 1, 0.5), "d": (0, 0, 1, 0),
             "e": (0, 0, 0, 0.5), "f": (0, 0.5, 0, 1), "g": (0, 0.5, 1, 0.5)}
_DIGITS = {"0": "abcdef", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc", "7": "abc",
           "8": "abcdefg", "9": "abcdfg"}


def _digit_rectangles(text):
    """Stroke rectangles (u0, v0, u1, v1) in metres of a one-digit label centred on the origin."""
    h, w, t = DIGIT['height'], DIGIT['width'], DIGIT['stroke']
    if text == "1":
        return [(-t / 2, -h / 2, t / 2, h / 2)]
    rects = []
    for name in _DIGITS[text]:
        u0, v0, u1, v1 = _SEGMENTS[name]
        rects.append((-w / 2 + u0 * (w - t), -h / 2 + v0 * (h - t), -w / 2 + u1 * (w - t) + t, -h / 2 + v1 * (h - t) + t))
    return rects


def _digit_quads(text, centre, z):
    """One digit as flat rectangles on the bar top, upright for the robot (glyph up -Y)."""
    import hooke_apparatus as hooke
    import numpy as np
    quads = []
    for u0, v0, u1, v1 in _digit_rectangles(text):
        corners = [(centre[0] - u, centre[1] - v, z) for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
        quads.append(hooke._oriented(corners, np.array([0.0, 0.0, 1.0])))
    return quads


def _bar(stage, path, foot):
    """The sample's visual housing, holes, numbers, scale and red end, and its colliders (boxes around the bores)."""
    import hooke_apparatus as hooke
    colour = OBJECTS['unknown_object'][1]
    base._box(stage, path + '/Housing', (LENGTH_M, WIDTH_M, HEIGHT_M), (0, 0, HEIGHT_M / 2), colour, collide=False)
    h, half_l, half_w = HOLE_HALF_M, LENGTH_M / 2, WIDTH_M / 2
    boxes = [((-half_l, half_l), (h, half_w)), ((-half_l, half_l), (-half_w, -h))]          # side strips
    xs = sorted(HOLES.values())
    edges = [-half_l] + [v for x in xs for v in (x - h, x + h)] + [half_l]
    boxes += [((a, b), (-h, h)) for a, b in zip(edges[0::2], edges[1::2])]                   # between the bores
    for i, ((x0, x1), (y0, y1)) in enumerate(boxes):
        geom = base._box(stage, path + f'/Collider{i}', (x1 - x0, y1 - y0, HEIGHT_M),
                         ((x0 + x1) / 2, (y0 + y1) / 2, HEIGHT_M / 2), colour, material=foot)
        geom.CreateVisibilityAttr('invisible')
    top = HEIGHT_M + 0.0001
    digits = []
    for n, x in HOLES.items():
        base._annulus_mesh(stage, path + f'/Hole{n}', 0.0, h, top, HOLE_INK, segments=40)
        digits += _digit_quads(str(n), (x, 0.021), top + 0.0001)
    hooke._quads_mesh(stage, path + '/HoleNumbers', digits, LABEL_INK)
    # A white scale (a tick every 10 mm, longer every 20 mm at the holes) beside the holes, and the red +x end.
    for i in range(-7, 8):
        base._box(stage, path + f'/Tick{i + 7}', (.0007, .012 if i % 2 == 0 else .008, .00015),
                  (i * .01, -.020, HEIGHT_M + .000075), (.88, .88, .88), collide=False)
    base._box(stage, path + '/PositiveX', (.010, .045, .0002), (.0745, 0, HEIGHT_M + .0001), (.75, .08, .05),
              collide=False)


def _disc_bore(stage, path, foot):
    """The standard disc's colliders with a centre bore from below: a full-radius plate above the bore and a square
    frame around it (the visual disc is unchanged; the handle stands over the bore)."""
    stage.GetPrimAtPath(path + '/Body/Collider').SetActive(False)
    plate = OBJECT_H - DISC_BORE_DEPTH_M
    geom = UsdGeom.Cylinder.Define(stage, path + '/BorePlate')
    UsdGeom.XformCommonAPI(geom).SetTranslate(Gf.Vec3d(0, 0, DISC_BORE_DEPTH_M + plate / 2))
    geom.CreateRadiusAttr(OBJECT_R)
    geom.CreateHeightAttr(plate)
    geom.CreateAxisAttr('Z')
    geom.CreateExtentAttr([Gf.Vec3f(-OBJECT_R, -OBJECT_R, -plate / 2), Gf.Vec3f(OBJECT_R, OBJECT_R, plate / 2)])
    geom.CreateVisibilityAttr('invisible')
    base._collider(geom.GetPrim(), foot)
    a, h, d = OBJECT_R / 2 ** 0.5, HOLE_HALF_M, DISC_BORE_DEPTH_M
    for i, ((x0, x1), (y0, y1)) in enumerate((((h, a), (-a, a)), ((-a, -h), (-a, a)), ((-h, h), (h, a)), ((-h, h), (-a, -h)))):
        frame = base._box(stage, path + f'/BoreFrame{i}', (x1 - x0, y1 - y0, d), ((x0 + x1) / 2, (y0 + y1) / 2, d / 2),
                          OBJECTS['standard_disc'][1], material=foot)
        frame.CreateVisibilityAttr('invisible')


def build(stage, *, turntable_inertia_kgm2, ballast_fraction, imbalance, ballast_halfspan_m, mark_rgb):
    props = mass_properties(ballast_fraction, imbalance, ballast_halfspan_m)
    paths = base.build(stage, turntable_inertia_kgm2=turntable_inertia_kgm2,
                       unknown_inertia_kgm2=props['diagonal_inertia_kgm2'][2], mark_rgb=mark_rgb,
                       storage_box=BOX_SEATS, pin=True)
    path = paths['unknown_object']
    # The shared builder's disc body is switched off and the bar authored at its own paths: removing a
    # prim and redefining it with another type at the same path in one batch can leave the renderer
    # without the new body.
    stage.GetPrimAtPath(path + '/Body').SetActive(False)
    foot = UsdShade.Material.Get(stage, '/World/PhysicsMaterials/TorsionFoot')
    _bar(stage, path, foot)
    mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(path))
    mass.GetCenterOfMassAttr().Set(Gf.Vec3f(*props['com_m']))
    mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*props['diagonal_inertia_kgm2']))
    _disc_bore(stage, paths['standard_disc'], foot)
    return paths

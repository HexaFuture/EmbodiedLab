"""Render-only launcher parts: coil spring, ratchet rack and pawl, numbered notch scale and pointer.

Nothing here has a collider, a mass or a joint, so the launcher dynamics are those of asset.build_launcher alone.
The coil spans the plunger head and the rear end block and is re-shaped from the plunger travel for every rendered
frame; the pawl (two render variants on the plunger) shows the backend's latched state. The scale sits on a plate
beside the knob guide on the robot's side: one tick per notch at the knob position that latches it, numbered in mm
(two staggered rows), read against the pointer on the knob.

Axis coordinates: u along the launch axis from the muzzle, v lateral (+y, the robot's side), w normal to the axis.
"""
import math
import numpy as np
from .geometry import (AXIS, MUZZLE, TILT, TILT_DEG, BALL_RADIUS, MUZZLE_GAP, NOTCHES, KNOB_OFFSET_Y, KNOB_SIZE)

NORMAL = (-math.sin(TILT), 0., math.cos(TILT))
FACE_U = -(BALL_RADIUS + MUZZLE_GAP)          # plunger face at the free length
HEAD_REAR_U = FACE_U - .030                   # rear face of the 30 mm plunger head at the free length
BLOCK_FRONT_U = -.150                         # front face of the spring end block
KNOB_U = FACE_U - .0175                       # knob centre (and pointer) at the free length
COIL_TURNS, COIL_RADIUS, WIRE_RADIUS = 6, .0075, .0006
COIL_SEGMENTS, WIRE_SIDES = COIL_TURNS * 20, 6
PLATE_TOP_W = -.014                           # scale plate top: level with the knob guide
# The scale, rack, pawl and pointer stand clear of the tall knob post (seen from above, the post would hide them):
# a render-only arm on the knob carries the pawl and the pointer out over the rack and the plate.
PLATE_V = (.088, .136)
ARM_V, ARM_W = (.056, .0945), -.0055
RACK_V = (.0895, .0935)
TOOTH_H = .0025
PAWL_LENGTH, PAWL_PIVOT_W, PAWL_V = .014, -.0045, (.0900, .0930)
POINTER_V, POINTER_W = (.0890, .1010), -.0090
TICK_V0, TICK_SHORT, TICK_LONG = .0940, .1020, .1135
TEXT_H, TEXT_SQUEEZE = .0060, .70
ROWS_V = (.1035, .1155)                       # top edge (towards the launch line) of the two label rows
INK, PLATE, STEEL, DARK, POINTER = (.03, .03, .03), (.93, .93, .90), (.70, .72, .74), (.10, .11, .13), (.00, .55, .12)


def local(u, v, w):
    """Bench-local point of axis coordinates."""
    p = [m + u * a + w * n for m, a, n in zip(MUZZLE, AXIS, NORMAL)]
    p[1] += v
    return tuple(p)


def tick_u(notch):
    """Pointer (knob centre) position along the axis when the pawl holds `notch`."""
    return KNOB_U - notch


def _mesh(stage, path, points, faces, colour):
    from pxr import Gf, UsdGeom
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([int(i) for f in faces for i in f])
    mesh.CreateSubdivisionSchemeAttr('none'); mesh.CreateDoubleSidedAttr(True)
    arr = np.asarray(points, float)
    mesh.CreateExtentAttr([Gf.Vec3f(*arr.min(0).tolist()), Gf.Vec3f(*arr.max(0).tolist())])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    return mesh


def _box_faces(base):
    b = base
    return [(b, b + 1, b + 2, b + 3), (b + 4, b + 7, b + 6, b + 5), (b, b + 4, b + 5, b + 1),
            (b + 1, b + 5, b + 6, b + 2), (b + 2, b + 6, b + 7, b + 3), (b + 3, b + 7, b + 4, b)]


def _axis_boxes(stage, path, boxes, colour):
    """One mesh of boxes given as (u0, u1, v0, v1, w0, w1) in axis coordinates."""
    points, faces = [], []
    for u0, u1, v0, v1, w0, w1 in boxes:
        base = len(points)
        for w in (w0, w1):
            points += [local(u0, v0, w), local(u1, v0, w), local(u1, v1, w), local(u0, v1, w)]
        faces += _box_faces(base)
    return _mesh(stage, path, points, faces, colour)


# --- coil ----------------------------------------------------------------------------------------------
def coil_faces():
    sides = WIRE_SIDES
    return [(i * sides + j, i * sides + (j + 1) % sides, (i + 1) * sides + (j + 1) % sides, (i + 1) * sides + j)
            for i in range(COIL_SEGMENTS) for j in range(sides)]


def coil_points(travel):
    """Tube along a helix from the plunger head's rear face (at `travel`) to the end block, bench-local frame."""
    front, rear = HEAD_REAR_U - travel, BLOCK_FRONT_U
    t = np.linspace(0., 1., COIL_SEGMENTS + 1)
    phi = 2 * math.pi * COIL_TURNS * t
    a, n, y = np.asarray(AXIS), np.asarray(NORMAL), np.array([0., 1., 0.])
    radial = np.cos(phi)[:, None] * y + np.sin(phi)[:, None] * n
    centre = np.asarray(MUZZLE) + (front + (rear - front) * t)[:, None] * a + COIL_RADIUS * radial
    psi = 2 * math.pi * np.arange(WIRE_SIDES) / WIRE_SIDES
    ring = np.cos(psi)[None, :, None] * radial[:, None, :] + np.sin(psi)[None, :, None] * a[None, None, :]
    return (centre[:, None, :] + WIRE_RADIUS * ring).reshape(-1, 3)


def set_coil(stage, path, travel):
    from pxr import Vt
    stage.GetPrimAtPath(path).GetAttribute('points').Set(Vt.Vec3fArray.FromNumpy(coil_points(travel).astype(np.float32)))


# --- text ----------------------------------------------------------------------------------------------
def _text_rects(text, height):
    """Pixel-font rectangles (x0, y0, x1, y1) of a one-line label, origin at its left baseline, y up."""
    import cv2
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    pad = 2
    image = np.zeros((h + base + 2 * pad, w + 2 * pad), np.uint8)
    cv2.putText(image, text, (pad, pad + h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_8)
    px = height / h
    rects = []
    for r, row in enumerate(image > 127):
        c = 0
        while c < row.size:
            if row[c]:
                s = c
                while c < row.size and row[c]:
                    c += 1
                top = (pad + h - r) * px
                rects.append(((s - pad) * px * TEXT_SQUEEZE, top - px, (c - pad) * px * TEXT_SQUEEZE, top))
            else:
                c += 1
    return rects, w * px * TEXT_SQUEEZE


def _label(text, centre_u, top_v, w):
    """Quads of a label on the scale plate, reading along -u (the robot's left to right), its top towards -v."""
    rects, width = _text_rects(text, TEXT_H)
    points, faces = [], []
    for x0, y0, x1, y1 in rects:
        base = len(points)
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            u = centre_u + width / 2 - x            # reading direction -u
            v = top_v + TEXT_H - y                  # glyph up is -v
            points.append(local(u, v, w))
        faces.append((base, base + 1, base + 2, base + 3))
    return points, faces


TRIGGER_COLOUR = (.42, .16, .62)


def trigger_label(stage, root, pos, table_z=-.040, near=.046, depth=.020, height=.011):
    """Render-only label plate lying on the table on the robot side of the trigger, clear of the cap's overhang seen
    from above: white plate, black `FIRE`, reading the same way as the scale labels."""
    rects, width = _text_rects('FIRE', height)
    half_x, y0, top = width / 2 + .006, pos[1] + near, table_z + .0015
    plate = [(pos[0] + sx * half_x, y, z) for z in (table_z + .0002, top)
             for sx, y in ((-1, y0), (1, y0), (1, y0 + depth), (-1, y0 + depth))]
    _mesh(stage, f'{root}/TriggerLabelPlate', plate, _box_faces(0), (.92, .93, .92))
    top_v = y0 + (depth - height) / 2
    points, faces = [], []
    for x0, y0_, x1, y1 in rects:
        base = len(points)
        for x, y in ((x0, y0_), (x1, y0_), (x1, y1), (x0, y1)):
            points.append((pos[0] + width / 2 - x, top_v + height - y, top + .0003))
        faces.append((base, base + 1, base + 2, base + 3))
    return _mesh(stage, f'{root}/TriggerLabel', points, faces, (.05, .05, .06))


# --- build ---------------------------------------------------------------------------------------------
def build(stage, root, plunger_path, head_local):
    """Author the render-only parts. head_local: bench-local plunger body origin (the head centre at free length)."""
    from pxr import Gf, UsdGeom
    # Coil spring, at the free length; the backend re-shapes it every rendered frame.
    _mesh(stage, root + '/SpringCoil', coil_points(0.), coil_faces(), STEEL)
    # The end block's bracket down to the cradle beam, and the beam's extension under it.
    _axis_boxes(stage, root + '/SpringBlockBracket', [(-.192, -.158, -.004, .004, -.050, -.015),
                                                      (-.200, -.170, -.030, .030, -.060, -.050)], (.42, .46, .49))
    # Scale plate beside the knob guide (beyond the rack), ticks and numbers.
    lo, hi = tick_u(NOTCHES[-1]) - .008, tick_u(NOTCHES[0]) + .008
    _axis_boxes(stage, root + '/NotchScalePlate', [(lo, hi, PLATE_V[0], PLATE_V[1], PLATE_TOP_W - .002, PLATE_TOP_W)], PLATE)
    ink_w = PLATE_TOP_W + .0002
    ticks = [(tick_u(d) - .0006, tick_u(d) + .0006, TICK_V0, TICK_LONG if i % 2 else TICK_SHORT, PLATE_TOP_W, ink_w)
             for i, d in enumerate(NOTCHES)]
    _axis_boxes(stage, root + '/NotchScaleTicks', ticks, INK)
    points, faces = [], []
    for i, d in enumerate(NOTCHES):
        p, f = _label(str(int(round(d * 1000))), tick_u(d), ROWS_V[i % 2], ink_w)
        faces += [tuple(len(points) + j for j in face) for face in f]; points += p
    _mesh(stage, root + '/NotchScaleNumbers', points, faces, INK)
    # Ratchet rack along the plate's inner edge: one tooth per notch. Each vertical face looks back (-u) at the pawl
    # tip that holds that notch; the ramp rises from the next notch towards it.
    points, faces = [], []
    for d in NOTCHES:
        face_u = tick_u(d) - PAWL_LENGTH * math.cos(math.radians(_ENGAGED)) + .0004
        u_low = face_u + .005
        base = len(points)
        for v in RACK_V:
            points += [local(face_u, v, PLATE_TOP_W), local(u_low, v, PLATE_TOP_W), local(face_u, v, PLATE_TOP_W + TOOTH_H)]
        faces += [(base, base + 1, base + 2), (base + 3, base + 5, base + 4), (base, base + 3, base + 4, base + 1),
                  (base, base + 2, base + 5, base + 3), (base + 1, base + 4, base + 5, base + 2)]
    _mesh(stage, root + '/RatchetRack', points, faces, DARK)
    # Pointer on the knob (render-only child of the plunger body) and the pawl in its two positions.
    origin = np.asarray(head_local, float)

    def child_box(name, centre_local, dims, angle_deg, colour):
        c = UsdGeom.Cube.Define(stage, f'{plunger_path}/{name}'); c.CreateSizeAttr(1.)
        c.AddTranslateOp().Set(Gf.Vec3d(*(np.asarray(centre_local, float) - origin)))
        c.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), angle_deg).GetQuat()))
        c.AddScaleOp().Set(Gf.Vec3f(*dims)); c.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
        return c
    child_box('ScaleArm', local(KNOB_U, sum(ARM_V) / 2, ARM_W), (.008, ARM_V[1] - ARM_V[0], .003), -TILT_DEG, DARK)
    child_box('PointerBracket', local(KNOB_U, POINTER_V[0] + .0008, (POINTER_W + ARM_W) / 2), (.008, .0016, ARM_W - POINTER_W), -TILT_DEG, DARK)
    child_box('Pointer', local(KNOB_U, sum(POINTER_V) / 2, POINTER_W), (.0020, POINTER_V[1] - POINTER_V[0], .0012), -TILT_DEG, POINTER)
    child_box('PawlPin', local(KNOB_U, sum(PAWL_V) / 2, PAWL_PIVOT_W), (.003, .006, .003), -TILT_DEG, DARK)
    for name, angle in (('PawlEngaged', _ENGAGED), ('PawlLifted', _LIFTED)):
        a = math.radians(angle)
        mid = local(KNOB_U - PAWL_LENGTH / 2 * math.cos(a), sum(PAWL_V) / 2, PAWL_PIVOT_W - PAWL_LENGTH / 2 * math.sin(a))
        pawl = child_box(name, mid, (PAWL_LENGTH, PAWL_V[1] - PAWL_V[0], .0025), -TILT_DEG - angle, (.80, .80, .82))
        if name == 'PawlEngaged':
            UsdGeom.Imageable(pawl.GetPrim()).MakeInvisible()


# Pawl angles below the launch axis: engaged, its tip in the tooth gap just behind a tooth face; lifted clear.
_ENGAGED = math.degrees(math.asin((PAWL_PIVOT_W - (PLATE_TOP_W + .0012)) / PAWL_LENGTH))
_LIFTED = _ENGAGED - 28.


def set_pawl(stage, plunger_path, engaged):
    from pxr import UsdGeom
    for name, show in (('PawlEngaged', engaged), ('PawlLifted', not engaged)):
        img = UsdGeom.Imageable(stage.GetPrimAtPath(f'{plunger_path}/{name}'))
        img.MakeVisible() if show else img.MakeInvisible()

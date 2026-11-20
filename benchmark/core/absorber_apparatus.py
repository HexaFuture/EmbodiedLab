"""Vibration absorber apparatus (vibration_absorber): a steel strip clamped in a cast-iron vise and driven by an
electrodynamic exciter, the tuned absorber clamped near its tip, four brass weights on a rack and the vibration analyzer.
Built with the chemistry kit's helpers on its root, so the chemistry backend's grasp and button machinery runs it. The
layout and every dimension are in absorber_bench (pxr-free); the render-only looks are in absorber_looks.

Collision geometry is what is drawn; bodies:
- static: the vise (base, jaws, spindle and T-handle) with the strip's clamped root, the exciter (base, trunnion
  uprights, body), the amplifier, the analyzer's case, the rack and its collars;
- kinematic, moved every physics tick by the backend to the computed motion (absorber_bench poses): the strip (SEGMENTS
  collider pieces; the accelerometer on the last), the mount (jaws round the strip, the blade's root clamp, both
  cheeks with their slots and rubber stop linings, the damper cylinder with its valve head), the blade (BLADE_PIECES
  pieces), the damper's yoke (piston rod and slotted plate), the exciter's armature and stinger, and the carrier's
  track (no collider: the blade's tangent at the carrier, the anchor of the carrier's joint);
- dynamic: the carrier (clamp halves round the blade, riser, cross tray with four sockets built of boxes, bumper pins,
  push handle) on a prismatic joint along its track (its drive: a stiff spring to the detent hole the index pin's ball
  sits in, capped, or the clamp's friction alone while the ball is out of a hole: a stiff spring to where the slide
  stands, capped at the friction, its anchor dragged along as the slide is pushed; the backend switches it); the weights (the beam kit's weights, brass, real masses); the key caps (SWEEP on the
  analyzer: the kit's world-jointed cap; the valve keys: caps jointed to the mount, spring-returned).
The carrier clamps the blade and its robot-side pin runs in the yoke's slot: those pairs are filtered (the parts are
assembled into each other). The stop linings' colliders lie STOP_RECESS behind their faces (the rubber's compression is
the model's stop stiffness), so the bumper pins never touch them in PhysX; the carrier's vertical motion is the
model's.
"""
from __future__ import annotations

import math

from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

import absorber_bench as B
import chem_apparatus as C
from chem_apparatus import GLYPHS, ROOT, SEGMENT_EMISSION, SEGMENT_ON, SEGMENTS as SEVEN, _bind, _remember_lit, \
    _render_material, _rigid, _text, collar
from circuit_apparatus import label
from torsion_apparatus import TABLE_TOP, _box, _cylinder, _material, repair_cube_extents, retire_hooke_apparatus, \
    table_top_z

STRIP_PATH = ROOT + '/Strip'
MOUNT = ROOT + '/Mount'
BLADE_PATH = ROOT + '/Blade'
TRACK = ROOT + '/CarrierTrack'
CARRIER = ROOT + '/Carrier'
YOKE = ROOT + '/Yoke'
STINGER = ROOT + '/Stinger'
ANALYZER = ROOT + '/Analyzer'
CABLE = ROOT + '/AccelCable'
CABLE_R = 0.0012                  # the cable's radius
CABLE_END_UV = (-0.070, 0.050)    # the accelerometer socket on the analyzer's case (from its centre, robot frame)
FOOT_CONTACT = (2.0e4, 22.0)      # the weights' feet: compliant contact per point (N/m, N s/m). The kit's cuvette layer
                                  # (1e3 N/m) let a hold-down's ~8 N sink a foot 1.7 mm into its socket floor; here
                                  # ~0.1 mm. Rigid feet rattled: a weight set down in an open socket bounced on the
                                  # floor (52 N spikes, the solver's push-out feeding it) and rang the blade's 30 Hz mode
                                  # at 1-3 mm. A weight pinched in the fingers while on its floor does not come to rest in
                                  # them at this stiffness; the hand lifts it clear first (the reference's pick)
CARRIER_JOINT = C.PHYSICS_ROOT + '/CarrierSlide'
CARRIER_S0_ATTR = 'embodiedlab:absorber_s0_m'   # the joint's zero: the carrier's s when authored
HOLD = ROOT + '/HoldDown'
CLAMP_FRICTION_N, CLAMP_FRICTION_DAMPING = 1.5, 2.0e3
DETENT_STIFFNESS, DETENT_PEAK_N = 1.0e5, 6.5      # the index pin (a spring ball plunger): while its ball sits in a hole
                                                   # the slide's drive is a stiff spring to that hole, its force (with the
                                                   # clamp's friction) at most DETENT_PEAK_N + CLAMP_FRICTION_N; out of
                                                   # the hole it rolls on the flat and the drive is the clamp's friction
                                                   # alone (the backend switches it): static, a spring as stiff to an
                                                   # anchor at most CLAMP_SLIP_M behind the slide, capped at
                                                   # CLAMP_FRICTION_N (a saturated damper alone let the slide walk
                                                   # along its track under PhysX's contact jitter, some 1 mm/s toward
                                                   # the blade's free end). The drive is implicit, so the
                                                   # spring can be as stiff as a real plunger: the blade's pitching at
                                                   # its 30 Hz mode (the carrier's mass rides above the blade) must not
                                                   # rock the carrier out of its hole.
DETENT_CAPTURE_M, DETENT_RELEASE_M = 0.0010, 0.0013  # the ball drops into a hole within this of its centre, leaves it
                                                   # beyond this
CLAMP_SLIP_M = 0.00005                             # out of a hole: the friction anchor trails the slide by at most this
                                                   # (the spring is at the friction cap 15 um from it)
STEADY_ITERATIONS = (32, 16)       # solver position / velocity iterations of the carrier and the weights
STEADY_DEPENETRATION_M_S = 0.5     # their largest depenetration speed
STEEL, DARK, RUBBER = (0.62, 0.63, 0.66), (0.16, 0.17, 0.19), (0.05, 0.05, 0.05)
ORANGE, WHITE, BRASS = (0.92, 0.40, 0.08), (0.95, 0.95, 0.93), (0.80, 0.62, 0.30)
LCD_GLASS = (0.004, 0.018, 0.010)
HIDE_M = 0.004
ANALYZER_DIGIT_H = 0.014
VALVE_DIGIT_H = 0.009
PEAK_CELLS, HZ_CELLS = 5, 3


def segment_path(i):
    return f'{STRIP_PATH}/S{i:02d}'


def blade_path(i):
    return f'{BLADE_PATH}/B{i:02d}'


def weight_path(w):
    return f'{ROOT}/Weight{w}'


def hold_path(n):
    """Socket n's hold-down slide (a rigid body jointed to the carrier)."""
    return f'{HOLD}{n}'


def hold_joint_path(n):
    return f'{C.PHYSICS_ROOT}/HoldDown{n}'


def body_top_local():
    """The top of a seated weight's body above the carrier frame's origin (the beam kit's weight: foot, body)."""
    import beam_apparatus as BA
    return socket_floor_local() + BA.WEIGHT_FOOT[2] + BA.WEIGHT_BODY[1]


def hold_origin_local(n):
    """Socket n's slide frame (closed, at rest) in the carrier frame."""
    return (0.0, B.SOCKET_Y[n - 1], body_top_local() + B.HOLD_REST_DZ)


def kinematic_paths():
    """Every kinematic body the backend moves, in the order of AbsorberBackend's targets."""
    return [segment_path(i) for i in range(B.SEGMENTS)] + [MOUNT] + [blade_path(i) for i in range(B.BLADE_PIECES)] + \
        [TRACK, YOKE, STINGER]


def seat_paths(stage=None):
    return {f'rack:{n}': f'{ROOT}/RackSeat{n}' for n in range(1, len(B.RACK['xs']) + 1)}


def button_layout():
    return {}


def handle_top_local():
    import beam_apparatus as BA
    return BA.handle_top_local()


# --- small helpers ---------------------------------------------------------------------------------------------
def _kinematic(prim):
    """A kinematic body whose contacts are reported (every contact, no force threshold)."""
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)


def _report(prim):
    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)


def _steady(prim):
    """More solver iterations and a bounded depenetration speed for a body riding the vibrating absorber (the carrier
    on its track, the weights in its sockets)."""
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    body.CreateSolverPositionIterationCountAttr().Set(STEADY_ITERATIONS[0])
    body.CreateSolverVelocityIterationCountAttr().Set(STEADY_ITERATIONS[1])
    body.CreateMaxDepenetrationVelocityAttr().Set(STEADY_DEPENETRATION_M_S)


def _fine(prim, offset=0.0005):
    """A tight contact offset (parts that sit close to others)."""
    api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    api.CreateContactOffsetAttr(offset)
    api.CreateRestOffsetAttr(0.0)


def _xform(stage, path, translate=(0, 0, 0)):
    xform = UsdGeom.Xform.Define(stage, path)
    api = UsdGeom.XformCommonAPI(xform)
    api.SetTranslate(Gf.Vec3d(*translate))
    api.SetRotate(Gf.Vec3f(0, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    return xform


def _span_box(stage, path, xs, ys, zs, colour, material=None, collide=True):
    """A box spanning the given (lo, hi) ranges in its parent's frame."""
    lo = [min(r) for r in (xs, ys, zs)]
    hi = [max(r) for r in (xs, ys, zs)]
    size = [b - a for a, b in zip(lo, hi)]
    centre = [(a + b) / 2 for a, b in zip(lo, hi)]
    return _box(stage, path, size, centre, colour, material=material, collide=collide)


def _cyl_y(stage, path, r, y0, y1, x, z, colour, material=None, collide=True):
    """A cylinder along y from y0 to y1 at (x, z) in its parent's frame."""
    xform = _cylinder(stage, path, r, abs(y1 - y0), (x, (y0 + y1) / 2, z), colour, material, collide=collide)
    UsdGeom.XformCommonAPI(xform).SetRotate(Gf.Vec3f(90, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    return xform


def _flat(stage, path, centre, size, z, colour):
    return _box(stage, path, (size[0], size[1], 0.0006), (centre[0], centre[1], z), colour, collide=False)


# --- seven-segment cells ------------------------------------------------------------------------------------------
def cell_rects(cells, point_after, digit_h, centre):
    """{(cell, seg): ((x, y), (dx, dy))} and the point's ((x, y), size) of a horizontal LCD row read from the robot
    (reading direction world -x, digits' tops toward -y), centred on `centre` (parent frame xy)."""
    w, t = digit_h * 0.5, digit_h * 0.14
    pitch, gap = digit_h * 0.74, digit_h * 0.22
    width = cells * pitch + (gap if point_after is not None else 0.0) - (pitch - w)
    out = {}
    for i in range(cells):
        u0 = -width / 2 + i * pitch + (gap if point_after is not None and i >= point_after else 0.0)
        for seg, (fu, fv, horizontal) in SEVEN.items():
            u, v = u0 + fu * w, -digit_h / 2 + fv * digit_h
            out[(i, seg)] = ((centre[0] - u, centre[1] - v), (w, t) if horizontal else (t, digit_h / 2))
    if point_after is None:
        return out, (None, t * 1.3)
    u = -width / 2 + point_after * pitch - (pitch - w) / 2 + gap / 2
    return out, ((centre[0] - u, centre[1] + digit_h / 2 - t / 2), t * 1.3)


def _lcd_row(stage, prefix, cells, point_after, digit_h, centre, z, looks):
    rects, (pxy, ps) = cell_rects(cells, point_after, digit_h, centre)
    for (i, seg), ((x, y), (dx, dy)) in rects.items():
        _bind(_box(stage, f'{prefix}{i}{seg}', (dx, dy, 0.0006), (x, y, z), SEGMENT_ON, collide=False), looks['lit'])
        _remember_lit(stage.GetPrimAtPath(f'{prefix}{i}{seg}'))
    if point_after is not None and point_after < cells:
        _bind(_box(stage, f'{prefix}Point', (ps, ps, 0.0006), (pxy[0], pxy[1], z), SEGMENT_ON, collide=False), looks['lit'])
        _remember_lit(stage.GetPrimAtPath(f'{prefix}Point'))


def show_cells(stage, prefix, text, previous, point=True):
    """Light the cells of a row (prefix + cell + segment) where text differs from previous; the point while a number
    shows."""
    if text == previous:
        return text
    for i, char in enumerate(text):
        if previous is not None and previous[i] == char:
            continue
        on = GLYPHS.get(char, '')
        for seg in SEVEN:
            C._light(stage, f'{prefix}{i}{seg}', seg in on)
    lit = lambda t: t is not None and bool(t.strip()) and 'E' not in t
    if point and (previous is None or lit(text) != lit(previous)):
        C._light(stage, prefix + 'Point', lit(text))
    return text


# --- the vise and the strip ----------------------------------------------------------------------------------------
def _vise(stage, m):
    x0, x1 = B.VISE['x']
    y, hw = B.BEAM_Y, B.VISE['y']
    bx, by, bz = B.VISE['base']
    _box(stage, ROOT + '/ViseBase', (bx, by, bz), ((x0 + x1) / 2 - 0.005, y, bz / 2), DARK, material=m['metal'])
    bottom = B.BEAM_TOP - B.STRIP[2]
    _span_box(stage, ROOT + '/ViseBody', (x0 + 0.015, x1), (y - hw, y + hw), (bz, bottom), DARK, m['metal'])
    _span_box(stage, ROOT + '/ViseJaw', (x0 + 0.015, x1), (y - hw, y + hw), (B.BEAM_TOP, B.BEAM_TOP + 0.022), DARK,
              m['metal'])
    _span_box(stage, ROOT + '/StripRoot', (x0 + 0.018, x1), (y - B.STRIP[1] / 2, y + B.STRIP[1] / 2), (bottom, B.BEAM_TOP),
              (0.78, 0.80, 0.83), m['metal'])
    top = B.BEAM_TOP + 0.022
    _cylinder(stage, ROOT + '/ViseSpindle', 0.007, 0.040, ((x0 + x1) / 2 + 0.008, y, top + 0.020), STEEL, m['metal'])
    _cylinder(stage, ROOT + '/ViseHandle', 0.004, 0.110, ((x0 + x1) / 2 + 0.008, y, top + 0.044), STEEL, m['metal'])
    UsdGeom.XformCommonAPI(stage.GetPrimAtPath(ROOT + '/ViseHandle')).SetRotate(Gf.Vec3f(90, 0, 0),
                                                                              UsdGeom.XformCommonAPI.RotationOrderXYZ)
    for k, dy in enumerate((-1, 1)):
        _cylinder(stage, f'{ROOT}/ViseBolt{k}', 0.006, 0.006, (x0 + 0.004, y + dy * (by / 2 - 0.014), bz + 0.003), STEEL,
                  m['metal'])


def _strip(stage, m):
    UsdGeom.Xform.Define(stage, STRIP_PATH)
    L, w, t = B.STRIP
    for i in range(B.SEGMENTS):
        piece = _xform(stage, segment_path(i), (B.CLAMP_X + B.segment_x(i), B.BEAM_Y, B.BEAM_TOP - t / 2))
        box = _box(stage, segment_path(i) + '/Box', (L / B.SEGMENTS + B.SEGMENT_OVERLAP, w, t), (0, 0, 0),
                   (0.78, 0.80, 0.83), material=m['metal'])
        _fine(box.GetPrim(), 0.002)
        _kinematic(piece.GetPrim())
    # the accelerometer on the last piece: a hexagonal block (collider: a cylinder of its across-corners)
    af, ah, ax = B.ACCEL
    last = segment_path(B.SEGMENTS - 1)
    dx = ax - (B.CLAMP_X + B.segment_x(B.SEGMENTS - 1))
    _cylinder(stage, last + '/Accel', af / 2, ah, (dx, 0.0, t / 2 + ah / 2), (0.20, 0.21, 0.23), m['metal'])


# --- the mount (mount frame) -------------------------------------------------------------------------------------
def _mount(stage, m, gap, looks):
    _xform(stage, MOUNT, (B.MOUNT_X, B.BEAM_Y, B.BEAM_TOP))
    t, hw, L = B.STRIP[2], B.MOUNT_W / 2, B.MOUNT_LEN / 2
    jaw = (0.30, 0.31, 0.34)
    top = B.BLADE_H - M_BLADE_T / 2
    _span_box(stage, MOUNT + '/LowerJaw', (-L, L), (-hw, hw), (-t - B.JAW_T, -t), jaw, m['metal'])
    _span_box(stage, MOUNT + '/UpperJaw', (-L, L), (-hw, hw), (0.0, top), jaw, m['metal'])
    for k, s in enumerate((-1, 1)):
        _span_box(stage, f'{MOUNT}/Side{k}', (-L, L), (s * (B.STRIP[1] / 2 + 0.0005), s * hw), (-t, 0.0), jaw, m['metal'])
    # the blade's root under its clamp plate
    r0, r1 = B.ROOT_X, B.ROOT_X + B.ROOT_CLAMP
    _span_box(stage, MOUNT + '/BladeRoot', (r0, r1), (-M_BLADE_W / 2, M_BLADE_W / 2), (top, top + M_BLADE_T),
              (0.55, 0.56, 0.60), m['metal'])
    _span_box(stage, MOUNT + '/ClampPlate', (r0, r1 + 0.004), (-0.011, 0.011), (top + M_BLADE_T, top + M_BLADE_T + 0.006),
              jaw, m['metal'])
    # cheeks: upper and lower bars, the rubber linings of the slot, the filler at the mount end, spacers
    gap_lo, gap_hi = B.BLADE_H - B.PIN_R - gap - B.STOP_RECESS, B.BLADE_H + B.PIN_R + gap + B.STOP_RECESS
    c0, c1 = B.CHEEK_X
    for k, s in enumerate((-1, 1)):
        ys = (s * B.CHEEK_Y[0], s * B.CHEEK_Y[1])
        _span_box(stage, f'{MOUNT}/CheekTop{k}', (c1, c0), ys, (gap_hi + B.RUBBER_T, B.CHEEK_Z[1]), jaw, m['metal'])
        _span_box(stage, f'{MOUNT}/CheekBottom{k}', (c1, c0), ys, (B.CHEEK_Z[0], gap_lo - B.RUBBER_T), jaw, m['metal'])
        _span_box(stage, f'{MOUNT}/StopTop{k}', (c1, B.ROOT_X - 0.012), ys, (gap_hi, gap_hi + B.RUBBER_T), RUBBER,
                  m['rubber'])
        _span_box(stage, f'{MOUNT}/StopBottom{k}', (c1, B.ROOT_X - 0.012), ys, (gap_lo - B.RUBBER_T, gap_lo), RUBBER,
                  m['rubber'])
        _span_box(stage, f'{MOUNT}/CheekFill{k}', (B.ROOT_X - 0.012, c0), ys, (gap_lo - B.RUBBER_T, gap_hi + B.RUBBER_T),
                  jaw, m['metal'])
        _span_box(stage, f'{MOUNT}/Spacer{k}', (-L, L), (s * hw, s * B.CHEEK_Y[0]), (B.CHEEK_Z[0], B.CHEEK_Z[1]), jaw,
                  m['metal'])
    # the damper: bracket, cylinder, valve head with the LCD (keys: caps of their own, jointed to the mount)
    cx, cy = B.CYL_XY
    _span_box(stage, MOUNT + '/DamperBracket', (L - 0.010, cx), (hw, cy), (B.CYL_Z[0], B.CYL_Z[0] + 0.006), jaw,
              m['metal'])
    _cylinder(stage, MOUNT + '/DamperCylinder', B.CYL_R, B.CYL_Z[1] - B.CYL_Z[0], (cx, cy, (B.CYL_Z[0] + B.CYL_Z[1]) / 2),
              (0.70, 0.12, 0.10), m['metal'])
    (hx0, hx1), (hy0, hy1), (hz0, hz1) = B.HEAD
    _span_box(stage, MOUNT + '/ValveHead', (hx0, hx1), (hy0, hy1), (hz0, hz1), (0.13, 0.14, 0.16), m['metal'])
    vx, vy = B.VALVE_LCD_XY
    _bind(_box(stage, MOUNT + '/ValveGlass', (0.008, 0.012, 0.0006), (vx, vy, hz1 + 0.0003), LCD_GLASS, collide=False),
          looks['glass'])
    _lcd_row(stage, MOUNT + '/Valve', 1, None, VALVE_DIGIT_H, (vx, vy), hz1 + 0.0007, looks)
    for bid, (x, y) in B.KEY_XY.items():
        _cylinder(stage, f'{MOUNT}/{bid.split(":")[1].title()}Housing', C.CAP_R + 0.002, C.CAP_LIFT - 0.001,
                  (x, y, hz1 + (C.CAP_LIFT - 0.001) / 2), (0.25, 0.25, 0.27), collide=False)
    _text(stage, MOUNT + '/ValveLabel', 'VALVE', (vx, vy - 0.009), hz1 + 0.0004, 0.0035, colour=WHITE)
    _kinematic(stage.GetPrimAtPath(MOUNT))


M_BLADE_T, M_BLADE_W = 0.0005, 0.012


def _blade(stage, m):
    UsdGeom.Xform.Define(stage, BLADE_PATH)
    piece = B.BLADE_LEN / B.BLADE_PIECES
    for i in range(B.BLADE_PIECES):
        s = B.blade_s(i)
        x = B.MOUNT_X + B.ROOT_X - s
        xform = _xform(stage, blade_path(i), (x, B.BEAM_Y, B.BEAM_TOP + B.BLADE_H))
        box = _box(stage, blade_path(i) + '/Box', (piece + 0.0002, M_BLADE_W, M_BLADE_T), (0, 0, 0), (0.55, 0.56, 0.60),
                   material=m['metal'])
        _fine(box.GetPrim(), 0.0005)
        _kinematic(xform.GetPrim())
        # detent holes (index pin) and their lengths, printed beside them
        for mm in B.DETENTS_MM:
            d = mm / 1000.0 - s
            if -piece / 2 <= d < piece / 2:
                _cylinder(stage, f'{blade_path(i)}/Hole{mm}', 0.0011, 0.0002, (-d, -0.003, M_BLADE_T / 2 + 0.0001),
                          (0.04, 0.04, 0.05), collide=False)
                label(stage, f'{blade_path(i)}/Mark{mm}', str(mm), (-d, 0.0025), M_BLADE_T / 2 + 0.0002, height_m=0.0028,
                      align='centre', colour=(0.05, 0.05, 0.06), up=(1.0, 0.0))


def _yoke(stage, m):
    _xform(stage, YOKE, (B.MOUNT_X, B.BEAM_Y, B.BEAM_TOP))
    ys = B.YOKE_Y
    x_slot0, x_far = B.ROOT_X - 0.015, B.YOKE_X[1]
    z0, z1 = B.YOKE_Z
    lo, hi = B.BLADE_H - B.YOKE_SLOT, B.BLADE_H + B.YOKE_SLOT
    grey = (0.62, 0.63, 0.66)
    _span_box(stage, YOKE + '/Upper', (x_far, x_slot0), ys, (hi, z1), grey, m['metal'])
    _span_box(stage, YOKE + '/Lower', (x_far, x_slot0), ys, (z0, lo), grey, m['metal'])
    _span_box(stage, YOKE + '/Arm', (x_slot0, B.YOKE_X[0]), ys, (z0, z1), grey, m['metal'])
    cx, cy = B.CYL_XY
    _cylinder(stage, YOKE + '/Rod', B.ROD_R, B.CYL_Z[0] + 0.006 - z1, (cx, cy, (z1 + B.CYL_Z[0] + 0.006) / 2), STEEL,
              m['metal'])
    _span_box(stage, YOKE + '/RodFoot', (cx - 0.005, cx + 0.005), (cy - 0.005, cy + 0.005), (z1 - 0.001, z1 + 0.003),
              grey, m['metal'])
    _kinematic(stage.GetPrimAtPath(YOKE))


def _stinger(stage, m):
    _xform(stage, STINGER, (B.STINGER_X, B.BEAM_Y, 0.0))
    r, z0, h = B.ARMATURE
    _cylinder(stage, STINGER + '/Armature', r, h, (0, 0, z0 + h / 2), (0.70, 0.71, 0.73), m['metal'])
    top = B.BEAM_TOP - B.STRIP[2]
    cz = B.STINGER_CLAMP[2]
    _cylinder(stage, STINGER + '/Rod', B.STINGER_R, top - cz - (z0 + h), (0, 0, (z0 + h + top - cz) / 2), STEEL, m['metal'])
    _box(stage, STINGER + '/Clamp', B.STINGER_CLAMP, (0, 0, top - cz / 2), DARK, material=m['metal'])
    _kinematic(stage.GetPrimAtPath(STINGER))


def _exciter(stage, m):
    (x, y), r, (z0, z1), (bx, by, bz) = B.SHAKER['xy'], B.SHAKER['r'], B.SHAKER['z'], B.SHAKER['base']
    _box(stage, ROOT + '/ShakerBase', (bx, by, bz), (x, y, bz / 2), DARK, material=m['metal'])
    for k, s in enumerate((-1, 1)):
        _span_box(stage, f'{ROOT}/ShakerTrunnion{k}', (x - 0.016, x + 0.016), (y + s * (r + 0.003), y + s * (r + 0.011)),
                  (bz, (z0 + z1) / 2 + 0.012), DARK, m['metal'])
    _cylinder(stage, ROOT + '/ShakerBody', r, z1 - z0, (x, y, (z0 + z1) / 2), (0.10, 0.22, 0.42), m['metal'])
    ax, ay = B.AMPLIFIER['xy']
    sx, sy, sz = B.AMPLIFIER['size']
    _box(stage, ROOT + '/Amplifier', (sx, sy, sz), (ax, ay, sz / 2), (0.75, 0.75, 0.73), material=m['metal'])


# --- the carrier (carrier frame) --------------------------------------------------------------------------------
def _carrier(stage, m, L):
    x0 = B.MOUNT_X + B.ROOT_X - L
    _xform(stage, TRACK, (B.MOUNT_X + B.ROOT_X, B.BEAM_Y, B.BEAM_TOP + B.BLADE_H))
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(TRACK)).CreateKinematicEnabledAttr(True)
    _xform(stage, CARRIER, (x0, B.BEAM_Y, B.BEAM_TOP + B.BLADE_H))
    bx, by, bz = B.CLAMP_BLOCK
    alu = (0.70, 0.71, 0.74)
    _span_box(stage, CARRIER + '/ClampUpper', (-bx / 2, bx / 2), (-by / 2, by / 2), (M_BLADE_T / 2, bz / 2), alu, m['metal'])
    _span_box(stage, CARRIER + '/ClampLower', (-bx / 2, bx / 2), (-by / 2, by / 2), (-bz / 2, -M_BLADE_T / 2), alu,
              m['metal'])
    rx, ry = B.RISER
    _span_box(stage, CARRIER + '/Riser', (-rx / 2, rx / 2), (-ry / 2, ry / 2), (bz / 2, B.TRAY_Z), alu, m['metal'])
    tx, ty, tt = B.TRAY
    z0, z1 = B.TRAY_Z, B.TRAY_Z + tt
    _span_box(stage, CARRIER + '/TrayBase', (-tx / 2, tx / 2), (-ty / 2, ty / 2), (z0, z1), alu, m['tray'])
    # the sockets, built of boxes: a lower layer with the wells (WELL_HALF), an upper layer with the mouths
    # (MOUTH_HALF), and between them 45-degree chamfers (slabs turned 45 degrees, their inner faces the chamfer planes)
    top = z1 + B.SOCKET_H
    wh, mh = B.WELL_HALF, B.MOUTH_HALF
    depth = mh - wh
    mid = top - depth
    for layer, half, (za, zb) in (('Low', wh, (z1, mid)), ('High', mh, (mid, top))):
        for k, s in enumerate((-1, 1)):
            _span_box(stage, f'{CARRIER}/Rail{layer}{k}', (s * half, s * tx / 2), (-ty / 2, ty / 2), (za, zb), alu,
                      m['tray'])
        edges = [-ty / 2] + [y for c in B.SOCKET_Y for y in (c - half, c + half)] + [ty / 2]
        for k in range(0, len(edges), 2):
            _span_box(stage, f'{CARRIER}/Wall{layer}{k // 2}', (-half, half), (edges[k], edges[k + 1]), (za, zb), alu,
                      m['tray'])
    face, back = depth * math.sqrt(2), 0.0015
    off = (wh + mh) / 2 + back / (2 * math.sqrt(2))
    zc = top - depth / 2 - back / (2 * math.sqrt(2))
    for n, c in enumerate(B.SOCKET_Y):
        for k, sx in enumerate((1, -1)):
            _box(stage, f'{CARRIER}/ChamferX{n}{k}', (back, 2 * mh, face), (sx * off, c, zc), alu,
                 rotate_xyz=(0.0, 45.0 if sx > 0 else 135.0, 0.0), material=m['tray'])
        for k, sy in enumerate((1, -1)):
            _box(stage, f'{CARRIER}/ChamferY{n}{k}', (2 * mh, back, face), (0.0, c + sy * off, zc), alu,
                 rotate_xyz=(-45.0 if sy > 0 else -135.0, 0.0, 0.0), material=m['tray'])
    # the hold-downs' rails: a riser clear of the weight's body (joined to the tray by a low web under the body) and a
    # rail under each slide: the notch floor along its length, the high section at its back, the slope between
    zb = body_top_local() + B.HOLD_REST_DZ
    (rx0, rx1), _, depth = B.HOLD_RUNNER
    high, floor = zb + B.HOLD_HIGH - depth, zb + B.HOLD_FLOOR - depth
    slope = math.radians(B.HOLD_SLOPE_DEG)
    run = (high - floor) / math.tan(slope)
    r = (rx1 - rx0) / 2
    # the slope's top: its foot where the closed runner pin, on the floor, just touches it (no play when closed)
    xa = (rx0 + rx1) / 2 - r * (1 - math.cos(slope)) / math.sin(slope) - run
    th = 0.002
    for n, c in enumerate(B.SOCKET_Y, start=1):
        hy = B.HOLD_RAIL_HALF_Y
        _span_box(stage, f'{CARRIER}/HoldWeb{n}', (B.HOLD_RISER_X[1], -tx / 2), (c - hy, c + hy), (z0, top), alu,
                  m['tray'])
        _span_box(stage, f'{CARRIER}/HoldRiser{n}', B.HOLD_RISER_X, (c - hy, c + hy), (z0, floor - 0.0035), alu,
                  m['tray'])
        _span_box(stage, f'{CARRIER}/HoldRail{n}', B.HOLD_RAIL_X, (c - hy, c + hy), (floor - 0.0035, floor), alu,
                  m['tray'])
        _span_box(stage, f'{CARRIER}/HoldRailHigh{n}', (B.HOLD_RAIL_X[0], xa), (c - hy, c + hy), (floor, high), alu,
                  m['tray'])
        # the slope: a slab whose top face runs from the high section's front edge down to the floor
        cx = xa + run / 2 - th / 2 * math.sin(slope)
        cz = (high + floor) / 2 - th / 2 * math.cos(slope)
        _box(stage, f'{CARRIER}/HoldRailSlope{n}', (math.hypot(run, high - floor), 2 * hy, th), (cx, c, cz), alu,
             rotate_xyz=(0.0, B.HOLD_SLOPE_DEG, 0.0), material=m['tray'])
    # bumper pins (along y) and the push handle at the tray's robot end
    for k, (y0, y1) in enumerate(((B.PIN_Y[0], -by / 2), (by / 2, B.PIN_Y[1]))):
        _cyl_y(stage, f'{CARRIER}/Pin{k}', B.PIN_R, y0, y1, 0.0, 0.0, STEEL, m['metal'])
    hx, hy, hh = B.HANDLE
    _span_box(stage, CARRIER + '/HandleTab', (-hx / 2, hx / 2), (ty / 2, B.HANDLE_Y + hy / 2), (z0, z1), alu, m['tray'])
    _span_box(stage, CARRIER + '/Handle', (-hx / 2, hx / 2), (B.HANDLE_Y - hy / 2, B.HANDLE_Y + hy / 2), (z1, z0 + hh),
              ORANGE, m['rubber'])
    prim = stage.GetPrimAtPath(CARRIER)
    _rigid(prim, B.CARRIER_MASS, (0.0, 0.0, B.TRAY_Z))
    _steady(prim)
    _report(prim)
    pairs = UsdPhysics.FilteredPairsAPI.Apply(prim)
    pairs.CreateFilteredPairsRel().SetTargets([Sdf.Path(blade_path(i)) for i in range(B.BLADE_PIECES)] + [Sdf.Path(YOKE)])


def _hold_downs(stage, m, x0):
    """The slides of the hold-downs, closed over their empty sockets (as the bench is left: the runner on the notch
    floor), each its own rigid body: two prongs with lead-in fronts, the bar joining them, the push tab at the back and
    the runner under it."""
    t = B.HOLD_T
    steel, tab = (0.20, 0.21, 0.23), (0.80, 0.08, 0.06)
    ramp_l, ramp_h = B.HOLD_RAMP
    px0, px1 = B.HOLD_PRONG_X
    a = math.atan2(ramp_h, ramp_l)
    paths = []
    for n in range(1, len(B.SOCKET_Y) + 1):
        ox, oy, oz = hold_origin_local(n)
        path = hold_path(n)
        _xform(stage, path, (x0 + ox, B.BEAM_Y + oy, B.BEAM_TOP + B.BLADE_H + oz + B.HOLD_FLOOR))
        for k, sy in enumerate((-1, 1)):
            y0, y1 = sorted((sy * B.HOLD_PRONG_Y[0], sy * B.HOLD_PRONG_Y[1]))
            _span_box(stage, f'{path}/Prong{k}', (px0, px1 - ramp_l), (y0, y1), (0.0, t), steel, m['ptfe'])
            _span_box(stage, f'{path}/Nose{k}', (px1 - ramp_l, px1), (y0, y1), (ramp_h, t), steel, m['ptfe'])
            # the ramp: a slab whose underside runs from the prong's underside up to the nose's
            th = 0.0015
            cx = px1 - ramp_l / 2 - th / 2 * math.sin(a)
            cz = ramp_h / 2 + th / 2 * math.cos(a)
            _box(stage, f'{path}/Ramp{k}', (math.hypot(ramp_l, ramp_h), y1 - y0, th), (cx, (y0 + y1) / 2, cz), steel,
                 rotate_xyz=(0.0, -math.degrees(a), 0.0), material=m['ptfe'])
        hy = B.HOLD_PRONG_Y[1]
        # the bar and bridge run on the rail on a PTFE-lined underside
        _span_box(stage, f'{path}/Bridge', (px0 - 0.004, px0), (-hy, hy), (0.0, t), steel, m['ptfe'])
        _span_box(stage, f'{path}/Bar', (B.HOLD_BAR_X[0], px0 - 0.004), (-B.HOLD_BAR_HALF_Y, B.HOLD_BAR_HALF_Y), (0.0, t),
                  steel, m['ptfe'])
        (tx0, tx1), thy, ttop = B.HOLD_TAB
        # the tab: red acetal, slippery against the finger that pushes it (the slide must rise freely along it)
        _span_box(stage, f'{path}/Tab', (tx0, tx1), (-thy, thy), (t, ttop), tab, m['ptfe'])
        # the runner: a PTFE cam-follower pin across the bar's back, its underside HOLD_RUNNER's depth below the bar
        (rx0, rx1), ry, depth = B.HOLD_RUNNER
        r = (rx1 - rx0) / 2
        _cyl_y(stage, f'{path}/Runner', r, -ry, ry, (rx0 + rx1) / 2, r - depth, steel, m['ptfe'])
        prim = stage.GetPrimAtPath(path)
        _rigid(prim, B.HOLD_MASS, (-0.02, 0.0, t / 2))
        _steady(prim)
        _report(prim)
        paths.append(path)
    return paths


def socket_floor_local():
    """The sockets' floor above the carrier frame's origin (the blade line)."""
    return B.TRAY_Z + B.TRAY[2]


def _foot_contact():
    """FOOT_CONTACT; debugging may set EMBODIEDLAB_ABSORBER_FOOT to 'k,c' or 'rigid'."""
    import os
    v = os.environ.get('EMBODIEDLAB_ABSORBER_FOOT', '').strip()
    if not v:
        return FOOT_CONTACT
    if v == 'rigid':
        return None
    k, c = (float(x) for x in v.split(','))
    return (k, c)


# --- analyzer -------------------------------------------------------------------------------------------------------
def _analyzer(stage, m, looks):
    (ax, ay), (sx, sy, sz) = B.ANALYZER['xy'], B.ANALYZER['size']
    _box(stage, ANALYZER, (sx, sy, sz), (ax, ay, sz / 2), (0.80, 0.80, 0.78), material=m['metal'])
    gx, gy = C._xy((ax, ay), *B.SCREEN_UV)
    gw, gh = B.SCREEN
    _bind(_box(stage, ROOT + '/AnalyzerGlass', (gw, gh, 0.0006), (gx, gy, sz + 0.0003), LCD_GLASS, collide=False),
          looks['glass'])
    z = sz + 0.0008
    row_v = {'peak': gy - 0.010, 'hz': gy + 0.013}           # world y: the PEAK row farther from the robot
    _lcd_row(stage, ROOT + '/AnalyzerPeak', PEAK_CELLS, 2, ANALYZER_DIGIT_H, (gx + 0.010, row_v['peak']), z, looks)
    _lcd_row(stage, ROOT + '/AnalyzerHz', HZ_CELLS, 2, ANALYZER_DIGIT_H * 0.8, (gx + 0.010, row_v['hz']), z, looks)
    dim = (0.55, 0.75, 0.60)
    for name, text, v in (('PeakLabel', 'PEAK', row_v['peak']), ('HzLabel', 'FREQ', row_v['hz'])):
        _text(stage, f'{ROOT}/Analyzer{name}', text, (gx + 0.050, v), z, 0.0055, colour=dim)
    _text(stage, ROOT + '/AnalyzerPeakUnit', 'mm', (gx - 0.048, row_v['peak']), z, 0.0055, colour=dim)
    _text(stage, ROOT + '/AnalyzerHzUnit', 'Hz', (gx - 0.048, row_v['hz']), z, 0.0055, colour=dim)
    _bind(_box(stage, ROOT + '/AnalyzerRun', (0.006, 0.004, 0.0006), (gx - 0.050, gy - 0.020, z), SEGMENT_ON,
               collide=False), looks['lit'])
    _remember_lit(stage.GetPrimAtPath(ROOT + '/AnalyzerRun'))
    _text(stage, ROOT + '/AnalyzerRunLabel', 'RUN', (gx - 0.040, gy - 0.020), z, 0.004, colour=dim)
    bx, by = C._xy((ax, ay), *B.SWEEP_UV)
    C._button(stage, B.SWEEP_ID, (bx, by), sz, 'round', (0.15, 0.62, 0.25), None, m)
    _text(stage, ROOT + '/AnalyzerSweepLabel', 'SWEEP', (bx, by + 0.021), sz + 0.0004, 0.0055, colour=(0.15, 0.15, 0.17))
    _text(stage, ROOT + '/AnalyzerName', 'VIBRATION ANALYZER', C._xy((ax, ay), -0.030, -0.052), sz + 0.0004, 0.006,
          colour=(0.15, 0.15, 0.17))


def _valve_keys(stage, m):
    """The valve's - and + caps above the valve head (their joints to the mount: author_joints)."""
    UsdGeom.Xform.Define(stage, ROOT + '/Buttons')
    hz1 = B.HEAD[2][1]
    out = {}
    for bid, (x, y) in B.KEY_XY.items():
        path = C.button_path(bid)
        cap = UsdGeom.Xform.Define(stage, path)
        UsdGeom.XformCommonAPI(cap).SetTranslate(Gf.Vec3d(B.MOUNT_X + x, B.BEAM_Y + y,
                                                          B.BEAM_TOP + hz1 + C.CAP_LIFT + C.CAP_H / 2))
        colour = (0.15, 0.30, 0.65)
        _cylinder(stage, path + '/Cap', C.CAP_R, C.CAP_H, (0, 0, 0), colour, m['rubber'])
        _cylinder(stage, path + '/Pad', C.PAD_R, C.PAD_H, (0, 0, C.CAP_H / 2 + C.PAD_H / 2), colour, m['rubber'])
        _text(stage, path + '/Text', '-' if bid.endswith('down') else '+', (0.0, 0.0), C.CAP_H / 2 + C.PAD_H + 0.0004,
              0.010, colour=WHITE)
        _rigid(cap.GetPrim(), C.CAP_MASS_KG)
        _report(cap.GetPrim())
        out[bid] = path
    return out


# --- build -----------------------------------------------------------------------------------------------------------
def build(stage, config):
    """config: {'weights': {w: label}, 'masses_kg': {w: kg}, 'weight_seats': {w: 'rack:<n>'}, 'carrier_mm': the detent
    the carrier starts on, 'gap_m': the stops' gap}. Returns objects (weights), buttons (SWEEP, valve keys), seats
    (rack collars)."""
    import beam_apparatus as BA
    retire_hooke_apparatus(stage)
    repair_cube_extents(stage)
    z0 = table_top_z(stage)
    root = UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(0, 0, z0))
    m = {'rubber': _material(stage, '/World/PhysicsMaterials/AbsorberRubber', 0.8, 0.7),
         'metal': _material(stage, '/World/PhysicsMaterials/AbsorberMetal', 0.40, 0.30),
         'tray': _material(stage, '/World/PhysicsMaterials/AbsorberTray', 0.30, 0.25),
         'pmma': _material(stage, '/World/PhysicsMaterials/AbsorberPMMA', 0.25, 0.20),
         'cell': _material(stage, '/World/PhysicsMaterials/AbsorberCell', 0.25, 0.20, compliant=_foot_contact()),
         'seat': _material(stage, '/World/PhysicsMaterials/AbsorberSeat', 0.20, 0.15),
         'ptfe': _material(stage, '/World/PhysicsMaterials/AbsorberHoldPad', *B.HOLD_PAD_MU)}
    PhysxSchema.PhysxMaterialAPI.Apply(m['ptfe'].GetPrim()).CreateFrictionCombineModeAttr('min')
    looks = {'glass': _render_material(stage, ROOT + '/Looks/LcdGlass', LCD_GLASS, matte=True),
             'lit': _render_material(stage, ROOT + '/Looks/LcdSegment', SEGMENT_ON, emission=SEGMENT_EMISSION, matte=True)}
    _vise(stage, m)
    _strip(stage, m)
    _mount(stage, m, float(config['gap_m']), looks)
    _blade(stage, m)
    _yoke(stage, m)
    _stinger(stage, m)
    _exciter(stage, m)
    _carrier(stage, m, int(config['carrier_mm']) / 1000.0)
    _hold_downs(stage, m, B.MOUNT_X + B.ROOT_X - int(config['carrier_mm']) / 1000.0)
    _analyzer(stage, m, looks)
    buttons = {B.SWEEP_ID: C.button_path(B.SWEEP_ID)}
    buttons.update(_valve_keys(stage, m))
    # the rack and the weights (the beam kit's weights: foot, body, white rim handle)
    bx, by, bz = B.RACK['base']
    xs = B.RACK['xs']
    _box(stage, ROOT + '/Rack', (bx, by, bz), (sum(xs) / len(xs), B.RACK['y'], bz / 2), (0.12, 0.13, 0.15),
         material=m['rubber'])
    seats = {}
    for n, x in enumerate(xs, start=1):
        seats[f'rack:{n}'] = str(collar(stage, f'{ROOT}/RackSeat{n}', (x, B.RACK['y']), bz, (0.03, 0.16, 0.20),
                                        m['seat']).GetPath())
    objects = {}
    for w, sid in config['weight_seats'].items():
        n = int(sid.split(':')[1])
        path = BA._weight(stage, w, config['weights'][w], (xs[n - 1], B.RACK['y']), bz + C.SEAT_FLOOR + 0.0003,
                          config['masses_kg'][w], m)
        _steady(stage.GetPrimAtPath(path))
        _report(stage.GetPrimAtPath(path))
        objects[w] = path
    # the accelerometer's cable (render only, a thin black rubber tube; the backend redraws it as the tip moves)
    import visual_kit as K
    cable = K.material(stage, ROOT + '/Looks/Cable', (0.03, 0.03, 0.03), roughness=0.6)
    tip = (B.ACCEL[2], B.BEAM_Y, B.BEAM_TOP + B.STRIP[2] + B.ACCEL[1])
    end = C._xy(B.ANALYZER['xy'], *CABLE_END_UV)
    K.define(stage, CABLE, K.tube(B.cable_points(tip, (end[0], end[1], 0.030)), CABLE_R, sides=8), cable)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(stage.GetPrimAtPath(TABLE_TOP))
    contact.CreateContactOffsetAttr(0.001)
    contact.CreateRestOffsetAttr(0.0)
    show_cells(stage, ROOT + '/AnalyzerPeak', ' ' * PEAK_CELLS, None)
    show_cells(stage, ROOT + '/AnalyzerHz', ' ' * HZ_CELLS, None)
    show_cells(stage, MOUNT + '/Valve', '0', None, point=False)
    C._light(stage, ROOT + '/AnalyzerRun', False)
    C.finish_looks(stage)
    try:
        import absorber_looks                          # render-only looks over the drawn colliders (build time only)
        absorber_looks.apply(stage, config)
    except ImportError:
        pass
    return {'objects': objects, 'buttons': buttons, 'seats': seats}


def _hold_joint(stage, n):
    """Socket n's slide on the carrier: a D6 joint, free along x within the stroke and along z within the spring's
    travel (a spring drive pressing the slide down onto its rail, or onto the weight: the preload), every other axis
    locked. The slide collides with the carrier: its runner rides the rail (held where it is left by the contact's
    friction on the high section, latched in the notch when closed); the joint's lower z stop (HOLD_BACKSTOP under
    the datum, below the notch floor) is only a backstop."""
    joint = UsdPhysics.Joint.Define(stage, hold_joint_path(n))
    joint.CreateBody0Rel().SetTargets([Sdf.Path(CARRIER)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(hold_path(n))])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*hold_origin_local(n)))
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateCollisionEnabledAttr(True)
    prim = joint.GetPrim()
    for dof, (lo, hi) in (('transX', (-B.HOLD_STROKE, 0.0)), ('transZ', (-B.HOLD_BACKSTOP, B.HOLD_LIFT)), ('transY', (1.0, -1.0)),
                          ('rotX', (1.0, -1.0)), ('rotY', (1.0, -1.0)), ('rotZ', (1.0, -1.0))):
        limit = UsdPhysics.LimitAPI.Apply(prim, dof)
        limit.CreateLowAttr(lo)
        limit.CreateHighAttr(hi)
    k, target, damping = B.HOLD_SPRING
    spring = UsdPhysics.DriveAPI.Apply(prim, 'transZ')
    spring.CreateTypeAttr('force')
    spring.CreateStiffnessAttr(k)
    spring.CreateDampingAttr(damping)
    spring.CreateTargetPositionAttr(target)
    spring.CreateTargetVelocityAttr(0.0)
    # the slide starts closed (over its empty socket), at the joint's origin


def author_joints(stage, buttons):
    """The carrier's prismatic joint along its track (end stops TRAVEL_MARGIN_M beyond the first and last detent; its
    drive starts as the index pin's spring to the starting hole, see DETENT_STIFFNESS), SWEEP's world-jointed cap (the kit's) and the valve keys' caps,
    spring-returned on vertical joints to the mount (preload KEY_PRELOAD_M)."""
    UsdGeom.Scope.Define(stage, C.PHYSICS_ROOT)
    cache = UsdGeom.XformCache()
    track = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(TRACK)).ExtractTranslation()
    carrier = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(CARRIER)).ExtractTranslation()
    joint = UsdPhysics.PrismaticJoint.Define(stage, CARRIER_JOINT)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(TRACK)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(CARRIER)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(carrier - track))
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateAxisAttr('X')
    s0 = B.MOUNT_X + B.ROOT_X - float(carrier[0])                  # the starting detent, m
    lo = B.DETENTS_MM[0] / 1000.0 - B.TRAVEL_MARGIN_M
    hi = B.DETENTS_MM[-1] / 1000.0 + B.TRAVEL_MARGIN_M
    joint.CreateLowerLimitAttr(-(hi - s0))                         # track x points to the mount: s grows toward -x
    joint.CreateUpperLimitAttr(s0 - lo)
    joint.GetPrim().CreateAttribute(CARRIER_S0_ATTR, Sdf.ValueTypeNames.Double).Set(s0)   # joint position = s0 - s
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
    drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(DETENT_STIFFNESS)                    # the ball starts in the starting detent's hole
    drive.CreateDampingAttr(CLAMP_FRICTION_DAMPING)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateMaxForceAttr(DETENT_PEAK_N + CLAMP_FRICTION_N)
    for n in range(1, len(B.SOCKET_Y) + 1):
        _hold_joint(stage, n)
    C.author_joints(stage, {B.SWEEP_ID: buttons[B.SWEEP_ID]})
    mount = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(MOUNT)).ExtractTranslation()
    for bid in B.KEY_XY:
        path = buttons[bid]
        cap = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation()
        key = UsdPhysics.PrismaticJoint.Define(stage, C.joint_path(bid))
        key.CreateBody0Rel().SetTargets([Sdf.Path(MOUNT)])
        key.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        key.CreateLocalPos0Attr(Gf.Vec3f(cap - mount))
        key.CreateLocalRot0Attr(Gf.Quatf(1))
        key.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        key.CreateLocalRot1Attr(Gf.Quatf(1))
        key.CreateAxisAttr('Z')
        key.CreateLowerLimitAttr(-C.TRAVEL_M)
        key.CreateUpperLimitAttr(0.0)
        spring = UsdPhysics.DriveAPI.Apply(key.GetPrim(), 'linear')
        spring.CreateTypeAttr('force')
        spring.CreateStiffnessAttr(C.SPRING_N_PER_M)
        spring.CreateDampingAttr(C.SPRING_DAMPING)
        spring.CreateTargetPositionAttr(B.KEY_PRELOAD_M)
        spring.CreateTargetVelocityAttr(0.0)
        spring.CreateMaxForceAttr(100.0)


seat_frames = C.seat_frames


# --- poses of the moving parts (render copies; the physics bodies get the same kinematic targets) -----------------
def set_render_poses(stage, paths, poses, table_z):
    for path, (x, y, z, qx, qy, qz, qw) in zip(paths, poses):
        api = UsdGeom.XformCommonAPI(UsdGeom.Xform(stage.GetPrimAtPath(path)))
        api.SetTranslate(Gf.Vec3d(x, y, z - table_z))
        api.SetRotate(Gf.Vec3f(0.0, math.degrees(2 * math.atan2(qy, qw)), 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def set_cable(stage, points, table_z, radius=None):
    """Redraw the cable through points (world) as a tube mesh (its material stays bound)."""
    import visual_kit as K
    pts = [(float(p[0]), float(p[1]), float(p[2] - table_z)) for p in points]
    K.define(stage, CABLE, K.tube(pts, radius or CABLE_R, sides=8))


def show(stage, analyzer, valve, previous):
    """Light the analyzer (PEAK, FREQ cells, RUN lamp) and the valve LCD where they differ from `previous`."""
    peak, hz, lamp = analyzer
    old = previous or {}
    out = {'peak': show_cells(stage, ROOT + '/AnalyzerPeak', peak, old.get('peak')),
           'hz': show_cells(stage, ROOT + '/AnalyzerHz', hz, old.get('hz')),
           'valve': show_cells(stage, MOUNT + '/Valve', valve, old.get('valve'), point=False), 'run': lamp}
    if old.get('run') != lamp:
        C._light(stage, ROOT + '/AnalyzerRun', lamp)
    return out

"""Truss bench (truss_bridge v2), independent of Isaac: the bench layout, the pegboard's geometry, where each loose pin
stands (in a board hole, in the pin stand) and where each bar lies (on two seated pins, mounted in the tester, in a
rack slot, in the scrap tray), and the bar tester.

World layout (the robot stands at (0.95, 0.35) facing -y; world metres on the table top, z above it):
- The pegboard, centre-right of the robot: holes truss_model.HOLES in the bench frame (x along the span from support A
  = world +x, y away from the load direction = world +y: the rows beyond the deck line lie toward the robot and the
  load pulls C away from it, where the showcase loading frame stands in view of the cameras). A, C and B are fixed
  pins in their holes; the other holes are open.
- The pin stand beside the board: a block with N_LOOSE_PINS holes like the board's, the loose pins standing in them.
- The bar racks: rows of slots, each bar lying flat across its row (rows along world x hold bars along y, rows along
  y bars along x), two rows per printed length (blue and orange slots in one, green and white in the other), the
  length printed at one end of each slot.
- The scrap tray at the front left, beside the platform.
- The bar tester on the right: a fixed pin on a pedestal and a pin on a crosshead carriage that runs along world y on a
  rail with ball detents at the four eye spacings; its console (closer to the robot) carries the display, the green
  START key and the two-position MODE slider (TENSION / COMPRESSION).

A pin (loose or fixed) is a straight post (radius PIN_R, PIN_H tall, a conical tip on top) on a collar (radius
COLLAR_R, COLLAR_H thick) that rests on the board; a loose pin also has a shank (radius SHANK_R, SHANK_L long, its end
chamfered) below the collar. A hole is round (radius HOLE_R) through the board, its mouth chamfered at 45 degrees out
to CHAMFER_R at the top. A loose pin's frame: origin on its axis at the collar's underside, z up its axis. A
loose pin is seated in a hole (a stand hole) when its axis is within SEAT_XY_M of the hole's centre at the board top,
it is within SEAT_TILT_DEG of upright, its collar's underside within SEAT_Z_M of the board top and it is not held.

A bar's frame: origin at its centre on its underside, x along it (eyes at +-s/2), z up out of its top. A bar is two
bodies joined at CUT_M (bar-local x, just beside the grip knob): the main body (the knob and the eye at +s/2, id <bar>)
and the tail (the eye at -s/2, id <bar>t), both in the bar's frame. A bar that breaks in tension comes apart there into
the two pieces; one that buckles stays one bar, kinked there. An eye is on a pin when its centre lies within
EYE_ON_PIN_M of the pin's axis (at the eye's height) and between the collar's top and the top of the straight post; a
bar is in the bridge when it is not held and both eyes are on two pins seated in two different holes of the board (the
bar then spans them: its eye spacing fits only holes that far apart). It is mounted in the tester the same way on the
tester's two pins. A broken bar is where its pieces are: in the tester while a piece's eye is on a tester pin, in the
tray when both pieces are.

The tester: START (a rising edge on its key, READY or DONE) latches the bar mounted in it (or none) and the MODE
slider's detent (a slider between detents ignores START), then the force rises at RATE_N_S. The bar fails when the force
reaches its limit in that mode (truss_model.tester_peak with its own properties); the run stops there: DONE, PEAK the
failure force (shown with Gaussian noise NOISE_N, resolution 0.1 N). A run with no bar rises to nothing: the force reads
0.0 for EMPTY_STROKE_S, then DONE with PEAK 0.0. A bar taken off during the run ends it (DONE, PEAK the force reached,
the bar intact). A failed bar is broken for good: it fails again at once and carries nothing in a bridge; the pieces
of a bar broken in two are never mounted (a run with only pieces on the pins is an empty run). The display:
line 1 the state (READY / RUN / DONE) and the mode (latched while RUN or DONE, else the slider's), line 2 the force (N),
line 3 PEAK (blank before the first run); the force line refreshes DISPLAY_HZ times per second.
"""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

import truss_model as M

# --- layout (world, metres) ------------------------------------------------------------------------------------
A_XY = (0.90, -0.05)                 # hole A (support pin A) on the table
LOAD_Y = -1.0                        # world y direction in which the load pulls C (away from the robot)
BOARD_MARGIN = 0.030                 # board edge beyond the outer holes
BOARD_H = 0.012                      # board top above the table (holes go through)
HOLE_R = 0.0034                      # round through hole, 6.8 mm (the 6 mm shank centres within 0.4 mm)
CHAMFER_R, CHAMFER_DEPTH = 0.0060, 0.0026        # 45 degree mouth: 12 mm across at the top, down to the hole
HOLE_SEGMENTS = 32                   # the mesh's polygon for a hole (its flats are 0.02 mm inside the circle)
PIN_R, PIN_H, PIN_TIP = 0.004, 0.030, 0.006      # straight post above the collar, then a conical tip (to r 1.5 mm)
COLLAR_R, COLLAR_H = 0.007, 0.003
SHANK_R, SHANK_L, SHANK_TIP = 0.003, 0.009, 0.0015   # below the collar; the last SHANK_TIP narrows to r 1.5 mm
PIN_MASS_KG = 0.0175                 # steel
PIN_FOOT = BOARD_H + COLLAR_H        # a seated pin's collar top: the lowest bar's underside
SEAT_XY_M, SEAT_TILT_DEG, SEAT_Z_M = 0.0012, 4.0, 0.0010
BAR_W, BAR_T = 0.012, 0.005
EYE_R, EYE_HOLE_R = 0.010, 0.010750 / 2
KNOB = (0.014, 0.010, 0.012)         # along the bar, across, height above the bar top
EYE_ON_PIN_M = 0.002
STAND_XY = (0.835, -0.05)            # pin stand: centre of its holes' line (along world y)
STAND_PITCH = 0.040                  # room for the open fingers between two pins
STAND_MARGIN = 0.014
# Racks: rows of RACK_SLOTS slots; two rows per printed length (group 0: blue in its first KIT_PER_KIND slots, orange
# in the rest; group 1: green, then white). {(label, group): (axis, world x, world y of the row's first slot)}: axis
# 'x', the slots run along +x and the bars lie along y; axis 'y', the slots run along +y and the bars lie along x.
# Layout: every slot within the arm's reach with >= 0.15 rad on every joint (IK reach map from the
# start pose, measured grid points only; the front rows checked by real picks), its knob and both bar ends in a fixed
# camera's frame, clear of the case, platform, rope, board, stand, tester, console and the robot's riser (x 0.79-1.11,
# y > 0.19), and 6 cm off the weights' transit lines (case -> (1.25, 0.12) -> stage -> platform). The side row 50/1
# sat at x 0.20, past the reach (no IK there, probe p9): now x 0.25 from y 0.12, the back rows 5 cm further right and
# 2 cm lower (their far slots stay in agentview_left's frame), 100/0 2 cm lower with them; the right row 100/1 1.5 cm
# lower (taking slot 20, link 5/6 brushed the case's near 2 kg stack, probe p12). Table top x -0.25..1.55,
# y -0.70..0.70; robot base (0.95, 0.35).
ROBOT_XY = (0.95, 0.35)
RACK_ROWS = {(112, 0): ('x', 0.50, -0.14), (50, 0): ('x', 0.50, -0.02), (71, 0): ('x', 0.50, 0.08),
             (112, 1): ('x', 0.33, 0.22), (71, 1): ('x', 0.33, 0.36),
             (100, 0): ('y', 0.40, -0.15), (50, 1): ('y', 0.25, 0.12), (100, 1): ('y', 1.40, -0.235)}
RACKS = {'front': ((112, 0), (50, 0), (71, 0), (100, 0)), 'back': ((112, 1), (71, 1)), 'side': ((50, 1),),
         'right': ((100, 1),)}             # plates: {name: rows}
RACK_PITCH, RACK_SLOTS = 0.030, 10   # slots per row; a length's slots are numbered 1-10 (group 0), 11-20 (group 1)
RACK_BASE_H = 0.006
RACK_EDGE, RACK_LABEL = 0.004, 0.015                # plate margin round the slots; the label strip at the bars' ends
SLOT_HALF = (0.012, 0.030)           # bar centre within this of the slot centre (across, along the bar) ...
SLOT_YAW_DEG, FLAT_TILT_DEG = 15.0, 10.0          # ... lying along the slot and flat
TRAY_XY, TRAY_INNER, TRAY_WALL, TRAY_H = (0.6875, -0.2775), (0.175, 0.105), 0.004, 0.040   # front left, left of
                                     # the platform; a bin deep enough for every scrap piece of a run. Its front and left
                                     # are at the arm's reach on the floor (solve_ik), the platform on its right
TRAY_PILE_H = 0.040                  # a scrap piece lies in the tray up to this high (pieces may lie on each other)
CUT_M = -(KNOB[0] / 2 + 0.002)       # bar-local x of the joint between a bar's main body and its tail
TESTER_X = 1.23
TESTER_FIXED_Y = -0.20               # the fixed pin; the carriage pin stands one eye spacing toward the robot
TESTER_SPACINGS_MM = {label: M.BAR_LENGTHS_MM[label] for label in M.LABELS}
TESTER_TOP = 0.0265                  # pedestal and carriage tops above the table (the bar's underside)
TESTER_PIN_H = 0.012                 # straight part above the tops
CONSOLE_XY = (1.23, 0.0)

# --- tester ----------------------------------------------------------------------------------------------------
RATE_N_S = 20.0
DISPLAY_HZ = 10
RESOLUTION_N = 0.1
NOISE_N = 0.2
EMPTY_STROKE_S = 5.0
MODES = ('tension', 'compression')
START_ID = 'start'
STATES = ('READY', 'RUN', 'DONE')


def pin_xy(hole):
    """World xy of a board hole (named as in truss_model: 'A', 'B', 'C', 'x<col>y<row>')."""
    x, y = M.HOLE_XY_MM[hole]
    return (A_XY[0] + x / 1000.0, A_XY[1] + y / 1000.0)


def board_holes():
    return {n: pin_xy(n) for n in M.HOLES}


def board_rect():
    """World (x0, y0, x1, y1) of the board."""
    xs = [p[0] for p in board_holes().values()]
    ys = [p[1] for p in board_holes().values()]
    return (min(xs) - BOARD_MARGIN, min(ys) - BOARD_MARGIN, max(xs) + BOARD_MARGIN, max(ys) + BOARD_MARGIN)


def stand_holes():
    """{'stand:<k>': world xy}, k = 1 .. N_LOOSE_PINS (along world y, the first nearest the robot)."""
    n = M.N_LOOSE_PINS
    return {f'stand:{k + 1}': (STAND_XY[0], STAND_XY[1] + ((n - 1) / 2 - k) * STAND_PITCH) for k in range(n)}


def stand_rect():
    ys = [p[1] for p in stand_holes().values()]
    return (STAND_XY[0] - STAND_MARGIN, min(ys) - STAND_MARGIN, STAND_XY[0] + STAND_MARGIN, max(ys) + STAND_MARGIN)


def initial_pins():
    """The loose pins at the start: {pin id: stand slot} (ids private: pin1 .. pin4)."""
    return {f'pin{k + 1}': f'stand:{k + 1}' for k in range(M.N_LOOSE_PINS)}


def tester_pins(carriage_y):
    """World xy of the tester's fixed pin and of the carriage pin (the carriage at world y)."""
    return {'fixed': (TESTER_X, TESTER_FIXED_Y), 'carriage': (TESTER_X, carriage_y)}


def detent_y(label):
    return TESTER_FIXED_Y + TESTER_SPACINGS_MM[label] / 1000.0


def slot_kind(k):
    """The kind in a length's slot k (0-based over both its rows)."""
    return M.KINDS[k // M.KIT_PER_KIND]


def rack_slots():
    """{'rack:<label>:<k>': (xy, label, kind)}, k = 1 .. 2 RACK_SLOTS (each row's slots along +x; the length printed
    beside each)."""
    out = {}
    for (label, group) in RACK_ROWS:
        for j, xy in enumerate(row_slots((label, group))):
            k = group * RACK_SLOTS + j
            out[f'rack:{label}:{k + 1}'] = (xy, label, slot_kind(k))
    return out


def row_slots(row):
    """World xy of a rack row's slot centres (the bars' knobs), in slot order."""
    axis, x0, y0 = RACK_ROWS[row]
    return [(round(x0 + j * RACK_PITCH, 6), y0) if axis == 'x' else (x0, round(y0 + j * RACK_PITCH, 6))
            for j in range(RACK_SLOTS)]


def slot_yaw_deg(slot):
    """World yaw of a bar lying in the slot 'rack:<label>:<k>' (its axis: 90 along y, 0 along x)."""
    _, label, k = slot.split(':')
    return 90.0 if RACK_ROWS[(int(label), (int(k) - 1) // RACK_SLOTS)][0] == 'x' else 0.0


def row_rect(row):
    """World (x0, y0, x1, y1) of a rack row: its slots with RACK_EDGE round them and the label strip at the bars'
    ends (behind them, +y, for a row along x; toward the robot for a row along y)."""
    axis, x0, y0 = RACK_ROWS[row]
    half = bar_length_m(row[0]) / 2
    xs, ys = zip(*row_slots(row))
    p = RACK_PITCH / 2 + RACK_EDGE
    if axis == 'x':
        return (min(xs) - p, y0 - half - RACK_EDGE, max(xs) + p, y0 + half + RACK_LABEL)
    toward = 1.0 if x0 < ROBOT_XY[0] else -1.0
    lo, hi = (RACK_EDGE, RACK_LABEL) if toward > 0 else (RACK_LABEL, RACK_EDGE)
    return (x0 - half - lo, min(ys) - p, x0 + half + hi, max(ys) + p)


def label_xy(row, j):
    """World xy of the length printed at a slot's end (row slot j, 0-based)."""
    axis, x0, _ = RACK_ROWS[row]
    sx, sy = row_slots(row)[j]
    off = bar_length_m(row[0]) / 2 + 0.008
    if axis == 'x':
        return (sx, sy + off)
    return (sx + (off if x0 < ROBOT_XY[0] else -off), sy)


def initial_bars():
    """The kit at the start: {bar id: {'kind', 'label', 'slot'}} (ids private: bar01 .. bar80)."""
    out, n = {}, 0
    for label in M.LABELS:
        for k in range(len(M.KINDS) * M.KIT_PER_KIND):
            n += 1
            out[f'bar{n:02d}'] = {'kind': slot_kind(k), 'label': label, 'slot': f'rack:{label}:{k + 1}'}
    return out


def bar_length_m(label):
    """Overall length: eye spacing plus the two eye bosses."""
    return M.BAR_LENGTHS_MM[label] / 1000.0 + 2 * EYE_R


# --- the holed plates (board and pin stand): a static triangle mesh -------------------------------------------
def holed_plate(rect, holes, top_z, bottom_z=0.0, tile_half=None, segments=HOLE_SEGMENTS, groups=None):
    """(points, faces) of a plate over rect (x0, y0, x1, y1) from bottom_z to top_z with round through holes (radius
    HOLE_R, as a `segments`-gon) at the world xy `holes`, each mouth chamfered at 45 degrees out to CHAMFER_R at the
    top. Each hole sits in a square tile (half side tile_half: at most half the smallest hole spacing); the rest of
    the plate is rectangles. Faces are counter-clockwise seen from outside the solid. groups (a list), if given,
    gets each face's part: 'plate', 'chamfer' or 'bore'."""
    x0, y0, x1, y1 = rect
    if tile_half is None:
        gaps = [math.dist(a, b) for i, a in enumerate(holes) for b in holes[i + 1:]]
        tile_half = min(gaps) / 2 if gaps else CHAMFER_R + 0.002
        tile_half = min([tile_half] + [min(h[0] - x0, x1 - h[0], h[1] - y0, y1 - h[1]) for h in holes])
    T = tile_half
    zc = top_z - CHAMFER_DEPTH
    pts, faces, index = [], [], {}

    def p(x, y, z):
        key = (round(x, 7), round(y, 7), round(z, 7))
        if key not in index:
            index[key] = len(pts)
            pts.append(key)
        return index[key]

    def cells(z, up):
        xs = sorted({x0, x1} | {h[0] + s * T for h in holes for s in (-1, 1)})
        ys = sorted({y0, y1} | {h[1] + s * T for h in holes for s in (-1, 1)})
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
                if any(abs(cx - hx) < T and abs(cy - hy) < T for hx, hy in holes):
                    continue
                q = [p(xs[i], ys[j], z), p(xs[i + 1], ys[j], z), p(xs[i + 1], ys[j + 1], z), p(xs[i], ys[j + 1], z)]
                faces.append(q if up else q[::-1])

    cells(top_z, True)
    cells(bottom_z, False)
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for k in range(4):                                   # outer walls
        (ax, ay), (bx, by) = corners[k], corners[(k + 1) % 4]
        faces.append([p(ax, ay, bottom_z), p(bx, by, bottom_z), p(bx, by, top_z), p(ax, ay, top_z)])
    angles = [2 * math.pi * k / segments for k in range(segments)]          # segments % 8 == 0: the corners are in
    for hx, hy in holes:
        sq = [(hx + T * math.cos(a) / max(abs(math.cos(a)), abs(math.sin(a))),
               hy + T * math.sin(a) / max(abs(math.cos(a)), abs(math.sin(a)))) for a in angles]
        ring = lambda r: [(hx + r * math.cos(a), hy + r * math.sin(a)) for a in angles]     # noqa: E731
        top, mid = ring(CHAMFER_R), ring(HOLE_R)
        for k in range(segments):
            n = (k + 1) % segments
            faces.append([p(*sq[n], top_z), p(*top[n], top_z), p(*top[k], top_z), p(*sq[k], top_z)])     # top
            faces.append([p(*sq[k], bottom_z), p(*mid[k], bottom_z), p(*mid[n], bottom_z), p(*sq[n], bottom_z)])
            faces.append([p(*top[n], top_z), p(*mid[n], zc), p(*mid[k], zc), p(*top[k], top_z)])          # chamfer
            faces.append([p(*mid[n], zc), p(*mid[n], bottom_z), p(*mid[k], bottom_z), p(*mid[k], zc)])    # bore
    if groups is not None:
        groups.extend(['plate'] * (len(faces) - 4 * segments * len(holes)) +
                      ['plate', 'plate', 'chamfer', 'bore'] * segments * len(holes))
    return pts, faces


# --- geometry of posed bodies ----------------------------------------------------------------------------------
def _rotation(quat_wxyz):
    w, x, y, z = [float(v) for v in quat_wxyz]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def tail_id(bar):
    return bar + 't'


def _local_point(position, quat_wxyz, local, table_z):
    c = np.asarray(position, float) + _rotation(quat_wxyz) @ np.asarray(local, float)
    return (float(c[0]), float(c[1]), float(c[2]) - table_z)


def eye_centres(position, quat_wxyz, label, table_z=0.0, tail=None):
    """World (x, y, z above the table) of a bar's two eye centres (mid-thickness): the eye at -s/2 from the tail's pose
    (position, quat_wxyz) if given, the eye at +s/2 from the main body's."""
    half = M.BAR_LENGTHS_MM[label] / 2000.0
    tail = tail or (position, quat_wxyz)
    return [_local_point(*tail, (-half, 0.0, BAR_T / 2), table_z),
            _local_point(position, quat_wxyz, (half, 0.0, BAR_T / 2), table_z)]


def piece_centres(main, tail, label, table_z=0.0):
    """World (x, y, z above the table) of the middles of a bar's tail and main body (each pose (position, quat))."""
    half = M.BAR_LENGTHS_MM[label] / 2000.0
    return [_local_point(*tail, ((CUT_M - half) / 2, 0.0, BAR_T / 2), table_z),
            _local_point(*main, ((CUT_M + half) / 2, 0.0, BAR_T / 2), table_z)]


def tilt_deg(quat_wxyz):
    R = _rotation(quat_wxyz)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))


def yaw_deg(quat_wxyz):
    """Heading of the bar's axis in the table plane (0 along world x), folded to [0, 90] (an axis, not a direction)."""
    R = _rotation(quat_wxyz)
    a = abs(math.degrees(math.atan2(R[1, 0], R[0, 0]))) % 180.0
    return min(a, 180.0 - a)


def pin_axis(position, quat_wxyz, table_z=0.0):
    """(foot (x, y, z above the table) of a loose pin's axis at its collar's underside, unit axis)."""
    R = _rotation(quat_wxyz)
    p = np.asarray(position, float)
    return (float(p[0]), float(p[1]), float(p[2]) - table_z), tuple(float(v) for v in R[:, 2])


def pin_seat(position, quat_wxyz, table_z=0.0):
    """The board hole ('A' .. 'x4y2') or stand slot ('stand:k') a loose pin is seated in, or None."""
    (x, y, z), _ = pin_axis(position, quat_wxyz, table_z)
    if tilt_deg(quat_wxyz) > SEAT_TILT_DEG or abs(z - BOARD_H) > SEAT_Z_M:
        return None
    for name, (hx, hy) in list(board_holes().items()) + list(stand_holes().items()):
        if math.hypot(x - hx, y - hy) <= SEAT_XY_M:
            return name
    return None


class _Post:
    """A pin's straight post as a segment: base (x, y, z above the table) and unit axis, length PIN_H (or h)."""

    def __init__(self, base, axis=(0.0, 0.0, 1.0), height=PIN_H):
        self.base, self.axis, self.height = np.asarray(base, float), np.asarray(axis, float), height

    def holds(self, eye):
        """The eye's centre lies within EYE_ON_PIN_M of the axis, between the post's base and its top."""
        d = np.asarray(eye, float) - self.base
        t = float(d @ self.axis)
        return 0.0 <= t <= self.height and float(np.linalg.norm(d - t * self.axis)) <= EYE_ON_PIN_M


def _pin_under(eye, posts):
    for name, post in posts.items():
        if post.holds(eye):
            return name
    return None


def _in_tray(x, y, z):
    tx, ty = TRAY_XY
    return abs(x - tx) <= TRAY_INNER[0] / 2 and abs(y - ty) <= TRAY_INNER[1] / 2 and z <= TRAY_PILE_H


def classify(poses, labels, held, carriage_y, table_z=0.0, split=(), pins=()):
    """Where each bar lies and each loose pin stands.

    Bars: {bar: ('bridge', (hole, hole)) | ('tester', None) | ('rack', slot) | ('tray', None) | ('held', None) |
    ('loose', None)}; loose pins (ids `pins`): {pin: ('hole', hole) | ('stand', slot) | ('held', None) |
    ('loose', None)}. poses: {body: (position, quat wxyz)}, a bar's main body under its id and its tail under tail_id
    (absent: the tail lies as the main body, as it does while the two are joined straight), each loose pin under its
    id; held: the bodies in or leaving the hand; split: the bars broken in two (a piece of one lies in the tester
    while its eye is on a tester pin, the bar in the tray when both pieces are, else loose)."""
    out = {}
    posts = {n: _Post((x, y, PIN_FOOT)) for n, (x, y) in board_holes().items() if n in M.FIXED}
    for pin in pins:
        if pin in held:
            out[pin] = ('held', None)
            continue
        seat = pin_seat(*poses[pin], table_z)
        if seat is None:
            out[pin] = ('loose', None)
        elif seat.startswith('stand:'):
            out[pin] = ('stand', seat)
        else:
            out[pin] = ('hole', seat)
            if seat not in posts:                   # a hole with two pins cannot be: the first seated one stands there
                foot, axis = pin_axis(*poses[pin], table_z)
                base = np.asarray(foot) + COLLAR_H * np.asarray(axis)
                posts[seat] = _Post(base, axis)
    tester = {n: _Post((x, y, TESTER_TOP), height=TESTER_PIN_H) for n, (x, y) in tester_pins(carriage_y).items()}
    slots = rack_slots()
    slot_yaw = {sid: slot_yaw_deg(sid) for sid in slots}
    for bar, label in labels.items():
        position, quat = poses[bar]
        tail = poses.get(tail_id(bar), (position, quat))
        if bar in held or tail_id(bar) in held:
            out[bar] = ('held', None)
            continue
        eyes = eye_centres(position, quat, label, table_z, tail=tail)
        if bar in split:
            if any(_pin_under(e, tester) for e in eyes):
                out[bar] = ('tester', None)
            elif all(_in_tray(*c) for c in piece_centres((position, quat), tail, label, table_z)):
                out[bar] = ('tray', None)
            else:
                out[bar] = ('loose', None)
            continue
        on = [_pin_under(e, posts) for e in eyes]
        if None not in on and on[0] != on[1]:
            out[bar] = ('bridge', tuple(sorted(on)))
            continue
        mounted = [_pin_under(e, tester) for e in eyes]
        if None not in mounted and mounted[0] != mounted[1]:
            out[bar] = ('tester', None)
            continue
        x, y, z = float(position[0]), float(position[1]), float(position[2]) - table_z
        if _in_tray(x, y, z):
            out[bar] = ('tray', None)
            continue
        slot = None
        if tilt_deg(quat) <= FLAT_TILT_DEG and z <= RACK_BASE_H + 0.004:
            yaw = yaw_deg(quat)
            for sid, ((sx, sy), _, _) in slots.items():
                along_y = slot_yaw[sid] == 90.0
                across, along = (abs(x - sx), abs(y - sy)) if along_y else (abs(y - sy), abs(x - sx))
                if (abs(yaw - slot_yaw[sid]) <= SLOT_YAW_DEG and across <= SLOT_HALF[0]
                        and along <= SLOT_HALF[1]):
                    slot = sid
                    break
        out[bar] = ('rack', slot) if slot else ('loose', None)
    return out


def tester_clear(poses, labels, carriage_y, table_z=0.0):
    """No eye of any bar or piece is on a tester pin."""
    tester = {n: _Post((x, y, TESTER_TOP), height=TESTER_PIN_H) for n, (x, y) in tester_pins(carriage_y).items()}
    for bar, label in labels.items():
        position, quat = poses[bar]
        for e in eye_centres(position, quat, label, table_z, tail=poses.get(tail_id(bar))):
            if _pin_under(e, tester):
                return False
    return True


def bridge_members(where):
    """[(bar, (hole, hole))] of the bars in the bridge, sorted by bar."""
    return sorted((bar, w[1]) for bar, w in where.items() if w[0] == 'bridge')


def bridge_capacity(where, bars, failed):
    """The bridge's failure load (N) from each bar's own properties, the governing bar and mode, and its mass (g).
    A failed bar is broken: it carries nothing and stiffens nothing, so the bridge is analysed without it (it still
    weighs). bars: {bar: {'kind', 'label', 'T_N', 'P100_N', 'N_N'}}."""
    members = bridge_members(where)
    mass = sum(M.bar_mass_g(bars[b]['kind'], bars[b]['label']) for b, _ in members)
    intact = [(b, pins) for b, pins in members if b not in failed]
    rows = [{'pins': pins, 'label': bars[b]['label'], 'T_N': bars[b]['T_N'], 'P100_N': bars[b]['P100_N'],
             'N_N': bars[b]['N_N']} for b, pins in intact]
    P, which = M.capacity(rows)
    governing = None if which is None else {'bar': intact[which[0]][0], 'mode': which[1]}
    return {'capacity_N': float(P), 'governing': governing, 'mass_g': round(mass, 4),
            'members': [[b, list(p)] for b, p in members]}


# --- the tester ------------------------------------------------------------------------------------------------
def force_text(value_n):
    """Four seven-segment cells of a force line: hundreds, tens, units, then one decimal (the point after the units
    cell); blank leading zeros; ' Err' above 999.9 N; None blank."""
    if value_n is None:
        return '    '
    v = round(max(0.0, float(value_n)), 1)
    if v > 999.9:
        return ' Err'
    return f'{v:5.1f}'.replace('.', '')


class TrussBench:
    def __init__(self, episode_id, config, *, physics_hz=240, noise_seed=0):
        if physics_hz not in (240, 480):
            raise ValueError('unsupported physics clock')
        self.episode_id = episode_id
        self.physics_hz = physics_hz
        self.bars = deepcopy(config['bars'])            # bar -> kind, label, T_N, P100_N, N_N (own properties)
        self.rng = np.random.default_rng(noise_seed)
        self.tick = 0
        self.state = 'READY'
        self.mode_now = config.get('mode', 'tension')
        self.run = None                                 # the present (or last) run
        self.runs = []
        self.failed = {}                                # bar -> mode it failed in
        self.where = {}
        self.pressed = set()
        self.force = 0.0
        self.force_shown, self.peak_shown = 0.0, None
        self._next_refresh = 0
        self.transitions, self.actions = [], []

    @property
    def t_s(self):
        return self.tick / self.physics_hz

    def _shown(self, value):
        v = float(value) + (self.rng.normal(0.0, NOISE_N) if value > 0 else 0.0)
        return round(max(0.0, round(v / RESOLUTION_N) * RESOLUTION_N), 1)

    def _limit(self, bar, mode):
        if bar in self.failed:
            return 0.0
        b = self.bars[bar]
        return M.tester_peak(mode, b['T_N'], b['P100_N'], b['N_N'], b['label'])

    def _start(self, mounted):
        bar = mounted[0] if len(mounted) == 1 else None
        self.run = {'run': len(self.runs) + 1, 'tick': self.tick, 't_s': round(self.t_s, 6), 'mode': self.mode_now,
                    'bar': bar, 'kind': self.bars[bar]['kind'] if bar else None,
                    'label': self.bars[bar]['label'] if bar else None,
                    'limit_N': None if bar is None else self._limit(bar, self.mode_now), 'mounted': list(mounted),
                    'end': None, 'peak_N': None, 'peak_shown_N': None}
        self.runs.append(self.run)
        self.state = 'RUN'
        self.force = 0.0
        self.peak_shown = None
        self.actions.append({'tick': self.tick, 't_s': round(self.t_s, 6), 'action': 'start', 'run': self.run['run'],
                             'mode': self.mode_now, 'mounted': len(mounted)})

    def _finish(self, end, peak):
        run = self.run
        run.update(end=end, peak_N=round(float(peak), 4), end_tick=self.tick, end_t_s=round(self.t_s, 6))
        run['peak_shown_N'] = self.peak_shown = self._shown(peak) if peak > 0 else 0.0
        if end == 'failed' and run['bar'] is not None and run['bar'] not in self.failed:
            self.failed[run['bar']] = run['mode']
        self.state = 'DONE'
        self.force = 0.0
        self.force_shown = 0.0

    def step(self, tick, world):
        """world: {'where': classify(...), 'mode': 'tension' | 'compression' | None (between detents),
        'pressed': set of button ids}."""
        if tick != self.tick + 1:
            raise ValueError('truss bench requires every consecutive physics tick')
        self.tick = tick
        where = world.get('where', {})
        if where != self.where:
            self.transitions.append({'tick': tick, 't_s': round(self.t_s, 6),
                                     'where': {b: [w[0], list(w[1]) if isinstance(w[1], tuple) else w[1]]
                                               for b, w in where.items()}})
            self.where = dict(where)
        if self.state != 'RUN':
            self.mode_now = world.get('mode')
        split = {b for b, m in self.failed.items() if m == 'tension'}
        mounted = sorted(b for b, w in where.items() if w[0] == 'tester' and b not in split)
        pressed = set(world.get('pressed', ()))
        if START_ID in pressed - self.pressed and self.state != 'RUN' and self.mode_now in MODES:
            self._start(mounted)
        self.pressed = pressed
        if self.state == 'RUN':
            run = self.run
            self.force = RATE_N_S * (self.t_s - run['t_s'])
            if run['bar'] is None:
                self.force = 0.0
                if self.t_s - run['t_s'] >= EMPTY_STROKE_S:
                    self._finish('empty', 0.0)
            elif run['bar'] not in mounted:
                self._finish('removed', max(0.0, self.force - RATE_N_S / self.physics_hz))
            elif self.force >= run['limit_N']:
                self._finish('failed', run['limit_N'])
        if self.state == 'RUN' and tick >= self._next_refresh:
            self._next_refresh = tick + self.physics_hz // DISPLAY_HZ
            self.force_shown = self._shown(self.force)

    def display(self):
        """What the tester shows: {'state', 'mode', 'force_N', 'peak_N'}."""
        mode = self.run['mode'] if self.state != 'READY' and self.run else self.mode_now
        return {'state': self.state, 'mode': mode, 'force_N': self.force_shown if self.state == 'RUN' else 0.0,
                'peak_N': self.peak_shown}

    def screen(self):
        d = self.display()
        return {'state': d['state'], 'mode': d['mode'], 'force': force_text(d['force_N']), 'peak': force_text(d['peak_N'])}

    def observation_record(self, tick):
        """Private record of one observation: the display then and the run it belongs to."""
        return {'tick': tick, 't_s': round(tick / self.physics_hz, 6), 'display': self.display(), 'screen': self.screen(),
                'run': deepcopy(self.run), 'where': {b: [w[0], list(w[1]) if isinstance(w[1], tuple) else w[1]]
                                                     for b, w in self.where.items()},
                'failed': dict(self.failed)}

    def private_record(self):
        return {'runs': deepcopy(self.runs), 'failed': dict(self.failed), 'transitions': deepcopy(self.transitions),
                'actions': deepcopy(self.actions), 'bars': deepcopy(self.bars)}

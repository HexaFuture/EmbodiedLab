"""Geometry of the circuit board bench (no Isaac): board, nodes and sockets, blocks, component tray, supply, black box,
logger unit. All positions are world metres on the lab table (top at TABLE_Z); the robot base stands at world
(0.95, 0.35) facing -y, so world +x is the robot's left and the wrist camera at the start pose shows world -x as image
up and world +y as image right. Printed text reads upright in that view (its up is world -x).

Board: COLS x ROWS nodes on a square grid of PITCH. Node (c, r), c = 1..COLS along +x, r = 1..ROWS from the row
nearest the robot (largest y) to the farthest. Each node is a brass pad with one socket toward each neighbour
(SOCKET_OFFSET from the node centre): a square conical recess (45 degree walls) that guides a block's contact pin in.
A block spans one edge (two adjacent nodes): its pins sit in the two facing sockets of that edge, PIN_SPACING apart.

Blocks: a base (BASE) with two spherical contact pins under its ends and, on top, a handle (HANDLE) or, on a meter, a
display housing (HEAD); the + pin (local +x) of polarised blocks is marked red. Local frame: origin at the centre of
the base's bottom face, x from the - pin to the + pin, z up. Printed text on a block reads along local +x, up local +y.
"""
from __future__ import annotations

import math

TABLE_Z = 0.820

# --- board --------------------------------------------------------------------------------------------------------
COLS, ROWS = 5, 5
PITCH = 0.060
NODE_X0, NODE_Y0 = 0.780, 0.020            # node (1, 1)
BOARD_H = 0.012
BOARD_TOP_Z = TABLE_Z + BOARD_H
BOARD_MARGIN = 0.012                       # plate edge beyond the outer pads
PAD_HALF = 0.024                           # brass pad half-width around each node
SOCKET_OFFSET = 0.015                      # socket centre from its node centre, toward the neighbour
SOCKET_MOUTH_HALF = 0.0065                 # square mouth half-width at the board surface
SOCKET_DEPTH = 0.0055                      # 45 degree walls down to a small floor
PIN_SPACING = PITCH - 2 * SOCKET_OFFSET    # 0.030
SEAT_TOL_M = 0.004                         # each pin within 4 mm of its socket centre
SEAT_TILT_DEG = 5.0
SEAT_DZ_M = 0.002                          # base bottom within 2 mm of the board surface

# Fixed terminals on board nodes.
SUPPLY_PLUS, SUPPLY_MINUS = (1, 2), (1, 3)
BOX_TERMINALS = {'A': (5, 2), 'B': (5, 3), 'C': (5, 4)}


def node_xy(node):
    c, r = node
    return (NODE_X0 + PITCH * (c - 1), NODE_Y0 - PITCH * (r - 1))


def node_name(node):
    return f'{node[0]},{node[1]}'


def all_nodes():
    return [(c, r) for r in range(1, ROWS + 1) for c in range(1, COLS + 1)]


def edges():
    """Every edge (node, node) once: horizontal (c, r)-(c+1, r) and vertical (c, r)-(c, r+1)."""
    out = []
    for c, r in all_nodes():
        if c < COLS:
            out.append(((c, r), (c + 1, r)))
        if r < ROWS:
            out.append(((c, r), (c, r + 1)))
    return out


def socket_xy(node, toward):
    (x0, y0), (x1, y1) = node_xy(node), node_xy(toward)
    d = math.hypot(x1 - x0, y1 - y0)
    return (x0 + SOCKET_OFFSET * (x1 - x0) / d, y0 + SOCKET_OFFSET * (y1 - y0) / d)


def sockets():
    """[(x, y, node, edge index)] of every socket."""
    out = []
    for k, (a, b) in enumerate(edges()):
        out.append((*socket_xy(a, b), a, k))
        out.append((*socket_xy(b, a), b, k))
    return out


def board_extent():
    x0, y0 = node_xy((1, 1))
    x1, y1 = node_xy((COLS, ROWS))
    m = PAD_HALF + BOARD_MARGIN
    return (x0 - m, x1 + m), (y1 - m, y0 + m)


# --- blocks -------------------------------------------------------------------------------------------------------
BASE = (0.034, 0.022, 0.012)               # x (along the pins), y, z
PIN_R = 0.003
PIN_Z = -0.0015                            # pin centres below the base bottom: they protrude 4.5 mm
HANDLE = (0.020, 0.012, 0.018)             # on top of the base, centred
HEAD = (0.034, 0.018, 0.018)               # meter display housing on top of the base (it is the meter's handle)
BLOCK_TOP = BASE[2] + HANDLE[2]            # 0.030 above the block's bottom, for every block
ROCKER = (0.026, 0.008, 0.004)             # switch rocker on the handle top, pivot across the block (local y)
ROCKER_ANGLE_DEG = 10.0                    # either end down; the end pressed down shows the state
ROCKER_PIVOT_Z = BLOCK_TOP + 0.0025

# id: (kind, printed label, seat class, mass kg)
BLOCKS = {
    'V1': ('voltmeter', 'V1', 'voltmeter', 0.060),
    'V2': ('voltmeter', 'V2', 'voltmeter', 0.060),
    'A': ('ammeter', 'A', 'ammeter', 0.060),
    'SW': ('switch', 'SW', 'switch', 0.045),
    'CH1': ('channel', 'CH1', 'channel', 0.040),
    'CH2': ('channel', 'CH2', 'channel', 0.040),
    'R1k': ('resistor', '1k', 'R1k', 0.040),
    'R10k': ('resistor', '10k', 'R10k', 0.040),
    'R100k': ('resistor', '100k', 'R100k', 0.040),
    **{f'W{k}': ('wire', '', 'wire', 0.035) for k in range(1, 11)},
}
POLARISED = ('voltmeter', 'ammeter', 'channel')
METERS = ('V1', 'V2', 'A')
CHANNELS = ('CH1', 'CH2')
RS_NOMINAL = {'R1k': 1000.0, 'R10k': 10000.0, 'R100k': 100000.0}


def block_path(root, block_id):
    return f'{root}/Blocks/{block_id}'


# Three-node blocks (the changeover switch of rc_capacitance / circuit_blackbox3; not in the resistor_blackbox tray):
# a straight block over three collinear nodes n0, n1, n2. Its pins sit in n0's socket toward n1 (throw T1), n1's socket
# toward n0 (common C) and n2's socket toward n1 (throw T2); the socket of n1 toward n2 lies free under the block. A
# lever on top selects the throw: break before make, the common is open while the lever is between its contact
# angles. Local frame: origin at the centre of the base bottom, right over n1; x from T1 toward T2.
BASE3 = (0.102, 0.022, 0.012)                       # x along the three pins
CHANGEOVER_PINS = {'T1': -0.045, 'C': -0.015, 'T2': 0.045}  # local x of each pin
LEVER = (0.040, 0.010, 0.006)
LEVER_ANGLE_DEG = 20.0                              # lever end stops; it rests against one of them
LEVER_CONTACT_DEG = 8.0                             # beyond +-8 deg the lever closes C-T2 (+) or C-T1 (-)
CHANGEOVER_BLOCKS = {'CO': ('changeover', 'C/O', 'changeover', 0.080)}


def changeover_pins():
    """{pin: local (x, y, z)} of a three-node block: origin at the base centre over n1, x toward T2."""
    return {name: (x, 0.0, PIN_Z) for name, x in CHANGEOVER_PINS.items()}


def lever_position(angle_deg):
    """Throw the common is connected to at a lever angle: 'T2', 'T1' or None (between the contacts: open)."""
    if angle_deg >= LEVER_CONTACT_DEG:
        return 'T2'
    if angle_deg <= -LEVER_CONTACT_DEG:
        return 'T1'
    return None


def pin_local(sign):
    """Local position of the + (sign +1) or - (sign -1) pin centre."""
    return (sign * PIN_SPACING / 2, 0.0, PIN_Z)


# --- component tray -----------------------------------------------------------------------------------------------
# Wells long along world y (the block's local x points to world +y, so printed text reads upright in the wrist view).
TRAY_ROW_X = (1.215, 1.263, 1.311, 1.359)          # well centres along x (image rows, top to bottom)
TRAY_COL_Y = (0.050, -0.006, -0.062, -0.118, -0.174)  # well centres along y (image columns, right to left)
TRAY_H = 0.020                                      # tray top above the table
TRAY_FLOOR = 0.010                                  # well floor above the table (the block's bottom rests here)
TRAY_GROOVE_FLOOR = 0.004                           # pin groove along each well
TRAY_GROOVE_HALF = 0.005
TRAY_WELL_HALF = (0.013, 0.019)                     # inner half-size (x, y): 2 mm round the 22 x 34 mm base
TRAY_MOUTH_HALF = (0.017, 0.023)                    # chamfered mouth at the tray top
TRAY_CHAMFER_DEPTH = 0.004
TRAY_LABEL_DX = 0.0235                              # label centre beyond the well, toward +x (below it in the image)
TRAY_MARGIN = {'x0': 0.010, 'x1': 0.014, 'y0': 0.010, 'y1': 0.010}
# seat (row, column) -> block that starts there; one blank position
TRAY_LAYOUT = (('V1', 'V2', 'A', 'CH1', 'CH2'),
               ('R1k', 'R10k', 'R100k', 'SW', 'W1'),
               ('W2', 'W3', 'W4', 'W5', 'W6'),
               ('W7', 'W8', 'W9', 'W10', None))
TRAY_YAW_DEG = 90.0                                 # block local +x -> world +y
SEAT_HOME_DXY_M, SEAT_HOME_DZ_M, SEAT_HOME_TILT_DEG = 0.020, 0.020, 10.0


def tray_seats():
    """{seat label: dict(xy, block that starts there, class)} in row-major order; label = the starting block's id."""
    out = {}
    for i, row in enumerate(TRAY_LAYOUT):
        for j, block in enumerate(row):
            if block is None:
                continue
            out[block] = {'xy': (TRAY_ROW_X[i], TRAY_COL_Y[j]), 'class': BLOCKS[block][2],
                          'printed': BLOCKS[block][1] or 'W'}
    return out


def tray_extent():
    m = TRAY_MARGIN
    return ((TRAY_ROW_X[0] - TRAY_MOUTH_HALF[0] - m['x0'], TRAY_ROW_X[-1] + TRAY_MOUTH_HALF[0] + m['x1']),
            (TRAY_COL_Y[-1] - TRAY_MOUTH_HALF[1] - m['y0'], TRAY_COL_Y[0] + TRAY_MOUTH_HALF[1] + m['y1']))


# --- supply, black box --------------------------------------------------------------------------------------------
SUPPLY_BOX = ((0.620, 0.733), (-0.140, 0.000), 0.045)     # x range, y range, height
SUPPLY_BUTTON_XY = (0.652, -0.070)
SUPPLY_LAMPS = {'ON': (0.690, -0.042), 'CC': (0.690, -0.098)}
BOX_BOX = ((1.062, 1.150), (-0.190, 0.010), 0.045)
BOX_TERMINAL_COLOURS = {'A': (0.80, 0.08, 0.06), 'B': (0.04, 0.04, 0.04), 'C': (0.10, 0.55, 0.20)}

# --- push buttons (supply ON/OFF, logger keys): the coupled-bench key mechanics -----------------------------------
CAP_LIFT = 0.010             # cap bottom above the housing top at rest
CAP_H = 0.010
KEY_SIZE = 0.022
TRAVEL_M = 0.009
SWITCH_DOWN_M = 0.004        # a press acts when the cap is this far down, once, until it rises above SWITCH_UP_M
SWITCH_UP_M = 0.002
CAP_MASS_KG = 0.010
SPRING_N_PER_M = 400.0
SPRING_PRELOAD_M = 0.001
SPRING_DAMPING = 4.0

# --- logger unit ----------------------------------------------------------------------------------------------------
LOGGER_BOX = ((0.445, 0.605), (-0.205, 0.080), 0.040)
LOGGER_KEY_ROW_X = (0.470, 0.505, 0.540, 0.575)          # image rows, top to bottom
LOGGER_KEY_COL_Y = (-0.045, -0.010, 0.025, 0.060)        # image columns, left to right
LOGGER_KEYS = (('1', '2', '3', 'rate'), ('4', '5', '6', 'clear'), ('7', '8', '9', 'rec'), (None, '0', None, None))
LOGGER_KEY_PRINT = {'rate': 'RATE', 'clear': 'CLR', 'rec': 'REC'}
LOGGER_DISPLAY = ((0.452, 0.598), (-0.198, -0.068))      # dark window on the logger top (x range, y range)


def logger_key_centres():
    return {key: (LOGGER_KEY_ROW_X[i], LOGGER_KEY_COL_Y[j]) for i, row in enumerate(LOGGER_KEYS)
            for j, key in enumerate(row) if key is not None}


def logger_top_z():
    return TABLE_Z + LOGGER_BOX[2]


def supply_top_z():
    return TABLE_Z + SUPPLY_BOX[2]


def cap_rest_top_z(housing_top):
    return housing_top + CAP_LIFT + CAP_H

# --- displays (render only) ----------------------------------------------------------------------------------------
# Meter display on the head top, in the block frame: u = local x (reading direction), v = local y (text up).
METER_WINDOW = (-0.0165, -0.0085, 0.0145, 0.0085)          # u0, v0, u1, v1
METER_DIGITS = dict(u0=-0.0155, v0=-0.0075, digit_w=0.0038, digit_h=0.0085, stroke=0.0008, pitch=0.0054, count=5,
                    points=(0, 1, 2))
METER_LABEL = (-0.0150, 0.0050, 0.0042)                    # u (left end), v (centre), height
METER_UNIT = (0.0140, 0.0050, 0.0042)                      # u (right end), v, height
PLUS_BAND = (0.0150, 0.0170)                               # red band at the + end (u range) on a meter head
BASE_PLUS_BAND = (0.0110, 0.0170)                          # red band at the + end of a channel block's base top
# Logger display: window LOGGER_DISPLAY on the logger top; frame origin at the window centre, u = world +y, v = -x.
LOGGER_CH1 = dict(u0=-0.040, v0=0.033, digit_w=0.0085, digit_h=0.017, stroke=0.0019, pitch=0.0115, count=4,
                  points=(0,))
LOGGER_FIELDS = {'set': dict(u0=-0.040, v0=-0.006, count=2), 'rate': dict(u0=0.022, v0=-0.006, count=3),
                 'no': dict(u0=-0.040, v0=-0.042, count=2), 'left': dict(u0=0.022, v0=-0.042, count=2)}
LOGGER_FIELD_DIGIT = dict(digit_w=0.0060, digit_h=0.012, stroke=0.0014, pitch=0.0085)


# --- design geometry (bench construction only; not in the task spec) -------------------------------------------------
def _r(v, n=4):
    return [round(float(c), n) for c in v]


def reference_geometry():
    """The bench's design geometry (board nodes and sockets, blocks, tray seats, supply, box, logger keys) in world
    metres. Internal: the task spec publishes only what each part is (public_apparatus)."""
    supply_top = supply_top_z()
    return {
        'frame': 'world metres; table top z = %.3f' % TABLE_Z,
        'board': {'columns': COLS, 'rows': ROWS, 'pitch_m': PITCH, 'surface_z_m': round(BOARD_TOP_Z, 4),
                  'node_xy_m': {node_name(n): _r(node_xy(n)) for n in all_nodes()},
                  'socket_offset_from_node_m': SOCKET_OFFSET, 'socket_mouth_m': 2 * SOCKET_MOUTH_HALF,
                  'socket_depth_m': SOCKET_DEPTH, 'extent_xy_m': [_r(e) for e in board_extent()]},
        'fixed_terminals': {'supply_plus': node_name(SUPPLY_PLUS), 'supply_minus': node_name(SUPPLY_MINUS),
                            **{'box_' + k: node_name(v) for k, v in BOX_TERMINALS.items()}},
        'block': {'base_m': list(BASE), 'handle_m': list(HANDLE), 'meter_head_m': list(HEAD),
                  'height_m': BLOCK_TOP, 'pin_spacing_m': PIN_SPACING, 'pin_radius_m': PIN_R,
                  'pin_protrusion_m': round(-PIN_Z + PIN_R, 4),
                  'seating': {'pin_to_socket_m': SEAT_TOL_M, 'tilt_deg': SEAT_TILT_DEG,
                              'base_above_surface_m': SEAT_DZ_M}},
        'blocks': {b: {'kind': k, 'printed': p or 'none', 'tray_seat': b,
                       'polarised': k in POLARISED} for b, (k, p, _, _) in BLOCKS.items()},
        'tray': {'top_z_m': round(TABLE_Z + TRAY_H, 4), 'seat_floor_z_m': round(TABLE_Z + TRAY_FLOOR, 4),
                 'seat_yaw_deg': TRAY_YAW_DEG, 'well_m': [2 * TRAY_WELL_HALF[0], 2 * TRAY_WELL_HALF[1]],
                 'seats': {k: {'xy_m': _r(v['xy']), 'takes': v['class']} for k, v in tray_seats().items()},
                 'extent_xy_m': [_r(e) for e in tray_extent()]},
        'supply': {'housing_xy_m': [list(SUPPLY_BOX[0]), list(SUPPLY_BOX[1])], 'housing_top_z_m': round(supply_top, 4),
                   'button_xy_m': list(SUPPLY_BUTTON_XY), 'button_top_z_m': round(cap_rest_top_z(supply_top), 4),
                   'button_size_m': KEY_SIZE, 'button_travel_m': TRAVEL_M, 'button_acts_at_m': SWITCH_DOWN_M,
                   'lamps_xy_m': {k: list(v) for k, v in SUPPLY_LAMPS.items()}},
        'black_box': {'housing_xy_m': [list(BOX_BOX[0]), list(BOX_BOX[1])],
                      'housing_top_z_m': round(TABLE_Z + BOX_BOX[2], 4)},
        'logger': {'housing_xy_m': [list(LOGGER_BOX[0]), list(LOGGER_BOX[1])],
                   'housing_top_z_m': round(logger_top_z(), 4),
                   'key_xy_m': {k: list(v) for k, v in logger_key_centres().items()},
                   'key_top_z_m': round(cap_rest_top_z(logger_top_z()), 4), 'key_size_m': KEY_SIZE,
                   'display_xy_m': [list(LOGGER_DISPLAY[0]), list(LOGGER_DISPLAY[1])]},
    }


# --- public apparatus (task spec `apparatus`): what each part is and how to use it; no positions or sizes ------------
BLOCK_LOOKS = {'voltmeter': 'dark blue-grey display head printed with its name in white, a red band at its + end',
               'ammeter': 'dark red display head printed with its name in white, a red band at its + end',
               'channel': 'red band at the + end of the base', 'resistor': 'value printed on the handle',
               'switch': 'blue, with a rocker printed I and O on top', 'wire': 'grey, no resistance'}
HANDLE_COLOUR_NAMES = {'CH1': 'yellow handle', 'CH2': 'cyan handle'}


def public_apparatus():
    """Identity, appearance and use of every part (the agent finds where each one is from the cameras)."""
    return {
        'board': {'what': 'dark green plate with brass node pads; column numbers and row numbers are printed on its '
                          'edges', 'connection': 'everything with a pin or a terminal on one pad is connected there; '
                                                 'pads are otherwise insulated from each other'},
        'fixed_terminals': {'supply_plus': 'red strap from the supply + terminal to a board pad',
                            'supply_minus': 'black strap from the supply - terminal to a board pad',
                            'box_A': 'red strap from box terminal A to a board pad',
                            'box_B': 'black strap from box terminal B to a board pad',
                            'box_C': 'green strap from box terminal C to a board pad'},
        'blocks': {b: {'kind': k, 'printed': p or 'none', 'looks': HANDLE_COLOUR_NAMES.get(b, BLOCK_LOOKS[k]),
                       'tray_seat': b, 'polarised': k in POLARISED} for b, (k, p, _, _) in BLOCKS.items()},
        'tray': {'what': 'green-grey tray with one labelled well per block',
                 'seats': {k: {'printed': v['printed'], 'takes': v['class']} for k, v in tray_seats().items()}},
        'supply': {'what': 'grey DC supply printed DC SUPPLY, 5.00 V, 0.10 A MAX',
                   'button': 'ON/OFF push button: each press toggles the output', 'button_travel_m': TRAVEL_M,
                   'button_acts_at_m': SWITCH_DOWN_M, 'lamps': {'ON': 'green: output on',
                                                               'CC': 'amber: output at its current limit'}},
        'black_box': {'what': 'near-black sealed box printed BOX with terminals A (red), B (black), C (green)'},
        'logger': {'what': 'dark blue two-channel logger unit printed LOGGER',
                   'keys': [LOGGER_KEY_PRINT.get(k, k) for row in LOGGER_KEYS for k in row if k is not None],
                   'key_travel_m': TRAVEL_M, 'key_acts_at_m': SWITCH_DOWN_M,
                   'display': ['CH1 live (V)', 'SET (s)', 'RATE (Hz)', 'NO', 'LEFT', 'REC lamp', 'ERR lamp']},
    }

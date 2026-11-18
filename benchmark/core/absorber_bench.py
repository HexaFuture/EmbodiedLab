"""Vibration absorber bench (vibration_absorber), independent of Isaac: the layout, the poses of the moving parts, and
the bench itself (the strip and absorber dynamics between and during sweeps, the damper valve, the analyzer and the
records).

Layout (world, metres; heights above the table top; the robot stands at (0.95, 0.35) facing -y). The strip runs along
world x at y = BEAM_Y from the vise's jaws at x = CLAMP_X, its top BEAM_TOP above the table when unloaded; the free
length is the model's L_BEAM. The exciter stands under the strip with its stinger at X_F, the absorber's mount is
clamped on the strip at X_A. Everything on the strip is placed in the mount frame: origin on the strip's top centre
line at X_A, x toward the tip, z up; the mount (and everything fixed to it) turns with the strip there.

The absorber (mount frame): the blade's root is clamped on the mount block, its clamp face at x = ROOT_X; the blade
runs from there toward the clamp (-x) at height BLADE_H, s = distance from the clamp face. The carrier clamps the blade
at s = L (the detent its index pin sits in, 30..80 mm); it carries the cross tray (four square sockets in a row across
the strip), the orange push handle at the tray's robot end and two bumper pins along y. The pins run in the slots of
the two cheeks (fixed to the mount), whose rubber linings are the stops (+-gap from the unloaded blade line); the
robot-side pin also runs in the slot of the damper's yoke. The yoke hangs from the damper's piston rod: the damper
cylinder stands on the mount's robot side near the tip, its valve head on top with the - and + keys and the valve
LCD. The yoke moves up and down with the carrier (it follows the absorber's z), the cylinder with the mount.

Poses (pure): strip pieces from the strip's deflection and slope at their centres; the mount from those at X_A; the
blade pieces from the absorber's z and L (a cantilever with a tip load up to the carrier, straight beyond it); the
carrier's track (the kinematic body the carrier's prismatic joint runs on: the blade's tangent at the carrier); the
yoke (the mount translated by z); the stinger (the exciter's armature and push rod, translated by the deflection at
X_F). A pose is (x, y, z, qx, qy, qz, qw), turned about +y only.

The bench: between sweeps a Plant (absorber_model) runs with the loads pressing on the strip, the mount and the
absorber; SWEEP starts a Sweep (the model's) from the present state; its curve is recorded (public: export_sweeps,
as the analyzer stores it; private: the configuration, the state at the start, PEAK and FREQ as shown). Any touch of
the strip or the absorber during a sweep, or a change of its configuration, voids it (the analyzer shows Err); SWEEP
pressed while the carrier is not at rest in a detent shows Err and does nothing else. The valve keys act on their
press edge, not during a sweep.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

import numpy as np

try:                                      # in the bench image core is flat on the path; on the host a package
    import absorber_model as M
except ImportError:                       # pragma: no cover
    from benchmark.core import absorber_model as M

# --- layout --------------------------------------------------------------------------------------------------------
CLAMP_X, BEAM_Y = 0.68, -0.11
BEAM_TOP = 0.125
STRIP = (M.L_BEAM, M.B_BEAM, M.T_BEAM)
SEGMENTS = 40
SEGMENT_OVERLAP = 0.0004
MOUNT_X = CLAMP_X + M.X_A
STINGER_X = CLAMP_X + M.X_F
TIP_X = CLAMP_X + M.L_BEAM
MOUNT_LEN, MOUNT_W = 0.028, 0.044
JAW_T = 0.012                        # the mount's jaws above and below the strip
BLADE_H = 0.025                      # blade centre line above the strip top (mount frame)
ROOT_X = -MOUNT_LEN / 2              # the blade's clamp face (s = 0)
ROOT_CLAMP = 0.012                   # length of the blade root under the clamp plate (mount frame x ROOT_X..ROOT_X+)
BLADE_LEN = 0.100
BLADE_PIECES = 20
DETENTS_MM = tuple(M.BLADE_L_MM)
TRAVEL_MARGIN_M = 0.002              # the carrier's end stops this far beyond the first and last detent
# carrier (carrier frame: origin on the blade's centre line at the clamp point, x along the blade toward the mount)
CLAMP_BLOCK = (0.016, 0.020, 0.012)
RISER = (0.012, 0.016)               # x, y of the column from the clamp block to the tray
TRAY_Z = 0.018                       # tray underside above the blade line
TRAY = (0.030, 0.180, 0.004)         # x, y, base thickness
SOCKET_H = 0.016                     # socket depth above the base: a straight well holding the weight's foot, a 45-degree
SOCKET_Y = (-0.075, -0.025, 0.025, 0.075)     # chamfered mouth above it
WELL_HALF, MOUTH_HALF = 0.0080, 0.0125   # the well 1.75 mm around the 12.5 mm foot (the arm's placing accuracy)
# hold-downs (real retention): one spring slide clamp per socket on the tray's -x side. A
# slide on a rail beside the socket carries a forked nose (two prongs straddling the weight's post) and is pressed down
# by a vertical spring. Its runner rides the rail's high section, holding the prongs clear above the weight's body
# while the slide moves; near the closed end the rail drops down a slope into a notch, so the spring lowers the prongs
# straight down onto the top of the body and presses the weight into its socket (no sideways wedge on the weight), and
# the runner held in the notch latches the slide closed. Pulled back, the runner climbs the slope and lifts the prongs
# off. Slide frame: origin at the socket's centre, on the prongs' underside with the slide at its datum height; x
# forward (closed at 0), z up.
HOLD_STROKE = 0.028                  # the slide's travel, open (-) to closed (0)
HOLD_REST_DZ = -0.001                # the datum: the prongs' underside this far below the top of a seated weight's body
HOLD_HIGH = 0.002                    # the slide's height on the rail's high section (prongs 1 mm above the body)
HOLD_FLOOR = -0.0008                 # the slide's height with its runner on the notch floor (no weight in the socket)
HOLD_SLOPE_DEG = 25.0                # the rail's slope from the high section down into the notch: the latch. At 10 deg
                                     # (2 N to pull back) the tray's shaking on the stops (100 g, valve 0) walked the
                                     # slides 7 mm back off the weights and a 40 g weight tipped out; 25 deg holds about
                                     # 4 N in the notch and takes about 5.5 N to pull back, still under the carrier's
                                     # detent (8 N)
HOLD_RUNNER = ((-0.055, -0.047), 0.004, 0.004)   # the runner pin across the bar's back: x range (its diameter), half
                                                 # length, its underside's depth below the bar
HOLD_LIFT = 0.006                    # the spring's travel up
HOLD_BACKSTOP = 0.003                # the joint's lower z stop under the datum (below the notch floor: never reached)
HOLD_SPRING = (800.0, -0.010, 3.0)   # stiffness N/m, target below the datum (preload), damping N s/m: 9.6 N on the high
                                     # rail, about 7.5 N on a seated weight (its foot gives some 1.5 mm), 7.4 N on the
                                     # notch floor (a 40 g weight on the worst-shaken setting stays within 5 deg)
HOLD_PAD_MU = (0.08, 0.06)           # the slide's PTFE-faced undersides (friction taken as the lower of the pair): it
                                     # stays where it is left on the high rail (preload x mu, 0.6 N); pulling it back
                                     # up the slope takes about 5.5 N (HOLD_SLOPE_DEG), under the carrier's detent
HOLD_MASS = 0.005                    # the slide: slim aluminium (bar, prongs and tab)
TOUCH_DAMPING_NS_M = 10.0            # a finger on the absorber (on the tray, a weight standing in it, the carrier or a hold-down)
                                     # damps its motion as a hand on a ringing part does: about 0.8 of critical at 100 g. The
                                     # finger's push itself reaches the beam slewed (backend), a step late, and against a
                                     # lightly damped absorber ringing into a fingertip (a hold-down's latch letting go under
                                     # the push) that lag alone fed it: seen 0.2 mm rising to 1.3 mm, the carrier knocked off
                                     # its detent
HOLD_PRONG_Y = (0.0065, 0.0105)      # each prong's |y| range: the slot clears the post (4 mm) and the foot's play
HOLD_PRONG_X = (-0.017, 0.008)       # the bar starts behind the body's edge (14 mm + play)
HOLD_RAMP = (0.002, 0.0005)          # the prongs' front: a small lead-in chamfer, the underside rising over this length
                                     # by this much
HOLD_T = 0.005                       # the slide's thickness
HOLD_BAR_X = (-0.055, -0.017)
HOLD_BAR_HALF_Y = 0.006
HOLD_TAB = ((-0.055, -0.051), 0.006, 0.030)   # the push tab at the slide's back: x range, half y, top above the origin
HOLD_RAIL_X = (-0.088, -0.019)       # the rail (carrier): under the slide, on a riser clear of the weight's body
HOLD_RAIL_HALF_Y = 0.005
HOLD_RISER_X = (-0.028, -0.019)
PIN_R = 0.002
PIN_Y = (-0.033, 0.043)              # bumper pins: -y through its cheek; +y through its cheek and the yoke
HANDLE_Y, HANDLE = 0.106, (0.012, 0.010, 0.050)   # push handle post: y, size (x, y, height above the tray base;
                                                  # a hand pushing it passes clear above the weights in the tray)
CARRIER_MASS = 0.040                 # the carrier's rigid body: POM clamp and tray, steel pins; the model's
                                     # moving mass absorber_model.M_CARRIER is this and the four slides (x mass_factor)
# cheeks and stops (mount frame)
CHEEK_Y = (0.027, 0.031)
CHEEK_X = (0.014, ROOT_X - 0.092)
CHEEK_Z = (BLADE_H - 0.010, BLADE_H + 0.009)
STOP_RECESS = 0.0006                 # the rubber's collider lies this far behind its face (the stop's compression)
RUBBER_T = 0.0015
# damper (mount frame): cylinder, valve head, keys; the yoke (moves with z)
CYL_XY, CYL_R = (0.024, 0.038), 0.008
CYL_Z = (BLADE_H + 0.012, BLADE_H + 0.034)   # its bottom clear of the yoke's top at the upper stop
HEAD = ((0.018, 0.046), (0.020, 0.088), (CYL_Z[1], CYL_Z[1] + 0.010))   # x, y, z ranges
KEY_XY = {'valve:down': (0.032, 0.035), 'valve:up': (0.032, 0.073)}
VALVE_LCD_XY = (0.032, 0.054)
YOKE_Y = (0.036, 0.040)
YOKE_X = (0.030, ROOT_X - 0.095)
YOKE_Z = (BLADE_H - 0.007, BLADE_H + 0.007)
YOKE_SLOT = PIN_R + 0.0003
ROD_R = 0.003
KEY_PRELOAD_M = 0.003                # valve keys: spring preload (the kit's cap otherwise); holds them under vibration
# exciter (world)
SHAKER = {'xy': (STINGER_X, BEAM_Y), 'r': 0.035, 'z': (0.022, 0.082), 'base': (0.110, 0.090, 0.012)}
ARMATURE = (0.012, 0.088, 0.004)     # disc radius, bottom, thickness (rest)
STINGER_R = 0.0015
STINGER_CLAMP = (0.010, 0.010, 0.004)
# vise, analyzer, amplifier, rack (world)
VISE = {'x': (0.600, CLAMP_X), 'y': 0.040, 'base': (0.120, 0.130, 0.015)}
ANALYZER = {'xy': (1.290, -0.020), 'size': (0.160, 0.130, 0.055)}
SWEEP_UV = (0.045, -0.035)           # SWEEP key from the analyzer centre (robot frame u right = -x, v away = -y)
SCREEN_UV, SCREEN = (0.0, 0.025), (0.120, 0.050)
AMPLIFIER = {'xy': (STINGER_X, -0.270), 'size': (0.130, 0.100, 0.070)}
RACK = {'xs': (0.58, 0.68, 0.78, 0.88), 'y': 0.045, 'base': (0.38, 0.07, 0.012)}
ACCEL = (0.010, 0.008, TIP_X - 0.007)   # accelerometer: across flats, height, x on the strip top
WEIGHT_IDS = ('w10', 'w20', 'w30', 'w40')
SWEEP_ID = 'sweep'
VALVE_KEYS = {'valve:down': -1, 'valve:up': +1}

# --- bench behaviour -----------------------------------------------------------------------------------------------
DETENT_TOL_M = 0.0005                # SWEEP: the carrier within this of a detent (its position averaged over the
                                     # speed's window: shaken by the blade it rocks a few tenths of a mm about it), ...
DETENT_SPEED_M_S = 0.002             # ... slower than this
TOUCH_N = 0.02                       # a contact pressing harder than this touches the strip or the absorber
LOG_EVERY_S = 0.1


def _quat_y(angle):
    return (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))


def _turn(angle, x, z):
    """(x, z) of the local vector (x, z) turned about +y by angle (x toward -z for angle > 0)."""
    c, s = math.cos(angle), math.sin(angle)
    return c * x + s * z, -s * x + c * z


def segment_x(i):
    """Centre of strip piece i, distance from the clamp edge."""
    return (i + 0.5) * STRIP[0] / SEGMENTS


def blade_s(i):
    return (i + 0.5) * BLADE_LEN / BLADE_PIECES


def strip_poses(table_z, w, slope):
    """World poses of the strip pieces (w up +, slope dw/dx at each centre)."""
    t = STRIP[2]
    return [(CLAMP_X + segment_x(i), BEAM_Y, table_z + BEAM_TOP - t / 2 + wi, *_quat_y(-math.atan(si)))
            for i, (wi, si) in enumerate(zip(w, slope))]


def mount_frame(table_z, w_a, slope_a):
    """(origin (x, y, z), angle about +y) of the mount frame."""
    return (MOUNT_X, BEAM_Y, table_z + BEAM_TOP + w_a), -math.atan(slope_a)


def in_frame(frame, x, y, z, turn=0.0):
    """World pose of a point (x, y, z) of a frame, turned further by `turn` about +y."""
    (ox, oy, oz), a = frame
    dx, dz = _turn(a, x, z)
    return (ox + dx, oy + y, oz + dz, *_quat_y(a + turn))


def blade_shape(z, L, s):
    """Height (m, up +) and slope dz/ds of the blade at s for the absorber at z, the carrier at L: a cantilever with a
    tip load up to L, straight beyond."""
    s = np.asarray(s, float)
    near = s <= L
    h = np.where(near, z * s ** 2 * (3 * L - s) / (2 * L ** 3), z + 1.5 * z / L * (s - L))
    d = np.where(near, z * (6 * L * s - 3 * s ** 2) / (2 * L ** 3), 1.5 * z / L)
    return h, d


def blade_poses(frame, z, L):
    """World poses of the blade pieces (centres at blade_s(i)); a piece turns with the blade's slope (dz/ds > 0: the
    end away from the mount higher, a turn about +y by atan(dz/ds) in the mount frame, whose x points to the mount)."""
    s = np.array([blade_s(i) for i in range(BLADE_PIECES)])
    h, d = blade_shape(z, L, s)
    return [in_frame(frame, ROOT_X - si, 0.0, BLADE_H + hi, math.atan(di)) for si, hi, di in zip(s, h, d)]


def carrier_pose(frame, z, L):
    """World pose of the carrier frame (the clamp point on the blade, turned with the blade's tip slope)."""
    return in_frame(frame, ROOT_X - L, 0.0, BLADE_H + z, math.atan(1.5 * z / L))


def track_pose(frame, z, L):
    """World pose of the carrier's track: the blade's tangent at the carrier, its origin where that tangent line meets
    the clamp face (s = 0), its x along the line toward the mount."""
    slope = 1.5 * z / L
    return in_frame(frame, ROOT_X, 0.0, BLADE_H + z - slope * L, math.atan(slope))


def yoke_pose(frame, z):
    return in_frame(frame, 0.0, 0.0, z)


def stinger_pose(table_z, w_f):
    return (STINGER_X, BEAM_Y, table_z + w_f, 0.0, 0.0, 0.0, 1.0)


def carrier_s(frame, carrier_xyz):
    """The carrier's clamp point s (m from the clamp face) from its world position."""
    (ox, oy, oz), a = frame
    dx, dz = float(carrier_xyz[0]) - ox, float(carrier_xyz[2]) - oz
    lx, _ = _turn(-a, dx, dz)
    return ROOT_X - lx


def nearest_detent(s_m):
    mm = min(DETENTS_MM, key=lambda d: abs(d / 1000.0 - s_m))
    return mm, s_m - mm / 1000.0


def cable_points(tip_xyz, end_xyz, n=24, sag=0.05):
    """A hanging cable from the accelerometer (tip_xyz) to the analyzer's socket: a parabola sagging `sag` below the
    straight line, never below the table (render only)."""
    a, b = np.asarray(tip_xyz, float), np.asarray(end_xyz, float)
    t = np.linspace(0.0, 1.0, n)
    p = a[None] + (b - a)[None] * t[:, None]
    p[:, 2] -= sag * 4 * t * (1 - t)
    p[:, 2] = np.maximum(p[:, 2], min(a[2], b[2]) - sag)
    return p


# --- displays ------------------------------------------------------------------------------------------------------
def peak_text(value_mm):
    """Five cells: tens, units | three decimals (blank tens below 10 mm); ' Err ' for a void sweep; blank before any."""
    if value_mm is None:
        return '     '
    if value_mm == 'Err':
        return ' Err '
    v = min(round(float(value_mm), 3), 99.999)
    digits = f'{v:06.3f}'.replace('.', '')
    return (' ' if digits[0] == '0' else digits[0]) + digits[1:]


def hz_text(value_hz):
    """Three cells: tens, units | one decimal."""
    if value_hz is None:
        return '   '
    digits = f'{min(round(float(value_hz), 1), 99.9):04.1f}'.replace('.', '')
    return (' ' if digits[0] == '0' else digits[0]) + digits[1:]


class AbsorberBench:
    """The bench's state. config: {'hidden': the model's hidden values, 'weights_g': {weight id: g},
    'carrier_mm': the detent the carrier starts on, 'valve': the valve's starting setting}."""

    def __init__(self, episode_id, config, *, physics_hz=240):
        if physics_hz not in (240, 480):
            raise ValueError('unsupported physics clock')
        self.episode_id = episode_id
        self.physics_hz = physics_hz
        self.h = M.hidden_of(config['hidden'])
        self.weights_g = {k: float(v) for k, v in config['weights_g'].items()}
        self.valve = int(config.get('valve', 0))
        self.length_m = int(config['carrier_mm']) / 1000.0
        self.plant = M.Plant(self.h, 0.0, self.length_m * 1000, self.valve)
        xs = [segment_x(i) for i in range(SEGMENTS)]
        self.seg_w, self.seg_s = M.mode_shapes(self.h, xs)
        (self.a_w, self.a_s), (self.f_w, _) = M.mode_shapes(self.h, [M.X_A]), M.mode_shapes(self.h, [M.X_F])
        self.tick = 0
        self.tray = {}                      # weight id -> socket, the weights standing in the tray's sockets
        self.carrier = {'s_m': self.length_m, 'v_m_s': 0.0}
        self.pressed = set()
        self.run = None                     # the sweep going on: {'sweep': M.Sweep, 'record': ..., 'tick0': ...}
        self.sweeps, self.sweeps_private = [], []
        self.display = {'state': 'idle', 'peak': None, 'hz': None}
        self.transitions, self.actions, self.log = [], [], []
        self._next_log = 0
        self._config = self.configuration()
        self.start_configuration = deepcopy(self._config)

    def process_extra(self):
        """For the process record (bench_logs.json): the absorber as it stood when the episode began."""
        return {'start_configuration': deepcopy(self.start_configuration)}

    @property
    def t_s(self):
        return self.tick / self.physics_hz

    # --- configuration -------------------------------------------------------------------------------------------
    def mass_g(self):
        return float(sum(self.weights_g[w] for w in self.tray))

    def configuration(self):
        """What the absorber is set to now: the weights in the tray, their total mass, the carrier's detent (None
        between detents or while it moves) and the valve setting."""
        mm, off = nearest_detent(self.carrier.get('s_avg_m', self.carrier['s_m']))
        locked = abs(off) <= DETENT_TOL_M and abs(self.carrier['v_m_s']) <= DETENT_SPEED_M_S
        return {'weights': sorted(self.tray), 'mass_g': self.mass_g(), 'length_mm': mm if locked else None,
                'valve': self.valve}

    def strip_state(self):
        """Deflection and slope of the strip pieces, the mount's deflection and slope, the stinger's deflection (m, up
        +), the absorber's z and the carrier's s (what the backend poses the moving parts from)."""
        if self.run is not None:
            sw = self.run['sweep']
            q, z = sw.q[:, 0], float(sw.z[0])
        else:
            q, z = self.plant.q, float(self.plant.z)
        return {'seg_w': (self.seg_w @ q).tolist(), 'seg_s': (self.seg_s @ q).tolist(), 'w_a': float(self.a_w[0] @ q),
                's_a': float(self.a_s[0] @ q), 'w_f': float(self.f_w[0] @ q), 'z': z, 'L': self.carrier['s_m']}

    # --- stepping -----------------------------------------------------------------------------------------------
    def step(self, tick, world):
        """world: {'tray': {weight: socket} standing in the sockets, 'carrier': {'s_m', 'v_m_s'}, 'pressed': set of
        key ids, 'loads': {'strip': [(x_m, F_N up +)], 'mount_N': up +, 'absorber_N': up +}, 'touch': [who touches the
        strip or the absorber, pressing at least TOUCH_N]}."""
        if tick != self.tick + 1:
            raise ValueError('absorber bench requires every consecutive physics tick')
        self.tick = tick
        self.tray = dict(world.get('tray') or {})
        if world.get('carrier') is not None:
            c = world['carrier']
            self.carrier = {'s_m': float(c['s_m']), 'v_m_s': float(c['v_m_s']),
                            's_avg_m': float(c.get('s_avg_m', c['s_m']))}
        pressed = set(world.get('pressed', ()))
        edges = pressed - self.pressed
        self.pressed = pressed
        for key in sorted(edges):
            self._press(key)
        config = self.configuration()
        if config != self._config:
            self.transitions.append({'tick': tick, 't_s': round(self.t_s, 6), **config,
                                     'carrier_mm': round(self.carrier['s_m'] * 1000, 3)})
            self._config = config
        touch = sorted(world.get('touch') or ())
        if self.run is not None:
            self._sweep_tick(config, touch)
        else:
            if not 0.5 * DETENTS_MM[0] <= self.carrier['s_m'] * 1000 <= 1.5 * DETENTS_MM[-1]:
                raise RuntimeError(f"the absorber's carrier left its track (s = {self.carrier['s_m'] * 1000:.1f} mm)")
            self.plant.set_config(config['mass_g'], self.carrier['s_m'] * 1000, self.valve)
            f_modal, f_abs = self._loads(world.get('loads') or {})
            self.plant.c_touch = TOUCH_DAMPING_NS_M if (world.get('loads') or {}).get('robot_on_absorber') else 0.0
            self.plant.advance_to(self.t_s, f_modal, f_abs)
        if tick >= self._next_log:
            self._next_log = tick + round(LOG_EVERY_S * self.physics_hz)
            st = self.strip_state()
            self.log.append({'t_s': round(self.t_s, 3), 'tip_mm': round(float(self._tip()) * 1000, 4),
                             'z_mm': round(st['z'] * 1000, 4), 'carrier_mm': round(self.carrier['s_m'] * 1000, 3),
                             'valve': self.valve, 'tray': sorted(self.tray), 'state': self.display['state']})

    def _tip(self):
        if self.run is not None:
            return float(self.run['sweep'].tip_mm()[0]) / 1000.0
        return float(self.plant.pt @ (self.plant.q - self.plant.static_state()['q']))

    def _loads(self, loads):
        """Modal forces (sum F phi(x)) of the loads on the strip and the mount, and the force on the absorber."""
        f = np.zeros(len(self.plant.om))
        strip = loads.get('strip') or []
        if strip:
            xs = np.array([x for x, _ in strip], float)
            F = np.array([F for _, F in strip], float)
            w, _ = M.mode_shapes(self.h, np.clip(xs, 0.0, M.L_BEAM))
            f += F @ w
        f += float(loads.get('mount_N', 0.0)) * self.a_w[0]
        return f, float(loads.get('absorber_N', 0.0))

    def _press(self, key):
        row = {'tick': self.tick, 't_s': round(self.t_s, 6), 'key': key}
        if key in VALVE_KEYS:
            if self.run is not None:
                row['ignored'] = 'sweep running'
            else:
                self.valve = min(9, max(0, self.valve + VALVE_KEYS[key]))
            row['valve'] = self.valve
        elif key == SWEEP_ID:
            if self.run is not None:
                row['ignored'] = 'sweep running'
            else:
                row.update(self._start())
        self.actions.append(row)

    def _start(self):
        config = self.configuration()
        n = len(self.sweeps) + 1
        eid = f'{self.episode_id}:sweep:{n}'
        if config['length_mm'] is None:
            self.display = {'state': 'err', 'peak': 'Err', 'hz': None}
            return {'started': False, 'reason': 'carrier not at rest in a detent'}
        state = self.plant.state()
        nm = len(self.plant.om)
        initial = {'q': state['q'].reshape(nm, 1), 'qd': state['qd'].reshape(nm, 1), 'z': [state['z']],
                   'zd': [state['zd']]}
        n_sub = int(np.max(M.substeps(self.h, config['mass_g'], config['length_mm'], config['valve'])))
        sweep = M.Sweep(self.h, config['mass_g'], float(config['length_mm']), config['valve'], curve=True,
                        n_sub=n_sub, initial=initial)
        dev = self.plant.deviation()
        record = {'evidence_id': eid, 'started_s': round(self.t_s, 4), 'ended_s': None, 'status': 'running'}
        private = {'evidence_id': eid, 'config': deepcopy(config), 'carrier_mm': round(self.carrier['s_m'] * 1000, 4),
                   'start_deviation': {'tip_mm': round(dev[0] * 1000, 6), 'z_mm': round(dev[1] * 1000, 6),
                                       'tip_mm_s': round(dev[2] * 1000, 6), 'z_mm_s': round(dev[3] * 1000, 6)},
                   'n_sub': n_sub, 'touch': None}
        self.sweeps.append(record)
        self.sweeps_private.append(private)
        self.run = {'sweep': sweep, 'record': record, 'private': private, 'tick0': self.tick, 'config': config}
        self.display = {'state': 'run', 'peak': 0.0, 'hz': M.F_LO, 'sweep': eid}
        return {'started': True, 'evidence_id': eid}

    def _sweep_tick(self, config, touch):
        run = self.run
        sw = run['sweep']
        sw.advance_to((self.tick - run['tick0']) / self.physics_hz)
        changed = {k: config[k] for k in ('weights', 'mass_g', 'length_mm')} != \
            {k: run['config'][k] for k in ('weights', 'mass_g', 'length_mm')}
        if touch or changed:
            import os
            import sys
            if os.environ.get('EMBODIEDLAB_ABSORBER_TRACE', '0') == '1':
                print(f"[absorber-void] sweep t {sw.t:.3f} touch {touch} config {config} at start {run['config']} "
                      f"carrier s {self.carrier['s_m'] * 1000:.3f} avg {self.carrier.get('s_avg_m', 0.0) * 1000:.3f} mm v {self.carrier['v_m_s'] * 1000:.2f} mm/s",
                      file=sys.stderr, flush=True)
            self._end('void', touch=touch or ['configuration changed'])
            return
        self.display = {'state': 'run', 'peak': float(sw.peak[0]) * 1000, 'hz': sw.frequency(),
                        'sweep': run['record']['evidence_id']}
        if sw.done:
            self._end('complete')

    def _end(self, status, touch=None):
        run, self.run = self.run, None
        sw, record, private = run['sweep'], run['record'], run['private']
        self.plant.set_state(sw.state())
        self.plant.t = (run['tick0'] / self.physics_hz) + sw.i * sw.dt
        record['ended_s'] = round(self.t_s, 4)
        record['status'] = status
        if status == 'complete':
            r = sw.result()
            record['curve'] = {'f_hz': [round(float(f), 2) for f in r['curve_hz']],
                               'amplitude_mm': [round(float(a), 4) for a in r['curve_mm'][0]]}
            private['peak_mm_exact'] = float(r['peak_mm'][0])
            private['peak_hz_exact'] = float(r['peak_hz'][0])
            private['peak_mm'] = round(float(r['peak_mm'][0]), 3)            # as the screen shows them
            private['peak_hz'] = round(float(r['peak_hz'][0]), 1)
            self.display = {'state': 'done', 'peak': float(r['peak_mm'][0]), 'hz': float(r['peak_hz'][0]),
                            'sweep': record['evidence_id']}
        else:
            record['curve'] = None
            private['touch'] = list(touch or [])
            private['void_at_s'] = round(sw.t, 4)
            self.display = {'state': 'err', 'peak': 'Err', 'hz': None, 'sweep': record['evidence_id']}
        record['sha256'] = hashlib.sha256(json.dumps({k: v for k, v in record.items() if k != 'sha256'},
                                                     sort_keys=True).encode()).hexdigest()

    # --- what the instruments show --------------------------------------------------------------------------------
    def analyzer_text(self):
        """(PEAK cells, Hz cells, RUN lamp)."""
        d = self.display
        return peak_text(d['peak']), hz_text(d['hz']), d['state'] == 'run'

    def valve_text(self):
        return str(self.valve)

    def export_sweeps(self):
        """Every sweep of this episode that has ended, as the analyzer stores it: evidence_id, status, start and end
        times and the amplitude-frequency curve (PEAK and FREQ are on its screen only; the valve setting on the
        damper's display)."""
        return {'sweeps': deepcopy([r for r in self.sweeps if r['status'] != 'running']),
                'running': self.run is not None}

    def observation_record(self, tick):
        peak, hz, lamp = self.analyzer_text()
        mm, off = nearest_detent(self.carrier['s_m'])
        return {'tick': tick, 't_s': round(tick / self.physics_hz, 6), 'analyzer': {'peak': peak, 'hz': hz, 'run': lamp},
                'valve_lcd': self.valve_text(), 'configuration': self.configuration(),
                'carrier': {'s_mm': round(self.carrier['s_m'] * 1000, 4), 'nearest_mm': mm,
                            'offset_mm': round(off * 1000, 4), 'v_mm_s': round(self.carrier['v_m_s'] * 1000, 4)},
                'display': deepcopy(self.display)}

    def private_record(self):
        return {'sweeps': deepcopy(self.sweeps_private), 'transitions': deepcopy(self.transitions),
                'actions': deepcopy(self.actions), 'log': deepcopy(self.log), 'configuration': self.configuration(),
                'valve': self.valve, 'running': self.run is not None}

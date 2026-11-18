"""Public apparatus geometry of the spring launcher, in metres (bench-local frame); no sampled private parameters.

Local x runs along the launch direction, z up; the origin lies under the muzzle, 4 cm above the table top. The
launcher is a spring plunger in a cradle tilted 10 deg up: the robot cocks it by pushing the side knob back until the
ratchet pawl latches a notch, sets the ball in the cradle (it rolls back against the plunger face), and presses the
trigger. The ball leaves the plunger at the spring's free length and flies off the muzzle onto the receiving deck.
"""
import math

BALL_RADIUS = 0.020
BALL_MASS = 0.025
PLUNGER_MASS = 0.040
TILT_DEG = 10.0
TILT = math.radians(TILT_DEG)
AXIS = (math.cos(TILT), 0.0, math.sin(TILT))          # launch direction, up the cradle
MUZZLE = (0.0, 0.0, 0.22)                              # ball centre over the rail ends
MUZZLE_GAP = 0.030                                     # ball travel from free length to the rail ends
RAIL_RADIUS = 0.003
RAIL_LENGTH = 0.12
RAIL_FRICTION = 0.05
FORWARD_LIMIT = 0.025                                  # plunger travel past the free length (stop)
REAR_LIMIT = 0.062                                     # deepest plunger travel (stop)
NOTCHES = tuple(round(0.015 + 0.005 * i, 3) for i in range(8))     # 15 ... 50 mm
TARGET_NOTCHES = (0.030, 0.035, 0.040)
LATCH_HYSTERESIS = 0.0006                              # travel past a notch before the pawl drops into it
DASHPOT_DELAY_S = 0.30                                 # the plunger dashpot engages this long after a release
DASHPOT_DAMPING = 3.0                                  # N s/m
KNOB_OFFSET_Y = 0.050                                  # cocking knob beside the cradle, on the robot's side
KNOB_TOP_ABOVE_AXIS = 0.075
KNOB_SIZE = 0.016
RECEIVER_CENTER_Z = 0.06
DECK_Z = -0.020
DECK_FLAT_END = 0.42
POCKET_END = 0.50
DECK_HALF_WIDTH = 0.12
CUP_RIM_Z = 0.026
CUP_CONTACT_TOLERANCE_M = 0.001
CUP_INNER_WIDTH = 0.100
CUP_GRIP_Y = .066
CUP_POST_TOP_Z = .120
CUP_JAW_Z = CUP_POST_TOP_Z + .018
GATE_X = (-0.005, 0.025)                               # optical beams: one over the rail ends, one in free flight
BALL_PAD_LOCAL = (-0.28, 0.20)
CUP_PAD_LOCAL = (0.15, -0.27)                         # far side of the deck: the post faces the deck, 0.68 m ahead of the robot
TRIGGER_LOCAL = (-0.16, 0.18, 0.045)
EXIT_X = MUZZLE[0]


def along(s, base=None):
    """Point s metres along the launch axis from base (default: the muzzle point)."""
    base = MUZZLE if base is None else base
    return tuple(b + s * a for b, a in zip(base, AXIS))


def face_rest():
    """Plunger face at the spring's free length (ball centre one radius ahead of it)."""
    return along(-(BALL_RADIUS + MUZZLE_GAP))


def ball_rest(compression):
    """Ball centre resting against the plunger face latched at a compression."""
    return along(BALL_RADIUS - compression, face_rest())


def cradle_loading_point(compression, ahead=0.008):
    """Bottom of the ball's footprint for setting it in the cradle `ahead` metres in front of the latched face."""
    c = ball_rest(compression - ahead)
    return (c[0], c[1], c[2] - BALL_RADIUS)


def recovery_floor_z(x, y):
    """Flat cup deck followed by a passive, shallow V recovery pocket at the far end."""
    return DECK_Z - .010 * max(0., min(1., (x - DECK_FLAT_END) / (POCKET_END - DECK_FLAT_END - .01))) * max(0., 1 - abs(y) / DECK_HALF_WIDTH)


def vacuum_range(vx, vz, center_z, datum_z=RECEIVER_CENTER_Z):
    if vx <= 0 or center_z <= datum_z:
        raise ValueError("invalid forward flight geometry")
    t = (vz + math.sqrt(vz * vz + 2 * 9.81 * (center_z - datum_z))) / 9.81
    return vx * t


# Grip the upper hemisphere: the jaw link origin lies 54 mm above the fingertip, and the pads clear the cradle rails.
BALL_JAW_OFFSET = 0.054 - BALL_RADIUS + 0.011

"""Air-track hand controls without Isaac: start catches, the spring launcher's pawl, the release lever.

Track coordinates: X along the rail from the scale origin (world X 0.45), Y and Z as the world.

- Start catches. One spring catch per glider at its start mark (A 0.100 m, B 0.520 m). While the release lever is at
  HOLD the catches are down: a glider brought to rest with its centre within CATCH_CAPTURE_M of its mark (moving at
  most CATCH_CAPTURE_SPEED_M_S) drops its catch pin into the catch notch and is held there by the catch spring
  (stiffness CATCH_STIFFNESS_N_PER_M, at most CATCH_MAX_N). A glider dragged more than CATCH_RELEASE_M out of the notch
  rides out of it. Pulling the release lever lifts both catches; a glider moving faster than the capture speed rides
  over a lowered catch.
- Spring launcher. A plunger on a guided rod behind A's start mark, pushed by a compression spring (no published
  constants). Pushing its knob back (towards -X) passes three notches; a pawl drops into the deepest notch passed
  (LATCH_HYSTERESIS beyond it) and holds the plunger there. The pawl is one-way: the knob can always be pushed deeper.
  The pawl only drops while the release lever is at HOLD. Moving the lever to RELEASE lifts the pawl: the spring drives
  the plunger forward and its face strikes A (a dead, non-bouncing face) at the spring's free length. A dashpot on the
  rod is disengaged for DASHPOT_DELAY_S after the release and otherwise damps the plunger, so the plunger face is also
  the absorbing left-end bumper. After a shot the launcher rests uncocked at free length.
- Release lever. On the robot's side of the bench (within easy reach, clear of the track). Two detents, HOLD (rest)
  and RELEASE, 40 mm apart along +Y (towards the robot), held by a detent
  spring (damper_lever.Lever runtime). It trips when its carriage passes the midpoint (the detent index changes).
"""
from __future__ import annotations

MARKS_M = (0.10, 0.52)                  # start marks of A and B: the catch positions (glider centres)
GLIDER_LENGTH_M = 0.100
CATCH_CAPTURE_M = 0.001
CATCH_CAPTURE_SPEED_M_S = 0.05
CATCH_RELEASE_M = 0.004
CATCH_STIFFNESS_N_PER_M = 2000.0
CATCH_DAMPING_NS_PER_M = 40.0
CATCH_MAX_N = 6.0

# Launcher (track X of the plunger face at the spring's free length; the face is the left-end bumper)
FACE_FREE_X_M = 0.0495                  # 0.5 mm behind A's rear face when A stands at its mark
PLUNGER_MASS_KG = 0.10
SPRING_N_PER_M = 90.0                   # private to the bench; never published
NOTCHES_M = (0.020, 0.032, 0.044)       # compression at notch 1, 2, 3
LATCH_HYSTERESIS_M = 0.0006
UNCOCKED_M = 0.003                      # within this of free length (pawl not latched) the launcher is uncocked
FORWARD_LIMIT_M = 0.025                 # plunger travel past free length (rod stop)
REAR_LIMIT_M = 0.050                    # deepest compression (rod stop)
DASHPOT_NS_PER_M = 120.0                # strong enough that a glider stopped by the plunger face is not pushed off again
DASHPOT_DELAY_S = 0.30
LATCH_STIFFNESS_N_PER_M = 2.0e4
LATCH_DAMPING_NS_PER_M = 200.0

# Release lever (track coordinates of detent HOLD; RELEASE lies LEVER_SPACING_M further along +Y)
LEVER_DETENT0 = (0.83, 0.23)            # on the table on the robot side, right of the robot base
LEVER_SPACING_M = 0.04
LEVER_LABELS = ('HOLD', 'RELEASE')


def catch_engages(down, offset_m, speed_m_s):
    """A free glider drops into its lowered catch: centre within CATCH_CAPTURE_M of the mark, nearly at rest."""
    return bool(down) and abs(offset_m) <= CATCH_CAPTURE_M and abs(speed_m_s) <= CATCH_CAPTURE_SPEED_M_S


def catch_holds(offset_m):
    """An engaged glider stays caught until it is dragged CATCH_RELEASE_M out of the notch."""
    return abs(offset_m) <= CATCH_RELEASE_M


def notch_number(compression_m):
    """1, 2 or 3 for a latched compression; None otherwise."""
    for k, d in enumerate(NOTCHES_M, 1):
        if compression_m is not None and abs(compression_m - d) < 1e-9:
            return k
    return None


class Pawl:
    """The launcher's ratchet pawl and the plunger drive it implies (one-way latch, as the projectile_landing launcher).

    compression: plunger travel behind free length (m, > 0 = cocked back). armed: the pawl may drop (lever at HOLD and
    not within the release window)."""

    def __init__(self):
        self.latched = None            # compression of the notch the pawl holds, or None

    def update(self, compression_m, armed):
        """Drop the pawl into the deepest notch passed; returns the new notch number when it latches a deeper one."""
        if not armed:
            return None
        passed = [d for d in NOTCHES_M if compression_m >= d + LATCH_HYSTERESIS_M]
        if passed and (self.latched is None or passed[-1] > self.latched):
            self.latched = passed[-1]
            return notch_number(self.latched)
        return None

    def drive(self, compression_m, window):
        """(mode, stiffness N/m, target joint position m, damping N s/m) for the plunger joint (position = -compression).

        Latched and not pushed deeper than the notch: a stiff stop at the notch. Otherwise the spring (free length 0),
        damped by the dashpot except within the release window."""
        if self.latched is not None and compression_m < self.latched + 0.0003:
            return ('latched', LATCH_STIFFNESS_N_PER_M, -self.latched, LATCH_DAMPING_NS_PER_M)
        return ('spring', SPRING_N_PER_M, 0.0, 0.0 if window else DASHPOT_NS_PER_M)

    def trip(self):
        """The release lifts the pawl. Returns the notch number that was latched (None: uncocked)."""
        notch = notch_number(self.latched)
        self.latched = None
        return notch

    def cocked(self):
        return self.latched is not None

    def notch(self):
        return notch_number(self.latched)


def launch_speed_estimate(compression_m, mass_A_kg=0.2):
    """Ideal speed of A after a dead (e = 0) impact of the free plunger at free length (design check only; private)."""
    import math
    v_plunger = compression_m * math.sqrt(SPRING_N_PER_M / PLUNGER_MASS_KG)
    return PLUNGER_MASS_KG * v_plunger / (PLUNGER_MASS_KG + mass_A_kg)

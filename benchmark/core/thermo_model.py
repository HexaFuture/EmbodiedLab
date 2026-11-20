"""Thermal-kit models, headless: the heater-only block controller (shared by the calorimetry heater and the dry
block), the dry-block bench (thermistor_beta, rod_expansion) and the bench ohmmeter.

HeaterController: a block with a heater and no cooler on a set-point controller. Heating is rate-limited
(HEAT_RATE_K_S, about 10 K/min) and settles first-order (TAU_CONTROL_S) near the set point. With the set point below
the block, the heater is off and the block cools by Newton cooling toward the room with the hidden tau_cool_s, until
it reaches the set point (the controller holds it there) or the room. The set point moves by one step per press of
UP or DOWN within its range.

DryBlock: the controller's block, the carried probe (it follows the block with its own time constant while it stands
in the block's well, the room with TAU_AIR_S elsewhere) and, for kind 'rod', a rod in the block's heating jacket
(tau_rod) whose free end moves a digital indicator g0 + alpha L0 (T_rod - T_room) (mm).
kind 'thermistor': the probe is an NTC thermistor, R = R25 exp(B (1/T - 1/298.15 K)), wired to the bench ohmmeter. The
meter's test current heats the probe: its temperature follows T_env + P / delta (P = I^2 R in the probe, delta the
hidden dissipation constant, W/K) with the probe's own lag.

Ohmmeter: 4.5 digits (19999 counts), ranges 200 ohm, 2 kohm, 20 kohm, 200 kohm with test currents 1 mA, 1 mA,
100 uA, 10 uA. AUTO (the start) picks the most sensitive range the resistance fits; RANGE leaves autorange holding
the present range, every further press steps to the next range (after 200 kohm back to 200 ohm); AUTO returns to
autorange. A resistance above the range's full scale shows OL.
"""
from __future__ import annotations

import math

import numpy as np

T25_K = 298.15
HEAT_RATE_K_S = 10.0 / 60.0             # heating ramp limit: about 10 K per minute
TAU_CONTROL_S = 15.0                    # settling near the set point
TAU_AIR_S = 60.0
SET_MIN_C, SET_MAX_C, SET_STEP_C = 25.0, 80.0, 5.0
NOISE_K, RESOLUTION_K = 0.004, 0.01
R_NOISE_REL = 1.0e-4
INDICATOR_NOISE_MM, INDICATOR_RESOLUTION_MM = 0.0003, 0.001

# name, full scale (ohm), resolution (ohm), test current (A), decimals on the display, unit shown
OHM_RANGES = (('200', 199.99, 0.01, 1.0e-3, 2, 'Ohm'), ('2k', 1999.9, 0.1, 1.0e-3, 4, 'kOhm'),
              ('20k', 19999.0, 1.0, 1.0e-4, 3, 'kOhm'), ('200k', 199990.0, 10.0, 1.0e-5, 2, 'kOhm'))
RANGE_NAMES = tuple(r[0] for r in OHM_RANGES)


def thermistor_ohm(r25_ohm, beta_k, t_c):
    return r25_ohm * math.exp(beta_k * (1.0 / (t_c + 273.15) - 1.0 / T25_K))


class HeaterController:
    """Heater-only block on a set-point controller (see the module docstring)."""

    def __init__(self, *, t_room_c, tau_cool_s, set_min_c=SET_MIN_C, set_max_c=SET_MAX_C, set_step_c=SET_STEP_C,
                 start_set_c=None, t0_c=None):
        self.t_room, self.tau_cool = float(t_room_c), float(tau_cool_s)
        self.set_min, self.set_max, self.set_step = float(set_min_c), float(set_max_c), float(set_step_c)
        self.set_c = float(self.set_min if start_set_c is None else start_set_c)
        self.T = float(self.t_room if t0_c is None else t0_c)

    def press(self, button):
        """'up' or 'down'; a press beyond the range is ignored. Returns whether the set point changed."""
        step = {'up': self.set_step, 'down': -self.set_step}[button]
        new = min(self.set_max, max(self.set_min, self.set_c + step))
        changed = new != self.set_c
        self.set_c = new
        return changed

    def heating(self):
        return self.set_c > self.T

    def step(self, dt):
        s, T = self.set_c, self.T
        if s >= T:
            if s - T > HEAT_RATE_K_S * TAU_CONTROL_S:
                T = min(s, T + HEAT_RATE_K_S * dt)
            else:
                T = s + (T - s) * math.exp(-dt / TAU_CONTROL_S)
        else:
            newton = self.t_room + (T - self.t_room) * math.exp(-dt / self.tau_cool)
            held = s + (T - s) * math.exp(-dt / TAU_CONTROL_S)
            T = max(newton, held)
        self.T = T


class Ohmmeter:
    """The bench ohmmeter's state: autorange or a held range (see the module docstring)."""

    def __init__(self):
        self.auto = True
        self.range = len(OHM_RANGES) - 1

    def press(self, button):
        if button == 'auto':
            self.auto = True
        elif button == 'range':
            if self.auto:
                self.auto = False
            else:
                self.range = (self.range + 1) % len(OHM_RANGES)
        else:
            raise ValueError(button)

    def autorange(self, r_ohm):
        if self.auto:
            self.range = next((k for k, spec in enumerate(OHM_RANGES) if r_ohm <= spec[1]), len(OHM_RANGES) - 1)

    @property
    def current_a(self):
        return OHM_RANGES[self.range][3]

    def power_w(self, r_ohm):
        """Heat in the probe: I^2 R within the range; over range the source runs at its full-scale voltage."""
        _, full, _, current, _, _ = OHM_RANGES[self.range]
        if r_ohm <= full:
            return current * current * r_ohm
        return (current * full) ** 2 / r_ohm

    def read(self, r_ohm):
        """(value in ohm or None for OL, display text): the reading on the present range."""
        name, full, resolution, _, decimals, unit = OHM_RANGES[self.range]
        if r_ohm > full:
            return None, {'digits': '  OL ', 'point': None, 'unit': unit, 'auto': self.auto}
        counts = int(round(r_ohm / resolution))
        value = round(counts * resolution, 2)
        digits = f'{counts:5d}'
        point = 4 - decimals                     # the point stands after this cell (cells 0-4)
        digits = digits.replace(' ', '0') if point >= 0 else digits
        leading = list(digits)
        for i in range(point):                    # blank the leading zeros before the units cell
            if leading[i] == '0':
                leading[i] = ' '
            else:
                break
        return value, {'digits': ''.join(leading), 'point': point, 'unit': unit, 'auto': self.auto}


class DryBlock:
    def __init__(self, *, kind, t_room_c, tau_probe_s, tau_cool_s, r25_ohm=None, beta_k=None, delta_w_k=None,
                 tau_rod_s=None, alpha_per_k=None, rod_l0_mm=None, indicator_zero_mm=None, noise_seed=0):
        if kind not in ('thermistor', 'rod'):
            raise ValueError(kind)
        self.kind = kind
        self.t_room, self.tau_probe = float(t_room_c), float(tau_probe_s)
        self.controller = HeaterController(t_room_c=t_room_c, tau_cool_s=tau_cool_s)
        self.r25, self.beta, self.delta = r25_ohm, beta_k, delta_w_k
        self.tau_rod, self.alpha, self.l0, self.g0 = tau_rod_s, alpha_per_k, rod_l0_mm, indicator_zero_mm
        self.probe = self.rod = self.t_room
        self.meter = Ohmmeter() if kind == 'thermistor' else None
        self.rng = np.random.default_rng(noise_seed)
        if self.meter is not None:
            self.meter.autorange(self.probe_ohm())

    @property
    def block(self):
        return self.controller.T

    @property
    def set_c(self):
        return self.controller.set_c

    def press(self, button):
        if button in ('up', 'down'):
            return self.controller.press(button)
        self.meter.press(button)
        return True

    def probe_ohm(self):
        return thermistor_ohm(self.r25, self.beta, self.probe)

    def self_heating_k(self):
        """Steady temperature rise of the thermistor from the meter's test current at its present resistance."""
        if self.meter is None:
            return 0.0
        return self.meter.power_w(self.probe_ohm()) / self.delta

    def step(self, dt, probe_in_well):
        self.controller.step(dt)
        rise = self.self_heating_k()
        if probe_in_well:
            self.probe = self.block + rise + (self.probe - self.block - rise) * math.exp(-dt / self.tau_probe)
        else:
            self.probe = self.t_room + rise + (self.probe - self.t_room - rise) * math.exp(-dt / TAU_AIR_S)
        if self.kind == 'rod':
            self.rod = self.block + (self.rod - self.block) * math.exp(-dt / self.tau_rod)

    def _thermometer(self, t):
        v = t + self.rng.normal(0.0, NOISE_K)
        return round(round(v / RESOLUTION_K) * RESOLUTION_K, 2)

    def reading(self):
        """One logged reading: (public row values, display state).
        thermistor {T_C: the block's reference thermometer, R_ohm: the ohmmeter (None when OL)};
        rod {T_C: the probe thermometer, indicator_mm: the digital indicator}."""
        if self.kind == 'thermistor':
            r = self.probe_ohm() * (1.0 + self.rng.normal(0.0, R_NOISE_REL))
            self.meter.autorange(r)
            value, shown = self.meter.read(r)
            t = self._thermometer(self.block)
            return {'T_C': t, 'R_ohm': value}, {'block_C': t, 'ohmmeter': shown}
        g = self.g0 + self.alpha * self.l0 * (self.rod - self.t_room) + self.rng.normal(0.0, INDICATOR_NOISE_MM)
        g = round(round(g / INDICATOR_RESOLUTION_MM) * INDICATOR_RESOLUTION_MM, 3)
        t = self._thermometer(self.probe)
        return {'T_C': t, 'indicator_mm': g}, {'block_C': self._thermometer(self.block), 'probe_C': t,
                                                'indicator_mm': g}

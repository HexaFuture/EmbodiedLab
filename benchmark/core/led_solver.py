"""LED board of the circuit panel (led_planck), headless.

Series loop: supply u -> R_s -> LED -> ground, the logger (10 MOhm) across the LED. Each LED is a Shockley
junction with ideality factor n and a series resistance r:
    I = I_s (exp(V_j / (n V_T)) - 1),   V_LED = V_j + I r,
whose saturation current puts the junction voltage at E/e when I = 1 mA, E being the LED's turn-on energy:
E = hc/lambda - delta + epsilon (a common offset of the process plus a small scatter per LED; both hidden).
The supply follows the MODE detent: SWEEP ramps from 0 V at RAMP_V_PER_S from the moment the detent is reached
up to SOURCE_V and holds; ON holds SOURCE_V; OFF is 0 V. A LED selector in transition opens the loop.
CH1 = V_LED, CH2 = V across R_s (= I R_s).
"""
from __future__ import annotations

import math

H_EV_S = 4.135667696e-15
C_M_S = 299792458.0
HC_EV_NM = H_EV_S * C_M_S * 1e9            # 1239.84 eV nm
K_B_EV = 8.617333262e-5
T_K = 298.15
V_T = K_B_EV * T_K
SOURCE_V = 5.0
RAMP_V_PER_S = 0.25                        # 0 -> 5 V in 20 s
I_REF_A = 1.0e-3
LOGGER_INPUT_OHM = 1.0e7
MODES = ('SWEEP', 'OFF', 'ON')
RS_NOMINAL_OHM = (100, 330, 1000)


class LED:
    def __init__(self, energy_eV, n, r_ohm):
        if not 1.0 < energy_eV < 4.0 or not 1.0 <= n <= 4.0 or not 0 <= r_ohm < 100:
            raise ValueError('LED parameters out of range')
        self.energy_eV, self.n, self.r = float(energy_eV), float(n), float(r_ohm)
        self.nvt = self.n * V_T
        self.i_s = I_REF_A / math.expm1(self.energy_eV / self.nvt)

    def voltage(self, current_a):
        """V_LED at a forward current (A)."""
        return self.nvt * math.log1p(max(current_a, 0.0) / self.i_s) + current_a * self.r

    def current(self, supply_v, rs_ohm):
        """Loop current for a supply voltage through R_s (bisection on a monotonic function)."""
        if supply_v <= 0:
            return 0.0
        lo, hi = 0.0, supply_v / rs_ohm
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if mid * rs_ohm + self.voltage(mid) > supply_v:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)


class LEDModel:
    """Electrical model of one led_planck instance: the LEDs by selector label, the actual R_s values."""

    def __init__(self, leds, rs_actual_ohm):
        self.leds = dict(leds)
        if set(rs_actual_ohm) != set(RS_NOMINAL_OHM):
            raise ValueError('actual values required for every R_s detent')
        self.rs_actual_ohm = {int(k): float(v) for k, v in rs_actual_ohm.items()}
        self.capacitor = None

    def advance(self, switch, state, dt):
        """state: seconds since the MODE detent reached SWEEP (0 outside SWEEP)."""
        led, rs, mode = switch
        return state + dt if mode == 'SWEEP' else 0.0

    def supply(self, mode, state):
        if mode == 'SWEEP':
            return min(SOURCE_V, RAMP_V_PER_S * state)
        return SOURCE_V if mode == 'ON' else 0.0

    def current(self, switch, state):
        led, rs, mode = switch
        if led is None or rs is None or mode is None:
            return 0.0
        return self.leds[led].current(self.supply(mode, state), self.rs_actual_ohm[rs])

    def channels(self, switch, state):
        led, rs, mode = switch
        if led is None or rs is None or mode is None:
            return 0.0, 0.0
        i = self.current(switch, state)
        return self.leds[led].voltage(i), i * self.rs_actual_ohm[rs]


def turn_on_energy(wavelength_nm, delta_eV, scatter_eV):
    return HC_EV_NM / wavelength_nm - delta_eV + scatter_eV

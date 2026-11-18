"""Cell board of the circuit panel (battery_emf), headless.

Loop: cell (EMF E, internal resistance r) -> SWITCH -> load R_L -> shunt R_sh -> cell. The logger (10 MOhm, which
draws a negligible 0.15 uA) has CH1 across the cell terminals and CH2 across the shunt. With the switch OFF, or a
selector in transition (break before make), the loop is open and CH1 reads the open-circuit voltage E:
    I = E / (r + R_L + R_sh),   CH1 = U = E - I r,   CH2 = I R_sh.
The cell is a steady source at the bench's currents and times (no polarisation or temperature drift modelled).
"""
from __future__ import annotations

LOAD_NOMINAL_OHM = ('20', '10', '5', '3', '2')
SHUNT_NOMINAL_OHM = (0.1, 0.5, 1.0)
MODES = ('OFF', 'ON')


def _key(value):
    """Shunt detents are numbers; configuration records (JSON) carry them as strings."""
    return f'{float(value):g}'


class CellModel:
    """Electrical model of one battery_emf instance: the cell, actual load and shunt values by detent."""

    def __init__(self, emf_v, r_ohm, load_actual_ohm, shunt_actual_ohm):
        if not 0.5 < emf_v < 5.0 or not 0 < r_ohm < 50:
            raise ValueError('cell parameters out of range')
        self.emf, self.r = float(emf_v), float(r_ohm)
        self.load = {str(k): float(v) for k, v in load_actual_ohm.items()}
        self.shunt = {_key(k): float(v) for k, v in shunt_actual_ohm.items()}
        if set(self.load) != set(LOAD_NOMINAL_OHM) or set(self.shunt) != {_key(s) for s in SHUNT_NOMINAL_OHM}:
            raise ValueError('actual values required for every load and shunt detent')
        self.capacitor = None

    def advance(self, switch, state, dt):
        return 0.0                                  # no internal state

    def current(self, switch, state=0.0):
        load, shunt, mode = switch
        if load is None or shunt is None or mode != 'ON':
            return 0.0
        return self.emf / (self.r + self.load[str(load)] + self.shunt[_key(shunt)])

    def channels(self, switch, state=0.0):
        i = self.current(switch)
        return self.emf - i * self.r, (i * self.shunt[_key(switch[1])] if i else 0.0)

"""Calorimetry bench thermal model (calorimetry_specific_heat), headless: method of mixtures with heat loss.

Bodies: the heater block (thermo_model.HeaterController: heater only, rate-limited heating, slow Newton cooling with
the hidden tau_cool_s), the metal sample (heat capacity m_s c_s), the stirred water with the cup (m_w c_w + C_cup),
the room at T_room. The sample is in exactly one place: in the heater well (it follows the block with the hidden
tau_sample_s, a metal cylinder in a dry well), in the air (held or standing elsewhere: Newton cooling with the hidden
coefficient h_air, W/K) or in the cup's water (exchange with conductance G_SW W/K; G_SW_STILL with the stirrer off).
The water loses heat to the room through the hidden k_loss (W/K). Every tick advances the coupled linear system
exactly over dt (2 x 2 matrix exponential). The thermometer reads the water through a first-order lag TAU_PROBE_S,
quantised to 0.01 K with noise; the heater's own thermometer reads the block to 0.1 K.
"""
from __future__ import annotations

import math

import numpy as np

from thermo_model import HeaterController

C_WATER = 4184.0                  # J/(kg K)
G_SW = 6.0                        # W/K sample-water exchange while immersed and stirred
G_SW_STILL = 1.5                  # W/K without stirring
TAU_PROBE_S = 2.0                 # thermometer lag in stirred water
TAU_PROBE_STILL_S = 12.0
NOISE_K = 0.004
RESOLUTION_K = 0.01
HEATER_NOISE_K, HEATER_RESOLUTION_K = 0.01, 0.1
HEATER_SET = {'min': 25.0, 'max': 95.0, 'step': 5.0}


class Calorimeter:
    def __init__(self, *, m_s_kg, c_s, m_w_kg, c_cup_j_k, t_room_c, h_air_w_k, k_loss_w_k, tau_sample_s, tau_cool_s,
                 t_water0_c=None, t_sample0_c=None, noise_seed=0):
        self.Cs = m_s_kg * c_s
        self.Cw = m_w_kg * C_WATER + c_cup_j_k
        self.t_room = float(t_room_c)
        self.h_air, self.k_loss, self.tau_sample = float(h_air_w_k), float(k_loss_w_k), float(tau_sample_s)
        self.heater = HeaterController(t_room_c=t_room_c, tau_cool_s=tau_cool_s, set_min_c=HEATER_SET['min'],
                                       set_max_c=HEATER_SET['max'], set_step_c=HEATER_SET['step'])
        self.Ts = float(t_room_c if t_sample0_c is None else t_sample0_c)
        self.Tw = float(t_room_c if t_water0_c is None else t_water0_c)
        self.probe = self.Tw
        self._propagators = {}
        self.rng = np.random.default_rng(noise_seed)

    def step(self, dt, place, stirred=True):
        """place: 'heater', 'air' or 'cup' (where the sample is during this tick); stirred: the cup's stirrer runs."""
        g = G_SW if stirred else G_SW_STILL
        self.heater.step(dt)
        if place == 'heater':
            self.Ts = self.heater.T + (self.Ts - self.heater.T) * math.exp(-dt / self.tau_sample)
            self.Tw = self.t_room + (self.Tw - self.t_room) * math.exp(-self.k_loss / self.Cw * dt)
        elif place == 'air':
            self.Ts = self.t_room + (self.Ts - self.t_room) * math.exp(-self.h_air / self.Cs * dt)
            self.Tw = self.t_room + (self.Tw - self.t_room) * math.exp(-self.k_loss / self.Cw * dt)
        elif place == 'cup':
            # d/dt [Ts, Tw] = M [Ts - T_room, Tw - T_room]; exp(M dt) is the same for every tick of one stirrer state
            key = (dt, g)
            if key not in self._propagators:
                m = np.array([[-g / self.Cs, g / self.Cs],
                              [g / self.Cw, -(g + self.k_loss) / self.Cw]])
                w, v = np.linalg.eig(m)
                self._propagators[key] = (v @ np.diag(np.exp(w * dt)) @ np.linalg.inv(v)).real
            x = self._propagators[key] @ np.array([self.Ts - self.t_room, self.Tw - self.t_room])
            self.Ts, self.Tw = self.t_room + float(x[0]), self.t_room + float(x[1])
        else:
            raise ValueError(place)
        self.probe = self.Tw + (self.probe - self.Tw) * math.exp(-dt / (TAU_PROBE_S if stirred else TAU_PROBE_STILL_S))

    def reading(self):
        v = self.probe + self.rng.normal(0.0, NOISE_K)
        return round(round(v / RESOLUTION_K) * RESOLUTION_K, 2)

    def heater_reading(self):
        v = self.heater.T + self.rng.normal(0.0, HEATER_NOISE_K)
        return round(round(v / HEATER_RESOLUTION_K) * HEATER_RESOLUTION_K, 1)


def mixture_c(m_s_kg, m_w_kg, c_cup_j_k, t_sample_c, t_water0_c, t_final_c):
    """The method of mixtures without loss: c_s = (m_w c_w + C_cup)(T_f - T_w0) / (m_s (T_s - T_f))."""
    return (m_w_kg * C_WATER + c_cup_j_k) * (t_final_c - t_water0_c) / (m_s_kg * (t_sample_c - t_final_c))

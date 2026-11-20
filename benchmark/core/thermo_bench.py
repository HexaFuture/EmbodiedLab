"""Dry-block bench (thermistor_beta, rod_expansion), independent of Isaac: the set-point buttons (and the ohmmeter's
RANGE and AUTO keys), where the probe is, the thermal state and the displays. There is no data logger: every reading
is taken from the displays in the camera images.

Every physics tick the backend reports where the probe is ('well': standing in the block's well; 'stand': on its
stand; 'air': anywhere else) and which buttons are down. A press of UP or DOWN moves the set point by 5 C. The meters
refresh at LOG_HZ: each refresh is kept privately (rows: thermistor: the block's reference thermometer T_C and the
ohmmeter R_ohm, None when it shows OL; rod: the probe thermometer T_C and the digital indicator indicator_mm; private:
the set point, the meter range, the probe's place and the true temperatures) for the process record.

Displays (display): the controller screen's block reading, the ohmmeter, the probe thermometer and the indicator
show the last refresh; the SET field and the ohmmeter's range keys change at once. observation_record(tick) is the
private record of one get_observation: what every display showed at that moment and the bench state.
"""
from __future__ import annotations

from copy import deepcopy

from thermo_model import DryBlock, OHM_RANGES

LOG_HZ = 1
PLACES = ('well', 'stand', 'air')


class ThermoBench:
    def __init__(self, episode_id, config, *, physics_hz=240, noise_seed=0):
        if physics_hz not in (240, 480):
            raise ValueError('unsupported physics clock')
        self.episode_id = episode_id
        self.physics_hz = physics_hz
        self.model = DryBlock(noise_seed=noise_seed, **config['thermal'])
        self.kind = self.model.kind
        self.value = 'R_ohm' if self.kind == 'thermistor' else 'indicator_mm'
        self.buttons = ('up', 'down', 'range', 'auto') if self.kind == 'thermistor' else ('up', 'down')
        self.tick = 0
        self.place = config.get('probe_start', 'stand')
        self.pressed = set()
        self.rows, self.private = [], []
        self.transitions, self.actions = [], []
        self.shown = {}
        self.shown_values = {}            # the last refresh's readings as numbers (what the displays show)
        self._next_log = 0

    @property
    def t_s(self):
        return self.tick / self.physics_hz

    def step(self, tick, world):
        """world: {'place': 'well' | 'stand' | 'air', 'pressed': set of button ids}."""
        if tick != self.tick + 1:
            raise ValueError('dry-block bench requires every consecutive physics tick')
        self.tick = tick
        place = world.get('place', 'air')
        if place not in PLACES:
            raise ValueError(place)
        pressed = set(world.get('pressed', ()))
        m = self.model
        for button in sorted((pressed - self.pressed) & set(self.buttons)):
            changed = m.press(button)
            row = {'tick': tick, 't_s': round(self.t_s, 6), 'action': button, 'set_C': m.set_c}
            if m.meter is not None:
                row |= {'range': OHM_RANGES[m.meter.range][0], 'auto': m.meter.auto}
            if button in ('up', 'down'):
                row['changed'] = changed
            self.actions.append(row)
        self.pressed = pressed
        if place != self.place:
            self.transitions.append(dict(tick=tick, t_s=round(self.t_s, 6), place=place, probe_C=round(m.probe, 4),
                                         block_C=round(m.block, 4)))
            self.place = place
        m.step(1.0 / self.physics_hz, place == 'well')
        if tick >= self._next_log:
            self._next_log = tick + self.physics_hz // LOG_HZ
            values, self.shown = m.reading()
            self.rows.append({'sample_id': len(self.rows) + 1, 't_s': round(self.t_s, 3), **values})
            if self.kind == 'thermistor':
                spec = OHM_RANGES[m.meter.range]
                self.shown_values = {'block_C': self.shown['block_C'], 'R_ohm': values['R_ohm'],
                                     'R_resolution_ohm': spec[2], 'R_range': spec[0], 'R_unit': spec[5],
                                     't_s': round(self.t_s, 3)}
            else:
                self.shown_values = {'block_C': self.shown['block_C'], 'probe_C': self.shown['probe_C'],
                                     'indicator_mm': self.shown['indicator_mm'], 't_s': round(self.t_s, 3)}
            private = {'place': place, 'set_C': m.set_c, 'block_C': round(m.block, 5), 'probe_C': round(m.probe, 5)}
            if m.meter is not None:
                private |= {'range': OHM_RANGES[m.meter.range][0], 'auto': m.meter.auto,
                            'current_A': m.meter.current_a, 'self_heating_K': round(m.self_heating_k(), 5)}
            else:
                private['rod_C'] = round(m.rod, 5)
            self.private.append(private)

    def display(self):
        """What the bench shows now (the last logged readings, the set point and the meter keys at once)."""
        out = dict(self.shown) | {'set_C': self.model.set_c}
        if self.model.meter is not None and 'ohmmeter' in out:
            out['ohmmeter'] = dict(out['ohmmeter'], auto=self.model.meter.auto)
        return out

    def state(self):
        """Control state for the terminal record."""
        out = {'set_C': self.model.set_c}
        if self.model.meter is not None:
            out |= {'range': OHM_RANGES[self.model.meter.range][0], 'auto': self.model.meter.auto}
        return out

    def observation_record(self, tick):
        """Private record of one observation: what the displays showed at that moment (the latest refresh, its time;
        the set point), where the probe stood and the true temperatures (the meter's range and self-heating)."""
        m = self.model
        shown = dict(self.shown_values, set_C=m.set_c)
        record = {'tick': tick, 't_s': round(tick / self.physics_hz, 6), 'shown': shown, 'place': self.place,
                  'set_C': m.set_c, 'block_C': round(m.block, 5), 'probe_C': round(m.probe, 5)}
        if m.meter is not None:
            record |= {'range': OHM_RANGES[m.meter.range][0], 'auto': m.meter.auto,
                       'self_heating_K': round(m.self_heating_k(), 5)}
        else:
            record['rod_C'] = round(m.rod, 5)
        return record

    def private_record(self):
        """Written to the private directory when the backend closes (chem_backend.close)."""
        m = self.model
        return {'transitions': deepcopy(self.transitions), 'actions': deepcopy(self.actions), 'rows': len(self.rows),
                'set_C': m.set_c, 'block_C': m.block, 'probe_C': m.probe, 'rod_C': m.rod, 'place': self.place}

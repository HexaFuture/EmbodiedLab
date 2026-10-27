"""Chemistry side of the spectrophotometry bench, independent of Isaac.

Every physics tick the backend reports the measured world: which container sits on which station seat (rack
slots, dispenser seats, mixing station, spectrophotometer slot), which one the gripper holds and its tilt,
and which buttons are depressed. A button acts on its press edge (up -> down), like a real push button:
dispense (1 s; the liquid lands in the container on that seat when the flow ends, otherwise in the waste
tray), MIX (3 s, void if the container leaves the seat), BLANK/READ (2 s), and the wavelength keypad (digits,
ENTER, CLEAR). A container tilted beyond 45 degrees spills, held or not (knocked over on the bench).

Records: the public events an operator could see (button presses) and the private state log (every
container change, spectrophotometer statuses) for the evaluator. Nothing private is displayed.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from chem_containers import DISPENSE_S, Container, Dispenser, Mixer, Source
from chem_solver import Dye, register_dye
from chem_spectro import Spectrophotometer

CUVETTE_ML = 3.5
SPILL_TILT_DEG = 45.0
KEYS = tuple(str(d) for d in range(10)) + ('enter', 'clear')


class ChemBench:
    def __init__(self, episode_id, config, *, physics_hz=240, noise_seed=0):
        if physics_hz not in (240, 480):
            raise ValueError('unsupported physics clock')
        self.physics_hz = physics_hz
        rng = np.random.default_rng(noise_seed)
        for name, spec in (config.get('dye_specs') or {}).items():            # dye_absorptivity: the unknown dye
            register_dye(name, *spec)
        self.dyes = {name: Dye(name, factor) for name, factor in config['dyes'].items()}
        self.sources = {name: Source(name, s['composition_M'], s['volume_mL']) for name, s in config['sources'].items()}
        self.dispensers = {name: Dispenser(self.sources[name], volumes_mL=tuple(config.get('dispense_mL', (1.00, 0.10))),
                                           rng=np.random.default_rng(rng.integers(2**63)))
                           for name in config['sources']}
        self.containers = {i: Container(i, CUVETTE_ML) for i in range(1, int(config['cuvettes']) + 1)}
        # spectro_mixture: a cuvette handed out already filled (the mixed unknown); its history starts with that sample.
        for cid, fill in (config.get('prefilled') or {}).items():
            self.containers[int(cid)].add(fill['label'], float(fill['volume_mL']), float(fill['volume_mL']),
                                          fill['composition_M'])
        # Seat ids: one per dispensing station by default; the spectro bench's lines share one seat with the mixer.
        self.dispense_seats = {name: (config.get('dispense_seats') or {}).get(name, f'disp:{name}')
                               for name in config['sources']}
        self.mixer_seat = config.get('mixer_seat', 'mixer')
        self.spectro_seat = config.get('spectro_seat', 'spectro')
        self.mixer = Mixer()
        self.reactions = {d: float(k) for d, k in (config.get('reactions') or {}).items()}
        for container in self.containers.values():      # contact model on reaction benches (kinetics): jet, layer exchange, beam band
            container.contact = bool(self.reactions)
        self.spectro = Spectrophotometer(episode_id, self.dyes, b0=config['b0'], b1=config['b1'],
                                         noise_seed=int(rng.integers(2**63)), start_nm=config.get('start_nm', 500),
                                         dual=bool(config.get('dual_wavelength', False)),
                                         stray=float(config['stray']))      # the instance's hidden stray light
        self.tick = 0
        self.pressed = set()
        self.pending = []           # (done tick, dispenser name, container id at press, level)
        self.keypad = ''
        self.keypad_error = False
        self.actions = []           # public: button presses
        self.state_log = []         # private

    @property
    def t_s(self):
        return self.tick / self.physics_hz

    def step(self, tick, world):
        """world: {'seats': {station: container id | None}, 'held': id | None, 'tilt_deg': float,
        'pressed': set of button ids currently down, 'tilts': {container id: tilt from upright in degrees}}.
        Without 'tilts' only the held container's tilt_deg is known."""
        if tick != self.tick + 1:
            raise ValueError('chemistry bench requires every consecutive physics tick')
        self.tick = tick
        seats = world['seats']
        down = set(world.get('pressed', ()))
        for button in sorted(down - self.pressed):
            self._press(button, seats)
        self.pressed = down
        held = world.get('held')
        tilts = world.get('tilts', {held: world.get('tilt_deg', 0.0)} if held is not None else {})
        for cid, tilt in sorted(tilts.items()):
            if tilt > SPILL_TILT_DEG and cid in self.containers and self.containers[cid].volume_mL > 0:
                lost = self.containers[cid].spill()
                self.state_log.append(dict(tick=tick, event='spill', container=cid, mL=lost, held=cid == held))
        for item in [p for p in self.pending if p[0] <= tick]:
            self.pending.remove(item)
            _, name, cid, level = item
            target = self.containers[cid] if cid is not None and seats.get(self.dispense_seats[name]) == cid else None
            event = self.dispensers[name].press(level, target)
            self.state_log.append(dict(tick=tick, event='dispense', source=name, **event))
        if self.reactions:
            for container in self.containers.values():
                container.react(1.0 / self.physics_hz, self.reactions)
        mixed = self.mixer.update(self.t_s, seats.get(self.mixer_seat), self.containers)
        if mixed is not None:
            self.state_log.append(dict(tick=tick, event='mixed', container=mixed))
        slot = seats.get(self.spectro_seat)
        done = self.spectro.update(self.t_s, None if slot is None else self.containers[slot])
        if done is not None:
            extra = {k: done[k] for k in ('sample_id', 'scan_id', 'run', 'point') if k in done}
            self.state_log.append(dict(tick=tick, event=done['kind'], **extra))

    def _press(self, button, seats):
        self.actions.append(dict(tick=self.tick, button=button))
        kind, _, rest = button.partition(':')
        if kind == 'disp':
            name, level = rest.split(':')
            cid = seats.get(self.dispense_seats[name])
            self.pending.append((self.tick + round(DISPENSE_S * self.physics_hz), name, cid,
                                 {'coarse': 0, 'fine': 1}[level]))
        elif button == 'mix':
            cid = seats.get(self.mixer_seat)
            self.mixer.press(self.t_s, None if cid is None else self.containers[cid])
        elif button in ('spectro:blank', 'spectro:read', 'spectro:scan', 'spectro:kin'):
            self.spectro.press(rest, self.t_s)
        elif kind == 'key':
            self._key(rest)
        else:
            raise ValueError(f'unknown button {button!r}')

    def _key(self, key):
        if key not in KEYS:
            raise ValueError(f'unknown key {key!r}')
        self.keypad_error = False
        if key == 'clear':
            self.keypad = ''
        elif key == 'enter':
            if not self.keypad or not self.spectro.set_wavelength(int(self.keypad)):
                self.keypad_error = True
            self.keypad = ''
        elif len(self.keypad) < 3:
            self.keypad += key

    def display(self):
        """What the spectrophotometer screen shows (also the text rendered on the apparatus)."""
        shown = self.spectro.display()
        shown['entry'] = self.keypad or None
        shown['error'] = 'ERR' if self.keypad_error else None
        return shown

    def consumables(self):
        """Reservoir levels (the graduated bottles are visible) and the waste tray."""
        return {'reservoirs_mL': {n: round(s.remaining_mL, 2) for n, s in self.sources.items()},
                'waste_mL': round(sum(d.wasted_mL for d in self.dispensers.values()), 2)}

    def private_record(self):
        """Evaluator-only: container histories and spectrophotometer statuses."""
        return {'containers': {cid: {'nominal': deepcopy(c.nominal), 'volume_mL': c.volume_mL,
                                     'mixed': c.mixed, 'overflowed': c.overflowed, 'spilled': c.spilled}
                               for cid, c in self.containers.items()},
                'spectro_private': deepcopy(self.spectro.private),
                'scan_private': deepcopy(self.spectro.scan_private),
                'kinetic_private': deepcopy(self.spectro.kinetic_private),
                'state_log': deepcopy(self.state_log)}

    def process_extra(self):
        """Evaluator-only records for the offline process score (logging only): the SCAN survey spectra (public
        values as exported) with the cuvette that was in the well and its private condition."""
        private = {s['scan_id']: s for s in self.spectro.scan_private}
        return {'spectro_scans': [dict(deepcopy(scan), cuvette=private.get(scan['scan_id'], {}).get('cuvette'),
                                       condition=private.get(scan['scan_id'], {}).get('condition'))
                                  for scan in self.spectro.scans]}

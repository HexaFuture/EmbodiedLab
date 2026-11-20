"""Titration add-on of the chemistry kit, independent of Isaac: pH solver, titrant dispenser
with its volume counter, pH electrode and meter, recorder. titration_basic and titration_weak_acid use it.

Solutions are ideal (activity coefficients 1, 25 C, Kw = 1e-14); no CO2 uptake, no carry-over between
beakers. Analytical concentrations per beaker (mol/L): 'HCl' (strong acid), 'NaOH' (strong base), 'HA' (the
monoprotic weak acid of titration_weak_acid, with its Ka), 'KHP' (potassium hydrogen phthalate: K+ and the phthalate system,
pKa1 = 2.95, pKa2 = 5.41), and the pH standard buffers: 'HB4' + 'NaB4' (pH 4.01) and 'HB7' + 'NaB7' (pH 7.00), each
an equimolar pair of a weak acid and its sodium salt whose pKa gives exactly the nominal pH (BUFFER_KA). The pH
follows from the charge balance
    [Na+] + [K+] + [H+] = [OH-] + [Cl-] + [A-] + [HP-] + 2 [P2-] + [B4-] + [B7-],
which is monotonic in pH and solved by bisection.

What the instruments show is what real ones show: the meter a pH (0.01) and a stability mark, the dispenser
its cumulative nominal volume (0.01 mL). The meter is not calibrated: it shows
7.00 + s (pH - 7.00) + b, the electrode's slope s (fraction of the Nernst slope) and offset b drawn per instance. Whether the electrode is immersed, which beaker stands on the stirrer
or how much liquid really left the dispenser is recorded privately for the evidence gates only.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
from copy import deepcopy

import numpy as np

from chem_containers import Container

KW = 1.0e-14
PKA_PHTHALIC = (2.95, 5.41)
# pH standard buffers: an ideal equimolar pair HBn / NaBn, BUFFER_M each, whose pKa makes the charge
# balance give exactly the nominal pH. Carrier labels B4 and B7 (their handle rims show 4.01 and 7.00).
BUFFER_M = 0.025
BUFFER_PH = {'B4': 4.01, 'B7': 7.00}


def _buffer_ka(ph, c=BUFFER_M):
    """Ka of the pair HB / NaB (c each) whose ideal solution has this pH: [Na+] + [H+] = [OH-] + [B-] gives the
    fraction f of the pair present as B-, and Ka = f [H+] / (1 - f)."""
    h = 10.0 ** -ph
    f = (c + h - KW / h) / (2.0 * c)
    return f * h / (1.0 - f)


BUFFER_KA = {label: _buffer_ka(ph) for label, ph in BUFFER_PH.items()}


def buffer_composition(label):
    """Analytical concentrations (mol/L) of a standard buffer."""
    return {'H' + label: BUFFER_M, 'Na' + label: BUFFER_M}
PH_RANGE = (-1.0, 15.0)

TITRANT_ML = (1.00, 0.10, 0.02)                 # buttons: large, coarse, fine
TITRANT_SIGMA_ML = (0.003, 0.0005, 0.0002)      # absolute repeatability per press (public)
TITRANT_LEVELS = ('large', 'coarse', 'fine')
SPILL_TILT_DEG = 45.0                           # a beaker tilted further loses its contents, held or knocked over
DELIVERY_S = 0.5                                # press to liquid in the beaker
MIX_TAU_S = 1.0                                 # stirred beaker: the zone at the electrode follows the bulk
ELECTRODE_TAU_S = 0.75
SETTLE_S = 3.0                                  # stability mark off this long after an addition or immersion
STABLE_DRIFT_PH_PER_S = 0.005                   # and until the membrane drifts slower than this
NOISE_PH = 0.005
RECORD_HZ = 2
BEAKER_AREA_CM2 = 16.0                          # square beaker, 40 x 40 mm inside
BEAKER_ML = BEAKER_AREA_CM2 * 2.2                # 22 mm deep
TIP_ABOVE_FLOOR_MM = 2.0
IMMERSION_MM = 3.0                              # the junction sits this far above the tip
AIR_DRIFT_PH_PER_SQRT_S = 0.35                  # an electrode in air wanders and never shows the stability mark
FIELDS = ('sample_id', 't_s', 'pH_display', 'stable')            # what a pH meter knows: no dispenser volume


def ph_of(concentrations, ka=None):
    """pH of a solution with analytical concentrations (mol/L) {'HCl', 'NaOH', 'HA', 'KHP', and the buffer pairs
    'HB4', 'NaB4', 'HB7', 'NaB7'}; ka: the weak acid's acid constant (required when 'HA' is present)."""
    c = {k: float(v) for k, v in concentrations.items() if v}
    buffers = {label: (c.get('H' + label, 0.0) + c.get('Na' + label, 0.0), ka_b) for label, ka_b in BUFFER_KA.items()}
    unknown = set(c) - {'HCl', 'NaOH', 'HA', 'KHP'} - {p + label for label in BUFFER_KA for p in ('H', 'Na')}
    if unknown:
        raise ValueError(f'species without acid-base model: {sorted(unknown)}')
    if any(v < 0 for v in c.values()):
        raise ValueError('negative concentration')
    c_ha = c.get('HA', 0.0)
    if c_ha and (ka is None or ka <= 0):
        raise ValueError('weak acid present without its Ka')
    cations = c.get('NaOH', 0.0) + c.get('KHP', 0.0) + sum(c.get('Na' + label, 0.0) for label in BUFFER_KA)
    chloride, c_p = c.get('HCl', 0.0), c.get('KHP', 0.0)
    k1, k2 = (10.0 ** -p for p in PKA_PHTHALIC)

    def excess_positive(ph):
        h = 10.0 ** -ph
        anions = KW / h + chloride
        if c_ha:
            anions += c_ha * ka / (ka + h)
        if c_p:
            d = h * h + k1 * h + k1 * k2
            anions += c_p * (k1 * h + 2.0 * k1 * k2) / d
        for total, ka_b in buffers.values():
            if total:
                anions += total * ka_b / (ka_b + h)
        return cations + h - anions

    low, high = PH_RANGE
    if excess_positive(low) < 0 or excess_positive(high) > 0:
        raise ValueError('solution outside the pH range of the model')
    for _ in range(60):                          # 16 / 2**60: far below the 1e-10 tolerance
        mid = 0.5 * (low + high)
        if excess_positive(mid) > 0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def liquid_height_mm(volume_mL):
    return volume_mL / BEAKER_AREA_CM2 * 10.0


def immersed(volume_mL):
    return liquid_height_mm(volume_mL) >= TIP_ABOVE_FLOOR_MM + IMMERSION_MM


class Beaker(Container):
    """A titration beaker: always treated as one phase (it is titrated on the stirrer)."""

    def __init__(self, label, contents=None, volume_mL=0.0):
        super().__init__(label, BEAKER_ML)
        self.sample_mL = float(volume_mL) if contents else 0.0
        self.titrant_nominal_mL = 0.0
        self.titrant_actual_mL = 0.0
        if contents and volume_mL > 0:
            self.layers = [('sample', float(volume_mL), dict(contents))]

    def add(self, source_name, nominal_mL, actual_mL, composition):
        lost = super().add(source_name, nominal_mL, actual_mL, composition)
        self.mix()
        return lost


class TitrantDispenser:
    """Burette-type dispenser over the stirrer: three buttons, a finite reservoir and a counter of the nominal
    volume delivered since its ZERO key was last pressed. Every press counts, also one with no beaker below the
    spout."""

    def __init__(self, composition_M, volume_mL, rng):
        self.composition_M = dict(composition_M)
        self.capacity_mL = float(volume_mL)
        self.remaining_mL = float(volume_mL)
        self.rng = rng
        self.counter_mL = 0.0
        self.wasted_mL = 0.0

    def press(self, level, beaker):
        nominal = TITRANT_ML[level]
        if self.remaining_mL + 1e-12 < nominal:
            return {'dispensed': False, 'reason': 'consumable_exhausted', 'nominal_mL': nominal}
        actual = max(0.0, nominal + TITRANT_SIGMA_ML[level] * float(self.rng.standard_normal()))
        actual = min(actual, self.remaining_mL)
        self.remaining_mL -= actual
        self.counter_mL = round(self.counter_mL + nominal, 2)
        if beaker is None:
            self.wasted_mL += actual
            return {'dispensed': True, 'into': None, 'nominal_mL': nominal, 'actual_mL': actual}
        lost = beaker.add('titrant', nominal, actual, self.composition_M)
        beaker.titrant_nominal_mL = round(beaker.titrant_nominal_mL + nominal, 2)
        beaker.titrant_actual_mL += actual - lost
        self.wasted_mL += lost
        return {'dispensed': True, 'into': beaker.id, 'nominal_mL': nominal, 'actual_mL': actual, 'overflow_mL': lost}

    def reset_counter(self):
        self.counter_mL = 0.0


class PHMeter:
    """Electrode and meter. In a stirred beaker the zone at the electrode follows the bulk composition with
    MIX_TAU_S, the glass membrane follows that zone's pH with ELECTRODE_TAU_S. The meter is not calibrated: it
    displays 7.00 + slope_frac (pH_membrane - 7.00) + offset_pH and noise (slope_frac: the electrode's slope as a
    fraction of the Nernst slope; both drawn per instance). The stability mark comes on, as on a real meter, when the reading has stopped drifting:
    at least SETTLE_S after the last addition or immersion and the membrane slower than STABLE_DRIFT_PH_PER_S
    (close to an endpoint that takes longer). In air the reading wanders and the mark stays off."""

    def __init__(self, *, offset_pH, ka, rng, slope_frac=1.0):
        self.offset, self.slope, self.ka, self.rng = float(offset_pH), float(slope_frac), ka, rng
        self.zone = None                 # composition at the electrode (mol/L), None when not in liquid
        self.sensed = 7.0                # membrane pH
        self.disturbed_s = None          # time of the last addition or immersion
        self.in_liquid = False
        self.drift = math.inf            # membrane drift, pH/s

    def disturb(self, t_s):
        self.disturbed_s = t_s

    def update(self, t_s, dt, beaker):
        """beaker: the beaker the electrode stands in (None: parked, held or over an empty stirrer)."""
        wet = beaker is not None and immersed(beaker.volume_mL)
        if not wet:
            self.zone, self.in_liquid, self.drift = None, False, math.inf
            self.sensed += AIR_DRIFT_PH_PER_SQRT_S * math.sqrt(dt) * float(self.rng.standard_normal())
            self.sensed = min(max(self.sensed, 2.0), 12.0)
            return
        bulk = beaker.composition_M()
        if not self.in_liquid:
            self.zone, self.in_liquid = dict(bulk), True
            self.disturb(t_s)
        else:
            a = 1.0 - math.exp(-dt / MIX_TAU_S)
            self.zone = {k: self.zone.get(k, 0.0) + a * (bulk.get(k, 0.0) - self.zone.get(k, 0.0))
                         for k in set(bulk) | set(self.zone)}
        target = ph_of(self.zone, self.ka)
        self.sensed += (1.0 - math.exp(-dt / ELECTRODE_TAU_S)) * (target - self.sensed)
        self.drift = abs(target - self.sensed) / ELECTRODE_TAU_S

    def stable(self, t_s):
        return bool(self.in_liquid and self.disturbed_s is not None and t_s - self.disturbed_s >= SETTLE_S - 1e-9
                    and self.drift < STABLE_DRIFT_PH_PER_S)

    def reading(self):
        """A displayed value (new noise each call)."""
        value = 7.0 + self.slope * (self.sensed - 7.0) + self.offset + NOISE_PH * float(self.rng.standard_normal())
        return round(min(max(value, 0.0), 14.0), 2) + 0.0


class Recorder:
    """REC key on the meter: the first press starts a record, the next press ends it. While a record runs the meter is
    logged at RECORD_HZ, whatever the electrode stands in. A row holds only what the meter knows (FIELDS): the meter
    cannot know the dispenser's counter, which the Agent reads on the counter screen. A record is cited as
    <episode>:titration:<n>, its rows by sample_id. Privately each row also keeps the counter value at that moment (as
    the counter screen shows it, 0.01 mL), where the electrode was and the beaker on the stirrer when it stood in that
    beaker's holder."""

    def __init__(self, episode_id):
        self.episode_id = episode_id
        self.rows, self.private = [], []
        self.records = []                # {'n', 'first', 'last', 'started_s', 'ended_s'} (sample ids)
        self.open = None
        self.exports = {}

    @property
    def recording(self):
        return self.open is not None

    def toggle(self, t_s):
        if self.open is None:
            self.open = {'n': len(self.records) + 1, 'first': None, 'last': None, 'started_s': round(t_s, 6),
                         'ended_s': None}
            self.records.append(self.open)
        else:
            self.open['ended_s'] = round(t_s, 6)
            self.open = None

    def record(self, t_s, stable, counter_mL, ph, private):
        if self.open is None:
            return None
        sample_id = len(self.rows) + 1
        self.rows.append({'sample_id': sample_id, 't_s': round(t_s, 6), 'pH_display': ph, 'stable': bool(stable)})
        self.private.append({'sample_id': sample_id, 'record': self.open['n'], 'counter_mL': round(counter_mL, 2),
                             **private})
        if self.open['first'] is None:
            self.open['first'] = sample_id
        self.open['last'] = sample_id
        return sample_id

    def export(self):
        n = len(self.exports) + 1
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        for row in self.rows:
            writer.writerow(row)
        text = buffer.getvalue()
        records = [{'evidence_id': f'{self.episode_id}:titration:{r["n"]}', 'first_sample_id': r['first'],
                    'last_sample_id': r['last']}
                   for r in self.records if r['first'] is not None]
        record = {'export_id': f'{self.episode_id}:titration_log:{n}', 'rows': deepcopy(self.rows), 'csv': text,
                  'sha256': hashlib.sha256(text.encode()).hexdigest(), 'records': records}
        self.exports[record['export_id']] = record
        return deepcopy(record)


class TitrationBench:
    """Chemistry side of the titration bench, advanced every physics tick with the measured world:
    {'stirrer': beaker label on the stirrer seat | None, 'electrode': 'holder' | 'park' | 'held' | 'free',
     'pressed': set of button ids currently down ('titrant:large' | 'titrant:coarse' | 'titrant:fine' |
                'counter:zero' | 'meter:rec'),
     'tilts': {beaker label: tilt from upright in degrees}}. A beaker tilted beyond 45 degrees spills, held or not."""

    def __init__(self, episode_id, config, *, physics_hz=240, noise_seed=0):
        if physics_hz not in (240, 480):
            raise ValueError('unsupported physics clock')
        self.physics_hz = physics_hz
        rng = np.random.default_rng(noise_seed)
        self.beakers = {label: Beaker(label, b.get('composition_M'), b.get('volume_mL', 0.0))
                        for label, b in config['beakers'].items()}
        titrant = config['titrant']
        self.titrant = TitrantDispenser(titrant['composition_M'], titrant['volume_mL'],
                                        np.random.default_rng(rng.integers(2 ** 63)))
        self.meter = PHMeter(offset_pH=config.get('ph_offset', 0.0), slope_frac=config.get('ph_slope_frac', 1.0),
                             ka=config.get('ka'), rng=np.random.default_rng(rng.integers(2 ** 63)))
        self.recorder = Recorder(episode_id)
        self.tick = 0
        self.pressed = set()
        self.pending = []                # (done tick, level, beaker label at press)
        self.on_stirrer = None
        self.shown_ph, self.shown_stable = self.meter.reading(), False
        self.actions, self.state_log = [], []
        self.electrode_at = None         # private record only: (place, beaker in whose holder, in liquid)

    @property
    def t_s(self):
        return self.tick / self.physics_hz

    def step(self, tick, world):
        if tick != self.tick + 1:
            raise ValueError('titration bench requires every consecutive physics tick')
        self.tick = tick
        for key, tilt in sorted(world.get('tilts', {}).items()):
            if tilt > SPILL_TILT_DEG and key in self.beakers and self.beakers[key].volume_mL > 0:
                lost = self.beakers[key].spill()
                self.state_log.append(dict(tick=tick, event='spill', beaker=key, mL=lost, held=key == world.get('held')))
        label = world.get('stirrer')
        if label != self.on_stirrer:
            self.on_stirrer = label
            self.state_log.append(dict(tick=tick, event='stirrer', beaker=label))
        for button in sorted(set(world.get('pressed', ())) - self.pressed):
            self._press(button, label)
        self.pressed = set(world.get('pressed', ()))
        for item in [p for p in self.pending if p[0] <= tick]:
            self.pending.remove(item)
            _, level, at_press = item
            target = self.beakers[label] if label is not None and label == at_press else None
            event = self.titrant.press(level, target)
            self.state_log.append(dict(tick=tick, event='titrant', level=TITRANT_LEVELS[level], **event))
            if event.get('into') is not None:
                self.meter.disturb(self.t_s)
        in_holder = world.get('electrode') == 'holder'
        beaker = self.beakers[label] if in_holder and label is not None else None
        self.meter.update(self.t_s, 1.0 / self.physics_hz, beaker)
        # Private record for the process score: every change of where the electrode is (holder / park /
        # held / free), the beaker in whose holder it stands and whether it is in liquid, with its tilt then.
        at = (world.get('electrode'), label if in_holder else None, bool(self.meter.in_liquid))
        if at != self.electrode_at:
            self.electrode_at = at
            tilt = world.get('electrode_tilt_deg')
            self.state_log.append(dict(tick=tick, event='electrode', place=at[0], beaker=at[1], immersed=at[2],
                                       tilt_deg=None if tilt is None else round(float(tilt), 2)))
        if tick % (self.physics_hz // RECORD_HZ) == 0:
            self.shown_ph, self.shown_stable = self.meter.reading(), self.meter.stable(self.t_s)
            self.recorder.record(self.t_s, self.shown_stable, self.titrant.counter_mL, self.shown_ph,
                                 {'electrode': world.get('electrode'), 'beaker': label if in_holder else None,
                                  'immersed': self.meter.in_liquid,
                                  'beaker_volume_mL': None if beaker is None else beaker.volume_mL,
                                  'titrant_nominal_mL': None if beaker is None else beaker.titrant_nominal_mL,
                                  'titrant_actual_mL': None if beaker is None else beaker.titrant_actual_mL,
                                  'overflowed': None if beaker is None else beaker.overflowed})

    def _press(self, button, label):
        self.actions.append(dict(tick=self.tick, button=button))
        if button == 'counter:zero':
            self.titrant.reset_counter()
            self.state_log.append(dict(tick=self.tick, event='counter_zero'))
            return
        if button == 'meter:rec':
            self.recorder.toggle(self.t_s)
            self.state_log.append(dict(tick=self.tick, event='record_start' if self.recorder.recording else 'record_end',
                                       record=self.recorder.records[-1]['n']))
            return
        kind, _, level = button.partition(':')
        if kind != 'titrant' or level not in TITRANT_LEVELS:
            raise ValueError(f'unknown button {button!r}')
        self.pending.append((self.tick + round(DELIVERY_S * self.physics_hz), TITRANT_LEVELS.index(level), label))

    def display(self):
        """What the two screens show: the meter (pH, stability mark, REC mark and the running record's number) and
        the dispenser's counter."""
        return {'pH': f'{self.shown_ph:.2f}', 'stable': self.shown_stable, 'counter_mL': f'{self.titrant.counter_mL:.2f}',
                'recording': self.recorder.recording,
                'record': self.recorder.open['n'] if self.recorder.recording else None}

    def consumables(self):
        return {'titrant_mL': round(self.titrant.remaining_mL, 2), 'waste_mL': round(self.titrant.wasted_mL, 2)}

    def screen_state(self):
        """Private record of what the two screens show now and where the electrode stands (one per observation of the
        bench, for the process score; never shown to the Agent): the meter's pH, stability mark, REC mark and record
        number, the counter, and the electrode's place, the beaker in whose holder it stands, whether it is in liquid,
        and that beaker's titrant and overflow."""
        place, label, wet = self.electrode_at or (None, None, False)
        beaker = self.beakers.get(label) if label is not None else None
        return {'pH': float(self.shown_ph), 'stable': bool(self.shown_stable),
                'counter_mL': round(self.titrant.counter_mL, 2), 'recording': self.recorder.recording,
                'record': self.recorder.open['n'] if self.recorder.recording else None,
                'electrode': place, 'stirrer': self.on_stirrer, 'beaker': label, 'immersed': bool(wet),
                'beaker_volume_mL': None if beaker is None else beaker.volume_mL,
                'titrant_nominal_mL': None if beaker is None else beaker.titrant_nominal_mL,
                'overflowed': None if beaker is None else beaker.overflowed}

    def private_record(self):
        return {'beakers': {label: {'nominal': deepcopy(b.nominal), 'volume_mL': b.volume_mL, 'overflowed': b.overflowed,
                                    'spilled': b.spilled} for label, b in self.beakers.items()},
                'recorder_private': deepcopy(self.recorder.private), 'records': deepcopy(self.recorder.records),
                'sample_mL': {label: b.sample_mL for label, b in self.beakers.items()},
                'state_log': deepcopy(self.state_log)}

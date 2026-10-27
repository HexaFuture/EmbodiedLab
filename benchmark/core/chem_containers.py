"""Containers, dispensing stations and the mixing station of the chemistry kit, independent of Isaac.

No fluid simulation: a container is a state vector updated by the mechanisms. Liquid enters a container only
from a dispensing station (or the titrant dispenser); it never moves between containers. Each dispense adds
nominal x (1 + 0.003 N(0, 1)) mL. A container that holds liquid from two sources and has not been mixed since
the last addition is layered in order of addition: a spectrophotometer beam then samples a random layer.
Beyond the capacity the excess goes to the waste tray and the container is marked overflowed. Everything here
is Bench-private: the Agent sees only the rendered liquid (level, colour) and what the instruments display.

Contact model (benches with a reaction only: kinetics_crystal_violet; Container.contact set by the bench).
The 10 x 10 mm cuvette (1 mL = 10 mm) holds a stack of well-mixed compartments in order of addition (the 0.1 M NaOH
is only 0.4 % denser than water: density ordering ignored):
- jet: a dispense of v mL mixes completely with JET_ENTRAIN x v mL taken from the top of the stack (all of it when less
  is present) and becomes the new top compartment (a free jet of ~0.5 m/s penetrates and entrains about its own volume);
- exchange: adjacent compartments exchange EXCHANGE_ML_S across each interface, dc_i/dt = Q (c_j - c_i) / V_i
  (unstirred residual / thermal convection ~50 um/s x 1 cm^2; two 1.5 mL layers equalise with tau = 150 s);
- the reaction runs in every compartment holding dye and OH- (react), before and after MIX;
- MIX mixes fully; compartments within MIXED_REL_TOL of each other count as mixed (merged);
- the beam (Z-height 15 mm, 2 mm tall) passes through BEAM_BAND_ML by volume above the floor: beam_parts gives the
  compartments in that band with their volume weights (the spectrophotometer averages their transmittance).
Without the flag (every other bench) the layering rule is unchanged: layers do not exchange and an unmixed cuvette
is read in one random layer.
"""
from __future__ import annotations

import math

import numpy as np

DISPENSE_REPEATABILITY = 0.003
DISPENSE_S = 1.0
MIX_S = 3.0
JET_ENTRAIN = 1.0               # contact model: a dispense's jet mixes with this multiple of its volume from the top
EXCHANGE_ML_S = 0.005           # contact model: volume exchanged per second across each compartment interface
BEAM_BAND_ML = (1.4, 1.6)       # contact model: the beam's band in the liquid, by volume above the floor
MIXED_REL_TOL = 1e-3            # contact model: compartments this close in composition count as mixed


class Source:
    """A reservoir: its solution composition (mol/L per species) and the remaining volume (mL)."""

    def __init__(self, name, composition_M, volume_mL):
        self.name = name
        self.composition_M = dict(composition_M)
        self.capacity_mL = float(volume_mL)
        self.remaining_mL = float(volume_mL)
        self.dispensed_mL = 0.0


class Container:
    def __init__(self, cid, capacity_mL):
        self.id = cid
        self.capacity_mL = float(capacity_mL)
        self.layers = []            # (source name, actual mL, composition) in order of addition
        self.nominal = []           # (source name, nominal mL): the public dispense history
        self.mixed = True
        self.overflowed = False
        self.spilled = False
        self.overflow_mL = 0.0
        self.contact = False        # contact model (benches with a reaction); set by the bench

    @property
    def volume_mL(self):
        return sum(v for _, v, _ in self.layers)

    @property
    def sources(self):
        return {name for name, _, _ in self.layers}

    def moles(self):
        total = {}
        for _, v, comp in self.layers:
            for species, c in comp.items():
                total[species] = total.get(species, 0.0) + c * v * 1e-3
        return total

    def composition_M(self):
        v = self.volume_mL
        return {k: n / (v * 1e-3) for k, n in self.moles().items()} if v > 0 else {}

    def add(self, source_name, nominal_mL, actual_mL, composition):
        """Add liquid; returns the volume that overflowed into the waste tray."""
        free = max(self.capacity_mL - self.volume_mL, 0.0)
        kept = min(actual_mL, free)
        lost = actual_mL - kept
        if kept > 0 and self.contact and self.layers:
            self._jet(source_name, kept, composition)
        elif kept > 0:
            if self.layers and self.layers[-1][2] == composition:
                name, v, comp = self.layers[-1]
                self.layers[-1] = (name, v + kept, comp)
            else:
                self.layers.append((source_name, kept, dict(composition)))
        self.nominal.append((source_name, nominal_mL))
        self.mixed = self._homogeneous()
        if self.contact and self.mixed and len(self.layers) > 1:
            self.mix()
        if lost > 0:
            self.overflowed = True
            self.overflow_mL += lost
        return lost

    def _homogeneous(self):
        """All layers have the same composition (e.g. more of the same stock added to it): nothing to mix."""
        if len(self.layers) < 2:
            return True
        tol = MIXED_REL_TOL if self.contact else 1e-12
        first = self.layers[0][2]
        for _, _, comp in self.layers[1:]:
            keys = set(first) | set(comp)
            if any(abs(first.get(k, 0.0) - comp.get(k, 0.0)) > tol * max(abs(first.get(k, 0.0)), abs(comp.get(k, 0.0)), 1e-30)
                   for k in keys):
                return False
        return True

    def _jet(self, source_name, volume_mL, composition):
        """Contact model: the added volume mixes with JET_ENTRAIN x its volume from the top of the stack."""
        want = min(JET_ENTRAIN * volume_mL, self.volume_mL)
        moles = {k: c * volume_mL * 1e-3 for k, c in composition.items()}
        taken = 0.0
        while want - taken > 1e-12 and self.layers:
            name, v, comp = self.layers[-1]
            use = min(v, want - taken)
            for k, c in comp.items():
                moles[k] = moles.get(k, 0.0) + c * use * 1e-3
            taken += use
            if v - use > 1e-12:
                self.layers[-1] = (name, v - use, comp)
            else:
                self.layers.pop()
        total = volume_mL + taken
        self.layers.append((source_name, total, {k: n / (total * 1e-3) for k, n in moles.items()}))

    def _exchange(self, dt_s):
        """Contact model: exchange EXCHANGE_ML_S between adjacent compartments over dt_s (fluxes from the state at the
        start of the step)."""
        if len(self.layers) < 2:
            return
        q = EXCHANGE_ML_S * dt_s * 1e-3                 # L exchanged each way per interface
        moles = [{k: c * v * 1e-3 for k, c in comp.items()} for _, v, comp in self.layers]
        for i in range(len(self.layers) - 1):
            low, high = self.layers[i][2], self.layers[i + 1][2]
            for k in set(low) | set(high):
                flux = q * (high.get(k, 0.0) - low.get(k, 0.0))       # mol moving down
                moles[i][k] = moles[i].get(k, 0.0) + flux
                moles[i + 1][k] = moles[i + 1].get(k, 0.0) - flux
        self.layers = [(name, v, {k: max(n, 0.0) / (v * 1e-3) for k, n in m.items()})
                       for (name, v, _), m in zip(self.layers, moles)]
        if self._homogeneous():
            self.mix()

    def beam_parts(self):
        """Contact model: [(volume weight, composition)] of the compartments in the beam band (mixed: the whole)."""
        if self.mixed or len(self.layers) < 2:
            return [(1.0, self.composition_M())]
        low, high = BEAM_BAND_ML
        parts, bottom = [], 0.0
        for _, v, comp in self.layers:
            overlap = min(bottom + v, high) - max(bottom, low)
            if overlap > 0:
                parts.append((overlap / (high - low), comp))
            bottom += v
        return parts

    def react(self, dt_s, rates):
        """Dye + OH- -> colourless product (crystal violet fading, kinetics experiment), layer by layer: each
        layer keeps its own composition until mixed. rates: {dye: k (L mol^-1 s^-1)} with OH- from 'NaOH'. Over
        one tick the dye decays exactly at the layer's OH- concentration, which loses the same amount."""
        changed = False
        if self.contact:
            self._exchange(dt_s)
        for _, _, comp in self.layers:
            oh = comp.get('NaOH', 0.0)
            if oh <= 0:
                continue
            for dye, k in rates.items():
                c = comp.get(dye, 0.0)
                if c > 0:
                    lost = c * -math.expm1(-k * oh * dt_s)
                    comp[dye] = c - lost
                    comp['NaOH'] = oh = max(oh - lost, 0.0)
                    changed = True
        return changed

    def mix(self):
        if self.layers:
            comp = self.composition_M()
            self.layers = [('mixed', self.volume_mL, comp)]
        self.mixed = True

    def layer_at(self, height_fraction):
        """Composition at a relative height of the liquid column (0 = bottom), layers stacked in order."""
        v = self.volume_mL
        if not self.layers or v <= 0:
            return {}
        target = height_fraction * v
        acc = 0.0
        for _, layer_v, comp in self.layers:
            acc += layer_v
            if target <= acc:
                return comp
        return self.layers[-1][2]

    def spill(self):
        """Tilted beyond 45 deg, held or knocked over: the contents are lost."""
        spilled = self.volume_mL
        self.layers = []
        self.spilled = True
        return spilled


class Dispenser:
    """A dispensing station: coarse and fine buttons with nominal volumes, a finite reservoir, a seat below
    the spout. Pressing with no container on the seat dispenses into the waste tray."""

    def __init__(self, source, volumes_mL=(1.00, 0.10), repeatability=DISPENSE_REPEATABILITY, rng=None):
        self.source = source
        self.volumes_mL = tuple(volumes_mL)
        self.repeatability = float(repeatability)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.wasted_mL = 0.0
        self.wasted_mol = {}

    def _waste(self, volume_mL):
        self.wasted_mL += volume_mL
        for species, c in self.source.composition_M.items():
            self.wasted_mol[species] = self.wasted_mol.get(species, 0.0) + c * volume_mL * 1e-3

    def press(self, level, container):
        """level: index into volumes_mL. Returns an event dict (Bench-private)."""
        nominal = self.volumes_mL[level]
        if self.source.remaining_mL + 1e-12 < nominal:
            return {'dispensed': False, 'reason': 'consumable_exhausted', 'nominal_mL': nominal}
        actual = nominal * (1.0 + self.repeatability * float(self.rng.standard_normal()))
        actual = min(actual, self.source.remaining_mL)
        self.source.remaining_mL -= actual
        self.source.dispensed_mL += actual
        if container is None:
            self._waste(actual)
            return {'dispensed': True, 'into': None, 'nominal_mL': nominal, 'actual_mL': actual}
        lost = container.add(self.source.name, nominal, actual, self.source.composition_M)
        self._waste(lost)
        return {'dispensed': True, 'into': container.id, 'nominal_mL': nominal, 'actual_mL': actual,
                'overflow_mL': lost}


class Mixer:
    """Mixing station: MIX with a container on the seat mixes it after MIX_S; removing it earlier voids it."""

    def __init__(self):
        self.pending = None         # (container id, due time s)

    def press(self, t_s, container):
        if container is None:
            return {'mixing': False}
        self.pending = (container.id, t_s + MIX_S)
        return {'mixing': True, 'container': container.id, 'done_s': t_s + MIX_S}

    def update(self, t_s, seated, containers):
        """seated: id of the container on the seat now (or None). Returns the id mixed at this time, if any."""
        if self.pending is None:
            return None
        cid, due = self.pending
        if seated != cid:
            self.pending = None
            return None
        if t_s + 1e-9 >= due:
            containers[cid].mix()
            self.pending = None
            return cid
        return None


def check_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')

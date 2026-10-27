"""Single-beam spectrophotometer of the chemistry kit, independent of Isaac.

Public face (what a real instrument shows): the set wavelength, the last absorbance (3 decimals) and a BLANK
prompt when the stored reference is not for the current wavelength. BLANK stores the detector signal through
the cuvette in the slot at the current wavelength; READ shows A = log10(S_ref / S_sample) plus noise, using the
factory reference when no BLANK is valid (then the instrument/cuvette baseline b(lambda) stays in the reading).
Both take 2 simulated seconds. The instrument does not know what is in a cuvette: a dyed blank, an unmixed
cuvette or liquid below the beam only change the signal. Those conditions are recorded in the private log
(status codes) for the evidence gates and are never displayed.

READ values reach the agent only on the screen (no READ log export). Every READ still keeps a private
row (sample_id, t_s, wavelength_nm, A_display, blank_wavelength_nm and its status); screen_record names, for each
absorbance the screen shows, the READ (or BLANK / KIN point) that produced it, so that a reading cited to an
observation can be checked against what the screen showed then.

Dual-wavelength mode (recipe switch): the instrument keeps the two wavelengths entered last
(the earlier first); BLANK stores a reference at each, READ measures at each (one private row per wavelength) and the
screen shows both wavelengths with their absorbances (the earlier entered first). Without it every BLANK replaces the
single stored reference, as before.

SCAN (spectro_mixture): a scanning instrument's survey spectrum of the cuvette in the slot, 400-700 nm in 1 nm
steps, against the factory reference (no blank subtraction: the instrument baseline stays in), 15 simulated
seconds. Scans are exported separately and only serve to choose wavelengths; they are not READs.
"""
from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import io

import numpy as np

import chem_solver as S

MEASURE_S = 2.0
SCAN_S = 15.0
SCAN_STEP_NM = 1
KIN_INTERVAL_S = 2.0
KIN_DURATION_S = 300.0
MIN_VOLUME_ML = 2.0
WAVELENGTH_NM = (400, 700)
FIELDS = ('sample_id', 't_s', 'wavelength_nm', 'A_display', 'blank_wavelength_nm')


class Spectrophotometer:
    def __init__(self, episode_id, dyes, *, b0, b1, noise_seed, start_nm=500, stray, dual=False):
        self.episode_id = episode_id
        self.dyes = dyes
        self.b0, self.b1, self.stray = float(b0), float(b1), float(stray)
        self.rng = np.random.default_rng(noise_seed)
        self.wavelength_nm = int(start_nm)
        self.dual = bool(dual)
        self.entered = [int(start_nm)]  # the wavelengths entered last (dual mode measures at the two last)
        self.references = {}            # wavelength -> reference stored by the last BLANK
        self.reference = None           # {'wavelength_nm', 's_ref', 'container', 'solvent_only'} (private): the last one
        self.busy = None                # ('blank' | 'read', done_s)
        self.last_display = None
        self.screen = {}                # wavelength -> {'A', 'source'}: the absorbance its row shows and what produced it
        self.rows = []                  # READ rows (the instrument's own values; never exported, see screen_record)
        self.private = []               # same order: status, cuvette, blank container, composition
        self.scans = []                 # public: {'scan_id', 't_s', 'wavelength_nm': [...], 'A': [...]}
        self.scan_private = []          # same order: cuvette, condition
        self.scan_exports = {}
        self.kinetic = None             # running KIN: {'run', 'next_s', 'until_s'}
        self.kinetic_runs = []          # public: {'run', 'wavelength_nm', 'blank_wavelength_nm', 'rows': [{'t_s', 'A_display'}]}
        self.kinetic_private = []       # same order: per run the cuvette and per point status and composition
        self.kinetic_exports = {}

    # --- display ---------------------------------------------------------------------------------
    def active(self):
        """The wavelengths a BLANK or READ measures at (the earlier first)."""
        return list(dict.fromkeys(self.entered[-2:])) if self.dual else [self.wavelength_nm]

    def blank_valid(self, nm=None):
        if nm is None:
            return all(n in self.references for n in self.active())
        return nm in self.references

    def display(self):
        shown = {'wavelength_nm': self.wavelength_nm,
                 'absorbance': None if self.last_display is None else f'{self.last_display:.3f}',
                 'prompt': None if self.blank_valid() else 'BLANK',
                 'busy': self.busy is not None or self.kinetic is not None}
        if self.dual:
            shown['wavelengths_nm'] = self.active()
            shown['rows'] = [{'wavelength_nm': nm,
                              'absorbance': None if nm not in self.screen else f"{self.screen[nm]['A']:.3f}"}
                             for nm in self.active()]
        return shown

    def screen_record(self):
        """Private: what each absorbance on the screen comes from. rows (one per wavelength row, the earlier entered
        first): wavelength_nm, A (None when the row is empty), source ({'kind': 'read', 'sample_id'} | {'kind':
        'blank'} | {'kind': 'kin', 'run', 'point'} | None) and, for a READ, its private row with the public one
        (key 'read'); for a KIN point its private point with the run's wavelengths (key 'kin')."""
        rows = []
        for nm in (self.active() if self.dual else [self.wavelength_nm]):
            shown = self.screen.get(nm)
            row = {'wavelength_nm': nm, 'A': None if shown is None else shown['A'],
                   'source': None if shown is None else deepcopy(shown['source'])}
            source = row['source'] or {}
            if source.get('kind') == 'read':
                k = source['sample_id'] - 1
                row['read'] = dict(deepcopy(self.private[k]), public=deepcopy(self.rows[k]))
            elif source.get('kind') == 'kin':
                run = self.kinetic_runs[source['run'] - 1]
                point = self.kinetic_private[source['run'] - 1]['points'][source['point'] - 1]
                row['kin'] = dict(deepcopy(point), wavelength_nm=run['wavelength_nm'],
                                  blank_wavelength_nm=run['blank_wavelength_nm'],
                                  public=deepcopy(run['rows'][source['point'] - 1]))
            rows.append(row)
        return {'rows': rows, 'wavelength_nm': self.wavelength_nm, 'busy': self.busy is not None or self.kinetic is not None,
                'blank_valid': self.blank_valid(), 'dual': self.dual}

    # --- controls ---------------------------------------------------------------------------------
    def set_wavelength(self, nm):
        # The wavelength is fixed while a measurement or a KIN run is going (ENT shows Err).
        if self.busy is not None or self.kinetic is not None:
            return False
        if isinstance(nm, bool) or int(nm) != nm or not WAVELENGTH_NM[0] <= nm <= WAVELENGTH_NM[1]:
            return False
        self.wavelength_nm = int(nm)
        self.entered = (self.entered + [int(nm)])[-2:]
        self.last_display = None
        self.screen = {}
        return True

    def press(self, kind, t_s):
        """kind: 'blank', 'read', 'scan' or 'kin'. Ignored while a measurement is running; KIN during a timed run
        ends it, other keys are ignored until it ends."""
        if kind not in ('blank', 'read', 'scan', 'kin'):
            raise ValueError(kind)
        if self.kinetic is not None:
            if kind == 'kin':
                self.kinetic = None
                return True
            return False
        if self.busy is not None:
            return False
        if kind == 'kin':
            n = len(self.kinetic_runs) + 1
            self.kinetic = {'run': n, 'next_s': t_s + KIN_INTERVAL_S, 'until_s': t_s + KIN_DURATION_S}
            self.kinetic_runs.append({'run': n, 'wavelength_nm': self.wavelength_nm,
                                      'blank_wavelength_nm': self.reference['wavelength_nm'] if self.reference else None,
                                      'rows': []})
            self.kinetic_private.append({'run': n, 'points': []})
            return True
        self.busy = (kind, t_s + (SCAN_S if kind == 'scan' else MEASURE_S))
        return True

    def update(self, t_s, container):
        """Complete a running BLANK/READ/SCAN at its due time with the cuvette now in the slot (or None); take the
        due point of a timed KIN run."""
        if self.kinetic is not None:
            return self._kinetic(t_s, container)
        if self.busy is None or t_s + 1e-9 < self.busy[1]:
            return None
        kind, _ = self.busy
        self.busy = None
        if kind == 'scan':
            return self._scan(t_s, container)
        if kind == 'blank':
            self.references = {}
            for nm in self.active():
                content, condition = self._content(container, nm)
                solvent_only = (condition == 'ok' and container is not None
                                and not any(v > 0 for v in container.composition_M().values()))
                self.references[nm] = {'wavelength_nm': nm, 's_ref': S.signal(content, nm, self.b0, self.b1, self.stray),
                                       'container': None if container is None else container.id,
                                       'solvent_only': bool(solvent_only), 'condition': condition}
            self.reference = self.references[self.active()[-1]]
            self.last_display = 0.0
            self.screen = {nm: {'A': 0.0, 'source': {'kind': 'blank'}} for nm in self.active()}
            return {'kind': 'blank'}
        for nm in self.active():
            content, condition = self._content(container, nm)
            s = S.signal(content, nm, self.b0, self.b1, self.stray)
            reference = self.references.get(nm)
            s_ref = reference['s_ref'] if reference else S.factory_reference(self.stray)
            a = S.reading(s_ref, s)
            a += S.noise_sigma(a) * float(self.rng.standard_normal())
            shown = round(a, 3) + 0.0          # no negative zero on the display
            status = condition
            if status == 'ok' and reference is None:
                status = 'no_blank_at_λ'
            elif status == 'ok' and not (reference['solvent_only'] and reference['condition'] == 'ok'):
                status = 'blank_ref_not_solvent'
            sample_id = len(self.rows) + 1
            # without a reference here the row names the wavelength of the last BLANK, as on a single-beam instrument
            blank_nm = nm if reference else (self.reference['wavelength_nm'] if self.reference else None)
            self.rows.append({'sample_id': sample_id, 't_s': round(t_s, 6), 'wavelength_nm': nm, 'A_display': shown,
                              'blank_wavelength_nm': blank_nm})
            self.private.append({'sample_id': sample_id, 'status': status,
                                 'cuvette': None if container is None else container.id,
                                 'nominal_history': [] if container is None else deepcopy(container.nominal),
                                 'volume_mL': None if container is None else container.volume_mL,
                                 'spilled': None if container is None else container.spilled,
                                 'blank_cuvette': reference['container'] if reference else None})
            self.screen[nm] = {'A': shown, 'source': {'kind': 'read', 'sample_id': sample_id}}
        self.last_display = shown
        return {'kind': 'read', 'sample_id': sample_id, 'A_display': shown}

    def _absorbance(self, container):
        """(displayed A, private status) of the cuvette in the slot at the current wavelength, as READ."""
        content, condition = self._content(container)
        s = S.signal(content, self.wavelength_nm, self.b0, self.b1, self.stray)
        s_ref = self.references[self.wavelength_nm]['s_ref'] if self.blank_valid() else S.factory_reference(self.stray)
        a = S.reading(s_ref, s)
        a += S.noise_sigma(a) * float(self.rng.standard_normal())
        status = condition
        if status == 'ok' and not self.blank_valid():
            status = 'no_blank_at_λ'
        elif status == 'ok' and not (self.reference['solvent_only'] and self.reference['condition'] == 'ok'):
            status = 'blank_ref_not_solvent'
        return round(a, 3) + 0.0, status

    def _kinetic(self, t_s, container):
        run = self.kinetic
        if t_s + 1e-9 < run['next_s']:
            return None
        shown, status = self._absorbance(container)
        self.last_display = shown
        point = len(self.kinetic_runs[-1]['rows']) + 1
        self.screen = {self.wavelength_nm: {'A': shown, 'source': {'kind': 'kin', 'run': run['run'], 'point': point}}}
        self.kinetic_runs[-1]['rows'].append({'point': point, 't_s': round(t_s, 6), 'A_display': shown})
        self.kinetic_private[-1]['points'].append({
            'point': point, 'status': status, 'cuvette': None if container is None else container.id,
            'nominal_history': [] if container is None else deepcopy(container.nominal),
            'volume_mL': None if container is None else container.volume_mL,
            'spilled': None if container is None else container.spilled,
            'blank_cuvette': self.reference['container'] if self.blank_valid() else None})
        run['next_s'] += KIN_INTERVAL_S
        if run['next_s'] > run['until_s'] + 1e-9:
            self.kinetic = None
        return {'kind': 'kin', 'run': run['run'], 'point': point}

    def export_kinetics(self):
        n = len(self.kinetic_exports) + 1
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator='\n')
        writer.writerow(('run', 'wavelength_nm', 'blank_wavelength_nm', 'point', 't_s', 'A_display'))
        for run in self.kinetic_runs:
            for row in run['rows']:
                writer.writerow((run['run'], run['wavelength_nm'], run['blank_wavelength_nm'], row['point'], row['t_s'],
                                 row['A_display']))
        text = buffer.getvalue()
        record = {'export_id': f'{self.episode_id}:kinetics_log:{n}', 'runs': deepcopy(self.kinetic_runs), 'csv': text,
                  'sha256': hashlib.sha256(text.encode()).hexdigest(),
                  'evidence_ids': [f'{self.episode_id}:kinetics:{r["run"]}' for r in self.kinetic_runs]}
        self.kinetic_exports[record['export_id']] = record
        return deepcopy(record)

    def _scan(self, t_s, container):
        grid = list(range(WAVELENGTH_NM[0], WAVELENGTH_NM[1] + 1, SCAN_STEP_NM))
        values, condition = [], None
        for nm in grid:
            content, condition = self._content(container, nm)
            a = S.reading(S.factory_reference(self.stray), S.signal(content, nm, self.b0, self.b1, self.stray))
            a += S.noise_sigma(a) * float(self.rng.standard_normal())
            values.append(round(a, 3) + 0.0)
        scan_id = len(self.scans) + 1
        self.scans.append({'scan_id': scan_id, 't_s': round(t_s, 6), 'wavelength_nm': grid, 'A': values})
        self.scan_private.append({'scan_id': scan_id, 'cuvette': None if container is None else container.id,
                                  'condition': condition})
        return {'kind': 'scan', 'scan_id': scan_id}

    def _content(self, container, wavelength_nm=None):
        """Absorbance of what the beam passes through and the private condition code."""
        if container is None:
            return None, 'slot_empty'
        if container.overflowed:
            condition = 'overflowed'
        elif container.volume_mL < MIN_VOLUME_ML:
            return None, 'low_volume'      # the beam passes above the liquid
        elif not container.mixed:
            condition = 'not_mixed'
        else:
            condition = 'ok'
        if getattr(container, 'contact', False):
            # contact model (reaction benches): the compartments in the beam band, their mean transmittance
            nm = self.wavelength_nm if wavelength_nm is None else wavelength_nm
            t = sum(w * 10.0 ** -S.absorbance({k: v for k, v in comp.items() if k in self.dyes}, self.dyes, nm)
                    for w, comp in container.beam_parts())
            return float(-np.log10(t)), condition
        comp = container.layer_at(float(self.rng.random())) if not container.mixed else container.composition_M()
        dyes = {k: v for k, v in comp.items() if k in self.dyes}
        return S.absorbance(dyes, self.dyes, self.wavelength_nm if wavelength_nm is None else wavelength_nm), condition

    # --- evidence ---------------------------------------------------------------------------------
    def export_scans(self):
        n = len(self.scan_exports) + 1
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator='\n')
        writer.writerow(('scan_id', 't_s', 'wavelength_nm', 'A'))
        for scan in self.scans:
            for nm, a in zip(scan['wavelength_nm'], scan['A']):
                writer.writerow((scan['scan_id'], scan['t_s'], nm, a))
        text = buffer.getvalue()
        record = {'export_id': f'{self.episode_id}:spectro_scans:{n}', 'scans': deepcopy(self.scans), 'csv': text,
                  'sha256': hashlib.sha256(text.encode()).hexdigest()}
        self.scan_exports[record['export_id']] = record
        return deepcopy(record)


# --- screen text (seven-segment strings; chem_apparatus draws them) -----------------------------------
def _absorbance_text(value, busy):
    if busy:
        return ' ----'
    if value is None:
        return '     '
    a = max(-9.999, min(9.999, float(value)))
    return ('-' if a < 0 else ' ') + f'{abs(a):.3f}'.replace('.', '')


def screen_text(shown):
    """Seven-segment strings of the spectrophotometer screen from ChemBench.display (the dual-wavelength screen when
    it carries 'rows': row 1 the earlier entered wavelength, row 2 the one entered last, which also shows the digits
    being typed and Err)."""
    if 'rows' in shown:
        rows = [None] * (2 - len(shown['rows'])) + list(shown['rows'])
        text = {'prompt': shown['prompt'] == 'BLANK', 'entry': bool(shown['entry'])}
        for k, row in enumerate(rows, start=1):
            if k == 2 and shown['error']:
                wl = 'Err'
            elif k == 2 and shown['entry']:
                wl = shown['entry'].rjust(3)
            else:
                wl = '   ' if row is None else f"{row['wavelength_nm']:3d}"
            value = None if row is None else row['absorbance']
            text[f'wavelength{k}'] = wl
            text[f'absorbance{k}'] = _absorbance_text(value, shown['busy'] and row is not None)
            text[f'point{k}'] = value is not None and not shown['busy']
        return text
    if shown['error']:
        wl = 'Err'
    elif shown['entry']:
        wl = shown['entry'].rjust(3)
    else:
        wl = f"{shown['wavelength_nm']:3d}"
    if shown['busy']:
        ab = ' ----'
    elif shown['absorbance'] is None:
        ab = '     '
    else:
        a = max(-9.999, min(9.999, float(shown['absorbance'])))
        ab = ('-' if a < 0 else ' ') + f'{abs(a):.3f}'.replace('.', '')
    return {'wavelength': wl, 'absorbance': ab, 'prompt': shown['prompt'] == 'BLANK', 'entry': bool(shown['entry']),
            'point': shown['absorbance'] is not None and not shown['busy']}

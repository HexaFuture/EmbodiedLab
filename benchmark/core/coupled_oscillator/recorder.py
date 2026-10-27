"""Public quantized encoder, driven only by the trusted physics tick loop.

No pose setter, dynamics parameter, analytical trajectory or fitted answer is
part of this interface. Display and CSV consume the same stored samples.
"""
import csv
from copy import deepcopy
import hashlib
import io
import math


class DisplacementRecorder:
    sample_hz = 120
    resolution_mm = 0.1
    range_mm = 100.0

    def __init__(self, episode_id, physics_hz=240):
        if physics_hz not in (240, 480):
            raise ValueError('unsupported physics clock')
        self.episode_id = episode_id
        self.physics_hz = physics_hz
        self.interval = physics_hz // self.sample_hz
        self.active = None
        self.latest = None
        self.records = {}
        self.ticks = {}       # evidence id -> [start tick, end tick] (process record only)
        self._last_tick = None
        self._pending = 0

    def begin_release(self, tick, duration_s, *, physical_release_valid):
        if self.active is not None:
            raise ValueError('acquisition already active')
        if type(tick) is not int or tick < 0:
            raise ValueError('invalid release tick')
        if (isinstance(duration_s, bool) or not isinstance(duration_s, (int, float))
                or not math.isfinite(duration_s) or not 5 <= duration_s <= 180):
            raise ValueError('recording duration must be 5 to 180 simulated seconds')
        if physical_release_valid is not True:
            raise ValueError('physical release was not confirmed')
        eid = f'{self.episode_id}:encoder:{len(self.records) + 1}'
        self.active = dict(evidence_id=eid, start_tick=tick,
                           end_tick=tick + math.ceil(duration_s * self.physics_hz),
                           rows=[], reasons=set(), validity_events={}, contact_summary={}, status=[])
        self._last_tick = None
        self._pending = 0
        return eid

    # Per-sample status (private, `sample_status` beside the rows): the worst condition seen on any
    # physics tick since the previous sample, by ascending severity; 'ok' = a clean sample. Record only: the record's
    # status and invalid_reasons are unchanged.
    SAMPLE_STATUSES = ('ok', 'brake_engaged', 'external_contact', 'over_range_or_nonfinite')

    def sample(self, tick, displacements_m, *, external_contact, brake_engaged, contact_diagnostics=None):
        if type(tick) is not int or tick < 0:
            raise ValueError('invalid sample tick')
        if type(external_contact) is not bool or type(brake_engaged) is not bool:
            raise ValueError('physical validity sensors must be boolean')
        if len(displacements_m) != 2:
            raise ValueError('two encoder channels required')
        finite = all(isinstance(x, (float, int)) and not isinstance(x, bool)
                     and math.isfinite(x) for x in displacements_m)
        valid = finite and all(abs(x * 1000) <= self.range_mm for x in displacements_m)
        a = self.active
        if a:
            if tick < a['start_tick'] or (self._last_tick is not None and tick != self._last_tick + 1):
                raise ValueError('encoder requires every consecutive physics tick')
            self._last_tick = tick
            if external_contact and contact_diagnostics is not None:
                c = a['contact_summary']
                c['categories'] = sorted(set(c.get('categories', [])) | set(contact_diagnostics['categories']))
                c['max_normal_force_n'] = max(c.get('max_normal_force_n', 0.), contact_diagnostics['max_normal_force_n'])
                c['min_separation_m'] = min(c.get('min_separation_m', float('inf')), contact_diagnostics['min_separation_m'])
            for reason, triggered in (
                ('external_contact_during_recording', external_contact),
                ('brake_engaged_during_recording', brake_engaged),
                ('over_range_or_nonfinite', not valid),
            ):
                if triggered:
                    a['reasons'].add(reason)
                    elapsed = (tick-a['start_tick'])/self.physics_hz
                    event = a['validity_events'].setdefault(reason, dict(
                        first_t_s=elapsed, last_t_s=elapsed, physics_tick_count=0))
                    event['last_t_s'] = elapsed
                    event['physics_tick_count'] += 1
            self._pending = max(self._pending, 3 if not valid else 2 if external_contact else 1 if brake_engaged else 0)
        origin = a['start_tick'] if a else 0
        if (tick - origin) % self.interval == 0:
            row = dict(t_s=(tick-origin)/self.physics_hz,
                       x1_mm=round(displacements_m[0]*1000/self.resolution_mm)*self.resolution_mm if valid else None,
                       x2_mm=round(displacements_m[1]*1000/self.resolution_mm)*self.resolution_mm if valid else None,
                       status='OK' if valid else 'OVER_RANGE')
            self.latest = row
            if a:
                a['rows'].append(row.copy())
                a['status'].append(self.SAMPLE_STATUSES[self._pending])
                self._pending = 0
        if a and tick >= a['end_tick']:
            if not a['rows'] or a['rows'][0]['t_s'] > 1/self.sample_hz:
                a['reasons'].add('missing_release_samples')
            self._finish()

    def _finish(self):
        a = self.active
        self.ticks[a['evidence_id']] = [a['start_tick'], self._last_tick]
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=('t_s','x1_mm','x2_mm','status'))
        writer.writeheader()
        writer.writerows(a['rows'])
        data = buffer.getvalue()
        self.records[a['evidence_id']] = dict(
            schema_version='embodiedlab.coupled-encoder/v1',
            episode_id=self.episode_id, evidence_id=a['evidence_id'],
            status='invalid' if a['reasons'] else 'complete',
            invalid_reasons=sorted(a['reasons']),
            validity_diagnostics=deepcopy(a['validity_events']),
            contact_diagnostics=deepcopy(a['contact_summary']), sample_status=list(a['status']), sample_hz=self.sample_hz,
            resolution_mm=self.resolution_mm, range_mm=[-self.range_mm,self.range_mm],
            clock='simulation_time', rows=deepcopy(a['rows']), csv=data,
            sha256=hashlib.sha256(data.encode()).hexdigest())
        self.active = None

    # What an encoder knows. status, invalid_reasons, validity_diagnostics, contact_diagnostics and sample_status
    # are the bench's own books: they stay in the sealed record (self.records) for the evaluator.
    PUBLIC_FIELDS = ('schema_version', 'episode_id', 'evidence_id', 'sample_hz', 'resolution_mm', 'range_mm',
                     'clock', 'rows', 'csv', 'sha256')

    def export(self, evidence_id):
        if evidence_id not in self.records:
            raise ValueError('unknown, unfinished or cross-episode acquisition')
        record = self.records[evidence_id]
        return {key: deepcopy(record[key]) for key in self.PUBLIC_FIELDS}

    def reading(self):
        """Private state for the bench logs (no tool returns it)."""
        return dict(latest=deepcopy(self.latest),
                    acquisition='recording' if self.active else 'idle',
                    completed_evidence_ids=list(self.records),
                    active_validity_diagnostics=deepcopy(self.active['validity_events']) if self.active else {},
                    active_contact_diagnostics=deepcopy(self.active['contact_summary']) if self.active else {},
                    active_evidence_id=self.active['evidence_id'] if self.active else None)

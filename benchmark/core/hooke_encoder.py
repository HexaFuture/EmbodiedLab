"""Public tray encoder of the Hooke bench (kit mechanics 0.5), driven by the physics tick loop.

One channel z_mm, positive downward, zero 75 mm below the start of travel (the
authored empty-tray height), so the +/-100 mm range covers 25 mm above to
175 mm below the start. 120 Hz, 0.1 mm ideal quantisation, no simulated noise.
The public rows and CSV hold sample_id, t_s and z_mm only (z_mm empty on overtravel), as a real encoder
file does. Every physics tick is checked and each sample's status, the worst seen since the previous sample
(overtravel > contact > brake > load_lifted > ok; brake: the damper lever not at OFF), is kept for the evaluator
only (`sample_status` and `status_counts` in the private record). The private record also keeps
`sample_load_ids`, parallel to the rows: the weights resting on the tray when each sample was taken (the same rule as
load_ids_at_end, backend.encoder_load_ids), so a weight put on or taken off mid-recording is placed in time
(record only; nothing else changes). A recording is started with REC on the bench's
recorder panel; neither the panel nor the lever appears in the record. No stiffness, mass or fitted period is part
of this interface.
"""
import csv
from copy import deepcopy
import hashlib
import io
import math

STATUSES = ('ok', 'load_lifted', 'brake', 'contact', 'overtravel')  # ascending severity


class TrayEncoder:
    sample_hz = 120
    resolution_mm = 0.1
    range_mm = 100.0
    zero_below_start_m = 0.075
    arm_s = (10.0, 120.0)

    def __init__(self, episode_id, physics_hz=240):
        if physics_hz not in (240, 480):
            raise ValueError('unsupported physics clock')
        self.episode_id = episode_id
        self.physics_hz = physics_hz
        self.interval = physics_hz // self.sample_hz
        self.active = None
        self.latest = None
        self.records = {}
        self.private = {}
        self._pending = 0
        self._last_tick = None
        self._samples = 0

    def arm(self, tick, duration_s, remaining_ticks, load_ids):
        if self.active is not None:
            raise ValueError('encoder acquisition already active')
        if (isinstance(duration_s, bool) or not isinstance(duration_s, (int, float))
                or not math.isfinite(duration_s) or not self.arm_s[0] <= duration_s <= self.arm_s[1]):
            raise ValueError(f'arm duration must be {self.arm_s[0]:g} to {self.arm_s[1]:g} simulated seconds')
        ticks = math.ceil(duration_s * self.physics_hz)
        if ticks > remaining_ticks:
            raise ValueError('arm duration exceeds the remaining episode budget')
        eid = f'{self.episode_id}:encoder:{len(self.records) + 1}'
        self.active = dict(evidence_id=eid, start_tick=tick, end_tick=tick + ticks, duration_s=float(duration_s),
                           rows=[], status=[], loads=[], counts={s: 0 for s in STATUSES}, load_ids_at_arm=sorted(load_ids))
        self._pending = 0
        return eid

    def sample(self, tick, displacement_m, *, contact, load_lifted, brake=False, at_limit=False, load_ids=None):
        """One physics tick. displacement_m: tray position below the start of travel (trusted physics).

        load_ids: callable returning the weights resting on the tray, called at every recorded sample (private
        sample_load_ids) and once more when an acquisition ends (load_ids_at_end).
        """
        if any(type(v) is not bool for v in (contact, load_lifted, brake, at_limit)):
            raise ValueError('encoder validity sensors must be boolean')
        a = self.active
        if a is not None:
            if tick != a.setdefault('next_tick', a['start_tick'] + 1):
                raise ValueError('encoder requires every consecutive physics tick')
            a['next_tick'] += 1
        z_mm = (displacement_m - self.zero_below_start_m) * 1000 if math.isfinite(displacement_m) else math.inf
        over = at_limit or not abs(z_mm) <= self.range_mm
        self._pending = max(self._pending, 4 if over else 3 if contact else 2 if brake else 1 if load_lifted else 0)
        origin = a['start_tick'] if a else 0
        if (tick - origin) % self.interval == 0:
            name = STATUSES[self._pending]
            self._pending = 0
            row = dict(sample_id=None, t_s=round((tick - origin) / self.physics_hz, 6),
                       z_mm=None if name == 'overtravel'
                       else round(round(z_mm / self.resolution_mm) * self.resolution_mm, 1))
            if a is not None:
                row['sample_id'] = len(a['rows']) + 1
                a['rows'].append(row)
                a['status'].append(name)
                a['counts'][name] += 1
                a['loads'].append(sorted(load_ids()) if load_ids is not None else None)
            self.latest = row
        if a is not None and tick >= a['end_tick']:
            self._finish(load_ids() if load_ids is not None else None)

    def _finish(self, load_ids_at_end):
        a = self.active
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=('sample_id', 't_s', 'z_mm'))
        writer.writeheader()
        writer.writerows(a['rows'])
        data = buffer.getvalue()
        eid = a['evidence_id']
        self.records[eid] = dict(
            schema_version='embodiedlab.tray-encoder/v1', episode_id=self.episode_id, evidence_id=eid,
            status='complete', duration_s=a['duration_s'], sample_hz=self.sample_hz,
            resolution_mm=self.resolution_mm, range_mm=[-self.range_mm, self.range_mm],
            zero='75 mm below the start of travel; positive downward', clock='simulation_time',
            rows=a['rows'], csv=data,
            sha256=hashlib.sha256(data.encode()).hexdigest())
        # Trusted load bookkeeping for the evaluator; never returned to the Agent.
        self.private[eid] = dict(start_tick=a['start_tick'], end_tick=a['end_tick'],
                                 load_ids_at_arm=a['load_ids_at_arm'],
                                 load_ids_at_end=sorted(load_ids_at_end) if load_ids_at_end is not None else None,
                                 sample_status=list(a['status']), status_counts=dict(a['counts']),
                                 sample_load_ids=deepcopy(a['loads']))
        self.active = None

    def export(self, evidence_id):
        if evidence_id not in self.records:
            raise ValueError('unknown, unfinished or cross-episode acquisition')
        return deepcopy(self.records[evidence_id])

    def reading(self):
        a = self.active
        return dict(latest=deepcopy(self.latest), acquisition='recording' if a else 'idle',
                    active_evidence_id=a['evidence_id'] if a else None,
                    active_elapsed_s=round(len(a['rows']) / self.sample_hz, 3) if a else None,
                    completed_evidence_ids=list(self.records),
                    sample_hz=self.sample_hz, resolution_mm=self.resolution_mm,
                    range_mm=[-self.range_mm, self.range_mm])

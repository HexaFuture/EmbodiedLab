"""Photogate of the torsion bench (kit mechanics, torsion_inertia and torsion_centroid), driven by the physics tick loop.

A flag on the turntable rim passes through a fixed fork at the rest angle, so
the beam is interrupted twice per oscillation period. Every physics tick is
checked. Beam edges are timed on the simulation clock and interpolated between
the bracketing ticks (sub-tick timing), reported to 0.1 ms. A record starts when
REC is keyed on the recorder panel (torsion_backend) and runs for the keyed length.

The public record is what a photogate can know: the transits (block and clear
times). Each transit also carries the worst bench condition seen during it and
since the previous transit, kept for the evaluator only (`transit_status` and
`status_counts` in the private record): contact > brake > load_changing > ok. Each transit also
keeps the set of conditions seen (`transit_conditions`, private; the bench passes them, e.g. which kind of object the
robot touched: one on the turntable or one elsewhere), for the evaluator only; the public record is unchanged.
No stiffness, inertia, angle or fitted period is part of this interface.
"""
import csv
from copy import deepcopy
import hashlib
import io
import math

STATUSES = ('ok', 'load_changing', 'brake', 'contact')      # ascending severity
RESOLUTION_S = 1e-4


class Photogate:
    arm_s = (5.0, 120.0)

    def __init__(self, episode_id, physics_hz=240, record_range=None):
        if physics_hz not in (240, 480):
            raise ValueError('unsupported physics clock')
        if record_range is not None:
            self.arm_s = tuple(float(v) for v in record_range)
        self.episode_id = episode_id
        self.physics_hz = physics_hz
        self.active = None
        self.records = {}
        self.private = {}
        self.blocked = None
        self._previous = None          # (tick, signed distance to the beam edge)
        self._pending = 0
        self._pending_conditions = set()

    def arm(self, tick, duration_s, remaining_ticks, loads):
        if self.active is not None:
            raise ValueError('photogate acquisition already active')
        if (isinstance(duration_s, bool) or not isinstance(duration_s, (int, float))
                or not math.isfinite(duration_s) or not self.arm_s[0] <= duration_s <= self.arm_s[1]):
            raise ValueError(f'arm duration must be {self.arm_s[0]:g} to {self.arm_s[1]:g} simulated seconds')
        ticks = math.ceil(duration_s * self.physics_hz)
        if remaining_ticks is not None and ticks > remaining_ticks:
            raise ValueError('arm duration exceeds the remaining episode budget')
        eid = f'{self.episode_id}:photogate:{len(self.records) + 1}'
        self.active = dict(evidence_id=eid, start_tick=tick, end_tick=tick + ticks, duration_s=float(duration_s),
                           rows=[], status=[], open=None, counts={s: 0 for s in STATUSES}, loads_at_arm=deepcopy(loads),
                           blocked_at_arm=bool(self.blocked), max_angle=0.0, conditions=[])
        self._pending = 0
        self._pending_conditions = set()
        return eid

    def sample(self, tick, angle_rad, half_angle_rad, *, contact, brake, load_changing, loads=None, conditions=None):
        """One physics tick with the trusted turntable angle relative to rest.

        loads: callable returning the private load bookkeeping, called when an acquisition ends.
        conditions: optional names of the bench conditions of this tick, kept per transit (private record only).
        """
        if any(type(v) is not bool for v in (contact, brake, load_changing)):
            raise ValueError('photogate validity sensors must be boolean')
        if not math.isfinite(angle_rad):
            raise ValueError('non-finite turntable angle')
        distance = abs(angle_rad) - half_angle_rad          # < 0 while the flag is in the beam
        blocked = distance <= 0
        a = self.active
        severity = 3 if contact else 2 if brake else 1 if load_changing else 0
        if a is not None:
            if tick != a.setdefault('next_tick', a['start_tick'] + 1):
                raise ValueError('photogate requires every consecutive physics tick')
            a['next_tick'] += 1
            a['max_angle'] = max(a['max_angle'], abs(angle_rad))
            self._pending = max(self._pending, severity)
            if conditions:
                self._pending_conditions.update(conditions)
            if self.blocked is not None and blocked != self.blocked and self._previous is not None:
                previous_tick, previous_distance = self._previous
                fraction = previous_distance / (previous_distance - distance) if previous_distance != distance else 1.0
                t_edge = (previous_tick + min(1.0, max(0.0, fraction)) - a['start_tick']) / self.physics_hz
                if blocked:
                    a['open'] = dict(t_block_s=t_edge, severity=self._pending)
                elif a['open'] is not None:
                    status = STATUSES[max(self._pending, a['open']['severity'])]
                    row = dict(transit_id=len(a['rows']) + 1, t_block_s=round(a['open']['t_block_s'], 4),
                               t_clear_s=round(t_edge, 4))
                    a['rows'].append(row)
                    a['status'].append(status)
                    a['counts'][status] += 1
                    a['conditions'].append(sorted(self._pending_conditions))
                    a['open'] = None
                    self._pending = 0
                    self._pending_conditions = set()
        self.blocked = blocked
        self._previous = (tick, distance)
        if a is not None and tick >= a['end_tick']:
            self._finish(loads() if loads is not None else None)

    def _finish(self, loads_at_end):
        a = self.active
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=('transit_id', 't_block_s', 't_clear_s'))
        writer.writeheader()
        writer.writerows(a['rows'])
        data = buffer.getvalue()
        eid = a['evidence_id']
        self.records[eid] = dict(
            schema_version='embodiedlab.photogate/v2', episode_id=self.episode_id, evidence_id=eid,
            record_number=int(eid.rsplit(':', 1)[-1]),
            duration_s=a['duration_s'], clock='simulation_time', time_resolution_s=RESOLUTION_S,
            timing='beam edges interpolated between physics ticks; t from REC',
            beam_blocked_at_arm=a['blocked_at_arm'], rows=a['rows'], csv=data,
            sha256=hashlib.sha256(data.encode()).hexdigest())
        # Trusted bookkeeping for the evaluator (loads, the bench condition of every transit); never returned to the
        # Agent.
        self.private[eid] = dict(status='complete', start_tick=a['start_tick'], end_tick=a['end_tick'], loads_at_arm=a['loads_at_arm'],
                                 loads_at_end=deepcopy(loads_at_end), max_angle_rad=a['max_angle'],
                                 transit_status=list(a['status']), status_counts=dict(a['counts']),
                                 transit_conditions=deepcopy(a['conditions']))
        self.active = None

    def resolve(self, evidence_id):
        """A finished record by its evidence id, its record number (the NO shown on the display) or 'latest'."""
        if evidence_id == 'latest':
            if not self.records:
                raise ValueError('no finished photogate record yet')
            return next(reversed(self.records))
        if isinstance(evidence_id, int) and not isinstance(evidence_id, bool) or (
                isinstance(evidence_id, str) and evidence_id.isdigit()):
            return f'{self.episode_id}:photogate:{int(evidence_id)}'
        return evidence_id

    def export(self, evidence_id):
        evidence_id = self.resolve(evidence_id)
        if evidence_id not in self.records:
            raise ValueError('unknown, unfinished or cross-episode record')
        return deepcopy(self.records[evidence_id])

    def reading(self):
        """Private state for the bench logs (no tool returns it)."""
        a = self.active
        return dict(beam_blocked=bool(self.blocked), acquisition='recording' if a else 'idle',
                    active_evidence_id=a['evidence_id'] if a else None,
                    active_transits=len(a['rows']) if a else None,
                    active_status_counts=dict(a['counts']) if a else None,
                    completed_evidence_ids=list(self.records), time_resolution_s=RESOLUTION_S)

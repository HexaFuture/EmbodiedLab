"""Photogate of torsion_centroid: the torsion photogate plus the evaluator's private mounting trace.

At 30 Hz during a record the bench's private `mount` (which object sits on the pin, through which hole) is kept in
the private record (`mount_samples`: [t_s, load, hole]); the Agent's export is the photogate record only.
"""
import hashlib
import json
from torsion_gate import Photogate


class CentroidPhotogate(Photogate):
    def __init__(self, episode_id, physics_hz, mount_reader, record_range=None):
        super().__init__(episode_id, physics_hz, record_range)
        self.mount_reader = mount_reader

    def arm(self, *args, **kwargs):
        eid = super().arm(*args, **kwargs)
        self.active['mount_samples'] = [[0.0, *self.mount_reader()]]
        return eid

    def sample(self, tick, *args, **kwargs):
        if self.active and tick % (self.physics_hz // 30) == 0:
            self.active['mount_samples'].append(
                [round((tick - self.active['start_tick']) / self.physics_hz, 4), *self.mount_reader()])
        return super().sample(tick, *args, **kwargs)

    def _finish(self, loads_at_end):
        eid = self.active['evidence_id']
        samples = self.active['mount_samples']
        super()._finish(loads_at_end)
        self.private[eid]['mount_samples'] = samples
        record = self.records[eid]
        # Hash of the whole public record, retaining the photogate CSV's own hash.
        record['payload_sha256'] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()

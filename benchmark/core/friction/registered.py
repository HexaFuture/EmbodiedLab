"""Registered incline_friction episode: instance-driven coefficients, submission freeze and sealing.

The development episode (episode.py) and the apparatus are unchanged; this adds what a registry experiment needs.
"""
from copy import deepcopy
import hashlib
import json

from instance_episodes import InstanceMixin
from .episode import FrictionDevelopmentEpisode, TOOLS
from .instrument import InclineRecorder


def friction_parameters(instance):
    """{'A': {'mu_s', 'mu_k'}, 'B': {...}}: each slider's underside against the deck (two sliders)."""
    h = instance['hidden']
    out = {}
    for k in ('A', 'B'):
        mu_s, mu_k = h['mu_s_' + k], h['mu_k_' + k]
        if abs(mu_k - mu_s * h['mu_k_over_mu_s_' + k]) > 1e-12 or not 0 < mu_k < mu_s:
            raise ValueError('friction instance is inconsistent')
        out[k] = {'mu_s': float(mu_s), 'mu_k': float(mu_k)}
    return out


def canonical_sha256(value):
    # Same canonical form as benchmark/evaluation/sealed.py.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def friction_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    class InclineEpisode(InstanceMixin, FrictionDevelopmentEpisode):
        def public_task(self):
            development = FrictionDevelopmentEpisode.public_task(self)
            return self.instance_public_task({'apparatus': development['apparatus']})

        def seal_payload(self):
            """Terminal state, device, evidence and files of the sealed record (also used at stop)."""
            recorder = self.backend.recorder
            evidence = {eid: {'kind': 'incline_record', 'public': deepcopy(record),
                              'sha256': canonical_sha256(record), 'private_rows': deepcopy(recorder.private[eid])}
                        for eid, record in recorder.records.items()}
            return (
                self.backend.bench_state('seal'),
                {'gravity_m_s2': 9.81, 'angle_resolution_deg': recorder.angle_resolution_deg,
                 'position_resolution_m': recorder.position_resolution_m}, evidence)

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if name == 'submit_result':
                return self.freeze_and_seal(args, *self.seal_payload())
            return super()._dispatch(name, args)

        def process_extra(self):
            return self.backend.process_extra()

        def evaluation(self):
            return self.instance_evaluation()

    episode = InclineEpisode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    # Evidence ids are bound to the episode; the recorder restarts empty (nothing was recorded during set-up).
    fresh = InclineRecorder(physics_hz, prefix=episode.episode_id)
    fresh.last_tick, fresh.latest = backend.recorder.last_tick, backend.recorder.latest
    backend.recorder = fresh
    return episode


PUBLIC_TOOLS = TOOLS + ('submit_result',)

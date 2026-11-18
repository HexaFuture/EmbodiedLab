"""Registered rectifier_supply episode: instance-driven components, programmable load, submission sealing.

The device (bench, solver, scope, submission checks, post-submission waveform) is unchanged. Added here: the
instance's hidden components, the electronic load's remote setpoint and the sealed record for offline evaluation.
"""
from copy import deepcopy
import json
import math

from instance_episodes import InstanceMixin
from .bench import RectifierBench
from .episode import RectifierEpisode

NOMINAL_F = (1.0e-4, 2.2e-4, 4.7e-4)
FACTORS = ('c100_factor', 'c220_factor', 'c470_factor')
LOAD_RANGE_OHM = (150.0, 3300.0)


def rectifier_configuration(instance):
    h = instance['hidden']
    constants = instance['public_task']['public_constants']
    return {'capacitances_F': [c * float(h[k]) for c, k in zip(NOMINAL_F, FACTORS)], 'diode_n': float(h['diode_n']),
            'final_load_ohm': float(constants['objective']['final_load_ohm']),
            'initial': {'CAP': 'C100', 'LOAD': '704', 'MODE': 'OFF'}}


def load_position_m(ohm):
    return 0.06 * math.log(ohm / LOAD_RANGE_OHM[0]) / math.log(LOAD_RANGE_OHM[1] / LOAD_RANGE_OHM[0])


def rectifier_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    class Episode(InstanceMixin, RectifierEpisode):
        def public_task(self):
            device = RectifierEpisode.public_task(self)
            apparatus = dict(device['apparatus'])
            apparatus['controls'] = ('CAP SELECT and MODE are posts with detents. The electronic load is programmed '
                                     'with set_load_resistance; its slider follows the setpoint.')
            return self.instance_public_task({'apparatus': apparatus, 'circuit': device['circuit'],
                                              'objective': device['objective']})

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if not isinstance(args, dict):
                raise ValueError('arguments must be an object')
            if name in ('start_recording', 'stop_recording') and args == {}:
                args = {'expected_state_id': self.state_id}
            if name == 'set_load_resistance':
                if set(args) != {'ohm'} or type(args['ohm']) not in (int, float) or not math.isfinite(args['ohm']) \
                        or not LOAD_RANGE_OHM[0] <= args['ohm'] <= LOAD_RANGE_OHM[1]:
                    raise ValueError('ohm must be a number within the load range')
                self.backend.set_load_target(load_position_m(float(args['ohm'])))
                self.revision += 1
                return {'setpoint_ohm': float(args['ohm']), 'state_id': self.state_id,
                        'note': 'The load slider moves to the setpoint; read_instruments shows the actual resistance.'}
            if name == 'submit_result':
                return self._submit_sealed(args)
            return super()._dispatch(name, args)

        def _submit_sealed(self, answer):
            bench = self.backend.bench
            problem = None
            try:
                RectifierEpisode.submit(self, {'expected_state_id': self.state_id, 'answer': answer})
            except ValueError as exc:
                problem = str(exc)
            waveform = None
            path = self.backend.private / 'post_submit_waveform.json'
            if problem is None and path.exists():
                full = json.loads(path.read_text())
                waveform = {k: full[k] for k in ('start_tick', 'end_tick', 'mean_voltage_V', 'ripple_pp_V', 'configuration_valid')}
            evidence = {key: {'kind': 'rectifier_scope', 'sha256': record['sha256'], 'public': bench.record_payload(key)}
                        for key, record in bench.records.items() if record['end_tick'] is not None}
            return self.freeze_and_seal(deepcopy(answer),
                {'switch': list(bench.switch), 'recording_active': bench.active is not None},
                {'validation_problem': problem, 'post_submit': waveform,
                 'final_load_ohm': self.backend.configuration['final_load_ohm']}, evidence)

        def seal_payload(self):
            """The records at stop; the submission's device acceptance and waveform do not run here."""
            bench = self.backend.bench
            evidence = {key: {'kind': 'rectifier_scope', 'sha256': record['sha256'], 'public': bench.record_payload(key)}
                        for key, record in bench.records.items() if record['end_tick'] is not None}
            return ({'switch': list(bench.switch), 'recording_active': bench.active is not None},
                    {'final_load_ohm': self.backend.configuration['final_load_ohm'],
                     'selector_actions': deepcopy(bench.actions)}, evidence)

        def evaluation(self):
            return self.instance_evaluation()

    episode = Episode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    # Evidence ids are bound to the episode: "<episode>:rectifier-scope-<n>".
    backend.bench = RectifierBench(episode.episode_id + ':rectifier', backend.configuration)
    return episode

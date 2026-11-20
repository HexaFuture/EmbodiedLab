"""Dry-block episodes (thermistor_beta, rod_expansion): configuration of one instance and the device runtime.

There is no bench tool: every reading is taken from the bench's displays in the camera images (the controller screen,
the ohmmeter, the thermometer, the indicator, the steel rule); the UP/DN caps and the ohmmeter's RANGE/AUTO keys are
pressed with the arm. Every get_observation is evidence of the state it shows: it returns evidence_id
<episode>:obs:<n> (kind bench_view). Its private record keeps what each display showed at that moment, where the probe
stood and the true temperatures (thermo_bench.observation_record), for the readings gates
(benchmark/evaluation/thermal.py) and the process rule; the three standard views are sealed with it.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

THERMO_TOOLS = ()


def thermo_configuration(instance):
    """Plain values of one instance: the thermal parameters (true values) and what the apparatus shows."""
    from instance_episodes import appearance_seed
    h, params = instance['hidden'], instance['scene_params']
    kind = params['kind']
    thermal = {'kind': kind, 't_room_c': float(h['t_room_C']), 'tau_probe_s': float(h['tau_probe_s']),
               'tau_cool_s': float(h['tau_cool_s'])}
    if kind == 'thermistor':
        thermal |= {'r25_ohm': float(h['R25_ohm']), 'beta_k': float(h['B_K']), 'delta_w_k': float(h['delta_mW_K']) / 1000.0}
        bench = {'kind': kind, 'probe_label': 'THERMISTOR PROBE'}
    else:
        thermal |= {'tau_rod_s': float(h['tau_rod_s']), 'alpha_per_k': float(h['alpha_per_K']),
                    'rod_l0_mm': float(h['rod_L0_mm']), 'indicator_zero_mm': float(h['indicator_zero_mm'])}
        bench = {'kind': kind, 'probe_label': 'THERMOMETER PROBE', 'rod_l0_mm': float(h['rod_L0_mm'])}
    return {'thermal': thermal, 'bench': bench, 'probe_start': 'stand',
            'noise_seed': appearance_seed('thermo-noise:' + instance['instance_id'])}


def thermo_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    from device_runtime import DeviceRuntime
    from instance_episodes import InstanceMixin
    from thermo_bench import ThermoBench

    config = backend.configuration

    class ThermoEpisode(InstanceMixin, DeviceRuntime):
        records = None

        def public_task(self):
            return self.instance_public_task({})

        def robot_state(self):
            return super().robot_state() | {'gripper_holding': bool(self.backend.robot_state()['gripper_holding'])}

        def observation(self):
            """The three standard views with an evidence_id bound to this physical state."""
            value = super().observation()
            if self.records is None:
                self.records = {}
            eid = f'{self.episode_id}:obs:{len(self.records) + 1}'
            number = eid.rsplit(':', 1)[-1]
            directory = backend.private / 'evidence'
            directory.mkdir(exist_ok=True)
            images = {}
            for frame in backend.last_frames:
                name = f"{number}_{frame['camera']}.png"
                (directory / name).write_bytes(frame['rgb_png'])
                images[frame['camera']] = {'file': f'evidence/{name}', 'sha256': hashlib.sha256(frame['rgb_png']).hexdigest()}
            record = backend.bench.observation_record(self.tick)
            record.update(kind='bench_view', evidence_id=eid, held=backend.held, images=images)
            self.records[eid] = record
            (directory / f'{number}.json').write_text(json.dumps(record))       # private (development checks)
            value['evidence_id'] = eid
            return value

        def seal_payload(self):
            """Terminal state, device, evidence and files of the sealed record (also used at stop)."""
            bench = backend.bench
            records = deepcopy(self.records or {})
            directory = backend.private / 'evidence'
            files = [(v['file'], (directory / v['file'].rsplit('/', 1)[-1]).read_bytes())
                     for r in records.values() for v in r['images'].values()]
            device = {'thermal': deepcopy(config['thermal']), 'transitions': deepcopy(bench.transitions),
                      'actions': deepcopy(bench.actions), 'display_hz': 1, 'rows': len(bench.rows)}
            return (backend.seat_audit() | bench.state(), device, records, files)

        def process_extra(self):
            return {'observations': len(self.records or {})}

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if not isinstance(args, dict):
                raise ValueError('arguments must be an object')
            if name == 'get_task_spec' and args == {}:
                return self.public_task()
            if name == 'submit_result':
                return self.freeze_and_seal(args, *self.seal_payload())
            return super()._dispatch(name, args)

        def evaluation(self):
            return self.instance_evaluation()

    backend.set_gripper(.08)
    for _ in range(40 * (physics_hz // 30)):
        backend.step(None)
        backend.after_step({}, backend.tick_index)
    if not backend.at_rest():
        raise RuntimeError('dry-block bench failed to settle')
    backend.tick_index = 0
    backend.history.clear()
    episode = ThermoEpisode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    backend.bench = ThermoBench(episode.episode_id, config, physics_hz=physics_hz, noise_seed=config['noise_seed'])
    return episode

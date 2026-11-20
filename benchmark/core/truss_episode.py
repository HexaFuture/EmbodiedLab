"""Truss bench episodes: configuration of one instance and the device runtime.

The tester is read on its display in the camera images; there is no bench tool. Every get_observation is evidence of
the state it shows: it returns evidence_id <episode>:obs:<n> (kind bench_view); its private record keeps what the
display showed then and the run it belongs to (truss_bench.observation_record).

The Agent loads its own bridge on the loading frame (truss_frame). The loading starts when the first weight
touches the platform: the backend's LoadMonitor freezes the bench then (the seat audit, the bridge record). The first
failure of the bridge (the rope's pull reaching its capacity) ends the episode at that tick: the running command ends
with an error, the episode is sealed with the submission {} and the frozen bench (terminal state), the device record
carries the loading (held load, failure, touches), and the bridge then collapses on camera (post-seal, truss_loading).
submit_result ends the episode without a failure (held load so far; nothing loaded: none).

Each bar's own properties are its kind's (hidden) tensile strength, 100 mm Euler load and squash load times its own
factors, uniform within +-SCATTER (independent for the three), drawn from the instance's appearance seed. The four
loose pins start in the pin stand.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np

TRUSS_TOOLS = ()
SCATTER = 0.0075                  # +-0.75 % (the recipe's public bar_scatter_pct)
SETTLE_MAX_FRAMES = 300


def truss_configuration(instance):
    from instance_episodes import appearance_seed
    import truss_bench as B
    h, params = instance['hidden'], instance['scene_params']
    rng = np.random.default_rng(appearance_seed('truss-scatter:' + instance['instance_id']))
    bars = {}
    for bar, spec in B.initial_bars().items():
        kind = spec['kind']
        bars[bar] = dict(spec, T_N=float(h[f'T_{kind}_N']) * float(1 + rng.uniform(-SCATTER, SCATTER)),
                         P100_N=float(h[f'P100_{kind}_N']) * float(1 + rng.uniform(-SCATTER, SCATTER)),
                         N_N=float(h[f'N_{kind}_N']) * float(1 + rng.uniform(-SCATTER, SCATTER)))
    bench = {'bars': bars, 'pins': B.initial_pins(), 'carriage_label': int(params['carriage_label']),
             'mode': str(params['mode'])}
    return {'bench': bench, 'noise_seed': appearance_seed('truss-noise:' + instance['instance_id'])}


def truss_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    from device_runtime import DeviceRuntime
    from instance_episodes import InstanceMixin
    from truss_bench import TrussBench, bridge_capacity

    config = backend.configuration

    class TrussEpisode(InstanceMixin, DeviceRuntime):
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
            value['evidence_id'] = eid
            return value

        def seal_payload(self):
            bench, frame = backend.bench, backend.frame
            records = deepcopy(self.records or {})
            directory = backend.private / 'evidence'
            files = [(v['file'], (directory / v['file'].rsplit('/', 1)[-1]).read_bytes())
                     for r in records.values() for v in r['images'].values()]
            if frame.started:                  # the bench as it stood when the loading started
                audit = deepcopy(frame.frozen['audit'])
                bridge = deepcopy(frame.frozen['bridge'])
            else:
                audit = backend.seat_audit()
                where = {b: (w[0], tuple(w[1]) if w[0] == 'bridge' else w[1]) for b, w in audit['where'].items()}
                bridge = bridge_capacity(where, bench.bars, bench.failed)
            device = {'bars': deepcopy(bench.bars), 'failed': dict(bench.failed), 'runs': deepcopy(bench.runs),
                      'actions': deepcopy(bench.actions), 'bridge': bridge, 'loading': frame.record(),
                      'at_seal': backend.seat_audit() if frame.started else None}
            return audit, device, records, files

        def _seal_failure(self):
            """The bridge failed this tick: seal (submission {}), then the collapse on camera."""
            payload = self.seal_payload()
            self.freeze_and_seal({}, *payload)
            frame = backend.frame
            plan = dict(frame.frozen['bridge'], failed=dict(backend.bench.failed))
            backend.collapse(plan, frame.fail_pull)

        def _step(self, target=None, velocity=None):
            try:
                super()._step(target, velocity)
            finally:
                if backend.frame.failed and not self.sealed:
                    self._seal_failure()
                    raise ValueError('the bridge failed: the loading is over and the episode is sealed')

        def _dispatch(self, name, args):
            if backend.frame.failed and not self.sealed:
                self._seal_failure()
            if self.terminated:
                raise ValueError('episode terminated')
            if not isinstance(args, dict):
                raise ValueError('arguments must be an object')
            if name == 'get_task_spec' and args == {}:
                return self.public_task()
            if name == 'submit_result':
                payload = self.seal_payload()
                return self.freeze_and_seal(args, *payload)
            return super()._dispatch(name, args)

        def process_extra(self):
            return {'observations': len(self.records or {})}

        def evaluation(self):
            return self.instance_evaluation()

    backend.set_gripper(.08)
    for frame in range(SETTLE_MAX_FRAMES):
        for _ in range(physics_hz // 30):
            backend.step(None)
            backend.after_step({}, backend.tick_index)
        if frame >= 40 and backend.at_rest():
            break
    else:
        last = list(backend.history)[-1]
        raise RuntimeError(f"truss bench failed to settle: drift {backend.drift_m()} m, max speed {last['max_speed_m_s']} "
                           f"m/s, pressed {last['pressed']}, xyz_mm {last['xyz_mm']}")
    backend.tick_index = 0
    backend.history.clear()
    episode = TrussEpisode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    backend.bench = TrussBench(episode.episode_id, config['bench'], physics_hz=physics_hz, noise_seed=config['noise_seed'])
    return episode

"""Vibration absorber episodes (vibration_absorber): configuration of one instance and the device runtime.

The analyzer is read on its screen in the camera images (PEAK mm, FREQ Hz); export_sweeps() returns every finished
sweep of the episode as the analyzer stores it (evidence_id, status, start and end times, the amplitude-frequency
curve). Evidence:
- every get_observation (<episode>:obs:<n>, kind bench_view): what the screens showed (as drawn, and the analyzer's
  display with the sweep it belongs to), the absorber's configuration and the carrier's position then
  (absorber_bench.observation_record), the three standard views;
- every sweep (<episode>:sweep:<n>, kind absorber_sweep): its public record and the private one (the configuration it ran on,
  the state it started from, who touched the bench if it is void).
The sealed device record carries the hidden values, the configuration on the bench at the seal (the evaluator sweeps
it again from rest), the configuration's transitions and the actions (keys), for the gates of
benchmark/evaluation/absorber.py.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

ABSORBER_TOOLS = ('export_sweeps',)
SETTLE_MAX_FRAMES = 300


def absorber_configuration(instance):
    from instance_episodes import appearance_seed
    import absorber_model as M
    h, params = instance['hidden'], instance['scene_params']
    weights = {w: float(g) for w, g in params['weights_g'].items()}
    published = instance['public_task']['public_constants']['weights_g']
    if sorted(weights.values()) != sorted(float(g) for g in published):
        raise ValueError(f'published weights {published} differ from the bench weights {weights}')
    carrier_mm = int(params['carrier_mm'])
    if carrier_mm not in M.BLADE_L_MM:
        raise ValueError(f'the carrier starts on {carrier_mm} mm, not a detent')
    bench = {'hidden': {k: float(h[k]) for k in M.HIDDEN}, 'weights_g': weights,
             'masses_kg': {w: g / 1000.0 for w, g in weights.items()},
             'weight_labels': {w: f'{g:g} g' for w, g in weights.items()},
             'weight_seats': dict(params['weight_seats']), 'carrier_mm': carrier_mm,
             'valve': int(params.get('valve', 0)), 'gap_m': M.GAP_M * float(h['gap_factor'])}
    return {'bench': bench, 'noise_seed': appearance_seed('absorber-noise:' + instance['instance_id'])}


def absorber_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    from absorber_bench import AbsorberBench
    from device_runtime import DeviceRuntime
    from instance_episodes import InstanceMixin, cited_ids

    config = backend.configuration

    class AbsorberEpisode(InstanceMixin, DeviceRuntime):
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
            record.update(kind='bench_view', evidence_id=eid, held=backend.held, images=images,
                          drawn=backend.drawn_displays(), tray=dict(backend.tray))
            self.records[eid] = record
            value['evidence_id'] = eid
            return value

        def seal_payload(self, cited=()):
            """Terminal state, device, evidence and files of the sealed record (also used at stop)."""
            bench = backend.bench
            evidence = deepcopy(self.records or {})
            directory = backend.private / 'evidence'
            files = [(v['file'], (directory / v['file'].rsplit('/', 1)[-1]).read_bytes())
                     for eid in sorted(set(cited) & set(evidence)) for v in evidence[eid]['images'].values()]
            for public, private in zip(bench.sweeps, bench.sweeps_private):
                row = {k: v for k, v in deepcopy(public).items() if k != 'sha256'}
                evidence[public['evidence_id']] = dict(deepcopy(private), kind='absorber_sweep', public=row,
                                                       sha256=public.get('sha256'))
            audit = backend.seat_audit()
            device = {'model_version': bench_model_version(), 'hidden': deepcopy(config['bench']['hidden']),
                      'weights_g': deepcopy(config['bench']['weights_g']), 'configuration': bench.configuration(),
                      'carrier_mm': round(bench.carrier['s_m'] * 1000, 4), 'valve': bench.valve,
                      'tray': dict(bench.tray), 'running': bench.run is not None,
                      'transitions': deepcopy(bench.transitions), 'actions': deepcopy(bench.actions),
                      'sweeps': [s['evidence_id'] for s in bench.sweeps]}
            return audit | {'running': bench.run is not None}, device, evidence, files

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if not isinstance(args, dict):
                raise ValueError('arguments must be an object')
            if name == 'get_task_spec' and args == {}:
                return self.public_task()
            if name == 'export_sweeps' and args == {}:
                return backend.bench.export_sweeps()
            if name in ABSORBER_TOOLS:
                raise ValueError('unexpected arguments')
            if name == 'submit_result':
                return self.freeze_and_seal(args, *self.seal_payload(cited_ids(args)))
            return super()._dispatch(name, args)

        def process_extra(self):
            return {'observations': len(self.records or {}), 'sweeps': len(backend.bench.sweeps)}

        def evaluation(self):
            return self.instance_evaluation()

    def bench_model_version():
        import absorber_model
        return absorber_model.MODEL_VERSION

    # Apparatus preparation (not charged): open the gripper and let the weights settle in the rack.
    backend.set_gripper(.08)
    for frame in range(SETTLE_MAX_FRAMES):
        for _ in range(physics_hz // 30):
            backend.step(None)
            backend.after_step({}, backend.tick_index)
            if frame < 2 and __import__('os').environ.get('EMBODIEDLAB_ABSORBER_TRACE', '0') == '1':
                print(f"[absorber-settle] step s_mm {backend.carrier['s_m'] * 1000:.3f} holds {backend.holds()}",
                      file=__import__('sys').stderr, flush=True)
        if __import__('os').environ.get('EMBODIEDLAB_ABSORBER_TRACE', '0') == '1':
            print(f"[absorber-settle] frame {frame} s_mm {backend.carrier['s_m'] * 1000:.3f} "
                  f"v_mm_s {backend.carrier['v_m_s'] * 1000:.2f} holds {backend.holds()}", file=__import__('sys').stderr,
                  flush=True)
        if frame >= 40 and backend.at_rest():
            break
    else:
        last = list(backend.history)[-1]
        raise RuntimeError(f"absorber bench failed to settle: pressed {last['pressed']} xyz_mm {last['xyz_mm']} "
                           f"carrier {backend.carrier}")
    backend._mark(f'settled after {frame + 1} frames')
    audit = backend.seat_audit()
    expected = dict(config['bench']['weight_seats'])
    if audit['weight_seat'] != expected:
        raise RuntimeError(f"weights not seated in the rack: {audit['weight_seat']} (expected {expected})")
    backend.tick_index = 0
    backend.history.clear()
    backend.carrier = None
    episode = AbsorberEpisode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    bench = config['bench']
    backend.bench = AbsorberBench(episode.episode_id, {'hidden': bench['hidden'], 'weights_g': bench['weights_g'],
                                                       'carrier_mm': bench['carrier_mm'], 'valve': bench['valve']},
                                  physics_hz=physics_hz)
    backend.private.joinpath('absorber_configuration.json').write_text(json.dumps(bench, indent=1))
    return episode

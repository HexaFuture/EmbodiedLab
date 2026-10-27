"""Beam bench episodes: configuration of one instance and the device runtime.

The indicator is read on its LCD in the camera images; there is no bench tool. Every get_observation is evidence of the
state it shows: it returns evidence_id <episode>:obs:<n> (kind bench_view). Its private record keeps what the LCD showed
at that moment, where the carriage stood and the load state (beam_bench.observation_record), for the readings gate
(benchmark/evaluation/beam.py) and the process rule; the three standard views are sealed with it. Parts are grasped, placed
and pressed with the arm; the carriage is pushed by hand.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy

BEAM_TOOLS = ()
LOAD_MODEL = 'contact'          # beam_backend: the strip is loaded by the contact normal forces on it
STRIP_LENGTH_M = 0.415             # constructor knowledge, not public (beam_apparatus.BEAM[0])
SETTLE_MAX_FRAMES = 300            # 10 s at 30 frames/s; the bench is ready once at rest (after >= 40 frames)


def beam_configuration(instance):
    from instance_episodes import appearance_seed
    h, params = instance['hidden'], instance['scene_params']
    # Stations and masses by object id come from the private scene parameters; the agent sees the stations printed on
    # the bench and the masses as an unnamed list (public_constants.weights_g), which must agree.
    weights = params['weights_g']
    published = instance['public_task']['public_constants']['weights_g']
    if sorted(float(g) for g in weights.values()) != sorted(float(g) for g in published):
        raise ValueError(f'published weights {published} differ from the bench weights {weights}')
    holes = [int(mm) for mm in params['holes_mm']]
    if int(params['carriage_hole_mm']) not in holes:
        raise ValueError(f"the carriage starts on hole {params['carriage_hole_mm']}, not one of {holes}")
    bench = {'EI': float(h['EI_Nm2']), 'k_theta': float(h['k_theta_Nm_per_rad']),
             'length_m': STRIP_LENGTH_M,
             'seats_m': {k: float(v) / 1000 for k, v in params['seats_mm'].items()},
             'holes_mm': holes, 'carriage_hole_mm': int(params['carriage_hole_mm']),
             'masses_kg': {w: float(g) / 1000 for w, g in weights.items()},
             'weight_labels': {w: f'{g:g} g' for w, g in weights.items()},
             'weight_seats': dict(params['weight_seats']),
             'gauge_zero_mm': -float(h['dial_offset_mm']),
             'profile': {'tilt_mm_per_m': float(h['rail_tilt_mm_per_m']), 'bow_mm': float(h['strip_bow_mm'])},
             'load_model': LOAD_MODEL}
    return {'bench': bench, 'noise_seed': appearance_seed('beam-noise:' + instance['instance_id'])}


def beam_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    from beam_bench import DISPLAY_HZ, BeamBench
    from device_runtime import DeviceRuntime
    from instance_episodes import InstanceMixin

    config = backend.configuration

    class BeamEpisode(InstanceMixin, DeviceRuntime):
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
            """Terminal state, device, evidence and files of the sealed record (also used at stop)."""
            bench = backend.bench
            records = deepcopy(self.records or {})
            directory = backend.private / 'evidence'
            files = [(v['file'], (directory / v['file'].rsplit('/', 1)[-1]).read_bytes())
                     for r in records.values() for v in r['images'].values()]
            device = {'EI': bench.beam.EI, 'k_theta': bench.beam.k_theta, 'seats_m': bench.seat_x, 'holes_m': bench.holes,
                      'masses_kg': bench.masses, 'profile': bench.profile, 'transitions': deepcopy(bench.transitions),
                      'actions': deepcopy(bench.actions), 'display_hz': DISPLAY_HZ, 'rows': len(bench.rows),
                      'load_model': backend.load_model}
            return backend.seat_audit(), device, records, files

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
        raise RuntimeError(f"beam bench failed to settle: pressed {last['pressed']} xyz_mm {last['xyz_mm']}")
    backend.tick_index = 0
    backend.history.clear()
    episode = BeamEpisode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    asset = backend.asset
    backend.bench = BeamBench(episode.episode_id, config['bench'], physics_hz=physics_hz, noise_seed=config['noise_seed'],
                              segment_x=[asset.segment_x(i) for i in range(asset.SEGMENTS)])
    return episode

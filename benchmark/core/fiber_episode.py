"""Fiber coupling episodes (fiber_coupling): configuration of one instance and the device runtime.

No instrument tools: the power meter is read on its LCD in the camera images. Evidence: every get_observation
(<episode>:obs:<n>, kind bench_view) is bound to the physical state at that moment: the private record keeps the
knob angles, the screws, the mirror tilts and focus, the seats, the light and the meter as shown then
(fiber_bench.record), and the three standard views. The sealed device record carries the hidden instance (model seed),
the bench's final state, its transitions and actions, for the gates of benchmark/evaluation/fiber.py.
"""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy

SETTLE_MAX_FRAMES = 300
DEFAULT_SEATS = {'lens_seats': {f'lens_{k + 1}': f'box:{k + 1}' for k in range(5)}, 'cap_seat': 'coupler:cap',
                 'target_seat': 'box:target'}


def fiber_configuration(instance):
    """Plain values of one instance: the model seed (hidden) and where each holder starts (read before Kit starts:
    no apparatus import here; the episode adds the apparatus' geometry)."""
    from instance_episodes import appearance_seed
    h, params = instance.get('hidden') or {}, instance.get('scene_params') or {}
    seed = h.get('model_seed')
    if seed is None:                       # probe on another experiment's instance (dev only): default seats
        seed, params = int(os.environ.get('EMBODIEDLAB_FIBER_SEED', '3')), {}
    seats = {k: deepcopy(params.get(k, v)) for k, v in DEFAULT_SEATS.items()}
    return {'bench': {'model_seed': int(seed), **seats},
            'noise_seed': appearance_seed('fiber-noise:' + instance['instance_id'])}


def bench_geometry(A):
    """The bench's geometry from the apparatus: cartridges' lenses and offsets, the cap, target and fiber positions."""
    return {'cartridges': {oid: k for k, oid in enumerate(A.LENS_IDS)},
            'lens_offsets_m': [A.lens_disc_y(k) - A.LENS_SEAT_Y for k in range(len(A.LENS_IDS))],
            'cap_id': A.CAP_ID, 'target_id': A.TARGET_ID, 'p_cap': A.P_CAP, 'p_target': A.P_TARGET,
            'p_fiber': A.P_FIBER, 'hole_r': A.HOLE_R}


def fiber_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    from device_runtime import DeviceRuntime
    from fiber_bench import FiberBench
    from instance_episodes import InstanceMixin, cited_ids
    import fiber_model

    config = backend.configuration
    LENS_TEXT = backend.asset.LENS_TEXT

    class FiberEpisode(InstanceMixin, DeviceRuntime):
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
                images[frame['camera']] = {'file': f'evidence/{name}',
                                           'sha256': hashlib.sha256(frame['rgb_png']).hexdigest()}
            record = backend.bench.record(self.tick)
            record.update(kind='bench_view', evidence_id=eid, held=backend.held, images=images,
                          drawn_meter=backend.drawn_meter())
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
            audit = backend.seat_audit()
            device = {'model_version': fiber_model.MODEL_VERSION, 'model_seed': bench.model_seed,
                      'final': bench.snapshot(), 'transitions': deepcopy(bench.transitions),
                      'actions': deepcopy(bench.actions), 'cartridges': dict(bench.cartridges),
                      'lens_offsets_m': list(bench.lens_offset), 'lens_text': list(LENS_TEXT)}
            return audit, device, evidence, files

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if not isinstance(args, dict):
                raise ValueError('arguments must be an object')
            if name == 'get_task_spec' and args == {}:
                return self.public_task()
            if name == 'submit_result':          # the submission cites its observation as power_evidence
                cited = cited_ids(args) | {v for k, v in args.items() if k == 'power_evidence' and isinstance(v, str)}
                return self.freeze_and_seal(args, *self.seal_payload(cited))
            return super()._dispatch(name, args)

        def process_extra(self):
            return {'observations': len(self.records or {})}

        def evaluation(self):
            return self.instance_evaluation()

    # Apparatus preparation (not charged): open the gripper and let the holders settle in their seats.
    backend.set_gripper(.08)
    for frame in range(SETTLE_MAX_FRAMES):
        for _ in range(physics_hz // 30):
            backend.step(None)
            backend.after_step({}, backend.tick_index)
        if frame >= 40 and backend.at_rest():
            break
    else:
        last = list(backend.history)[-1]
        import numpy as np
        speeds = np.linalg.norm(np.asarray(backend.cuvettes.get_linear_velocities(), float), axis=1)
        spins = np.linalg.norm(np.asarray(backend.cuvettes.get_angular_velocities(), float), axis=1)
        moving = {i: (round(float(v), 4), round(float(w), 3)) for i, v, w in zip(backend.ids, speeds, spins) if v > 0.002}
        raise RuntimeError(f"fiber bench failed to settle: pressed {last['pressed']} moving (m/s, rad/s) {moving} "
                           f"xyz_mm {last['xyz_mm']}")
    import faulthandler
    faulthandler.cancel_dump_traceback_later()
    bench = config['bench'] | bench_geometry(backend.asset)
    audit = backend.seat_audit()
    expected = {oid: seat for oid, seat in bench['lens_seats'].items()}
    expected[bench['cap_id']] = bench['cap_seat']
    expected[bench['target_id']] = bench['target_seat']
    if audit['holder_seat'] != expected:
        raise RuntimeError(f"holders not seated: {audit['holder_seat']} (expected {expected})")
    drift = backend.knob_reader.drift_mm if backend.knob_reader is not None else {}
    if any(v > 0.05 for v in drift.values()):
        raise RuntimeError(f'knobs moved off their joints while settling: {drift}')
    backend.tick_index = 0
    backend.history.clear()
    backend.knob_reader = None              # the knobs' angles count from the episode's start
    backend.blocked = None
    episode = FiberEpisode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    backend.bench = FiberBench(episode.episode_id, {k: bench[k] for k in (
        'model_seed', 'cartridges', 'lens_offsets_m', 'cap_id', 'target_id', 'p_cap', 'p_target', 'p_fiber', 'hole_r')}
        | {'noise_seed': config['noise_seed']}, physics_hz=physics_hz)
    backend._redraw()                       # the meter, lamp and beam as the bench starts (the display's segments
                                            # otherwise show their authored all-lit state until the first tick)
    backend.private.joinpath('fiber_configuration.json').write_text(json.dumps(bench, indent=1))
    return episode

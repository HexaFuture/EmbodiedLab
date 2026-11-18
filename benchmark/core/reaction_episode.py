"""Reaction bench episodes (reaction_yield): configuration of one instance and the device runtime.

The kinetics bench's evidence is kept: every observation (<episode>:obs:<n>, the screen as drawn), every READ
(<episode>:spectro:<n>) and every KIN run (<episode>:kinetics:<run>). The block's data logger keeps, while its LOG key
has it on, one row a second (t_s, SET, BLOCK display), exported with export_block_log; the sealed device record adds the
bench's private chemistry (each cuvette's compartments, its temperature history, the block's true temperature) from
which the evaluator takes the product cuvette's yield.

Lines: S STOCK (internal 'stock'), BUFFER ('water'), R STOCK ('unknown'), QUENCH ('quench', the fourth line of
reaction_apparatus). The P standard comes as a filled cuvette standing in rack well PSTD_WELL.
"""
from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy

import reaction_model as RM

REACTION_TOOLS = ('export_spectro_kinetics', 'export_block_log')
PSTD_WELL = 9
PSTD_ML = 3.0
KINETIC_KEYS = tuple(k for k in RM.HIDDEN)


def reaction_configuration(instance):
    from instance_episodes import appearance_seed
    h, params = instance['hidden'], instance['scene_params']
    rng = random.Random(appearance_seed(instance['instance_id']) ^ 0x7E4C)
    sources = {'stock': {'composition_M': {RM.S: float(h['c_S_uM']) * 1e-6}, 'volume_mL': float(params['s_mL'])},
               'water': {'composition_M': {}, 'volume_mL': float(params['buffer_mL'])},
               'unknown': {'composition_M': {RM.RG: float(h['c_R_uM']) * 1e-6}, 'volume_mL': float(params['r_mL'])},
               'quench': {'composition_M': {RM.K: float(h['c_K_mM']) * 1e-3}, 'volume_mL': float(params['quench_mL'])}}
    bench = {'dyes': {RM.S: float(h['eps_S']), RM.P: float(h['eps_P']), RM.Q: float(h['eps_Q'])}, 'sources': sources,
             'dispense_mL': [1.00, 0.10], 'cuvettes': int(params['n_cuvettes']), 'b0': float(h['b0']), 'b1': float(h['b1']),
             'start_nm': rng.randrange(400, 701), 'dispense_seats': {name: 'station' for name in sources},
             'mixer_seat': 'station', 'spectro_seat': 'spectro', 'extra_key': 'kin',
             'labels': {'stock': 'S STOCK', 'water': 'BUFFER', 'unknown': 'R STOCK', 'quench': 'QUENCH'},
             'prefilled': {PSTD_WELL: {'label': 'P STD', 'volume_mL': PSTD_ML,
                                       'composition_M': {RM.P: float(h['c_P_uM']) * 1e-6}}},
             'kinetics': {k: float(h[k]) for k in KINETIC_KEYS}, 'stray': float(h['stray_light'])}
    return {'bench': bench, 'noise_seed': appearance_seed('chem-noise:' + instance['instance_id'])}


def reaction_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    from device_runtime import DeviceRuntime
    from instance_episodes import InstanceMixin, cited_ids, screen_view
    from reaction_bench import ReactionBench

    config = backend.configuration

    class ReactionEpisode(InstanceMixin, DeviceRuntime):
        records = None

        def public_task(self):
            return self.instance_public_task({})

        def robot_state(self):
            return super().robot_state() | {'gripper_holding': bool(self.backend.robot_state()['gripper_holding'])}

        def observation(self):
            """The three standard views with an evidence_id bound to this state (kind bench_view)."""
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
            self.records[eid] = screen_view(backend, self.tick, physics_hz) | {
                'kind': 'bench_view', 'evidence_id': eid, 'images': images, 'block': backend.bench.block_display()}
            value['evidence_id'] = eid
            return value

        def process_extra(self):
            return {'observations': len(self.records or {})}

        def seal_payload(self, cited=()):
            bench = backend.bench
            evidence = deepcopy(self.records or {})
            directory = backend.private / 'evidence'
            files = [(v['file'], (directory / v['file'].rsplit('/', 1)[-1]).read_bytes())
                     for eid in sorted(set(cited) & set(evidence)) for v in evidence[eid]['images'].values()]
            for row, private in zip(bench.spectro.rows, bench.spectro.private):
                eid = f"{self.episode_id}:spectro:{row['sample_id']}"
                public = deepcopy(row)
                evidence[eid] = dict(deepcopy(private), kind='spectro_read', evidence_id=eid, public=public,
                                     sha256=hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest())
            for run, private in zip(bench.spectro.kinetic_runs, bench.spectro.kinetic_private):
                eid = f"{self.episode_id}:kinetics:{run['run']}"
                public = deepcopy(run)
                evidence[eid] = {'kind': 'kinetic_run', 'evidence_id': eid, 'public': public,
                                 'private_points': deepcopy(private['points']),
                                 'sha256': hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()}
            h = instance['hidden']
            sources = config['bench']['sources']
            device = {'c_label_uM': {'S': float(h['c_S_uM']), 'R': float(h['c_R_uM']), 'P': float(h['c_P_uM']),
                                     'K': float(h['c_K_mM']) * 1e3},
                      'source_of': {'S': 'stock', 'R': 'unknown', 'buffer': 'water', 'quench': 'quench'},
                      'p_standard': {'cuvette': PSTD_WELL, 'volume_mL': PSTD_ML},
                      'sources': {n: {'volume_mL': s['volume_mL']} for n, s in sources.items()},
                      'dispense_mL': list(config['bench']['dispense_mL']), 'min_read_mL': 2.0, 'capacity_mL': 3.5,
                      'screen_readings': True, 'block_log': bench.export_block(),
                      'chemistry': bench.private_record(), 'actions': deepcopy(bench.actions)}
            return (backend.seat_audit() | {'busy': bench.spectro.busy is not None, 'block_logging': bench.logging,
                                            'block_set_C': bench.block.set_c, 'room_C': RM.ROOM_C},
                    device, evidence, files)

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if not isinstance(args, dict):
                raise ValueError('arguments must be an object')
            bench = backend.bench
            if name == 'get_task_spec' and args == {}:
                return self.public_task()
            if name == 'export_spectro_kinetics' and args == {}:
                return bench.spectro.export_kinetics()
            if name == 'export_block_log' and args == {}:
                return bench.export_block()
            if name in REACTION_TOOLS:
                raise ValueError('unexpected arguments')
            if name == 'submit_result':
                return self.freeze_and_seal(args, *self.seal_payload(cited_ids(args)))
            return super()._dispatch(name, args)

        def evaluation(self):
            return self.instance_evaluation()

    # Apparatus preparation (not charged): open the gripper and let the cuvettes settle in the rack.
    backend.set_gripper(.08)
    for _ in range(40 * (physics_hz // 30)):
        backend.step(None)
        backend.after_step({}, backend.tick_index)
    if not backend.at_rest():
        raise RuntimeError('reaction bench failed to settle')
    audit = backend.seat_audit()
    if sorted(v for v in audit['cuvette_seat'].values() if v) != sorted(f'rack:{i}' for i in audit['cuvette_seat']):
        raise RuntimeError(f"cuvettes not seated in the rack: {audit['cuvette_seat']}")
    backend.tick_index = 0
    backend.history.clear()
    episode = ReactionEpisode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    backend.bench = ReactionBench(episode.episode_id, config['bench'], physics_hz=physics_hz, noise_seed=config['noise_seed'])
    return episode

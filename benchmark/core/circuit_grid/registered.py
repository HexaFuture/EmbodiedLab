"""Registered circuit-board episode (resistor_blackbox): instance values, observation evidence, the logger record
export, submission freeze and sealing.

Device interface: the standard observation and motion tools, set_gripper, wait, export_logger(evidence_id) and
submit_result. Everything on the bench is operated by hand: blocks are picked from the tray and placed on the board,
the supply's ON/OFF button and the logger keys are pressed, the switch rocker is pushed. Readings are taken from the
displays in the camera images.

Evidence: every get_observation is a capture `<episode_id>:obs:<n>` of that physical state: its public record (tick,
SHA-256 of the three view images) is hash-sealed with the bench's private state at that tick (placements, supply,
true and shown value of every instrument, the time of the last change). Logger records are `<episode_id>:logger:<n>`
(CSV, hash-sealed) with the private state snapshots in force while they ran. The images of the captures cited in the
submission are sealed with it.
"""
from copy import deepcopy
import hashlib
import json

from device_runtime import DeviceRuntime
from instance_episodes import InstanceMixin, appearance_seed

from . import layout as L
from .board import GridBench, I_MIN_A, STEADY_S
from .instruments import (AMP_RANGES, CONVERSION_S, LOGGER_INPUT_OHM, LOGGER_NOISE_V, LOGGER_RANGE_V,
                          LOGGER_RESOLUTION_V, VOLT_RANGES, VOLTMETER_OHM)

TOOLS = ('get_task_spec', 'get_observation', 'get_robot_state', 'execute_joint_trajectory', 'set_gripper', 'wait',
         'export_logger', 'submit_result')
SUPPLY_V_NOMINAL, SUPPLY_LIMIT_A = 5.0, 0.10


def canonical_sha256(value):
    # Same canonical form as benchmark/evaluation/sealed.py.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def grid_configuration(instance):
    """Hidden element values of one instance (box resistor, actual R_s values, supply set point) and the noise seed."""
    h = instance['hidden']
    return {'r_box_ohm': float(h['R_ohm']),
            'rs_actual_ohm': {'R1k': 1000.0 * float(h['rs_factor_1k']), 'R10k': 10000.0 * float(h['rs_factor_10k']),
                              'R100k': 100000.0 * float(h['rs_factor_100k'])},
            'supply_v': SUPPLY_V_NOMINAL * float(h.get('supply_factor', 1.0)),
            'supply_limit_a': SUPPLY_LIMIT_A,
            'noise_seed': appearance_seed('grid-noise:' + instance['instance_id'])}


def public_record(record):
    """A finished logger record as exported (the CSV carries the rows; no summary counts, as a real logger)."""
    return {k: deepcopy(v) for k, v in record.items() if k not in ('rows', 'status', 'status_counts')}


def grid_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    config = backend.configuration

    class GridEpisode(InstanceMixin, DeviceRuntime):
        def public_task(self):
            return self.instance_public_task({'apparatus': L.public_apparatus()})

        def robot_state(self):
            return super().robot_state() | {'gripper_holding': backend.held is not None}

        def observation(self):
            value = super().observation()
            bench = backend.bench
            n = len(self.captures) + 1
            eid = f'{self.episode_id}:obs:{n}'
            views = backend.view_pngs
            images = {camera: hashlib.sha256(png).hexdigest() for camera, png in views.items()}
            public = {'evidence_id': eid, 'physics_tick': self.tick, 'sim_time_s': self.tick / physics_hz,
                      'images_sha256': images}
            directory = backend.private / 'evidence'
            directory.mkdir(exist_ok=True)
            for camera, png in views.items():
                (directory / f'obs_{n}_{camera}.png').write_bytes(png)
            self.captures[eid] = {
                'kind': 'screen_capture', 'public': public, 'sha256': canonical_sha256(public),
                'state': bench.observation_state(self.tick),
                'images': {camera: {'file': f'evidence/obs_{n}_{camera}.png', 'sha256': digest}
                           for camera, digest in images.items()}}
            value['evidence_id'] = eid
            return value

        def terminal_state(self):
            bench = backend.bench
            placement = backend.placement()
            seats = {}
            for block, place in placement.items():
                if place[0] == 'tray' and place[2]:
                    seats.setdefault(place[1], []).append(block)
            return {'supply_on': bool(backend.supply_on), 'switch_closed': bool(backend.switch_closed),
                    'logger_recording': bench.logger.active is not None,
                    'held': backend.held or backend.releasing,
                    'placement': {b: (('board', L.node_name(p[1]), L.node_name(p[2])) if p[0] == 'board' else
                                      ('tray', p[1], bool(p[2])) if p[0] == 'tray' else ('loose',))
                                  for b, p in placement.items()},
                    'tray_seats': seats,
                    'all_in_tray': all(p[0] == 'tray' and p[2] for p in placement.values())}

        def seal_payload(self, submission=None):
            """Terminal state, device, evidence and files of the sealed record (also used at stop)."""
            bench = backend.bench
            logger = bench.logger
            evidence = {eid: dict(public_record(record), kind='circuit_logger', **deepcopy(logger.private[eid]),
                                  states=bench.record_states(logger.private[eid]['start_tick'],
                                                             logger.private[eid]['end_tick']))
                        for eid, record in logger.records.items()}
            evidence.update(deepcopy(self.captures))
            cited = set()
            if isinstance(submission, dict) and isinstance(submission.get('measurements'), list):
                cited = {m.get('evidence_id') for m in submission['measurements'] if isinstance(m, dict)}
            directory = backend.private / 'evidence'
            files = [(image['file'], (directory / image['file'].rsplit('/', 1)[-1]).read_bytes())
                     for eid, record in self.captures.items() if eid in cited for image in record['images'].values()]
            device = {'supply_v_nominal': SUPPLY_V_NOMINAL, 'supply_limit_a': SUPPLY_LIMIT_A,
                      'voltmeter_input_ohm': VOLTMETER_OHM, 'logger_input_ohm': LOGGER_INPUT_OHM,
                      'voltmeter_ranges': [list(r[:4]) for r in VOLT_RANGES],
                      'ammeter_ranges': [list(r) for r in AMP_RANGES], 'conversion_s': CONVERSION_S,
                      'logger': {'range_v': LOGGER_RANGE_V, 'resolution_v': LOGGER_RESOLUTION_V,
                                 'noise_v': LOGGER_NOISE_V},
                      'rs_nominal_ohm': dict(L.RS_NOMINAL), 'i_min_a': I_MIN_A, 'steady_s': STEADY_S,
                      'physics_hz': physics_hz}
            return self.terminal_state(), device, evidence, files

        def process_extra(self):
            bench = backend.bench
            return {'logger_open': bench.logger.open_record(), 'display_log': list(bench.display_log),
                    'logger_presses': list(bench.logger.presses)}

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if name not in TOOLS or not isinstance(args, dict):
                raise ValueError('unknown circuit board tool')
            if name == 'get_task_spec' and args == {}:
                return self.public_task()
            if name == 'export_logger':
                if set(args) != {'evidence_id'}:
                    raise ValueError('export_logger takes evidence_id')
                return public_record(backend.bench.logger.export(args['evidence_id']))
            if name == 'submit_result':
                return self.freeze_and_seal(args, *self.seal_payload(args))
            return super()._dispatch(name, args)

        def evaluation(self):
            return self.instance_evaluation()

    # Apparatus preparation (not charged): open the gripper and let the blocks settle in their tray seats.
    backend.set_gripper(.08)
    for _ in range(40 * (physics_hz // 30)):
        backend.step(None)
        backend.after_step({}, backend.tick_index)
    if not backend.at_rest():
        raise RuntimeError('circuit board blocks failed to settle')
    placement = backend.placement()
    away = [b for b, p in placement.items() if not (p[0] == 'tray' and p[1] == b)]
    if away:
        raise RuntimeError(f'blocks not in their tray seats after set-up: {away}')
    backend.tick_index = 0
    backend.history.clear()
    episode = GridEpisode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    episode.captures = {}
    backend.horizon_ticks = horizon_ticks
    backend.bench = GridBench(episode.episode_id, r_box_ohm=config['r_box_ohm'], rs_actual_ohm=config['rs_actual_ohm'],
                              supply_v=config['supply_v'], supply_limit_a=config['supply_limit_a'],
                              noise_seed=config['noise_seed'], physics_hz=physics_hz)
    backend.bench.update(0, placement, None, backend.supply_on, backend.switch_closed, reason='start')
    return episode


PUBLIC_TOOLS = TOOLS

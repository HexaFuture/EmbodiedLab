"""Optics kit episodes: configuration of one instance and the device runtime with the optics tools.

Keypad bench (optics_rail_bench, slit_rail_bench, plate_rail_bench): read_controller — the stage controller as it
is now (selected axis, entry, ERR, carriage positions to 0.1 mm, moving, the sweep trigger); capture_screen — the
screen image (PNG) with the positions it shows, an instrument record cited as <episode>:screen:<n>; arm_sweep(every_mm)
— captures during the next carriage move; get_capture(capture) — any capture's record and image. None advances time.
Parts are grasped, placed and pressed with the arm.

Hand-slid bench (optics_hand_bench, scene parameter drive: hand): no instrument tools. get_observation returns, with
the three standard views, an evidence_id (<episode>:obs:<n>) bound to the physical state at that moment: the private
record keeps the riders' true index positions, the seats, the source state and what the screen shows (optics_hand).
"""
from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy

OPTICS_TOOLS = ('read_controller', 'capture_screen', 'arm_sweep', 'get_capture')
LENS_COLOURS = {'L1': [0.62, 0.80, 0.92], 'L2': [0.95, 0.70, 0.35]}      # lens tints: pale blue, amber
LENS_APERTURE_MM = 25.0       # constructor knowledge, not public: the default; a scene may set aperture_mm (not public)


def optics_configuration(instance):
    """Plain values of one instance: carriage travel and start, lenses (focal length, aperture, principal-plane
    offset) and where each lens holder starts."""
    from instance_episodes import appearance_seed
    h, params = instance['hidden'], instance['scene_params']
    constants = instance['public_task']['public_constants']
    lenses = {}
    kinds = params.get('element_kinds') or {}
    for lens in params['lenses']:
        if kinds.get(lens, 'lens') == 'plate':
            lenses[lens] = {'kind': 'plate', 'thickness_m': float(constants['plate']['thickness_mm']) / 1000.0,
                            'n': float(h[f'n_{lens}']), 'offset_mm': 0.0}
        elif kinds.get(lens, 'lens') == 'slit':
            lenses[lens] = {'kind': 'slit', 'width_m': float(h['slit_width_mm']) / 1000.0,
                            'offset_mm': float(h[f'offset_{lens}_mm'])}
        elif kinds.get(lens, 'lens') == 'grating':
            lenses[lens] = {'kind': 'grating', 'lines_per_mm': float(constants['grating']['lines_per_mm']),
                            'offset_mm': float(h[f'offset_{lens}_mm'])}
        else:
            lenses[lens] = {'kind': 'lens', 'f_m': float(h[f'f_{lens}_mm']) / 1000.0,
                            'aperture_m': float(params.get('aperture_mm') or (constants.get('lens') or {}).get(
                                'aperture_mm', LENS_APERTURE_MM)) / 2000.0,
                            'offset_mm': float(h[f'offset_{lens}_mm'])}
    bench = {'drive': params.get('drive', 'motor'), 'source': params.get('source', 'object'), 'laser_nm': float(h.get('laser_nm', (constants.get('laser') or {}).get('nm', 0.0))),
             'travel_mm': {k: list(v) for k, v in constants['travel_mm'].items()},
             'initial_mm': {k: float(v) for k, v in params['initial_mm'].items()},
             'lenses': lenses, 'lens_axes': list(params['lens_axes']), 'screen_axis': params['screen_axis'],
             'lens_seats': dict(params['lens_seats']),
             'lens_colours': {k: v for k, v in LENS_COLOURS.items() if k in lenses}}
    return {'bench': bench, 'noise_seed': appearance_seed('optics-noise:' + instance['instance_id'])}


def optics_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    from device_runtime import DeviceRuntime
    from instance_episodes import InstanceMixin
    from optics_bench import OpticsBench

    config = backend.configuration
    if config['bench'].get('drive') == 'hand':
        return optics_hand_episode(backend, instance, output, physics_hz=physics_hz, horizon_ticks=horizon_ticks)

    def _image(episode_id, public, png):
        eid = f"{episode_id}:screen:{public['capture']}"
        return public | {'evidence_id': eid, 'image': {'encoding': 'base64', 'mime_type': 'image/png',
                                                       'sha256': public['image_sha256'],
                                                       'data': base64.b64encode(png).decode('ascii')}}

    class OpticsEpisode(InstanceMixin, DeviceRuntime):
        def public_task(self):
            return self.instance_public_task({})

        def robot_state(self):
            return super().robot_state() | {'gripper_holding': bool(self.backend.robot_state()['gripper_holding'])}

        def seal_payload(self):
            """Terminal state, device, evidence and files of the sealed record (also used at stop)."""
            bench = backend.bench
            evidence, files = {}, []
            for public, png, private in zip(bench.captures, bench.pngs, bench.private):
                eid = f"{self.episode_id}:screen:{public['capture']}"
                rel = f"evidence/screen_{public['capture']:03d}.png"
                files.append((rel, png))
                evidence[eid] = {'kind': 'screen_capture', 'evidence_id': eid, 'public': deepcopy(public),
                                 'image_file': rel, 'private': deepcopy(private),
                                 'sha256': hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()}
            device = {'lenses': deepcopy(bench.lenses), 'source': bench.source, 'laser_nm': bench.laser_m * 1e9,
                      'travel_mm': bench.travel, 'actions': deepcopy(bench.actions), 'state_log': deepcopy(bench.state_log)}
            return (backend.seat_audit() | {'moving': bench.moving()}, device, evidence, files)

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if not isinstance(args, dict):
                raise ValueError('arguments must be an object')
            bench = backend.bench
            if name == 'get_task_spec' and args == {}:
                return self.public_task()
            if name == 'read_controller' and args == {}:
                return bench.display() | {'sim_time_s': self.tick / self.limits.physics_hz}
            if name == 'capture_screen' and args == {}:
                return _image(self.episode_id, *bench.capture())
            if name == 'arm_sweep' and set(args) == {'every_mm'}:
                return bench.arm_sweep(args['every_mm'])
            if name == 'get_capture' and set(args) == {'capture'}:
                return _image(self.episode_id, *bench.get_capture(args['capture']))
            if name in OPTICS_TOOLS:
                raise ValueError('unexpected arguments')
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
        raise RuntimeError('optics bench failed to settle')
    backend.tick_index = 0
    backend.history.clear()
    episode = OpticsEpisode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    backend.bench = OpticsBench(episode.episode_id, config['bench'], physics_hz=physics_hz, noise_seed=config['noise_seed'])
    return episode


def optics_hand_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    from device_runtime import DeviceRuntime
    from instance_episodes import InstanceMixin
    from optics_hand import HandOpticsBench

    config = backend.configuration

    class HandOpticsEpisode(InstanceMixin, DeviceRuntime):
        records = None

        def public_task(self):
            return self.instance_public_task({})

        def robot_state(self):
            return super().robot_state() | {'gripper_holding': bool(self.backend.robot_state()['gripper_holding'])}

        def observation(self):
            """The three standard views with an evidence_id bound to this physical state."""
            bench = backend.bench
            bench.current()
            value = super().observation()
            record = bench.record(self.tick)
            if self.records is None:
                self.records = {}
            eid = f'{self.episode_id}:obs:{len(self.records) + 1}'
            number = eid.rsplit(':', 1)[-1]
            directory = backend.private / 'evidence'
            directory.mkdir(exist_ok=True)
            images = {}
            for frame in backend.last_frames:
                rel = f"evidence/{number}_{frame['camera']}.png"
                (directory / f"{number}_{frame['camera']}.png").write_bytes(frame['rgb_png'])
                images[frame['camera']] = {'file': rel, 'sha256': hashlib.sha256(frame['rgb_png']).hexdigest()}
            record.update(kind='bench_view', evidence_id=eid, holders=backend.seat_audit(), held=backend.held,
                          rider_speed_mm_s=backend.rider_speeds_mm_s(), images=images)
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
            device = {'lenses': deepcopy(bench.lenses), 'source': bench.source, 'laser_nm': bench.laser_m * 1e9,
                      'travel_mm': bench.travel, 'drive': 'hand', 'actions': deepcopy(bench.actions),
                      'state_log': deepcopy(bench.state_log)}
            terminal = backend.seat_audit() | bench.snapshot() | {'rider_speed_mm_s': backend.rider_speeds_mm_s()}
            return terminal, device, records, files

        def process_extra(self):
            return {'observations': len(self.records or {})}

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if not isinstance(args, dict):
                raise ValueError('arguments must be an object')
            if name == 'get_task_spec' and args == {}:
                return self.public_task()
            if name in OPTICS_TOOLS:
                raise ValueError('not a tool of this bench')
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
        raise RuntimeError('optics bench failed to settle')
    backend.tick_index = 0
    backend.history.clear()
    episode = HandOpticsEpisode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    bench_config = dict(config['bench'], initial_mm=backend._read_riders())
    backend.bench = HandOpticsBench(episode.episode_id, bench_config, physics_hz=physics_hz)
    return episode

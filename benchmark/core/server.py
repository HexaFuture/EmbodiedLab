"""Standalone participant-neutral pendulum service. Physics stays on main thread.

Public endpoints expose only the published device contract. Evaluation and
capture finalization require the benchmark host's separate administration key.
No participant SDK, planner or segmentation service is imported or launched.
"""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import random
import shutil
import time

from control_contract import PUBLIC_TOOLS
from instance_episodes import EXPERIMENT_RUNTIMES, finish_process_record, appearance_seed, chem_configuration, titration_configuration, circuit_configuration, coupled_parameters, hooke_dynamics, load_instance, torsion_dynamics

LEGACY_TASKS = ('pendulum_gravity_v1','hooke_identification_v2','coupled_oscillator_modes_v1')

MAX_REQUEST_BYTES = 4 * 1024 * 1024
# A process-record checkpoint (process_checkpoint/, read by the lifecycle adapter's salvage when this process dies
# without finalizing) is written before commands that change the episode, at most this often (wall seconds).
CHECKPOINT_S = 15.0
READ_ONLY_TOOLS = ('get_task_spec', 'get_observation', 'get_robot_state')


def make_handler(episode, admin_key, capture, spectator=None, host_observer=None, checkpoint=None):
    task = episode.public_task()
    allowed_tools = tuple(task.get('tools', PUBLIC_TOOLS))
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            data = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer '+admin_key)

        def do_GET(self):
            if self.path == '/health':
                return self.reply(200, {'ready': True, 'task_id': task.get('task_id', 'pendulum_gravity_v1'),
                    'episode_id': episode.episode_id, 'terminated': episode.terminated})
            if self.path == '/v1/task':
                return self.reply(200, episode.public_task())
            if self.path in ('/v1/host-observation', '/v1/host-contact'):
                if not self.authorized():
                    return self.reply(403, {'error': 'benchmark host credentials required'})
                if host_observer is None:
                    return self.reply(404, {'error': 'no optional host observer'})
                value=(host_observer.snapshot(contact_only=True) if self.path=='/v1/host-contact'
                       else host_observer.snapshot())
                return self.reply(200, value)
            if self.path == '/v1/spectator':
                if not self.authorized():
                    return self.reply(403, {'error': 'benchmark host credentials required'})
                return self.reply(200, spectator.metadata() if spectator else {'enabled': False})
            if self.path == '/v1/evaluation':
                if not self.authorized():
                    return self.reply(403, {'error': 'benchmark host credentials required'})
                return self.reply(200, episode.evaluation())
            return self.reply(404, {'error': 'unknown endpoint'})

        def do_POST(self):
            if self.path == '/v1/finalize':
                if not self.authorized():
                    return self.reply(403, {'error': 'benchmark host credentials required'})
                result = capture()
                episode.terminated = True
                return self.reply(200, result)
            if self.path != '/v1/call':
                return self.reply(404, {'error': 'unknown endpoint'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    return self.reply(413, {'error': 'command exceeds public size limit'})
                value = json.loads(self.rfile.read(length), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('non-JSON number')))
                if not isinstance(value, dict) or set(value) != {'request_id', 'name', 'arguments'}:
                    raise ValueError('expected request_id, name and arguments')
                if value['name'] not in allowed_tools:
                    raise ValueError('not a public benchmark command')
                if checkpoint is not None and value['name'] not in READ_ONLY_TOOLS:
                    checkpoint()
                result = episode.call(value['request_id'], value['name'], value['arguments'])
            except (ValueError, TypeError, KeyError) as exc:
                return self.reply(400, {'ok': False, 'error': str(exc)})
            return self.reply(200, result)

        def log_message(self, format, *args):
            # Never log bodies, credentials or private physical state.
            pass

    return Handler


def main(*, host_observer_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--admin-key-file', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', choices=LEGACY_TASKS+tuple(EXPERIMENT_RUNTIMES), default='pendulum_gravity_v1')
    parser.add_argument('--instance-file', help='private instance record (registry experiments only)')
    parser.add_argument('--physics-hz', type=int, choices=(240, 480), default=240)
    parser.add_argument('--horizon-ticks', type=int, default=144000)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--spectator', action='store_true', help='Enable optional host-only read-only Viser mirror')
    args = parser.parse_args()
    admin_key = Path(args.admin_key_file).read_text().strip()
    if len(admin_key) < 32:
        parser.error('administration key must contain at least 32 characters')
    Path(args.output).mkdir(parents=True, exist_ok=True)
    # Development distribution, recorded privately and repeatable per seed.
    args.gravity = random.Random(args.seed).uniform(8.5, 11.0)
    args.record, args.preview_only, args.cycles = True, False, 10
    instance = None
    if args.task in EXPERIMENT_RUNTIMES:
        if not args.instance_file:
            parser.error('registry experiments require --instance-file')
        # Hidden values come from the instance service; the seed only keys cosmetics.
        instance = load_instance(args.instance_file, args.task, Path(args.output)/'private')
        args.seed = appearance_seed(instance['instance_id'])
        args.layout_candidates = instance.get('layout_candidates')
        # The experiment's own time budget (00 section 6 allows one per task) sets the episode horizon.
        if (instance.get('budget') or {}).get('sim_s'):
            args.horizon_ticks = int(round(float(instance['budget']['sim_s']) * args.physics_hz))
        if instance['runtime'] == 'pendulum':
            args.gravity = float(instance['hidden']['g_m_per_s2'])
        elif instance['runtime'] == 'coupled':
            args.coupled_parameters = coupled_parameters(instance)
        if instance['exp_id'] == 'airtrack_collision':
            args.airtrack_parameters = instance['hidden']
        if instance['exp_id'] == 'hooke_dynamic_k':
            args.hooke_dynamics = hooke_dynamics(instance)
        elif instance['runtime'] == 'rectifier':
            from rectifier.registered import rectifier_configuration
            args.rectifier_configuration = rectifier_configuration(instance)
        elif instance['runtime'] == 'friction':
            from friction.registered import friction_parameters
            args.friction_parameters = friction_parameters(instance)
        elif instance['runtime'] == 'torsion_centroid':
            from centroid_episode import centroid_dynamics
            args.centroid_dynamics = centroid_dynamics(instance)
        elif instance['runtime'] == 'torsion':
            args.torsion_dynamics = torsion_dynamics(instance)
        elif instance['runtime'] == 'circuit_grid':
            from circuit_grid.variants import grid_configuration_for
            args.grid_configuration = grid_configuration_for(instance)
        elif instance['runtime'] == 'circuit':
            args.circuit_configuration = circuit_configuration(instance)
        elif instance['runtime'] == 'optics':
            from optics_episode import optics_configuration
            args.chem_configuration = optics_configuration(instance)
        elif instance['runtime'] == 'calorimetry':
            from calor_episode import calor_configuration
            args.chem_configuration = calor_configuration(instance)
        elif instance['runtime'] == 'thermo':
            from thermo_episode import thermo_configuration
            args.chem_configuration = thermo_configuration(instance)
        elif instance['runtime'] == 'beam':
            from beam_episode import beam_configuration
            args.chem_configuration = beam_configuration(instance)
        elif instance['runtime'] == 'chem':
            titration = instance['recipe'] in ('chem_titration_bench', 'chem_titration_khp_bench')
            args.chem_configuration = (titration_configuration if titration else chem_configuration)(instance)
    elif args.instance_file:
        parser.error('--instance-file is only for registry experiments')
    runtime = instance['runtime'] if instance else {'hooke_identification_v2': 'hooke',
        'coupled_oscillator_modes_v1': 'coupled'}.get(args.task, 'pendulum')
    if args.spectator:
        # Kit prepends its older websockets package during initialization.
        # Resolve the optional viewer's dependencies before Kit changes paths.
        import viser  # noqa: F401
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 640, 'height': 640,
        'extra_args': ['--/app/asyncRendering=false', '--/renderer/multiGpu/enabled=false']})
    from isaac_backend import IsaacBackend
    from episode_runtime import PendulumEpisode
    backend = None
    spectator = None
    try:
        if args.task == 'hooke_identification_v2':
            from hooke_backend import HookeBackend
            from hooke_scoring import HookeEpisode
            from hooke_runtime import initialize_hooke_episode
            score = HookeEpisode(args.seed,args.horizon_ticks//(args.physics_hz//30),task_id=args.task)
            backend = HookeBackend(args,app,score)
            episode = initialize_hooke_episode(backend,score,physics_hz=args.physics_hz,horizon_ticks=args.horizon_ticks)
            scene = backend
        elif runtime == 'hooke':
            from hooke_backend import HookeBackend
            from instance_episodes import hooke_score, hooke_runtime, hooke_dynamic_runtime
            score = hooke_score(instance, args.horizon_ticks//(args.physics_hz//30))
            backend = HookeBackend(args,app,score)
            episode = (hooke_dynamic_runtime if instance['exp_id'] == 'hooke_dynamic_k' else hooke_runtime)(backend,score,instance,args.output,physics_hz=args.physics_hz,horizon_ticks=args.horizon_ticks)
            scene = backend
        elif runtime == 'rectifier':
            from rectifier.backend import RectifierBackend
            from rectifier.registered import rectifier_episode
            backend = RectifierBackend(args, app, args.rectifier_configuration)
            episode = rectifier_episode(backend, instance, args.output, physics_hz=args.physics_hz, horizon_ticks=args.horizon_ticks)
            scene = backend
        elif runtime == 'friction':
            from friction.backend import FrictionBackend
            from friction.registered import friction_episode
            backend = FrictionBackend(args, app, args.friction_parameters)
            episode = friction_episode(backend, instance, args.output, physics_hz=args.physics_hz, horizon_ticks=args.horizon_ticks)
            scene = backend
        elif runtime == 'torsion_centroid':
            from centroid_backend import CentroidBackend
            from centroid_episode import centroid_episode
            backend = CentroidBackend(args, app, args.centroid_dynamics)
            episode = centroid_episode(backend, instance, args.output, physics_hz=args.physics_hz, horizon_ticks=args.horizon_ticks)
            scene = backend
        elif runtime == 'torsion':
            from torsion_backend import TorsionBackend
            from instance_episodes import torsion_episode
            backend = TorsionBackend(args, app, args.torsion_dynamics)
            episode = torsion_episode(backend, instance, args.output, physics_hz=args.physics_hz, horizon_ticks=args.horizon_ticks)
            scene = backend
        elif runtime == 'circuit_grid':
            from circuit_grid.variants import grid_bench_for     # resistor_blackbox or a profile's bench
            backend, episode = grid_bench_for(args, app, instance)
            scene = backend
        elif runtime == 'circuit':
            from circuit_backend import CircuitBackend
            from instance_episodes import circuit_episode
            backend = CircuitBackend(args, app, args.circuit_configuration)
            episode = circuit_episode(backend, instance, args.output, physics_hz=args.physics_hz, horizon_ticks=args.horizon_ticks)
            scene = backend
        elif runtime == 'chem':
            if instance['recipe'] in ('chem_titration_bench', 'chem_titration_khp_bench'):
                from titration_backend import TitrationBackend
                from instance_episodes import titration_episode
                backend = TitrationBackend(args, app, args.chem_configuration)
                episode = titration_episode(backend, instance, args.output, physics_hz=args.physics_hz, horizon_ticks=args.horizon_ticks)
            else:
                from chem_backend import ChemBackend
                from instance_episodes import chem_episode
                backend = ChemBackend(args, app, args.chem_configuration)
                episode = chem_episode(backend, instance, args.output, physics_hz=args.physics_hz, horizon_ticks=args.horizon_ticks)
            scene = backend
        elif runtime == 'optics':
            from optics_backend import OpticsBackend
            from optics_episode import optics_episode
            backend = OpticsBackend(args, app, args.chem_configuration)
            episode = optics_episode(backend, instance, args.output, physics_hz=args.physics_hz, horizon_ticks=args.horizon_ticks)
            scene = backend
        elif runtime == 'calorimetry':
            from calor_backend import CalorBackend
            from calor_episode import calor_episode
            backend = CalorBackend(args, app, args.chem_configuration)
            episode = calor_episode(backend, instance, args.output, physics_hz=args.physics_hz, horizon_ticks=args.horizon_ticks)
            scene = backend
        elif runtime == 'thermo':
            from thermo_backend import ThermoBackend
            from thermo_episode import thermo_episode
            backend = ThermoBackend(args, app, args.chem_configuration)
            episode = thermo_episode(backend, instance, args.output, physics_hz=args.physics_hz, horizon_ticks=args.horizon_ticks)
            scene = backend
        elif runtime == 'beam':
            from beam_backend import BeamBackend
            from beam_episode import beam_episode
            backend = BeamBackend(args, app, args.chem_configuration)
            episode = beam_episode(backend, instance, args.output, physics_hz=args.physics_hz, horizon_ticks=args.horizon_ticks)
            scene = backend
        elif runtime == 'projectile':
            from projectile.backend import ProjectileBackend
            from projectile.episode import ProjectileEpisode
            backend = ProjectileBackend(args, app, instance['hidden'])
            episode = ProjectileEpisode(backend, instance, args.output, physics_hz=args.physics_hz, horizon_ticks=args.horizon_ticks)
            scene = backend
        elif runtime == 'airtrack':
            from airtrack.backend import AirtrackBackend
            from airtrack.episode import AirtrackEpisode
            backend = AirtrackBackend(args, app)
            episode = AirtrackEpisode(backend, instance, args.output, physics_hz=args.physics_hz, horizon_ticks=args.horizon_ticks)
            scene = backend
        elif runtime == 'coupled':
            from coupled_oscillator.backend import CoupledBackend
            from coupled_oscillator.episode import CoupledEpisode
            backend = CoupledBackend(args, app)
            if instance:
                from instance_episodes import coupled_episode
                episode = coupled_episode(backend, instance, args.output, physics_hz=args.physics_hz, horizon_ticks=args.horizon_ticks)
            else:
                episode = CoupledEpisode(backend, physics_hz=args.physics_hz, horizon_ticks=args.horizon_ticks)
            scene = backend
        else:
            backend = IsaacBackend(args, app)
            calibration = {d['id']: d['effective_length_m'] for d in backend.scene.descriptors}
            if instance:
                from instance_episodes import pendulum_episode
                episode = pendulum_episode(backend, instance, args.output, calibration=calibration,
                    physics_hz=args.physics_hz, horizon_ticks=args.horizon_ticks)
            else:
                episode = PendulumEpisode(backend, gravity_m_s2=args.gravity, calibration=calibration,
                    physics_hz=args.physics_hz, horizon_ticks=args.horizon_ticks)
            scene = backend.scene
            scene._render()

        if args.spectator:
            from spectator import SceneViewer
            spectator = SceneViewer(scene.stage)
            after_step = backend.after_step
            def update_spectator(timers, tick):
                after_step(timers, tick)
                if tick % (args.physics_hz // 5) == 0:
                    spectator.update()
            backend.after_step = update_spectator

        process_dir = Path(args.output) / 'process'

        def capture():
            # The run's records, submitted or not (offline process score): read before the backend closes, the
            # private logs it writes at close added after.
            writer = getattr(episode, 'write_process_record', None)
            if writer is not None:
                try:
                    writer(process_dir)
                except Exception as exc:
                    process_dir.mkdir(parents=True, exist_ok=True)
                    (process_dir / 'error.txt').write_text(f'{type(exc).__name__}: {exc}')
            backend.close()
            if writer is not None:
                finish_process_record(Path(args.output) / 'private', process_dir)
            return {'video_path': str(scene.video_path),
                    'frame_count': scene.frame_index, 'finalized': True,
                    'sealed_dir': str(episode.sealed_dir) if getattr(episode, 'sealed', False) else None,
                    'process_dir': str(process_dir) if writer is not None else None}

        checkpoint_state = {'last': -CHECKPOINT_S}

        def checkpoint():
            """The process record as it stands, for a process that dies before /v1/finalize (salvage.py)."""
            writer = getattr(episode, 'write_process_record', None)
            now = time.monotonic()
            if writer is None or episode.terminated or now - checkpoint_state['last'] < CHECKPOINT_S:
                return
            checkpoint_state['last'] = now
            final, staging = Path(args.output) / 'process_checkpoint', Path(args.output) / 'process_checkpoint.tmp'
            try:
                shutil.rmtree(staging, ignore_errors=True)
                writer(staging)
                video = getattr(scene, 'video_path', None)
                (staging / 'checkpoint.json').write_text(json.dumps({
                    'wall_time': time.time(), 'physics_tick': getattr(episode, 'tick', None),
                    'video_path': str(video) if video else None, 'frame_count': getattr(scene, 'frame_index', None)}))
                for handle in (getattr(scene, 'audit_file', None), getattr(scene, 'events_file', None)):
                    if handle is not None and not handle.closed:
                        handle.flush()
                shutil.rmtree(final, ignore_errors=True)
                staging.rename(final)
            except Exception as exc:
                print(f'EMBODIEDLAB_CHECKPOINT_ERROR {type(exc).__name__}: {exc}', flush=True)

        host_observer = host_observer_factory(backend) if host_observer_factory else None
        if host_observer is not None and hasattr(host_observer, 'after_step'):
            base_after_step = backend.after_step
            def update_host_observer(timers, tick):
                base_after_step(timers, tick)
                host_observer.after_step(tick)
            backend.after_step = update_host_observer
        server = HTTPServer((args.host, args.port), make_handler(episode, admin_key, capture, spectator, host_observer, checkpoint))
        print('EMBODIEDLAB_BENCH_PUBLIC_API_READY', flush=True)
        server.serve_forever()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if spectator is not None:
            spectator.close()
        if backend is not None:
            backend.close()
        app.close()


if __name__ == '__main__':
    main()

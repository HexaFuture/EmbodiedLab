"""Registry experiments on the apparatus runtimes.

The instance service materializes one private record per instance (hidden
values, scene parameters, the generated public task). The Bench reads it
through the read-only instance mount, writes private/truth.json, and builds the
apparatus with those values instead of sampling from a seed. It does not score:
the first submit_result freezes the episode and seals the evidence into
<output>/sealed (format `embodiedlab.sealed-episode/v1`, read by
benchmark/evaluation), with the state of the loose items on the bench at that moment (bench_order). At stop, submitted or not, the same records go to
<output>/process (format `embodiedlab.process-record/v1`) with the end of the run
and the backend's private logs, for the offline process score. Legacy task ids
keep their own code paths untouched.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import shutil
import time
from copy import deepcopy
from pathlib import Path

SEALED_SCHEMA = 'embodiedlab.sealed-episode/v1'
PROCESS_SCHEMA = 'embodiedlab.process-record/v1'
EXPERIMENT_RUNTIMES = {'hooke_k': 'hooke', 'pendulum_g': 'pendulum', 'coupled_modes': 'coupled',
                       'hooke_dynamic_k': 'hooke', 'torsion_inertia': 'torsion', 'torsion_centroid': 'torsion_centroid',
                       'rc_capacitance': 'circuit_grid', 'resistor_blackbox': 'circuit_grid', 'circuit_blackbox3': 'circuit_grid',
                       'airtrack_collision': 'airtrack', 'spectro_basic': 'chem',
                       'titration_basic': 'chem', 'incline_friction': 'friction',
                       'projectile_landing': 'projectile', 'spectro_mixture': 'chem', 'dye_absorptivity': 'chem',
                       'kinetics_crystal_violet': 'chem', 'rectifier_supply': 'circuit_grid',
                       'titration_weak_acid': 'chem', 'led_planck': 'circuit_grid', 'battery_emf': 'circuit_grid', 'lens_focal': 'optics',
                       'grating_wavelength': 'optics', 'slit_width': 'optics', 'plate_index': 'optics', 'optical_relay': 'optics',
                       'calorimetry_specific_heat': 'calorimetry', 'thermistor_beta': 'thermo', 'rod_expansion': 'thermo',
                       'beam_compliance': 'beam'}


def load_instance(path, exp_id, private_dir):
    record = json.loads(Path(path).read_text())
    if record.get('schema_version') != 'embodiedlab.instance/v1' or record.get('exp_id') != exp_id:
        raise ValueError('instance record does not match the requested experiment')
    if record.get('runtime') != EXPERIMENT_RUNTIMES.get(exp_id):
        raise ValueError('instance record names a different apparatus runtime')
    private_dir = Path(private_dir)
    private_dir.mkdir(parents=True, exist_ok=True)
    truth = {'exp_id': record['exp_id'], 'instance_id': record['instance_id'], 'hidden': record['hidden']}
    digest = hashlib.sha256(json.dumps(truth, sort_keys=True).encode()).hexdigest()
    if digest != record['truth_sha256']:
        raise ValueError('instance truth does not match its recorded hash')
    (private_dir / 'truth.json').write_text(json.dumps(truth, indent=2))
    return record


PROCESS_PRIVATE_SKIP = ('.usd', '.usda', '.usdc')   # the scene, rebuilt from the instance; tens of MB per run
PROCESS_PRIVATE_MAX_BYTES = 50_000_000


def finish_process_record(private_dir, root):
    """Add the backend's private logs (written at close; the truth and the scene stay behind) and the manifest."""
    root, private_dir = Path(root), Path(private_dir)
    target = root / 'private'
    target.mkdir(parents=True, exist_ok=True)
    skipped = {}
    for path in sorted(private_dir.glob('*')) if private_dir.is_dir() else ():
        if not path.is_file() or path.name == 'truth.json':
            continue
        if path.suffix in PROCESS_PRIVATE_SKIP:
            skipped[path.name] = path.stat().st_size
            continue
        if path.suffix == '.jsonl':
            # Frame logs of long per-tick runs exceed the cap raw (100 MB and more) but compress 10-15x: the cap
            # applies to the stored file.
            stored = target / (path.name + '.gz')
            with path.open('rb') as source, gzip.open(stored, 'wb') as sink:
                shutil.copyfileobj(source, sink)
            if stored.stat().st_size > PROCESS_PRIVATE_MAX_BYTES:
                stored.unlink()
                skipped[path.name] = path.stat().st_size
        elif path.stat().st_size > PROCESS_PRIVATE_MAX_BYTES:
            skipped[path.name] = path.stat().st_size
        else:
            shutil.copy2(path, target / path.name)
    if skipped:
        (root / 'private_skipped.json').write_text(json.dumps(skipped, indent=2, sort_keys=True))
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))


def appearance_seed(instance_id):
    """Cosmetic randomness (e.g. colours) keyed by the opaque id, never by the index."""
    return int(hashlib.sha256(('appearance:' + instance_id).encode()).hexdigest()[:8], 16)


class InstanceMixin:
    """Shared public task, submission freeze and sealing for registry experiments."""

    def _instance_init(self, instance, output):
        self.instance = instance
        self._instance_wall_start = time.monotonic()
        self.sealed_dir = Path(output) / 'sealed'
        self.submission_attempts = 0
        self.sealed = False
        self._bench_order = None
        self._bench_order_error = None

    def _order(self):
        """The bench-order tracker, made before the first tool call moves anything; None if it cannot read the stage."""
        if getattr(self, '_bench_order', None) is None and getattr(self, '_bench_order_error', None) is None:
            self._bench_order = None
            try:
                from bench_order import BenchOrder
                self._bench_order = BenchOrder()
                self._bench_order.observe(self.tick, None)
            except Exception as exc:
                self._bench_order_error = f'{type(exc).__name__}: {exc}'
        return self._bench_order

    def call(self, request_id, name, arguments):
        order = self._order()
        logged = len(getattr(self, 'call_log', ()))
        response = super().call(request_id, name, arguments)
        self._log_export(logged, name, arguments, response)
        if order is not None:
            try:
                order.observe(self.tick, name)
            except Exception as exc:
                self._bench_order_error = f'{type(exc).__name__}: {exc}'
        return response

    def _log_export(self, logged, name, arguments, response):
        """Export log of the process record (private, never returned): one entry per successful
        export_* call that ran (a replayed request_id adds none). `call` is its index in the call log; `evidence_ids`
        the records the export returned (the `evidence_id` of the result, or of each item of a list in it);
        `result_sha256` the SHA-256 of the result as canonical JSON, for exports without an evidence id."""
        try:
            log = getattr(self, 'call_log', None)
            if (not name.startswith('export_') or not isinstance(response, dict) or not response.get('ok')
                    or log is None or len(log) != logged + 1):
                return
            result = response.get('result')
            ids = []
            if isinstance(result, dict):
                ids += [result['evidence_id']] if isinstance(result.get('evidence_id'), str) else []
                for value in result.values():
                    if isinstance(value, list):
                        ids += [v['evidence_id'] for v in value
                                if isinstance(v, dict) and isinstance(v.get('evidence_id'), str)]
            elif isinstance(result, list):
                ids += [v['evidence_id'] for v in result if isinstance(v, dict) and isinstance(v.get('evidence_id'), str)]
            digest = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=True,
                                               default=repr).encode()).hexdigest()
            if not hasattr(self, 'export_log'):
                self.export_log = []
            self.export_log.append({'call': logged, 'name': name, 'tick': self.tick, 'arguments': deepcopy(arguments),
                                    'evidence_ids': list(dict.fromkeys(ids)), 'result_sha256': digest})
        except Exception as exc:
            self._export_log_error = f'{type(exc).__name__}: {exc}'

    def bench_order_snapshot(self, with_start=False):
        """The bench now; with_start adds start_items, the items before the first tool call (for the seal)."""
        order = self._order()
        try:
            if order is None:
                return {'error': getattr(self, '_bench_order_error', None)}
            snapshot = order.snapshot()
            if with_start:
                snapshot['start_items'] = order.start_items()
            return snapshot
        except Exception as exc:
            return {'error': f'{type(exc).__name__}: {exc}'}

    def instance_public_task(self, extra):
        task = deepcopy(self.instance['public_task'])
        task.update(extra)
        task.update(episode_id=self.episode_id, physics_hz=self.limits.physics_hz,
                    horizon_ticks=self.horizon_ticks, horizon=self.horizon_ticks // (self.limits.physics_hz // 30),
                    horizon_note='the simulated-time cap of this episode (budget.sim_s) counted as physics ticks '
                                 '(horizon_ticks at physics_hz) and as 30 Hz frames (horizon); it is not a step limit of its own',
                    device_rpc_limit=65536, control_track='public_joint_trajectory',
                    device_calibration=deepcopy(self.backend.public_calibration()))
        return task

    def freeze_and_seal(self, submission, terminal_state, device, evidence, files=()):
        self.submission_attempts += 1
        self.submit_tick = self.tick
        self.terminated = True
        root = self.sealed_dir
        (root / 'evidence').mkdir(parents=True, exist_ok=True)
        for relative, data in files:
            (root / relative).write_bytes(data)
        episode = {
            'schema_version': SEALED_SCHEMA, 'exp_id': self.instance['exp_id'],
            'instance_id': self.instance['instance_id'], 'spec_digest': self.instance['spec_digest'],
            'runtime': self.instance['runtime'], 'episode_id': self.episode_id,
            'physics_hz': self.limits.physics_hz, 'horizon_ticks': self.horizon_ticks,
            'submit_tick': self.tick, 'request_count': len(self.requests) + 1,
            'submission_attempts': self.submission_attempts,
            'terminal_state': terminal_state, 'device': device,
            'layout_jitter_applied': False,
            'bench_order': self.bench_order_snapshot(with_start=True),
        }
        if self.instance['exp_id'] == 'circuit_blackbox3' and self.instance['runtime'] == 'circuit':   # the old panel
            episode['wall_s_used'] = time.monotonic() - self._instance_wall_start
            logger = self.backend.bench.logger
            episode['logger_arms_used'] = logger.max_arms - logger.remaining_arms
        (root / 'episode.json').write_text(json.dumps(episode, indent=2))
        # Keep the submission exactly as received, including non-finite numbers.
        (root / 'submission.json').write_text(json.dumps(submission, indent=2, allow_nan=True))
        (root / 'evidence.json').write_text(json.dumps(evidence, indent=2))
        manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
        self.sealed = True
        # The Agent receives only a receipt; scoring happens offline.
        return {'accepted': True, 'terminated': True, 'episode_id': self.episode_id}

    def end_reason(self):
        if self.sealed:
            return 'submitted'
        return self.termination_reason or 'stopped'

    def write_process_record(self, root):
        """The run's records at stop, submitted or not, for the offline process score.

        Terminal state, device and evidence come from seal_payload, as a submission at this moment would seal them;
        the bench's own logs and the call log are added, and the server adds the backend's private logs once it has
        closed the backend (finish_process_record). The Agent never sees this directory."""
        root = Path(root)
        (root / 'evidence').mkdir(parents=True, exist_ok=True)
        record = {
            'schema_version': PROCESS_SCHEMA, 'exp_id': self.instance['exp_id'],
            'instance_id': self.instance['instance_id'], 'spec_digest': self.instance['spec_digest'],
            'runtime': self.instance['runtime'], 'episode_id': self.episode_id,
            'physics_hz': self.limits.physics_hz, 'horizon_ticks': self.horizon_ticks,
            'end_tick': self.tick, 'end_reason': self.end_reason(),
            'termination_detail': getattr(self, 'termination_detail', None),
            'submitted': bool(self.sealed), 'submit_tick': getattr(self, 'submit_tick', None),
            'request_count': len(self.requests), 'calls': list(getattr(self, 'call_log', [])),
            'exports': list(getattr(self, 'export_log', [])),     # export call -> record(s), see _log_export
        }
        if getattr(self, '_export_log_error', None):
            record['exports_error'] = self._export_log_error
        evidence = {}
        try:
            payload = self.seal_payload()
            record.update(terminal_state=payload[0], device=payload[1])
            evidence = payload[2]
            for relative, data in (payload[3] if len(payload) > 3 else ()):
                (root / relative).write_bytes(data)
        except Exception as exc:
            record['payload_error'] = f'{type(exc).__name__}: {exc}'
        bench = getattr(self.backend, 'bench', None)
        logs = {name: getattr(bench, name) for name in ('rows', 'private', 'state_log', 'actions', 'transitions')
                if isinstance(getattr(bench, name, None), list)}
        if getattr(self, 'gates', None):
            logs['gates'] = self.gates
        if getattr(self, 'failure_traceback', None):      # private: the traceback behind termination_detail
            logs['backend_failure'] = self.failure_traceback
        order, error = getattr(self, '_bench_order', None), getattr(self, '_bench_order_error', None)
        if order is not None or error is not None:
            logs['bench_order'] = dict(order.record() if order is not None else {}, error=error,
                                       at_stop=self.bench_order_snapshot() if order is not None else None)
        # Records a bench or an episode keeps outside these lists (scans, open recordings, timer states): each
        # source names them in process_extra -> {name: JSON-able value}.
        for source in (self, bench):
            extra = getattr(source, 'process_extra', None)
            if callable(extra):
                try:
                    logs.update(extra())
                except Exception as exc:
                    logs.setdefault('extra_errors', []).append(f'{type(source).__name__}: {type(exc).__name__}: {exc}')
        (root / 'episode.json').write_text(json.dumps(record, indent=2, default=repr))
        (root / 'evidence.json').write_text(json.dumps(evidence, indent=2, default=repr))
        (root / 'bench_logs.json').write_text(json.dumps(logs, default=repr))

    def instance_evaluation(self):
        return {'reward': 0., 'task_completed': False, 'terminated': self.terminated,
                'sim_step': self.tick // (self.limits.physics_hz // 30),
                'evaluator': 'offline:benchmark.evaluation', 'sealed': self.sealed,
                'sealed_dir': str(self.sealed_dir) if self.sealed else None,
                'error': self.termination_reason}


# --- hooke_k: Hooke ------------------------------------------------------------------------
def hooke_score(instance, horizon):
    from hooke_scoring import HookeEpisode, WEIGHT_IDS

    class InstanceHookeScore(HookeEpisode):
        def __init__(self):
            hidden = instance['hidden']
            n = int(hidden['n_weights'])
            super().__init__(0, horizon, task_id='hooke_identification_v2')
            self.configuration = {'stiffness_n_m': float(hidden['k_N_per_m']), 'weight_count': n,
                                  'weight_mass_kg': 0.05, 'randomization_version': 'registry:' + instance['spec_digest'][:12]}
            self.truth = self.configuration['stiffness_n_m']
            self.task_id = instance['exp_id']
            self.weight_ids = WEIGHT_IDS[:n]
            self.records = {}

        def add_evidence(self, frame, supported, stable, released, reading_cm, image_sha256s):
            """image_sha256s: {camera: SHA-256 of its PNG} of the three standard views of this state."""
            eid = f'{self.episode_id}:obs:{len(self.records) + 1}'
            number = eid.rsplit(':', 1)[-1]
            self.records[eid] = {'kind': 'ruler_image', 'evidence_id': eid, 'frame': frame,
                                 'supported': sorted(supported), 'stable': bool(stable), 'released': bool(released),
                                 'automatic_reading_cm': reading_cm,
                                 'images': {camera: {'file': f'evidence/{number}_{camera}.png', 'sha256': sha}
                                            for camera, sha in sorted(image_sha256s.items())}}
            return eid

    return InstanceHookeScore()


def hooke_runtime(backend, score, instance, output, *, physics_hz, horizon_ticks):
    from hooke_runtime import HookeRuntime, initialize_hooke_episode, PUBLIC_TOOLS

    class InstanceHookeRuntime(InstanceMixin, HookeRuntime):
        def public_task(self):
            return self.instance_public_task({
                'weights': [{'id': w, 'mass_kg': 0.05} for w in self.score.weight_ids],
                'gravity_m_s2': 9.81, **({} if getattr(self.backend, 'controls', False) else {'ruler_unit': 'cm'})})

        def observation(self):
            value = super(HookeRuntime, self).observation()
            supported, stable, released = self.backend.support_state()
            # The evidence images are the three standard views of this state (no extra camera or crop).
            views = self.backend.view_pngs
            eid = self.score.add_evidence(self.tick // (self.limits.physics_hz // 30), supported, stable, released,
                                          self.backend.automatic_reading,
                                          {camera: hashlib.sha256(png).hexdigest() for camera, png in views.items()})
            if hasattr(self.backend, 'tray_free'):   # sealed record only (six levels): see HookeBackend.tray_free
                self.score.records[eid]['tray_free'] = bool(self.backend.tray_free())
            directory = self.backend.private / 'evidence'
            directory.mkdir(exist_ok=True)
            for camera, png in views.items():
                (directory / f"{eid.rsplit(':', 1)[-1]}_{camera}.png").write_bytes(png)
            value['evidence_id'] = eid
            return value

        def seal_payload(self):
            """Terminal state, device, evidence and files of the sealed record (also used at stop)."""
            supported, stable, released = self.backend.support_state()
            directory = self.backend.private / 'evidence'
            files = [(v['file'], (directory / Path(v['file']).name).read_bytes())
                     for r in self.score.records.values() for v in r['images'].values()]
            return (
                {'weight_ids': list(self.score.weight_ids), 'supported': sorted(supported),
                 'stable': bool(stable), 'released': bool(released),
                 'weight_box_seats': self.backend.seat_occupancy()},
                {'weight_mass_kg': 0.05, 'gravity_for_conversion_m_s2': 9.81,
                 'weight_box': deepcopy(getattr(self.backend, 'weight_box', None))},
                deepcopy(self.score.records), files)

        def process_extra(self):
            """Weight box occupancy for the process record: at start, seat_in / seat_out events, at stop."""
            return {'weight_box_seats': self.backend.end_seat_log()}

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if name not in PUBLIC_TOOLS:
                raise ValueError('not a Hooke device command')
            if name == 'get_task_spec' and args == {}:
                return self.public_task()
            if name == 'submit_result':
                return self.freeze_and_seal(args, *self.seal_payload())
            return super()._dispatch(name, args)

        def evaluation(self):
            return self.instance_evaluation()

    # Reuse the validated empty-apparatus initialisation, then rebind the class.
    episode = initialize_hooke_episode(backend, score, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode.__class__ = InstanceHookeRuntime
    episode._instance_init(instance, output)
    # Evidence recorded during initialisation (none: the settle call is a wait) stays empty.
    return episode


# --- hooke_dynamic_k: Hooke, dynamic (tray encoder) ------------------------------------------------------
TRAY_MASS_PUBLIC_KG = 0.080     # the tray rigid body without the spring's effective mass (hooke_physics.HANGER_MASS_KG)
ENCODER_TOOLS = ('export_encoder',)   # recordings are keyed on the bench's recorder panel; the damper is a lever


def hooke_dynamics(instance):
    """Tray body m_t + m_s/3 and drive damping 2*gamma*(m_t + m_s/3), gamma of the empty tray."""
    hidden = instance['hidden']
    mass = TRAY_MASS_PUBLIC_KG + float(hidden['m_s_kg']) / 3
    return {'body_mass_kg': mass, 'damping_n_s_m': 2 * float(hidden['gamma_per_s']) * mass, 'encoder': True}


def hooke_dynamic_runtime(backend, score, instance, output, *, physics_hz, horizon_ticks):
    from hooke_encoder import TrayEncoder

    episode = hooke_runtime(backend, score, instance, output, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    base = type(episode)

    class InstanceHookeDynamicRuntime(base):
        def public_task(self):
            return super().public_task() | {'tray_mass_public_kg': TRAY_MASS_PUBLIC_KG}

        def seal_payload(self):
            """Terminal state, device, evidence and files of the sealed record (also used at stop)."""
            encoder = self.backend.encoder
            supported, stable, released = self.backend.support_state()
            directory = self.backend.private / 'evidence'
            files = [(v['file'], (directory / Path(v['file']).name).read_bytes())
                     for r in self.score.records.values() for v in r['images'].values()]
            evidence = deepcopy(self.score.records)
            for eid, record in encoder.records.items():
                evidence[eid] = dict(deepcopy(record), kind='tray_encoder', **deepcopy(encoder.private[eid]))
            return (
                {'weight_ids': list(self.score.weight_ids), 'supported': sorted(supported),
                 'stable': bool(stable), 'released': bool(released),
                 **self.backend.bench_state(),   # damper lever and recorder panel (encoder_bench_restored)
                 'weight_box_seats': self.backend.seat_occupancy()},
                {'weight_mass_kg': 0.05, 'tray_mass_public_kg': TRAY_MASS_PUBLIC_KG,
                 'gravity_for_conversion_m_s2': 9.81,
                 'weight_box': deepcopy(getattr(self.backend, 'weight_box', None))}, evidence, files)

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if name == 'export_encoder' and isinstance(args, dict) and set(args) == {'evidence_id'}:
                return self.backend.encoder.export(self.record_id(args['evidence_id']))
            if name in ENCODER_TOOLS:
                raise ValueError('unexpected arguments')
            if name == 'submit_result':
                return self.freeze_and_seal(args, *self.seal_payload())
            return super()._dispatch(name, args)

        def record_id(self, ref):
            """A finished record by its evidence_id, its number (the display's NO field) or "latest"."""
            records = self.backend.encoder.records
            if ref == 'latest':
                if not records:
                    raise ValueError('no finished record yet')
                return list(records)[-1]
            if (isinstance(ref, int) and not isinstance(ref, bool)) or (isinstance(ref, str) and ref.isdigit()):
                return f'{self.episode_id}:encoder:{int(ref)}'
            if not isinstance(ref, str):
                raise ValueError('evidence_id must be an evidence id, a record number or "latest"')
            return ref

    episode.__class__ = InstanceHookeDynamicRuntime
    backend.release_settling_damping()
    backend.encoder = TrayEncoder(episode.episode_id, physics_hz)
    return episode


# --- pendulum_g: pendulum ------------------------------------------------------------------------
def pendulum_episode(backend, instance, output, *, calibration, **kwargs):
    from episode_runtime import PendulumEpisode
    from apparatus_manual import apparatus_description

    labels = {key: f'P{index}' for index, (key, _) in
              enumerate(sorted(calibration.items(), key=lambda item: item[1]), 1)}

    def face_panel(gate):
        """(elapsed s, cycles, word) a timer face draws for a gate's acquisition (None: idle), as isaac_scene._render."""
        if gate is None:
            return 0.0, 0, 'IDLE'
        r = gate['timer'].reading()
        return (round(r.elapsed_s, 3), r.cycles, 'ERR' if r.status == 'invalid' else 'DONE' if r.valid else
                'RUN' if r.status == 'timing' else 'ARM')

    class InstancePendulumEpisode(InstanceMixin, PendulumEpisode):
        """The timers are read on their faces only (no export_gate). Every get_observation is
        evidence of the state it shows: evidence_id <episode>:obs:<n> (kind bench_view), whose private record keeps
        what each timer face showed (state word, TIME, CYCLES) and which acquisition it showed (timer_faces); the three
        standard views are sealed with it. The acquisitions stay private evidence (kind optical_gate) with their
        validity audit, for the gate (benchmark/evaluation/gates.py _timer_display) and the process rule."""
        records = None          # get_observation evidence (private)
        shown = None            # per pendulum: the acquisition its face has shown since the last display refresh

        def public_task(self):
            return self.instance_public_task({
                'effective_lengths_m': deepcopy(self.calibration),
                'apparatus': apparatus_description(self.calibration)})

        def seal_payload(self):
            """Terminal state, device, evidence and files of the sealed record (also used at stop)."""
            evidence = {}
            for eid, record in self.evidence.items():
                evidence[eid] = dict(deepcopy(record), kind='optical_gate', evidence_id=eid,
                                     pendulum_label=labels[record['pendulum_id']])
            # Each gate timer as it stands (idle: reset, display at zero) for the gate_timers_reset gate.
            timers = {labels[k]: self.gates[k]['timer'].reading().status if k in self.gates else 'idle'
                      for k in self.calibration}
            records = deepcopy(self.records or {})
            evidence.update(records)
            directory = self._evidence_dir()
            files = [(v['file'], (directory / v['file'].rsplit('/', 1)[-1]).read_bytes())
                     for r in records.values() for v in r['images'].values()]
            return ({'gate_timers': timers},
                {'effective_lengths_m': {labels[k]: v for k, v in self.calibration.items()},
                 'pendulum_labels': dict(labels)}, evidence, files)

        def _evidence_dir(self):
            private = getattr(self.backend, 'private', None) or self.backend.scene.private
            return Path(private) / 'evidence'

        def _step(self, target=None, velocity=None):
            """One tick. The backend redraws the timer faces on every video frame (physics_hz/30 ticks) from the gates
            as they stand after the tick: note which acquisition each face shows from then on."""
            try:
                super()._step(target, velocity)
            finally:
                ticks = getattr(getattr(self.backend, 'scene', None), 'frame_ticks', None) or self.limits.physics_hz // 30
                if self.tick % ticks == 0:
                    self.shown = {key: self.gates.get(key) for key in self.backend.pendulum_ids}

        def timer_faces(self):
            """What each timer face shows now (private): P label -> state word, TIME (s) and CYCLES as drawn (None
            where ERR shows dashes) and the acquisition shown (its private evidence id once it has finished)."""
            drawn = getattr(getattr(self.backend, 'scene', None), 'last_display', None) or {}
            shown = self.shown or {}
            faces = {}
            for index, key in enumerate(self.backend.pendulum_ids):
                gate = shown.get(key)
                elapsed, cycles, word = drawn.get(index) or face_panel(gate)
                dashes = word == 'ERR'
                faces[labels[key]] = {'pendulum_id': key, 'word': word,
                                      'time_s': None if dashes else min(99999, int(elapsed * 1000 + .5)) / 1000,
                                      'cycles': None if dashes else int(cycles),
                                      'acquisition': gate.get('evidence_id') if gate else None}
            return faces

        def observation(self):
            """The three standard views with an evidence_id bound to this physical state (see the class docstring)."""
            value = super().observation()
            if self.records is None:
                self.records = {}
            eid = f'{self.episode_id}:obs:{len(self.records) + 1}'
            number = eid.rsplit(':', 1)[-1]
            directory = self._evidence_dir()
            directory.mkdir(parents=True, exist_ok=True)
            images = {}
            for frame in value['frames']:
                png = base64.b64decode(frame['rgb']['data'])
                name = f"{number}_{frame['camera']}.png"
                (directory / name).write_bytes(png)
                images[frame['camera']] = {'file': f'evidence/{name}', 'sha256': hashlib.sha256(png).hexdigest()}
            self.records[eid] = {'kind': 'bench_view', 'evidence_id': eid, 'tick': self.tick,
                                 't_s': round(self.tick / self.limits.physics_hz, 6), 'timers': self.timer_faces(),
                                 'images': images}
            value['evidence_id'] = eid
            return value

        def process_extra(self):
            return super().process_extra() | {'observations': len(self.records or {})}

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if name == 'get_task_spec' and args == {}:
                return self.public_task()
            if name == 'record_prediction':
                raise ValueError('not a tool of this experiment')
            if name == 'submit_result':
                return self.freeze_and_seal(args, *self.seal_payload())
            return super()._dispatch(name, args)

        def evaluation(self):
            return self.instance_evaluation()

    episode = InstancePendulumEpisode(backend, gravity_m_s2=instance['hidden']['g_m_per_s2'],
                                      calibration=calibration, **kwargs)
    episode._instance_init(instance, output)
    return episode


# --- coupled_modes: coupled oscillators --------------------------------------------------------------
def coupled_parameters(instance):
    from coupled_oscillator.model import CoupledParameters
    h = instance['hidden']
    # Local (uncoupled) frequencies, as in the legacy sampler (dynamics.Dynamics local_omega_*).
    return CoupledParameters(float(h['omega_a']), float(h['omega_b']), float(h['c_per_s2']),
                             float(h['gamma_per_s']), bool(h['identical']))


def coupled_episode(backend, instance, output, **kwargs):
    from coupled_oscillator.episode import CoupledEpisode

    class InstanceCoupledEpisode(InstanceMixin, CoupledEpisode):
        def public_task(self):
            return self.instance_public_task({})

        def seal_payload(self):
            """Terminal state, device, evidence and files of the sealed record (also used at stop)."""
            evidence = {eid: dict(deepcopy(record), kind='coupled_recorder')
                        for eid, record in self.recorder.records.items()}
            return (
                self.backend.bench_state('seal'),
                {'slider_mass_kg': self.backend.dynamics.mass_kg}, evidence)

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if name == 'get_task_spec' and not args:
                return self.public_task()
            if name == 'submit_result':
                return self.freeze_and_seal(args, *self.seal_payload())
            return super()._dispatch(name, args)

        def evaluation(self):
            return self.instance_evaluation()

    episode = InstanceCoupledEpisode(backend, **kwargs)
    episode._instance_init(instance, output)
    return episode


# --- torsion_inertia: torsion pendulum ----------------------------------------------------------------------
TORSION_DAMPING_NMS_PER_RAD = 1.2e-4     # small spring damping: gamma = c / 2I ~ 0.02-0.06 1/s
TORSION_TOOLS = ('export_photogate',)    # REC on the recorder panel, the damper lever by hand


def torsion_dynamics(instance):
    """Spring constant, turntable I0 and the unknown object's Izz from the instance; brake near critical."""
    import math
    h = instance['hidden']
    kappa, inertia = float(h['kappa_Nm_per_rad']), float(h['I0_kgm2'])
    return {'kappa_nm_per_rad': kappa, 'turntable_inertia_kgm2': inertia,
            'unknown_inertia_kgm2': float(h['I_unknown_kgm2']),
            'damping_nms_per_rad': TORSION_DAMPING_NMS_PER_RAD,
            'brake_damping_nms_per_rad': 2 * math.sqrt(kappa * (inertia + 0.0015)),
            'mark_rgb': torsion_mark_colour(instance['instance_id'])}


def torsion_mark_colour(instance_id):
    """Colour of the painted centre circle: one fixed saturated magenta for every instance (audit R4),
    clearly distinct from the grey platter, the yellow push post and the blue/green objects."""
    del instance_id
    return [0.80, 0.10, 0.70]


def torsion_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    from device_runtime import DeviceRuntime
    from torsion_gate import Photogate
    import torsion_apparatus as asset

    class TorsionEpisode(InstanceMixin, DeviceRuntime):
        def public_task(self):
            return self.instance_public_task({
                'objects': [
                    {'id': 'standard_disc', 'mass_kg': asset.OBJECTS['standard_disc'][2],
                     'moment_of_inertia_kgm2': asset.STANDARD_INERTIA_KGM2,
                     'description': 'blue uniform solid disc with a handle on its axis; moment of inertia about its axis'},
                    {'id': 'unknown_object', 'mass_kg': asset.OBJECTS['unknown_object'][2],
                     'description': 'green; same outer shape and handle as the disc; internal mass distribution not known'}]})

        def robot_state(self):
            return super().robot_state() | {'gripper_holding': bool(self.backend.robot_state()['gripper_holding'])}

        def seal_payload(self):
            """Terminal state, device, evidence and files of the sealed record (also used at stop)."""
            gate = self.backend.photogate
            evidence = {eid: dict(deepcopy(record), kind='torsion_photogate', **deepcopy(gate.private[eid]))
                        for eid, record in gate.records.items()}
            # Bench state (TorsionBackend.bench_state): storage box seats, damper lever, recorder, held object.
            return (
                {'loads': self.backend.loads(), 'photogate_active': gate.active is not None,
                 **self.backend.bench_state('seal')},
                {'standard_disc_inertia_kgm2': asset.STANDARD_INERTIA_KGM2,
                 'object_masses_kg': {k: v[2] for k, v in asset.OBJECTS.items()},
                 'flag_half_angle_rad': asset.FLAG_HALF_ANGLE_RAD,
                 'storage_box': {'seats': {str(i + 1): key for i, key in enumerate(asset.BOX_SEATS)},
                                 'dxy_m': asset.SEAT_DXY_M, 'abs_dz_m': asset.SEAT_DZ_M,
                                 'tilt_deg': asset.SEAT_TILT_DEG}}, evidence)

        def process_extra(self):
            """Storage box occupancy for the process record: at start, seat_in / seat_out events, at stop."""
            return {'storage_box_seats': self.backend.end_seat_log()}

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if not isinstance(args, dict):
                raise ValueError('arguments must be an object')
            gate = self.backend.photogate
            if name == 'get_task_spec' and args == {}:
                return self.public_task()
            if name == 'export_photogate' and set(args) == {'evidence_id'}:
                return gate.export(args['evidence_id'])
            if name in TORSION_TOOLS:
                raise ValueError('unexpected arguments')
            if name == 'submit_result':
                return self.freeze_and_seal(args, *self.seal_payload())
            return super()._dispatch(name, args)

        def evaluation(self):
            return self.instance_evaluation()

    # Apparatus preparation (not charged): open gripper, settle with the preparation damping, then release it (the
    # damper lever stands at OFF).
    backend.set_gripper(.08)
    for _ in range(130 * (physics_hz // 30)):
        backend.step(None)
        backend.after_step({}, backend.tick_index)
    if not backend.at_rest() or backend.loads():
        raise RuntimeError('empty turntable failed to settle')
    backend.tick_index = 0
    backend.history.clear()
    episode = TorsionEpisode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    backend.release_settling_damping()
    backend.photogate = Photogate(episode.episode_id, physics_hz, record_range=asset.RECORD_S_RANGE)
    backend.start_seat_log()
    backend.bench_state('start')
    return episode


# --- rc_capacitance: circuit panel ---------------------------------------------------------------------------
CIRCUIT_TOOLS = ('arm_logger', 'read_logger', 'export_logger')


LED_LABELS = ('L1', 'L2', 'L3', 'L4', 'X')


def led_configuration(instance):
    """led_planck: the LED board (turn-on energy, ideality and series resistance per LED), actual R_s values and
    the public initial slider positions (LED and R_s drawn per instance, supply OFF)."""
    import random
    from led_solver import turn_on_energy
    h = instance['hidden']
    known = instance['public_task']['public_constants']['leds']
    rng = random.Random(appearance_seed(instance['instance_id']) ^ 0x1ED5)
    leds = {}
    for label in LED_LABELS:
        nm = float(h['lambda_x_nm']) if label == 'X' else float(known[label])
        leds[label] = {'energy_eV': turn_on_energy(nm, float(h['delta_eV']), float(h[f'eps_{label}_eV'])),
                       'n': float(h[f'n_{label}']), 'r_ohm': float(h[f'r_{label}_ohm'])}
    printed = {'PAIR': list(LED_LABELS), 'RS': ['100', '330', '1k'], 'MODE': ['SWEEP', 'OFF', 'ON']}
    panel = {'detent_labels': {'PAIR': list(LED_LABELS), 'RS': [100, 330, 1000], 'MODE': ['SWEEP', 'OFF', 'ON']},
             'printed': printed, 'names': {'PAIR': 'LED', 'RS': 'RS', 'MODE': 'SUPPLY'},
             'board': {'leds': [{'label': label, 'nm': (float(h['lambda_x_nm']) if label == 'X' else float(known[label])),
                                 'printed': ('' if label == 'X' else f'{known[label]}'), 'clear': label == 'X'}
                                for label in LED_LABELS]}}
    return {'leds': leds, 'panel': panel,
            'rs_actual_ohm': {100: 100.0 * float(h['rs_factor_100']), 330: 330.0 * float(h['rs_factor_330']),
                              1000: 1000.0 * float(h['rs_factor_1k'])},
            'initial': {'PAIR': rng.choice(list(LED_LABELS)), 'RS': rng.choice([100, 330, 1000]), 'MODE': 'OFF'},
            'noise_seed': appearance_seed('logger-noise:' + instance['instance_id'])}


CELL_LOADS = ('20', '10', '5', '3', '2')
CELL_SHUNTS = (0.1, 0.5, 1.0)


def battery_configuration(instance):
    """battery_emf: the cell board (EMF, internal resistance), actual load and shunt values and the public initial
    slider positions (load and shunt drawn per instance, switch OFF)."""
    import random
    h = instance['hidden']
    rng = random.Random(appearance_seed(instance['instance_id']) ^ 0xCE11)
    panel = {'detent_labels': {'PAIR': list(CELL_LOADS), 'RS': list(CELL_SHUNTS), 'MODE': ['OFF', 'ON']},
             'printed': {'PAIR': list(CELL_LOADS), 'RS': ['0.1', '0.5', '1.0'], 'MODE': ['OFF', 'ON']},
             'names': {'PAIR': 'LOAD', 'RS': 'SHUNT', 'MODE': 'SWITCH'}, 'board': {'kind': 'cell'}}
    return {'cell': {'emf_V': float(h['E_V']), 'r_ohm': float(h['r_ohm'])}, 'panel': panel,
            'load_actual_ohm': {k: float(k) * float(h[f'load_factor_{k}']) for k in CELL_LOADS},
            'shunt_actual_ohm': {f'{s:g}': s * float(h[f'shunt_factor_{tag}'])
                                 for s, tag in zip(CELL_SHUNTS, ('01', '05', '10'))},
            'initial': {'PAIR': rng.choice(CELL_LOADS), 'RS': rng.choice(CELL_SHUNTS), 'MODE': 'OFF'},
            'noise_seed': appearance_seed('logger-noise:' + instance['instance_id'])}


def circuit_configuration(instance):
    """Box network, actual R_s values and the public initial slider positions of one instance.

    The initial detents (PAIR, RS and, with initial_mode random, MODE) and the logger noise stream are
    public-state randomness keyed by the opaque instance id. Element values come from the hidden record.
    """
    import random
    if instance['recipe'] == 'led_bench':
        return led_configuration(instance)
    if instance['recipe'] == 'battery_bench':
        return battery_configuration(instance)
    h, params = instance['hidden'], instance['scene_params']
    single = {'single_c': {'C1': 'C_F'}, 'single_r': {'R1': 'R_ohm'}}     # element -> hidden value (rc_capacitance, resistor_blackbox)
    if params['box_kind'] == 'network':
        networks = {'T1': {'R1': 'R1_ohm', 'R2': 'R2_ohm'},
                    'T2': {'R1': 'R1_ohm', 'C1': 'C1_F'},
                    'T4': {'R1': 'R1_ohm', 'R2': 'R2_ohm', 'C1': 'C1_F'}}
        fields = networks[params['topology']]
    elif params['box_kind'] in single and params['topology'] == params['box_kind']:
        fields = single[params['box_kind']]
    else:
        raise ValueError('unsupported circuit box')
    rng = random.Random(appearance_seed(instance['instance_id']) ^ 0xB7C1)
    return {'topology': params['topology'],
            'values': {name: float(h[key]) for name, key in fields.items()},
            'rs_actual_ohm': {1000: 1000.0 * float(h['rs_factor_1k']), 10000: 10000.0 * float(h['rs_factor_10k']),
                              100000: 100000.0 * float(h['rs_factor_100k'])},
            'initial': {'PAIR': rng.choice(['A-B', 'B-C', 'A-C']), 'RS': rng.choice([1000, 10000, 100000]),
                        'MODE': rng.choice(['CHARGE', 'OFF', 'DISCHARGE']) if params['initial_mode'] == 'random'
                        else params['initial_mode']},
            'noise_seed': appearance_seed('logger-noise:' + instance['instance_id'])}


def circuit_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    from device_runtime import DeviceRuntime
    from circuit_bench import CircuitBench
    from circuit_solver import CircuitModel, box_network

    config = backend.configuration
    if config.get('cell'):
        from battery_solver import CellModel
        model = CellModel(config['cell']['emf_V'], config['cell']['r_ohm'], config['load_actual_ohm'],
                          config['shunt_actual_ohm'])
    elif config.get('leds'):
        from led_solver import LED, LEDModel
        model = LEDModel({label: LED(v['energy_eV'], v['n'], v['r_ohm']) for label, v in config['leds'].items()},
                         config['rs_actual_ohm'])
    else:
        model = CircuitModel(box_network(config['topology'], config['values']), config['rs_actual_ohm'])
    max_arms = int(instance['public_task']['public_constants']['logger']['max_arms'])

    class CircuitEpisode(InstanceMixin, DeviceRuntime):
        def public_task(self):
            return self.instance_public_task({})

        def robot_state(self):
            return super().robot_state() | {'gripper_holding': False}

        def seal_payload(self):
            """Terminal state, device, evidence and files of the sealed record (also used at stop)."""
            logger = backend.bench.logger
            evidence = {eid: dict(deepcopy(record), kind='circuit_logger', **deepcopy(logger.private[eid]))
                        for eid, record in logger.records.items()}
            taus = ({str(rs): model.time_constant_s('A-B', rs) for rs in (1000, 10000, 100000)}
                    if model.capacitor is not None else {})
            extra = ({'rs_nominal_ohm': [100, 330, 1000], 'led_labels': list(config['leds']),
                      'known_nm': {k: v for k, v in instance['public_task']['public_constants']['leds'].items() if v}}
                     if config.get('leds') else
                     {'shunt_nominal_ohm': list(CELL_SHUNTS), 'load_nominal_ohm': list(CELL_LOADS)}
                     if config.get('cell') else {})
            return (
                {'switch': backend.bench.labels(), 'logger_active': logger.active is not None},
                {'logger_input_ohm': 1.0e7, 'tau_ab_charge_s': taus,
                 'selector_actions': deepcopy(backend.bench.actions), **extra}, evidence)

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if not isinstance(args, dict):
                raise ValueError('arguments must be an object')
            logger = backend.bench.logger
            if name == 'get_task_spec' and args == {}:
                return self.public_task()
            if name == 'arm_logger' and set(args) == {'duration_s', 'rate_hz'}:
                # Arming changes no physical state: plans and observations stay valid, so sliders can be
                # moved while the logger records (card).
                armed = logger.arm(self.tick, args['duration_s'], args['rate_hz'], self.horizon_ticks - self.tick)
                return armed | {'note': 'Recording starts now; advance time (wait) and move sliders as needed.'}
            if name == 'read_logger' and 'arm_id' in args and set(args) <= {'arm_id', 't_from_s', 't_to_s', 'max_points'}:
                return logger.read(args['arm_id'], args.get('t_from_s'), args.get('t_to_s'),
                                   args.get('max_points', 600)) | {'logger': logger.reading(),
                                                                   'sim_time_s': self.tick / self.limits.physics_hz}
            if name == 'export_logger' and set(args) == {'arm_id'}:
                return logger.export(args['arm_id'])
            if name in CIRCUIT_TOOLS:
                raise ValueError('unexpected arguments')
            if name == 'submit_result':
                return self.freeze_and_seal(args, *self.seal_payload())
            return super()._dispatch(name, args)

        def evaluation(self):
            return self.instance_evaluation()

    # Apparatus preparation (not charged): open the gripper and let the carriages settle in their detents.
    backend.set_gripper(.08)
    for _ in range(40 * (physics_hz // 30)):
        backend.step(None)
        backend.after_step({}, backend.tick_index)
    if not backend.at_rest():
        raise RuntimeError('circuit panel sliders failed to settle')
    backend.tick_index = 0
    backend.history.clear()
    episode = CircuitEpisode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    backend.bench = CircuitBench(episode.episode_id, model, initial=config['initial'], physics_hz=physics_hz,
                                 max_arms=max_arms, noise_seed=config['noise_seed'],
                                 detent_labels=(config.get('panel') or {}).get('detent_labels'))
    return episode


# --- spectro_basic: spectrophotometry bench ----------------------------------------------------------------------
# READ values are read on the screen in the camera images (no READ log export); SCAN and KIN runs are curves
# and keep their exports.
CHEM_TOOLS = ('export_spectro_scans', 'export_spectro_kinetics')


def _screen_number(text):
    return int(text) if text.strip().isdigit() else None


def _screen_absorbance(text, point):
    """The value a drawn absorbance row shows (sign place and four digits, the point after the first) or None."""
    if not point or not text[1:].isdigit() or text[0] not in ' -':
        return None
    return (-1 if text[0] == '-' else 1) * int(text[1:]) / 1000.0


def screen_view(backend, tick, physics_hz):
    """Private record of one observation of a spectrophotometry bench (kind bench_view): the screen as drawn in its
    images (the seven-segment strings), per row the wavelength and absorbance those strings show (wavelength_nm_shown,
    A_shown; None while typing, Err, ---- or empty) next to the instrument's own row (Spectrophotometer.screen_record:
    wavelength_nm, A and the READ / BLANK / KIN point behind it), the seats and what the robot holds."""
    audit = backend.seat_audit()
    return screen_view_record(backend.shown, backend.bench.spectro.screen_record(), tick=tick, physics_hz=physics_hz,
                              seats=audit['seats'], held=audit['held'])


def screen_view_record(drawn, screen, *, tick, physics_hz, seats, held):
    """screen_view from the drawn strings (chem_spectro.screen_text) and Spectrophotometer.screen_record."""
    drawn = deepcopy(drawn)
    keys = ([('wavelength1', 'absorbance1', 'point1'), ('wavelength2', 'absorbance2', 'point2')] if 'wavelength1' in drawn
            else [('wavelength', 'absorbance', 'point')])
    private = [None] * (len(keys) - len(screen['rows'])) + list(screen['rows'])
    rows = []
    for k, ((wl, ab, pt), row) in enumerate(zip(keys, private)):
        typing = k == len(keys) - 1 and (drawn['entry'] or drawn[wl] == 'Err')
        rows.append(dict(deepcopy(row) if row else {'wavelength_nm': None, 'A': None, 'source': None},
                         wavelength_nm_shown=None if typing else _screen_number(drawn[wl]),
                         A_shown=_screen_absorbance(drawn[ab], drawn[pt])))
    return {'tick': tick, 't_s': round(tick / physics_hz, 6), 'drawn': drawn, 'rows': rows,
            'busy': screen['busy'], 'blank_prompt': bool(drawn['prompt']), 'dual': screen['dual'],
            'seats': deepcopy(seats), 'held': held}


def kinetics_configuration(instance):
    """Kinetics bench: the spectro_basic bench with CV STOCK / WATER / NaOH on its three lines (internal names stock / water /
    unknown), the KIN key, and crystal violet fading in the cuvettes at the hidden rate constant."""
    import random
    h, params = instance['hidden'], instance['scene_params']
    dye = params['dye']
    rng = random.Random(appearance_seed(instance['instance_id']) ^ 0x4B1E)
    naoh_M = float(instance['public_task']['public_constants']['naoh']['label_M']) * float(h['naoh_cert_factor'])
    sources = {'stock': {'composition_M': {dye: float(h['c_stock_uM']) * 1e-6}, 'volume_mL': float(params['stock_mL'])},
               'water': {'composition_M': {}, 'volume_mL': float(params['water_mL'])},
               'unknown': {'composition_M': {'NaOH': naoh_M}, 'volume_mL': float(params['naoh_mL'])}}
    bench = {'dyes': {dye: float(h['eps_factor'])}, 'sources': sources, 'dispense_mL': [1.00, 0.10],
             'cuvettes': int(params['n_cuvettes']), 'b0': float(h['b0']), 'b1': float(h['b1']),
             'start_nm': rng.randrange(400, 701), 'dispense_seats': {name: 'station' for name in sources},
             'mixer_seat': 'station', 'spectro_seat': 'spectro',
             'labels': {'stock': 'CV STOCK', 'water': 'WATER', 'unknown': 'NaOH'}, 'extra_key': 'kin',
             'reactions': {dye: float(h['k_M_s'])}, 'stray': float(h['stray_light'])}
    return {'bench': bench, 'noise_seed': appearance_seed('chem-noise:' + instance['instance_id'])}
UNKNOWN_DYE = 'unknown_dye'


def dye_configuration(instance):
    """dye_absorptivity: the spectro_basic bench with STOCK 1 / WATER / STOCK 2 (two stocks of one unknown dye, labelled
    concentrations; internal line names stock / water / unknown) and the SCAN key. The dye's spectrum: a main band at
    the hidden peak and a weaker shoulder 40 nm to the blue, scaled to the hidden peak absorptivity."""
    import random
    h, params = instance['hidden'], instance['scene_params']
    rng = random.Random(appearance_seed(instance['instance_id']) ^ 0xD7E5)
    lam = float(h['lambda_max_nm'])
    spec = [lam, float(h['eps_max']), [[lam, float(h['band_sigma_nm']), 1.0], [lam - 40.0, 25.0, float(h['shoulder'])]]]
    sources = {'stock': {'composition_M': {UNKNOWN_DYE: float(h['c_stock1_uM']) * 1e-6}, 'volume_mL': float(params['stock1_mL'])},
               'water': {'composition_M': {}, 'volume_mL': float(params['water_mL'])},
               'unknown': {'composition_M': {UNKNOWN_DYE: float(h['c_stock2_uM']) * 1e-6}, 'volume_mL': float(params['stock2_mL'])}}
    bench = {'dyes': {UNKNOWN_DYE: 1.0}, 'dye_specs': {UNKNOWN_DYE: spec}, 'sources': sources,
             'dispense_mL': [1.00, 0.10], 'cuvettes': int(params['n_cuvettes']), 'b0': float(h['b0']), 'b1': float(h['b1']),
             'start_nm': rng.randrange(400, 701), 'dispense_seats': {name: 'station' for name in sources},
             'mixer_seat': 'station', 'spectro_seat': 'spectro',
             'labels': {'stock': 'STOCK 1', 'water': 'WATER', 'unknown': 'STOCK 2'}, 'scan_key': True,
             'stray': float(h['stray_light'])}          # hidden per instance, as on every spectrophotometer bench
    return {'bench': bench, 'noise_seed': appearance_seed('chem-noise:' + instance['instance_id'])}


MIXTURE_LABEL = 'mixture'


def mixture_configuration(instance):
    """spectro_mixture: the spectro_basic bench with STOCK X / WATER / STOCK Y on its three lines (internal line names stock / water /
    unknown, the spectro_basic geometry), the mixture handed out in one cuvette, and the SCAN key."""
    import random
    h, params = instance['hidden'], instance['scene_params']
    dx, dy = params['dye_x'], params['dye_y']
    rng = random.Random(appearance_seed(instance['instance_id']) ^ 0xA7A7)
    sources = {'stock': {'composition_M': {dx: float(h['c_stock_x_uM']) * 1e-6}, 'volume_mL': float(params['stock_x_mL'])},
               'water': {'composition_M': {}, 'volume_mL': float(params['water_mL'])},
               'unknown': {'composition_M': {dy: float(h['c_stock_y_uM']) * 1e-6}, 'volume_mL': float(params['stock_y_mL'])}}
    mixture = {dx: float(h['c_X_uM']) * 1e-6, dy: float(h['c_Y_uM']) * 1e-6}
    bench = {'dyes': {dx: float(h['eps_factor_x']), dy: float(h['eps_factor_y'])}, 'sources': sources,
             'dispense_mL': [1.00, 0.10], 'cuvettes': int(params['n_cuvettes']), 'b0': float(h['b0']), 'b1': float(h['b1']),
             'start_nm': rng.randrange(400, 701), 'dispense_seats': {name: 'station' for name in sources},
             'mixer_seat': 'station', 'spectro_seat': 'spectro',
             'labels': {'stock': 'STOCK X', 'water': 'WATER', 'unknown': 'STOCK Y'}, 'scan_key': True,
             'dual_wavelength': True,
             'prefilled': {int(params['mixture_cuvette']): {'label': MIXTURE_LABEL, 'volume_mL': float(params['mixture_mL']),
                                                            'composition_M': mixture}},
             'stray': float(h['stray_light'])}
    return {'bench': bench, 'noise_seed': appearance_seed('chem-noise:' + instance['instance_id'])}


def chem_configuration(instance):
    """Reagents, dye batch, instrument baseline and the public start wavelength of one instance.

    Plain values only: the Bench builds dye spectra and colours after the simulator has started. The start
    wavelength and the noise streams are public-state randomness keyed by the opaque instance id.
    """
    import random
    if instance['recipe'] == 'chem_spectro_scan_bench':
        return mixture_configuration(instance)
    if instance['recipe'] == 'chem_spectro_dye_bench':
        return dye_configuration(instance)
    if instance['recipe'] == 'chem_spectro_kin_bench':
        return kinetics_configuration(instance)
    h, params = instance['hidden'], instance['scene_params']
    dye = params['dye']
    rng = random.Random(appearance_seed(instance['instance_id']) ^ 0xC9E1)
    sources = {'stock': {'composition_M': {dye: float(h['c_stock_uM']) * 1e-6}, 'volume_mL': float(params['stock_mL'])},
               'water': {'composition_M': {}, 'volume_mL': float(params['water_mL'])},
               'unknown': {'composition_M': {dye: float(h['c_u_uM']) * 1e-6}, 'volume_mL': float(params['unknown_mL'])}}
    bench = {'dyes': {dye: float(h['eps_factor'])}, 'sources': sources, 'dispense_mL': [1.00, 0.10],
             'cuvettes': int(params['n_cuvettes']), 'b0': float(h['b0']), 'b1': float(h['b1']),
             'start_nm': rng.randrange(400, 701), 'dispense_seats': {name: 'station' for name in sources},
             'mixer_seat': 'station', 'spectro_seat': 'spectro', 'stray': float(h['stray_light'])}
    return {'bench': bench, 'noise_seed': appearance_seed('chem-noise:' + instance['instance_id'])}


KHP_G_PER_MOL = 204.22


def weak_acid_configuration(instance):
    """titration_weak_acid: the titration_basic bench with the KHP standard (label mass in 20 mL water) in S1, the weak acid in S2 and S3, and
    NaOH labelled 'about 0.1 M' (plain values only)."""
    h, params = instance['hidden'], instance['scene_params']
    constants = instance['public_task']['public_constants']
    std = constants['standard']
    khp_M = float(h['khp_mass_g']) / KHP_G_PER_MOL / (float(params['khp_water_mL']) * 1e-3)
    beakers = {std['beaker']: {'composition_M': {'KHP': khp_M}, 'volume_mL': float(params['khp_water_mL'])}}
    for label in constants['sample']['beakers']:
        beakers[label] = {'composition_M': {'HA': float(h['c_HA_M'])}, 'volume_mL': float(h[f'V_{label}_mL'])}
    beakers.update(titration_buffers(constants))
    bench = {'beakers': beakers,
             'titrant': {'composition_M': {'NaOH': 0.1000 * float(h['naoh_factor'])}, 'volume_mL': float(params['naoh_mL'])},
             'ph_offset': float(h['ph_offset']), 'ph_slope_frac': float(h['ph_slope_frac']), 'ka': 10.0 ** -float(h['pKa']),
             'titrant_label_M': float(constants['titrant']['label_M']), 'sample_nominal_mL': float(params['sample_mL']),
             'titrant_label_text': str(constants['titrant']['label_text']),
             'khp_mass_g': float(h['khp_mass_g']), 'khp_beaker': std['beaker'],
             'sample_beakers': list(constants['sample']['beakers'])}
    return {'bench': bench, 'noise_seed': appearance_seed('titration-noise:' + instance['instance_id'])}


def titration_buffers(constants):
    """The two pH standard buffers of the titration benches (the meter is not calibrated), each in a
    beaker on its carrier: {label: {'composition_M', 'volume_mL'}}; labels B4 / B7 (nominal pH as public)."""
    from chem_titration import BUFFER_PH, buffer_composition
    public = constants['buffers']
    nominal = {'B4': float(public['low_pH']), 'B7': float(public['high_pH'])}
    if nominal != BUFFER_PH:
        raise ValueError(f'buffer pH {nominal} differs from the chemistry model {BUFFER_PH}')
    return {label: {'composition_M': buffer_composition(label), 'volume_mL': float(public['volume_mL'])}
            for label in BUFFER_PH}


def titration_configuration(instance):
    """Titrant, sample beakers, buffers and the meter's offset and slope of one titration instance (plain values
    only)."""
    if instance['recipe'] == 'chem_titration_khp_bench':
        return weak_acid_configuration(instance)
    h, params = instance['hidden'], instance['scene_params']
    constants = instance['public_task']['public_constants']
    label_M = float(constants['titrant']['label_M'])
    acid = params['sample_acid']
    beakers = {label: {'composition_M': {acid: float(h[f'c_{acid}_M'])}, 'volume_mL': float(h[f'V_{label}_mL'])}
               for label in constants['sample']['beakers'][:int(params['n_sample_beakers'])]}
    beakers.update(titration_buffers(constants))
    bench = {'beakers': beakers,
             'titrant': {'composition_M': {'NaOH': label_M * float(h['naoh_cert_factor'])}, 'volume_mL': float(params['naoh_mL'])},
             'ph_offset': float(h['ph_offset']), 'ph_slope_frac': float(h['ph_slope_frac']), 'titrant_label_M': label_M,
             'titrant_label_text': str(constants['titrant']['label_text']),
             'sample_nominal_mL': float(params['sample_mL'])}
    return {'bench': bench, 'noise_seed': appearance_seed('titration-noise:' + instance['instance_id'])}


def titration_evidence(episode_id, bench):
    """Sealed evidence of a titration bench: one record per REC record (started and ended with the meter's REC key),
    cited as <episode>:titration:<n>; private rows keep the electrode's place and the beaker it stood in."""
    private = {p['sample_id']: p for p in bench.recorder.private}
    evidence = {}
    for record in bench.recorder.records:
        if record['first'] is None:
            continue
        eid = f"{episode_id}:titration:{record['n']}"
        rows = [deepcopy(r) for r in bench.recorder.rows if record['first'] <= r['sample_id'] <= record['last']]
        public = {'record': record['n'], 'rows': rows}
        evidence[eid] = {'kind': 'titration_record', 'evidence_id': eid, 'public': public,
                         'sha256': hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest(),
                         'private_rows': [deepcopy(private[r['sample_id']]) for r in rows]}
    return evidence


def cited_ids(value):
    """Every string under an 'evidence_id' key anywhere in a submission."""
    if isinstance(value, dict):
        return {v for k, v in value.items() if k == 'evidence_id' and isinstance(v, str)} | \
            {i for v in value.values() for i in cited_ids(v)}
    if isinstance(value, list):
        return {i for v in value for i in cited_ids(v)}
    return set()


def chem_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    import base64
    from device_runtime import DeviceRuntime
    from chem_bench import ChemBench

    config = backend.configuration
    mixture = bool(config['bench'].get('prefilled'))
    kinetics = config['bench'].get('extra_key') == 'kin'

    class ChemEpisode(InstanceMixin, DeviceRuntime):
        records = None

        def public_task(self):
            return self.instance_public_task({})

        def robot_state(self):
            return super().robot_state() | {'gripper_holding': bool(self.backend.robot_state()['gripper_holding'])}

        def observation(self):
            """The three standard views with an evidence_id <episode>:obs:<n> bound to this state (kind bench_view): its
            private record keeps what the spectrophotometer screen showed (screen_view), for the readings gate."""
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
            self.records[eid] = screen_view(backend, self.tick, physics_hz) | {'kind': 'bench_view', 'evidence_id': eid,
                                                                              'images': images}
            value['evidence_id'] = eid
            return value

        def process_extra(self):
            return {'observations': len(self.records or {})}

        def seal_payload(self, cited=()):
            """Terminal state, device, evidence and files of the sealed record (also used at stop). Every observation is
            sealed as evidence; the images of the cited ones (cited: the submission's evidence ids) go with it."""
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
            sources = config['bench']['sources']
            if mixture:
                h = instance['hidden']
                (cid, fill), = config['bench']['prefilled'].items()
                device = {'c_stock_uM_label': {'X': float(h['c_stock_x_uM']), 'Y': float(h['c_stock_y_uM'])},
                          'source_of': {'X': 'stock', 'Y': 'unknown'},
                          'mixture': {'cuvette': int(cid), 'label': fill['label'], 'volume_mL': fill['volume_mL']}}
            elif config['bench'].get('dye_specs'):
                h = instance['hidden']
                device = {'c_label_uM': {'stock': float(h['c_stock1_uM']), 'unknown': float(h['c_stock2_uM'])}}
            else:
                device = {'c_stock_uM_label': float(instance['hidden']['c_stock_uM'])}
            if kinetics:
                constants = instance['public_task']['public_constants']
                device['naoh_label_M'] = float(constants['naoh']['label_M'])
                device['source_of'] = {'dye': 'stock', 'NaOH': 'unknown'}
                for run, private in zip(bench.spectro.kinetic_runs, bench.spectro.kinetic_private):
                    eid = f"{self.episode_id}:kinetics:{run['run']}"
                    public = deepcopy(run)
                    evidence[eid] = {'kind': 'kinetic_run', 'evidence_id': eid, 'public': public,
                                     'private_points': deepcopy(private['points']),
                                     'sha256': hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()}
            return (backend.seat_audit() | {'busy': bench.spectro.busy is not None},
                device | {
                 'sources': {n: {'volume_mL': s['volume_mL']} for n, s in sources.items()},
                 'dispense_mL': list(config['bench']['dispense_mL']), 'min_read_mL': 2.0, 'capacity_mL': 3.5,
                 'screen_readings': True,
                 'chemistry': bench.private_record(), 'actions': deepcopy(bench.actions)}, evidence, files)

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if not isinstance(args, dict):
                raise ValueError('arguments must be an object')
            bench = backend.bench
            if name == 'get_task_spec' and args == {}:
                return self.public_task()
            if name == 'export_spectro_scans' and args == {} and config['bench'].get('scan_key'):
                return bench.spectro.export_scans()
            if name == 'export_spectro_kinetics' and args == {} and kinetics:
                return bench.spectro.export_kinetics()
            if name in CHEM_TOOLS:
                raise ValueError({'export_spectro_scans': 'this spectrophotometer has no SCAN',
                                  'export_spectro_kinetics': 'this spectrophotometer has no KIN'}.get(name, 'unexpected arguments'))
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
        raise RuntimeError('chemistry bench failed to settle')
    audit = backend.seat_audit()
    if sorted(v for v in audit['cuvette_seat'].values() if v) != sorted(f'rack:{i}' for i in audit['cuvette_seat']):
        raise RuntimeError(f"cuvettes not seated in the rack: {audit['cuvette_seat']}")
    backend.tick_index = 0
    backend.history.clear()
    episode = ChemEpisode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    backend.bench = ChemBench(episode.episode_id, config['bench'], physics_hz=physics_hz, noise_seed=config['noise_seed'])
    return episode


def titration_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    import base64
    from device_runtime import DeviceRuntime
    from chem_titration import TitrationBench

    config = backend.configuration
    tools = ('export_titration_log',)

    class TitrationEpisode(InstanceMixin, DeviceRuntime):
        screen_views = None     # private: what the screens showed at each observation (process score)

        def public_task(self):
            return self.instance_public_task({})

        def observation(self):
            """The three standard views (unchanged); privately, what the meter and counter screens showed then and where
            the electrode stood, for the offline process score (the Agent's result carries nothing new)."""
            value = super().observation()
            if backend.bench is not None:
                if self.screen_views is None:
                    self.screen_views = []
                self.screen_views.append({'tick': self.tick, **backend.bench.screen_state()})
            return value

        def process_extra(self):
            return {'screen_views': list(self.screen_views or [])}

        def robot_state(self):
            return super().robot_state() | {'gripper_holding': bool(self.backend.robot_state()['gripper_holding'])}

        def seal_payload(self):
            """Terminal state, device, evidence and files of the sealed record (also used at stop)."""
            bench = backend.bench
            return (backend.seat_audit(),
                {'titrant_label_M': config['bench']['titrant_label_M'],
                 'sample_nominal_mL': config['bench']['sample_nominal_mL'],
                 **{k: config['bench'][k] for k in ('khp_mass_g', 'khp_beaker', 'sample_beakers') if k in config['bench']},
                 'chemistry': bench.private_record(), 'actions': deepcopy(bench.actions)},
                titration_evidence(self.episode_id, bench))

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if not isinstance(args, dict):
                raise ValueError('arguments must be an object')
            bench = backend.bench
            if name == 'get_task_spec' and args == {}:
                return self.public_task()
            if name == 'export_titration_log' and args == {}:
                return bench.recorder.export()
            if name in tools:
                raise ValueError('unexpected arguments')
            if name == 'submit_result':
                return self.freeze_and_seal(args, *self.seal_payload())
            return super()._dispatch(name, args)

        def evaluation(self):
            return self.instance_evaluation()

    # Apparatus preparation (not charged): open the gripper and let the carriers and the electrode settle.
    backend.set_gripper(.08)
    for _ in range(40 * (physics_hz // 30)):
        backend.step(None)
        backend.after_step({}, backend.tick_index)
    if not backend.at_rest():
        raise RuntimeError('titration bench failed to settle')
    audit = backend.seat_audit()
    if sorted(v for v in audit['carrier_seat'].values() if v) != sorted(f'rack:{label}' for label in audit['carrier_seat']) \
            or audit['electrode'] != 'park':
        raise RuntimeError(f"apparatus not in its starting places: {audit['carrier_seat']}, electrode {audit['electrode']}")
    backend.tick_index = 0
    backend.history.clear()
    episode = TitrationEpisode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    backend.bench = TitrationBench(episode.episode_id, config['bench'], physics_hz=physics_hz, noise_seed=config['noise_seed'])
    return episode

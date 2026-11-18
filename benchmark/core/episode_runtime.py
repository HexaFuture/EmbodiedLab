"""Development pendulum evaluation. Truth is injected only by the trusted host.

No evaluator condition depends on EmbodiedLab skills, planner labels, or tool success
claims. Parameter and prediction tolerances are provisional calibration goals.
"""
from copy import deepcopy
import hashlib
import json

from control_contract import finite, PUBLIC_TOOLS
from device_runtime import DeviceRuntime
from apparatus_manual import apparatus_description


class PendulumEpisode(DeviceRuntime):
    def __init__(self, backend, *, gravity_m_s2, calibration, **kwargs):
        if not finite(gravity_m_s2) or gravity_m_s2 <= 0:
            raise ValueError('private gravity must be finite and positive')
        super().__init__(backend, **kwargs)
        self._truth = float(gravity_m_s2)
        self.calibration = deepcopy(calibration)
        if set(self.calibration) != set(backend.pendulum_ids) or len(self.calibration) != 3:
            raise ValueError('three calibrated pendula required')
        if any(not finite(v) or v <= 0 for v in self.calibration.values()):
            raise ValueError('positive effective lengths required')
        if len(set(self.calibration.values())) != 3:
            raise ValueError('three distinct effective lengths required')
        self.prediction = None
        self.report = None

    def public_task(self):
        return {
            'schema_version': 'embodiedlab.pendulum-task/v1-dev',
            'task_id': 'pendulum_gravity_v1', 'episode_id': self.episode_id,
            'evaluation_ready': False, 'threshold_status': 'provisional_pending_physical_calibration',
            'tools': list(PUBLIC_TOOLS), 'effective_lengths_m': deepcopy(self.calibration),
            'apparatus': apparatus_description(self.calibration),
            'horizon': self.horizon_ticks//(self.limits.physics_hz//30),
            'physics_hz': self.limits.physics_hz, 'horizon_ticks': self.horizon_ticks,
            'device_rpc_limit': 65536,
            'measurement_requirements': {'fit_lengths': 2, 'held_out_lengths': 1,
                'full_cycles': 10, 'warmup_cycles': 1, 'amplitude_strictly_below_deg': 5,
                'free_timing_interval': True, 'physical_robot_excitation': True,
                'prediction_before_held_out_excitation': True},
            'provisional_tolerances': {'parameter_relative_error': .03,
                                       'held_out_prediction_relative_error': .015},
            'submission_fields': ['estimate', 'unit', 'model', 'uncertainty_statement',
                                  'measurement_evidence_ids', 'validation_evidence_id', 'prediction_id'],
            'parameter': 'gravity', 'unit': 'm/s^2',
            'control_track': 'public_joint_trajectory',
            'device_calibration': deepcopy(self.backend.public_calibration())
                if hasattr(self.backend, 'public_calibration') else {},
        }

    def process_extra(self):
        """Started gates (finished or not) and the pendulums the robot touched, for the process record: gate_states
        holds the current acquisition of each gate, gate_history every started acquisition in order (those cleared
        by RESET included), button_presses every START/RESET switch closure (tick, pendulum, button, gate state
        before, effect), gate_final each gate's timer at the end of the record (idle = reset, display at zero),
        gate_events the gates' state-machine log (DeviceRuntime._gate_event); peak_deg is the largest |angle| of the
        pendulum from the START press to the end of the acquisition (or now)."""
        def state(gate):
            reading = gate['timer'].reading()
            return {'start_tick': gate['start_tick'], 'status': reading.status, 'cycles': reading.cycles,
                    'elapsed_s': reading.elapsed_s, 'timing_started': gate['timing_started'],
                    'invalid_reasons': sorted(gate['invalid_reasons']),
                    'contact_excitation': gate['contact_excitation'], 'evidence_id': gate.get('evidence_id'),
                    'peak_deg': round(gate.get('peak_deg', 0.0), 4)}
        history = [dict(state(g), pendulum_id=k) for k, g in getattr(self, 'gate_history', [])]
        history += [dict(state(g), pendulum_id=k) for k, g in self.gates.items()]
        final = {key: {'status': 'idle', 'cycles': 0, 'elapsed_s': 0.0} for key in sorted(self.calibration)}
        for key, gate in self.gates.items():
            final[key] = {k: state(gate)[k] for k in ('status', 'cycles', 'elapsed_s')}
        return {'gate_states': {key: state(gate) for key, gate in self.gates.items()},
                'gate_history': sorted(history, key=lambda g: g['start_tick']), 'touched': sorted(self.touched),
                'button_presses': list(getattr(self, 'button_presses', [])), 'gate_final': final,
                'gate_events': list(getattr(self, 'gate_events', []))}

    def _dispatch(self, name, args):
        if self.terminated:
            raise ValueError('episode terminated')
        if name == 'get_task_spec' and args == {}:
            return self.public_task()
        if name == 'record_prediction':
            if self.prediction is not None:
                raise ValueError('prediction already frozen')
            if not isinstance(args, dict) or set(args) != {'pendulum_id', 'period_s', 'measurement_evidence_ids'}:
                raise ValueError('invalid prediction schema')
            key = args['pendulum_id']
            if (key not in self.calibration or key in self.touched or key in self.gates
                    or any(k == key for k, _ in self.gate_history)):
                raise ValueError('validation pendulum already excited or instrumented')
            if not finite(args['period_s']) or args['period_s'] <= 0:
                raise ValueError('prediction period must be positive seconds')
            ids = args['measurement_evidence_ids']
            records = self._measurement_records(ids, count=2)
            if key in {r['pendulum_id'] for r in records}:
                raise ValueError('validation length was used for fitting')
            self.prediction = deepcopy(args) | {'prediction_id': f'{self.episode_id}:prediction:1',
                                                'physics_tick': self.tick}
            self.revision += 1
            return deepcopy(self.prediction)
        if name == 'submit_result':
            return self._submit(args)
        return super()._dispatch(name, args)

    def _measurement_records(self, ids, count):
        if not isinstance(ids, list) or len(ids) != count or any(not isinstance(eid, str) for eid in ids):
            raise ValueError('wrong measurement evidence count')
        if len(set(ids)) != count or any(eid not in self.evidence for eid in ids):
            raise ValueError('duplicate, unknown or cross-episode evidence')
        records = [self.evidence[eid] for eid in ids]
        if len({r['pendulum_id'] for r in records}) != count:
            raise ValueError('distinct measured lengths required')
        for record in records:
            if record['public']['status'] != 'complete' or not 8 <= record['public']['cycles'] <= 10:
                raise ValueError('incomplete optical-gate measurement')
            if record['invalid_reasons'] or not record['contact_excitation']:
                raise ValueError('measurement failed physical validity')
        return records

    def _submit(self, value):
        # First submission freezes the episode even when its contents are invalid.
        self.terminated = True
        gates = dict(valid_submission=False, measurement_evidence_valid=False,
                     held_out_prediction_valid=False, within_budget=self.tick <= self.horizon_ticks)
        parameter_error = prediction_error = None
        estimate = None
        reason = None
        try:
            fields = set(self.public_task()['submission_fields'])
            if not isinstance(value, dict) or set(value) != fields:
                raise ValueError('submission must match public schema')
            estimate = value['estimate']
            if not finite(estimate) or estimate <= 0 or value['unit'] != 'm/s^2':
                raise ValueError('positive finite estimate and unit m/s^2 required')
            for key in ('model', 'uncertainty_statement'):
                if not isinstance(value[key], str) or not 1 <= len(value[key].strip()) <= 4000:
                    raise ValueError('bounded nonempty analysis statements required')
            gates['valid_submission'] = True
            self._measurement_records(value['measurement_evidence_ids'], 2)
            validation = self._measurement_records([value['validation_evidence_id']], 1)[0]
            if value['validation_evidence_id'] in value['measurement_evidence_ids']:
                raise ValueError('validation evidence cannot also be fit evidence')
            gates['measurement_evidence_valid'] = True
            prediction = self.prediction
            if prediction is None or value['prediction_id'] != prediction['prediction_id']:
                raise ValueError('missing or foreign frozen prediction')
            if set(value['measurement_evidence_ids']) != set(prediction['measurement_evidence_ids']):
                raise ValueError('fit evidence changed after prediction')
            if validation['pendulum_id'] != prediction['pendulum_id'] or validation['start_tick'] < prediction['physics_tick']:
                raise ValueError('validation does not match prior prediction')
            gates['held_out_prediction_valid'] = True
            measured_period = validation['public']['elapsed_s']/validation['public']['cycles']
            prediction_error = abs(prediction['period_s']-measured_period)/measured_period
            parameter_error = abs(estimate-self._truth)/self._truth
        except (ValueError, TypeError, KeyError) as exc:
            reason = str(exc)
        valid = all(gates.values())
        reward = max(0., 1-parameter_error) if valid else 0.
        self.report = {
            'schema_version': 'embodiedlab.evaluation/v1',
            'episode_id': self.episode_id, 'task_id': 'pendulum_gravity_v1',
            'evaluator': 'embodiedlab-pendulum-gravity-v1-dev', 'terminated': True,
            'task_completed': bool(valid and parameter_error <= .03 and prediction_error <= .015),
            'reward': reward, 'score_percent': 100*reward,
            'parameter_relative_error': parameter_error, 'held_out_prediction_error': prediction_error,
            'gates': gates, 'error': reason,
            'sim_step': self.tick//(self.limits.physics_hz//30),
            'simulated_seconds': self.tick/self.limits.physics_hz,
            'tool_calls': len(self.requests)+1, 'final_state_id': self.state_id,
            'evaluation_ready': False, 'threshold_status': 'provisional_pending_physical_calibration',
            'evidence_sha256': hashlib.sha256(json.dumps(
                {eid: r['sha256'] for eid, r in self.evidence.items()}, sort_keys=True).encode()).hexdigest(),
        }
        # Agent receives the receipt. Only management reads the final report.
        return {'submitted': True, 'terminated': True, 'episode_id': self.episode_id}

    def evaluation(self):
        if self.report is not None:
            return deepcopy(self.report)
        return {'reward': 0., 'task_completed': False, 'sim_step': self.tick//(self.limits.physics_hz//30),
                'evaluator': 'embodiedlab-pendulum-gravity-v1-dev', 'terminated': self.terminated,
                'error': self.termination_reason,
                'infrastructure_failure': self.termination_reason == 'physics_backend_failure'}

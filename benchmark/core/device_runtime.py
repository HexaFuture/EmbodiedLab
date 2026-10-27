"""Participant-neutral device API over a trusted, single-thread physics backend.

Backend.step advances exactly one fixed physics tick and returns private
contact/angle audits, binary ray occlusions and the switch state of each gate's
START and RESET push buttons. No wall-clock polling runs physics. Request
retries return the original result without repeating actions.

Gate timers (one per pendulum) are operated only by their physical buttons:
idle -> (START) acquisition: waiting for the first crossing, warm-up cycle,
TIMED_CYCLES timed cycles -> complete | invalid. START acts only on an idle
gate. RESET returns the gate to idle (display zero); an acquisition still
running ends invalid (`reset_during_timing`). Every finished acquisition is
immutable private evidence from the tick it finishes. No tool reads or exports a gate (export_gate
removed): its timer face is read in the camera images.

For the process record every gate logs its state machine (gate_events, also sent to the backend's private event
log): button presses with their effect, the start of an acquisition, each status change and completed cycle,
each validity reason when it first applies, and the finished acquisition with its peak swing amplitude.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import uuid

from control_contract import (PROTECTIVE_STOP_BRAKE_S, PROTECTIVE_STOP_REST_RAD_S, PROTECTIVE_STOP_REST_S, ControlLimits,
                              SimDiverged, duration_ticks, finite, validate_trajectory, validate_velocities)
from scripts.pendulum_instruments import GateConfig, OpticalGateTimer, TIMED_CYCLES
from observation import public_frame


class DeviceRuntime:
    def __init__(self, backend, *, horizon_ticks=144000, physics_hz=240):
        if type(horizon_ticks) is not int or horizon_ticks <= 0:
            raise ValueError('horizon_ticks must be a positive integer')
        self.backend = backend
        self.limits = ControlLimits(physics_hz)
        self.horizon_ticks = horizon_ticks
        self.episode_id = uuid.uuid4().hex
        self.tick = 0
        self.revision = 0
        self.terminated = False
        self.termination_reason = None
        self.termination_detail = None
        self.call_log = []          # (name, tick before, tick after, ok, error type), kept by the Bench
        self.requests = {}
        self.gates = {}                 # pendulum -> its current acquisition (absent: the gate is idle)
        self.gate_history = []          # (pendulum, acquisition) cleared from its gate by RESET, in order
        self.button_presses = []        # every switch closure: tick, pendulum, button, gate state before, effect
        self._switches = {}             # (pendulum, button) -> switch closed on the previous tick
        self.gate_events = []           # gate state machine log for the process record (see the module docstring)
        self.evidence = {}
        self.touched = set()

    @property
    def state_id(self):
        return f'{self.episode_id}:{self.revision}'

    def robot_state(self):
        value = self.backend.robot_state()
        # Explicit allowlist: never return body poses, USD paths or private g.
        keys = ('joint_positions_rad', 'gripper_width_m')
        return {key: deepcopy(value[key]) for key in keys} | {
                    'episode_id': self.episode_id, 'state_id': self.state_id,
                    'physics_tick': self.tick, 'physics_hz': self.limits.physics_hz,
                    'sim_time_s': self.tick/self.limits.physics_hz}

    def observation(self):
        frames = self.backend.observation()
        if not isinstance(frames, list) or not 1 <= len(frames) <= 4:
            raise RuntimeError('camera frames unavailable')
        public = [public_frame(frame, episode_id=self.episode_id,
                    state_id=self.state_id, index=index) for index, frame in enumerate(frames)]
        return {'schema_version': 'embodiedlab.observation/v1',
                'episode_id': self.episode_id, 'state_id': self.state_id,
                'physics_tick': self.tick, 'sim_time_s': self.tick/self.limits.physics_hz,
                'robot': self.robot_state(), 'frames': public}

    def _record_failure(self, where):
        """The full traceback behind termination_detail, kept private: the container log (stderr) and the process
        record's bench_logs (backend_failure). The agent sees only the short termination_detail."""
        import sys
        import traceback
        text = traceback.format_exc()
        self.failure_traceback = {'tick': self.tick, 'where': where, 'traceback': text}
        print(f'EMBODIEDLAB_BACKEND_FAILURE tick={self.tick} in {where}\n{text}', file=sys.stderr, flush=True)

    def _step(self, target=None, velocity=None):
        stop = self._physics_tick(target, velocity)
        if stop is not None and not getattr(self, '_braking', False):
            # Protective stop: the arm brakes to rest, then the running command ends with this error. The episode
            # continues.
            self._brake_to_rest()
            raise stop

    def _brake_to_rest(self):
        """After a protective stop: hold the measured pose of the stop tick (fixed, no feed-forward) and keep stepping
        until every arm joint moves slower than PROTECTIVE_STOP_REST_RAD_S for PROTECTIVE_STOP_REST_S, for at most
        PROTECTIVE_STOP_BRAKE_S or the rest of the budget. Every tick is an ordinary tick through _step (charged,
        recorded, gripper and hold as before); no protective stop is raised while braking. Without this the arm kept
        its stop-tick velocity into the next command, which then stopped on its first tick again (47% of
        stops repeated)."""
        hold = list(self.backend.robot_state()['joint_positions_rad'])
        still = (0.0,) * len(hold)
        last, quiet = hold, 0
        # At rest = slower than the limit on PROTECTIVE_STOP_REST_S of consecutive ticks: a joint that chatters in
        # contact passes zero speed for a single tick without being at rest.
        rest_ticks = max(1, round(PROTECTIVE_STOP_REST_S * self.limits.physics_hz))
        self._braking = True
        try:
            for _ in range(round(PROTECTIVE_STOP_BRAKE_S * self.limits.physics_hz)):
                if self.tick >= self.horizon_ticks:
                    break
                self._step(hold, still)
                q = self.backend.robot_state()['joint_positions_rad']
                slow = max(abs(a - b) for a, b in zip(q, last)) * self.limits.physics_hz < PROTECTIVE_STOP_REST_RAD_S
                quiet = quiet + 1 if slow else 0
                last = q
                if quiet >= rest_ticks:
                    break
        finally:
            self._braking = False

    def _physics_tick(self, target=None, velocity=None):
        """One physics tick; returns the backend's protective stop for this tick (None without one)."""
        if self.tick >= self.horizon_ticks:
            self.terminated = True
            self.termination_reason = self.termination_reason or 'sim_budget_exhausted'
            raise ValueError('episode tick budget exhausted')
        # Charge before step, including a step that raises after physical work.
        self.backend.pending_protective_stop = None
        self.tick += 1
        self.revision += 1
        try:
            sample = self.backend.step(target) if velocity is None else self.backend.step(target, velocity=velocity)
            if not isinstance(sample, dict) or set(sample) != set(self.backend.pendulum_ids):
                raise RuntimeError('physics tick omitted a pendulum sample')
            for physical in sample.values():
                if not isinstance(physical, dict) or any(type(physical.get(key)) is not bool for key in
                        ('blocked', 'robot_contact', 'any_contact', 'unexpected_occluder')):
                    raise RuntimeError('physics tick returned invalid contact or gate data')
                if not finite(physical.get('angle_deg')):
                    if isinstance(physical.get('angle_deg'), float):     # NaN or infinite: the physics diverged
                        raise SimDiverged('physics tick returned a non-finite pendulum angle')
                    raise RuntimeError('physics tick returned invalid angle data')
                if any(type(physical.get(key, False)) is not bool for key in ('start_button', 'reset_button')):
                    raise RuntimeError('physics tick returned invalid button data')
        except Exception as exc:
            self.terminated = True
            self.termination_reason = 'sim_diverged' if isinstance(exc, SimDiverged) else 'physics_backend_failure'
            self.termination_detail = f'{type(exc).__name__}: {exc}'
            self._record_failure('step')
            raise
        for key, physical in sample.items():
            if physical['robot_contact']:    # a swing started by any part of the robot
                self.touched.add(key)
            if key not in self.gates:
                continue
            gate = self.gates[key]
            if gate['timer'].reading().status in ('complete', 'invalid'):
                continue
            reading = gate['timer'].update(self.tick/self.limits.physics_hz, physical['blocked'])
            reasons = set(gate['invalid_reasons'])
            if not finite(physical['angle_deg']) or abs(physical['angle_deg']) >= 5:
                gate['invalid_reasons'].add('amplitude_not_below_5_degrees')
            if finite(physical['angle_deg']):
                gate['peak_deg'] = max(gate['peak_deg'], abs(physical['angle_deg']))
            if physical['unexpected_occluder']:
                gate['invalid_reasons'].add('unexpected_gate_occluder')
            # Warmup permits withdrawal. All counted timing ticks must be free.
            if gate['timing_started'] or reading.status in ('timing', 'complete'):
                gate['timing_started'] = True
                if physical['any_contact'] or gate['previous_contact']:
                    gate['invalid_reasons'].add('contact_during_timing')
            gate['previous_contact'] = physical['any_contact']
            for reason in sorted(gate['invalid_reasons'] - reasons):
                self._gate_event('gate_invalid_reason', key, reason=reason)
            if reading.status != gate['last_status'] or reading.cycles != gate['last_cycles']:
                self._gate_event('gate_status', key, status=reading.status, cycles=reading.cycles,
                                 instrument_error=reading.error)
                gate['last_status'], gate['last_cycles'] = reading.status, reading.cycles
            if reading.status == 'complete':
                gate['contact_excitation'] = key in self.touched
            if reading.status in ('complete', 'invalid'):
                self._export_gate(key)          # evidence from the tick the acquisition finishes
        # Buttons act on the closing edge of their switch, after this tick's timing: a START begins counting
        # with the next tick. RESET before START when both close on the same tick.
        for key, physical in sample.items():
            for button in ('reset', 'start'):
                closed = physical.get(button + '_button', False)
                if closed and not self._switches.get((key, button), False):
                    self._press(key, button)
                self._switches[(key, button)] = closed
        after_step = getattr(self.backend, 'after_step', None)
        if after_step is not None:
            try:
                after_step({key: gate['timer'] for key, gate in self.gates.items()}, self.tick)
            except Exception as exc:
                self.terminated = True
                self.termination_reason = 'physics_backend_failure'
                self.termination_detail = f'{type(exc).__name__}: {exc}'
                self._record_failure('after_step')
                raise
        stop = getattr(self.backend, 'pending_protective_stop', None)
        if stop is not None:
            # The tick completed with the arm held at its measured pose; _step brakes it to rest and ends the
            # running command with this error. The episode continues.
            self.backend.pending_protective_stop = None
        return stop

    def call(self, request_id, name, arguments):
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise ValueError('bounded request_id required')
        # This string is only a retry key, never an RPC response. Preserve
        # non-finite inputs here so submit_result can reject AND lock them.
        fingerprint = hashlib.sha256(json.dumps([name, arguments], sort_keys=True,
                                                allow_nan=True).encode()).hexdigest()
        if request_id in self.requests:
            previous, result = self.requests[request_id]
            if fingerprint != previous:
                raise ValueError('request_id reused with different arguments')
            return deepcopy(result)
        # This is an RPC replay-cache bound, not an Agent step budget. A basic
        # 30 Hz feedback controller needs 18,000 packets over a 600 s episode.
        # The old 2,000 cap accidentally penalized moving control into an SDK.
        if len(self.requests) >= 65536:
            self.terminated = True
            self.termination_reason = self.termination_reason or 'request_budget_exhausted'
            raise ValueError('tool call budget exhausted')
        tick_before = self.tick
        try:
            result = self._dispatch(name, arguments)
            response = {'ok': True, 'result': result}
        except Exception as exc:
            response = {'ok': False, 'error': type(exc).__name__, 'message': str(exc),
                        'state_id': self.state_id, 'physics_tick': self.tick,
                        'terminated': self.terminated}
        self.call_log.append((name, tick_before, self.tick, response['ok'], response.get('error')))
        self.requests[request_id] = fingerprint, deepcopy(response)
        return response

    def _dispatch(self, name, args):
        if not isinstance(args, dict):
            raise ValueError('arguments must be an object')
        if self.terminated:
            raise ValueError('episode terminated')
        if name == 'get_robot_state' and not args:
            return self.robot_state()
        if name == 'get_observation' and not args:
            return self.observation()
        if name == 'wait' and set(args) == {'seconds'}:
            count = duration_ticks(args['seconds'], self.horizon_ticks-self.tick, self.limits)
            for _ in range(count):
                self._step()
            return self.robot_state()
        if name == 'execute_joint_trajectory' and set(args) in (
                {'expected_state_id', 'targets_rad'},
                {'expected_state_id', 'targets_rad', 'velocities_rad_s'}):
            if args['expected_state_id'] != self.state_id:
                raise ValueError('stale robot state; observe and replan')
            targets = validate_trajectory(args['targets_rad'], self.backend.robot_state()['joint_positions_rad'],
                                          self.horizon_ticks-self.tick, self.limits)
            if 'velocities_rad_s' in args:
                velocities = validate_velocities(args['velocities_rad_s'], len(targets), self.limits)
            else:
                # No feed-forward given: none on the first tick, then the backend's finite difference between
                # consecutive waypoints. The first waypoint usually differs from the previous command by the
                # arm's gravity sag; (first - previous)/dt there was a one-tick velocity spike.
                velocities = [(0.0,)*7] + [None]*(len(targets)-1)
            for target, velocity in zip(targets, velocities):
                self._step(target, velocity)
            return self.robot_state()
        if name == 'set_gripper' and set(args) == {'width_m', 'expected_state_id'}:
            width = args['width_m']
            if not finite(width) or not 0 <= width <= .08 or args['expected_state_id'] != self.state_id:
                raise ValueError('invalid gripper command or stale state')
            self.backend.set_gripper(width)
            self.revision += 1
            return self.robot_state()
        raise ValueError('unknown device command or unexpected arguments')

    def _press(self, key, button):
        """A button's switch closed: START starts an acquisition on an idle gate; RESET clears the gate."""
        gate = self.gates.get(key)
        before = gate['timer'].reading().status if gate else 'idle'
        if button == 'start':
            effect = 'no_effect'
            if gate is None:
                timer = OpticalGateTimer(GateConfig(target_cycles=TIMED_CYCLES,
                           max_sample_gap_s=1/self.limits.physics_hz + 1e-9))
                timer.reset(self.tick/self.limits.physics_hz)
                self.gates[key] = dict(timer=timer, start_tick=self.tick, timing_started=False,
                                       invalid_reasons=set(), contact_excitation=False, previous_contact=False,
                                       peak_deg=0.0, last_status=timer.reading().status, last_cycles=0)
                effect = 'started'
        elif gate is None:
            effect = 'no_effect'
        else:
            effect = 'cleared'
            if before not in ('complete', 'invalid'):
                gate['invalid_reasons'].add('reset_during_timing')
                gate['timer'].abort('reset_during_timing')
                self._export_gate(key)
                effect = 'ended_invalid'
            self.gate_history.append((key, self.gates.pop(key)))
        self.button_presses.append({'tick': self.tick, 'pendulum_id': key, 'button': button,
                                    'gate_before': before, 'effect': effect})
        self._gate_event('button_press', key, button=button, gate_before=before, effect=effect)

    def _gate_event(self, event, key, **data):
        """One row of the gate log (process record), also written to the backend's private event file."""
        row = {'tick': self.tick, 'pendulum_id': key, 'event': event, **data}
        self.gate_events.append(row)
        log = getattr(self.backend, 'log_event', None)
        if log is not None:
            log(event, **{k: v for k, v in row.items() if k != 'event'})

    def _export_gate(self, key):
        """The finished current acquisition of `key` as immutable evidence (made once; later calls return the same)."""
        gate = self.gates[key]
        if 'evidence_id' not in gate:
            value = self._gate_reading(key, gate)
            eid = f'{self.episode_id}:gate:{len(self.evidence)+1}'
            public = dict(value, evidence_id=eid, episode_id=self.episode_id)
            self.evidence[eid] = {
                'public': public, 'pendulum_id': key,
                'instrument_error': gate['timer'].reading().error,     # private: the face shows only ERR
                'contact_excitation': gate['contact_excitation'],
                'invalid_reasons': sorted(gate['invalid_reasons']),
                'start_tick': gate['start_tick'], 'end_tick': self.tick,
                'peak_amplitude_deg': round(gate.get('peak_deg', 0.0), 4),
                'sha256': hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()}
            gate['evidence_id'] = eid
            self._gate_event('gate_finished', key, status=value['status'], cycles=value['cycles'],
                             elapsed_s=value['elapsed_s'], instrument_error=self.evidence[eid]['instrument_error'],
                             invalid_reasons=sorted(gate['invalid_reasons']), start_tick=gate['start_tick'],
                             peak_amplitude_deg=round(gate.get('peak_deg', 0.0), 4), evidence_id=eid)
        return deepcopy(self.evidence[gate['evidence_id']]['public'])

    def _previous_ids(self, key):
        return [gate['evidence_id'] for k, gate in self.gate_history if k == key and 'evidence_id' in gate]

    def _gate_reading(self, key, gate):
        reading = gate['timer'].reading()
        audit = gate['timer'].audit()
        return {'pendulum_id': key, 'physics_tick': self.tick,
                'status': reading.status, 'cycles': reading.cycles,
                'elapsed_s': float(reading.elapsed_text),
                'rising_edge_times_s': [round(t, 3) for t in audit['rising_edge_times_s']],
                'time_resolution_s': .001,
                'edge_uncertainty_bound_s': .5/self.limits.physics_hz + .0005}

"""Calorimetry episodes: configuration of one instance and the device runtime with the recorder tool.

Tool: export_temperature_log — the recorded segments so far (calor_bench: a record runs from one press of the REC
cap to the next; 1 Hz: the water thermometer T_C, 0.01 C, and the heater's block thermometer heater_C, 0.1 C), each
with its record id <episode>:thermo_rec:<n> and its CSV with SHA-256. It does not advance time. Everything else is
read from the bench's displays in the camera images (the water thermometer's LCD with the REC mark and the record
number, the heater's screen, the stirrer lamp); the heater's UP/DN caps, the STIR cap and the REC cap are pressed with
the arm. Sealed evidence: every exported record as its latest export returned it (kind thermo_record), with the
private state at each of its rows.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

CALOR_TOOLS = ('export_temperature_log',)
CONTACT_REFERENCE_KG = 0.030      # the kit's compliant foot contact was tuned for a 30 g part (chem_apparatus)


def calor_configuration(instance):
    """Plain values of one instance: the thermal parameters (true values) and what the apparatus shows."""
    from instance_episodes import appearance_seed
    h = instance['hidden']
    thermal = {'m_s_kg': float(h['m_s_g']) / 1000.0, 'c_s': float(h['c_s_J_kgK']), 'm_w_kg': float(h['m_w_g']) / 1000.0,
               'c_cup_j_k': float(h['c_cup_J_K']), 't_room_c': float(h['t_room_C']),
               'h_air_w_k': float(h['h_air_W_K']), 'k_loss_w_k': float(h['k_loss_W_K']),
               'tau_sample_s': float(h['tau_sample_s']), 'tau_cool_s': float(h['tau_cool_s']),
               't_water0_c': float(h['t_room_C']) - float(h['water_below_room_K']), 't_sample0_c': float(h['t_room_C'])}
    bench = {'water_g': float(h['m_w_g']), 'sample_g': float(h['m_s_g']), 'cup_j_k': float(h['c_cup_J_K']),
             'sample_kg': float(h['m_s_g']) / 1000.0}
    return {'thermal': thermal, 'bench': bench, 'noise_seed': appearance_seed('calor-noise:' + instance['instance_id'])}


def calor_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    from calor_bench import CalorimetryBench
    from device_runtime import DeviceRuntime
    from instance_episodes import InstanceMixin

    config = backend.configuration

    class CalorEpisode(InstanceMixin, DeviceRuntime):
        def public_task(self):
            return self.instance_public_task({})

        def robot_state(self):
            return super().robot_state() | {'gripper_holding': bool(self.backend.robot_state()['gripper_holding'])}

        def seal_payload(self):
            """Terminal state, device, evidence and files of the sealed record (also used at stop)."""
            bench = backend.bench
            evidence = {}
            for eid, (record, tick) in bench.exports.items():
                public = deepcopy(record)
                n = public['record']
                evidence[eid] = {'kind': 'thermo_record', 'evidence_id': eid, 'public': public, 'export_tick': tick,
                                 'private': deepcopy(bench.records[n - 1]['private'][:len(public['rows'])]),
                                 'sha256': hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()}
            device = {'thermal': deepcopy(config['thermal']), 'transitions': deepcopy(bench.transitions),
                      'actions': deepcopy(bench.actions), 'display_hz': 1, 'rows': len(bench.rows),
                      'records': [{'record': r['n'], 'started_s': r['started_s'], 'ended_s': r['ended_s'],
                                   'rows': len(r['rows'])} for r in bench.records]}
            return (backend.seat_audit() | bench.state(), device, evidence)

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if not isinstance(args, dict):
                raise ValueError('arguments must be an object')
            bench = backend.bench
            if name == 'get_task_spec' and args == {}:
                return self.public_task()
            if name == 'export_temperature_log' and args == {}:
                return bench.export()
            if name in CALOR_TOOLS:
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
        raise RuntimeError('calorimetry bench failed to settle')
    backend.tick_index = 0
    backend.history.clear()
    episode = CalorEpisode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    backend.bench = CalorimetryBench(episode.episode_id, config, physics_hz=physics_hz, noise_seed=config['noise_seed'])
    return episode

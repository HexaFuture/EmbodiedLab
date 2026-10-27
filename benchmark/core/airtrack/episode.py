"""airtrack_collision participant-neutral runtime and sealed evidence; no online scoring.

The bench is operated by hand only (start catches, launcher knob, release lever, recorder keypad). The only device
tools beyond the robot are export_recording (the gate timers' memory of a run) and submit_result.
"""
from device_runtime import DeviceRuntime
from instance_episodes import InstanceMixin
from .recorder import GateRecorder
from .measurement import BEAM_SPACING_M, FLAG_LENGTH_M
from . import mechanism as M

TOOLS = ('get_task_spec', 'get_observation', 'get_robot_state', 'execute_joint_trajectory', 'set_gripper', 'wait',
         'export_recording', 'submit_result')


class AirtrackEpisode(InstanceMixin, DeviceRuntime):
    def __init__(self, backend, instance, output, **kwargs):
        super().__init__(backend, **kwargs)
        self._instance_init(instance, output)
        self.recorder = GateRecorder(self.episode_id, self.limits.physics_hz)
        backend.recorder = self.recorder

    def public_task(self):
        """What is on the bench and how it is used; no positions, layout or sizes (the cameras are calibrated)."""
        return self.instance_public_task({
            'apparatus': {
                'sign_convention': 'velocities are positive away from the launcher, negative towards it',
                'gliders': {'A': {'colour': 'blue', 'mass_kg': .2}, 'B': {'colour': 'red'}},
                'start_catches': 'one spring catch at each glider\'s start mark (labelled A, B); with the release '
                                 'lever at HOLD a glider brought to rest on its mark is caught and held there; moving '
                                 'the lever to RELEASE lifts both catches',
                'launcher': 'spring launcher (LAUNCHER) whose plunger strikes A when fired; pushing its yellow knob '
                            'towards the housing cocks it into notch 1, 2 or 3 (scale 0-3 beside the knob); RELEASE '
                            'fires it',
                'release_lever': {'detents': list(M.LEVER_LABELS), 'operation': 'push the yellow post along its slot'},
                'bumpers': 'absorbing: the launcher plunger face and a black rubber bumper',
                'air_supply': 'ideal air-bearing support; the gas-box dial is schematic and gives no reading'},
            'instrument': {
                'gates': {'1': 'double-beam photogate labelled 1', '2': 'double-beam photogate labelled 2'},
                'beam_spacing_m': BEAM_SPACING_M, 'time_resolution_ms': 0.01,
                'direction': '+ for a flag moving away from the launcher, - towards it',
                'recorder': {'keypad': 'digits, CLR, REC', 'record_s_range': [5, 60], 'start': 'release lever'},
                'clock': 'simulation_time'},
            'mass_A_kg': .2})

    def robot_state(self):
        value = super().robot_state()
        s = self.backend.robot_state()
        return value | {'gripper_holding': s['gripper_holding']}

    def _dispatch(self, name, args):
        if self.terminated:
            raise ValueError('episode terminated')
        if name not in TOOLS or not isinstance(args, dict):
            raise ValueError('unknown public airtrack command')
        if name == 'get_task_spec' and not args:
            return self.public_task()
        if name == 'export_recording' and set(args) == {'evidence_id'}:
            return self.recorder.export(args['evidence_id'])
        if name == 'submit_result':
            return self.freeze_and_seal(args, *self.seal_payload())
        return super()._dispatch(name, args)

    def seal_payload(self):
        """Terminal state, device and evidence of the sealed record (also used at stop)."""
        return (self.backend.bench_state('seal'),
                {'mass_A_kg': .2, 'beam_spacing_m': BEAM_SPACING_M, 'flag_length_m': FLAG_LENGTH_M,
                 'timer_quantum_s': 1e-5},
                self.recorder.evidence())

    def process_extra(self):
        """Run ticks for the process record: [start, end] per run (end None while recording)."""
        return {'recordings': self.recorder.ticks()}

    def evaluation(self):
        return self.instance_evaluation()

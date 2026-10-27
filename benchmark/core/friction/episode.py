"""Public device interface of the incline bench.

The ramp, the gate and the recorder are operated by hand (console buttons, recorder keypad); export_recording is the
only instrument call."""
from device_runtime import DeviceRuntime
from .bench import GATE_SPEED_M_S, RECORD_S_RANGE

TOOLS=('get_task_spec','get_observation','get_robot_state','execute_joint_trajectory',
       'set_gripper','wait','export_recording')


class FrictionDevelopmentEpisode(DeviceRuntime):
    def public_task(self):
        return dict(task_id='incline_friction',development_only=True,
            acceptance_status='unvalidated',tools=list(TOOLS),physics_hz=self.limits.physics_hz,
            horizon_ticks=self.horizon_ticks,episode_id=self.episode_id,
            device_calibration=self.backend.public_calibration(),
            # Identity and operation only; positions, layout and sizes are found from the calibrated cameras.
            apparatus=dict(sliders='two sliders, A and B: each a green plate with a magenta square handle on top, its plate '
                                   'top printed "A" or "B"; each starts on its own white loading pad',
                handle_color='magenta',
                angle_range_deg=[0.,35.],tilt_rate_deg_s=.5,gate_retraction_speed_m_s=GATE_SPEED_M_S,
                controls='Push buttons: green raises the ramp, orange holds it, blue lowers it; white toggles the retractable gate. '
                         'Recorder keypad (keys printed 0-9, CLR, REC; REC is red): digits key the record length in '
                         'seconds, CLR clears, REC starts the record. The recorder display on its post shows the state.',
                record_length_s=list(RECORD_S_RANGE),
                gravity_m_s2=9.81))

    def robot_state(self):
        return super().robot_state()|{'gripper_holding':self.backend.held is not None}

    def _dispatch(self,name,args):
        if self.terminated:raise ValueError('episode terminated')
        if name not in TOOLS or not isinstance(args,dict):raise ValueError('unknown incline tool')
        if name=='get_task_spec' and not args:return self.public_task()
        if name=='export_recording' and set(args)=={'evidence_id'}:
            return self.backend.recorder.export(self.record_id(args['evidence_id']))
        return super()._dispatch(name,args)

    def record_id(self,ref):
        """A finished record by its evidence_id, its number (1, 2, ...) or "latest" (as the coupled bench)."""
        recorder=self.backend.recorder
        if ref=='latest':
            if not recorder.records:raise ValueError('no finished record yet')
            return list(recorder.records)[-1]
        if (isinstance(ref,int) and not isinstance(ref,bool)) or (isinstance(ref,str) and ref.isdigit()):
            return recorder.evidence_id(int(ref))
        if not isinstance(ref,str):raise ValueError('evidence_id must be an evidence id, a record number or "latest"')
        return ref

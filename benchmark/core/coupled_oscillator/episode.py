"""Coupled scientific evidence over neutral basic robot/device controls."""
import math
from copy import deepcopy
from device_runtime import DeviceRuntime
from control_contract import finite
from .recorder import DisplacementRecorder

TASK='coupled_oscillator_modes_v1'
# The bench itself is operated by hand (catches, CLAMP buttons, release post, recorder keypad, damper lever);
# export_recording is the only instrument call.
TOOLS=('get_task_spec','get_observation','get_robot_state','execute_joint_trajectory',
       'set_gripper','wait','export_recording','submit_result')


class CoupledEpisode(DeviceRuntime):
    def __init__(self,backend,**kwargs):
        super().__init__(backend,**kwargs)
        self.recorder=DisplacementRecorder(self.episode_id,self.limits.physics_hz)
        backend.recorder=self.recorder
        self.report=None

    def public_task(self):
        return dict(schema_version='embodiedlab.coupled-task/v1-dev',task_id=TASK,episode_id=self.episode_id,
            evaluation_ready=False,threshold_status='provisional_pending_physical_calibration',
            tools=list(TOOLS),horizon=self.horizon_ticks//(self.limits.physics_hz//30),
            physics_hz=self.limits.physics_hz,horizon_ticks=self.horizon_ticks,device_rpc_limit=65536,
            device_calibration=self.backend.public_calibration(),
            objective='Measure both undamped normal-mode angular frequencies of the coupled two-slider apparatus, in ascending order, using real manipulation and public encoder records.',
            apparatus=dict(channels=['CH1','CH2'],
                controls='Each slider hooks to a catch carriage with a CLAMP push button; the record length (40-180 s) is keyed on the recorder keypad, then REC, and the recorder shows its state on its display; an eddy-current damper lever has OFF and ON detents.',
                release='The yellow release post slides in one direction against a spring. With the recorder READY (REC keyed), sustained physical grip and at least 35 mm bar travel opens both hooks and starts sampling; otherwise the post moves but nothing fires. The spring returns the bar after release.',
                measurement='Both channels are quantized simulated linear encoders, not optical image-derived readings. Display and CSV share one buffer. No true stiffness, frequencies or velocities are exposed.'),
            instrument=dict(sample_hz=120,resolution_mm=.1,range_mm=[-100,100],clock='simulation_time'),
            model_convention='Linear coupled oscillators with equal masses and proportional damping acceleration -2*gamma*v. Local uncoupled frequencies may differ. Fit both channels jointly; undamped omega=sqrt(damped_omega^2+gamma^2).',
            measurement_requirements=dict(distinct_releases=2,min_duration_s=40,recommended_duration_s=120,
                independent_initial_conditions=True,free_interval=True),
            provisional_tolerances=dict(each_mode_relative_error=.01),
            submission_fields=['angular_frequencies','unit','model','uncertainty_statement','measurement_evidence_ids'])

    def robot_state(self):
        state=self.backend.robot_state()
        return super().robot_state() | dict(gripper_holding=state['gripper_holding'])

    def _dispatch(self,name,args):
        if self.terminated:raise ValueError('episode terminated')
        if name not in TOOLS or not isinstance(args,dict):raise ValueError('unknown public coupled command')
        if name=='get_task_spec' and not args:return self.public_task()
        if name=='export_recording' and set(args)=={'evidence_id'}:return self.recorder.export(self.record_id(args['evidence_id']))
        if name=='submit_result':return self._submit(args)
        return super()._dispatch(name,args)

    def record_id(self,ref):
        """A finished record by its evidence_id, its number (1, 2, ...) or "latest"."""
        if ref=='latest':
            if not self.recorder.records:raise ValueError('no finished record yet')
            return list(self.recorder.records)[-1]
        if (isinstance(ref,int) and not isinstance(ref,bool)) or (isinstance(ref,str) and ref.isdigit()):
            return f'{self.episode_id}:encoder:{int(ref)}'
        if not isinstance(ref,str):raise ValueError('evidence_id must be an evidence id, a record number or "latest"')
        return ref

    def _submit(self,args):
        self.terminated=True
        gates=dict(valid_submission=False,measurement_evidence_valid=False,
                   independent_excitation=False,within_budget=self.tick<=self.horizon_ticks)
        errors=None;reason=None
        try:
            if set(args)!=set(self.public_task()['submission_fields']):raise ValueError('submission must match public schema')
            estimates=args['angular_frequencies']
            if not isinstance(estimates,list) or len(estimates)!=2 or not all(finite(v) and v>0 for v in estimates) or estimates[0]>=estimates[1] or args['unit']!='rad/s':
                raise ValueError('two finite positive ascending frequencies with unit rad/s required')
            for key in ('model','uncertainty_statement'):
                if not isinstance(args[key],str) or not 1<=len(args[key].strip())<=4000:raise ValueError('analysis statements required')
            gates['valid_submission']=True
            ids=args['measurement_evidence_ids']
            if not isinstance(ids,list) or len(ids)!=2 or any(not isinstance(x,str) for x in ids) or len(set(ids))!=2:raise ValueError('two distinct release evidence IDs required')
            records=[self.recorder.export(eid) for eid in ids]
            for rec in records:
                if rec['status']!='complete' or rec['invalid_reasons'] or rec['rows'][-1]['t_s']<40:raise ValueError('invalid or short free recording')
                if any(max(row[key] for row in rec['rows'])-min(row[key] for row in rec['rows'])<1 for key in ('x1_mm','x2_mm')):raise ValueError('insufficient measurable channel motion')
            gates['measurement_evidence_valid']=True
            vectors=[[r['rows'][0]['x1_mm'],r['rows'][0]['x2_mm']] for r in records]
            a,b=vectors;den=math.hypot(*a)*math.hypot(*b)
            if den<1 or abs(a[0]*b[1]-a[1]*b[0])/den<.2:raise ValueError('initial displacements do not provide independent excitation')
            gates['independent_excitation']=True
            truth=self.backend.dynamics.normal_angular_frequencies()
            errors=[abs(v-t)/t for v,t in zip(estimates,truth)]
        except (ValueError,TypeError,KeyError,IndexError) as exc:reason=str(exc)
        success=all(gates.values()) and errors is not None and max(errors)<=.1
        self.report=dict(schema_version='embodiedlab.evaluation/v1',evaluator='embodiedlab-coupled-modes-v1-dev',
            task_id=TASK,episode_id=self.episode_id,evaluation_ready=False,terminated=True,
            task_completed=success,reward=max(0.,1-max(errors)) if errors is not None and all(gates.values()) else 0.,
            parameter_relative_error=max(errors) if errors is not None else None,mode_relative_errors=errors,
            gates=gates,error=reason,sim_step=self.tick//(self.limits.physics_hz//30))
        return dict(accepted=True,terminated=True,episode_id=self.episode_id)

    def process_extra(self):
        """Recorder ticks for the process record: [start, end] per finished recording and the open one."""
        a=self.recorder.active
        return dict(recordings=dict(self.recorder.ticks),
                    recording_active=[a['evidence_id'],a['start_tick'],a['end_tick']] if a else None,
                    ready_s=self.backend.panel.ready_s,recorder_panel=self.backend.panel.record())

    def evaluation(self):
        if self.report is not None:return deepcopy(self.report)
        return dict(evaluator='embodiedlab-coupled-modes-v1-dev',reward=0.,task_completed=False,
                    sim_step=self.tick//(self.limits.physics_hz//30),terminated=self.terminated,
                    error=self.termination_reason,evaluation_ready=False)

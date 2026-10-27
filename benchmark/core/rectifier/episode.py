"""Public device API. Hidden component parameters never enter the task spec."""
from device_runtime import DeviceRuntime
import json
import numpy as np
from .submission import digest,validate,LOAD_TOLERANCE

TOOLS=('get_task_spec','get_observation','get_robot_state','execute_joint_trajectory','set_gripper','wait',
       'read_instruments','start_recording','stop_recording','export_recording','submit_result')

class RectifierEpisode(DeviceRuntime):
    def public_task(self):
        return dict(task_id='rectifier_supply',development_only=True,tools=list(TOOLS),physics_hz=240,
            episode_id=self.episode_id,horizon_ticks=self.horizon_ticks,wall_clock_seconds=7200,device_calibration=self.backend.public_calibration(),
            apparatus=dict(selectors=['CAP','LOAD','MODE'],rail_direction_world=[0,1,0],
                nominal_rail_x_world_m={'CAP':.75,'LOAD':.90,'MODE':1.05},
                table_top_world_z_m=self.backend.asset.table_top_z(self.backend.stage),post_top_above_table_m=.093,
                cap_labels=['C100','C220','C470'],mode_labels=['ON','OFF','DISCHARGE'],detent_spacing_m=.03,
                post_radius_m=.012,post_height_m=.060,post_colour='yellow',load_travel_m=[0,.060],
                load_range_ohm=[150,3300],load_encoder_resolution_m=.00001,
                load_mapping='R=150*22**(travel_m/.06)',
                controls='Push the labelled posts with the robot. CAP and MODE have detents. LOAD moves continuously.',
                physical_modules='Blue C1/C2/C3 capacitors on green PCB; separate AC source, bridge and R LOAD enclosure. CAP SELECT chooses a capacitor; LOAD controls the constant-resistance electronic load.',
                wiring='AC source -> 25 ohm source resistance -> four-diode bridge -> selected capacitor in parallel with R LOAD. Red leads are electrical; blue link is the load control.',
                off_behavior='Source opens; selected capacitor continues discharging through load. Other capacitors stay isolated.',
                discharge_resistor_ohm=100),
            circuit=dict(source_peak_V=12.,frequency_Hz=50.,source_resistance_ohm=25.,
                nominal_capacitances_uF=[100,220,470],capacitance_tolerance_fraction=.1,
                bridge='Four equal Shockley diodes; two conduct in series. No reverse recovery or leakage.',
                diode_n_range=[1.6,2.],diode_Is_A=1e-8,temperature_K=298.,
                scope=dict(sample_hz=12000,voltage_range_V=[-20,20],current_range_A=[-1,1],voltage_step_V=.002,current_step_A=.0001,max_record_seconds=30)),
            objective=dict(mean_voltage_min_V=8.5,ripple_pp_max_V=.8,final_load_ohm=self.backend.configuration['final_load_ohm'],
                submit='Minimum feasible load resistance for each capacitor, chosen final capacitor, predicted mean voltage and peak-to-peak ripple, evidence IDs.'),
            submission=dict(tool='submit_result',arguments=['expected_state_id','answer'],
                answer_schema={'experiment_id':'rectifier_supply',
                    'load_limits':[{'module_id':n,'minimum_load_ohm':'finite number in [150,3300]'} for n in ('C100','C220','C470')],
                    'final_configuration':{'module_id':'one capacitor label','load_ohm':'task load'},
                    'final_measurement':{'mean_voltage_V':'nonnegative finite number','ripple_pp_V':'nonnegative finite number'},
                    'evidence_ids':'nonempty unique list of completed raw records from this episode'},
                load_configuration_tolerance_fraction=LOAD_TOLERANCE,
                minimum_load_relative_tolerance=.05,mean_relative_tolerance=.03,
                ripple_absolute_tolerance_V='max(0.03, 0.05 * sealed true ripple)',
                reserved_simulation_seconds=1.3,
                terminal='One submission locks the episode, including invalid submissions. Retry with the same request ID is idempotent.',
                evaluation='After submission: hold actual configuration, settle 1 second, align to a full source period, then record 0.2 seconds privately at 24 kHz. No parameter fitting consistency gate.',
                raw_evidence='Export complete scope pages with controls and record hash. Analysis methods and number of measurements are unrestricted within public budget.'))

    def submit(self,args):
        if set(args)!={'expected_state_id','answer'} or args['expected_state_id']!=self.state_id:
            self.terminated=True
            raise ValueError('Invalid submission arguments or stale state; episode locked')
        path=self.backend.private/'submission.json'
        try:
            answer=json.loads(json.dumps(args['answer'],allow_nan=False))
            bindings=validate(answer,self.backend.bench,self.backend.configuration['final_load_ohm'])
            if self.backend.bench.active is not None:raise ValueError('Stop active recording before submission')
            if self.horizon_ticks-self.tick<312:raise ValueError('Reserve 1.3 seconds of public budget for submission')
            sealed=dict(answer=answer,submitted_tick=self.tick,evidence_sha256=bindings)
            sealed['answer_sha256']=digest(answer)
            with path.open('x') as f:json.dump(sealed,f,indent=2,allow_nan=False)
            for _ in range(240):self._step()
            while self.tick%24:self._step()
            start=self.tick;rows=[];controls=[]
            for _ in range(48):
                applied=list(self.backend.bench.switch)
                self._step();rows.extend(self.backend.bench.last_rows.tolist())
                controls.append(dict(start_tick=self.tick-1,cap=applied[0],load_ohm=applied[1],mode=applied[2]))
            v=np.asarray(rows)[:,2]
            target=self.backend.configuration['final_load_ohm'];module=answer['final_configuration']['module_id']
            stable=all(c['cap']==module and c['mode']=='ON' and
                abs(c['load_ohm']/target-1)<=LOAD_TOLERANCE and
                abs(c['load_ohm']/answer['final_configuration']['load_ohm']-1)<=LOAD_TOLERANCE for c in controls)
            payload=dict(start_tick=start,end_tick=self.tick,sample_hz=24000,samples=rows,controls=controls,
                         mean_voltage_V=float(v.mean()),ripple_pp_V=float(np.ptp(v)),configuration_valid=stable)
            payload['sha256']=digest(payload)
            with (self.backend.private/'post_submit_waveform.json').open('x') as f:json.dump(payload,f,allow_nan=False)
            receipt=dict(accepted=True,sealed=True,submission_id=self.episode_id+'-submission',
                         answer_sha256=sealed['answer_sha256'],physics_tick=self.tick)
            return receipt
        finally:
            self.terminated=True;self.termination_reason='submitted'

    def _dispatch(self,name,args):
        if self.terminated:raise ValueError('episode terminated')
        if name not in TOOLS or not isinstance(args,dict):raise ValueError('unknown rectifier tool')
        if name=='submit_result':return self.submit(args)
        if name=='get_task_spec' and not args:return self.public_task()
        if name=='read_instruments' and not args:return self.backend.bench.reading()
        if name=='export_recording' and set(args)<= {'evidence_id','offset','limit'} and 'evidence_id' in args:
            return self.backend.bench.export(args['evidence_id'],args.get('offset',0),args.get('limit',512))
        if name in ('start_recording','stop_recording') and set(args)=={'expected_state_id'}:
            if args['expected_state_id']!=self.state_id:raise ValueError('stale device state')
            key=(self.backend.bench.start if name=='start_recording' else self.backend.bench.stop)()
            if name=='stop_recording':
                directory=self.backend.output/'evidence';directory.mkdir(exist_ok=True)
                with (directory/(key+'.json')).open('x') as f:
                    json.dump(self.backend.bench.record_payload(key),f,allow_nan=False)
            self.revision+=1;return dict(evidence_id=key,state_id=self.state_id)
        return super()._dispatch(name,args)

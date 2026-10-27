"""incline_friction backend: physical robot, incline motor, two free contact sliders and the recorder front panel.

The participant receives only public instruments; material coefficients are consumed once when constructing the native
surfaces. Every physics tick reads the four console buttons and the twelve recorder keys (the shared panel,
RecorderPanel with start_on_rec: a valid REC starts a record of the keyed length at once), logs every press, and
samples the recorder.

Two sliders, A and B. The position sensor measures only the slider on its track (lying flat on the deck,
in the sensor's span) that is nearest the lower end, as an ultrasonic range finder sees the first object in its beam.
Each record sample keeps a private row: which slider was measured and whether it was free (not held, not touched by
the robot, the gate or the other slider); the evaluator attributes a record to a slider from these rows.
"""
from pathlib import Path
import json
import math
import numpy as np
import collision_reflex
import grip_contact
from torsion_backend import TorsionBackend
from .bench import (build_bench, ROOT, ORIGIN, BUTTONS, BLOCK_DIMS, GATE_SPEED_M_S, block_projection_half_extent,
                    RECORDER_PATH, RECORD_S_RANGE, PAD_CENTERS_XY, PAD_TOP_Z, LOADING_PAD_DIMS, SLIDERS,
                    slider_path, pad_path)

# The sensor's track: a slider lies flat on the deck (centre within 3 mm of its resting height) inside this span.
TRACK_S_M = (-.60, .14)
TRACK_LATERAL_M = .105
FLAT_NORMAL_M, FLAT_TOL_M = .005, .003


def on_track(along, normal, lateral):
    return TRACK_S_M[0] < along < TRACK_S_M[1] and abs(lateral) < TRACK_LATERAL_M and abs(normal-FLAT_NORMAL_M) < FLAT_TOL_M


def measured_slider(poses):
    """The slider the position sensor measures: of the sliders on its track, the one nearest the lower end (largest
    along-deck coordinate); None when no slider is on the track. poses: {name: (along, normal, lateral)}."""
    on = [(along, name) for name, (along, normal, lateral) in poses.items() if on_track(along, normal, lateral)]
    return max(on)[1] if on else None
from .instrument import InclineRecorder


class FrictionBackend(TorsionBackend):
    # The ramp tilts: the planner gets every collider as a box in its own frame, not its world-aligned bound.
    oriented_planning_boxes = True

    def __init__(self, args, app, parameters):
        import carb, cv2, omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema
        from isaacsim.core.api import World
        from isaacsim.core.api.physics_context import PhysicsContext
        from isaacsim.core.prims import SingleRigidPrim, RigidPrim
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot.manipulators import SingleManipulator
        from isaacsim.robot.manipulators.grippers import ParallelGripper
        from isaacsim.sensors.camera import Camera
        import hooke_apparatus as lab
        import hooke_physics
        from coupled_oscillator.controls import KEYS,RecorderPanel
        from coupled_oscillator import controls_asset as CA
        self.CA,self.KEYS=CA,KEYS
        self.panel=RecorderPanel(start_on_rec=True,record_range=RECORD_S_RANGE);self.panel_state=self.panel.state
        self.angle_rate=0.;self.last_angle=None;self.slider_poses={};self.gate_position=0.
        self.lab,self.Gf,self.UsdGeom,self.Action,self.cv2=lab,Gf,UsdGeom,ArticulationAction,cv2
        self.dt=1/args.physics_hz;self.tick_index=0;self.frame_ticks=args.physics_hz//30
        self.output=Path(args.output);self.private=self.output/'private';self.private.mkdir(parents=True,exist_ok=True)
        self.audit_file=(self.private/'physics.jsonl').open('w')
        self.closed=False;self.frame_index=0;self.held=self.releasing=None
        self.grasp_broken=False;self.previous_jaw=None;self.width_target=.08
        self.target_angle=0.;self.direction=0;self.gate_target=0.;self.gate_command=0.
        self.recorder=InclineRecorder(args.physics_hz)
        self.press_ticks={k:0 for k in BUTTONS};self.pressed=set()
        source=Usd.Stage.Open(args.stage)
        if source.GetPrimAtPath('/World/HookeApparatus'):
            source.GetPrimAtPath('/World/HookeApparatus').SetActive(False)
        # The shared gripper creates its joint below HookePhysics. Disable
        # obsolete spring joints individually, retaining an active parent.
        physics_root=source.GetPrimAtPath('/World/HookePhysics')
        if physics_root:
            physics_root.SetActive(True)
            for child in physics_root.GetChildren():child.SetActive(False)
        else:UsdGeom.Scope.Define(source,'/World/HookePhysics')
        if not source.GetPrimAtPath('/World/RobotRiser'):raise RuntimeError('original robot base missing')
        source.GetPrimAtPath('/World/RobotRiser').SetActive(True)
        for prim in list(source.Traverse()):
            if prim.IsA(UsdPhysics.Scene):prim.SetActive(False)
            if str(prim.GetPath()).startswith(lab.ROBOT_PATH) and prim.IsInstanceable():prim.SetInstanceable(False)
            for attr in prim.GetAttributes():
                if attr.GetTypeName()!=Sdf.ValueTypeNames.Asset:continue
                value=attr.Get()
                if value and str(value.path).lower().endswith(('.png','.jpg','.exr','.hdr')):
                    if str(value.path).endswith('normal.png'):attr.Set(Sdf.AssetPath('/workspace/normal.png'))
                    elif not value.resolvedPath or str(value.resolvedPath).startswith(('http:','https:')):attr.Set(Sdf.AssetPath(''))
        build_bench(source,{k:(float(parameters[k]['mu_s']),float(parameters[k]['mu_k'])) for k in SLIDERS})
        hooke_physics.configure_arm(source,lab.ROBOT_PATH)
        lab.setup_neutral_lab_lighting(source)
        source.Flatten().Export(str(self.private/'initial.usda'))
        omni.usd.get_context().open_stage(str(self.private/'initial.usda'))
        for _ in range(30):app.update()
        self.stage=omni.usd.get_context().get_stage()
        self.hinge=UsdPhysics.DriveAPI(self.stage.GetPrimAtPath(ROOT+'/Hinge'),'angular')
        self.gate_drive=UsdPhysics.DriveAPI(self.stage.GetPrimAtPath(ROOT+'/GateGuide'),'linear')
        carb.settings.get_settings().set_bool('/physics/updateToUsd',True)
        carb.settings.get_settings().set_bool('/physics/updateVelocitiesToUsd',True)
        self.world=World(stage_units_in_meters=1.,physics_dt=self.dt,rendering_dt=1/30)
        self.paths={k:slider_path(k) for k in SLIDERS}
        self.bodies={k:self.world.scene.add(SingleRigidPrim(p,name='friction_'+k,reset_xform_properties=False)) for k,p in self.paths.items()}
        self.ramp=self.world.scene.add(SingleRigidPrim(ROOT+'/Ramp',name='incline_ramp',reset_xform_properties=False))
        self.gate=self.world.scene.add(SingleRigidPrim(ROOT+'/Gate',name='incline_gate',reset_xform_properties=False))
        self.buttons={k:self.world.scene.add(SingleRigidPrim(ROOT+'/Button_'+k,name='incline_button_'+k,reset_xform_properties=False)) for k in BUTTONS}
        self.key_view=self.world.scene.add(RigidPrim([RECORDER_PATH+'/Keypad/Key_'+k for k in KEYS],name='incline_keys',reset_xform_properties=False))
        robot_parts=[lab.ROBOT_PATH+'/'+n for n in ('panda_hand','panda_leftfinger','panda_rightfinger')]+[lab.ROBOT_PATH+'/panda_link'+str(i) for i in range(8)]
        robot_parts=[p for p in robot_parts if self.stage.GetPrimAtPath(p)]
        self.robot_contact_parts=robot_parts
        def sensor(path,name,filters):
            return self.world.scene.add(RigidPrim(path,name=name,reset_xform_properties=False,track_contact_forces=True,
                contact_filter_prim_paths_expr=filters,max_contact_count=128))
        self.robot_sensors={k:sensor(p,k+'_robot_contact',robot_parts) for k,p in self.paths.items()}
        self.finger_sensors=grip_contact.finger_contact_sensors(self.world.scene,RigidPrim,lab.ROBOT_PATH,self.paths,'friction')
        self.grip=grip_contact.GripState()
        self.button_sensors={k:sensor(ROOT+'/Button_'+k,'button_contact_'+k,robot_parts) for k in BUTTONS}
        self.gate_sensors={k:sensor(p,'slider_gate_contact_'+k,[ROOT+'/Gate']) for k,p in self.paths.items()}
        self.pair_sensor=sensor(self.paths['A'],'slider_pair_contact',[self.paths['B']])
        self.structure_sensors={name:sensor(ROOT+'/'+name,'structure_'+name,robot_parts) for name in ('Ramp','Gate')}
        self.support_sensors={k:sensor(p,'slider_support_'+k,[ROOT+'/Ramp']+[pad_path(n) for n in SLIDERS]) for k,p in self.paths.items()}
        gripper=ParallelGripper(end_effector_prim_path=lab.ROBOT_PATH+'/panda_rightfinger',
            joint_prim_names=['panda_finger_joint1','panda_finger_joint2'],joint_opened_positions=np.array([.04,.04]),
            joint_closed_positions=np.array([.009,.009]),action_deltas=np.array([.01,.01]))
        self.robot=self.world.scene.add(SingleManipulator(prim_path=lab.ROBOT_PATH,name='friction_franka',
            end_effector_prim_path=lab.ROBOT_PATH+'/panda_rightfinger',gripper=gripper))
        self.cameras=[self.world.scene.add(Camera(prim_path=p,name='friction_cam'+str(i),resolution=(512,512))) for i,p in enumerate(lab.CAMERA_PATHS)]
        if getattr(self.world,'_physics_context',None) is None:self.world._physics_context=PhysicsContext(physics_dt=self.dt,set_defaults=False)
        self.world.get_physics_context().set_solver_type(getattr(args,'solver_type','PGS'))
        collision_reflex.prepare_scene(self.stage, lab.ROBOT_PATH, self.private, self.paths.values())   # colliders as drawn; contact reports
        self.world.initialize_physics();self.world.reset()
        self.reflex = collision_reflex.Reflex(lab.ROBOT_PATH, self.dt, self.private)   # collision reflex
        standby=np.array([0.,-.785,0.,-2.356,0.,1.571,.785,.04,.04])
        self.robot.set_joints_default_state(positions=standby);self.robot.set_joint_positions(standby)
        self.robot.initialize();self.target=standby[:7].copy()
        for camera in self.cameras:camera.initialize();camera.add_distance_to_image_plane_to_frame()
        for _ in range(60):
            self.robot.get_articulation_controller().apply_action(self.Action(joint_positions=self.target,joint_indices=np.arange(7)))
            self.world.step(render=True)
        self.last_q=np.asarray(self.robot.get_joint_positions()[:7]).copy()
        # Key caps at rest after the warm-up (as CoupledBackend): a press is the cap's drop below this height.
        self.key_rest_z=np.asarray(self.key_view.get_world_poses()[0],float)[:,2].copy();self.key_down=[False]*len(KEYS)
        self.video_path=self.output/'friction_three_views_raw.mp4'
        self.writer=cv2.VideoWriter(str(self.video_path),cv2.VideoWriter_fourcc(*'mp4v'),30,(1536,512))
        if not self.writer.isOpened():raise RuntimeError('friction video writer unavailable')
        from omni.physx import get_physx_interface
        from omni.physx.bindings._physx import SimulationEvent
        from pxr import PhysicsSchemaTools
        def joint_event(event):
            if event.type==int(SimulationEvent.JOINT_BREAK) and str(PhysicsSchemaTools.decodeSdfPath(*event.payload['jointPath']))=='/World/HookePhysics/GraspConstraint':
                self.grasp_broken=True
        self._subscription=get_physx_interface().get_simulation_event_stream_v2().create_subscription_to_pop(joint_event)
        self.stage.GetRootLayer().Export(str(self.private/'runtime_scene.usda'))
        (self.private/'configuration.json').write_text(json.dumps(parameters,indent=2))

    def coordinates(self):
        """Ramp angle and, per slider, (along, normal, lateral) in the deck frame from the hinge axis (along positive
        downhill); keeps each slider's world pose, yaw, tilt and lateral extent for the audit and the bench state."""
        rp,rq=self.ramp.get_world_pose()
        w,x,y,z=map(float,rq)
        angle=math.degrees(math.atan2(2*(w*y+x*z),1-2*(y*y+z*z)))
        theta=math.radians(angle)
        coords={}
        for name in SLIDERS:
            bp,bq=self.bodies[name].get_world_pose()
            delta=np.asarray(bp)-rp
            # Exact lateral projection of the rectangular plate, for private
            # side-guard clearance auditing; the ramp rotates only around Y.
            bw,bx,by,bz=map(float,bq)
            row_y=np.array([2*(bx*by+bw*bz),1-2*(bx*bx+bz*bz),2*(by*bz-bw*bx)])
            self.slider_poses[name]=dict(position=np.asarray(bp,float).copy(),quat=np.asarray(bq,float).copy(),
                yaw_deg=round(math.degrees(math.atan2(2*(bw*bz+bx*by),1-2*(by*by+bz*bz))),3),
                tilt_deg=round(math.degrees(math.acos(max(-1.,min(1.,1-2*(bx*bx+by*by))))),3),
                lateral_extent_m=float(abs(delta[1])+block_projection_half_extent(row_y)))
            coords[name]=(float(delta[0]*math.cos(theta)-delta[2]*math.sin(theta)),
                          float(delta[0]*math.sin(theta)+delta[2]*math.cos(theta)),float(delta[1]))
        return angle,coords

    def step(self,target,velocity=None):
        if self.closed:raise RuntimeError('backend closed')
        previous=self.target.copy()
        if target is not None:self.target=np.asarray(target)
        controller=self.robot.get_articulation_controller()
        controller.apply_action(self.Action(joint_positions=self.target,
            joint_velocities=(self.target-previous)/self.dt if velocity is None else np.asarray(velocity),joint_indices=np.arange(7)))
        controller.apply_action(self.Action(joint_positions=np.full(2,self.grip.drive_width(self.width_target,self.robot)/2),joint_indices=np.array([7,8])))
        self.target_angle=float(np.clip(self.target_angle+self.direction*.5*self.dt,0.,35.))
        self.gate_command+=float(np.clip(self.gate_target-self.gate_command,-GATE_SPEED_M_S*self.dt,GATE_SPEED_M_S*self.dt))
        self.hinge.GetTargetPositionAttr().Set(self.target_angle)
        self.gate_drive.GetTargetPositionAttr().Set(self.gate_command)
        self.world.step(render=False);self.tick_index+=1
        if self.tick_index % 8 == 0:
            for name,contact in self.structure_sensors.items():
                forces=np.asarray(contact.get_contact_force_matrix(dt=self.dt))
                if np.max(np.abs(forces))>.1:
                    self.audit_file.write(json.dumps(dict(event='robot_structure_contact',tick=self.tick_index,
                        structure=name,robot_parts=self.robot_contact_parts,forces_N=forces.tolist()))+'\n')
        self._gripper_mechanism()
        q=np.asarray(self.robot.get_joint_positions()[:7]);rates=np.abs(q-self.last_q)/self.dt;self.last_q=q.copy()
        if not np.isfinite(q).all():raise RuntimeError('robot joint state is not finite')
        from control_contract import reset_window, window_rates; mean_rates = window_rates(self, q, self.dt)
        if float(np.max(mean_rates))>.9:
            self.audit_file.write(json.dumps(dict(event='robot_velocity_guard',tick=self.tick_index,
                robot_joints=q.tolist(),robot_rates=rates.tolist()))+'\n');self.audit_file.flush()
            from control_contract import protective_stop
            self.target=q.copy()   # protective stop: hold the measured pose
            self.pending_protective_stop=protective_stop(mean_rates); reset_window(self, q, self.dt)   # raised by the runtime after this tick
        stop = self.reflex.check(self.tick_index, collision_reflex.held_path(self))
        if stop is not None:   # collision reflex: hold the measured pose; raised by the runtime after this tick
            self.target = q.copy()
            if getattr(self, 'pending_protective_stop', None) is None:
                self.pending_protective_stop = stop
        for name,body in self.buttons.items():
            travel=float(body.get_world_pose()[0][2]-(ORIGIN[2]+BUTTONS[name][2]))
            down=travel<-.005 and self._touching(self.button_sensors[name])
            self.press_ticks[name]=self.press_ticks[name]+1 if down else 0
            if not down:self.pressed.discard(name)
            if self.press_ticks[name]>=round(.04/self.dt) and name not in self.pressed:
                self.pressed.add(name)
                self.audit_file.write(json.dumps(dict(event='physical_button_press',tick=self.tick_index,
                    button=name,cap_travel_m=travel))+'\n')
                if name=='raise':self.direction=1
                elif name=='lower':self.direction=-1
                elif name=='hold':self.direction=0
                elif name=='release':self.gate_target=0. if self.gate_target else .1
        self._keys()
        angle,coords=self.coordinates()
        self.angle_rate=0. if self.last_angle is None else (angle-self.last_angle)/self.dt;self.last_angle=angle
        ramp_position=self.ramp.get_world_pose()[0]
        gate_delta=self.gate.get_world_pose()[0]-ramp_position
        radians=math.radians(angle)
        gate_position=float(gate_delta[0]*math.sin(radians)+gate_delta[2]*math.cos(radians))-.025
        self.gate_position=gate_position
        pair_contact=self._touching(self.pair_sensor)
        contact={k:dict(robot=self._touching(self.robot_sensors[k]),gate=self._touching(self.gate_sensors[k])) for k in SLIDERS}
        free={k:bool(self.held!=k and not contact[k]['robot'] and not contact[k]['gate'] and not pair_contact) for k in SLIDERS}
        sensed=measured_slider(coords)
        # Private row of the sample: the measured slider and whether it was free (never exported).
        self._record_tick(angle,None if sensed is None else coords[sensed][0],
                          dict(slider=sensed,free=bool(sensed is not None and free[sensed])))
        sliders={k:dict(s_m=round(coords[k][0],6),normal_m=round(coords[k][1],6),lateral_m=round(coords[k][2],6),
                        lateral_extent_m=round(self.slider_poses[k]['lateral_extent_m'],5),robot_contact=contact[k]['robot'],
                        gate_contact=contact[k]['gate'],yaw_deg=self.slider_poses[k]['yaw_deg'],
                        tilt_deg=self.slider_poses[k]['tilt_deg']) for k in SLIDERS}
        self.audit_file.write(json.dumps(dict(tick=self.tick_index,angle_deg=angle,sliders=sliders,sensed=sensed,
            pair_contact=pair_contact,held=self.held,robot_peak_rate=float(max(rates)),direction=self.direction,
            gate_command_m=self.gate_command,gate_actual_m=gate_position,recorder=self.panel.state))+'\n')
        return {}

    # --- recorder keypad ---------------------------------------------------------------------------------------------
    def _event(self,kind,fields):
        if self.audit_file.closed:return
        self.audit_file.write(json.dumps(dict(event=kind,tick=self.tick_index,**fields))+'\n')

    def _keys(self):
        """Recorder keys (CoupledBackend._controls): a key acts once when its cap is down >= SWITCH_DOWN_M and again
        only after rising back to <= SWITCH_UP_M."""
        depth=self.key_rest_z-np.asarray(self.key_view.get_world_poses()[0],float)[:,2]
        for k,(key,d) in enumerate(zip(self.KEYS,depth)):
            if not self.key_down[k] and d>=self.CA.SWITCH_DOWN_M:
                self.key_down[k]=True;self._key(key)
            elif self.key_down[k] and d<=self.CA.SWITCH_UP_M:self.key_down[k]=False

    def _key(self,key):
        effect=self.panel.press(key)
        fields=dict(key=key,effect=effect,entry=self.panel.entry)
        if effect=='start':
            # REC took a valid length: the record of that length starts with this tick's sample.
            number=len(self.recorder.records)+1
            fields['evidence_id']=self.recorder.start(number,self.panel.ready_s)
            self.panel.start(number);fields['length_s']=self.panel.recording_s
        self._event('key',fields)
        self._panel_changed()

    def _panel_changed(self):
        if self.panel.state!=self.panel_state:
            self._event('recorder',dict(old=self.panel_state,new=self.panel.state,number=self.panel.number,
                                        recording_s=self.panel.recording_s))
            self.panel_state=self.panel.state

    def _record_tick(self,angle,position,private):
        """This tick's recorder sample (private: the audit row kept with it); a record that reaches its keyed length
        ends and the panel shows DONE."""
        finished=self.recorder.sample(self.tick_index,angle,position,private=private)
        if finished is not None:
            self.panel.finish();self._event('record_end',dict(evidence_id=finished,length_s=self.panel.last_length_s))
            self._panel_changed()
        elif self.recorder.active is not None:self.panel.tick(self.recorder.elapsed_s)

    def _update_display(self):
        """Recorder display of the incline panel. Also the display hook the torsion parent calls from after_step and
        observation (its own panel path and state do not exist here); redraws only when the shown state changes."""
        shown=self.panel.display()
        if shown!=getattr(self,'_shown_display',None):
            self.CA.update_recorder_display(self.stage,RECORDER_PATH,shown);self._shown_display=shown
            # DLSS keeps a history of static pixels: hold its reset for a few renders, and an observation renders until
            # it has settled (TorsionBackend._display_render / _render_for_capture), or the digits stay stale.
            self._display_changed()

    def _show_panel(self):
        self._update_display()

    def after_step(self,timers,tick):
        if tick%self.frame_ticks==0:self._update_display()
        return super().after_step(timers,tick)

    def observation(self):
        self._update_display()
        return super().observation()

    def process_extra(self):
        """Recorder ticks for the process record: [first, last] per finished record, and the panel."""
        return dict(recordings=dict(self.recorder.ticks),recorder_panel=self.panel.record())

    # --- the bench at submission ---------------------------------------------------------------------------------------
    def _slider_state(self,name):
        """Where a slider lies relative to the loading pads: the pad its centre is over (or the nearest pad), the
        offset from that pad's centre, its underside above the pad top, its tilt, and whether it is held."""
        pose=self.slider_poses.get(name)
        position,quat=(pose['position'],pose['quat']) if pose is not None else (np.full(3,np.inf),np.array([1.,0.,0.,0.]))
        w,x,y,z=map(float,quat)
        tilt=math.degrees(math.acos(max(-1.,min(1.,1-2*(x*x+y*y)))))
        pad=min(SLIDERS,key=lambda k:math.hypot(position[0]-PAD_CENTERS_XY[k][0],position[1]-PAD_CENTERS_XY[k][1]))
        centre=PAD_CENTERS_XY[pad]
        return dict(held=self.held==name,pad=pad,
                    pad_offset_m=[round(float(position[0]-centre[0]),4),round(float(position[1]-centre[1]),4)],
                    pad_half_m=[LOADING_PAD_DIMS[0]/2,LOADING_PAD_DIMS[1]/2],
                    gap_above_pad_m=round(float(position[2]-BLOCK_DIMS[2]/2-PAD_TOP_Z),4),tilt_deg=round(tilt,3))

    def bench_state(self,reason=None):
        """Ramp, gate, recorder and the two sliders as the tidy-up gate reads them (sealed as the terminal state).
        Values of the last physics tick, so this also works after the simulation has closed."""
        state=dict(
            ramp=dict(angle_deg=round(float(self.last_angle or 0.),4),rate_deg_s=round(float(self.angle_rate),4),
                      motor={1:'raise',-1:'lower',0:'hold'}[self.direction]),
            gate=dict(in_place=bool(self.gate_target==0. and abs(self.gate_position)<=.002),
                      target='in' if self.gate_target==0. else 'out',position_m=round(float(self.gate_position),4)),
            recorder=dict(self.panel.record(),ready=self.panel.ready,recording=self.panel.recording),
            sliders={k:self._slider_state(k) for k in SLIDERS},
            held=self.held,recorder_active=self.recorder.active is not None)
        if reason is not None:self._event('bench_state',dict(reason=reason,**{k:v for k,v in state.items() if k!='held'}))
        return state

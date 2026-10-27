"""Independent Hooke physics device. No SDK, IK, planner or scripted policy."""
from collections import deque
from pathlib import Path
import json
import numpy as np
import collision_reflex
import grip_contact

# Dynamic bench displays (recorder and live tray field): RTX's DLSS keeps a history of static pixels, so redrawn
# digits stayed stale in the cameras (zdisp-dbg-1015). As chem/circuit_grid: hold the history reset for a few renders
# after a display change, and take an observation only after the reset has settled.
HISTORY_RESET='/rtx-transient/post/dlss/forceParamReset'
DISPLAY_RESET_RENDERS=3
DISPLAY_SETTLE_RENDERS=9



def tray_free_row(row):
    """One frame-log row: no robot contact on the tray or on a weight on its deck, no held weight pressing on the deck
    (support forces > 0.1 N). A row without `tray_contact` (or with a log error) is not free."""
    if row.get('tray_contact') is not False or 'log_error' in row:return False
    for key,force in row['forces'].items():
        if force>.1 and (key==row.get('held') or row['robot_contacts'].get(key)):return False
    return True

class HookeBackend:
    pendulum_ids = ()  # Shared fixed-tick runtime has no optical gates here.

    def __init__(self, args, app, score):
        import carb
        import cv2
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.physics_context import PhysicsContext
        from isaacsim.core.prims import SingleRigidPrim, RigidPrim
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot.manipulators import SingleManipulator
        from isaacsim.robot.manipulators.grippers import ParallelGripper
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
        import hooke_apparatus as asset
        import hooke_physics as physical
        from hooke_scale_reader import VisualScaleReader
        from isaac_scene import json_default
        self.Gf,self.UsdGeom,self.Action,self.asset,self.physical=Gf,UsdGeom,ArticulationAction,asset,physical
        self.cv2,self.json_default,self.score=cv2,json_default,score
        self.dt,self.tick_index=1/args.physics_hz,0
        self.frame_ticks=args.physics_hz//30
        self.output=Path(args.output);self.private=self.output/'private'
        self.private.mkdir(parents=True,exist_ok=True)
        self.audit_file=(self.private/'physics.jsonl').open('w')
        self.history=deque(maxlen=30)
        self.held,self.releasing=None,None
        self.previous_jaw=None
        self.grasp_broken=False
        self.width_target=.08
        self.closed=False
        self.frame_index=0
        self.weight_ids=score.weight_ids
        asset.WEIGHT_SEQUENCE=asset.WEIGHT_SEQUENCE[:len(self.weight_ids)]
        asset.WEIGHTS={g:asset.WEIGHTS[g] for g in asset.WEIGHT_SEQUENCE}
        self.paths=dict(zip(self.weight_ids,asset.WEIGHTS.values()))
        source=Usd.Stage.Open(args.stage)
        for prim in source.Traverse():
            if str(prim.GetPath()).startswith(asset.ROBOT_PATH):
                if prim.IsInstanceable():prim.SetInstanceable(False)
                visibility=prim.GetAttribute('visibility')
                if visibility:visibility.Set('inherited')
            for attribute in prim.GetAttributes():
                if attribute.GetTypeName()!=Sdf.ValueTypeNames.Asset:continue
                value=attribute.Get()
                if value and str(value.path).lower().endswith(('.png','.jpg','.exr','.hdr')):
                    if str(value.path).endswith('normal.png'):attribute.Set(Sdf.AssetPath('/workspace/normal.png'))
                    elif not value.resolvedPath or str(value.resolvedPath).startswith(('http:','https:')):attribute.Set(Sdf.AssetPath(''))
        prepared=self.private/'initial.usda';source.Flatten().Export(str(prepared))
        omni.usd.get_context().open_stage(str(prepared))
        for _ in range(240):
            app.update();self.stage=omni.usd.get_context().get_stage()
            if self.stage and self.stage.GetPrimAtPath(asset.HANGER_PATH):break
        else:raise RuntimeError('Hooke stage failed to load')
        asset.prepare_cumulative_weights(self.stage)
        asset.prepare_runtime_tray(self.stage)
        # hooke_dynamic_k: the tray encoder's recorder panel and the damper lever, authored before the layout so that it
        # moves them with the bench. hooke_k has neither.
        self.controls=bool((getattr(args,'hooke_dynamics',None) or {}).get('encoder'))
        if self.controls:asset.build_encoder_controls(self.stage)
        # Registry instances move the whole apparatus rigidly before its spring joint is authored.
        self.placement=None
        if getattr(args,'layout_candidates',None):
            import layout
            self.placement=layout.place_hooke(self.stage,args.layout_candidates,self.private,asset)
        dynamics=getattr(args,'hooke_dynamics',None) or {}
        # The apparatus is prepared with the settling damping; a dynamic experiment switches to its
        # instance damping once the empty tray rests, before the Agent's first action.
        self.free_damping=dynamics.get('damping_n_s_m')
        self.travel_start_z=physical.configure_contacts(self.stage,list(self.paths.values()),asset.TRAY_PATH,score.truth,
            **{k:dynamics[k] for k in ('body_mass_kg',) if k in dynamics})
        self.encoder=None  # hooke_encoder.TrayEncoder, attached by the dynamic experiment's runtime
        self.panel=self.lever=None  # recorder panel and damper lever runtimes (hooke_dynamic_k), set after the warm-up
        self.settling,self.joint_damping=True,physical.SPRING_DAMPING_NS_M
        physical.configure_arm(self.stage,asset.ROBOT_PATH)
        asset.setup_neutral_lab_lighting(self.stage)
        carb.settings.get_settings().set_bool('/physics/updateToUsd',True)
        carb.settings.get_settings().set_bool('/physics/updateVelocitiesToUsd',True)
        self.world=World(stage_units_in_meters=1.,physics_dt=self.dt,rendering_dt=1/30)
        self.hanger=self.world.scene.add(SingleRigidPrim(asset.HANGER_PATH,name='hanger',reset_xform_properties=False))
        self.bodies={key:self.world.scene.add(SingleRigidPrim(path,name=key,reset_xform_properties=False)) for key,path in self.paths.items()}
        self.support_sensors={key:self.world.scene.add(RigidPrim(path,name=key+'_support',reset_xform_properties=False,
            track_contact_forces=True,contact_filter_prim_paths_expr=[asset.HANGER_PATH],max_contact_count=64)) for key,path in self.paths.items()}
        robot_parts=[asset.ROBOT_PATH+'/'+n for n in ('panda_hand','panda_leftfinger','panda_rightfinger')]+[asset.ROBOT_PATH+'/panda_link'+str(i) for i in range(8)]
        robot_parts=[p for p in robot_parts if self.stage.GetPrimAtPath(p)]
        self.finger_sensors=grip_contact.finger_contact_sensors(self.world.scene,RigidPrim,asset.ROBOT_PATH,self.paths,'hooke')
        self.grip=grip_contact.GripState()
        self.robot_sensors={key:self.world.scene.add(RigidPrim(path,name=key+'_robot_contact',reset_xform_properties=False,
            track_contact_forces=True,contact_filter_prim_paths_expr=robot_parts,max_contact_count=128)) for key,path in self.paths.items()}
        # Robot contact with the tray body (deck, stem, pointer): the encoder's contact status on the dynamic bench, and
        # the frame log's `tray_contact` on both benches (also hooke_k, where fingertips pressing the
        # tray went unrecorded; a contact report only, no physical effect; the static rows then also carry
        # damper/recorder None).
        self.hanger_contact=self.world.scene.add(RigidPrim(asset.HANGER_PATH,name='hanger_robot_contact',reset_xform_properties=False,
            track_contact_forces=True,contact_filter_prim_paths_expr=robot_parts,max_contact_count=128))
        if self.controls:   # hand-operated controls: the keypad caps and the damper lever carriage (positions only)
            from coupled_oscillator.controls import KEYS
            self.cap_names=list(KEYS)
            self.cap_view=self.world.scene.add(RigidPrim([asset.CONTROLS_PANEL_PATH+'/Keypad/Key_'+k for k in KEYS],
                                                         name='hooke_keys',reset_xform_properties=False))
            self.lever_view=self.world.scene.add(RigidPrim(asset.LEVER_PATH+'/Carriage',name='hooke_lever',reset_xform_properties=False))
        gripper=ParallelGripper(end_effector_prim_path=asset.ROBOT_PATH+'/panda_rightfinger',
            joint_prim_names=['panda_finger_joint1','panda_finger_joint2'],joint_opened_positions=np.array([.04,.04]),
            joint_closed_positions=np.array([.009,.009]),action_deltas=np.array([.01,.01]))
        self.robot=self.world.scene.add(SingleManipulator(prim_path=asset.ROBOT_PATH,name='hooke_franka',
            end_effector_prim_path=asset.ROBOT_PATH+'/panda_rightfinger',gripper=gripper))
        self.cameras=[self.world.scene.add(Camera(prim_path=p,name='hooke_cam_'+str(i),resolution=(512,512))) for i,p in enumerate(asset.CAMERA_PATHS)]
        if getattr(self.world,'_physics_context',None) is None:
            self.world._physics_context=PhysicsContext(physics_dt=self.dt,set_defaults=False)
            self.world.set_simulation_dt(physics_dt=self.dt,rendering_dt=1/30)
        collision_reflex.prepare_scene(self.stage, asset.ROBOT_PATH, self.private, self.paths.values())   # colliders as drawn; contact reports
        self.world.initialize_physics();self.world.reset()
        self.reflex = collision_reflex.Reflex(asset.ROBOT_PATH, self.dt, self.private)   # collision reflex
        standby=np.array([0.,-.785,0.,-2.356,0.,1.571,.785,.04,.04])
        self.robot.set_joints_default_state(positions=standby)
        self.robot.set_joint_positions(standby) # Initial conditions only.
        self.robot.initialize();self.target=standby[:7].copy()
        for camera in self.cameras:
            camera.initialize();camera.add_distance_to_image_plane_to_frame()
        for _ in range(60):
            self.robot.get_articulation_controller().apply_action(self.Action(joint_positions=self.target,joint_indices=np.arange(7)))
            self.world.step(render=True)
        if self.controls:self._init_controls()
        self.initial_hanger_z=float(self.hanger.get_world_pose()[0][2])
        bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
        self.bottom_offsets={key:float(bounds.ComputeWorldBound(self.stage.GetPrimAtPath(path)).ComputeAlignedRange().GetMin()[2])-float(self.bodies[key].get_world_pose()[0][2]) for key,path in self.paths.items()}
        # Weight box seats in the world (after layout jitter), for the sealed device record and the offline process rules:
        # every seat's floor centre, the root position of a weight standing on it (the frame log's xyz_mm is the root;
        # the weights are identical) and the weight that starts in it (None for the seats beyond the weight count).
        xforms=UsdGeom.XformCache()
        self.weight_box={'path':asset.WEIGHT_BOX_PATH,'grid_columns_rows':[asset.WEIGHT_SEAT_COLUMNS,asset.WEIGHT_SEAT_ROWS],
                         'seat_pitch_xy_m':[asset.WEIGHT_SEAT_PITCH_M,asset.WEIGHT_SEAT_ROW_PITCH_M],
                         'well_m':2*asset.WEIGHT_SEAT_WELL_HALF_M,'seats':{}}
        seated_offset=self.bottom_offsets[self.weight_ids[0]]
        for index in range(asset.WEIGHT_SEAT_COUNT):
            floor=np.asarray(xforms.GetLocalToWorldTransform(self.stage.GetPrimAtPath(asset.weight_seat_path(index))).ExtractTranslation(),float)
            self.weight_box['seats'][str(index+1)]={'label':str(index+1),
                                                   'starts_with':self.weight_ids[index] if index<len(self.weight_ids) else None,
                                                   'floor_centre_m':[round(float(v),4) for v in floor],
                                                   'seated_weight_xyz_m':[round(float(v),4) for v in floor-[0,0,seated_offset]]}
        self.reader=VisualScaleReader()
        self.video_path=self.output/'hooke_three_views_raw.mp4'
        self.writer=cv2.VideoWriter(str(self.video_path),cv2.VideoWriter_fourcc(*'mp4v'),30,(1536,512))
        if not self.writer.isOpened():raise RuntimeError('Hooke video writer unavailable')
        for sensor in list(self.support_sensors.values())+list(self.robot_sensors.values()):
            data=sensor.get_contact_force_data(dt=self.dt)
            if data is None or np.asarray(data[4]).size==0:raise RuntimeError('Hooke contact sensor unavailable')
        self.last_q=np.asarray(self.robot.get_joint_positions()[:7]).copy()
        from omni.physx import get_physx_interface
        from omni.physx.bindings._physx import SimulationEvent
        from pxr import PhysicsSchemaTools
        def on_joint_event(event):
            if event.type == int(SimulationEvent.JOINT_BREAK):
                path = str(PhysicsSchemaTools.decodeSdfPath(*event.payload['jointPath']))
                if path == '/World/HookePhysics/GraspConstraint':
                    self.grasp_broken = True
        self._subscription = get_physx_interface().get_simulation_event_stream_v2().create_subscription_to_pop(on_joint_event)
        (self.private/'configuration.json').write_text(json.dumps(score.configuration,indent=2))
        self.stage.GetRootLayer().Export(str(self.private/'runtime_scene.usda'))

    def _transform(self,path):
        return self.UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))

    def public_calibration(self):
        from robot_calibration import public_robot_calibration
        return public_robot_calibration(self) | {'gripper_mechanism':grip_contact.public_gripper_mechanism()}

    def robot_state(self):
        q=np.asarray(self.robot.get_joint_positions())
        return dict(joint_positions_rad=q[:7].tolist(),gripper_width_m=float(sum(q[7:9])),gripper_holding=self.held is not None)

    def set_gripper(self,width_m):
        opens=self.grip.opens(width_m,np.asarray(self.robot.get_joint_positions())[7:9])
        self.width_target=width_m
        if opens:self.grasp_broken=False
        if opens and self.held is not None:
            self.physical.release_grasp_joint(self.stage,self.paths[self.held])
            self.releasing,self.held=self.held,None

    def set_magnet(self,enabled):
        raise ValueError('Hooke apparatus has no switchable magnetic probe')

    def _gripper_mechanism(self):
        q=np.asarray(self.robot.get_joint_positions())
        if self.releasing is not None and self.grip.break_cleared(q[7:9]):
            self.physical.finish_release(self.stage,self.paths[self.releasing])   # broken hold: fingers collide again
        if self.releasing is not None and self.grip.cleared(q[7:9]):
            self.physical.finish_release(self.stage,self.paths[self.releasing]);self.releasing=None
        fingers=[self._transform(self.asset.ROBOT_PATH+'/'+n).ExtractTranslation() for n in ('panda_leftfinger','panda_rightfinger')]
        jaw=np.asarray((fingers[0]+fingers[1])/2)
        jaw_velocity=np.zeros(3) if self.previous_jaw is None else (jaw-self.previous_jaw)/self.dt
        self.previous_jaw=jaw.copy()
        if self.grasp_broken and self.held is not None:
            self.physical.release_grasp_joint(self.stage,self.paths[self.held])
            self.releasing,self.held=self.held,None
            self.grip.broke=True
        # Contact-conditioned grasp (grip_contact): the fingers close onto a body, both touch it and it
        # stops them. The rigid hold stands in for friction; there is no hidden grasp point.
        pressing=self.grip.pressing(q[7:9],np.asarray(self.robot.get_joint_velocities())[7:9],self.width_target)
        if self.held is not None or self.releasing is not None or self.grasp_broken or not pressing:return
        key=grip_contact.pinched_body({k:(self.finger_sensors[k],np.asarray(b.get_world_pose()[0]),np.asarray(b.get_linear_velocity()))
                                       for k,b in self.bodies.items()},jaw,jaw_velocity,self.dt)
        if key is None:return
        self.physical.attach_grasp_joint(self.stage,self.asset.HAND_PATH,self.paths[key],jaw)
        self.grip.hold(q[7:9])
        self.held=key

    def step(self,target,velocity=None):
        if self.closed:raise RuntimeError('Hooke backend closed')
        previous=self.target.copy()
        if target is not None:self.target=np.asarray(target)
        controller=self.robot.get_articulation_controller()
        controller.apply_action(self.Action(joint_positions=self.target,joint_velocities=(self.target-previous)/self.dt if velocity is None else np.asarray(velocity),joint_indices=np.arange(7)))
        controller.apply_action(self.Action(joint_positions=np.full(2,self.grip.drive_width(self.width_target,self.robot)/2),joint_indices=np.array([7,8])))
        self.world.step(render=False);self.tick_index+=1
        self._gripper_mechanism()
        if self.encoder is not None:self._encoder_tick()
        if self.panel is not None:self._controls()
        q=np.asarray(self.robot.get_joint_positions()[:7])
        rates=np.abs(q-self.last_q)/self.dt;speed=float(np.max(rates))
        # Private diagnostics for the arm-speed guard: per-frame peak joint rate and the commanded target.
        if speed>getattr(self,'frame_peak',(-1.,0))[0]:self.frame_peak=(speed,int(np.argmax(rates)))
        from control_contract import reset_window, window_rates; mean_rates = window_rates(self, q, self.dt)
        if float(np.max(mean_rates))>.9:
            self.audit_file.write(json.dumps({'event':'arm_speed_guard','tick':self.tick_index,'joint_rates_rad_s':rates.tolist(),
                'q_rad':q.tolist(),'previous_q_rad':self.last_q.tolist(),'target_rad':np.asarray(self.target).tolist(),
                'held':self.held,'releasing':self.releasing})+'\n');self.audit_file.flush()
            from control_contract import protective_stop
            self.target = q.copy();self.last_q = q.copy()   # protective stop: hold the measured pose
            self.pending_protective_stop=protective_stop(mean_rates); reset_window(self, q, self.dt)   # raised by the runtime after this tick
        stop = self.reflex.check(self.tick_index, collision_reflex.held_path(self))
        if stop is not None:   # collision reflex: hold the measured pose; raised by the runtime after this tick
            self.target = q.copy()
            if getattr(self, 'pending_protective_stop', None) is None:
                self.pending_protective_stop = stop
        self.last_q=q.copy()
        if self.tick_index%self.frame_ticks==0:
            forces={key:abs(float(sensor.get_contact_force_matrix(dt=self.dt)[0,0,2])) for key,sensor in self.support_sensors.items()}
            counts={key:int(np.asarray(sensor.get_contact_force_data(dt=self.dt)[4]).sum()) for key,sensor in self.robot_sensors.items()}
            if any(count<0 for count in counts.values()):raise RuntimeError('Hooke contact buffer invalid')   # a full buffer counts as contact
            row={'frame':self.tick_index//self.frame_ticks,'z':float(self.hanger.get_world_pose()[0][2]),
                 'forces':forces,'released':self.held is None and self.releasing is None and not any(counts.values()),
                 'robot_contacts':counts,'q_rad':[round(float(v),5) for v in q],
                 'peak_joint_rate_rad_s':round(self.frame_peak[0],4),'peak_joint':self.frame_peak[1]}
            self.frame_peak=(-1.,0)
            # Process-score log fields (read-only): held weight, weight positions (integer mm); on the dynamic
            # bench the tray's robot contact this tick, the damper lever's overlap and the recorder state word.
            try:
                row['held']=self.held
                row['xyz_mm']={k:[round(float(c)*1000) for c in b.get_world_pose()[0]] for k,b in self.bodies.items()}
                if self.hanger_contact is not None:
                    row['tray_contact']=self._contact_flag(self.hanger_contact)
                    row['damper']=round(self.lever.fraction,3) if self.lever is not None else None
                    row['recorder']=self.panel.state if self.panel is not None else None
            except Exception as exc:
                row['log_error']=f'{type(exc).__name__}: {exc}'
            self.history.append(row);self.audit_file.write(json.dumps(row)+'\n')
            self._seat_frame()
        return {}

    def _contact_flag(self,sensor):
        """_touching for the frame log only: never raises (None when the contact buffer is invalid)."""
        count=int(np.asarray(sensor.get_contact_force_data(dt=self.dt)[4]).sum())
        force=float(np.linalg.norm(np.asarray(sensor.get_contact_force_matrix(dt=self.dt))))
        if count<0 or not np.isfinite(force):return None
        return count>0 or force>1e-5

    @staticmethod
    def _touching(sensor,dt):
        # Same rule as contact_reporting: a contact pair or any non-zero force, no minimum force.
        count=int(np.asarray(sensor.get_contact_force_data(dt=dt)[4]).sum())
        force=float(np.linalg.norm(np.asarray(sensor.get_contact_force_matrix(dt=dt))))
        if count<0 or not np.isfinite(force):raise RuntimeError('Hooke contact buffer invalid')   # a full buffer counts as contact
        return count>0 or force>1e-5

    # Weight box occupancy for the process-score records. A seat holds a weight by the bench_restore home thresholds:
    # within 20 mm horizontally and vertically of a weight standing on that seat, tilted at most 10 degrees, not held.
    SEAT_DXY_M,SEAT_DZ_M,SEAT_TILT_DEG=.02,.02,10.

    def seat_occupancy(self):
        """{seat number: id of the weight seated in it, or None}."""
        poses={k:(np.asarray(p,float),np.asarray(q,float)) for k,(p,q) in ((k,b.get_world_pose()) for k,b in self.bodies.items())}
        out={}
        for seat in self.weight_box['seats'].values():
            home=np.asarray(seat['seated_weight_xyz_m'],float);out[seat['label']]=None
            for key,(xyz,q) in poses.items():
                w,x,y,z=q/max(float(np.linalg.norm(q)),1e-12)          # world z component of the weight's own up axis
                tilt=float(np.degrees(np.arccos(np.clip(1-2*(x*x+y*y),-1.,1.))))
                if key in (self.held,self.releasing) or tilt>self.SEAT_TILT_DEG:continue
                if np.hypot(*(xyz[:2]-home[:2]))<=self.SEAT_DXY_M and abs(xyz[2]-home[2])<=self.SEAT_DZ_M:
                    out[seat['label']]=key;break
        return out

    def start_seat_log(self):
        """At the episode clock's zero: the start occupancy (logged after the first frame row), then seat_in / seat_out
        events {tick, seat, weight} on every change, checked at 30 Hz; end_seat_log adds the occupancy at stop."""
        self.seat_state=self.seat_occupancy()
        self.seat_log={'thresholds':{'dxy_m':self.SEAT_DXY_M,'abs_dz_m':self.SEAT_DZ_M,'tilt_deg':self.SEAT_TILT_DEG},
                       'start':{'tick':self.tick_index,'seats':dict(self.seat_state)},'events':[],'end':None}
        self.seat_start_pending=True

    def _seat_audit(self,event):
        if not self.audit_file.closed:self.audit_file.write(json.dumps(event)+'\n')

    def _seat_frame(self):
        if getattr(self,'seat_state',None) is None:return
        try:
            if self.seat_start_pending:
                self.seat_start_pending=False
                self._seat_audit({'event':'seat_occupancy','phase':'start',**self.seat_log['start']})
            now=self.seat_occupancy()
            for seat,before in self.seat_state.items():
                if now[seat]==before:continue
                for kind,weight in (('seat_out',before),('seat_in',now[seat])):
                    if weight is not None:
                        event={'event':kind,'tick':self.tick_index,'seat':seat,'weight':weight}
                        self.seat_log['events'].append(event);self._seat_audit(event)
            self.seat_state=now
        except Exception as exc:
            self.seat_log.setdefault('errors',[]).append(f'{self.tick_index}: {type(exc).__name__}: {exc}')

    def end_seat_log(self):
        """The seat record with the occupancy now (at stop), logged once."""
        if getattr(self,'seat_state',None) is None:return None
        if self.seat_log['end'] is None:
            self.seat_log['end']={'tick':self.tick_index,'seats':self.seat_occupancy()}
            self._seat_audit({'event':'seat_occupancy','phase':'end',**self.seat_log['end']})
        return json.loads(json.dumps(self.seat_log))

    # --- hooke_dynamic_k bench controls: recorder keypad and damper lever ----------------------------------------------
    def _init_controls(self):
        """After the warm-up: key cap rest heights, the lever's detent 0 where its carriage stands, the panel."""
        from coupled_oscillator.controls import RecorderPanel
        from damper_lever import Lever,lever_info
        self.cap_rest_z=np.asarray(self.cap_view.get_world_poses()[0],float)[:,2].copy()
        self.cap_down=[False]*len(self.cap_names)
        carriage=np.asarray(self.lever_view.get_world_poses()[0],float)[0]
        self.lever=Lever(lever_info(self.asset.LEVER_PATH,**dict(self.asset.LEVER,detent0=(float(carriage[0]),float(carriage[1])))),self.stage)
        self.panel=RecorderPanel(start_on_rec=True,record_range=self.asset.RECORD_S_RANGE)
        self.panel_state=self.panel.state

    def _event(self,kind,fields):
        if not self.audit_file.closed:self.audit_file.write(json.dumps({'event':kind,'tick':self.tick_index,**fields})+'\n')

    def _controls(self):
        """Every physics tick, after the encoder sample: the key caps (each acts once per press: down >= SWITCH_DOWN_M,
        up again <= SWITCH_UP_M), the damper lever and the spring damping it sets, and the recorder's progress."""
        from coupled_oscillator.controls_asset import SWITCH_DOWN_M,SWITCH_UP_M
        tick=self.tick_index
        depth=self.cap_rest_z-np.asarray(self.cap_view.get_world_poses()[0],float)[:,2]
        for k,(name,d) in enumerate(zip(self.cap_names,depth)):
            if not self.cap_down[k] and d>=SWITCH_DOWN_M:
                self.cap_down[k]=True;self._key(name)
            elif self.cap_down[k] and d<=SWITCH_UP_M:self.cap_down[k]=False
        carriage=np.asarray(self.lever_view.get_world_poses()[0],float)[0]
        for change in self.lever.update(tick,carriage[:2]):self._event('damper',change)
        self._apply_damping()
        encoder=self.encoder
        if self.panel.recording and encoder is not None:
            if encoder.active is None:
                self.panel.finish();self._event('record_end',dict(number=self.panel.number))
            else:self.panel.tick((tick-encoder.active['start_tick'])*self.dt)
        self._panel_changed()

    def _key(self,key):
        """A key's closing edge. REC with a valid length starts a tray-encoder recording at once (start_on_rec)."""
        effect=self.panel.press(key)
        fields=dict(key=key,effect=effect,entry=self.panel.entry)
        if effect=='start':
            if self.encoder is None:raise RuntimeError('tray encoder not attached')
            # No budget check: a recording the episode does not outlast never finishes (and is never exported).
            eid=self.encoder.arm(self.tick_index,self.panel.ready_s,10**12,self.encoder_load_ids())
            self.panel.start(int(eid.rsplit(':',1)[-1]))
            fields.update(evidence_id=eid,record_s=self.panel.recording_s,load_ids=self.encoder.active['load_ids_at_arm'])
        self._event('key',fields)
        self._panel_changed()

    def _panel_changed(self):
        if self.panel.state!=self.panel_state:
            self._event('recorder',dict(old=self.panel_state,new=self.panel.state,number=self.panel.number,
                                        recording_s=self.panel.recording_s))
            self.panel_state=self.panel.state

    def _apply_damping(self):
        """Spring drive damping: the settling damping while the apparatus is prepared, then the instance damping plus
        the damper lever's share, in proportion to its magnet's overlap with the vane (ON: the settling damping, as
        the former damping brake)."""
        fraction=1.0 if self.settling else self.lever.fraction
        value=float(self.free_damping+fraction*(self.physical.SPRING_DAMPING_NS_M-self.free_damping))
        if abs(value-self.joint_damping)>1e-9:
            self._set_spring_damping(value);self.joint_damping=value

    def _set_spring_damping(self,value):
        from pxr import UsdPhysics
        drive=UsdPhysics.DriveAPI.Get(self.stage.GetPrimAtPath(self.physical.SPRING_JOINT),'linear')
        drive.GetDampingAttr().Set(value)
        if abs(float(drive.GetDampingAttr().Get())-value)>1e-6*max(1.,value):raise RuntimeError('spring damping not applied')

    def release_settling_damping(self):
        """End of the apparatus preparation (dynamic experiment): the damper lever, at OFF, alone adds damping from now on."""
        if self.lever is None:raise RuntimeError('damper lever not built')
        self.settling=False;self._apply_damping()

    def bench_state(self):
        """Recorder panel and damper lever now (part of the sealed terminal state); {} without them (hooke_k)."""
        if self.panel is None:return {}
        return dict(damper=self.lever.record(),
                    recorder=dict(self.panel.record(),ready=self.panel.ready,recording=self.panel.recording))

    def encoder_load_ids(self):
        """Weights resting on the tray: support force in every one of the last 10 frames, not held or touched.

        Unlike hooke_k's static support test (mean force equal to m g, tray at rest) this holds while the
        tray oscillates, when the support force varies as m (g - a) but never vanishes.
        """
        rows=list(self.history)[-10:]
        if len(rows)<10:return []
        return sorted(k for k in self.bodies if k!=self.held and k!=self.releasing
                      and all(r['forces'][k]>.1 and not r['robot_contacts'][k] for r in rows))

    def _encoder_tick(self):
        encoder=self.encoder;active=encoder.active
        z=float(self.hanger.get_world_pose()[0][2])
        contact=lifted=False
        if active is not None or self.tick_index%encoder.interval==0:
            loaded=active['load_ids_at_arm'] if active is not None else self.support_state()[0]
            # contact: the robot on the tray, on a weight counted at arming, or on any weight pressing on the deck now
            # (a weight touched at arming is not counted as load, and touching it must still show).
            pressing=[k for k,s in self.support_sensors.items() if abs(float(s.get_contact_force_matrix(dt=self.dt)[0,0,2]))>.1]
            contact=self._touching(self.hanger_contact,self.dt) or any(self._touching(self.robot_sensors[k],self.dt) for k in set(loaded)|set(pressing))
            # A resting weight presses on the deck; zero support force means it left the tray.
            lifted=any(abs(float(self.support_sensors[k].get_contact_force_matrix(dt=self.dt)[0,0,2]))<.02 for k in loaded)
        # brake: the damper lever not at OFF (its carriage more than 4 mm from the OFF detent) while sampling
        brake=self.lever is not None and self.lever.state!='OFF'
        encoder.sample(self.tick_index,self.travel_start_z-z,contact=bool(contact),load_lifted=bool(lifted),brake=bool(brake),
                       load_ids=self.encoder_load_ids)

    def _tray_reading(self):
        """Live tray field (hooke_dynamic_k display): the encoder model applied to the tray position now (same zero,
        sign, quantisation and range as a record; z_mm None outside the range), not a cached sample. None before the
        encoder is attached."""
        enc=self.encoder
        if enc is None:return None
        z=(self.travel_start_z-float(self.hanger.get_world_pose()[0][2])-enc.zero_below_start_m)*1000
        if not (np.isfinite(z) and abs(z)<=enc.range_mm):return {'z_mm':None}
        return {'z_mm':round(round(z/enc.resolution_mm)*enc.resolution_mm,1)}

    def _render(self):
        """world.render; on the dynamic bench with the DLSS history reset held after a display change."""
        index=getattr(self,'render_index',0)
        if self.panel is not None:
            import carb
            settings=carb.settings.get_settings()
            reset=index<getattr(self,'display_reset_until',0);resetting=getattr(self,'display_resetting',False)
            if reset and (getattr(self,'display_reset_edge',False) or not resetting):
                if resetting:settings.set_bool(HISTORY_RESET,False)
                settings.set_bool(HISTORY_RESET,True);self.display_resetting,self.display_reset_edge=True,False
            elif not reset and resetting:
                settings.set_bool(HISTORY_RESET,False);self.display_resetting=False
        self.world.render();self.render_index=index+1

    def after_step(self,timers,tick):
        if tick!=self.tick_index:raise RuntimeError('Hooke physics clock mismatch')
        if tick%self.frame_ticks:return
        self.asset.update_spring_visual(self.stage,self.hanger,self.initial_hanger_z)
        if self.panel is not None:
            from coupled_oscillator.controls_asset import update_recorder_display
            shown=self.panel.display()
            update_recorder_display(self.stage,self.asset.CONTROLS_PANEL_PATH,shown)
            key=json.dumps(shown,sort_keys=True,default=str);changed=key!=getattr(self,'display_key',None);self.display_key=key
            if tick%(3*self.frame_ticks)==0:   # live tray field, 10 Hz
                changed=self.asset.update_tray_readout(self.stage,self._tray_reading()) or changed
            if changed:self.display_reset_until,self.display_reset_edge=getattr(self,'render_index',0)+DISPLAY_RESET_RENDERS,True
        self._render()
        views=[np.asarray(c.get_rgba())[...,:3].astype(np.uint8) for c in self.cameras]
        if any(v.shape!=(512,512,3) for v in views):raise RuntimeError('Hooke camera frame unavailable')
        self.writer.write(self.cv2.cvtColor(np.hstack(views),self.cv2.COLOR_RGB2BGR));self.frame_index+=1

    def support_state(self):
        rows=list(self.history)
        if len(rows)<30:return [],False,False
        z=np.array([r['z'] for r in rows])
        stable=np.ptp(z)<.0008 and np.max(np.abs(np.diff(z)))*30<.002
        supported=[]
        deck=self.asset.tray_deck_top_z(self.stage)
        for key,body in self.bodies.items():
            forces=np.array([r['forces'][key] for r in rows])
            gap=float(body.get_world_pose()[0][2])+self.bottom_offsets[key]-deck
            if np.isfinite(forces).all() and abs(float(forces.mean())-.05*9.81)<.06 and forces.min()>.2 and abs(gap)<.004:supported.append(key)
        return supported,bool(stable),all(row['released'] for row in rows)

    def tray_free(self):
        """Process record only (six levels), from the last 30 frame-log rows: the robot touched neither the
        tray (`tray_contact`, None = contact buffer invalid counts as touched) nor a weight standing on the tray deck
        (support force > 0.1 N), and no held weight pressed on the deck (> 0.1 N). A weight held clear of the tray does
        not matter. Reads the log only; no effect on the bench."""
        rows=list(self.history)
        if len(rows)<30:return False
        return all(tray_free_row(row) for row in rows)

    def observation(self):
        if self.panel is None:
            for _ in range(4):self.world.render()
        else:   # the displays as they are now, not the denoiser's memory of earlier digits
            until=max(getattr(self,'render_index',0)+4,getattr(self,'display_reset_until',0)+DISPLAY_SETTLE_RENDERS)
            while getattr(self,'render_index',0)<until or getattr(self,'display_resetting',False):self._render()
        frames=[]
        for path,camera in zip(self.asset.CAMERA_PATHS,self.cameras):
            rgba=np.asarray(camera.get_rgba());depth=np.asarray(camera.get_depth(device='cpu'))
            if rgba.shape!=(512,512,4) or depth.shape!=(512,512):raise RuntimeError('Hooke RGB-D unavailable')
            valid=np.isfinite(depth)&(depth>0)&(depth<65.535)
            mm=np.zeros(depth.shape,dtype='<u2');mm[valid]=np.maximum(1,np.rint(depth[valid]*1000)).astype('<u2')
            ok,png=self.cv2.imencode('.png',self.cv2.cvtColor(rgba[...,:3],self.cv2.COLOR_RGB2BGR))
            if not ok:raise RuntimeError('Hooke camera encoding failed')
            frames.append(dict(camera=path.rsplit('/',1)[-1],resolution=[512,512],rgb_png=png.tobytes(),depth_u16le_mm=mm.tobytes(),
                intrinsics=np.asarray(camera.get_intrinsics_matrix(device='cpu')).tolist(),world_from_camera_ros=np.linalg.inv(camera.get_view_matrix_ros(device='cpu')).tolist()))
        # Evidence images of this state: the three standard views (camera name -> PNG), nothing else.
        self.view_pngs={f['camera']:f['rgb_png'] for f in frames}
        right=self.cv2.cvtColor(self.cv2.imdecode(np.frombuffer(frames[1]['rgb_png'],np.uint8),self.cv2.IMREAD_COLOR),self.cv2.COLOR_BGR2RGB)
        reading,confidence,_=self.reader.read(right)
        self.automatic_reading=float(reading) if reading is not None and confidence>=.8 else None
        return frames

    def close(self):
        if self.closed:return
        self.closed=True;self.writer.release();self.audit_file.close()
        if getattr(self, 'reflex', None) is not None: self.reflex.close()   # private contact log

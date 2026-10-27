"""Coupled apparatus physical backend. No IK, task solver or robot policy.

The bench is operated by hand only (controls_asset.py): catches with CLAMP buttons and spring hooks, the release
post, the recorder keypad and the eddy-current damper lever. Each physics tick reads the caps, the lever carriage
and the catches, applies their effects (clamp toggles, key presses, hooks, release, damper) and logs every change
as an event row of private/physics.jsonl next to a compact state row (design: experiments/coupled_modes/design.md).
"""
from pathlib import Path
import json
import numpy as np
import collision_reflex
from .dynamics import Dynamics
from .model import sample_parameters

# Displays: RTX's DLSS keeps a history of static pixels, so redrawn digits stayed stale in the cameras. As
# chem/circuit_grid: hold the history reset for a few renders after a display change, and take an observation only
# after the reset has settled.
HISTORY_RESET='/rtx-transient/post/dlss/forceParamReset'
DISPLAY_RESET_RENDERS=3
DISPLAY_SETTLE_RENDERS=9

ROOT='/World/CoupledOscillatorLab'
ROBOT='/World/Franka'
HAND=ROBOT+'/panda_hand'
FINGERS=[ROBOT+'/'+n for n in ('panda_leftfinger','panda_rightfinger')]
CAMERAS=[ROBOT+'/RoboCasaCameras/robot0_agentview_left',
         ROBOT+'/RoboCasaCameras/robot0_agentview_right',HAND+'/robot0_eye_in_hand']
CENTERS=np.array([.43,.97])


class CoupledBackend:
    pendulum_ids=()

    def __init__(self,args,app):
        import carb,cv2,omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.physics_context import PhysicsContext
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot.manipulators import SingleManipulator
        from isaacsim.robot.manipulators.grippers import ParallelGripper
        from isaacsim.sensors.camera import Camera
        from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema
        from hooke_physics import configure_arm
        from .asset import build_apparatus
        from .instrument_visuals import update_visuals
        from . import controls_asset as CA
        from .controls import KEYS,RecorderPanel
        from damper_lever import Lever,lever_info
        self.CA=CA
        self.Gf,self.UsdGeom,self.Action=Gf,UsdGeom,ArticulationAction
        self.UsdPhysics,self.cv2,self.update_visuals=UsdPhysics,cv2,update_visuals
        self.output=Path(args.output);self.private=self.output/'private';self.private.mkdir(parents=True,exist_ok=True)
        # Registry instances inject parameters; legacy task ids sample from the public seed.
        params=getattr(args,'coupled_parameters',None) or sample_parameters(args.seed)
        self.dynamics=Dynamics(params.omega1,params.omega2,params.coupling_c,params.damping_gamma,params.mass_kg)
        (self.private/'parameters.json').write_text(json.dumps(vars(self.dynamics),indent=2))
        self.dt=1/args.physics_hz;self.tick_index=0;self.frame_ticks=args.physics_hz//30
        self.frame_index=0;self.closed=False;self.width_target=.08;self.recorder=None
        self.release_count=0;self.bar_contact_ticks=0
        self.bar_reset_observed=True;self.last_release_tick=None
        self.grip_ticks=[0,0,0,0]
        # Hand-operated controls: both sliders hooked to their catches, both catches clamped at zero, lever OFF.
        self.hooked=[True,True];self.clamped=[True,True];self.panel=RecorderPanel();self.panel_state=self.panel.state
        self.moved=False;self.post_gripped=False;self.audit=None;self.snapshot=None
        asset_path=self.private/'apparatus.usda'
        stage=Usd.Stage.CreateNew(str(asset_path));UsdGeom.SetStageMetersPerUnit(stage,1.)
        UsdGeom.SetStageUpAxis(stage,'Z');build_apparatus(stage,params,seed=args.seed)
        stage.SetDefaultPrim(stage.GetPrimAtPath(ROOT));stage.GetRootLayer().Save()
        source=Usd.Stage.Open(args.stage)
        for path in ('/World/HookeApparatus','/World/HookePhysics','/World/RobotRiser'):
            prim=source.GetPrimAtPath(path)
            if prim:prim.SetActive(False)
        source.DefinePrim(ROOT,'Xform').GetReferences().AddReference(str(asset_path))
        for prim in list(Usd.PrimRange(source.GetPrimAtPath(ROBOT))):
            if prim.IsInstance():prim.SetInstanceable(False)
        configure_arm(source,ROBOT)
        material=UsdShade.Material(source.GetPrimAtPath(ROOT+'/Materials/ContactMaterial'))
        for prim in Usd.PrimRange(source.GetPrimAtPath(ROBOT)):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                api=PhysxSchema.PhysxArticulationAPI.Apply(prim)
                # Preserve the proven robot velocity solver configuration.
                # Apparatus bodies keep their separately authored settings.
                api.CreateSolverPositionIterationCountAttr(64);api.CreateSolverVelocityIterationCountAttr(32)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api=UsdPhysics.RigidBodyAPI(prim);api.CreateVelocityAttr(Gf.Vec3f(0));api.CreateAngularVelocityAttr(Gf.Vec3f(0))
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                api=PhysxSchema.PhysxCollisionAPI.Apply(prim);api.CreateContactOffsetAttr(.0005);api.CreateRestOffsetAttr(0.)
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,materialPurpose='physics')
            if prim.IsA(UsdPhysics.PrismaticJoint) and 'finger' in prim.GetName():
                for api in list(prim.GetAppliedSchemas()):
                    if api.startswith('PhysxMimicJointAPI:'):prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI,api.split(':',1)[1])
                drive=UsdPhysics.DriveAPI.Apply(prim,'linear')
                drive.CreateStiffnessAttr(400.);drive.CreateDampingAttr(20.);drive.CreateMaxForceAttr(3.)
        # Resolve textures without modifying the frozen input stage.
        for prim in source.Traverse():
            for attr in prim.GetAttributes():
                if attr.GetTypeName()!=Sdf.ValueTypeNames.Asset:continue
                v=attr.Get()
                if v and str(v.path).lower().endswith(('.png','.jpg','.exr','.hdr')):
                    if str(v.path).endswith('normal.png'):attr.Set(Sdf.AssetPath('/workspace/normal.png'))
                    elif not v.resolvedPath or str(v.resolvedPath).startswith(('http:','https:')):attr.Set(Sdf.AssetPath(''))
        scene_path=self.private/'initial.usda';source.Flatten().Export(str(scene_path))
        omni.usd.get_context().open_stage(str(scene_path))
        for _ in range(30):app.update()
        self.stage=omni.usd.get_context().get_stage()
        carb.settings.get_settings().set_bool('/physics/updateToUsd',True)
        carb.settings.get_settings().set_bool('/physics/updateVelocitiesToUsd',True)
        self.world=World(stage_units_in_meters=1.,physics_dt=self.dt,rendering_dt=1/30)
        self.views=[self.world.scene.add(RigidPrim(ROOT+f'/Slider{i}',name=f'coupled_slider{i}',
                    reset_xform_properties=False,track_contact_forces=True,
                    contact_filter_prim_paths_expr=FINGERS,max_contact_count=1024)) for i in (1,2)]
        self.bar=self.world.scene.add(RigidPrim(ROOT+'/ReleaseBar',name='coupled_bar',reset_xform_properties=False,
                    track_contact_forces=True,contact_filter_prim_paths_expr=FINGERS,max_contact_count=1024))
        self.catch_view=self.world.scene.add(RigidPrim([ROOT+f'/Catch{i}' for i in (1,2)],name='coupled_catches',reset_xform_properties=False))
        self.cap_names=['clamp1','clamp2']+['key_'+k for k in KEYS]
        self.cap_view=self.world.scene.add(RigidPrim([ROOT+'/Clamp1Cap',ROOT+'/Clamp2Cap']+[ROOT+'/Keypad/Key_'+k for k in KEYS],
                    name='coupled_caps',reset_xform_properties=False))
        self.lever=Lever(lever_info(ROOT+'/DamperLever',**CA.DAMPER_LEVER),self.stage)
        self.lever_view=self.world.scene.add(RigidPrim(self.lever.info['carriage'],name='coupled_lever',reset_xform_properties=False,
                    track_contact_forces=True,contact_filter_prim_paths_expr=FINGERS,max_contact_count=1024))
        # Separate whole-scene contact sensors from the two-finger grip sensors.
        # Exact collider/body paths avoid wildcard expansion and include palms,
        # arm links and apparatus obstacles, not only the gripping fingers.
        contact_targets=set()
        for prim in self.stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            target=prim;ancestor=prim
            while ancestor and str(ancestor.GetPath())!='/':
                if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    target=ancestor;break
                ancestor=ancestor.GetParent()
            contact_targets.add(str(target.GetPath()))
        self.external_views=[];self.external_filters=[]
        for i in (1,2):
            path=ROOT+f'/Slider{i}'
            filters=sorted(p for p in contact_targets if p!=path and not p.startswith(path+'/'))
            self.external_filters.append(filters)
            self.external_views.append(self.world.scene.add(RigidPrim(path,name=f'coupled_external{i}',
                reset_xform_properties=False,track_contact_forces=True,
                contact_filter_prim_paths_expr=filters,max_contact_count=1024)))
        gripper=ParallelGripper(end_effector_prim_path=HAND,joint_prim_names=['panda_finger_joint1','panda_finger_joint2'],
                   joint_opened_positions=np.array([.04,.04]),joint_closed_positions=np.array([.008,.008]),action_deltas=np.array([.005,.005]))
        self.robot=self.world.scene.add(SingleManipulator(prim_path=ROBOT,name='coupled_franka',end_effector_prim_path=HAND,gripper=gripper))
        self.cameras=[self.world.scene.add(Camera(prim_path=p,name=f'coupled_camera{i}',resolution=(512,512))) for i,p in enumerate(CAMERAS)]
        if getattr(self.world,'_physics_context',None) is None:
            self.world._physics_context=PhysicsContext(physics_dt=self.dt,set_defaults=False)
        collision_reflex.prepare_scene(self.stage, ROBOT, self.private, ())   # colliders as drawn; contact reports
        self.world.initialize_physics();self.world.reset()
        self.reflex = collision_reflex.Reflex(ROBOT, self.dt, self.private)   # collision reflex
        standby=np.array([0.,-.785,0.,-2.356,0.,1.571,.785,.04,.04])
        self.robot.set_joints_default_state(positions=standby);self.robot.set_joint_positions(standby)
        self.robot.initialize();self.target=standby[:7].copy();self.velocity=np.zeros(7)
        self.latch_joints=[UsdPhysics.PrismaticJoint(self.stage.GetPrimAtPath(ROOT+f'/ReturnJoint{i}')) for i in (1,2)]
        for j in self.latch_joints:
            if j.GetBody0Rel().GetTargets():raise RuntimeError('rail must be world anchored')
            for name in ('stiffness','damping','maxForce'):
                if j.GetPrim().GetAttribute('drive:linear:physics:'+name).Get()!=0.:raise RuntimeError('duplicate rail drive')
        self.catch_joints=[UsdPhysics.PrismaticJoint(self.stage.GetPrimAtPath(ROOT+f'/CatchJoint{i}')) for i in (1,2)]
        self.hook_drives=[UsdPhysics.DriveAPI.Get(self.stage.GetPrimAtPath(ROOT+f'/HookJoint{i}'),'transX') for i in (1,2)]
        for c in self.cameras:c.initialize();c.add_distance_to_image_plane_to_frame()
        for _ in range(60):
            self._apply_robot();self.world.step(render=True)
        self.last_q=np.asarray(self.robot.get_joint_positions()[:7]).copy()
        self.cap_rest_z=np.asarray(self.cap_view.get_world_poses()[0],float)[:,2].copy();self.cap_down=[False]*len(self.cap_names)
        for sensor in [*self.views,self.bar,self.lever_view]:self._finger_forces(sensor)
        self.video_path=self.output/'coupled_three_views_raw.mp4'
        self.writer=cv2.VideoWriter(str(self.video_path),cv2.VideoWriter_fourcc(*'mp4v'),30,(1536,512))
        if not self.writer.isOpened():raise RuntimeError('video writer unavailable')
        self.audit=(self.private/'physics.jsonl').open('w')
        self._event('bench_state',dict(reason='start',**self.bench_state()))
        self.curves=[]
        for name,mat in [('ReturnSpring1','Return'),('CouplingSpring','Coupling'),('ReturnSpring2','Return')]:
            curve=UsdGeom.BasisCurves.Define(self.stage,ROOT+'/'+name)
            curve.CreateTypeAttr('linear');curve.CreateCurveVertexCountsAttr([61]);curve.CreateWidthsAttr([.004]);curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(UsdShade.Material(self.stage.GetPrimAtPath(ROOT+'/Materials/'+mat)))
            self.curves.append(curve)
        self._render()

    def _transform(self,path):
        return self.UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))

    def _state(self,view):
        return np.asarray(view.get_world_poses()[0][0]),np.asarray(view.get_linear_velocities()[0])

    def _finger_forces(self,view):
        v=np.asarray(view.get_contact_force_matrix(dt=self.dt))
        if v.size!=6 or not np.isfinite(v).all():raise RuntimeError('invalid finger contact sensor')
        return np.linalg.norm(v.reshape(2,3),axis=1)

    def public_calibration(self):
        from robot_calibration import public_robot_calibration
        return public_robot_calibration(self)

    def robot_state(self):
        q=np.asarray(self.robot.get_joint_positions())
        # Proprioception only, the same fields as on every bench: gripper_holding; no forces, no part names.
        return dict(joint_positions_rad=q[:7].tolist(),gripper_width_m=float(sum(q[7:9])),
                    gripper_holding=any(n>=round(.1/self.dt) for n in self.grip_ticks))

    def contact_channels(self):
        # Public physical contact instrumentation, identical for all participants.
        # This reports existing sensors; it does not identify a visual target.
        pairs=[self._finger_forces(v) for v in [*self.views,self.bar,self.lever_view]]
        return {name:dict(finger_force_n=[float(x) for x in force],
                         sustained_bilateral=bool(ticks>=round(.1/self.dt)))
                for name,force,ticks in zip(('CH1','CH2','release_bar','lever'),pairs,self.grip_ticks)}

    def set_gripper(self,width_m):self.width_target=width_m

    def _apply_robot(self):
        c=self.robot.get_articulation_controller()
        c.apply_action(self.Action(joint_positions=self.target,joint_velocities=self.velocity,joint_indices=np.arange(7)))
        c.apply_action(self.Action(joint_positions=np.array([self.width_target/2]*2),joint_indices=np.array([7,8])))

    def step(self,target,velocity=None):
        if self.closed:raise RuntimeError('backend closed')
        if target is not None:
            q=np.asarray(target);self.velocity=(q-self.target)/self.dt if velocity is None else np.asarray(velocity);self.target=q
        else:self.velocity=np.zeros(7)
        states=[self._state(v) for v in self.views]
        x=[float(s[0][0]-c) for s,c in zip(states,CENTERS)];v=[float(s[1][0]) for s in states]
        force=self.dynamics.forces_newton(x,v)
        # Eddy-current damper: viscous force in proportion to the magnet's overlap with the copper vane.
        damping=self.CA.DAMPING_ON_PER_S*self.lever.fraction
        for body,f,speed in zip(self.views,force,v):body.apply_forces(np.array([[f-self.dynamics.mass_kg*damping*speed,0,0]],dtype=np.float32),is_global=True)
        self._apply_robot();self.world.step(render=False);self.tick_index+=1;tick=self.tick_index
        q=np.asarray(self.robot.get_joint_positions()[:7]);rates=np.abs(q-self.last_q)/self.dt;speed=float(np.max(rates));self.last_q=q.copy()
        if not np.isfinite(q).all():raise RuntimeError('robot joint state is not finite')
        from control_contract import reset_window, window_rates; mean_rates = window_rates(self, q, self.dt)
        if float(np.max(mean_rates))>.9:
            from control_contract import protective_stop
            self.target=q.copy();self.velocity=np.zeros(7)   # protective stop: hold the measured pose
            self.pending_protective_stop=protective_stop(mean_rates); reset_window(self, q, self.dt)   # raised by the runtime after this tick
        stop = self.reflex.check(self.tick_index, collision_reflex.held_path(self))
        if stop is not None:   # collision reflex: hold the measured pose; raised by the runtime after this tick
            self.target = q.copy(); self.velocity = np.zeros(7)
            if getattr(self, 'pending_protective_stop', None) is None:
                self.pending_protective_stop = stop
        pairs=[self._finger_forces(v) for v in [*self.views,self.bar,self.lever_view]]
        self.grip_ticks=[n+1 if min(f)>.02 else 0 for n,f in zip(self.grip_ticks,pairs)]
        bar_y=float(self._state(self.bar)[0][1])
        if bar_y<-.225:self.bar_reset_observed=True
        gripped=self.grip_ticks[2]>=round(.1/self.dt)
        if gripped!=self.post_gripped:
            self.post_gripped=gripped;self._event('post',dict(gripped=gripped,post_mm=round((bar_y+.23)*1000,1)))
        self._controls(tick)
        # The release post starts the record keyed with REC: >= 35 mm of travel under a sustained two-finger grip, after the post
        # has been back at rest since the previous record. It opens both hooks in the same tick and starts the record.
        if self.panel.ready and bar_y>-.195 and gripped and self.bar_reset_observed:
            for i in (0,1):
                if self.hooked[i]:self._hook(i,False,'release')
            eid=self.recorder.begin_release(tick,self.panel.ready_s,physical_release_valid=True)
            self.panel.start(int(eid.rsplit(':',1)[-1]))
            self.release_count+=1;self.bar_reset_observed=False;self.last_release_tick=tick
            self._event('release',dict(evidence_id=eid,length_s=self.panel.recording_s,post_mm=round((bar_y+.23)*1000,1)))
        states=[self._state(v) for v in self.views]
        positions=[float(s[0][0]-c) for s,c in zip(states,CENTERS)];speeds=[float(s[1][0]) for s in states]
        catches,catch_v=self._catches()
        recording=bool(self.recorder and self.recorder.active)
        if not recording:   # the pawls are held open from the release until the record has ended
            for i in (0,1):
                if not self.hooked[i] and self.hook_rule(positions[i],catches[i],speeds[i],catch_v[i]):self._hook(i,True,'met')
        if not self.moved and (any(abs(c)>.001 for c in catches) or any(h and abs(p)>.001 for h,p in zip(self.hooked,positions))):
            self.moved=True;self._event('robot_moved',dict(x_mm=[round(p*1000,2) for p in positions],c_mm=[round(c*1000,2) for c in catches]))
        from .contact_summary import summarize_contacts
        external=False; summaries=[]
        for sensor,state,filters in zip(self.external_views,states,self.external_filters):
            count,summary=summarize_contacts(sensor.get_contact_force_data(dt=self.dt),filters,ROBOT)
            external=external or count>0
            if count:summaries.append(summary)
            if max(abs(float(state[0][1])),abs(float(state[0][2])-.99))>.0002:raise RuntimeError('slider left rail')
        summary=dict(categories=sorted({x for d in summaries for x in d['categories']}),
                     max_normal_force_n=max((d['max_normal_force_n'] for d in summaries),default=0.),
                     min_separation_m=min((d['min_separation_m'] for d in summaries),default=0.))
        if self.recorder:
            active=self.recorder.active
            self.recorder.sample(tick,positions,external_contact=external,brake_engaged=any(self.hooked) or self.lever.fraction>0,contact_diagnostics=summary)
            if active and not self.recorder.active:
                self.panel.finish();record=self.recorder.records[active['evidence_id']]
                self._event('record_end',dict(evidence_id=active['evidence_id'],status=record['status'],invalid_reasons=record['invalid_reasons']))
            elif self.recorder.active:self.panel.tick((tick-self.recorder.active['start_tick'])*self.dt)
        self._panel_changed()
        self.snapshot=dict(positions=positions,speeds=speeds,catches=catches,bar_y=bar_y)
        self.audit.write(json.dumps(dict(tick=tick,x_mm=[round(p*1000,2) for p in positions],c_mm=[round(c*1000,2) for c in catches],
            hooked=self.hooked,clamped=self.clamped,post_mm=round((bar_y+.23)*1000,1),grip=gripped,ext=external,
            damper=round(self.lever.fraction,3),speed=round(speed,3)))+'\n')
        return {}

    # --- hand-operated controls ---------------------------------------------------------------------------------
    @staticmethod
    def hook_rule(slider_m,catch_m,v_slider,v_catch):
        from .controls import hooks
        return hooks(slider_m,catch_m,v_slider,v_catch)

    def _event(self,kind,fields):
        if self.audit is None or self.audit.closed:return
        self.audit.write(json.dumps({'event':kind,'tick':self.tick_index,**fields})+'\n')   # a field's own tick wins

    def _catches(self):
        """Catch offsets from their zero marks along X and their X speeds (trusted, private)."""
        p=np.asarray(self.catch_view.get_world_poses()[0],float);v=np.asarray(self.catch_view.get_linear_velocities(),float)
        return [float(p[i,0]-self.CA.CATCH_ZERO_X[i]) for i in (0,1)],[float(v[i,0]) for i in (0,1)]

    def _controls(self,tick):
        """Push buttons (each acts once per press: down >= SWITCH_DOWN_M, up again <= SWITCH_UP_M) and the lever."""
        depth=self.cap_rest_z-np.asarray(self.cap_view.get_world_poses()[0],float)[:,2]
        for k,(name,d) in enumerate(zip(self.cap_names,depth)):
            if not self.cap_down[k] and d>=self.CA.SWITCH_DOWN_M:
                self.cap_down[k]=True
                if name.startswith('clamp'):self._clamp(int(name[-1])-1,not self.clamped[int(name[-1])-1])
                else:self._key(name[4:])
            elif self.cap_down[k] and d<=self.CA.SWITCH_UP_M:self.cap_down[k]=False
        carriage=np.asarray(self.lever_view.get_world_poses()[0],float)[0]
        for change in self.lever.update(tick,carriage[:2]):self._event('damper',change)

    def _clamp(self,i,on):
        offset=self._catches()[0][i];j=self.catch_joints[i];travel=self.CA.CATCH_TRAVEL_M
        j.GetLowerLimitAttr().Set(offset if on else -travel);j.GetUpperLimitAttr().Set(offset if on else travel)
        self.clamped[i]=on
        self._event('clamp',dict(catch=f'CH{i+1}',state='clamped' if on else 'free',c_mm=round(offset*1000,2)))

    def _hook(self,i,on,cause):
        drive=self.hook_drives[i]
        drive.GetStiffnessAttr().Set(self.CA.HOOK_K if on else 0.);drive.GetDampingAttr().Set(self.CA.HOOK_C if on else 0.)
        self.hooked[i]=on
        x=float(self._state(self.views[i])[0][0]-CENTERS[i])
        self._event('hook',dict(slider=f'CH{i+1}',state='hooked' if on else 'unhooked',cause=cause,x_mm=round(x*1000,2),
                                c_mm=round(self._catches()[0][i]*1000,2)))

    def _key(self,key):
        effect=self.panel.press(key)
        x=[round(float(self._state(v)[0][0]-c)*1000,2) for v,c in zip(self.views,CENTERS)]
        self._event('key',dict(key=key,effect=effect,entry=self.panel.entry,ready_s=self.panel.ready_s,hooked=list(self.hooked),x_mm=x))
        self._panel_changed()

    def _panel_changed(self):
        if self.panel.state!=self.panel_state:
            self._event('recorder',dict(old=self.panel_state,new=self.panel.state,number=self.panel.number,
                                        ready_s=self.panel.ready_s,recording_s=self.panel.recording_s))
            self.panel_state=self.panel.state

    def bench_state(self,reason=None):
        """State of the hand-operated controls (sealed as the terminal state; logged at start, stop and submission)."""
        snap=self.snapshot   # values of the last physics tick (also valid after the simulation has closed)
        if snap is None:
            states=[self._state(v) for v in self.views]
            snap=dict(positions=[float(s[0][0]-c) for s,c in zip(states,CENTERS)],speeds=[float(s[1][0]) for s in states],
                      catches=self._catches()[0],bar_y=float(self._state(self.bar)[0][1]))
        # released: no two-finger grip (both fingers in contact for 0.1 s) on a catch, the release post or the lever, as on airtrack_collision
        state=dict(released=not any(n>=round(.1/self.dt) for n in self.grip_ticks),
                   catches=[dict(offset_mm=round(c*1000,2),clamped=bool(k)) for c,k in zip(snap['catches'],self.clamped)],
                   sliders=[dict(offset_mm=round(p*1000,2),hooked=bool(h),speed_mm_s=round(abs(v)*1000,2))
                            for p,v,h in zip(snap['positions'],snap['speeds'],self.hooked)],
                   damper=self.lever.record(),
                   recorder=dict(self.panel.record(),ready=self.panel.ready,recording=self.panel.recording),
                   post_mm=round((snap['bar_y']+.23)*1000,1))
        if reason is not None:self._event('bench_state',dict(reason=reason,**state))
        return state

    def _display_render(self):
        """world.render with the DLSS history reset held for a few renders after a display change."""
        import carb
        settings = carb.settings.get_settings()
        index = getattr(self, 'render_index', 0)
        reset = index < getattr(self, 'display_reset_until', 0); resetting = getattr(self, 'display_resetting', False)
        if reset and (getattr(self, 'display_reset_edge', False) or not resetting):
            if resetting: settings.set_bool(HISTORY_RESET, False)
            settings.set_bool(HISTORY_RESET, True); self.display_resetting, self.display_reset_edge = True, False
        elif not reset and resetting:
            settings.set_bool(HISTORY_RESET, False); self.display_resetting = False
        self.world.render(); self.render_index = index + 1

    def _display_changed(self):
        self.display_reset_until, self.display_reset_edge = getattr(self, 'render_index', 0) + DISPLAY_RESET_RENDERS, True

    def _render_for_capture(self):
        """Render until the cameras show the displays as they are now (at least 4 renders, as before)."""
        until = max(getattr(self, 'render_index', 0) + 4, getattr(self, 'display_reset_until', 0) + DISPLAY_SETTLE_RENDERS)
        while getattr(self, 'render_index', 0) < until or getattr(self, 'display_resetting', False): self._display_render()

    def after_step(self,timers,tick):
        if tick!=self.tick_index:raise RuntimeError('clock mismatch')
        if tick%self.frame_ticks==0:self._render()

    def _render(self):
        x1,x2=[self._state(v)[0][0] for v in self.views]
        for curve,(a,b) in zip(self.curves,[(.05,x1-.12),(x1+.12,x2-.12),(x2+.12,1.35)]):
            curve.GetPointsAttr().Set([self.Gf.Vec3f(float(a+(b-a)*t),float(.04+.018*np.sin(t*18*np.pi)),float(1.005+.018*np.cos(t*18*np.pi))) for t in np.linspace(0,1,61)])
        if self.recorder:
            self.update_visuals(self.stage,[ROOT+f'/EncoderDisplay{i}/Digits' for i in (1,2)],self.recorder.latest)
        self.CA.update_recorder_display(self.stage,ROOT,self.panel.display())
        for i in (0,1):self.CA.set_lamp(self.stage,ROOT,i,self.clamped[i])
        row=self.recorder.latest if self.recorder else None
        key=json.dumps([None if row is None else [None if row[k] is None else round(row[k]*10) for k in ('x1_mm','x2_mm')],
                        self.panel.display(),[bool(c) for c in self.clamped]],sort_keys=True,default=str)
        if key!=getattr(self,'display_key',None):self.display_key=key;self._display_changed()
        for _ in range(4):self._display_render()
        rgba=[np.asarray(c.get_rgba()) for c in self.cameras]
        if any(a.shape!=(512,512,4) for a in rgba):raise RuntimeError('camera frames unavailable')
        self.writer.write(self.cv2.cvtColor(np.concatenate([a[:,:,:3] for a in rgba],axis=1),self.cv2.COLOR_RGB2BGR))
        self.frame_index+=1

    def observation(self):
        self._render_for_capture()
        result=[]
        for path,c in zip(CAMERAS,self.cameras):
            rgba=np.asarray(c.get_rgba());depth=np.asarray(c.get_depth(device='cpu'))
            if rgba.shape!=(512,512,4) or depth.shape!=(512,512):raise RuntimeError('RGB-D unavailable')
            valid=np.isfinite(depth)&(depth>0)&(depth<65.535);mm=np.zeros(depth.shape,dtype='<u2')
            mm[valid]=np.maximum(1,np.rint(depth[valid]*1000)).astype('<u2')
            ok,png=self.cv2.imencode('.png',self.cv2.cvtColor(rgba[:,:,:3],self.cv2.COLOR_RGB2BGR))
            if not ok:raise RuntimeError('image encoding failed')
            result.append(dict(camera=path.rsplit('/',1)[-1],resolution=[512,512],rgb_png=png.tobytes(),depth_u16le_mm=mm.tobytes(),
                  intrinsics=np.asarray(c.get_intrinsics_matrix(device='cpu')).tolist(),world_from_camera_ros=np.linalg.inv(c.get_view_matrix_ros(device='cpu')).tolist()))
        return result

    def close(self):
        if self.closed:return
        self.closed=True;self.writer.release();self.audit.close()
        if getattr(self, 'reflex', None) is not None: self.reflex.close()   # private contact log

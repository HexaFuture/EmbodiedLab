"""projectile_landing robot backend. Launch, ball flight and cup contacts are native rigid physics.

The spring is the plunger joint's drive (private stiffness). The ratchet pawl is a stiff stop on the same drive at the
deepest notch the plunger has been pushed past; the trigger hands the plunger back to the spring. The published
short-range constrained gripper is shared with the torsion lab. Nothing assigns an object pose or velocity after the
initial conditions, with one integrator correction: in the tick where the ball centre passes the rail ends, its state at
the end of that tick is replaced by the exact free flight from the sub-tick moment it left the rails (rail support
does not end on a tick boundary in nature), and the rails do not collide with the ball until it is clear of them.
Without it the range would depend on which 240 Hz tick the ball leaves on (a sawtooth in compression). The coil spring and the pawl are drawn from this state before every rendered frame (render only).
"""
from pathlib import Path
import json
import numpy as np
import collision_reflex
import grip_contact
from torsion_backend import TorsionBackend
from .bench import build_bench, ROOT, ORIGIN
from .geometry import (AXIS, BALL_RADIUS, NOTCHES, LATCH_HYSTERESIS, DASHPOT_DELAY_S, DASHPOT_DAMPING, CUP_CONTACT_TOLERANCE_M,
                       CUP_INNER_WIDTH, TRIGGER_LOCAL, MUZZLE, ball_rest)
from .measurement import FlightRecorder
from .rail_exit import exact_exit, RAIL_CLEAR_U


class ProjectileBackend(TorsionBackend):
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
        self.lab,self.Gf,self.UsdGeom,self.Action,self.cv2=lab,Gf,UsdGeom,ArticulationAction,cv2
        self.dt=1/args.physics_hz;self.tick_index=0;self.frame_ticks=args.physics_hz//30
        self.output=Path(args.output);self.private=self.output/'private';self.private.mkdir(parents=True,exist_ok=True)
        self.audit_file=(self.private/'physics.jsonl').open('w')
        self.closed=False;self.frame_index=0;self.held=self.releasing=None
        self.grasp_broken=False;self.previous_jaw=None;self.width_target=.08
        self.spring_k=float(parameters['spring_k']);self.target_notch=float(parameters['target_notch_m'])
        self.records={};self.flight=None;self.trigger_ticks=0
        self.panel=None   # the shared torsion runtime's recorder panel: none on this bench
        # Launcher state: the notch the pawl holds (None: free), the tick of the last release, and whether the pawl
        # is armed again (the plunger came back to its free length after a release).
        self.latched=None;self.release_tick=None;self.pawl_armed=True;self.drive_mode=None
        self.episode_id=None;self.private_continuation=False;self.outcome=None
        self.contact_during_flight=False;self.cup_contact_before_measurement=False
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
        build_bench(source,self.spring_k)
        hooke_physics.configure_arm(source,lab.ROBOT_PATH)
        lab.setup_neutral_lab_lighting(source)
        source.Flatten().Export(str(self.private/'initial.usda'))
        omni.usd.get_context().open_stage(str(self.private/'initial.usda'))
        for _ in range(30):app.update()
        self.stage=omni.usd.get_context().get_stage()
        self.spring=UsdPhysics.DriveAPI(self.stage.GetPrimAtPath(ROOT+'/Spring'),'linear')
        carb.settings.get_settings().set_bool('/physics/updateToUsd',True)
        carb.settings.get_settings().set_bool('/physics/updateVelocitiesToUsd',True)
        self.world=World(stage_units_in_meters=1.,physics_dt=self.dt,rendering_dt=1/30)
        self.paths={'ball':ROOT+'/Ball','cup':ROOT+'/Cup'}
        self.bodies={k:self.world.scene.add(SingleRigidPrim(p,name='projectile_'+k,reset_xform_properties=False)) for k,p in self.paths.items()}
        self.button=self.world.scene.add(SingleRigidPrim(ROOT+'/TriggerButton',name='projectile_button',reset_xform_properties=False))
        self.plunger=self.world.scene.add(SingleRigidPrim(ROOT+'/Plunger',name='projectile_plunger',reset_xform_properties=False))
        robot_parts=[lab.ROBOT_PATH+'/'+n for n in ('panda_hand','panda_leftfinger','panda_rightfinger')]+[lab.ROBOT_PATH+'/panda_link'+str(i) for i in range(8)]
        robot_parts=[p for p in robot_parts if self.stage.GetPrimAtPath(p)]
        def sensor(path,name,filters):
            return self.world.scene.add(RigidPrim(path,name=name,reset_xform_properties=False,track_contact_forces=True,
                contact_filter_prim_paths_expr=filters,max_contact_count=128))
        self.robot_sensors={k:sensor(p,k+'_robot_contact',robot_parts) for k,p in self.paths.items()}
        self.finger_sensors=grip_contact.finger_contact_sensors(self.world.scene,RigidPrim,lab.ROBOT_PATH,self.paths,'projectile')
        self.grip=grip_contact.GripState()
        self.trigger_sensor=sensor(ROOT+'/TriggerButton','trigger_robot_contact',robot_parts)
        self.cup_sensor=sensor(ROOT+'/Ball','ball_cup_contact',[ROOT+'/Cup'])
        self.face_sensor=sensor(ROOT+'/Ball','ball_plunger_contact',[ROOT+'/Plunger'])
        self.plunger_robot_sensor=sensor(ROOT+'/Plunger','plunger_robot_contact',robot_parts)
        supports=[ROOT+'/RailL',ROOT+'/RailR',ROOT+'/BallLoadingPad',ROOT+'/CupLoadingPad',ROOT+'/ReceivingDeck',ROOT+'/Cup']
        supports=[p for p in supports if self.stage.GetPrimAtPath(p)]
        from torsion_backend import _SupportSum
        self.support_sensors={k:_SupportSum(sensor(p,k+'_support',[t for t in supports if t!=p])) for k,p in self.paths.items()}
        gripper=ParallelGripper(end_effector_prim_path=lab.ROBOT_PATH+'/panda_rightfinger',
            joint_prim_names=['panda_finger_joint1','panda_finger_joint2'],joint_opened_positions=np.array([.04,.04]),
            joint_closed_positions=np.array([.009,.009]),action_deltas=np.array([.01,.01]))
        self.robot=self.world.scene.add(SingleManipulator(prim_path=lab.ROBOT_PATH,name='projectile_franka',
            end_effector_prim_path=lab.ROBOT_PATH+'/panda_rightfinger',gripper=gripper))
        self.cameras=[self.world.scene.add(Camera(prim_path=p,name='projectile_cam'+str(i),resolution=(512,512))) for i,p in enumerate(lab.CAMERA_PATHS)]
        if getattr(self.world,'_physics_context',None) is None:self.world._physics_context=PhysicsContext(physics_dt=self.dt,set_defaults=False)
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
        from .geometry import MUZZLE, face_rest
        self.axis=np.asarray(AXIS);self.plunger_rest=np.asarray(ORIGIN)+np.asarray(face_rest())-.015*self.axis
        self.video_path=self.output/'projectile_three_views_raw.mp4'
        self.writer=cv2.VideoWriter(str(self.video_path),cv2.VideoWriter_fourcc(*'mp4v'),30,(1536,512))
        if not self.writer.isOpened():raise RuntimeError('projectile video writer unavailable')
        from omni.physx import get_physx_interface
        from omni.physx.bindings._physx import SimulationEvent
        from pxr import PhysicsSchemaTools
        def joint_event(event):
            if event.type==int(SimulationEvent.JOINT_BREAK) and str(PhysicsSchemaTools.decodeSdfPath(*event.payload['jointPath']))=='/World/HookePhysics/GraspConstraint':
                self.grasp_broken=True
        self._subscription=get_physx_interface().get_simulation_event_stream_v2().create_subscription_to_pop(joint_event)
        self.stage.GetRootLayer().Export(str(self.private/'runtime_scene.usda'))
        (self.private/'configuration.json').write_text(json.dumps(parameters,indent=2))

    def local_ball(self):return np.asarray(self.bodies['ball'].get_world_pose()[0])-ORIGIN

    # --- render-only launcher parts (visuals.py): coil follows the plunger, pawl shows the latched state ---------
    def _update_visuals(self):
        from . import visuals
        travel=round(self.travel(),5)
        if travel!=getattr(self,'_coil_travel',None):
            visuals.set_coil(self.stage,ROOT+'/SpringCoil',travel);self._coil_travel=travel;self._display_changed()
        engaged=self.latched is not None
        if engaged!=getattr(self,'_pawl_engaged',None):
            visuals.set_pawl(self.stage,ROOT+'/Plunger',engaged);self._pawl_engaged=engaged;self._display_changed()

    def after_step(self,timers,tick):
        """Video frame of the three standard views every rendered tick. A camera whose render product has no frame
        yet (seen on a loaded host right after the start) is rendered again; if it still has none, that video frame
        is skipped and noted in the private audit instead of failing the episode."""
        if tick!=self.tick_index:raise RuntimeError('projectile physics clock mismatch')
        if tick%self.frame_ticks:return
        self._update_visuals()
        for attempt in range(4):
            self._display_render()
            views=[np.asarray(c.get_rgba()) for c in self.cameras]
            if all(v.ndim==3 and v.shape[:2]==(512,512) for v in views):break
        else:
            self.audit_file.write(json.dumps(dict(event='video_frame_skipped',tick=tick,
                shapes=[list(v.shape) for v in views]))+'\n');self.audit_file.flush()
            return
        views=[v[...,:3].astype(np.uint8) for v in views]
        self.writer.write(self.cv2.cvtColor(np.hstack(views),self.cv2.COLOR_RGB2BGR));self.frame_index+=1

    def observation(self):
        self._update_visuals()
        return super().observation()

    # --- launcher ---------------------------------------------------------------------------------
    def travel(self):
        """Plunger compression from its free length along the launch axis (m, positive backwards)."""
        return float(-(np.asarray(self.plunger.get_world_pose()[0])-self.plunger_rest)@self.axis)

    def _in_launch_window(self):
        return self.release_tick is not None and (self.tick_index-self.release_tick)*self.dt<DASHPOT_DELAY_S

    def _launcher_drive(self):
        """Pawl and dashpot for the coming tick: a stiff stop at the latched notch, else the spring."""
        s=self.travel();window=self._in_launch_window()
        if not self.pawl_armed and not window and s<.003:self.pawl_armed=True
        if self.pawl_armed:
            passed=[n for n in NOTCHES if s>=n+LATCH_HYSTERESIS]
            if passed and (self.latched is None or passed[-1]>self.latched):self.latched=passed[-1]
        if self.latched is not None and s<self.latched+.0003:mode=('latched',self.latched)
        else:mode=('spring',window)
        if mode==self.drive_mode:return
        stiffness,target,damping=((2e4,-self.latched,200.) if mode[0]=='latched' else
                                  (self.spring_k,0.,0. if window else DASHPOT_DAMPING))
        self.spring.GetStiffnessAttr().Set(float(stiffness));self.spring.GetTargetPositionAttr().Set(float(target))
        self.spring.GetDampingAttr().Set(float(damping));self.drive_mode=mode

    def _rail_collision(self,enabled):
        from pxr import UsdPhysics
        for name in ('RailL','RailR'):
            UsdPhysics.CollisionAPI(self.stage.GetPrimAtPath(ROOT+'/'+name)).GetCollisionEnabledAttr().Set(bool(enabled))
        self.rails_off=not enabled

    def _rail_exit(self,p0,v0):
        """Exact sub-tick rail exit (see the module note): once, in the tick the ball centre passes the rail ends."""
        p1=self.local_ball();v1=np.asarray(self.bodies['ball'].get_linear_velocity(),float)
        if not float((p0-MUZZLE)@self.axis)<0.<=float((p1-MUZZLE)@self.axis):return
        position,velocity,fraction,pe,ue=exact_exit(p0,p1,v0,v1,self.dt,np.asarray(MUZZLE),self.axis)
        orientation=self.bodies['ball'].get_world_pose()[1]
        self.bodies['ball'].set_world_pose(position=np.asarray(ORIGIN)+position,orientation=orientation)
        self.bodies['ball'].set_linear_velocity(velocity)
        self._rail_collision(False);self.exit_done=True
        self.exit_state=dict(tick=self.tick_index,fraction=fraction,exit_position=pe.tolist(),exit_velocity=ue.tolist(),
                             engine_position=p1.tolist(),engine_velocity=v1.tolist())
        self.audit_file.write(json.dumps(dict(event='rail_exit',**self.exit_state))+'\n')

    def ball_in_cradle(self):
        """Ball on the rails ahead of the plunger, not held (the shot is recorded); seated: at rest against the face."""
        if self.held=='ball':return False,False
        rest=np.asarray(ball_rest(self.travel()))
        offset=self.local_ball()-rest;along=float(offset@self.axis);across=offset-along*self.axis
        on_rails=bool(np.linalg.norm(across)<.006 and -.004<along<.15)
        seated=bool(on_rails and along<.003 and np.linalg.norm(self.bodies['ball'].get_linear_velocity())<.003)
        return on_rails,seated

    def fire(self,private=False):
        """The trigger: the pawl lets go and the spring drives the plunger. Records the shot if the ball is in the cradle."""
        if self.flight is not None:return False
        notch=self.latched;on_rails,seated=self.ball_in_cradle();travel=self.travel()
        self.latched=None;self.pawl_armed=False;self.release_tick=self.tick_index
        if notch is None or not on_rails:return False
        self.flight=FlightRecorder(self.local_ball(),self.dt,notch,travel);self.flight_seated=seated
        self.private_continuation=private;self.contact_during_flight=False;self.cup_contact_before_measurement=False
        self.entered=False;self.retained_ticks=0;self.exit_done=False
        self.cup_at_release=[np.asarray(x).tolist() for x in self.bodies['cup'].get_world_pose()]
        self.cup_moved=False
        return True

    def _flight_step(self):
        if self.flight is None:return
        previous_landing=self.flight.landing
        self.flight.step(self.local_ball())
        robot_contact=bool(any(self._touching(s) for s in self.robot_sensors.values())
                           or self._touching(self.plunger_robot_sensor) or self.held is not None)
        self.contact_during_flight |= robot_contact
        # Private record only: the same robot contact from the trigger up to and including the tick
        # of the curtain crossing, and its first tick; robot_contact (invalid_reasons) still covers the whole 4 s.
        if getattr(self,'_contact_flight',None) is not self.flight:
            self._contact_flight=self.flight;self.contact_before_curtain=False;self.contact_first_tick=None
            self.ball_plunger_contact_before_curtain=False;self.ball_cup_contact_before_curtain=False
        if robot_contact and self.contact_first_tick is None:self.contact_first_tick=self.tick_index
        # Check contacts from the same tick as the first optical crossing too.
        if previous_landing is None:
            self.contact_before_curtain |= robot_contact
            cup_touch=self._touching(self.cup_sensor)
            self.cup_contact_before_measurement |= cup_touch
            # Private record only (six-level scoring): what makes a trial shot invalid is the robot on the
            # ball or the plunger, or the ball on the cup, from the trigger to the curtain; holding the cup is not.
            self.ball_plunger_contact_before_curtain |= bool(self._touching(self.robot_sensors['ball'])
                or self._touching(self.plunger_robot_sensor) or self.held=='ball')
            self.ball_cup_contact_before_curtain |= bool(cup_touch)
        cup_pose=self.bodies['cup'].get_world_pose()
        self.cup_moved |= previous_landing is None and bool(np.linalg.norm(cup_pose[0]-self.cup_at_release[0])>.002 or abs(np.dot(cup_pose[1],self.cup_at_release[1]))<.999)
        local=self._transform(ROOT+'/Cup').GetInverse().Transform(self.Gf.Vec3d(*map(float,self.bodies['ball'].get_world_pose()[0])))
        inside=abs(local[0])<CUP_INNER_WIDTH/2-BALL_RADIUS+CUP_CONTACT_TOLERANCE_M and abs(local[1])<CUP_INNER_WIDTH/2-BALL_RADIUS+CUP_CONTACT_TOLERANCE_M and -.016+BALL_RADIUS-.002<local[2]<.027
        self.entered |= bool(inside)
        self.retained_ticks=self.retained_ticks+1 if inside else 0
        if self.tick_index-self.release_tick>=round(4/self.dt):
            public=self.flight.export();private=self.flight.private();invalid=[]
            if self.flight.landing is None or len(self.flight.events)!=2:invalid.append('incomplete_flight')
            if self.contact_during_flight:invalid.append('robot_contact')
            if not self.flight_seated:invalid.append('ball_not_seated_against_plunger')
            if not self.private_continuation and abs(private['notch_m']-self.target_notch)<.0005:invalid.append('target_notch_used_before_submission')
            if self.private_continuation:
                self.outcome=dict(episode_id=self.episode_id,completed=True,launch_tick=self.release_tick,
                    notch_m=private['notch_m'],ball_seated=self.flight_seated,cup_frozen=not self.cup_moved,
                    no_robot_contact=not self.contact_during_flight,range_measured_before_cup_contact=not self.cup_contact_before_measurement,
                    entered_cup=self.entered,retained_in_cup=self.retained_ticks>=round(.5/self.dt),
                    distance_m=None if public['landing'] is None else public['landing']['distance_m'],public_measurement=public,
                    private_measurement=private)
            else:
                from .recording import seal_record
                number=len(self.records)+1;eid=f'{self.episode_id}:flight:{number}'
                private=dict(private,robot_contact_before_curtain=bool(self.contact_before_curtain),
                             robot_contact_first_tick=self.contact_first_tick,
                             robot_ball_or_plunger_contact_before_curtain=bool(self.ball_plunger_contact_before_curtain),
                             ball_cup_contact_before_curtain=bool(self.ball_cup_contact_before_curtain))
                self.records[eid]=seal_record(eid,self.episode_id,public,self.release_tick,self.tick_index,invalid,private,number)
            self.flight=None

    def step(self,target,velocity=None):
        if self.closed:raise RuntimeError('backend closed')
        previous=self.target.copy()
        if target is not None:self.target=np.asarray(target)
        controller=self.robot.get_articulation_controller()
        controller.apply_action(self.Action(joint_positions=self.target,joint_velocities=(self.target-previous)/self.dt if velocity is None else np.asarray(velocity),joint_indices=np.arange(7)))
        controller.apply_action(self.Action(joint_positions=np.full(2,self.grip.drive_width(self.width_target,self.robot)/2),joint_indices=np.array([7,8])))
        self._launcher_drive()
        on_rails=self.flight is not None and not self.exit_done and self.held is None
        if on_rails:before=(self.local_ball(),np.asarray(self.bodies['ball'].get_linear_velocity(),float))
        self.world.step(render=False);self.tick_index+=1
        if on_rails:self._rail_exit(*before)
        if getattr(self,'rails_off',False) and (self.flight is None or float((self.local_ball()-MUZZLE)@self.axis)>RAIL_CLEAR_U):
            self._rail_collision(True)
        # Trusted diagnosis of contact overloads, before a broken grip is
        # removed. These vectors never enter the public observation/tools.
        held_before=self.held
        support_before=(np.asarray(self.support_sensors[held_before].get_contact_force_matrix(dt=self.dt)).tolist()
                        if held_before else None)
        face_force=np.asarray(self.face_sensor.get_contact_force_matrix(dt=self.dt)).tolist()
        self._gripper_mechanism()
        q=np.asarray(self.robot.get_joint_positions()[:7]);rates=np.abs(q-self.last_q)/self.dt;self.last_q=q.copy()
        if not np.isfinite(q).all():raise RuntimeError('robot joint state is not finite')
        from control_contract import reset_window, window_rates; mean_rates = window_rates(self, q, self.dt)
        if float(np.max(mean_rates))>.9:
            self.audit_file.write(json.dumps(dict(event='robot_velocity_guard',tick=self.tick_index,
                robot_joints=q.tolist(),robot_rates=rates.tolist(),
                target_joints=self.target.tolist(),target_velocity=((self.target-previous)/self.dt if velocity is None else np.asarray(velocity)).tolist(),
                native_joint_velocity=np.asarray(self.robot.get_joint_velocities()[:7]).tolist()))+'\n')
            self.audit_file.flush()
            from control_contract import protective_stop
            self.target=q.copy()   # protective stop: hold the measured pose
            self.pending_protective_stop=protective_stop(mean_rates); reset_window(self, q, self.dt)   # raised by the runtime after this tick
        stop = self.reflex.check(self.tick_index, collision_reflex.held_path(self))
        if stop is not None:   # collision reflex: hold the measured pose; raised by the runtime after this tick
            self.target = q.copy()
            if getattr(self, 'pending_protective_stop', None) is None:
                self.pending_protective_stop = stop
        travel=float(self.button.get_world_pose()[0][2]-(ORIGIN[2]+TRIGGER_LOCAL[2]))
        self.trigger_ticks=self.trigger_ticks+1 if travel<-.005 and self._touching(self.trigger_sensor) else 0
        if self.trigger_ticks==round(.04/self.dt):self.fire()
        self._flight_step()
        self.audit_file.write(json.dumps(dict(tick=self.tick_index,ball=self.local_ball().tolist(),ball_velocity=self.bodies['ball'].get_linear_velocity().tolist(),held=self.held,
            held_before_step=held_before,held_support_force=support_before,ball_face_force=face_force,plunger_travel=self.travel(),
            latched=self.latched,drive_mode=list(self.drive_mode),trigger_travel=travel,robot_peak_rate=float(max(rates)),robot_joints=q.tolist(),
            target_joints=self.target.tolist()))+'\n')
        return {}

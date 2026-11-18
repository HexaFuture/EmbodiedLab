"""Public calibration tools and sealed private post-submission validation."""
import json
from copy import deepcopy
from device_runtime import DeviceRuntime
from instance_episodes import InstanceMixin
from .geometry import BALL_MASS, PLUNGER_MASS, TILT_DEG, NOTCHES, RAIL_FRICTION, DASHPOT_DELAY_S

TOOLS=('get_task_spec','get_observation','get_robot_state','execute_joint_trajectory','set_gripper','wait',
       'export_flight','submit_result')


class ProjectileEpisode(InstanceMixin,DeviceRuntime):
    def __init__(self,backend,instance,output,**kwargs):
        super().__init__(backend,**kwargs);self._instance_init(instance,output)
        backend.episode_id=self.episode_id

    def public_task(self):
        # Identity, operation and public physical quantities only: no positions, layout or part sizes (the Agent
        # finds where things are from the calibrated cameras).
        return self.instance_public_task(dict(tools=list(TOOLS),apparatus=dict(
            ball_mass_kg=BALL_MASS,ball_model='homogeneous solid sphere',plunger_mass_kg=PLUNGER_MASS,
            launch_tilt_deg=TILT_DEG,rail_friction_coefficient=RAIL_FRICTION,
            notches_m=list(NOTCHES),target_notch_m=self.backend.target_notch,
            notch_scale=('White scale plate with one black tick per notch, numbered in mm, at the knob position that '
                         'latches that notch. The green pointer on the knob reads the plunger compression against it.'),
            cocking=('Pushing the knob back along the launch axis compresses the spring; the ratchet pawl on the knob '
                     'drops into the deepest rack tooth passed by at least 0.6 mm and holds that notch when the knob is '
                     'let go. The pawl cannot be lifted by hand: only the trigger frees a latched plunger.'),
            release=('A shot is recorded when the trigger frees a latched plunger with the ball in the cradle in front of '
                     f'the plunger face; a dashpot settles the plunger {DASHPOT_DELAY_S} s after the release, and the pawl '
                     're-arms once the plunger is back at its free length. The trigger with no ball fires empty.'),
            trigger='Purple push-button cap, with a white plate labelled FIRE lying beside it.',
            recovery='A shallow V pocket in the receiving deck gathers the ball after a shot.',
            final=('After the submission a private four-second continuation releases the same pawl and verifies the '
                   'catch; nothing places or pushes the ball: the spring launches it as in a shot of your own.'))))

    def robot_state(self):return super().robot_state()|{'gripper_holding':self.backend.held is not None}

    def _dispatch(self,name,args):
        if self.terminated:raise ValueError('episode terminated')
        if name not in TOOLS or not isinstance(args,dict):raise ValueError('unknown projectile tool')
        if name=='get_task_spec' and not args:return self.public_task()
        if name=='export_flight' and not args:
            # The timer memory: every completed shot record of this episode, oldest first.
            return dict(records=[deepcopy(r['public']) for r in self.backend.records.values()])
        if name=='export_flight' and set(args)=={'evidence_id'}:
            if args['evidence_id'] not in self.backend.records:raise ValueError('unknown evidence_id')
            return deepcopy(self.backend.records[args['evidence_id']]['public'])
        if name=='submit_result':
            # Freeze first, even for malformed answers or an invalid setup.
            self.terminated=True
            outcome=dict(completed=False,episode_id=self.episode_id)
            try:
                # This settling tick belongs to the trusted continuation and is
                # separately recorded; no further participant actions are allowed.
                self.backend.step(None);self.backend.after_step({},self.backend.tick_index)
                if not self.backend.fire(private=True):raise ValueError('no latched plunger with the ball in the cradle')
                for _ in range(round(4*self.limits.physics_hz)):
                    self.backend.step(None);self.backend.after_step({},self.backend.tick_index)
                outcome=self.backend.outcome or outcome
            except Exception as exc:
                outcome['error']=type(exc).__name__+': '+str(exc)
            return self.freeze_and_seal(args,{'private_validation_completed':outcome.get('completed',False)},
                {'target_notch_m':self.backend.target_notch,'private_validation_s':4.},self.backend.records,
                files=(('outcome.json',json.dumps(outcome,allow_nan=False).encode()),))
        return super()._dispatch(name,args)

    def seal_payload(self):
        """The records at stop; the submission's private continuation does not run here."""
        return ({'private_validation':'not_run'},
                {'target_notch_m':self.backend.target_notch,'private_validation_s':4.},self.backend.records)

    def evaluation(self):return self.instance_evaluation()

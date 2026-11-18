"""Hooke evidence and scoring over participant-neutral fixed-tick controls."""
import base64
import hashlib
from device_runtime import DeviceRuntime

PUBLIC_TOOLS=('get_task_spec','get_observation','get_robot_state','execute_joint_trajectory','set_gripper','wait','submit_result')


def initialize_hooke_episode(backend, score, *, physics_hz, horizon_ticks):
    """Restore the original empty-apparatus initialization, before user actions.

    Ten open-gripper frames + 120 settle frames are apparatus preparation.
    The subsequent 30 recorded frames count against the episode budget, as in
    the working Hooke worker. No experiment, loading or reading is automated.
    """
    backend.set_gripper(.08)
    for _ in range(130*(physics_hz//30)):
        backend.step(None)
        backend.after_step({}, backend.tick_index)
    supported, stable, released = backend.support_state()
    if supported or not stable or not released:
        raise RuntimeError('empty spring failed to settle')
    backend.tick_index = 0
    backend.history.clear()
    backend.start_seat_log()    # weight box occupancy from the episode clock's zero (process-score records)
    episode = HookeRuntime(backend, score, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    receipt = episode.call('initial-empty-stabilization', 'wait', {'seconds': 1.})
    if not receipt['ok'] or not backend.support_state()[1]:
        raise RuntimeError('initial empty evidence window is not stable')
    return episode


class HookeRuntime(DeviceRuntime):
    def __init__(self,backend,score,**kwargs):
        super().__init__(backend,**kwargs)
        self.score=score
        self.episode_id=score.episode_id

    def public_task(self):
        return self.score.public_task() | {'tools':list(PUBLIC_TOOLS),'physics_hz':self.limits.physics_hz,
            'horizon_ticks':self.horizon_ticks,'device_rpc_limit':65536,'evaluation_ready':False,
            'benchmark_version':'hooke-neutral-device/v1-dev',
            'device_calibration':self.backend.public_calibration(),
            'control_track':'public_joint_trajectory',
            'note':'New independent device implementation; historical combined-adapter runs are retained separately.'}

    def robot_state(self):
        return super().robot_state() | {'gripper_holding':bool(self.backend.robot_state()['gripper_holding'])}

    def observation(self):
        value=super().observation()
        supported,stable,released=self.backend.support_state()
        # The evidence images are the three standard views of this state.
        views=self.backend.view_pngs
        digest=hashlib.sha256(''.join(hashlib.sha256(views[c]).hexdigest() for c in sorted(views)).encode()).hexdigest()
        eid=self.score.add_evidence(self.tick//(self.limits.physics_hz//30),supported,stable and released,
            self.backend.automatic_reading,digest)
        directory=self.backend.private/'evidence';directory.mkdir(exist_ok=True)
        for camera,png in views.items():(directory/f"{eid.rsplit(':',1)[-1]}_{camera}.png").write_bytes(png)
        value['evidence_id']=eid
        return value

    def _dispatch(self,name,args):
        if self.terminated:raise ValueError('episode terminated')
        if name not in PUBLIC_TOOLS:raise ValueError('not a Hooke device command')
        if name=='get_task_spec' and args=={}:return self.public_task()
        if name=='submit_result':
            supported,stable,released=self.backend.support_state()
            self.terminated=True
            report=self.score.submit(args,frame=self.tick//(self.limits.physics_hz//30),
                all_supported=set(supported)==set(self.score.weight_ids),stable=stable,released=released,
                budget_ok=self.tick<=self.horizon_ticks)
            # Scoring and hidden parameter diagnostics belong to the host.
            return {'accepted':True,'episode_id':self.episode_id,'terminated':True}
        return super()._dispatch(name,args)

    def evaluation(self):
        if self.terminated and self.score.report is None:
            self.score.terminate(self.tick//(self.limits.physics_hz//30),self.termination_reason or 'device terminated')
        return self.score.evaluation(self.tick//(self.limits.physics_hz//30))

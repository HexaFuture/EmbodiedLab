"""Trusted EmbodiedLab envd lifecycle adapter; participant calls use public EmbodiedLab HTTP.

The neutral benchmark runs as a child main-thread Isaac process. Only lifecycle
hooks retain the host administration key. Arbitrary env.exec is disabled.
"""
import atexit
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
import urllib.error

from ..motion.transport import EmbodiedLabTransport
from .participant import Participant
from .tools import FAMILIES, LEGACY_OF, catalog, family

INSTANCE_ROOT = '/var/lib/embodiedlab/instances'

# Tool exposure level, fixed at environment creation and enforced here (not by the
# tool-visibility API, which any client could undo).  T1: bench tools plus the
# perception and planning primitives; the agent writes every manipulation skill
# itself.  T0: bench tools only.
EXPOSURE = os.environ.get('EMBODIEDLAB_EXPOSURE', 'T1').strip().upper() or 'T1'
if EXPOSURE not in ('T0', 'T1'):
    raise ValueError('EMBODIEDLAB_EXPOSURE must be T0 or T1')
BENCH_CORE = {'get_task_spec', 'get_observation', 'get_robot_state', 'execute_joint_trajectory',
              'set_gripper', 'wait', 'submit_result', 'record_prediction', 'solve_ik'}
PRIMITIVES = {'segment_sam3_point_prompt', 'inspect_mask_image', 'localize_rgbd_point', 'mask_geometry',
              'get_robot_pose', 'move_hand', 'plan_return_to_start', 'translate_contact'}
# The read-only Viser view of the whole scene, for the operator only (trial start --viewer). Off by default: it shows
# more than the three cameras, and any process that reaches the host's 127.0.0.1 can open it.
VIEWER = os.environ.get('EMBODIEDLAB_VIEWER', 'off').strip().lower() == 'on'


def exposed(name, task):
    """Whether tool `name` is visible and callable at this environment's exposure level."""
    if name in BENCH_CORE or name in set(task.get('tools') or ()):
        return True
    return EXPOSURE == 'T1' and name in PRIMITIVES


class Worker:
    def __init__(self):
        task_id=os.environ.get('EMBODIEDLAB_TASK_ID','pendulum_gravity_v1')
        if task_id not in FAMILIES:
            raise ValueError('unsupported neutral EmbodiedLab task')
        self.lock=threading.RLock()
        self.closed=False
        self.capture=None
        self.root=Path(os.environ.get('EMBODIEDLAB_BENCH_PRIVATE_ROOT','/var/lib/embodiedlab/private'))/secrets.token_hex(12)
        self.root.mkdir(parents=True,mode=0o700)
        self.key=secrets.token_hex(32)
        keyfile=self.root/'admin.key'
        fd=os.open(keyfile,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(self.key)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        self.url=f'http://127.0.0.1:{port}'
        launcher=os.environ.get('EMBODIEDLAB_BENCH_ISAAC_PYTHON','/isaac-sim/python.sh')
        entry=os.environ.get('EMBODIEDLAB_BENCH_DEVICE_MODULE','embodiedlab_sdk.robot.embodiedlab.device_entry')
        self.log=(self.root/'device.log').open('wb')
        hz=int(os.environ.get('EMBODIEDLAB_BENCH_PHYSICS_HZ','240'))
        horizon=int(os.environ.get('EMBODIEDLAB_MAX_STEPS','18000'))*(hz//30)
        instance=[]
        if task_id not in LEGACY_OF.values():
            # Registry experiment: the opaque instance id selects the private record on the
            # read-only instance mount; hidden values never travel in the environment.
            instance_id=os.environ.get('EMBODIEDLAB_INSTANCE_ID','')
            if not instance_id.startswith('inst-') or not instance_id[5:].isalnum():
                raise ValueError('registry experiments require EMBODIEDLAB_INSTANCE_ID')
            path=Path(os.environ.get('EMBODIEDLAB_INSTANCE_ROOT',INSTANCE_ROOT))/(instance_id+'.json')
            if not path.is_file():raise FileNotFoundError('instance record is not mounted')
            instance=['--instance-file',str(path)]
            if task_id == 'circuit_blackbox3' and 'EMBODIEDLAB_MAX_STEPS' not in os.environ:
                # The published budget of circuit_blackbox3 also applies to ordinary Agent launches.
                horizon = int(json.loads(path.read_text())['budget']['sim_s'] * hz)
        command=[launcher,'-m',entry,'--stage',os.environ.get('EMBODIEDLAB_BENCH_STAGE','/isaac-sim/documents/hooke_scene_layout.usda'),
                 '--output',str(self.root/'run'),'--admin-key-file',str(keyfile),
                 '--seed',os.environ.get('EMBODIEDLAB_SEED','0'),'--task',task_id,'--physics-hz',str(hz),
                 '--horizon-ticks',str(horizon),'--port',str(port),*(['--spectator'] if VIEWER else []),*instance]
        # Do not inject model/API credentials into the benchmark process.
        environment={k:v for k,v in os.environ.items() if not any(t in k.upper() for t in ('API_KEY','TOKEN','SECRET','PASSWORD'))}
        self.process=subprocess.Popen(command,stdout=self.log,stderr=subprocess.STDOUT,env=environment)
        try:
            # Just under the Worker's 900 s readiness wait: many benches starting at once on a loaded host can take ten minutes.
            deadline=time.monotonic()+870
            while time.monotonic()<deadline:
                if self.process.poll() is not None:raise RuntimeError('neutral benchmark process exited; inspect private device.log')
                try:
                    with urllib.request.urlopen(self.url+'/health',timeout=3) as response:
                        if json.load(response).get('ready'):break
                except (OSError,ValueError):pass
                time.sleep(1)
            else:raise RuntimeError('neutral benchmark startup timed out')
            self.transport=EmbodiedLabTransport(self.url,timeout=900)
            self.participant=Participant(self.transport,sam_url=os.environ.get('EMBODIEDLAB_SAM_URL',''),
                                         curobo_url=os.environ.get('EMBODIEDLAB_CUROBO_URL',''),
                                         geometry_provider=lambda:self.administration('/v1/host-observation'))
            self.task=self.participant.task
            self.viewer=self.administration('/v1/spectator')
            port=self.viewer.get('port') if VIEWER and isinstance(self.viewer,dict) else None
            if port and os.environ.get('EMBODIEDLAB_CAPTURE_DIR'):
                # The operator's capture directory (outside the agent's workspace): trial viewer reads the address here.
                try:(Path(os.environ['EMBODIEDLAB_CAPTURE_DIR'])/'viewer.json').write_text(
                    json.dumps({'url':f'http://127.0.0.1:{port}','port':port}))
                except OSError:pass
        except BaseException:
            # Worker removes failed containers; retain device diagnostics in the
            # operator-owned capture directory before lifecycle cleanup.
            target = os.environ.get('EMBODIEDLAB_CAPTURE_DIR')
            if target:
                try:
                    destination = Path(target) / 'diagnostics'
                    destination.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(self.root / 'device.log', destination / 'startup-device.log')
                except OSError:
                    pass
            self.close()
            raise
        atexit.register(self.close)

    def call(self,name,args):
        with self.lock:
            if self.closed:raise RuntimeError('environment closed')
            if not exposed(name,self.task):
                raise PermissionError(f'{name} is not exposed at level {EXPOSURE}; see get_task_spec()["exposure"]')
            self._check_alive()
            try:result=self.participant.call(name,args)
            except OSError:
                # A dropped connection from a benchmark process that just died: say so, not a socket error.
                try:self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
                self._check_alive()
                raise
            if name=='get_task_spec' and isinstance(result,dict):
                tools=list(result.get('tools') or [])
                if 'solve_ik' not in tools and exposed('solve_ik',self.task):
                    at=tools.index('execute_joint_trajectory')+1 if 'execute_joint_trajectory' in tools else len(tools)
                    tools.insert(at,'solve_ik')
                result=dict(result,exposure=EXPOSURE,tools=tools)
            return result

    def _check_alive(self):
        code=self.process.poll()
        if code is not None:
            raise RuntimeError(f'the simulator process has exited ({_exit_status(code)}); the episode has ended and '
                               'no further command can run')

    def administration(self,path,*,post=False):
        request=urllib.request.Request(self.url+path,data=b'{}' if post else None,
                    headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=180) as response:return json.load(response)

    def finalize(self):
        with self.lock:
            if self.capture is not None:return self.capture
            try:
                if self.process.poll() is not None:
                    raise ConnectionError('benchmark process exited before finalizing')
                record=self.administration('/v1/finalize',post=True)
            except OSError as exc:
                if not hasattr(self,'participant'):raise
                self.capture=self.salvage(exc)
                return self.capture
            target=os.environ.get('EMBODIEDLAB_CAPTURE_DIR')
            output={'supported':True,'enabled':True,'finalized':True,'frame_count':record['frame_count']}
            if target:
                directory=Path(target)/'video';directory.mkdir(parents=True,exist_ok=True)
                destination=directory/'video_three-view.mp4'
                shutil.copyfile(record['video_path'],destination)
                output['artifacts']={'video':'video/video_three-view.mp4'}
                if record.get('sealed_dir'):
                    # Sealed evidence for the offline evaluation service.
                    shutil.copytree(record['sealed_dir'],Path(target)/'sealed',dirs_exist_ok=True)
                    output['artifacts']['sealed']='sealed'
                if record.get('process_dir') and Path(record['process_dir']).is_dir():
                    # The run's records, submitted or not, for the offline process score.
                    shutil.copytree(record['process_dir'],Path(target)/'process',dirs_exist_ok=True)
                    output['artifacts']['process']='process'
            self.capture=output
            return output

    def salvage(self,cause):
        """The benchmark process died (or stopped answering): end it and wrap the episode up from what it left on
        disk, in a separate interpreter (salvage.py: last process-record checkpoint marked sim_diverged,
        private logs, the video up to the crash)."""
        if self.process.poll() is None:
            self.process.kill();self.process.wait()
        status=f'{_exit_status(self.process.returncode)}; {type(cause).__name__}: {cause}'
        target=os.environ.get('EMBODIEDLAB_CAPTURE_DIR')
        output={'supported':True,'enabled':True,'finalized':False,'salvaged':False,'error':status}
        if not target:return output
        try:
            self.log.flush()
            destination=Path(target)/'diagnostics';destination.mkdir(parents=True,exist_ok=True)
            lines=(self.root/'device.log').read_bytes().splitlines(keepends=True)
            (destination/'crash-device.log').write_bytes(b''.join(lines[-3000:]))
            launcher=os.environ.get('EMBODIEDLAB_BENCH_ISAAC_PYTHON','/isaac-sim/python.sh')
            result=subprocess.run([launcher,'-m','embodiedlab_sdk.robot.embodiedlab.salvage','--run',str(self.root/'run'),
                                   '--capture',target,'--status',status,'--device-log',str(self.root/'device.log')],
                                  capture_output=True,text=True,timeout=1800)
            (destination/'salvage.log').write_text(result.stdout[-20000:]+result.stderr[-20000:])
            output=json.loads(result.stdout.strip().splitlines()[-1])
        except Exception as exc:
            output['salvage_error']=f'{type(exc).__name__}: {exc}'
        return output

    def close(self):
        if self.closed:return
        self.closed=True
        if hasattr(self,'process') and self.process.poll() is not None and hasattr(self,'participant'):
            try:self.finalize()     # salvage what a crashed benchmark left
            except Exception:pass
        if hasattr(self,'process') and self.process.poll() is None:
            try:self.finalize()
            except (OSError,ValueError):pass
            self.process.terminate()
            try:self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        if hasattr(self,'log'):self.log.close()


def _exit_status(code):
    return f'exit status {code}' if code is None or code>=0 else f'signal {-code}'


def bootstrap(namespace):
    worker=Worker()
    task=json.dumps(worker.task,ensure_ascii=False,indent=2)
    def execute(code,filename,context=None):
        if code.strip()!='print(TASK)':
            return {'ok':False,'error':{'type':'PermissionError','message':'Use public tools; analysis belongs in the separate participant workspace.'}}
        return {'ok':True,'stdout':task+'\n','stderr':'','result':None}
    def evaluate():
        value=worker.administration('/v1/evaluation')
        return {k:value[k] for k in ('reward','task_completed','sim_step','evaluator')} | (
            {'scientific_evaluation':value} if value.get('terminated') else {})
    def restore(_snapshot):raise RuntimeError('evidence snapshots cannot reset an episode')
    namespace.update({
        'TASK':task,'EMBODIEDLAB_RESTRICTED_TOOLS':True,
        'EMBODIEDLAB_ENV_METADATA':{'profile':'embodiedlab-'+('airtrack' if worker.task['task_id']=='airtrack_collision' else family(worker.task['task_id'])),'horizon':worker.task['horizon'],
            'evaluation_ready':False,'mode':'development','episode_id':worker.task['episode_id'],
            'participant':'EmbodiedLab','benchmark_interface':'public-device-v1','exposure':EXPOSURE,'viser':worker.viewer},
        'embodiedlab_execute_code':execute,'embodiedlab_tools':lambda:[e for e in catalog(worker.task['task_id'],worker.task) if exposed(e['name'],worker.task)],
        'embodiedlab_observe_with_images':lambda:worker.call('get_observation',{}),
        'embodiedlab_call_tool_with_images':worker.call,
        'embodiedlab_evaluate_state':evaluate,
        'embodiedlab_save_state':lambda:{'episode_id':worker.task['episode_id']},
        'embodiedlab_restore_state':restore,
        'embodiedlab_capture_status':lambda:worker.capture or {'supported':True,'enabled':True,'finalized':False},
        'embodiedlab_finalize_capture':worker.finalize,'embodiedlab_close':worker.close,
    })

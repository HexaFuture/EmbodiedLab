"""One agent trial on EmbodiedLab: start an environment, hand the agent a workspace, score it offline.

usage (normally through ./embodiedlab trial ...):
  trial.py start  --exp EXP --seed N [--exposure T1|T0] [--agent LABEL] [--gpu G|any]
                  [--image IMG] [--root DIR] [--wall-s S] [--no-watch] [--viewer]
  trial.py prompt TRIAL_ID            print the agent prompt for this trial (generic, no experiment knowledge)
  trial.py status TRIAL_ID            calls so far, submitted or not, environment state
  trial.py viewer TRIAL_ID            the address of the trial's Viser view (only for a trial started with --viewer)
  trial.py watch  TRIAL_ID            block until the episode closes (submission or wall limit), then finish
  trial.py finish TRIAL_ID            stop the environment, evaluate sealed evidence, write result.json
  trial.py pack   TRIAL_ID            tar the workspace for an agent machine that tunnels to the Controller
  trial.py catalog [--json]           list the experiments (id, tier, kit, budgets) as a table or JSON
  trial.py score  [--root DIR] [--json]  success, process score and error of every finished trial, per agent and
                                      exposure level, by experiment, discipline and overall

Layout of a trial directory <root>/<trial_id>/:
  trial.json          exp, seed, instance_id, exposure, agent, environment_id, image, budget, timestamps
  workspace/          what the agent sees: env.json, py, README.md, task_card.md, task_spec.json, tools.json,
                      images/, calls.jsonl, and whatever the agent writes (tools/, notes, scripts)
  captures/           Controller capture dir: video, sealed evidence and the run's process record (written at stop)
  evaluation.json     offline evaluation of the sealed evidence against the instance truth
  process_score.json  the six process levels (L1-L6), the level and the process score
  result.json         summary row (status, G, e_rel, e, reward, task_completed, process_level, process_score,
                      quantities, sim_s_used, wall_s, tool_calls, failed_gates)

The agent never sees anything above workspace/; the runner is the only process that reads the instance store.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import os
import shutil
import subprocess
import tarfile
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sdk" / "src"))

from benchmark.evaluation.evaluate import evaluate  # noqa: E402
from benchmark.evaluation.process import score_trial  # noqa: E402
from benchmark.evaluation.bench_restore import from_process_logs  # noqa: E402
from benchmark.instances.service import InstanceRejected, derive, materialize, read  # noqa: E402
from benchmark.instances import snapshot as instance_snapshot  # noqa: E402
from benchmark.registry.loader import load  # noqa: E402
from embodiedlab_sdk import EmbodiedLabClient  # noqa: E402

CONTROLLER = os.environ.get("EMBODIEDLAB_CONTROLLER_URL", "http://127.0.0.1:18731")
DATA = Path(os.environ.get("EMBODIEDLAB_DATA", str(REPO / "data")))
STORE = Path(os.environ.get("EMBODIEDLAB_INSTANCE_STORE", str(DATA / "instances")))
DEFAULT_ROOT = Path(os.environ.get("EMBODIEDLAB_TRIALS", str(DATA / "trials")))
DEFAULT_IMAGE = os.environ.get("EMBODIEDLAB_IMAGE", "embodiedlab/lab:0.1.0")
PYTHON = os.environ.get("EMBODIEDLAB_TRIAL_PYTHON", sys.executable)
EXPOSURE_NOTES = {
    "T1": "bench tools plus the perception and planning primitives listed in README.md; there are no grasp, place, push or press skills, any manipulation you need you implement yourself from the primitives",
    "T0": "bench tools only (observation, joint control, gripper, instruments, submit); perception, planning and every skill are yours to implement",
}

# What README.md says about manipulation: per exposure level the interface facts (frames, kinematics) and how the
# perception and planning tools fit together, then the robot limits and, where the bench has one, the gripper
# mechanism. No how-to: approach, grasp, place, push and recovery are the Agent's own.
MANIPULATION_INTRO = {
    'T1': '## Manipulating at this exposure level (the skills are yours to write)\n\n**Frames.** Perception tools answer in WORLD coordinates, motion tools use ROBOT BASE coordinates; convert with\n`device_calibration.world_from_robot_base`. `move_hand` takes the hand-frame pose `[qw,qx,qy,qz,x,y,z]`; the fingertips\nsit `fingertip_center_hand_m` along the hand z axis.\n\n**Perception and planning (a suggestion, not a rule).** `segment_sam3_point_prompt` turns a pixel into\nmasks of the whole part (check them with `inspect_mask_image`), and `mask_geometry` reads the WORLD position and\nextent of the visible part from a mask, which is far more reliable than an eyeballed pixel. `move_hand` plans a\ncollision-free path (cuRobo) to a hand pose given in the ROBOT BASE frame. Masks and `rgb_ref`s expire with the\nframe, so re-observe after each motion.\n',
    'T0': '## Manipulating at this exposure level (perception, motion and skills are yours to write)\n\n**Frames and kinematics.** `device_calibration` (robot model, joint chain, `world_from_robot_base`, fingertip and\njaw offsets along the hand z axis, and `cameras`: resolution and intrinsics of every camera, the world pose of the two\nfixed cameras and the pose of the wrist camera on the hand, given once for the episode) is all you get; observations are\nRGB images only, there is no depth. The robot state carries the hand pose (forward kinematics) and `solve_ik` returns\njoint angles for a hand pose without checking collisions; scene geometry, collision avoidance, paths and trajectories\nare yours. `execute_joint_trajectory` takes one seven-joint target per\nphysics tick (240 Hz); targets beyond a joint range are clamped, nothing checks collisions.\n',
}


# How the two fingers move (Franka hand asset: panda_finger_joint1 drive, maxJointVelocity 0.2 m/s; joint2 its mimic).
FINGER_MOTION = ('The two fingers are coupled, like a real Franka Hand driven by one motor: they always move together, symmetrically about the hand centre, each at most 0.2 m/s. If one finger meets something that does not give way, both fingers stop there; a loose part is pushed along by the finger that meets it first. The hand itself never moves to centre on a part.')


def gripper_note(task_spec: dict) -> str:
    """How this simulation grasps: part of the benchmark, stated at every exposure level."""
    mechanism = (task_spec.get("device_calibration") or {}).get("gripper_mechanism") or {}
    if mechanism.get("type") != "contact_conditioned_constrained_gripper":
        return ""
    return ("**Gripper (how this simulation grasps, the same at every exposure level).** "
            + mechanism["description"] + " " + FINGER_MOTION
            + " The numbers are in `device_calibration.gripper_mechanism`.\n")


ROBOT_NOTE = ("**Robot limits (the same at every exposure level).** Joint targets outside the joint range are "
              "clamped to it, and one command lasts at most 30 s. The simulated joint drives cannot exceed "
              "0.8 rad/s, so a large target jump is followed at that speed rather than instantly (an optional "
              "velocity feed-forward must stay within 0.65 rad/s). If a joint's mean speed over the last 0.1 s (its net "
              "travel in that time divided by 0.1 s) reaches 0.9 rad/s, "
              "for example after an impact or when a joint overshoots while following a large jump, the arm makes a "
              "protective stop: it brakes toward the pose where the stop occurred for at most 0.25 s (charged to "
              "the time budget), then holds that pose as its target, and the running command returns an error "
              "naming the joint; the rest of the command is not executed. The arm may not be fully at rest when "
              "the error returns. The gripper keeps its state (a held part stays held). A protective stop is not "
              "a failure: nothing is damaged, it is not scored, the episode continues, and the next command runs "
              "normally from the current pose.\n")


def manipulation_note(exposure: str, task_spec: dict) -> str:
    text = MANIPULATION_INTRO[exposure] + "\n" + ROBOT_NOTE
    gripper = gripper_note(task_spec)
    if gripper:
        text += "\n" + gripper
    return text

PROMPT = """You are operating a robot in a simulated physics laboratory to answer a measurement question.

Workspace: {workspace}

Rules
- Read README.md (the API and the tools you have) and task_card.md (the question, the apparatus, the acceptance conditions, the budget and the submission format) before doing anything else.
- The only way to act is the Python API described in README.md (`from embodiedlab_api import connect; env = connect()`), run through the workspace wrapper `./py`. Every call is logged to calls.jsonl; images a tool returns are written to images/.
- The answer is a property of this particular apparatus. It is not in the task card and cannot be looked up; it must be obtained with the tools, and the evidence the task card asks for must come from this episode.
- Keep your notes in NOTES.md. Code you want to reuse goes in tools/*.py; it is yours for this trial only.
- You may submit exactly once (`env.submit(...)`, schema in task_spec.json). The first submission closes the episode even if it is invalid.
- Do not read, modify or probe anything outside the workspace, and do not try to reach the simulator, other controller endpoints or evaluation data.

Exposure level {exposure}: {exposure_note}.

When the submission is accepted, write a short SUMMARY.md in the workspace (what you measured, the value, the evidence ids) and stop.
"""

README = """# EmbodiedLab trial workspace

Environment `{environment_id}` on `{controller}`; experiment `{exp_id}`; exposure level **{exposure}**
({exposure_note}).

## Python API

```python
from embodiedlab_api import connect
env = connect()                 # resolved from env.json in this directory
spec = env.task()               # public task specification (also in task_spec.json; the card is task_card.md)
                                # spec["budget"]["sim_s"] is the simulated-time budget; horizon_ticks is the same in
                                # physics ticks (physics_hz per second), horizon in 30 Hz frames
env.tools()                     # the tools you have right now: name, signature, description (also tools.json)
obs = env.observe()             # three calibrated RGB views (calibration in device_calibration.cameras; no depth); obs["images"] lists the PNG paths (images/)
env.state()                     # joint positions, gripper width and hand pose; carries the current state_id
env.call("wait", seconds=5)     # advance simulated time (charged to the budget)
env.call("<tool>", **arguments) # any tool from env.tools()
env.submit(**fields)            # once; see the submission schema in task_spec.json
```

Run Python with `./py` (it sets the path and the environment variables):

    ./py -c "from embodiedlab_api import connect; print(connect().state())"

Every call is appended to `calls.jsonl`; every image a tool returns is saved under `images/` and listed in the
returned dict under `"images"`. A tool that refuses or fails raises `embodiedlab_api.ToolError` with the tool's
own message. Tool arguments that take a `state_id` (or, at T1, an `rgb_ref`) must come from the same physical state: any
physical action invalidates earlier observations, so observe again after acting.

## Tools at this exposure level

{tool_table}

{manipulation_note}
Simulated-time budget: {sim_s} s (every physical action and every wait is charged; observing is free). A command
that needs more simulated time than remains is refused; once none remains, a further physical action ends the
episode and nothing, not even a submission, is accepted after that.
Wall-clock limit: {wall_min} min of real time for the whole trial; when it runs out the trial is closed without a
submission. Submissions: 1.
In the task card's scoring, G is 1 when every acceptance condition holds and 0 otherwise.
"""


def _trial_dir(root: Path, trial_id: str) -> Path:
    directory = root / trial_id
    if not (directory / "trial.json").is_file():
        raise SystemExit(f"no trial at {directory}")
    return directory


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) if not isinstance(value, str) else value)


def _call(client: EmbodiedLabClient, environment_id: str, name: str, arguments: dict | None = None, timeout: float = 600):
    operation = client.wait_operation(client.call_tool(environment_id, name, arguments or {}).id, timeout=timeout)
    output = operation.output or {}
    if operation.status.value != "completed" or output.get("ok") is False:
        raise RuntimeError(f"{name} failed: {output.get('error') or operation.error}")
    return output.get("result")


def _tool_table(tools: list[dict]) -> str:
    lines = ["| tool | signature | what it does |", "|---|---|---|"]
    for tool in tools:
        description = str(tool.get("description") or "").replace("|", "\\|").replace("\n", " ")
        signature = str(tool.get("signature") or "").replace("|", "\\|")
        lines.append(f"| `{tool['name']}` | `{signature}` | {description} |")
    return "\n".join(lines)


def _task_card(spec: dict) -> str:
    for key in ("task_card", "card", "task_card_markdown"):
        if isinstance(spec.get(key), str):
            return spec[key]
    parts = [f"# {spec.get('task_id', 'task')}"]
    for key in ("objective", "apparatus", "instrument", "measurement_requirements", "model_convention",
                "provisional_tolerances", "submission_fields", "device_calibration"):
        if key in spec:
            value = spec[key]
            parts.append(f"\n## {key}\n")
            parts.append(value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False))
    return "\n".join(parts) + "\n"


def _source_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def start(args) -> int:
    spec = load(args.exp)
    try:
        instance = derive(args.exp, "dev", args.seed)
    except InstanceRejected as exc:
        raise SystemExit(f"dev seed {args.seed} rejected for {args.exp}: {exc.failed}")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    trial_id = args.name or f"{args.exp}-s{args.seed}-{args.exposure}-{args.agent}-{stamp}"
    root = Path(args.root)
    trial = root / trial_id
    workspace = trial / "workspace"
    (workspace / "images").mkdir(parents=True)
    (workspace / "tools").mkdir()
    materialize(instance, STORE)
    instance_snapshot.save(instance, trial)     # the record evaluation reads later (the shared store can change)
    # The wall-clock limit in force: the caller's, else the experiment's own budget.
    wall_s = float(args.wall_s) if args.wall_s else float(spec.budget.get("wall_s") or 10800)
    record = {"trial_id": trial_id, "exp_id": args.exp, "seed": args.seed, "split": "dev",
              "instance_id": instance.instance_id, "spec_digest": instance.spec_digest,
              "exposure": args.exposure, "agent": args.agent, "image": args.image, "gpu": args.gpu,
              "budget": dict(spec.budget), "wall_s": wall_s, "viewer": bool(args.viewer), "status": "starting",
              "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "environment_id": None,
              "source_commit": _source_commit()}
    _write(trial / "trial.json", record)
    started = time.time()
    with EmbodiedLabClient(CONTROLLER, timeout=60) as client:
        response = client.start_environments(
            profile="embodiedlab", count=1, seed=0, task_id=args.exp, image=args.image,
            gpu=None if args.gpu in (None, "any") else args.gpu, record_video=True,
            parameters={"capture_dir": str(trial / "captures"), "horizon": int(spec.budget["sim_s"] * 30),
                        "environment": {"EMBODIEDLAB_INSTANCE_ID": instance.instance_id,
                                        "EMBODIEDLAB_EXPOSURE": args.exposure,
                                        "EMBODIEDLAB_MAX_STEPS": str(int(spec.budget["sim_s"] * 30)),
                                        "EMBODIEDLAB_VIEWER": "on" if args.viewer else "off"}})
        environment_id = response["environments"][0]["id"]
        record.update(environment_id=environment_id, start_operation_id=response["operation"]["id"])
        _write(trial / "trial.json", record)
        operation = client.wait_operation(response["operation"]["id"], timeout=1500)
        if operation.status.value != "completed":
            record.update(status="start_failed", error=str(operation.error))
            _write(trial / "trial.json", record)
            raise SystemExit(f"environment start failed: {operation.error}")
        for _ in range(60):  # the start operation can complete before the Controller marks the environment ready
            try:
                _call(client, environment_id, "get_robot_state", timeout=120)
                break
            except Exception as exc:
                if "409" not in str(exc):
                    raise
                time.sleep(2)
        task_spec = _call(client, environment_id, "get_task_spec")
        tools = client.request("GET", "/v1/tools", params={"environment_id": environment_id})
    tools = [{k: t.get(k) for k in ("name", "signature", "description", "effect")} for t in tools]
    exposure_seen = task_spec.get("exposure") if isinstance(task_spec, dict) else None
    if exposure_seen != args.exposure:
        record.update(status="exposure_mismatch", error=f"environment reports exposure {exposure_seen!r}; the image must carry the exposure filter")
        _write(trial / "trial.json", record)
        raise SystemExit(record["error"])
    _write(workspace / "env.json", {"environment_id": environment_id, "controller": CONTROLLER, "workspace": str(workspace),
                                     "exp_id": args.exp, "exposure": args.exposure})
    _write(workspace / "task_spec.json", task_spec)
    _write(workspace / "tools.json", tools)
    (workspace / "task_card.md").write_text(_task_card(task_spec))
    (workspace / "README.md").write_text(README.format(
        environment_id=environment_id, controller=CONTROLLER, exp_id=args.exp, exposure=args.exposure,
        exposure_note=EXPOSURE_NOTES[args.exposure], tool_table=_tool_table(tools), sim_s=spec.budget["sim_s"],
        wall_min=int(round(wall_s / 60)), manipulation_note=manipulation_note(args.exposure, task_spec)))
    shutil.copytree(REPO / "sdk" / "src" / "embodiedlab_api", workspace / "embodiedlab_api",
                    ignore=shutil.ignore_patterns("__pycache__"))
    _write_wrapper(workspace, PYTHON)
    (workspace / "NOTES.md").write_text("# Notes\n")
    record.update(status="ready", start_s=round(time.time() - started, 1), ready_at=dt.datetime.now(dt.timezone.utc).isoformat())
    _write(trial / "trial.json", record)
    if not args.no_watch:
        log = (trial / "watch.log").open("a")
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "watch", trial_id, "--root", str(root)],
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"trial_id": trial_id, "workspace": str(workspace), "environment_id": environment_id,
                      "exposure": args.exposure, "tools": len(tools), "start_s": record["start_s"], "wall_s": wall_s}))
    print("\n--- agent prompt ---\n")
    print(prompt_text(record, workspace))
    return 0


def _write_wrapper(workspace: Path, python: str) -> None:
    """`./py`: Python bound to this workspace; the only way the agent talks to the benchmark.

    The workspace carries its own copy of embodiedlab_api (standard library only), so the
    same wrapper works on the server and on a laptop that reaches the Controller through a tunnel.
    """
    wrapper = workspace / "py"
    wrapper.write_text("#!/bin/bash\n# Python bound to this trial's workspace (env.json names the environment and the Controller).\n"
                       'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
                       'export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"\n'
                       'export EMBODIEDLAB_WORKSPACE="$HERE"\n'
                       'cd "$HERE"\n'
                       f'exec {python} "$@"\n')
    wrapper.chmod(0o755)


def pack(args) -> int:
    """Bundle the agent-facing workspace for a machine that reaches the Controller through an SSH tunnel."""
    trial = _trial_dir(Path(args.root), args.trial_id)
    record = json.loads((trial / "trial.json").read_text())
    workspace = trial / "workspace"
    staging = trial / "pack" / "workspace"
    if staging.parent.exists():
        shutil.rmtree(staging.parent)
    shutil.copytree(workspace, staging, ignore=shutil.ignore_patterns("__pycache__", "images", "calls.jsonl"))
    (staging / "images").mkdir()
    env_record = json.loads((staging / "env.json").read_text())
    env_record["controller"] = args.controller
    env_record["tunnel"] = f"ssh -N -L {args.port}:{CONTROLLER.split('://', 1)[-1]} <benchmark host>"
    _write(staging / "env.json", env_record)
    _write_wrapper(staging, args.python)
    readme = (staging / "README.md").read_text().replace(CONTROLLER, args.controller)
    (staging / "README.md").write_text(readme)
    archive = trial / f"{record['trial_id']}-workspace.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staging, arcname="workspace")
    print(json.dumps({"archive": str(archive), "controller": args.controller, "tunnel": env_record["tunnel"]}))
    return 0


# Trial status (result.json "status"):
#   submitted          the Bench sealed a submission
#   not_submitted      the Agent stopped, or the trial was closed, without submitting
#   timeout            the wall-clock limit ran out without a submission
#   budget_exhausted   the episode's simulated-time budget (or the tool-call cap) ran out without a submission
#   agent_terminated   the Bench ended the episode on a physical protection the Agent's actions triggered
#   not_scored         infrastructure failure (the environment failed or did not finalize, the simulation
#                      process crashed or hit a GPU fault, or the agent runner failed: see AGENT_INFRA_CAUSES);
#                      retried, not counted
# Every status except not_scored is a scored trial: success 0 unless submitted and accurate.
STATUS_BY_END_REASON = {"sim_budget_exhausted": "budget_exhausted", "request_budget_exhausted": "budget_exhausted",
                        "physics_backend_failure": "agent_terminated", "sim_diverged": "agent_terminated"}
# A benchmark process that died (native crash, GPU out of memory, kill) is salvaged from its last checkpoint and
# marked "salvaged"; that, and a backend failure reported by CUDA, is an infrastructure failure, not the Agent's.
GPU_FAILURE = re.compile(r"CUDA|out of memory|cudaError|CUBLAS|illegal memory access", re.I)

# The agent runner's own failure, reported by the runner (not the Agent) in `finish --agent-end FILE`: a JSON object
# {"kind": "agent_infra_error", "cause": <one of these>, "detail": <the evidence>, ...}. It makes a trial not_scored
# only when the Agent did not submit and the runner did not stop the Agent itself (wall-clock limit, after the
# submission). Any other kind, cause or an empty detail leaves the status to the rules below. The runner decides
# from unambiguous evidence only, never from the wall time: an "I give up" final message with exit 0 stays
# not_submitted.
AGENT_INFRA_CAUSES = {
    "gateway_transport": "the model gateway failed (stream disconnected, reconnects exhausted, 5xx) and ended the turn",
    "gateway_usage_limit": "the model gateway refused for the usage limit or quota and ended the turn",
    "gateway_auth": "the model gateway refused the runner's credentials and ended the turn",
    "signal": "the agent process was killed by a signal it did not send itself",
    "crash": "the agent process ended in the middle of a turn without reporting an error",
    "startup": "the agent process failed before its first turn",
}


def read_agent_end(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def agent_infra_error(agent_end: dict | None) -> str | None:
    """The agent runner's failure that ended the Agent's process ("<cause>: <detail>"), or None."""
    if not isinstance(agent_end, dict) or agent_end.get("kind") != "agent_infra_error":
        return None
    cause, detail = agent_end.get("cause"), " ".join(str(agent_end.get("detail") or "").split())
    if cause not in AGENT_INFRA_CAUSES or not detail:
        return None
    return f"{cause}: {detail[:300]}"


def infrastructure_end(end: dict, logs: dict | None = None) -> bool:
    """Whether the episode ended because the simulation host failed (process crash, GPU fault, or a failure while
    rendering after a completed physics tick, bench_logs backend_failure.where == "after_step"), not by the Agent."""
    if ((logs or {}).get("backend_failure") or {}).get("where") == "after_step":
        return True
    return bool(end.get("salvaged")) or (end.get("end_reason") in ("physics_backend_failure", "sim_diverged")
                                         and bool(GPU_FAILURE.search(str(end.get("termination_detail") or ""))))


def episode_closed(client: EmbodiedLabClient, environment_id: str) -> str | None:
    """Why the Bench ended the episode ("submitted", or its termination reason), None while it runs."""
    try:
        operation = client.wait_operation(client.environment_command(environment_id, "evaluate").id, timeout=120)
    except Exception:
        return None
    report = (operation.output or {}).get("scientific_evaluation")
    if not report:
        return None
    return "submitted" if report.get("sealed") else (report.get("error") or "terminated")


def prompt_text(record: dict, workspace: Path) -> str:
    return PROMPT.format(workspace=workspace, exposure=record["exposure"], exposure_note=EXPOSURE_NOTES[record["exposure"]])


def submitted(workspace: Path) -> bool:
    log = workspace / "calls.jsonl"
    if not log.is_file():
        return False
    with log.open() as handle:
        return any(json.loads(line).get("tool") == "submit_result" for line in handle if line.strip())


def calls(workspace: Path) -> int:
    log = workspace / "calls.jsonl"
    if not log.is_file():
        return 0
    with log.open() as handle:
        return sum(1 for line in handle if line.strip())


STOP_RETRY_S = 180      # how long finish keeps stopping an environment whose container did not report its exit


def environment_state(client: EmbodiedLabClient, environment_id: str) -> str | None:
    try:
        return client.request("GET", f"/v1/environments/{environment_id}").get("state")
    except Exception:
        return None


def status(args) -> int:
    trial = _trial_dir(Path(args.root), args.trial_id)
    record = json.loads((trial / "trial.json").read_text())
    workspace = trial / "workspace"
    with EmbodiedLabClient(CONTROLLER, timeout=60) as client:
        state = environment_state(client, record["environment_id"]) if record.get("environment_id") else None
    print(json.dumps({"trial_id": record["trial_id"], "status": record["status"], "environment_state": state,
                      "tool_calls": calls(workspace), "submitted": submitted(workspace),
                      "result": json.loads((trial / "result.json").read_text()) if (trial / "result.json").is_file() else None},
                     indent=2, default=str))
    return 0


def viewer(args) -> int:
    """The Viser view of a running trial: its address on the benchmark host and the tunnel for another machine."""
    trial = _trial_dir(Path(args.root), args.trial_id)
    record = json.loads((trial / "trial.json").read_text())
    if not record.get("viewer"):
        raise SystemExit(f"{args.trial_id} was started without --viewer; the view is off for this trial")
    found = sorted((trial / "captures").glob("*/viewer.json"))
    address = trial / "captures" / str(record.get("environment_id")) / "viewer.json"
    address = address if address.is_file() else (found[0] if found else None)
    if address is None:
        raise SystemExit(f"{args.trial_id}: the view is not up yet (the environment is still starting) or failed to start")
    port = json.loads(address.read_text())["port"]
    with EmbodiedLabClient(CONTROLLER, timeout=60) as client:
        state = environment_state(client, record["environment_id"]) if record.get("environment_id") else None
    print(json.dumps({"trial_id": record["trial_id"], "environment_state": state, "url": f"http://127.0.0.1:{port}",
                      "from_another_machine": f"ssh -N -L {port}:127.0.0.1:{port} <benchmark host>, then open "
                                              f"http://127.0.0.1:{port}"}, indent=2))
    return 0


def watch(args) -> int:
    trial = _trial_dir(Path(args.root), args.trial_id)
    record = json.loads((trial / "trial.json").read_text())
    workspace = trial / "workspace"
    deadline = time.time() + float(record.get("wall_s") or 10800)
    reason = "wall_limit"
    with EmbodiedLabClient(CONTROLLER, timeout=60) as client:
        polls = 0
        while time.time() < deadline:
            if submitted(workspace):
                reason = "submitted"
                break
            state = environment_state(client, record["environment_id"])
            if state not in ("ready", "starting", "running", None):
                reason = f"environment_{state}"
                break
            polls += 1
            if polls % 4 == 0:
                closed = episode_closed(client, record["environment_id"])
                if closed:
                    reason = closed
                    break
            time.sleep(30)
    print(json.dumps({"trial_id": record["trial_id"], "close_reason": reason}))
    args.reason = reason
    return finish(args)


def finish(args) -> int:
    trial = _trial_dir(Path(args.root), args.trial_id)
    record = json.loads((trial / "trial.json").read_text())
    if (trial / "result.json").is_file() and not getattr(args, "force", False):
        print(f"{trial} already finished")
        return 0
    workspace = trial / "workspace"
    environment_id = record["environment_id"]
    result = {k: record.get(k) for k in ("trial_id", "exp_id", "seed", "instance_id", "exposure", "agent", "image", "gpu",
                                          "environment_id", "source_commit", "started_at")}
    reason = getattr(args, "reason", "manual")
    agent_failure = (None if reason in ("wall_limit", "submitted")
                     else agent_infra_error(read_agent_end(getattr(args, "agent_end", None))))
    with EmbodiedLabClient(CONTROLLER, timeout=60) as client:
        was_submitted = submitted(workspace) or reason == "submitted" or episode_closed(client, environment_id) == "submitted"
        result.update(close_reason=reason, tool_calls=calls(workspace), error=None)
        # Docker on a loaded host can miss the container's exit event and fail the stop, which leaves the
        # environment holding its Worker; stop again until it is released (a stop of an exited container completes).
        deadline, attempt = time.time() + STOP_RETRY_S, 0
        while True:
            try:
                operation = client.wait_operation(client.environment_command(environment_id, "stop").id,
                                                  timeout=900 if attempt == 0 else 120)
                error = None if operation.status.value == "completed" else str(operation.error)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            if error is None or environment_state(client, environment_id) in ("stopped", "failed"):
                break
            if attempt == 0:
                result["first_stop_error"] = error
            if time.time() >= deadline:
                result["stop_error"] = error
                print(f"environment {environment_id} did not stop and still holds its Worker: {error}\n"
                      f"Check `docker ps` for its container; once the container has exited, run "
                      f"`./embodiedlab trial finish {args.trial_id} --force` to stop it and score again.",
                      file=sys.stderr)
                break
            attempt += 1
            time.sleep(15)
    started = dt.datetime.fromisoformat(record["started_at"])
    result["wall_s"] = round((dt.datetime.now(dt.timezone.utc) - started).total_seconds(), 1)
    sealed = [p for p in (trial / "captures").glob("*/sealed") if (p / "manifest.json").is_file()]
    process = [p for p in (trial / "captures").glob("*/process") if (p / "episode.json").is_file()]
    end = json.loads((process[0] / "episode.json").read_text()) if process else {}
    logs_path = process[0] / "bench_logs.json" if process else None
    try:
        bench_logs = json.loads(logs_path.read_text()) if logs_path and logs_path.is_file() else {}
    except ValueError:
        bench_logs = {}
    if (end.get("end_reason") == "stopped" and "end_tick" in end and "horizon_ticks" in end
            and end["horizon_ticks"] - end["end_tick"] < end.get("physics_hz", 240)):
        # Less than one simulated second left: the budget ran out (a wait longer than what is left is refused
        # without ending the episode, so the Bench itself rarely records the exhaustion).
        end["end_reason"] = "sim_budget_exhausted"
    result.update(end_reason=end.get("end_reason"), termination_detail=end.get("termination_detail"),
                  process_record=str(process[0]) if process else None)
    if sealed:
        result["status"] = "submitted"
    elif infrastructure_end(end, bench_logs):
        result["status"] = "not_scored"      # the simulation process crashed or the GPU failed
    elif end.get("end_reason") in STATUS_BY_END_REASON:
        result["status"] = STATUS_BY_END_REASON[end["end_reason"]]
    elif not process and (was_submitted or reason.startswith("environment_")):
        result["status"] = "not_scored"      # the environment failed, or did not finalize what the Agent submitted
    elif agent_failure and not was_submitted:
        result.update(status="not_scored", agent_infra_error=agent_failure)   # the agent runner failed, not the Agent
    else:
        result["status"] = "timeout" if reason == "wall_limit" else "not_submitted"
    if sealed:
        try:
            report = evaluate(sealed[0], instance_snapshot.for_evaluation(trial, STORE, record["instance_id"]))
            _write(trial / "evaluation.json", report)
            result.update({k: report.get(k) for k in ("G", "e_rel", "e", "reward", "task_completed", "sim_s_used")})
            result["quantities"] = [{k: q.get(k) for k in ("name", "submitted", "truth", "abs_error", "rel_error",
                                                           "normalized_error", "correct") if k in q}
                                    for q in report.get("quantities") or []]
            result["failed_gates"] = [k for k, v in report["gates"].items() if not v]
            result["bench_restore"] = report.get("bench_restore")
        except Exception as exc:
            result["error"] = f"evaluation: {type(exc).__name__}: {exc}"
            (trial / "evaluation_traceback.txt").write_text(traceback.format_exc())
    elif result["status"] == "not_scored":
        result["error"] = (f"infrastructure: {end.get('termination_detail')}" if infrastructure_end(end, bench_logs)
                           else "no sealed evidence (the environment did not finalize)" if was_submitted
                           else f"agent infrastructure: {agent_failure}" if result.get("agent_infra_error")
                           else f"infrastructure: {result.get('stop_error') or reason}")
    if process and "bench_restore" not in result:
        # Not submitted: which items are back where they started, from the bench as it stood at stop.
        try:
            logs = process[0] / "bench_logs.json"
            result["bench_restore"] = from_process_logs(json.loads(logs.read_text()) if logs.is_file() else {})
        except Exception as exc:
            result["bench_restore"] = {"error": f"{type(exc).__name__}: {exc}"}
    if process:
        # The process score: how far the run got (six levels over the exported records and the evaluation; see
        # benchmark/evaluation/process.py). Written next to the result; a scorer failure never loses the result.
        try:
            score = score_trial(trial, STORE)
            _write(trial / "process_score.json", score)
            result.update(process_level=score["level"], process_score=round(score["process_score"], 4),
                          levels_reached=[n for n, m in score["levels"].items() if m["reached"]],
                          process_flags=score["flags"])
        except Exception as exc:
            result["process_error"] = f"{type(exc).__name__}: {exc}"
    videos = list((trial / "captures").glob("*/video/*.mp4"))
    result["video"] = str(videos[0]) if videos else None
    _write(trial / "result.json", result)
    record.update(status="finished", finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
    _write(trial / "trial.json", record)
    print(json.dumps({k: result.get(k) for k in ("trial_id", "status", "G", "e_rel", "e", "reward", "task_completed",
                                                   "process_level", "process_score", "levels_reached",
                                                   "sim_s_used", "wall_s", "tool_calls", "failed_gates", "error")}, default=str))
    return 0


def prompt(args) -> int:
    trial = _trial_dir(Path(args.root), args.trial_id)
    record = json.loads((trial / "trial.json").read_text())
    print(prompt_text(record, trial / "workspace"))
    return 0


def catalog(args) -> int:
    """The experiments in the registry: a table, or JSON with --json."""
    from benchmark.registry.loader import list_experiments
    rows = []
    for exp_id in list_experiments():
        spec = load(exp_id)
        rows.append({"id": spec.id, "tier": spec.tier, "kit": spec.kit,
                     "title": spec.raw.get("title") or spec.id, "runtime": spec.runtime,
                     "sim_s": spec.budget.get("sim_s"), "wall_s": spec.budget.get("wall_s")})
    if getattr(args, "json", False):
        print(json.dumps({"experiments": rows}, ensure_ascii=False))
        return 0
    print(f"{'id':26}{'tier':10}{'kit':11}{'sim_s':>7}{'wall_s':>8}  title")
    for row in rows:
        print(f"{row['id']:26}{row['tier']:10}{row['kit']:11}{row['sim_s']!s:>7}{row['wall_s']!s:>8}  {row['title']}")
    return 0


def _median(values: list) -> float | None:
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def _mean(values: list) -> float | None:
    return sum(values) / len(values) if values else None


def score(args) -> int:
    """The three headline scores of every finished trial under the root, per agent and exposure level.

    success  every gate passes and e <= 1 (task_completed); the share of runs that succeeded
    process  the process score, cumulative level weights 10/10/10/10/30/30 (benchmark/evaluation/process.py)
    error    e, the largest error divided by its tolerance, over the valid submissions (G = 1); the median
    Experiments average their seeds; the overall line averages the experiments, so that every experiment counts
    equally (the error: the median of every valid submission); a discipline line averages its experiments (the
    error: the median of its valid submissions) and is a breakdown only.
    Runs closed as not_scored (infrastructure failures) are left out and listed for a rerun."""
    root = Path(args.root)
    groups: dict = {}
    rerun = []
    for path in sorted(root.glob("*/result.json")):
        result = json.loads(path.read_text())
        if result.get("status") == "not_scored":
            rerun.append(path.parent.name)
            continue
        if result.get("process_score") is None:
            try:                                    # trials finished before the runner reported it
                result["process_score"] = score_trial(path.parent, STORE)["process_score"]
            except Exception:
                pass
        key = f"{result.get('agent') or '?'} {result.get('exposure') or '?'}"
        groups.setdefault(key, []).append(result)
    kits: dict = {}
    report = {"root": str(root), "groups": {}, "not_scored": rerun}
    for key, rows in sorted(groups.items()):
        by_exp: dict = {}
        for row in rows:
            by_exp.setdefault(row["exp_id"], []).append(row)
        experiments = {}
        for exp, runs in sorted(by_exp.items()):
            if exp not in kits:
                kits[exp] = load(exp).kit
            processed = [r["process_score"] for r in runs if isinstance(r.get("process_score"), (int, float))]
            valid_e = [r["e"] for r in runs if r.get("G") == 1 and isinstance(r.get("e"), (int, float))]
            experiments[exp] = {"kit": kits[exp], "runs": len(runs),
                                "success": sum(r.get("task_completed") is True for r in runs) / len(runs),
                                "process_score": _mean(processed), "valid_submissions": len(valid_e),
                                "median_e": _median(valid_e), "_e": valid_e}
        disciplines = {}
        for kit in sorted({v["kit"] for v in experiments.values()}):
            members = [v for v in experiments.values() if v["kit"] == kit]
            disciplines[kit] = {"experiments": len(members),
                                "success": _mean([m["success"] for m in members]),
                                "process_score": _mean([m["process_score"] for m in members if m["process_score"] is not None]),
                                "median_e": _median([e for m in members for e in m["_e"]])}
        overall = {"runs": len(rows), "experiments": len(experiments),
                   "success": _mean([v["success"] for v in experiments.values()]),
                   "process_score": _mean([v["process_score"] for v in experiments.values() if v["process_score"] is not None]),
                   "median_e": _median([e for v in experiments.values() for e in v["_e"]])}
        for v in experiments.values():
            del v["_e"]
        report["groups"][key] = {"overall": overall, "disciplines": disciplines, "experiments": experiments}
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    fmt = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
    for key, group in report["groups"].items():
        o = group["overall"]
        print(f"== {key}: {o['runs']} runs, {o['experiments']} experiments")
        print(f"   {'':28} {'success':>8} {'process':>8} {'median e':>9}")
        print(f"   {'overall':28} {fmt(o['success']):>8} {fmt(o['process_score']):>8} {fmt(o['median_e']):>9}")
        for kit, d in group["disciplines"].items():
            print(f"   {kit:28} {fmt(d['success']):>8} {fmt(d['process_score']):>8} {fmt(d['median_e']):>9}")
        for exp, v in group["experiments"].items():
            print(f"     {exp:26} {fmt(v['success']):>8} {fmt(v['process_score']):>8} {fmt(v['median_e']):>9}"
                  f"   ({v['runs']} runs, {v['valid_submissions']} valid)")
    if rerun:
        print(f"not scored (infrastructure, rerun): {', '.join(rerun)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("start")
    s.add_argument("--exp", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--exposure", choices=("T0", "T1"), default="T1")
    s.add_argument("--agent", default="manual", help="free-form label of who drives the workspace (manual, my-agent, ...)")
    s.add_argument("--gpu", default="any", help="GPU index of the Worker to run on, or any")
    s.add_argument("--image", default=DEFAULT_IMAGE)
    s.add_argument("--root", default=str(DEFAULT_ROOT))
    s.add_argument("--wall-s", type=float, default=None,
                   help="wall-clock limit in seconds (default: the experiment's budget.wall_s)")
    s.add_argument("--name", help="trial id (default: exp-seed-exposure-agent-timestamp)")
    s.add_argument("--no-watch", action="store_true", help="do not start the background watcher that finishes the trial")
    s.add_argument("--viewer", action="store_true",
                   help="start the read-only Viser view of the whole scene (off by default; see README.md before using it)")
    s.set_defaults(func=start)
    for name, func in (("prompt", prompt), ("status", status), ("viewer", viewer), ("watch", watch), ("finish", finish),
                       ("pack", pack)):
        p = sub.add_parser(name)
        p.add_argument("trial_id")
        p.add_argument("--root", default=str(DEFAULT_ROOT))
        if name == "finish":
            p.add_argument("--force", action="store_true")
            p.add_argument("--reason", default="manual")
            p.add_argument("--agent-end", help="JSON file from the agent runner: why the agent process ended "
                                               "(kind agent_infra_error with a cause and detail -> not_scored)")
        if name == "pack":
            p.add_argument("--controller", default="http://127.0.0.1:18731", help="Controller URL as seen from the agent machine")
            p.add_argument("--port", type=int, default=18731)
            p.add_argument("--python", default="python3", help="Python on the agent machine")
        p.set_defaults(func=func)
    k = sub.add_parser("catalog")
    k.add_argument("--json", action="store_true", help="print JSON instead of a table")
    k.set_defaults(func=catalog)
    c = sub.add_parser("score", help="success, process score and error of every finished trial under the root")
    c.add_argument("--root", default=str(DEFAULT_ROOT))
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=score)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""The stable `embodiedlab` command-line surface."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

from . import __version__
from .client import EmbodiedLabAPIError, EmbodiedLabClient
from .models import TERMINAL_OPERATION_STATUSES, Operation, OperationStatus
from .skills import install_skills, skill_status
from .tool_bundle import package_tool_bundle
from .workflow import package_workflow

_CAPTURE_DIR_HELP = (
    "directory for this run's videos and camera frames; a relative path is "
    "resolved against the current working directory (one sub-directory per "
    "environment). Defaults to the Worker's EMBODIEDLAB_CAPTURE_ROOT."
)


def _start_parameters(args: Any) -> dict[str, Any]:
    """Fold --capture-dir into the parameters the Worker already receives.

    The flag is sugar over ``--parameters '{"capture_dir": ...}'``; an explicit
    entry in --parameters wins so a caller can always reach the raw channel.
    """

    parameters = dict(args.parameters)
    if args.capture_dir and "capture_dir" not in parameters:
        # Resolved here, not on the Worker: "relative to where the CLI ran" is
        # only knowable at the client, and the Worker's own working directory
        # would otherwise silently decide where a relative path lands.
        parameters["capture_dir"] = str(Path(args.capture_dir).expanduser().resolve())
    return parameters


def _json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def _load_json_file(path: str) -> dict[str, Any]:
    parsed = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("JSON file must contain an object")
    return parsed


def _operation_id(value: Any) -> str | None:
    if isinstance(value, Operation):
        return value.id
    if isinstance(value, dict):
        if isinstance(value.get("operation"), dict):
            return value["operation"].get("id")
        if isinstance(value.get("id"), str) and str(value["id"]).startswith("op_"):
            return value["id"]
    return None


def _print(value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _add_global(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--controller-url",
        default=os.getenv("EMBODIEDLAB_CONTROLLER_URL", "http://127.0.0.1:8130"),
    )
    parser.add_argument("--api-key", default=os.getenv("EMBODIEDLAB_API_KEY"))
    parser.add_argument("--project", default=os.getenv("EMBODIEDLAB_PROJECT_ID", "default"))
    parser.add_argument(
        "--role", choices=("user", "operator"), default=os.getenv("EMBODIEDLAB_ROLE", "user")
    )
    parser.add_argument(
        "--no-wait", action="store_true", help="return the operation_id immediately"
    )
    parser.add_argument("--timeout", type=float, help="maximum seconds to wait for an operation")


def _add_env_commands(
    parent: argparse._SubParsersAction, name: str, *, deprecated: bool = False
) -> None:
    env = parent.add_parser(
        name, help="environment lifecycle" + (" (deprecated alias)" if deprecated else "")
    )
    env.set_defaults(deprecated_simulator=deprecated)
    commands = env.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start")
    start.add_argument("--profile", "--backend", dest="profile", default="robocasa")
    start.add_argument("--count", type=int, default=1)
    start.add_argument("--seed", type=int, default=0)
    start.add_argument("--task-id")
    start.add_argument("--image")
    start.add_argument("--gpu")
    start.add_argument(
        "--record-video",
        action="store_true",
        help="record the environment control loop and persist video artifacts",
    )
    start.add_argument("--workflow-digest")
    start.add_argument("--capture-dir", help=_CAPTURE_DIR_HELP)
    start.add_argument("--parameters", type=_json_object, default={})
    commands.add_parser("list")
    status = commands.add_parser("status")
    status.add_argument("env_id")
    for command in ("observe", "states", "verify", "evaluate", "capture", "stop"):
        item = commands.add_parser(command)
        item.add_argument("env_id")
    truth = commands.add_parser("truth")
    truth.add_argument("env_id")
    truth.add_argument("--relation")
    truth.add_argument("--obj-a", dest="obj_a")
    truth.add_argument("--obj-b", dest="obj_b")
    truth.add_argument("--threshold", type=float)
    restore = commands.add_parser("restore")
    restore.add_argument("env_id")
    restore.add_argument("state_id")
    execute = commands.add_parser("exec")
    execute.add_argument("env_id")
    source = execute.add_mutually_exclusive_group(required=True)
    source.add_argument("--file")
    source.add_argument("--code")
    source.add_argument("--stdin", action="store_true")
    run_workflow = commands.add_parser("run-workflow")
    run_workflow.add_argument("env_id")
    run_workflow.add_argument("--task")
    run_workflow.add_argument("--parameters", type=_json_object, default={})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embodiedlab")
    _add_global(parser)
    parser.add_argument("--version", action="version", version=f"embodiedlab-sdk {__version__}")
    areas = parser.add_subparsers(dest="area", required=True)

    controller = areas.add_parser("controller")
    controller.add_subparsers(dest="command", required=True).add_parser("status")
    runtime = areas.add_parser("runtime", help="deprecated controller alias")
    runtime.add_subparsers(dest="command", required=True).add_parser("status")

    _add_env_commands(areas, "env")
    _add_env_commands(areas, "simulator", deprecated=True)

    operation = areas.add_parser("operation")
    operation_commands = operation.add_subparsers(dest="command", required=True)
    operation_commands.add_parser("list")
    for command in ("status", "cancel", "wait", "logs"):
        item = operation_commands.add_parser(command)
        item.add_argument("operation_id")
        if command == "logs":
            item.add_argument("--follow", action="store_true")
            item.add_argument("--after", type=int, default=0)

    tool = areas.add_parser("tool")
    tool_commands = tool.add_subparsers(dest="command", required=True)
    tool_list = tool_commands.add_parser("list")
    tool_list.add_argument("--env", dest="env_id")
    tool_call = tool_commands.add_parser("call")
    tool_call.add_argument("name")
    tool_call.add_argument("--env", dest="env_id", required=True)
    tool_call.add_argument("--args", type=_json_object, default={})
    tool_add = tool_commands.add_parser("add")
    tool_add.add_argument("directory")
    tool_add.add_argument("--env", dest="env_id", required=True)
    tool_remove = tool_commands.add_parser("remove")
    tool_remove.add_argument("--env", dest="env_id", required=True)
    tool_target = tool_remove.add_mutually_exclusive_group(required=True)
    tool_target.add_argument("--digest")
    tool_target.add_argument("--name")
    for command, visible in (("enable", True), ("disable", False)):
        item = tool_commands.add_parser(command)
        item.add_argument("name")
        item.add_argument("--env", dest="env_id", required=True)
        item.set_defaults(visible=visible)
    tool_status = tool_commands.add_parser("status")
    tool_status.add_argument("operation_id")
    tool_status.add_argument("--follow", action="store_true")

    tool_bundle = areas.add_parser("tool-bundle", help="package and publish ToolBundles")
    tool_bundle_commands = tool_bundle.add_subparsers(dest="command", required=True)
    bundle_pack = tool_bundle_commands.add_parser("pack")
    bundle_pack.add_argument("directory")
    bundle_pack.add_argument("--output")
    bundle_upload = tool_bundle_commands.add_parser("upload")
    bundle_upload.add_argument("directory")
    bundle_inspect = tool_bundle_commands.add_parser("inspect")
    bundle_inspect.add_argument("directory")

    service = areas.add_parser("service")
    service_commands = service.add_subparsers(dest="command", required=True)
    service_commands.add_parser("catalog")
    service_commands.add_parser("list")
    for command in ("start", "stop", "status", "logs"):
        item = service_commands.add_parser(command)
        item.add_argument("name")
        if command == "start":
            item.add_argument("--replicas", type=int, default=1)
            item.add_argument("--image")
            item.add_argument("--gpu")
            item.add_argument("--parameters", type=_json_object, default={})

    batch = areas.add_parser("batch")
    batch_commands = batch.add_subparsers(dest="command", required=True)
    batch_start = batch_commands.add_parser("start")
    batch_start.add_argument("--spec", required=True)
    batch_start.add_argument("--max-concurrency", type=int)
    batch_commands.add_parser("list")
    for command in ("status", "wait", "stop"):
        item = batch_commands.add_parser(command)
        item.add_argument("batch_id")

    skill = areas.add_parser("skill")
    skill_commands = skill.add_subparsers(dest="command", required=True)
    for command in ("install", "status", "update"):
        item = skill_commands.add_parser(command)
        item.add_argument("--target")
    skill_commands.add_parser("list")

    workflow = areas.add_parser("workflow")
    workflow_commands = workflow.add_subparsers(dest="command", required=True)
    for command in ("pack", "upload"):
        item = workflow_commands.add_parser(command)
        item.add_argument("directory")
        item.add_argument("--output")
    workflow_run = workflow_commands.add_parser("run")
    workflow_run.add_argument("directory")
    workflow_run.add_argument("--profile", default="python-code")
    workflow_run.add_argument("--seed", type=int, default=0)
    workflow_run.add_argument("--task-id")
    workflow_run.add_argument("--task")
    workflow_run.add_argument("--image")
    workflow_run.add_argument("--gpu")
    workflow_run.add_argument(
        "--record-video",
        action="store_true",
        help="record the environment control loop and persist video artifacts",
    )
    workflow_run.add_argument("--capture-dir", help=_CAPTURE_DIR_HELP)
    workflow_run.add_argument("--parameters", type=_json_object, default={})
    workflow_run.add_argument("--workflow-parameters", type=_json_object, default={})
    workflow_run.add_argument("--keep-env", action="store_true")
    return parser


def _wait_if_needed(client: EmbodiedLabClient, value: Any, args: argparse.Namespace) -> Any:
    if isinstance(value, dict) and not args.no_wait:
        pending = value.get("operations")
        if isinstance(pending, list) and len(pending) > 1:
            completed = []
            for item in pending:
                operation_id = _operation_id(item)
                if operation_id is None:
                    raise RuntimeError("Controller returned an invalid environment operation")
                completed.append(
                    client.wait_operation(operation_id, timeout=args.timeout).model_dump(
                        mode="json"
                    )
                )
            return {**value, "operation": completed[0], "operations": completed}
    operation_id = _operation_id(value)
    if operation_id and not args.no_wait:
        return client.wait_operation(operation_id, timeout=args.timeout)
    return value


def _attach_tool_bundle(client: EmbodiedLabClient, args: argparse.Namespace) -> dict[str, Any]:
    """Pack, idempotently upload, and attach one ToolBundle, then wait.

    The full operation is embedded in the output so a failed attach surfaces
    envd's machine-readable error (code/message/detail) verbatim and drives the
    non-zero exit code in main.
    """

    # Resolved here, not on the Controller: a relative bundle directory is only
    # meaningful where the operator ran the CLI.
    directory = Path(args.directory).expanduser().resolve()
    bundle = package_tool_bundle(directory)
    upload = client.upload_tool_bundle(bundle)
    operation = _wait_if_needed(client, client.attach_tools(args.env_id, bundle.digest), args)
    result: dict[str, Any] = {
        "digest": bundle.digest,
        "name": bundle.manifest.metadata.name,
        "environment_id": args.env_id,
        "upload": upload,
        "operation_id": operation.id,
        "operation": operation.model_dump(mode="json"),
    }
    if operation.status == OperationStatus.COMPLETED:
        revision = operation.output.get("registry_revision")
        if revision is None:
            environment = client.request("GET", f"/v1/environments/{args.env_id}")
            revision = (environment.get("metadata") or {}).get("registry_revision")
        result["registry_revision"] = revision
    return result


def _env_source(args: argparse.Namespace) -> dict[str, Any]:
    if args.file:
        source = Path(args.file).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"code file does not exist: {source}")
        return {"code": source.read_text(encoding="utf-8"), "filename": source.name}
    if args.code is not None:
        return {"code": args.code, "filename": "inline.py"}
    return {"code": sys.stdin.read(), "filename": "stdin.py"}


def _handle(client: EmbodiedLabClient, args: argparse.Namespace) -> Any:
    if args.area in {"controller", "runtime"}:
        if args.area == "runtime":
            warnings.warn(
                "`embodiedlab runtime` is deprecated; use `embodiedlab controller`",
                DeprecationWarning,
                stacklevel=2,
            )
        return client.ready()
    if args.area in {"env", "simulator"}:
        if args.area == "simulator":
            print("warning: `embodiedlab simulator` is deprecated; use `embodiedlab env`", file=sys.stderr)
        if args.command == "start":
            return _wait_if_needed(
                client,
                client.start_environments(
                    profile=args.profile,
                    count=args.count,
                    seed=args.seed,
                    task_id=args.task_id,
                    image=args.image,
                    gpu=args.gpu,
                    record_video=args.record_video,
                    workflow_digest=args.workflow_digest,
                    parameters=_start_parameters(args),
                ),
                args,
            )
        if args.command == "list":
            return client.request(
                "GET", "/v1/environments", params={"project_id": client.project_id}
            )
        if args.command == "status":
            return client.request("GET", f"/v1/environments/{args.env_id}")
        arguments = _env_source(args) if args.command == "exec" else {}
        if args.command == "restore":
            arguments = {"state_id": args.state_id}
        if args.command == "run-workflow":
            arguments = {"task": args.task, "parameters": args.parameters}
        if args.command == "truth" and args.relation:
            arguments = {
                "relation": args.relation,
                "obj_a": args.obj_a,
                "obj_b": args.obj_b,
                "threshold": args.threshold,
            }
        return _wait_if_needed(
            client, client.environment_command(args.env_id, args.command, arguments), args
        )
    if args.area == "operation":
        if args.command == "list":
            return client.request(
                "GET",
                "/v1/operations",
                params={"project_id": client.project_id, "include_output": False},
            )
        if args.command == "status":
            return client.operation(args.operation_id)
        if args.command == "cancel":
            return client.cancel_operation(args.operation_id)
        if args.command == "wait":
            return client.wait_operation(args.operation_id, timeout=args.timeout)
        if args.command == "logs":
            after = args.after
            while True:
                result = client.request(
                    "GET", f"/v1/operations/{args.operation_id}/logs", params={"after": after}
                )
                for item in result["entries"]:
                    _print(item)
                    after = item["offset"]
                operation = client.operation(args.operation_id)
                if not args.follow or operation.status in TERMINAL_OPERATION_STATUSES:
                    return {"operation": operation, "last_offset": after}
                time.sleep(0.5)
    if args.area == "tool":
        if args.command == "list":
            tools = client.request("GET", "/v1/tools", params={"environment_id": args.env_id})
            if not args.env_id:
                return tools
            # /v1/tools only carries the tool list; the registry fields the
            # Controller reconciled after each mutation live on the environment.
            metadata = client.request("GET", f"/v1/environments/{args.env_id}").get("metadata")
            metadata = metadata or {}
            return {
                "tools": tools,
                "registry_revision": metadata.get("registry_revision"),
                "disabled": metadata.get("disabled_tools", []),
                "bundles": metadata.get("tool_bundles", []),
            }
        if args.command == "add":
            return _attach_tool_bundle(client, args)
        if args.command == "remove":
            return _wait_if_needed(
                client, client.detach_tools(args.env_id, digest=args.digest, name=args.name), args
            )
        if args.command in {"enable", "disable"}:
            return _wait_if_needed(
                client,
                client.set_tool_visibility(args.env_id, args.name, visible=args.visible),
                args,
            )
        if args.command == "status":
            if args.follow:
                return client.wait_operation(args.operation_id, timeout=args.timeout)
            return client.operation(args.operation_id)
        return _wait_if_needed(client, client.call_tool(args.env_id, args.name, args.args), args)
    if args.area == "tool-bundle":
        bundle = package_tool_bundle(args.directory)
        if args.command == "inspect":
            return bundle.manifest
        if args.command == "upload":
            return client.upload_tool_bundle(bundle)
        output = (
            Path(args.output).expanduser().resolve()
            if args.output
            else Path(args.directory).expanduser().resolve().with_suffix(".tar.gz")
        )
        output.write_bytes(bundle.content)
        return {"digest": bundle.digest, "output": str(output), "files": list(bundle.files)}
    if args.area == "service":
        if args.command == "catalog":
            return client.request("GET", "/v1/services/catalog")
        if args.command == "list":
            return client.request("GET", "/v1/services", params={"project_id": client.project_id})
        if args.command in {"status", "logs"}:
            services = client.request(
                "GET", "/v1/services", params={"project_id": client.project_id}
            )
            matches = [item for item in services if item.get("name") == args.name]
            return matches[0] if matches else {"name": args.name, "status": "not-deployed"}
        replicas = args.replicas if args.command == "start" else 0
        value = client.request(
            "POST",
            "/v1/services/actions",
            json={
                "project_id": client.project_id,
                "name": args.name,
                "replicas": replicas,
                "image": getattr(args, "image", None),
                "gpu": getattr(args, "gpu", None),
                "parameters": getattr(args, "parameters", {}),
            },
        )
        return _wait_if_needed(client, value, args)
    if args.area == "batch":
        if args.command == "start":
            spec = _load_json_file(args.spec)
            if args.max_concurrency is not None:
                spec["max_concurrency"] = args.max_concurrency
            return client.request(
                "POST", "/v1/batches", json={"project_id": client.project_id, **spec}
            )
        if args.command == "list":
            return client.request("GET", "/v1/batches", params={"project_id": client.project_id})
        if args.command == "stop":
            return client.request("POST", f"/v1/batches/{args.batch_id}/cancel")
        deadline = time.monotonic() + args.timeout if args.timeout is not None else None
        while True:
            value = client.request("GET", f"/v1/batches/{args.batch_id}")
            if args.command == "status" or value["state"] in {
                "completed",
                "failed",
                "interrupted",
                "cancelled",
            }:
                return value
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("batch wait timed out")
            time.sleep(0.5)
    if args.area == "skill":
        if args.command == "install":
            return install_skills(target=args.target)
        if args.command == "update":
            return install_skills(target=args.target, update=True)
        if args.command == "status":
            return skill_status(target=args.target)
        return client.request("GET", "/v1/skills")
    if args.area == "workflow":
        bundle = package_workflow(args.directory)
        output_arg = getattr(args, "output", None)
        output = Path(output_arg).expanduser().resolve() if output_arg else None
        if output:
            output.write_bytes(bundle.content)
        result: dict[str, Any] = {
            "digest": bundle.digest,
            "files": list(bundle.files),
            "size": len(bundle.content),
            "output": str(output) if output else None,
        }
        if args.command in {"upload", "run"}:
            result["upload"] = client.upload_workflow(bundle)
        if args.command == "run":
            if args.no_wait:
                raise ValueError("embodiedlab workflow run does not support --no-wait")
            started = client.start_environments(
                profile=args.profile,
                count=1,
                seed=args.seed,
                task_id=args.task_id,
                image=args.image,
                gpu=args.gpu,
                record_video=args.record_video,
                workflow_digest=bundle.digest,
                parameters=_start_parameters(args),
            )
            start_id = _operation_id(started)
            if start_id is None:
                raise RuntimeError("Controller did not return an environment start operation")
            start = client.wait_operation(start_id, timeout=args.timeout)
            result["start"] = start.model_dump(mode="json")
            environment_ids = start.input.get("environment_ids", [])
            if not environment_ids:
                raise RuntimeError("environment start did not allocate an environment")
            env_id = str(environment_ids[0])
            result["environment_id"] = env_id
            if start.status != OperationStatus.COMPLETED:
                return result
            try:
                run = client.environment_command(
                    env_id,
                    "run-workflow",
                    {"task": args.task, "parameters": args.workflow_parameters},
                )
                run = client.wait_operation(run.id, timeout=args.timeout)
                result["run"] = run.model_dump(mode="json")
            finally:
                if not args.keep_env:
                    stopped = client.environment_command(env_id, "stop")
                    stopped = client.wait_operation(stopped.id, timeout=args.timeout)
                    result["stop"] = stopped.model_dump(mode="json")
        return result
    raise ValueError("unsupported command")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.area == "skill" and args.command in {"install", "status", "update"}:
            _print(_handle(None, args))  # type: ignore[arg-type]
            return 0
        with EmbodiedLabClient(
            args.controller_url,
            api_key=args.api_key,
            project_id=args.project,
            role=args.role,
        ) as client:
            value = _handle(client, args)
        _print(value)
        failed_statuses = {
            OperationStatus.FAILED.value,
            OperationStatus.INTERRUPTED.value,
            OperationStatus.CANCELLED.value,
        }
        def contains_failed_operation(item: Any) -> bool:
            if isinstance(item, Operation):
                return item.status.value in failed_statuses
            if isinstance(item, dict):
                if item.get("status") in failed_statuses:
                    return True
                return any(contains_failed_operation(child) for child in item.values())
            if isinstance(item, list):
                return any(contains_failed_operation(child) for child in item)
            return False

        if contains_failed_operation(value):
            return 2
        return 0
    except (EmbodiedLabAPIError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        _print({"ok": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

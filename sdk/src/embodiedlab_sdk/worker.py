"""Role-separated Worker daemon with registration, heartbeat, and assignment polling."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import logging
import os
import socket
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from . import __version__
from .progress import ExecutionContext, OperationProgressReporter
from .sim_bridge import SimHeartbeatClient
from .tool_bundle import ToolBundleCache
from .viewer import LocalViewerTransport
from .viewer_tunnel_worker import WorkerViewerTunnelClient
from .workflow import WorkflowCache

Executor = Callable[[dict[str, Any]], dict[str, Any]]
LOGGER = logging.getLogger(__name__)

# How long a Worker keeps retrying a status report the Controller does not answer. A completed result is
# only held in this process, so it is retried well past a Controller pause instead of being dropped.
REPORT_RETRY_SECONDS = 900.0
REPORT_RETRY_MAX_DELAY = 10.0
TERMINAL_TRANSITION_DETAIL = "terminal operation cannot transition"


def _transient_controller_error(exc: httpx.HTTPError) -> bool:
    """A timeout, dropped connection or 5xx: the Controller may still be fine a moment later."""

    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


def _response_detail(response: Any) -> str:
    try:
        detail = response.json().get("detail")
    except Exception:
        return ""
    return detail if isinstance(detail, str) else ""


def _load_executor(spec: str | None) -> Executor:
    if not spec:

        def unavailable(operation: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError(
                f"no adapter executor installed for {operation['kind']}; "
                "install a matching OCI/profile plugin"
            )

        return unavailable
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError("executor must use module:function syntax")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"executor {spec} is not callable")
    return function


class WorkerDaemon:
    def __init__(
        self,
        controller_url: str,
        role: str,
        *,
        name: str,
        executor: Executor,
        labels: dict[str, str] | None = None,
        capacity: dict[str, Any] | None = None,
        registration_key: str | None = None,
        heartbeat_interval: float = 5.0,
        api_key: str | None = None,
        workflow_cache_dir: str | Path | None = None,
        tool_bundle_cache_dir: str | Path | None = None,
        progress_spool_dir: str | Path | None = None,
        viewer_tunnel_enabled: bool = False,
        sim_bridge_enabled: bool = False,
        sim_bridge_url: str | None = None,
        sim_worker_id: str | None = None,
        sim_bridge_api_key: str | None = None,
        sim_heartbeat_interval: float = 5.0,
    ) -> None:
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(base_url=controller_url.rstrip("/"), timeout=30, headers=headers)
        self.role = role
        self.name = name
        self.executor = executor
        self.labels = labels or {}
        self.capacity = capacity or {}
        self.registration_key = registration_key
        self.heartbeat_interval = heartbeat_interval
        self.api_key = api_key
        self.viewer_tunnel_enabled = viewer_tunnel_enabled
        self.worker_id: str | None = None
        self.report_retry_seconds = REPORT_RETRY_SECONDS
        self._viewer_tunnel: WorkerViewerTunnelClient | None = None
        self._sim_bridge: SimHeartbeatClient | None = None
        self._sim_bridge_enabled = sim_bridge_enabled
        self._sim_bridge_url = sim_bridge_url
        self._sim_worker_id = sim_worker_id
        self._sim_bridge_api_key = sim_bridge_api_key
        self._sim_heartbeat_interval = sim_heartbeat_interval
        self._agent_generation = uuid4().hex
        self._last_heartbeat = 0.0
        self._progress_reporters: list[OperationProgressReporter] = []
        cache_root = workflow_cache_dir or os.getenv("EMBODIEDLAB_WORKFLOW_CACHE_DIR", "~/.embodiedlab/workflows")
        self.workflow_cache = WorkflowCache(cache_root)
        bundle_cache_root = tool_bundle_cache_dir or os.getenv(
            "EMBODIEDLAB_TOOL_BUNDLE_CACHE_DIR", "~/.embodiedlab/tool-bundles"
        )
        self.tool_bundle_cache = ToolBundleCache(bundle_cache_root)
        self._progress_spool_dir = Path(
            progress_spool_dir or os.getenv("EMBODIEDLAB_PROGRESS_SPOOL_DIR", "~/.embodiedlab/progress")
        ).expanduser()

    def _start_viewer_tunnel(self) -> None:
        if not self.viewer_tunnel_enabled or self._viewer_tunnel is not None:
            return
        if self.role != "simulation-worker":
            raise RuntimeError("Viewer tunnel is only valid for simulation Workers")
        if not self.worker_id or not self.api_key or not self.registration_key:
            raise RuntimeError(
                "Viewer tunnel requires a registered Worker, API key, and stable registration key"
            )
        transport = self.executor
        if not isinstance(transport, LocalViewerTransport):
            raise RuntimeError(
                "Viewer tunnel executor must implement the Worker-local Viewer transport contract"
            )
        self._viewer_tunnel = WorkerViewerTunnelClient(
            str(self.http.base_url), self.worker_id, self.api_key, transport
        )
        self._viewer_tunnel.start()

    def _environment_ids(self) -> list[str]:
        inventory = getattr(self.executor, "list_environment_ids", None)
        if not callable(inventory):
            return []
        values = inventory()
        return [value for value in values if isinstance(value, str) and value]

    def _start_sim_bridge(self) -> None:
        if not self._sim_bridge_enabled or self._sim_bridge is not None:
            return
        if self.role != "simulation-worker":
            raise RuntimeError("Sim bridge is only valid for simulation Workers")
        if not all(
            (
                self._sim_bridge_url,
                self._sim_worker_id,
                self._sim_bridge_api_key,
                self.registration_key,
            )
        ):
            raise RuntimeError(
                "Sim bridge requires URL, Sim Worker ID, API key, and registration key"
            )
        self._sim_bridge = SimHeartbeatClient(
            self._sim_bridge_url,
            self._sim_worker_id,
            self._sim_bridge_api_key,
            self.registration_key,
            sdk_worker_id=lambda: self.worker_id,
            environment_ids=self._environment_ids,
            capacity=self.capacity,
            agent_generation=self._agent_generation,
            interval=self._sim_heartbeat_interval,
        )
        self._sim_bridge.start()

    def register(self) -> str:
        response = self.http.post(
            "/v1/workers/register",
            json={
                "name": self.name,
                "role": self.role,
                "labels": self.labels,
                "capacity": self.capacity,
                "registration_key": self.registration_key,
                "version": __version__,
            },
        )
        response.raise_for_status()
        self.worker_id = response.json()["id"]
        self._last_heartbeat = 0.0
        self._start_viewer_tunnel()
        self._start_sim_bridge()
        return self.worker_id

    def heartbeat(self) -> None:
        assert self.worker_id
        response = self.http.post(
            f"/v1/workers/{self.worker_id}/heartbeat",
            json={
                "state": "ready",
                "capacity": self.capacity,
                "agent_generation": self._agent_generation,
            },
        )
        response.raise_for_status()
        self._last_heartbeat = time.monotonic()

    def _heartbeat_if_due(self) -> None:
        if time.monotonic() - self._last_heartbeat >= self.heartbeat_interval:
            self.heartbeat()
        self._flush_pending_progress()

    def _tool_bundle_dir(self, digest: str, project_id: str) -> Path:
        cached = self.tool_bundle_cache.path(digest)
        if cached is not None:
            return cached
        with self.http.stream(
            "GET",
            f"/v1/tool-bundles/{digest}",
            headers={"X-EmbodiedLab-Project": project_id},
        ) as response:
            response.raise_for_status()
            return self.tool_bundle_cache.materialize(
                digest,
                response.iter_bytes(chunk_size=1024 * 1024),
            )

    def _flush_pending_progress(self) -> None:
        pending: list[OperationProgressReporter] = []
        for reporter in self._progress_reporters:
            try:
                if not reporter.flush():
                    pending.append(reporter)
            except httpx.TransportError:
                pending.append(reporter)
        self._progress_reporters = pending

    def _execute(
        self, operation: dict[str, Any], context: ExecutionContext | None = None
    ) -> dict[str, Any]:
        values = operation["input"]
        project_id = operation.get("project_id", "default")
        digest = values.get("workflow_digest")
        if digest is not None:
            workflow_dir = self.workflow_cache.path(digest)
            if workflow_dir is None:
                with self.http.stream(
                    "GET",
                    f"/v1/workflows/{digest}",
                    headers={"X-EmbodiedLab-Project": project_id},
                ) as response:
                    response.raise_for_status()
                    workflow_dir = self.workflow_cache.materialize(
                        digest,
                        response.iter_bytes(chunk_size=1024 * 1024),
                    )
            values["workflow_dir"] = str(workflow_dir)
        bundle_digests = values.get("tool_bundle_digests")
        if bundle_digests:
            if not isinstance(bundle_digests, list):
                raise ValueError("tool_bundle_digests must be a list of sha256 digests")
            values["tool_bundle_dirs"] = {
                item: str(self._tool_bundle_dir(item, project_id)) for item in bundle_digests
            }
        execute_with_context = getattr(self.executor, "execute_with_context", None)
        if callable(execute_with_context) and context is not None:
            return execute_with_context(operation, context)
        return self.executor(operation)

    def _report_transition(
        self,
        operation_id: str,
        patch: dict[str, Any],
        *,
        applied_details: tuple[str, ...] = (),
        retry_seconds: float | None = None,
    ) -> None:
        """Post an operation transition, retrying a Controller that does not answer.

        A timed-out report may still have been applied by a Controller that answered too late, so a
        409 whose detail starts with one of ``applied_details`` counts as already applied.
        """

        deadline = time.monotonic() + (
            self.report_retry_seconds if retry_seconds is None else retry_seconds
        )
        delay = 0.5
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self.http.post(
                    f"/v1/operations/{operation_id}/transition", json=patch
                )
                status_code = getattr(response, "status_code", 200)
                if status_code == 409 and applied_details:
                    detail = _response_detail(response)
                    if detail.startswith(applied_details):
                        LOGGER.warning(
                            "Operation %s %s report answered 409 (%s) on attempt %d; "
                            "treating it as already applied",
                            operation_id,
                            patch.get("status"),
                            detail,
                            attempt,
                        )
                        return
                response.raise_for_status()
                if attempt > 1:
                    LOGGER.warning(
                        "Operation %s %s report delivered on attempt %d",
                        operation_id,
                        patch.get("status"),
                        attempt,
                    )
                return
            except httpx.HTTPError as exc:
                if not _transient_controller_error(exc) or time.monotonic() + delay > deadline:
                    raise
                LOGGER.warning(
                    "Operation %s %s report failed (error=%s, attempt %d); retrying in %.1fs",
                    operation_id,
                    patch.get("status"),
                    type(exc).__name__,
                    attempt,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, REPORT_RETRY_MAX_DELAY)

    def run_once(self) -> bool:
        assert self.worker_id
        response = self.http.post(f"/v1/workers/{self.worker_id}/assignments/next")
        response.raise_for_status()
        operation = response.json()
        if operation is None:
            return False
        operation_id = operation["id"]
        # A retry after a lost answer finds the Operation already running for this Worker.
        self._report_transition(
            operation_id,
            {"status": "running", "worker_id": self.worker_id},
            applied_details=("invalid operation transition running->running",),
        )
        executor_operation = copy.deepcopy(operation)
        executor_operation["input"] = dict(executor_operation.get("input") or {})
        executor_operation["input"].pop("workflow_dir", None)
        executor_operation["input"].pop("tool_bundle_dirs", None)
        reporter = OperationProgressReporter(
            self.http,
            operation_id,
            self.worker_id,
            self._agent_generation,
            attempt=int(executor_operation.get("input", {}).get("attempt", 1)),
            trace_id=executor_operation.get("input", {}).get("trace_id"),
            spool_dir=self._progress_spool_dir,
        )
        context = ExecutionContext(progress=reporter)
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self._execute, executor_operation, context)
        controller_unreachable_since: float | None = None
        while not future.done():
            heartbeat_in = max(
                0.0,
                self.heartbeat_interval - (time.monotonic() - self._last_heartbeat),
            )
            time.sleep(min(0.25, heartbeat_in))
            try:
                self._heartbeat_if_due()
                current = self.http.get(f"/v1/operations/{operation_id}")
                current.raise_for_status()
            except httpx.HTTPError as exc:
                # The executor still owns the running Operation: a Controller pause must not make this
                # Worker drop it and claim the next one, or its result is never reported.
                if not _transient_controller_error(exc):
                    raise
                if controller_unreachable_since is None:
                    controller_unreachable_since = time.monotonic()
                    LOGGER.warning(
                        "Controller request failed while Worker %s runs operation %s; "
                        "keeping the operation (error=%s)",
                        self.worker_id,
                        operation_id,
                        type(exc).__name__,
                    )
                time.sleep(min(1.0, self.heartbeat_interval))
                continue
            if controller_unreachable_since is not None:
                LOGGER.warning(
                    "Controller reachable again after %.1fs; Worker %s still runs operation %s",
                    time.monotonic() - controller_unreachable_since,
                    self.worker_id,
                    operation_id,
                )
                controller_unreachable_since = None
            if current.json()["status"] == "cancelled":
                cancel = getattr(self.executor, "cancel", None)
                if callable(cancel):
                    cancel(operation_id, executor_operation)
                pool.shutdown(wait=False, cancel_futures=True)
                return True
        pool.shutdown(wait=True)
        progress_error: httpx.HTTPError | None = None
        try:
            if not reporter.flush():
                self._progress_reporters.append(reporter)
        except httpx.HTTPError as exc:
            progress_error = exc
        try:
            output = future.result()
            if progress_error is not None:
                raise progress_error
            patch = {
                "status": "completed",
                "worker_id": self.worker_id,
                "output": output,
            }
        except KeyboardInterrupt:
            patch = {
                "status": "interrupted",
                "worker_id": self.worker_id,
                "error": {"type": "KeyboardInterrupt"},
            }
            self._report_transition(
                operation_id,
                patch,
                applied_details=(TERMINAL_TRANSITION_DETAIL,),
                retry_seconds=10.0,
            )
            raise
        except Exception as exc:
            error: dict[str, Any] = {"type": type(exc).__name__, "message": str(exc)}
            # Executors that fail with a machine-readable payload (for example
            # EnvironmentCommandError from a ToolGuard rejection) expose it as
            # ``detail``; pass it through so Agents can react to error codes
            # instead of parsing message strings.
            detail = getattr(exc, "detail", None)
            if isinstance(detail, dict):
                error["detail"] = detail
            patch = {
                "status": "failed",
                "worker_id": self.worker_id,
                "error": error,
            }
        # A 409 for a terminal Operation means an earlier attempt landed, or the client cancelled it.
        self._report_transition(
            operation_id, patch, applied_details=(TERMINAL_TRANSITION_DETAIL,)
        )
        return True

    def _controller_forgot_worker(self, exc: httpx.HTTPError) -> bool:
        """Return whether a response proves that this Worker identity is gone."""

        if not self.worker_id or not isinstance(exc, httpx.HTTPStatusError):
            return False
        if exc.response.status_code not in {404, 410}:
            return False
        worker_path = f"/v1/workers/{self.worker_id}"
        request_path = exc.request.url.path.rstrip("/")
        return request_path.endswith(
            (f"{worker_path}/heartbeat", f"{worker_path}/assignments/next")
        )

    def run_forever(self) -> None:
        while True:
            try:
                if self.worker_id is None:
                    self.register()
                self._heartbeat_if_due()
                if not self.run_once():
                    time.sleep(min(1.0, self.heartbeat_interval))
            except httpx.HTTPError as exc:
                # A timeout or transient Controller failure does not invalidate an
                # already registered identity. Re-register only when the Controller
                # explicitly says that this Worker no longer exists.
                if self._controller_forgot_worker(exc):
                    LOGGER.warning(
                        "Controller no longer recognizes Worker %s; registering again",
                        self.worker_id,
                    )
                    self.worker_id = None
                else:
                    status = (
                        exc.response.status_code
                        if isinstance(exc, httpx.HTTPStatusError)
                        else None
                    )
                    LOGGER.warning(
                        "Controller request failed for Worker %s; retaining identity "
                        "(error=%s, status=%s)",
                        self.worker_id or "unregistered",
                        type(exc).__name__,
                        status,
                    )
                time.sleep(min(5.0, max(0.2, self.heartbeat_interval)))


def _mapping(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def _enabled(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embodiedlab-worker")
    parser.add_argument(
        "--controller-url", default=os.getenv("EMBODIEDLAB_CONTROLLER_URL", "http://127.0.0.1:8130")
    )
    parser.add_argument("--role", required=True, choices=("simulation-worker", "model-worker"))
    parser.add_argument("--name", default=socket.gethostname())
    parser.add_argument("--executor", default=os.getenv("EMBODIEDLAB_WORKER_EXECUTOR"))
    parser.add_argument("--labels", type=_mapping, default={})
    parser.add_argument("--capacity", type=_mapping, default={})
    parser.add_argument("--registration-key", default=os.getenv("EMBODIEDLAB_WORKER_REGISTRATION_KEY"))
    parser.add_argument("--heartbeat-interval", type=float, default=5.0)
    parser.add_argument("--api-key", default=os.getenv("EMBODIEDLAB_API_KEY"))
    parser.add_argument("--workflow-cache-dir", default=os.getenv("EMBODIEDLAB_WORKFLOW_CACHE_DIR"))
    parser.add_argument("--progress-spool-dir", default=os.getenv("EMBODIEDLAB_PROGRESS_SPOOL_DIR"))
    parser.add_argument(
        "--tool-bundle-cache-dir", default=os.getenv("EMBODIEDLAB_TOOL_BUNDLE_CACHE_DIR")
    )
    parser.add_argument(
        "--viewer-tunnel",
        action="store_true",
        default=_enabled(os.getenv("EMBODIEDLAB_WORKER_VIEWER_TUNNEL_ENABLED")),
    )
    parser.add_argument("--sim-bridge-url", default=os.getenv("EMBODIEDLAB_SIM_CONTROL_PLANE_URL"))
    parser.add_argument(
        "--sim-bridge",
        action="store_true",
        default=_enabled(os.getenv("EMBODIEDLAB_SIM_BRIDGE_ENABLED")),
    )
    parser.add_argument("--sim-worker-id", default=os.getenv("EMBODIEDLAB_SIM_WORKER_ID"))
    parser.add_argument("--sim-bridge-api-key", default=os.getenv("EMBODIEDLAB_SIM_WORKER_API_KEY"))
    parser.add_argument(
        "--sim-heartbeat-interval",
        type=float,
        default=float(os.getenv("EMBODIEDLAB_SIM_HEARTBEAT_INTERVAL", "5")),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    WorkerDaemon(
        args.controller_url,
        args.role,
        name=args.name,
        executor=_load_executor(args.executor),
        labels=args.labels,
        capacity=args.capacity,
        registration_key=args.registration_key,
        heartbeat_interval=args.heartbeat_interval,
        api_key=args.api_key,
        workflow_cache_dir=args.workflow_cache_dir,
        tool_bundle_cache_dir=args.tool_bundle_cache_dir,
        progress_spool_dir=args.progress_spool_dir,
        viewer_tunnel_enabled=args.viewer_tunnel,
        sim_bridge_enabled=args.sim_bridge,
        sim_bridge_url=args.sim_bridge_url,
        sim_worker_id=args.sim_worker_id,
        sim_bridge_api_key=args.sim_bridge_api_key,
        sim_heartbeat_interval=args.sim_heartbeat_interval,
    ).run_forever()


if __name__ == "__main__":
    main()

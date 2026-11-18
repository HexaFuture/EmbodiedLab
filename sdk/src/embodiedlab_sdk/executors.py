"""Built-in generic OCI executor; profile-specific behavior stays in thin plugins/images."""

from __future__ import annotations

import base64
import json
import os
import socket
import time
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import quote

import httpx

from .catalog import environment_profile, service_spec
from .container_runtime import ContainerRuntimeError, ContainerSpec, OCIRuntime
from .dynamic_tools import MAX_TOOL_BUNDLE_BYTES
from .tool_bundle import package_tool_bundle
from .viewer import (
    ViewerHTTPResponse,
    ViewerTransportUnavailable,
    sanitize_viewer_response_headers,
)


class EnvironmentCommandError(RuntimeError):
    """A command inside the environment failed with a machine-readable error.

    ``detail`` carries envd's structured payload (``code``/``message`` plus
    fields like ``filename``/``line`` for ToolGuard rejections) so the Worker
    can hand it back on ``operation.error.detail`` and an Agent can repair its
    bundle programmatically instead of parsing a stringified traceback.
    """

    def __init__(self, detail: dict[str, Any]) -> None:
        super().__init__(
            str(detail.get("message") or json.dumps(detail, ensure_ascii=False, sort_keys=True))
        )
        self.detail = detail

# Where an environment finds its own writable output directory.  The host
# side is per environment (see env.start); inside the container the path is
# fixed so image bootstraps can hard-code it.
CAPTURE_TARGET = "/opt/embodiedlab/capture"


def _capture_base(
    root: str | None, parameters: dict[str, Any], project_id: str
) -> Path | None:
    """Resolve where this request's captures live, or None when capture is off.

    ``capture_dir`` names a host directory on this Worker and is used as given.
    It must already be absolute: the CLI resolves a relative path against the
    operator's working directory, and re-resolving here would instead pick up
    the Worker process's own directory, which nobody typed.

    Note that the caller therefore chooses which host directory gets bind
    mounted read-write into the container, and the container writes as root.
    """

    label = parameters.get("capture_dir")
    if label:
        candidate = Path(str(label)).expanduser()
        if not candidate.is_absolute():
            raise ValueError(
                f"capture_dir {label!r} must be absolute; the CLI resolves a "
                "relative path against its own working directory"
            )
        return candidate.resolve()
    if not root:
        return None
    return Path(root).expanduser().resolve() / project_id


def _port_open(host: str, port: int, *, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _reserve_loopback_port() -> int:
    """Select an ephemeral host port for one host-network environment daemon."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _requested_horizon(parameters: dict[str, Any]) -> int | None:
    """Validate the optional generic environment step budget."""

    if "horizon" not in parameters:
        return None
    horizon = parameters["horizon"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("parameters.horizon must be a positive integer")
    return horizon


class GenericOCIExecutor:
    def __init__(self) -> None:
        self._runtime: OCIRuntime | None = None
        self._containers: dict[str, str] = {}
        # Host capture directories of --record-video environments; env.stop
        # finalizes these and env.capture maps container paths back to them.
        self._capture_dirs: dict[str, Path] = {}
        self._viewer_ports: dict[str, int] = {}
        self._lock = RLock()

    @property
    def runtime(self) -> OCIRuntime:
        if self._runtime is None:
            self._runtime = OCIRuntime()
        return self._runtime

    def __call__(self, operation: dict[str, Any]) -> dict[str, Any]:
        kind = operation["kind"]
        values = operation.get("input", {})
        if kind == "env.start":
            profile = environment_profile(values["profile"])
            parameters = values.get("parameters", {})
            requested_horizon = _requested_horizon(parameters)
            image = values.get("image") or profile.get("default_image")
            if not image:
                raise ValueError(f"environment profile {profile['name']} requires an OCI image")
            workflow_digest = values.get("workflow_digest")
            workflow_dir = values.get("workflow_dir") if workflow_digest else None
            if workflow_digest and not workflow_dir:
                raise RuntimeError("Worker did not materialize the Workflow artifact")
            mounts: list[tuple[str, str, str]] = []
            # Writable scratch mounts get one subdirectory per environment (e.g. the
            # simulator's shader cache, kept off the Docker root file system); an
            # optional mount is skipped when the Worker setting is absent.
            per_environment: list[tuple[str, str, str]] = []
            for mount in profile.get("mounts", []):
                source = os.getenv(str(mount["source_env"]))
                if not source:
                    if mount.get("optional"):
                        continue
                    raise ValueError(
                        f"environment profile {profile['name']} requires Worker setting "
                        f"{mount['source_env']}"
                    )
                entry = (source, str(mount["target"]), str(mount.get("mode", "readonly")))
                (per_environment if mount.get("per_environment") else mounts).append(entry)
            if workflow_dir:
                mounts.append((str(workflow_dir), "/opt/embodiedlab/workflow", "readonly"))
            capture_base = _capture_base(
                os.getenv("EMBODIEDLAB_CAPTURE_ROOT"), parameters, str(operation["project_id"])
            )
            record_video = bool(values.get("record_video", False))
            results = []
            started: list[tuple[str, str]] = []
            try:
                for index, environment_id in enumerate(values.get("environment_ids", [])):
                    name = f"embodiedlab-{environment_id}"
                    # The profile-level mounts are shared by every environment in
                    # the batch, so the writable capture directory has to be
                    # appended per environment: one shared output mount would put
                    # every replica's video_exec-0001.mp4 on the same path.
                    environment_mounts = list(mounts)
                    for root, target, mode in per_environment:
                        scratch = Path(root).expanduser() / environment_id
                        scratch.mkdir(parents=True, exist_ok=True)
                        environment_mounts.append((str(scratch), target, mode))
                    capture_dir: Path | None = None
                    if capture_base is not None:
                        capture_dir = capture_base / environment_id
                        # "--mount type=bind" refuses to start the container when
                        # the source is missing; unlike "-v" it never creates it.
                        capture_dir.mkdir(parents=True, exist_ok=True)
                        environment_mounts.append(
                            (str(capture_dir), CAPTURE_TARGET, "readwrite")
                        )
                    if record_video and capture_dir is None:
                        # --record-video always needs a writable capture mount;
                        # without an operator-selected capture directory fall
                        # back to the Worker's environment-capture root.
                        capture_root = Path(
                            os.getenv(
                                "EMBODIEDLAB_ENV_CAPTURE_ROOT",
                                "~/.embodiedlab/environment-captures",
                            )
                        ).expanduser()
                        capture_dir = (capture_root / environment_id).resolve()
                        capture_dir.mkdir(parents=True, exist_ok=True)
                        environment_mounts.append(
                            (str(capture_dir), CAPTURE_TARGET, "readwrite")
                        )
                    network = profile.get("network")
                    command = list(parameters.get("command", profile.get("command", [])))
                    parameter_environment = dict(parameters.get("environment") or {})
                    if requested_horizon is not None:
                        configured_horizon = parameter_environment.get("EMBODIEDLAB_MAX_STEPS")
                        if configured_horizon is not None:
                            try:
                                configured_horizon_value = int(configured_horizon)
                            except (TypeError, ValueError) as exc:
                                raise ValueError(
                                    "parameters.environment.EMBODIEDLAB_MAX_STEPS must be an integer"
                                ) from exc
                            if (
                                isinstance(configured_horizon, bool)
                                or configured_horizon_value != requested_horizon
                            ):
                                raise ValueError(
                                    "parameters.horizon conflicts with "
                                    "parameters.environment.EMBODIEDLAB_MAX_STEPS"
                                )
                        parameter_environment["EMBODIEDLAB_MAX_STEPS"] = str(requested_horizon)
                    environment = {
                        "EMBODIEDLAB_ENV_ID": environment_id,
                        "EMBODIEDLAB_ENV_PROFILE": profile["name"],
                        "EMBODIEDLAB_SEED": str(int(values.get("seed", 0)) + index),
                        **profile.get("environment", {}),
                        **parameter_environment,
                    }
                    # Trusted profiles may pass selected Worker-side secrets to
                    # the container without serializing them into an operation
                    # input or committing them to the catalog.
                    for variable in profile.get("environment_from_worker", []):
                        value = os.getenv(str(variable))
                        if value:
                            environment[str(variable)] = value
                    if values.get("task_id") is not None:
                        environment["EMBODIEDLAB_TASK_ID"] = str(values["task_id"])
                    if capture_dir is not None:
                        environment["EMBODIEDLAB_CAPTURE_DIR"] = CAPTURE_TARGET
                    if record_video:
                        environment["EMBODIEDLAB_RECORD_VIDEO"] = "1"
                    if workflow_dir:
                        environment.update(
                            EMBODIEDLAB_WORKFLOW_DIR="/opt/embodiedlab/workflow",
                            EMBODIEDLAB_WORKFLOW_DIGEST=str(workflow_digest),
                        )
                    daemon_endpoint = None
                    if network == "host":
                        daemon_port = _reserve_loopback_port()
                        daemon_endpoint = f"http://127.0.0.1:{daemon_port}"
                        environment["EMBODIEDLAB_ENVD_URL"] = daemon_endpoint
                        if not command:
                            command = [
                                "--host",
                                "127.0.0.1",
                                "--port",
                                str(daemon_port),
                            ]
                    labels = {
                        "io.embodiedlab.environment-id": environment_id,
                        "io.embodiedlab.project-id": operation["project_id"],
                        "io.embodiedlab.profile": profile["name"],
                    }
                    ownership_labels = values.get("ownership_labels", {})
                    if not isinstance(ownership_labels, dict) or any(
                        key
                        not in {
                            "io.embodiedlab.assignment-id",
                            "io.embodiedlab.execution-id",
                            "io.embodiedlab.lease-id",
                            "io.embodiedlab.worker-id",
                        }
                        or not isinstance(value, str)
                        or not value
                        for key, value in ownership_labels.items()
                    ):
                        raise ValueError("invalid managed OCI ownership labels")
                    labels.update(ownership_labels)
                    if workflow_digest:
                        labels["io.embodiedlab.workflow-digest"] = str(workflow_digest)
                    container_id = self.runtime.start(
                        ContainerSpec(
                            name=name,
                            image=image,
                            command=command,
                            environment={key: str(value) for key, value in environment.items()},
                            labels=labels,
                            mounts=[tuple(item) for item in environment_mounts],
                            network=network,
                            shm_size=profile.get("shm_size"),
                            gpu=values.get("gpu"),
                            egl=profile.get("gpu_mode") == "egl",
                        )
                    )
                    started.append((environment_id, container_id))
                    with self._lock:
                        self._containers[environment_id] = container_id
                        if record_video and capture_dir is not None:
                            self._capture_dirs[environment_id] = capture_dir
                    ready = self._wait_ready(
                        container_id,
                        float(parameters.get("ready_timeout", profile.get("ready_timeout", 60))),
                    )
                    tool_catalog = self._client(container_id, ["tools"]).get("tools", [])
                    metadata = dict(ready.get("metadata") or {})
                    viewer_port = self._viewer_port(metadata)
                    public_metadata = self._public_metadata(metadata)
                    if viewer_port is not None:
                        with self._lock:
                            self._viewer_ports[environment_id] = viewer_port
                        public_metadata["viewer"] = {"state": "ready"}
                    results.append(
                        {
                            "environment_id": environment_id,
                            "container_id": container_id,
                            "daemon_endpoint": daemon_endpoint,
                            "capture_dir": str(capture_dir) if capture_dir else None,
                            "tools": tool_catalog,
                            **public_metadata,
                        }
                    )
            except Exception:
                for environment_id, container_id in reversed(started):
                    try:
                        self.runtime.stop(container_id, timeout=5)
                    finally:
                        with self._lock:
                            self._containers.pop(environment_id, None)
                            self._capture_dirs.pop(environment_id, None)
                            self._viewer_ports.pop(environment_id, None)
                raise
            return {"environments": results}
        environment_id = values.get("environment_id")
        if not environment_id:
            raise ValueError(f"{kind} requires environment_id")
        container_id = self._container_for(environment_id)
        if kind == "env.exec":
            return self._client(
                container_id,
                ["exec", "--filename", values.get("filename", "inline.py")],
                input_text=values["code"],
            )
        if kind == "env.run-workflow":
            arguments = ["run-workflow", "--workflow-dir", "/opt/embodiedlab/workflow"]
            if values.get("task"):
                arguments.extend(["--task", str(values["task"])])
            return self._client(
                container_id,
                arguments,
                input_text=json.dumps(values.get("parameters", {})),
            )
        if kind == "env.observe":
            return self._client(container_id, ["observe"])
        if kind == "env.states":
            return self._client(container_id, ["states"])
        if kind == "env.verify":
            return self._client(container_id, ["verify"])
        if kind == "env.evaluate":
            return self._client(container_id, ["evaluate"])
        if kind == "env.truth":
            if values.get("relation"):
                return self._client(
                    container_id,
                    ["truth"],
                    input_text=json.dumps(
                        {
                            "relation": values["relation"],
                            "obj_a": values["obj_a"],
                            "obj_b": values.get("obj_b"),
                            "threshold": values.get("threshold"),
                        }
                    ),
                )
            return self._client(container_id, ["truth"], input_text="")
        if kind == "env.restore":
            return self._client(container_id, ["restore", values["state_id"]])
        if kind == "env.capture":
            capture = self._client(container_id, ["capture"])
            return self._host_capture_result(environment_id, capture)
        if kind == "env.attach-tools":
            digest = str(values["digest"])
            bundle_dir = (values.get("tool_bundle_dirs") or {}).get(digest)
            if not bundle_dir:
                raise RuntimeError("Worker did not materialize the tool bundle artifact")
            # Repack the materialized directory into the deterministic archive;
            # the digest must round-trip or the Worker cache served other content.
            artifact = package_tool_bundle(bundle_dir)
            if artifact.digest != digest:
                raise RuntimeError(
                    f"tool bundle repack digest mismatch: requested {digest}, "
                    f"packaged {artifact.digest}"
                )
            if len(artifact.content) > MAX_TOOL_BUNDLE_BYTES:
                raise ValueError(
                    f"tool bundle exceeds the {MAX_TOOL_BUNDLE_BYTES} byte attach limit"
                )
            attached = self._client(
                container_id,
                ["tool-attach"],
                input_text=json.dumps(
                    {
                        "digest": digest,
                        "content_b64": base64.b64encode(artifact.content).decode("ascii"),
                    }
                ),
            )
            return {**attached, **self._client(container_id, ["tools"])}
        if kind == "env.detach-tools":
            detached = self._client(
                container_id,
                ["tool-detach"],
                input_text=json.dumps(
                    {"digest": values.get("digest"), "name": values.get("name")}
                ),
            )
            return {**detached, **self._client(container_id, ["tools"])}
        if kind == "env.tool-visibility":
            changed = self._client(
                container_id,
                ["tool-visibility"],
                input_text=json.dumps(
                    {"name": values["name"], "visible": bool(values["visible"])}
                ),
            )
            return {**changed, **self._client(container_id, ["tools"])}
        if kind == "env.stop":
            capture = None
            capture_error = None
            if environment_id in self._capture_dirs:
                try:
                    capture = self._host_capture_result(
                        environment_id,
                        self._client(container_id, ["capture"]),
                    )
                except Exception as exc:
                    capture_error = str(exc)
            # Keep ownership when stop fails, so a later cleanup can retry.
            self.runtime.stop(container_id)
            with self._lock:
                self._containers.pop(environment_id, None)
                self._capture_dirs.pop(environment_id, None)
                self._viewer_ports.pop(environment_id, None)
            return {
                "environment_id": environment_id,
                "stopped": True,
                **({"capture": capture} if capture is not None else {}),
                **(
                    {"capture_error": capture_error}
                    if capture_error is not None
                    else {}
                ),
            }
        if kind.startswith("tool."):
            return self._client(
                container_id,
                ["tool", "--name", kind.removeprefix("tool.")],
                input_text=json.dumps(values.get("arguments", {})),
            )
        raise RuntimeError(f"generic OCI executor requires a profile plugin for {kind}")

    def _host_capture_result(
        self, environment_id: str, value: dict[str, Any]
    ) -> dict[str, Any]:
        """Map container capture paths onto this Worker's host capture dir."""

        capture_dir = self._capture_dirs.get(environment_id)
        if capture_dir is None:
            return value
        result = dict(value)
        result["capture_dir"] = str(capture_dir)
        artifacts = result.get("artifacts")
        if isinstance(artifacts, dict):
            result["artifacts"] = {
                name: str(
                    capture_dir
                    / str(path).removeprefix(CAPTURE_TARGET + "/").lstrip("/")
                )
                for name, path in artifacts.items()
            }
        return result

    def list_environment_ids(self) -> list[str]:
        """Return only running environments this executor can safely recover."""

        candidates = self.runtime.list_containers(label_keys=["io.embodiedlab.environment-id"])
        discovered: set[str] = set()
        for container_id in candidates:
            inspected = self.runtime.inspect(container_id)
            labels = inspected.get("Config", {}).get("Labels") or {}
            environment_id = labels.get("io.embodiedlab.environment-id")
            if not isinstance(environment_id, str) or not environment_id:
                continue
            try:
                self._recover_environment(environment_id)
            except Exception:
                continue
            discovered.add(environment_id)
        return sorted(discovered)

    def _client(
        self,
        container_id: str,
        arguments: list[str],
        *,
        input_text: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        result = self.runtime.exec_command(
            container_id,
            ["embodiedlab-env-client", *arguments],
            input_text=input_text,
            timeout=timeout,
            # embodiedlab-env-client exits 2 with {"ok": false, "error": {...}} on
            # stdout; raising on the exit code here would bury that payload
            # inside an OCI exec failure string.
            check=False,
        )
        try:
            value = json.loads(result["stdout"])
        except json.JSONDecodeError:
            raise ContainerRuntimeError(
                f"embodiedlab-env-client produced no JSON (exit {result['returncode']}): "
                f"stdout={result['stdout'].strip()} stderr={result['stderr'].strip()}"
            ) from None
        if value.get("ok") is False:
            error = value.get("error")
            raise EnvironmentCommandError(
                error if isinstance(error, dict) else {"message": str(error)}
            )
        return value

    def _wait_ready(self, container_id: str, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                value = self._client(container_id, ["ready"], timeout=2)
                if value.get("ready") is True:
                    return value
            except Exception as exc:
                error = exc
                details = self.runtime.inspect(container_id)
                state = details.get("State") if isinstance(details, dict) else None
                if isinstance(state, dict) and state.get("Running") is False:
                    status = state.get("Status", "exited")
                    exit_code = state.get("ExitCode", "unknown")
                    raise ContainerRuntimeError(
                        "environment container exited before readiness: "
                        f"status={status} exit_code={exit_code}"
                    ) from exc
            time.sleep(0.2)
        raise TimeoutError(f"environment daemon did not become ready: {error}")

    def _container_for(self, environment_id: str) -> str:
        with self._lock:
            container_id = self._containers.get(environment_id)
        if container_id is not None:
            return container_id
        container_id, _viewer_port = self._recover_environment(environment_id)
        return container_id

    def _recover_environment(
        self, environment_id: str, *, require_viewer: bool = False
    ) -> tuple[str, int | None]:
        candidates = self.runtime.list_containers(labels={"io.embodiedlab.environment-id": environment_id})
        running: list[str] = []
        for container_id in candidates:
            details = self.runtime.inspect(container_id)
            config = details.get("Config")
            state = details.get("State")
            labels = config.get("Labels") if isinstance(config, dict) else None
            if (
                not isinstance(labels, dict)
                or labels.get("io.embodiedlab.environment-id") != environment_id
            ):
                raise RuntimeError(f"environment {environment_id} has ambiguous OCI ownership")
            if isinstance(state, dict) and state.get("Running") is True:
                running.append(container_id)
        if len(running) != 1:
            raise RuntimeError(f"environment {environment_id} is not owned by this Worker")

        container_id = running[0]
        ready = self._client(container_id, ["ready"], timeout=2)
        if ready.get("ready") is not True:
            raise RuntimeError(f"environment {environment_id} is not ready")
        metadata = ready.get("metadata")
        viewer_port = self._viewer_port(metadata if isinstance(metadata, dict) else {})
        if require_viewer and viewer_port is None:
            raise RuntimeError(f"environment {environment_id} has no valid Viewer")

        with self._lock:
            existing = self._containers.get(environment_id)
            if existing is not None and existing != container_id:
                raise RuntimeError(f"environment {environment_id} has ambiguous OCI ownership")
            self._containers[environment_id] = container_id
            if viewer_port is not None:
                self._viewer_ports[environment_id] = viewer_port
        return container_id, viewer_port

    def cancel(self, operation_id: str, operation: dict[str, Any]) -> None:
        values = operation.get("input", {})
        environment_ids = list(values.get("environment_ids", []))
        if values.get("environment_id"):
            environment_ids.append(values["environment_id"])
        for environment_id in environment_ids:
            container_id = self._containers.get(environment_id)
            if not container_id:
                continue
            try:
                self.runtime.stop(container_id, timeout=5)
            finally:
                with self._lock:
                    self._containers.pop(environment_id, None)
                    self._viewer_ports.pop(environment_id, None)

    @staticmethod
    def _viewer_port(metadata: dict[str, Any]) -> int | None:
        candidates = [metadata.get("viewer"), metadata.get("viser")]
        worker = metadata.get("worker")
        if isinstance(worker, dict):
            candidates.append({"port": worker.get("viser_port")})
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            port = candidate.get("port")
            if isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535:
                return port
        return None

    @staticmethod
    def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Keep local Viser routing data on the Worker; Controller metadata gets capability only."""

        public = dict(metadata)
        public.pop("viewer", None)
        worker = public.get("worker")
        if isinstance(worker, dict):
            sanitized_worker = {key: value for key, value in worker.items() if key != "viser_port"}
            if sanitized_worker:
                public["worker"] = sanitized_worker
            else:
                public.pop("worker", None)
        return public

    def _viewer_port_for(self, environment_id: str) -> int:
        with self._lock:
            port = self._viewer_ports.get(environment_id)
        if port is None:
            try:
                _container_id, port = self._recover_environment(environment_id, require_viewer=True)
            except Exception as exc:
                raise ViewerTransportUnavailable("viewer_not_owned_by_worker") from exc
        if port is None:  # pragma: no cover - require_viewer guarantees this invariant
            raise ViewerTransportUnavailable("viewer_not_owned_by_worker")
        return port

    @staticmethod
    def _viewer_url(port: int, path: str, query: str, *, websocket: bool = False) -> str:
        if any(segment in {".", ".."} for segment in path.split("/")):
            raise ViewerTransportUnavailable("viewer_path_not_found")
        scheme = "ws" if websocket else "http"
        target = f"{scheme}://127.0.0.1:{port}/{quote(path.lstrip('/'), safe='/%:@')}"
        return f"{target}?{query}" if query else target

    async def fetch_viewer_http(
        self, environment_id: str, path: str, query: str, method: str
    ) -> ViewerHTTPResponse:
        if method not in {"GET", "HEAD"}:
            raise ViewerTransportUnavailable("viewer_method_not_allowed")
        url = self._viewer_url(self._viewer_port_for(environment_id), path, query)
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                response = await client.request(method, url, headers={"Accept": "*/*"})
        except httpx.HTTPError as exc:
            raise ViewerTransportUnavailable("viewer_not_available") from exc
        return ViewerHTTPResponse(
            status_code=response.status_code,
            content=response.content,
            media_type=response.headers.get("content-type"),
            headers=sanitize_viewer_response_headers(response.headers),
        )

    async def connect_viewer_websocket(
        self, environment_id: str, path: str, query: str, subprotocols: list[str]
    ) -> _LocalViewerWebSocket:
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:  # pragma: no cover - packaging gate
            raise ViewerTransportUnavailable("viewer_websocket_dependency_missing") from exc
        try:
            url = self._viewer_url(
                self._viewer_port_for(environment_id), path, query, websocket=True
            )
            socket = await connect(
                url,
                subprotocols=subprotocols or None,
                compression=None,
                max_size=16 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=60,
            )
        except Exception as exc:
            raise ViewerTransportUnavailable("viewer_not_available") from exc
        return _LocalViewerWebSocket(socket)


class _LocalViewerWebSocket:
    def __init__(self, socket: Any) -> None:
        self._socket = socket
        self.subprotocol = socket.subprotocol

    async def receive(self) -> str | bytes | None:
        try:
            return await self._socket.recv()
        except Exception:
            return None

    async def send(self, message: str | bytes) -> None:
        await self._socket.send(message)

    async def close(self) -> None:
        await self._socket.close()


_EXECUTOR = GenericOCIExecutor()
executor = _EXECUTOR


def execute(operation: dict[str, Any]) -> dict[str, Any]:
    return _EXECUTOR(operation)


class ModelOCIExecutor:
    """Reconcile long-lived service replicas without importing model packages."""

    def __init__(self) -> None:
        self._runtime: OCIRuntime | None = None
        self._containers: dict[str, list[dict[str, Any]]] = {}
        self._lock = RLock()

    @property
    def runtime(self) -> OCIRuntime:
        if self._runtime is None:
            self._runtime = OCIRuntime()
        return self._runtime

    def _stop_service(self, name: str) -> None:
        container_ids = [instance["container_id"] for instance in self._containers.get(name, [])]
        list_containers = getattr(self.runtime, "list_containers", None)
        if callable(list_containers):
            container_ids.extend(list_containers(labels={"io.embodiedlab.service": name}))
        for container_id in dict.fromkeys(container_ids):
            self.runtime.stop(container_id)
        self._containers.pop(name, None)

    def _wait_ready(
        self,
        container_id: str,
        host: str,
        port: int,
        timeout: float,
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.runtime.inspect(container_id).get("State", {})
            if not state.get("Running", False):
                raise RuntimeError(
                    f"service container exited before ready (exit_code={state.get('ExitCode')})"
                )
            health = state.get("Health")
            if isinstance(health, dict):
                status = health.get("Status")
                if status == "healthy":
                    return
                if status == "unhealthy":
                    raise RuntimeError("service container health check failed")
                time.sleep(0.1)
                continue

            if _port_open(host, port):
                return
            time.sleep(0.1)
        raise TimeoutError(f"service did not listen on {host}:{port} in {timeout}s")

    def __call__(self, operation: dict[str, Any]) -> dict[str, Any]:
        if operation["kind"] != "service.reconcile":
            raise RuntimeError(f"model OCI executor cannot handle {operation['kind']}")
        values = operation["input"]
        spec = service_spec(values["name"])
        name = spec["name"]
        replicas = int(values.get("replicas", 1))
        parameters = {
            **dict(spec.get("parameters") or {}),
            **dict(values.get("parameters") or {}),
        }
        with self._lock:
            self._stop_service(name)
            if replicas == 0:
                return {"instances": []}
            image = values.get("image") or parameters.get("image") or spec.get("default_image")
            if not image:
                raise ValueError(f"service {name} requires an OCI image")
            container_port = int(parameters.get("container_port", spec["default_port"]))
            host_ports = parameters.get("host_ports")
            if host_ports is None:
                first_port = int(parameters.get("host_port", spec["default_port"]))
                host_ports = [first_port + index for index in range(replicas)]
            if len(host_ports) != replicas:
                raise ValueError("parameters.host_ports length must match replicas")
            network = parameters.get("network", spec.get("network"))
            if network == "host" and replicas > 1:
                raise ValueError(
                    "host network cannot serve multiple replicas: "
                    "every replica would bind the same host port"
                )
            # Docker gives a container only 64 MB of /dev/shm; services that share memory
            # across processes need more, set per service.
            shm_size = parameters.get("shm_size", spec.get("shm_size"))
            advertise_host = parameters.get("advertise_host", "127.0.0.1")
            ready_host = parameters.get("ready_host", "127.0.0.1")
            ready_timeout = float(parameters.get("ready_timeout", 120))
            publish_host = str(parameters.get("publish_host", "127.0.0.1"))
            command = list(parameters.get("command", spec.get("command", [])))
            environment = {
                **dict(spec.get("environment") or {}),
                **dict(parameters.get("environment") or {}),
            }
            mounts: list[tuple[str, str, str]] = []
            for mount in [
                *list(spec.get("mounts") or []),
                *list(parameters.get("mounts") or []),
            ]:
                source = mount.get("source")
                source_env = mount.get("source_env")
                if source_env:
                    source = os.getenv(str(source_env))
                    if not source:
                        raise ValueError(
                            f"model service {name} requires Worker setting {source_env}"
                        )
                if not source:
                    raise ValueError(f"model service {name} mount requires source or source_env")
                mounts.append(
                    (str(source), str(mount["target"]), str(mount.get("mode", "readonly")))
                )
            instances: list[dict[str, Any]] = []
            try:
                for index, host_port in enumerate(host_ports):
                    instance_id = f"svc_{name}_{index}"
                    container_id = self.runtime.start(
                        ContainerSpec(
                            name=f"embodiedlab-{instance_id}",
                            image=image,
                            command=command,
                            environment={key: str(value) for key, value in environment.items()},
                            labels={
                                "io.embodiedlab.service": name,
                                "io.embodiedlab.service-instance": instance_id,
                                "io.embodiedlab.project-id": operation["project_id"],
                            },
                            mounts=mounts,
                            # With host networking the container binds host ports directly from its command;
                            # docker would drop a -p mapping, which would only make the setting look effective.
                            ports=(
                                []
                                if network == "host"
                                else [(int(host_port), container_port)]
                            ),
                            publish_host=publish_host,
                            network=network,
                            shm_size=shm_size,
                            gpu=values.get("gpu"),
                        )
                    )
                    scheme = spec["protocol"].replace("openai-http", "http")
                    instance = {
                        "id": instance_id,
                        "container_id": container_id,
                        "endpoint": f"{scheme}://{advertise_host}:{host_port}",
                        "protocol": spec["protocol"],
                        "port": int(host_port),
                    }
                    instances.append(instance)
                    self._wait_ready(
                        container_id,
                        str(ready_host),
                        int(host_port),
                        ready_timeout,
                    )
            except Exception:
                for instance in instances:
                    self.runtime.stop(instance["container_id"])
                raise
            self._containers[name] = instances
            return {"instances": instances}

    def cancel(self, operation_id: str, operation: dict[str, Any]) -> None:
        name = service_spec(operation["input"]["name"])["name"]
        with self._lock:
            self._stop_service(name)


service_executor = ModelOCIExecutor()

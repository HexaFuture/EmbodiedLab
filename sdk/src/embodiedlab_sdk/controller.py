"""FastAPI Controller: the only stateful control-plane API."""

from __future__ import annotations

import os

import asyncio
import hashlib
import io
import tempfile
from datetime import timedelta
from typing import Any

from fastapi import (
    Body,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import __version__
from .catalog import capabilities, load_catalog, service_spec
from .external_services import ExternalServiceRegistry
from .models import (
    TERMINAL_OPERATION_STATUSES,
    Batch,
    BatchCreate,
    CommandCreate,
    Environment,
    EnvironmentCreate,
    EvaluationCreate,
    Heartbeat,
    Operation,
    OperationCreate,
    OperationPatch,
    OperationProgressEvent,
    OperationProgressEventCreate,
    OperationStatus,
    ServiceAction,
    ToolBundleAttach,
    ToolBundleDetach,
    ToolCall,
    ToolVisibility,
    Worker,
    WorkerRegister,
    WorkerRole,
    resource_id,
    utcnow,
)
from .object_store import LocalObjectStore, ObjectStore
from .security import APIIdentity
from .store import InMemoryStore, OperationEventConflict, Store
from .viewer import (
    ViewerTransport,
    ViewerTransportUnavailable,
    ViewerWebSocket,
    WorkerViewerTunnelRegistry,
    sanitize_viewer_response_headers,
)


def _not_found(kind: str, item_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{kind} {item_id} was not found")


def _worker_with_liveness(worker: Worker, stale_after_seconds: float) -> Worker:
    """Project missed heartbeats to offline without mutating stored Worker data."""

    if worker.state == "offline":
        return worker
    if utcnow() - worker.last_heartbeat_at <= timedelta(seconds=stale_after_seconds):
        return worker
    return worker.model_copy(update={"state": "offline"})


# envd is the sole authority for the tool registry revision; the Controller
# only mirrors each completed mutation into environment metadata for reads.
TOOL_REGISTRY_OPERATION_KINDS = {
    "env.attach-tools",
    "env.detach-tools",
    "env.tool-visibility",
}
TOOL_REGISTRY_HISTORY_LIMIT = 50


async def _relay_viewer_websocket(websocket: WebSocket, upstream: ViewerWebSocket) -> None:
    async def from_browser() -> None:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            text = message.get("text")
            binary = message.get("bytes")
            if isinstance(text, str):
                await upstream.send(text)
            elif isinstance(binary, bytes):
                await upstream.send(binary)

    async def from_upstream() -> None:
        while True:
            message = await upstream.receive()
            if message is None:
                return
            if isinstance(message, str):
                await websocket.send_text(message)
            else:
                await websocket.send_bytes(message)

    tasks = [asyncio.create_task(from_browser()), asyncio.create_task(from_upstream())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except WebSocketDisconnect:
        pass
    finally:
        for task in tasks:
            task.cancel()
        await upstream.close()
        if websocket.client_state.name == "CONNECTED":
            await websocket.close(code=1000)


def _new_operation(store: Store, value: OperationCreate) -> Operation:
    operation = Operation(
        id=resource_id("op"),
        project_id=value.project_id,
        kind=value.kind,
        requested_by=value.requested_by,
        target_worker_id=value.target_worker_id,
        input=value.input,
    )
    return store.put_operation(operation)


def _required_role(kind: str) -> WorkerRole | None:
    if kind.startswith(("env.", "tool.", "evaluate.")):
        return WorkerRole.SIMULATION
    if kind.startswith("service."):
        return WorkerRole.MODEL
    return None


def _transition(operation: Operation, patch: OperationPatch) -> Operation:
    if operation.status in TERMINAL_OPERATION_STATUSES:
        raise HTTPException(status_code=409, detail="terminal operation cannot transition")
    allowed = {
        OperationStatus.PENDING: {
            OperationStatus.SCHEDULED,
            OperationStatus.RUNNING,
            OperationStatus.CANCELLED,
            OperationStatus.FAILED,
        },
        OperationStatus.SCHEDULED: {
            OperationStatus.RUNNING,
            OperationStatus.CANCELLED,
            OperationStatus.FAILED,
            OperationStatus.INTERRUPTED,
        },
        OperationStatus.RUNNING: TERMINAL_OPERATION_STATUSES,
    }
    if patch.status not in allowed.get(operation.status, set()):
        raise HTTPException(
            status_code=409,
            detail=f"invalid operation transition {operation.status.value}->{patch.status.value}",
        )
    now = utcnow()
    update: dict[str, Any] = {
        "status": patch.status,
        "worker_id": patch.worker_id or operation.worker_id,
        "updated_at": now,
        "output": patch.output,
        "error": patch.error,
    }
    if patch.status == OperationStatus.RUNNING and operation.started_at is None:
        update["started_at"] = now
    if patch.status in TERMINAL_OPERATION_STATUSES:
        update["completed_at"] = now
    # Worker completion describes execution only. It can never set task success.
    update["success"] = None
    update["success_source"] = None
    return operation.model_copy(update=update)


def create_app(
    store: Store | None = None,
    object_store: ObjectStore | None = None,
    api_keys: dict[str, APIIdentity] | None = None,
    external_services: list[dict[str, Any]] | None = None,
    managed_services_enabled: bool = True,
    viewer_transport: ViewerTransport | None = None,
    worker_stale_after_seconds: float = 30.0,
) -> FastAPI:
    if worker_stale_after_seconds <= 0:
        raise ValueError("worker_stale_after_seconds must be positive")
    state = store or InMemoryStore()
    artifacts = object_store or LocalObjectStore(tempfile.mkdtemp(prefix="embodiedlab-artifacts-"))
    app = FastAPI(title="EmbodiedLab Controller", version=__version__)
    app.state.store = state
    configured_keys = api_keys or {}
    external_registry = ExternalServiceRegistry(state, external_services or [])
    external_registry.refresh()
    app.state.viewer_tunnel_registry = WorkerViewerTunnelRegistry()
    app.state.viewer_transport = viewer_transport or app.state.viewer_tunnel_registry

    def observed_worker(worker_id: str) -> Worker | None:
        worker = state.get_worker(worker_id)
        if worker is None:
            return None
        return _worker_with_liveness(worker, worker_stale_after_seconds)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if not configured_keys or request.url.path in {"/ready", "/version"}:
            return await call_next(request)
        authorization = request.headers.get("authorization", "")
        token = authorization.removeprefix("Bearer ").strip()
        identity = configured_keys.get(token)
        if identity is None:
            return JSONResponse(status_code=401, content={"detail": "valid EmbodiedLab API key required"})
        headers = [
            (key, value)
            for key, value in request.scope["headers"]
            if key not in {b"x-embodiedlab-role", b"x-embodiedlab-project"}
        ]
        headers.extend(
            [
                (b"x-embodiedlab-role", identity.role.encode()),
                (b"x-embodiedlab-project", identity.project_id.encode()),
                (b"x-embodiedlab-subject", identity.subject.encode()),
            ]
        )
        if identity.worker_registration_key:
            headers.append(
                (
                    b"x-embodiedlab-worker-registration-key",
                    identity.worker_registration_key.encode(),
                )
            )
        request.scope["headers"] = headers
        return await call_next(request)

    def require_operation_access(operation: Operation, request: Request) -> None:
        if not configured_keys:
            return
        role = request.headers.get("x-embodiedlab-role", "")
        project_id = request.headers.get("x-embodiedlab-project", "")
        if role != "operator" and project_id != operation.project_id:
            raise HTTPException(status_code=403, detail="operation project access denied")

    def require_worker_event_identity(
        operation: Operation,
        event: OperationProgressEventCreate,
        request: Request,
    ) -> Worker:
        if operation.status != OperationStatus.RUNNING:
            raise HTTPException(status_code=409, detail="operation is not running")
        if operation.worker_id != event.worker_id:
            raise HTTPException(status_code=403, detail="operation worker ownership mismatch")
        worker = state.get_worker(event.worker_id)
        if worker is None:
            raise _not_found("worker", event.worker_id)
        if configured_keys:
            if request.headers.get("x-embodiedlab-role") != "worker":
                raise HTTPException(status_code=403, detail="worker identity required")
            registration_key = request.headers.get("x-embodiedlab-worker-registration-key")
            if not registration_key or registration_key != worker.registration_key:
                raise HTTPException(status_code=403, detail="worker registration identity mismatch")
            if request.headers.get("x-embodiedlab-project") != operation.project_id:
                raise HTTPException(status_code=403, detail="worker project mismatch")
        if not worker.agent_generation or worker.agent_generation != event.agent_generation:
            raise HTTPException(status_code=409, detail="worker generation conflict")
        return worker

    def reconcile_resources(operation: Operation) -> None:
        if operation.kind == "env.start":
            environment_ids = operation.input.get("environment_ids", [])
            output_by_id = {
                item.get("environment_id"): item
                for item in operation.output.get("environments", [])
            }
            for environment_id in environment_ids:
                environment = state.get_resource("environment", environment_id)
                if environment is None:
                    continue
                if operation.status == OperationStatus.COMPLETED:
                    environment.update(
                        state="ready",
                        worker_id=operation.worker_id,
                        metadata={
                            **environment.get("metadata", {}),
                            "gpu": operation.input.get("gpu"),
                            **output_by_id.get(environment_id, {}),
                        },
                    )
                elif operation.status in {
                    OperationStatus.FAILED,
                    OperationStatus.INTERRUPTED,
                    OperationStatus.CANCELLED,
                }:
                    environment.update(state="failed")
                state.put_resource("environment", environment_id, environment)
        elif operation.kind == "env.stop" and operation.status == OperationStatus.COMPLETED:
            environment_id = operation.input.get("environment_id")
            environment = state.get_resource("environment", environment_id)
            if environment:
                environment.update(
                    state="stopped",
                    updated_at=utcnow().isoformat(),
                    metadata={
                        **(environment.get("metadata") or {}),
                        **(
                            {"capture": operation.output.get("capture")}
                            if (operation.output or {}).get("capture")
                            else {}
                        ),
                    },
                )
                state.put_resource("environment", environment_id, environment)
        elif operation.kind == "env.capture" and operation.status == OperationStatus.COMPLETED:
            environment_id = operation.input.get("environment_id")
            environment = state.get_resource("environment", environment_id)
            if environment:
                environment.update(
                    metadata={
                        **(environment.get("metadata") or {}),
                        "capture": operation.output,
                    },
                    updated_at=utcnow().isoformat(),
                )
                state.put_resource("environment", environment_id, environment)
        elif (
            operation.kind in TOOL_REGISTRY_OPERATION_KINDS
            and operation.status == OperationStatus.COMPLETED
        ):
            environment_id = operation.input.get("environment_id")
            environment = state.get_resource("environment", environment_id)
            if environment is None:
                return
            metadata = dict(environment.get("metadata") or {})
            output = operation.output or {}
            if isinstance(output.get("tools"), list):
                metadata["tools"] = output["tools"]
            if output.get("registry_revision") is not None:
                metadata["registry_revision"] = output["registry_revision"]
            if isinstance(output.get("disabled"), list):
                metadata["disabled_tools"] = output["disabled"]
            if isinstance(output.get("bundles"), list):
                metadata["tool_bundles"] = output["bundles"]
            entry: dict[str, Any] = {
                "operation_id": operation.id,
                "kind": operation.kind,
                "revision": output.get("registry_revision"),
                "recorded_at": utcnow().isoformat(),
            }
            for field in ("digest", "name"):
                if operation.input.get(field):
                    entry[field] = operation.input[field]
            history = list(metadata.get("tool_registry_history") or [])
            history.append(entry)
            metadata["tool_registry_history"] = history[-TOOL_REGISTRY_HISTORY_LIMIT:]
            environment.update(metadata=metadata, updated_at=utcnow().isoformat())
            state.put_resource("environment", environment_id, environment)
        elif (
            operation.kind == "service.reconcile" and operation.status == OperationStatus.COMPLETED
        ):
            name = operation.input["name"]
            state.put_resource(
                "service",
                name,
                {
                    "id": name,
                    "name": name,
                    "project_id": operation.project_id,
                    "replicas": operation.input.get("replicas", 1),
                    "instances": operation.output.get("instances", []),
                    "state": "running" if operation.input.get("replicas", 1) else "stopped",
                    "operation_id": operation.id,
                    "updated_at": utcnow().isoformat(),
                },
            )

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        return {"ready": True, "version": __version__}

    @app.get("/version")
    def version() -> dict[str, str]:
        return {"name": "embodiedlab-sdk", "version": __version__, "api_version": "v1"}

    @app.get("/capabilities")
    def get_capabilities() -> dict[str, Any]:
        return capabilities()

    @app.post("/v1/workers/register", response_model=Worker)
    def register_worker(value: WorkerRegister) -> Worker:
        worker = Worker(id=resource_id("worker"), **value.model_dump())
        try:
            return state.register_worker(worker)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/workers/{worker_id}/heartbeat", response_model=Worker)
    def heartbeat(worker_id: str, value: Heartbeat) -> Worker:
        worker = state.get_worker(worker_id)
        if worker is None:
            raise _not_found("worker", worker_id)
        updated = worker.model_copy(
            update={
                "state": value.state,
                "capacity": value.capacity or worker.capacity,
                "agent_generation": value.agent_generation or worker.agent_generation,
                "last_heartbeat_at": utcnow(),
                "updated_at": utcnow(),
            }
        )
        return state.put_worker(updated)

    @app.get("/v1/workers", response_model=list[Worker])
    def list_workers(role: WorkerRole | None = None) -> list[Worker]:
        values = [
            _worker_with_liveness(item, worker_stale_after_seconds)
            for item in state.list_workers()
        ]
        return [item for item in values if role is None or item.role == role]

    @app.post("/v1/workers/{worker_id}/assignments/next", response_model=Operation | None)
    def next_assignment(worker_id: str) -> Operation | None:
        worker = observed_worker(worker_id)
        if worker is None:
            raise _not_found("worker", worker_id)
        if worker.state != "ready":
            return None
        return state.claim_operation(worker)

    @app.post("/v1/operations", response_model=Operation)
    def create_operation(value: OperationCreate) -> Operation:
        return _new_operation(state, value)

    @app.get("/v1/operations", response_model=list[Operation])
    def list_operations(
        project_id: str | None = None,
        include_output: bool = True,
    ) -> list[Operation]:
        """List Operations, optionally omitting their potentially huge outputs.

        Observation and robot-tool Operations can contain RGB-D images, masks,
        and other multi-megabyte values.  Polling clients only need lifecycle
        metadata, so they can request ``include_output=false`` without forcing
        the Controller to serialize and transfer every retained payload.
        """

        operations = state.list_operations(project_id, include_output=include_output)
        if include_output:
            return operations
        return [
            operation.model_copy(update={"output": {}, "output_included": False})
            for operation in operations
        ]

    @app.get("/v1/operations/{operation_id}", response_model=Operation)
    def get_operation(operation_id: str) -> Operation:
        operation = state.get_operation(operation_id)
        if operation is None:
            raise _not_found("operation", operation_id)
        return operation

    @app.post("/v1/operations/{operation_id}/transition", response_model=Operation)
    def transition_operation(operation_id: str, patch: OperationPatch) -> Operation:
        operation = state.get_operation(operation_id)
        if operation is None:
            raise _not_found("operation", operation_id)
        updated = _transition(operation, patch)
        state.append_log(operation_id, "system", f"status={updated.status.value}")
        saved = state.put_operation(updated)
        reconcile_resources(saved)
        return saved

    @app.post("/v1/operations/{operation_id}/cancel", response_model=Operation)
    def cancel_operation(operation_id: str) -> Operation:
        operation = state.get_operation(operation_id)
        if operation is None:
            raise _not_found("operation", operation_id)
        if operation.status in TERMINAL_OPERATION_STATUSES:
            return operation
        updated = _transition(operation, OperationPatch(status=OperationStatus.CANCELLED))
        state.append_log(operation_id, "system", "cancelled by client")
        return state.put_operation(updated)

    @app.post("/v1/operations/{operation_id}/evaluation", response_model=Operation)
    def evaluate_operation(operation_id: str, value: EvaluationCreate) -> Operation:
        operation = state.get_operation(operation_id)
        if operation is None:
            raise _not_found("operation", operation_id)
        if operation.status != OperationStatus.COMPLETED:
            raise HTTPException(status_code=409, detail="only completed execution can be evaluated")
        updated = operation.model_copy(
            update={
                "success": value.success,
                "success_source": value.source,
                "output": {**operation.output, "evaluation": value.details},
                "updated_at": utcnow(),
            }
        )
        state.append_log(
            operation_id, "evaluation", f"success={value.success} source={value.source}"
        )
        return state.put_operation(updated)

    @app.get("/v1/operations/{operation_id}/logs")
    def operation_logs(
        operation_id: str,
        request: Request,
        after: int = Query(default=0, ge=0),
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> dict[str, Any]:
        operation = state.get_operation(operation_id)
        if operation is None:
            raise _not_found("operation", operation_id)
        require_operation_access(operation, request)
        return {
            "operation_id": operation_id,
            "entries": state.list_logs(operation_id, after)[:limit],
        }

    @app.get("/v1/operations/{operation_id}/events")
    def operation_events(
        operation_id: str,
        request: Request,
        after: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        operation = state.get_operation(operation_id)
        if operation is None:
            raise _not_found("operation", operation_id)
        require_operation_access(operation, request)
        return {
            "operation_id": operation_id,
            "entries": state.list_operation_events(operation_id, after, limit),
        }

    @app.post(
        "/internal/v1/operations/{operation_id}/events",
        response_model=OperationProgressEvent,
        status_code=201,
    )
    def append_operation_event(
        operation_id: str,
        value: OperationProgressEventCreate,
        request: Request,
        response: Response,
    ) -> OperationProgressEvent:
        operation = state.get_operation(operation_id)
        if operation is None:
            raise _not_found("operation", operation_id)
        require_worker_event_identity(operation, value, request)
        event = OperationProgressEvent(
            **value.model_dump(),
            operation_id=operation.id,
            project_id=operation.project_id,
        )
        try:
            persisted, created = state.append_operation_event(event)
        except OperationEventConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not created:
            response.status_code = 200
        return persisted

    def registry_parameters(task_id, seed, parameters):
        """Registry experiments started by seed get the dev instance of that seed (operator convenience).

        Only with EMBODIEDLAB_REGISTRY_ROOT set. An explicit EMBODIEDLAB_INSTANCE_ID (e.g. a test-split
        instance issued by the keyed scheduler) is kept as given; the dev split is derived from the public
        namespace and never needs a key. The record goes to EMBODIEDLAB_INSTANCE_STORE, mounted read-only.
        """
        root = os.environ.get("EMBODIEDLAB_REGISTRY_ROOT")
        if not root or not task_id:
            return parameters
        import sys
        if root not in sys.path:
            sys.path.insert(0, root)
        from benchmark.instances.service import InstanceRejected, derive, materialize
        from benchmark.registry.loader import EXPERIMENTS, load
        # Only the requested spec is loaded: a broken or unfinished spec of another experiment must not
        # block every environment start.
        if not (EXPERIMENTS / task_id / "experiment.yaml").is_file():
            return parameters
        environment = dict(parameters.get("environment") or {})
        if environment.get("EMBODIEDLAB_INSTANCE_ID"):
            return parameters
        try:
            instance = derive(task_id, "dev", seed)
        except InstanceRejected as exc:
            raise HTTPException(status_code=422, detail=(
                f"{task_id} dev seed {seed} is rejected by the experiment constraints ({exc.failed}); "
                "use another seed")) from exc
        materialize(instance, os.environ["EMBODIEDLAB_INSTANCE_STORE"])
        environment["EMBODIEDLAB_INSTANCE_ID"] = instance.instance_id
        result = {**parameters, "environment": environment}
        result.setdefault("horizon", int(load(task_id).budget["sim_s"] * 30))
        return result

    @app.post("/v1/environments")
    def create_environments(value: EnvironmentCreate) -> dict[str, Any]:
        if value.workflow_digest:
            if not value.workflow_digest.startswith("sha256:") or len(value.workflow_digest) != 71:
                raise HTTPException(
                    status_code=422,
                    detail="workflow digest must use sha256:<hex>",
                )
            workflow_key = (
                f"projects/{value.project_id}/workflows/{value.workflow_digest[7:]}.tar.gz"
            )
            if not artifacts.exists(workflow_key):
                raise HTTPException(status_code=404, detail="workflow was not uploaded")
        catalog_names = {item["name"] for item in load_catalog("environment_profiles")}
        if value.profile not in catalog_names:
            raise HTTPException(
                status_code=422, detail=f"unknown environment profile {value.profile}"
            )
        if value.target_worker_id:
            target_worker = observed_worker(value.target_worker_id)
            if target_worker is None:
                raise _not_found("worker", value.target_worker_id)
            if target_worker.role != WorkerRole.SIMULATION:
                raise HTTPException(
                    status_code=422,
                    detail="target_worker_id must identify a simulation Worker",
                )
            if target_worker.state != "ready":
                raise HTTPException(
                    status_code=409,
                    detail="target_worker_id is not ready",
                )
        pool_id = resource_id("pool")
        environments: list[Environment] = []
        operations: list[Operation] = []
        # Validate every seed before creating any operation.
        instance_parameters = [registry_parameters(value.task_id, value.seed + index, value.parameters)
                               for index in range(value.count)]
        for index in range(value.count):
            environment_id = resource_id("env")
            operation = _new_operation(
                state,
                OperationCreate(
                    kind="env.start",
                    project_id=value.project_id,
                    target_worker_id=value.target_worker_id,
                    input={
                        **value.model_dump(),
                        "parameters": instance_parameters[index],
                        "count": 1,
                        "seed": value.seed + index,
                        "pool_id": pool_id,
                        "environment_ids": [environment_id],
                    },
                ),
            )
            environment = Environment(
                id=environment_id,
                project_id=value.project_id,
                profile=value.profile,
                state="pending",
                seed=value.seed + index,
                task_id=value.task_id,
                operation_id=operation.id,
                metadata={
                    "pool_id": pool_id,
                    "image": value.image,
                    "gpu": value.gpu,
                    "correlation_id": value.correlation_id,
                },
            )
            state.put_resource("environment", environment.id, environment.model_dump(mode="json"))
            environments.append(environment)
            operations.append(operation)
        return {
            "pool_id": pool_id,
            "environments": environments,
            # Keep the legacy key for count=1 and older clients.
            "operation": operations[0],
            "operations": operations,
        }

    @app.get("/v1/environments")
    def list_environments(
        project_id: str | None = None, correlation_id: str | None = None
    ) -> list[dict[str, Any]]:
        environments = state.list_resources("environment", project_id)
        if correlation_id is None:
            return environments
        return [
            environment
            for environment in environments
            if environment.get("metadata", {}).get("correlation_id") == correlation_id
        ]

    @app.get("/v1/environments/{environment_id}")
    def get_environment(environment_id: str) -> dict[str, Any]:
        value = state.get_resource("environment", environment_id)
        if value is None:
            raise _not_found("environment", environment_id)
        return value

    @app.websocket("/internal/v1/workers/{worker_id}/viewer-tunnel")
    async def register_worker_viewer_tunnel(websocket: WebSocket, worker_id: str) -> None:
        """Accept a private channel opened by a pre-registered simulation Worker only."""

        if not configured_keys:
            await websocket.close(code=4404, reason="viewer_tunnel_not_configured")
            return
        authorization = websocket.headers.get("authorization", "")
        identity = configured_keys.get(authorization.removeprefix("Bearer ").strip())
        worker = observed_worker(worker_id)
        if (
            identity is None
            or identity.role != "worker"
            or not identity.worker_registration_key
            or worker is None
            or worker.role != WorkerRole.SIMULATION
            or worker.state != "ready"
            or worker.registration_key != identity.worker_registration_key
        ):
            await websocket.close(code=4403, reason="viewer_tunnel_identity_required")
            return
        await websocket.accept()
        registry: WorkerViewerTunnelRegistry = app.state.viewer_tunnel_registry
        connection = await registry.register(worker_id, websocket)
        try:
            while True:
                payload = await websocket.receive_json()
                if isinstance(payload, dict):
                    connection.receive(payload)
        except WebSocketDisconnect:
            pass
        finally:
            await registry.unregister(connection)

    @app.api_route(
        "/internal/v1/environments/{environment_id}/viewer/{path:path}",
        methods=["GET", "HEAD"],
        include_in_schema=False,
    )
    async def proxy_environment_viewer(
        environment_id: str,
        path: str,
        request: Request,
        x_embodiedlab_role: str | None = Header(default=None),
        x_embodiedlab_project: str | None = Header(default=None),
    ) -> Response:
        # The auth middleware injects these headers only when an API key matches.
        # Do not permit this endpoint in controller instances without configured keys.
        if not configured_keys:
            raise HTTPException(status_code=404, detail="viewer_gateway_not_configured")
        if x_embodiedlab_role != "gateway" or not x_embodiedlab_project:
            raise HTTPException(status_code=403, detail="viewer_gateway_identity_required")
        if any(segment in {".", ".."} for segment in path.split("/")):
            raise HTTPException(status_code=404, detail="viewer_path_not_found")
        environment_value = state.get_resource("environment", environment_id)
        if environment_value is None or environment_value.get("project_id") != x_embodiedlab_project:
            raise _not_found("environment", environment_id)
        environment = Environment.model_validate(environment_value)
        viewer_metadata = environment.metadata.get("viewer")
        worker = observed_worker(environment.worker_id) if environment.worker_id else None
        if (
            environment.state != "ready"
            or worker is None
            or worker.state != "ready"
            or not isinstance(viewer_metadata, dict)
            or viewer_metadata.get("state") != "ready"
        ):
            raise HTTPException(status_code=409, detail="viewer_not_available")
        transport = app.state.viewer_transport
        if transport is None:
            raise HTTPException(status_code=503, detail="viewer_transport_not_configured")
        try:
            upstream = await transport.fetch_http(
                environment,
                path,
                request.url.query,
                request.method,
            )
        except ViewerTransportUnavailable as exc:
            raise HTTPException(status_code=503, detail="viewer_not_available") from exc
        return Response(
            content=b"" if request.method == "HEAD" else upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.media_type,
            headers=sanitize_viewer_response_headers(upstream.headers),
        )

    @app.websocket("/internal/v1/environments/{environment_id}/viewer/{path:path}")
    async def proxy_environment_viewer_websocket(
        websocket: WebSocket, environment_id: str, path: str
    ) -> None:
        if not configured_keys:
            await websocket.close(code=4404, reason="viewer_gateway_not_configured")
            return
        authorization = websocket.headers.get("authorization", "")
        identity = configured_keys.get(authorization.removeprefix("Bearer ").strip())
        if identity is None or identity.role != "gateway":
            await websocket.close(code=4403, reason="viewer_gateway_identity_required")
            return
        if any(segment in {".", ".."} for segment in path.split("/")):
            await websocket.close(code=4404, reason="viewer_path_not_found")
            return
        environment_value = state.get_resource("environment", environment_id)
        if environment_value is None or environment_value.get("project_id") != identity.project_id:
            await websocket.close(code=4404, reason="environment_not_found")
            return
        environment = Environment.model_validate(environment_value)
        viewer_metadata = environment.metadata.get("viewer")
        worker = observed_worker(environment.worker_id) if environment.worker_id else None
        if (
            environment.state != "ready"
            or worker is None
            or worker.state != "ready"
            or not isinstance(viewer_metadata, dict)
            or viewer_metadata.get("state") != "ready"
        ):
            await websocket.close(code=4409, reason="viewer_not_available")
            return
        transport = app.state.viewer_transport
        if transport is None:
            await websocket.close(code=1013, reason="viewer_transport_not_configured")
            return
        try:
            upstream = await transport.connect_websocket(
                environment,
                path,
                websocket.url.query,
                list(websocket.scope.get("subprotocols", [])),
            )
        except ViewerTransportUnavailable:
            await websocket.close(code=1013, reason="viewer_not_available")
            return
        await websocket.accept(subprotocol=getattr(upstream, "subprotocol", None))
        await _relay_viewer_websocket(websocket, upstream)

    def environment_command(environment_id: str, command: str, value: CommandCreate) -> Operation:
        environment = state.get_resource("environment", environment_id)
        if environment is None:
            raise _not_found("environment", environment_id)
        if command != "stop" and environment.get("state") != "ready":
            raise HTTPException(status_code=409, detail="environment is not ready")
        worker_id = environment.get("worker_id")
        if not worker_id:
            raise HTTPException(status_code=409, detail="environment has no owning Worker")
        return _new_operation(
            state,
            OperationCreate(
                kind=f"env.{command}",
                project_id=value.project_id,
                target_worker_id=worker_id,
                input={"environment_id": environment_id, **value.arguments},
            ),
        )

    @app.post("/v1/environments/{environment_id}/exec", response_model=Operation)
    def exec_environment(environment_id: str, value: CommandCreate) -> Operation:
        return environment_command(environment_id, "exec", value)

    @app.post("/v1/environments/{environment_id}/run-workflow", response_model=Operation)
    def run_workflow_environment(environment_id: str, value: CommandCreate) -> Operation:
        return environment_command(environment_id, "run-workflow", value)

    @app.post("/v1/environments/{environment_id}/observe", response_model=Operation)
    def observe_environment(environment_id: str, value: CommandCreate) -> Operation:
        return environment_command(environment_id, "observe", value)

    @app.post("/v1/environments/{environment_id}/states", response_model=Operation)
    def states_environment(environment_id: str, value: CommandCreate) -> Operation:
        return environment_command(environment_id, "states", value)

    @app.post("/v1/environments/{environment_id}/verify", response_model=Operation)
    def verify_environment(environment_id: str, value: CommandCreate) -> Operation:
        return environment_command(environment_id, "verify", value)

    @app.post("/v1/environments/{environment_id}/evaluate", response_model=Operation)
    def evaluate_environment(environment_id: str, value: CommandCreate) -> Operation:
        return environment_command(environment_id, "evaluate", value)

    @app.post("/v1/environments/{environment_id}/truth", response_model=Operation)
    def truth_environment(environment_id: str, value: CommandCreate) -> Operation:
        return environment_command(environment_id, "truth", value)

    @app.post("/v1/environments/{environment_id}/capture", response_model=Operation)
    def capture_environment(environment_id: str, value: CommandCreate) -> Operation:
        return environment_command(environment_id, "capture", value)

    @app.post("/v1/environments/{environment_id}/restore", response_model=Operation)
    def restore_environment(environment_id: str, value: CommandCreate) -> Operation:
        return environment_command(environment_id, "restore", value)

    @app.post("/v1/environments/{environment_id}/stop", response_model=Operation)
    def stop_environment(environment_id: str, value: CommandCreate) -> Operation:
        return environment_command(environment_id, "stop", value)

    @app.post("/v1/environments/{environment_id}/tools/attach", response_model=Operation)
    def attach_environment_tools(environment_id: str, value: ToolBundleAttach) -> Operation:
        if not value.digest.startswith("sha256:") or len(value.digest) != 71:
            raise HTTPException(
                status_code=422, detail="tool bundle digest must use sha256:<hex>"
            )
        key = f"projects/{value.project_id}/tool-bundles/{value.digest[7:]}.tar.gz"
        if not artifacts.exists(key):
            raise _not_found("tool bundle", value.digest)
        # tool_bundle_digests triggers the Worker-side materialization (M1);
        # the executor repacks that directory and verifies the digest.
        return environment_command(
            environment_id,
            "attach-tools",
            CommandCreate(
                project_id=value.project_id,
                arguments={"digest": value.digest, "tool_bundle_digests": [value.digest]},
            ),
        )

    @app.post("/v1/environments/{environment_id}/tools/detach", response_model=Operation)
    def detach_environment_tools(environment_id: str, value: ToolBundleDetach) -> Operation:
        arguments = {
            key: item for key, item in (("digest", value.digest), ("name", value.name)) if item
        }
        return environment_command(
            environment_id,
            "detach-tools",
            CommandCreate(project_id=value.project_id, arguments=arguments),
        )

    @app.post(
        "/v1/environments/{environment_id}/tools/visibility", response_model=Operation
    )
    def set_environment_tool_visibility(
        environment_id: str, value: ToolVisibility
    ) -> Operation:
        return environment_command(
            environment_id,
            "tool-visibility",
            CommandCreate(
                project_id=value.project_id,
                arguments={"name": value.name, "visible": value.visible},
            ),
        )

    @app.get("/v1/tools")
    def list_tools(environment_id: str | None = None) -> list[dict[str, Any]]:
        if environment_id:
            environment = state.get_resource("environment", environment_id)
            if environment is None:
                raise _not_found("environment", environment_id)
            tools = environment.get("metadata", {}).get("tools")
            if isinstance(tools, list):
                return tools
        return load_catalog("robot_tools")

    @app.post("/v1/tools/call", response_model=Operation)
    def call_tool(value: ToolCall) -> Operation:
        environment = state.get_resource("environment", value.environment_id)
        if environment is None:
            raise _not_found("environment", value.environment_id)
        if environment.get("state") != "ready" or not environment.get("worker_id"):
            raise HTTPException(status_code=409, detail="environment is not ready")
        return _new_operation(
            state,
            OperationCreate(
                kind=f"tool.{value.name}",
                project_id=value.project_id,
                target_worker_id=environment["worker_id"],
                input=value.model_dump(),
            ),
        )

    @app.get("/v1/services/catalog")
    def service_catalog() -> list[dict[str, Any]]:
        return load_catalog("services")

    @app.get("/v1/services")
    def list_services(project_id: str | None = None) -> list[dict[str, Any]]:
        external_registry.refresh()
        values = state.list_resources("service", project_id)
        return [
            item
            for item in values
            if item.get("source") != "external" or item.get("name") in external_registry.names
        ]

    @app.post("/v1/services/actions", response_model=Operation)
    def service_action(
        value: ServiceAction, x_embodiedlab_role: str = Header(default="user")
    ) -> Operation:
        if x_embodiedlab_role != "operator":
            raise HTTPException(status_code=403, detail="service lifecycle requires operator role")
        try:
            canonical = service_spec(value.name)["name"]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not managed_services_enabled:
            raise HTTPException(
                status_code=409,
                detail="model service lifecycle is disabled; use configured external services",
            )
        if canonical in external_registry.names:
            raise HTTPException(
                status_code=409,
                detail=f"service {canonical} is externally managed",
            )
        value = value.model_copy(update={"name": canonical})
        return _new_operation(
            state,
            OperationCreate(
                kind="service.reconcile", project_id=value.project_id, input=value.model_dump()
            ),
        )

    @app.post("/v1/batches")
    def create_batch(value: BatchCreate) -> Batch:
        batch_id = resource_id("batch")
        operations = [
            _new_operation(
                state,
                OperationCreate(
                    kind="evaluate.batch-job",
                    project_id=value.project_id,
                    input={**job, "batch_id": batch_id, "job_index": index},
                ),
            )
            for index, job in enumerate(value.jobs)
        ]
        batch = Batch(
            id=batch_id,
            project_id=value.project_id,
            state="queued",
            operation_ids=[item.id for item in operations],
            max_concurrency=value.max_concurrency,
        )
        state.put_resource("batch", batch.id, batch.model_dump(mode="json"))
        return batch

    @app.get("/v1/batches")
    def list_batches(project_id: str | None = None) -> list[dict[str, Any]]:
        return state.list_resources("batch", project_id)

    @app.get("/v1/batches/{batch_id}")
    def get_batch(batch_id: str) -> dict[str, Any]:
        value = state.get_resource("batch", batch_id)
        if value is None:
            raise _not_found("batch", batch_id)
        operations = [state.get_operation(item) for item in value["operation_ids"]]
        present = [item for item in operations if item]
        statuses = {item.status for item in present}
        if statuses & {OperationStatus.RUNNING, OperationStatus.SCHEDULED}:
            value["state"] = "running"
        elif present and all(item.status in TERMINAL_OPERATION_STATUSES for item in present):
            if OperationStatus.FAILED in statuses:
                value["state"] = "failed"
            elif OperationStatus.INTERRUPTED in statuses:
                value["state"] = "interrupted"
            elif statuses == {OperationStatus.CANCELLED}:
                value["state"] = "cancelled"
            else:
                value["state"] = "completed"
        value["summary"] = {
            "jobs": len(present),
            "completed": sum(item.status == OperationStatus.COMPLETED for item in present),
            "succeeded": sum(item.success is True for item in present),
            "unsuccessful": sum(item.success is False for item in present),
            "unevaluated": sum(
                item.status == OperationStatus.COMPLETED and item.success is None
                for item in present
            ),
        }
        value["operations"] = [item.model_dump(mode="json") for item in present]
        state.put_resource(
            "batch", batch_id, {key: item for key, item in value.items() if key != "operations"}
        )
        return value

    @app.post("/v1/batches/{batch_id}/cancel")
    def cancel_batch(batch_id: str) -> dict[str, Any]:
        value = state.get_resource("batch", batch_id)
        if value is None:
            raise _not_found("batch", batch_id)
        for operation_id in value["operation_ids"]:
            operation = state.get_operation(operation_id)
            if operation and operation.status not in TERMINAL_OPERATION_STATUSES:
                state.put_operation(
                    _transition(operation, OperationPatch(status=OperationStatus.CANCELLED))
                )
        value.update(state="cancelled", updated_at=utcnow().isoformat())
        return state.put_resource("batch", batch_id, value)

    @app.get("/v1/skills")
    def list_skills() -> list[dict[str, Any]]:
        return load_catalog("skills")

    @app.put("/v1/workflows/{digest}")
    def upload_workflow(
        digest: str,
        content: bytes = Body(media_type="application/octet-stream"),
        x_embodiedlab_project: str = Header(default="default"),
    ) -> dict[str, Any]:
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise HTTPException(status_code=422, detail="workflow digest must use sha256:<hex>")
        actual = f"sha256:{hashlib.sha256(content).hexdigest()}"
        if actual != digest:
            raise HTTPException(status_code=422, detail="workflow content digest mismatch")
        key = f"projects/{x_embodiedlab_project}/workflows/{digest[7:]}.tar.gz"
        existed = artifacts.exists(key)
        uri = key if existed else artifacts.put(key, io.BytesIO(content))
        value = {
            "digest": digest,
            "project_id": x_embodiedlab_project,
            "key": key,
            "uri": uri,
            "cached": existed,
            "read_only": True,
        }
        state.put_resource("workflow", f"{x_embodiedlab_project}:{digest}", value)
        return value

    @app.get("/v1/workflows/{digest}")
    def download_workflow(
        digest: str,
        x_embodiedlab_project: str = Header(default="default"),
    ) -> StreamingResponse:
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise HTTPException(
                status_code=422,
                detail="workflow digest must use sha256:<hex>",
            )
        key = f"projects/{x_embodiedlab_project}/workflows/{digest[7:]}.tar.gz"
        if not artifacts.exists(key):
            raise _not_found("workflow", digest)
        return StreamingResponse(
            artifacts.iter_bytes(key),
            media_type="application/gzip",
            headers={
                "X-EmbodiedLab-Workflow-Digest": digest,
                "Content-Disposition": f'attachment; filename="{digest[7:]}.tar.gz"',
            },
        )

    @app.put("/v1/tool-bundles/{digest}")
    def upload_tool_bundle(
        digest: str,
        content: bytes = Body(media_type="application/octet-stream"),
        x_embodiedlab_project: str = Header(default="default"),
    ) -> dict[str, Any]:
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise HTTPException(status_code=422, detail="tool bundle digest must use sha256:<hex>")
        actual = f"sha256:{hashlib.sha256(content).hexdigest()}"
        if actual != digest:
            raise HTTPException(status_code=422, detail="tool bundle content digest mismatch")
        key = f"projects/{x_embodiedlab_project}/tool-bundles/{digest[7:]}.tar.gz"
        existed = artifacts.exists(key)
        uri = key if existed else artifacts.put(key, io.BytesIO(content))
        value = {
            "digest": digest,
            "project_id": x_embodiedlab_project,
            "key": key,
            "uri": uri,
            "cached": existed,
            "read_only": True,
        }
        state.put_resource("tool_bundle", f"{x_embodiedlab_project}:{digest}", value)
        return value

    @app.get("/v1/tool-bundles/{digest}")
    def download_tool_bundle(
        digest: str,
        x_embodiedlab_project: str = Header(default="default"),
    ) -> StreamingResponse:
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise HTTPException(status_code=422, detail="tool bundle digest must use sha256:<hex>")
        key = f"projects/{x_embodiedlab_project}/tool-bundles/{digest[7:]}.tar.gz"
        if not artifacts.exists(key):
            raise _not_found("tool bundle", digest)
        return StreamingResponse(
            artifacts.iter_bytes(key),
            media_type="application/gzip",
            headers={
                "X-EmbodiedLab-Tool-Bundle-Digest": digest,
                "Content-Disposition": f'attachment; filename="{digest[7:]}.tar.gz"',
            },
        )

    return app


app = create_app()

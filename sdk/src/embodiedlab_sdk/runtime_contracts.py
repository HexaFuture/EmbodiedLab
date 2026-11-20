"""Versioned wire contracts for Sim-managed Worker runtimes."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RUNTIME_PROTOCOL_FAMILY = "embodiedlab.runtime"
RUNTIME_PROTOCOL_VERSION = "2.0"
RUNTIME_CONTRACT_REVISION = "1"
SUPPORTED_RUNTIME_PROTOCOL_VERSIONS = (RUNTIME_PROTOCOL_VERSION,)

Digest = str
ProtocolVersion = Literal["2.0"]


class RuntimeWireModel(BaseModel):
    """Fail closed when a peer sends fields this contract does not understand."""

    model_config = ConfigDict(extra="forbid")


class WorkerRuntimeState(str, Enum):
    READY = "ready"
    BUSY = "busy"
    DRAINING = "draining"
    DEGRADED = "degraded"


class RuntimeCommandType(str, Enum):
    ENVIRONMENT_START = "environment.start"
    ENVIRONMENT_STOP = "environment.stop"
    WORKLOAD_RUN = "workload.run"
    WORKLOAD_CANCEL = "workload.cancel"
    INVENTORY_REFRESH = "inventory.refresh"


class RuntimeCommandState(str, Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RuntimeEventType(str, Enum):
    STATE = "state"
    PHASE = "phase"
    PROGRESS = "progress"
    METRIC = "metric"
    CHECKPOINT = "checkpoint"
    MESSAGE = "message"
    HEARTBEAT = "heartbeat"
    INVENTORY = "inventory"


class GpuDeviceAssignment(RuntimeWireModel):
    physical_gpu_id: int = Field(ge=0)
    logical_index: int = Field(ge=0)
    gpu_worker_index: int | None = Field(default=None, ge=0)
    device_uuid: str | None = Field(default=None, min_length=1, max_length=128)
    mig_uuid: str | None = Field(default=None, min_length=1, max_length=128)


class ResourceAssignment(RuntimeWireModel):
    cpu_ids: list[int] = Field(default_factory=list, max_length=4096)
    gpu_devices: list[GpuDeviceAssignment] = Field(default_factory=list, max_length=64)
    memory_mb: int = Field(ge=128)

    @field_validator("cpu_ids")
    @classmethod
    def cpu_ids_are_unique(cls, value: list[int]) -> list[int]:
        if any(cpu_id < 0 for cpu_id in value):
            raise ValueError("cpu_ids must be non-negative")
        if len(value) != len(set(value)):
            raise ValueError("cpu_ids must be unique")
        return value

    @field_validator("gpu_devices")
    @classmethod
    def gpu_devices_are_unique(cls, value: list[GpuDeviceAssignment]) -> list[GpuDeviceAssignment]:
        physical_ids = [device.physical_gpu_id for device in value]
        logical_ids = [device.logical_index for device in value]
        if len(physical_ids) != len(set(physical_ids)):
            raise ValueError("physical GPU assignments must be unique")
        if len(logical_ids) != len(set(logical_ids)):
            raise ValueError("logical GPU assignments must be unique")
        return value


class GpuInventoryItem(RuntimeWireModel):
    physical_gpu_id: int = Field(ge=0)
    device_uuid: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=128)
    memory_total_mb: int = Field(ge=1)
    memory_free_mb: int = Field(ge=0)
    healthy: bool = True
    mig_enabled: bool = False

    @model_validator(mode="after")
    def free_memory_fits_total(self) -> GpuInventoryItem:
        if self.memory_free_mb > self.memory_total_mb:
            raise ValueError("GPU free memory cannot exceed total memory")
        return self


class WorkerCapabilities(RuntimeWireModel):
    simulator_backends: list[str] = Field(default_factory=list, max_length=128)
    runtimes: list[str] = Field(default_factory=list, max_length=32)
    viewer_protocols: list[str] = Field(default_factory=list, max_length=16)
    features: list[str] = Field(default_factory=list, max_length=128)


class WorkerInventory(RuntimeWireModel):
    total_cpu: int = Field(ge=1)
    available_cpu: int = Field(ge=0)
    total_memory_mb: int = Field(ge=128)
    available_memory_mb: int = Field(ge=0)
    gpu_devices: list[GpuInventoryItem] = Field(default_factory=list, max_length=64)
    environment_ids: list[str] = Field(default_factory=list, max_length=4096)

    @model_validator(mode="after")
    def available_capacity_fits_total(self) -> WorkerInventory:
        if self.available_cpu > self.total_cpu:
            raise ValueError("available CPU cannot exceed total CPU")
        if self.available_memory_mb > self.total_memory_mb:
            raise ValueError("available memory cannot exceed total memory")
        return self


class WorkerRegistrationRequest(RuntimeWireModel):
    agent_instance_id: str = Field(min_length=1, max_length=128)
    offered_protocol_versions: list[str] = Field(min_length=1, max_length=16)
    sdk_version: str = Field(min_length=1, max_length=64)
    hostname: str = Field(min_length=1, max_length=255)
    labels: dict[str, str] = Field(default_factory=dict)
    capabilities: WorkerCapabilities
    inventory: WorkerInventory
    capability_digest: Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    inventory_digest: Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class WorkerRegistrationResponse(RuntimeWireModel):
    worker_id: str = Field(min_length=1, max_length=128)
    agent_generation: int = Field(ge=1)
    protocol_version: ProtocolVersion = RUNTIME_PROTOCOL_VERSION
    contract_revision: Literal["1"] = RUNTIME_CONTRACT_REVISION
    heartbeat_interval_seconds: int = Field(default=5, ge=1, le=300)
    claim_timeout_seconds: int = Field(default=30, ge=5, le=300)
    server_time: datetime


class WorkerHeartbeatRequest(RuntimeWireModel):
    protocol_version: ProtocolVersion = RUNTIME_PROTOCOL_VERSION
    agent_instance_id: str = Field(min_length=1, max_length=128)
    agent_generation: int = Field(ge=1)
    state: WorkerRuntimeState
    capability_digest: Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    inventory_digest: Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    inventory: WorkerInventory | None = None
    observed_at: datetime


class WorkerHeartbeatResponse(RuntimeWireModel):
    accepted: Literal[True] = True
    agent_generation: int = Field(ge=1)
    next_heartbeat_seconds: int = Field(default=5, ge=1, le=300)
    inventory_refresh_required: bool = False
    server_time: datetime


class ViewerTunnelCredentialRequest(RuntimeWireModel):
    agent_instance_id: str = Field(min_length=1, max_length=128)
    agent_generation: int = Field(ge=1)


class ViewerTunnelCredentialResponse(RuntimeWireModel):
    route_key: str = Field(min_length=32, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    tunnel_path: str = Field(
        min_length=1,
        max_length=512,
        pattern=r"^/internal/v1/tunnels/[A-Za-z0-9_-]+$",
    )
    ticket: str = Field(min_length=32, max_length=4096)
    expires_at: int = Field(ge=1)


class RuntimeCommand(RuntimeWireModel):
    protocol_version: ProtocolVersion = RUNTIME_PROTOCOL_VERSION
    command_id: str = Field(min_length=1, max_length=128)
    command_type: RuntimeCommandType
    execution_id: str = Field(min_length=1, max_length=128)
    environment_id: str = Field(min_length=1, max_length=128)
    worker_id: str = Field(min_length=1, max_length=128)
    agent_generation: int = Field(ge=1)
    assignment_id: str = Field(min_length=1, max_length=128)
    lease_id: str = Field(min_length=1, max_length=128)
    lease_revision: int = Field(ge=1)
    runtime_revision: int = Field(ge=1)
    idempotency_key: str = Field(min_length=1, max_length=256)
    delivery_id: str = Field(min_length=1, max_length=128)
    delivery_attempt: int = Field(ge=1)
    resource_assignment: ResourceAssignment
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    claim_expires_at: datetime

    @model_validator(mode="after")
    def claim_expiry_follows_creation(self) -> RuntimeCommand:
        if self.claim_expires_at <= self.created_at:
            raise ValueError("claim expiry must follow command creation")
        return self


class RuntimeCommandClaimRequest(RuntimeWireModel):
    protocol_version: ProtocolVersion = RUNTIME_PROTOCOL_VERSION
    agent_instance_id: str = Field(min_length=1, max_length=128)
    agent_generation: int = Field(ge=1)
    max_wait_seconds: int = Field(default=20, ge=0, le=30)


class RuntimeCommandClaimResponse(RuntimeWireModel):
    command: RuntimeCommand | None = None
    retry_after_seconds: int = Field(default=1, ge=0, le=30)
    server_time: datetime


class RuntimeDeliveryIdentity(RuntimeWireModel):
    protocol_version: ProtocolVersion = RUNTIME_PROTOCOL_VERSION
    agent_instance_id: str = Field(min_length=1, max_length=128)
    agent_generation: int = Field(ge=1)
    delivery_id: str = Field(min_length=1, max_length=128)
    runtime_revision: int = Field(ge=1)


class RuntimeCommandAckRequest(RuntimeDeliveryIdentity):
    acknowledged_at: datetime


class RuntimeCommandRenewRequest(RuntimeDeliveryIdentity):
    extend_seconds: int = Field(default=30, ge=5, le=300)


class RuntimeEvent(RuntimeWireModel):
    event_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    sequence: int = Field(ge=1)
    event_type: RuntimeEventType
    state: RuntimeCommandState | None = None
    phase: str | None = Field(default=None, min_length=1, max_length=128)
    current: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=1)
    unit: str | None = Field(default=None, min_length=1, max_length=32)
    message: str | None = Field(default=None, max_length=1024)
    metrics: dict[str, float] = Field(default_factory=dict)
    checkpoint_ref: str | None = Field(default=None, min_length=1, max_length=1024)
    payload: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime

    @model_validator(mode="after")
    def event_fields_match_type(self) -> RuntimeEvent:
        if self.event_type == RuntimeEventType.STATE and self.state is None:
            raise ValueError("state events require state")
        if self.event_type == RuntimeEventType.PHASE and self.phase is None:
            raise ValueError("phase events require phase")
        if self.event_type == RuntimeEventType.PROGRESS and self.current is None:
            raise ValueError("progress events require current")
        if self.total is not None and self.current is not None and self.current > self.total:
            raise ValueError("progress current cannot exceed total")
        if self.total is not None and not self.unit:
            raise ValueError("determinate progress requires unit")
        if self.event_type == RuntimeEventType.CHECKPOINT and not self.checkpoint_ref:
            raise ValueError("checkpoint events require checkpoint_ref")
        if len(self.metrics) > 32:
            raise ValueError("events may contain at most 32 metrics")
        return self


class RuntimeCommandEventsRequest(RuntimeDeliveryIdentity):
    events: list[RuntimeEvent] = Field(min_length=1, max_length=100)

    @field_validator("events")
    @classmethod
    def event_sequences_are_strictly_increasing(
        cls, value: list[RuntimeEvent]
    ) -> list[RuntimeEvent]:
        sequences = [event.sequence for event in value]
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise ValueError("event sequences must be strictly increasing")
        if len({event.event_id for event in value}) != len(value):
            raise ValueError("event IDs must be unique within a batch")
        return value


class RuntimeErrorDetail(RuntimeWireModel):
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=1024)
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class RuntimeCommandCompleteRequest(RuntimeDeliveryIdentity):
    event_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    sequence: int = Field(ge=1)
    terminal_state: Literal["succeeded", "failed", "cancelled"]
    output: dict[str, Any] = Field(default_factory=dict)
    error: RuntimeErrorDetail | None = None
    completed_at: datetime

    @model_validator(mode="after")
    def terminal_result_is_coherent(self) -> RuntimeCommandCompleteRequest:
        if self.terminal_state == "failed" and self.error is None:
            raise ValueError("failed completion requires an error")
        if self.terminal_state != "failed" and self.error is not None:
            raise ValueError("only failed completion may contain an error")
        return self


class RuntimeCommandReceipt(RuntimeWireModel):
    command_id: str = Field(min_length=1, max_length=128)
    state: RuntimeCommandState
    accepted_sequence: int = Field(default=0, ge=0)
    claim_expires_at: datetime | None = None
    duplicate: bool = False
    server_time: datetime


class RuntimeApiError(RuntimeWireModel):
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=1024)
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class RuntimeApiErrorEnvelope(RuntimeWireModel):
    error: RuntimeApiError


class RuntimeProtocolUnsupported(RuntimeWireModel):
    error: RuntimeApiError
    supported_protocol_versions: list[ProtocolVersion]
    contract_revision: Literal["1"] = RUNTIME_CONTRACT_REVISION


def select_runtime_protocol(offered_versions: list[str]) -> ProtocolVersion:
    for version in SUPPORTED_RUNTIME_PROTOCOL_VERSIONS:
        if version in offered_versions:
            return RUNTIME_PROTOCOL_VERSION
    raise ValueError("runtime_protocol_unsupported")


_SCHEMA_MODELS: tuple[type[BaseModel], ...] = (
    GpuDeviceAssignment,
    ResourceAssignment,
    GpuInventoryItem,
    WorkerCapabilities,
    WorkerInventory,
    WorkerRegistrationRequest,
    WorkerRegistrationResponse,
    WorkerHeartbeatRequest,
    WorkerHeartbeatResponse,
    ViewerTunnelCredentialRequest,
    ViewerTunnelCredentialResponse,
    RuntimeCommand,
    RuntimeCommandClaimRequest,
    RuntimeCommandClaimResponse,
    RuntimeDeliveryIdentity,
    RuntimeCommandAckRequest,
    RuntimeCommandRenewRequest,
    RuntimeEvent,
    RuntimeCommandEventsRequest,
    RuntimeErrorDetail,
    RuntimeCommandCompleteRequest,
    RuntimeCommandReceipt,
    RuntimeApiError,
    RuntimeApiErrorEnvelope,
    RuntimeProtocolUnsupported,
)


def runtime_schema_components() -> dict[str, Any]:
    components: dict[str, Any] = {}
    for model in _SCHEMA_MODELS:
        schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
        definitions = schema.pop("$defs", {})
        for name, definition in definitions.items():
            existing = components.get(name)
            if existing is not None and existing != definition:
                raise ValueError(f"conflicting JSON Schema definition: {name}")
            components[name] = definition
        components[model.__name__] = schema
    return dict(sorted(components.items()))


def _request(schema_name: str) -> dict[str, Any]:
    return {
        "required": True,
        "content": {
            "application/json": {"schema": {"$ref": f"#/components/schemas/{schema_name}"}}
        },
    }


def _responses(success_schema: str, *, created: bool = False) -> dict[str, Any]:
    status = "201" if created else "200"
    response = {
        "description": "Accepted" if not created else "Registered",
        "content": {
            "application/json": {"schema": {"$ref": f"#/components/schemas/{success_schema}"}}
        },
    }
    errors = {
        code: {
            "description": description,
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/RuntimeApiErrorEnvelope"}
                }
            },
        }
        for code, description in {
            "400": "Malformed runtime request",
            "401": "Missing or invalid runtime credential",
            "403": "Worker identity or scope rejected",
            "404": "Runtime resource not found",
            "409": "Generation, delivery, revision, or state conflict",
            "410": "Delivery claim expired",
            "422": "Runtime contract validation failed",
            "429": "Runtime request rate limited",
            "503": "Runtime control plane temporarily unavailable",
        }.items()
    }
    return {status: response, **errors}


def runtime_openapi_document() -> dict[str, Any]:
    worker_parameter = {
        "name": "worker_id",
        "in": "path",
        "required": True,
        "schema": {"type": "string", "minLength": 1, "maxLength": 128},
    }
    command_parameter = {
        "name": "command_id",
        "in": "path",
        "required": True,
        "schema": {"type": "string", "minLength": 1, "maxLength": 128},
    }
    worker_security = [{"WorkerBearer": [], "WorkerRegistrationKey": []}]
    paths: dict[str, Any] = {
        "/internal/v2/runtime/workers/register": {
            "post": {
                "operationId": "registerRuntimeWorkerV2",
                "security": worker_security,
                "requestBody": _request("WorkerRegistrationRequest"),
                "responses": {
                    **_responses("WorkerRegistrationResponse", created=True),
                    "426": {
                        "description": "No mutually supported runtime protocol",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/RuntimeProtocolUnsupported"
                                }
                            }
                        },
                    },
                },
            }
        },
        "/internal/v2/runtime/workers/{worker_id}/heartbeat": {
            "post": {
                "operationId": "heartbeatRuntimeWorkerV2",
                "security": worker_security,
                "parameters": [worker_parameter],
                "requestBody": _request("WorkerHeartbeatRequest"),
                "responses": _responses("WorkerHeartbeatResponse"),
            }
        },
        "/internal/v2/runtime/workers/{worker_id}/viewer-credentials": {
            "post": {
                "operationId": "issueRuntimeViewerTunnelCredentialV2",
                "security": worker_security,
                "parameters": [worker_parameter],
                "requestBody": _request("ViewerTunnelCredentialRequest"),
                "responses": _responses("ViewerTunnelCredentialResponse"),
            }
        },
        "/internal/v2/runtime/workers/{worker_id}/commands/claim": {
            "post": {
                "operationId": "claimRuntimeCommandV2",
                "security": worker_security,
                "parameters": [worker_parameter],
                "requestBody": _request("RuntimeCommandClaimRequest"),
                "responses": _responses("RuntimeCommandClaimResponse"),
            }
        },
    }
    command_actions = {
        "ack": ("ackRuntimeCommandV2", "RuntimeCommandAckRequest"),
        "renew": ("renewRuntimeCommandV2", "RuntimeCommandRenewRequest"),
        "events": ("appendRuntimeCommandEventsV2", "RuntimeCommandEventsRequest"),
        "complete": ("completeRuntimeCommandV2", "RuntimeCommandCompleteRequest"),
    }
    for action, (operation_id, request_schema) in command_actions.items():
        paths[f"/internal/v2/runtime/commands/{{command_id}}/{action}"] = {
            "post": {
                "operationId": operation_id,
                "security": worker_security,
                "parameters": [command_parameter],
                "requestBody": _request(request_schema),
                "responses": _responses("RuntimeCommandReceipt"),
            }
        }
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "EmbodiedLab Sim Managed Runtime API",
            "version": RUNTIME_PROTOCOL_VERSION,
            "x-contract-revision": RUNTIME_CONTRACT_REVISION,
        },
        "servers": [{"url": "/", "description": "Sim control plane"}],
        "paths": dict(sorted(paths.items())),
        "components": {
            "securitySchemes": {
                "WorkerBearer": {"type": "http", "scheme": "bearer"},
                "WorkerRegistrationKey": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-EmbodiedLab-Worker-Registration-Key",
                },
            },
            "schemas": runtime_schema_components(),
        },
    }


def runtime_compatibility_document() -> dict[str, Any]:
    return {
        "schema_version": "embodiedlab.runtime.compatibility/v1",
        "contract_revision": RUNTIME_CONTRACT_REVISION,
        "protocol_family": RUNTIME_PROTOCOL_FAMILY,
        "managed": {
            "supported_protocol_versions": list(SUPPORTED_RUNTIME_PROTOCOL_VERSIONS),
            "sdk_versions": [">=0.2.0,<0.3.0"],
            "unknown_fields": "reject",
            "unknown_protocol_versions": "reject_with_426",
            "standalone_fallback": "forbidden",
        },
        "standalone": {
            "sdk_controller_api": "v1",
            "status": "compatibility_only",
            "sim_managed_state_authority": False,
        },
    }

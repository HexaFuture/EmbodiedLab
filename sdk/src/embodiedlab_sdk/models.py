"""Versioned public resource models shared by Controller, clients, and workers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from math import isfinite
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def resource_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class WorkerRole(str, Enum):
    SIMULATION = "simulation-worker"
    MODEL = "model-worker"


class OperationStatus(str, Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class ProgressEventType(str, Enum):
    PHASE = "phase"
    PROGRESS = "progress"
    METRIC = "metric"
    CHECKPOINT = "checkpoint"
    MESSAGE = "message"
    HEARTBEAT = "heartbeat"


class OperationPhase(str, Enum):
    SUBMITTED = "submitted"
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    ENVIRONMENT_STARTING = "environment_starting"
    ENVIRONMENT_READY = "environment_ready"
    WORKLOAD_STARTING = "workload_starting"
    WORKLOAD_RUNNING = "workload_running"
    COLLECTING = "collecting"
    EVALUATING = "evaluating"
    CLEANUP = "cleanup"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


MetricName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")]


TERMINAL_OPERATION_STATUSES = {
    OperationStatus.COMPLETED,
    OperationStatus.FAILED,
    OperationStatus.INTERRUPTED,
    OperationStatus.CANCELLED,
}


class ResourceMeta(BaseModel):
    id: str
    project_id: str = "default"
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class OperationCreate(BaseModel):
    kind: str = Field(min_length=1, max_length=128)
    project_id: str = "default"
    requested_by: str = "cli"
    target_worker_id: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)


class Operation(ResourceMeta):
    kind: str
    requested_by: str
    status: OperationStatus = OperationStatus.PENDING
    success: bool | None = None
    success_source: str | None = None
    worker_id: str | None = None
    target_worker_id: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    output_included: bool = True
    error: dict[str, Any] | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_success_semantics(self) -> Operation:
        if self.success is not None and not self.success_source:
            raise ValueError("success requires an evaluation source")
        if self.success is not None and self.status != OperationStatus.COMPLETED:
            raise ValueError("success may only be attached to a completed operation")
        return self


class OperationPatch(BaseModel):
    status: OperationStatus
    worker_id: str | None = None
    output: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None


class OperationProgressEventCreate(BaseModel):
    event_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    sequence: int = Field(ge=1)
    event_type: ProgressEventType
    phase: OperationPhase | None = None
    current: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=1)
    unit: str | None = Field(default=None, min_length=1, max_length=32)
    message: str | None = Field(default=None, max_length=512)
    metrics: dict[MetricName, float] = Field(default_factory=dict)
    checkpoint_ref: str | None = Field(default=None, min_length=1, max_length=512)
    episode_id: str | None = Field(default=None, min_length=1, max_length=128)
    episode_index: int | None = Field(default=None, ge=0)
    step_index: int | None = Field(default=None, ge=0)
    sample_count: int | None = Field(default=None, ge=0)
    worker_id: str = Field(min_length=1, max_length=128)
    agent_generation: str = Field(min_length=1, max_length=128)
    attempt: int = Field(default=1, ge=1)
    trace_id: str | None = Field(default=None, min_length=1, max_length=128)
    occurred_at: datetime

    @field_validator("metrics")
    @classmethod
    def metrics_are_bounded(cls, value: dict[str, float]) -> dict[str, float]:
        if len(value) > 32:
            raise ValueError("progress event metrics may contain at most 32 values")
        if any(not isfinite(metric) for metric in value.values()):
            raise ValueError("progress event metrics must be finite")
        return value

    @model_validator(mode="after")
    def validate_progress_semantics(self) -> OperationProgressEventCreate:
        if self.event_type == ProgressEventType.PHASE and self.phase is None:
            raise ValueError("phase events require phase")
        if self.event_type == ProgressEventType.PROGRESS and self.current is None:
            raise ValueError("progress events require current")
        if self.total is not None and self.current is not None and self.current > self.total:
            raise ValueError("progress current cannot exceed total")
        if self.total is not None and not self.unit:
            raise ValueError("determinate progress requires unit")
        if self.event_type == ProgressEventType.CHECKPOINT and not self.checkpoint_ref:
            raise ValueError("checkpoint events require checkpoint_ref")
        if self.occurred_at > utcnow() + timedelta(minutes=5):
            raise ValueError("progress event timestamp is too far in the future")
        return self


class OperationProgressEvent(OperationProgressEventCreate):
    operation_id: str
    project_id: str
    created_at: datetime = Field(default_factory=utcnow)


class EvaluationCreate(BaseModel):
    success: bool
    source: str = Field(min_length=1, max_length=128)
    details: dict[str, Any] = Field(default_factory=dict)


class WorkerRegister(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    role: WorkerRole
    registration_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    endpoint: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    capacity: dict[str, Any] = Field(default_factory=dict)
    version: str


class Worker(ResourceMeta):
    name: str
    role: WorkerRole
    registration_key: str | None = None
    endpoint: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    capacity: dict[str, Any] = Field(default_factory=dict)
    version: str
    state: Literal["ready", "draining", "offline"] = "ready"
    last_heartbeat_at: datetime = Field(default_factory=utcnow)
    agent_generation: str | None = None


class Heartbeat(BaseModel):
    state: Literal["ready", "draining"] = "ready"
    capacity: dict[str, Any] = Field(default_factory=dict)
    agent_generation: str | None = Field(default=None, min_length=1, max_length=128)


class EnvironmentCreate(BaseModel):
    profile: str
    count: int = Field(default=1, ge=1, le=1024)
    project_id: str = "default"
    seed: int = 0
    task_id: str | None = None
    image: str | None = None
    gpu: str | None = None
    record_video: bool = False
    target_worker_id: str | None = None
    correlation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    workflow_digest: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class Environment(ResourceMeta):
    profile: str
    state: Literal["pending", "starting", "ready", "stopping", "stopped", "failed"]
    worker_id: str | None = None
    seed: int
    task_id: str | None = None
    operation_id: str
    endpoint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CommandCreate(BaseModel):
    project_id: str = "default"
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    project_id: str = "default"
    environment_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolBundleAttach(BaseModel):
    project_id: str = "default"
    digest: str


class ToolBundleDetach(BaseModel):
    project_id: str = "default"
    digest: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def validate_target(self) -> ToolBundleDetach:
        if (self.digest is None) == (self.name is None):
            raise ValueError("detach requires exactly one of digest or name")
        return self


class ToolVisibility(BaseModel):
    project_id: str = "default"
    name: str = Field(min_length=1, max_length=128)
    visible: bool


class ServiceAction(BaseModel):
    project_id: str = "default"
    name: str
    replicas: int = Field(default=1, ge=0, le=128)
    image: str | None = None
    gpu: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class BatchCreate(BaseModel):
    project_id: str = "default"
    jobs: list[dict[str, Any]] = Field(min_length=1)
    max_concurrency: int = Field(default=3, ge=1, le=1024)


class Batch(ResourceMeta):
    state: Literal["queued", "running", "completed", "failed", "interrupted", "cancelled"]
    operation_ids: list[str]
    max_concurrency: int
    summary: dict[str, Any] = Field(default_factory=dict)

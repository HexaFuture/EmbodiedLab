"""Versioned Observation/Action/StepResult contracts for RobotEndpoints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .artifacts import validate_digest
from .models import utcnow

ENDPOINT_SCHEMA = "embodiedlab.endpoint/v1alpha1"
OBSERVATION_SCHEMA = ENDPOINT_SCHEMA
ACTION_SCHEMA = ENDPOINT_SCHEMA
STEP_RESULT_SCHEMA = ENDPOINT_SCHEMA


def _identifier(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def canonical_json(value: BaseModel | Mapping[str, Any]) -> bytes:
    """Encode a contract value deterministically before hashing or transport."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def contract_digest(value: BaseModel | Mapping[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value)).hexdigest()}"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BlobRef(ContractModel):
    """Reference to a large immutable observation channel."""

    digest: str
    media_type: str = Field(min_length=1)
    size_bytes: int | None = Field(default=None, ge=0)
    dtype: str | None = None
    shape: list[int] = Field(default_factory=list)

    @field_validator("digest")
    @classmethod
    def validate_blob_digest(cls, value: str) -> str:
        validate_digest(value, label="blob")
        return value

    @field_validator("shape")
    @classmethod
    def validate_blob_shape(cls, value: list[int]) -> list[int]:
        if any(dimension < 0 for dimension in value):
            raise ValueError("blob shape dimensions must be non-negative")
        return value


class ObservationChannelSpec(ContractModel):
    channel: str = Field(min_length=1, max_length=256)
    dtype: str = Field(min_length=1, max_length=64)
    shape: list[int | None] = Field(default_factory=list)
    frame: str | None = Field(default=None, min_length=1, max_length=128)
    unit: str | None = Field(default=None, min_length=1, max_length=64)
    required: bool = True

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, value: list[int | None]) -> list[int | None]:
        if any(dimension is not None and dimension < 0 for dimension in value):
            raise ValueError("channel shape dimensions must be non-negative or null")
        return value

    def validate_value(self, value: Any) -> None:
        if not self.shape:
            return
        actual_shape = _declared_shape(value)
        _validate_shape(self.shape, actual_shape, f"observation {self.channel}")
        declared_dtype = getattr(value, "dtype", None)
        if declared_dtype is None and isinstance(value, Mapping):
            declared_dtype = value.get("dtype")
        if declared_dtype is not None and str(declared_dtype) != self.dtype:
            raise ValueError(
                f"observation {self.channel} dtype {declared_dtype!s} does not match {self.dtype}"
            )


class ObservationSpec(ContractModel):
    schema_version: Literal[OBSERVATION_SCHEMA] = OBSERVATION_SCHEMA
    channels: list[ObservationChannelSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_channels(self) -> ObservationSpec:
        names = [item.channel for item in self.channels]
        if len(names) != len(set(names)):
            raise ValueError("observation channels must have unique names")
        return self

    def channel(self, name: str) -> ObservationChannelSpec:
        for item in self.channels:
            if item.channel == name:
                return item
        raise KeyError(name)

    def validate_observation(self, observation: ObservationEnvelope) -> ObservationEnvelope:
        observation = ObservationEnvelope.model_validate(observation)
        known = {item.channel for item in self.channels}
        present = observation.channel_names
        unknown = sorted(present - known)
        if unknown:
            raise ValueError(f"observation contains unknown channels: {', '.join(unknown)}")
        missing = sorted(
            item.channel for item in self.channels if item.required and item.channel not in present
        )
        if missing:
            raise ValueError(f"observation is missing required channels: {', '.join(missing)}")
        for item in self.channels:
            if item.channel in observation.channels:
                item.validate_value(observation.channels[item.channel])
            elif (
                item.channel in observation.blobs
                and observation.blobs[item.channel].dtype not in {None, item.dtype}
            ):
                raise ValueError(
                    f"observation {item.channel} blob dtype does not match {item.dtype}"
                )
            elif item.channel in observation.blobs and observation.blobs[item.channel].shape:
                _validate_shape(
                    item.shape,
                    observation.blobs[item.channel].shape,
                    f"observation {item.channel}",
                )
        return observation


class ObservationEnvelope(ContractModel):
    schema_version: Literal[OBSERVATION_SCHEMA] = OBSERVATION_SCHEMA
    observation_id: str = Field(default_factory=lambda: _identifier("obs"), min_length=1)
    episode_id: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    captured_at: datetime = Field(default_factory=utcnow)
    channels: dict[str, Any] = Field(default_factory=dict)
    blobs: dict[str, BlobRef] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_channel_storage(self) -> ObservationEnvelope:
        overlap = sorted(set(self.channels).intersection(self.blobs))
        if overlap:
            raise ValueError(
                f"observation channels cannot be both inline and blob-backed: {', '.join(overlap)}"
            )
        if not self.channels and not self.blobs:
            raise ValueError("observation must contain at least one channel")
        return self

    @property
    def channel_names(self) -> set[str]:
        return set(self.channels).union(self.blobs)


class ActionSpec(ContractModel):
    schema_version: Literal[ACTION_SCHEMA] = ACTION_SCHEMA
    kind: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]*/v[0-9]+$",
    )
    dtype: str = Field(min_length=1, max_length=64)
    shape: list[int | None] = Field(default_factory=list)
    frame: str | None = Field(default=None, min_length=1, max_length=128)
    unit: str | None = Field(default=None, min_length=1, max_length=64)
    minimum: float | list[float] | None = None
    maximum: float | list[float] | None = None
    control_frequency_hz: float = Field(gt=0)
    horizon: int = Field(default=1, ge=1)
    limits: dict[str, Any] = Field(default_factory=dict)

    @field_validator("minimum", "maximum")
    @classmethod
    def validate_bound_vector(cls, value: float | list[float] | None) -> float | list[float] | None:
        if isinstance(value, list) and not value:
            raise ValueError("action bounds must not be empty")
        return value

    @model_validator(mode="after")
    def validate_bounds(self) -> ActionSpec:
        if self.minimum is not None and self.maximum is not None:
            minimum = self.minimum if isinstance(self.minimum, list) else [self.minimum]
            maximum = self.maximum if isinstance(self.maximum, list) else [self.maximum]
            if len(minimum) not in {1, len(maximum)} and len(maximum) not in {1, len(minimum)}:
                raise ValueError("action minimum and maximum must have compatible lengths")
            if any(low > high for low, high in zip(minimum, maximum, strict=False)):
                raise ValueError("action minimum cannot exceed maximum")
        if any(dimension is not None and dimension < 0 for dimension in self.shape):
            raise ValueError("action shape dimensions must be non-negative or null")
        return self

    def validate_payload(self, payload: Any) -> None:
        if self.shape:
            actual_shape = _declared_shape(payload)
            _validate_shape(self.shape, actual_shape, f"action {self.kind}")
        numbers = list(_numeric_values(payload))
        if not numbers:
            return
        minimum = self.minimum if isinstance(self.minimum, list) else [self.minimum]
        maximum = self.maximum if isinstance(self.maximum, list) else [self.maximum]
        for index, number in enumerate(numbers):
            if minimum != [None] and minimum[0] is not None:
                lower = minimum[index] if len(minimum) > 1 else minimum[0]
                if lower is not None and number < lower:
                    raise ValueError(f"action {self.kind} payload is below its lower limit")
            if maximum != [None] and maximum[0] is not None:
                upper = maximum[index] if len(maximum) > 1 else maximum[0]
                if upper is not None and number > upper:
                    raise ValueError(f"action {self.kind} payload is above its upper limit")


def _validate_shape(
    expected: list[int | None], actual: list[int] | None, label: str
) -> None:
    if actual is None or len(actual) != len(expected) or any(
        dimension is not None and dimension != current
        for dimension, current in zip(expected, actual or [], strict=True)
    ):
        raise ValueError(f"{label} value has an invalid shape")


def _declared_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None and isinstance(value, Mapping):
        shape = value.get("shape")
    if shape is not None:
        try:
            return [int(dimension) for dimension in shape]
        except (TypeError, ValueError):
            return None
    return _payload_shape(value)


def _payload_shape(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)):
        return []
    if not value:
        return [0]
    child_shapes = [_payload_shape(item) for item in value]
    if any(shape is None for shape in child_shapes):
        return None
    first = child_shapes[0]
    if any(shape != first for shape in child_shapes[1:]):
        return None
    return [len(value), *(first or [])]


def _numeric_values(value: Any) -> Sequence[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        values: list[float] = []
        for item in value:
            values.extend(_numeric_values(item))
        return values
    return []


class Action(ContractModel):
    schema_version: Literal[ACTION_SCHEMA] = ACTION_SCHEMA
    action_id: str = Field(default_factory=lambda: _identifier("action"), min_length=1)
    kind: str = Field(min_length=1, max_length=128)
    based_on_observation_id: str = Field(min_length=1)
    sequence_id: str = Field(default_factory=lambda: _identifier("sequence"), min_length=1)
    sequence_index: int = Field(default=0, ge=0)
    valid_from: datetime = Field(default_factory=utcnow)
    valid_until: datetime | None = None
    horizon: int = Field(default=1, ge=1)
    payload: Any
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_validity_window(self) -> Action:
        if self.valid_until is not None and self.valid_until <= self.valid_from:
            raise ValueError("action valid_until must be later than valid_from")
        return self

    def validate_for(self, spec: ActionSpec) -> Action:
        if self.kind != spec.kind:
            raise ValueError(f"action kind {self.kind!r} is not supported by endpoint")
        if self.horizon > spec.horizon:
            raise ValueError(
                f"action horizon {self.horizon} exceeds {self.kind} limit {spec.horizon}"
            )
        spec.validate_payload(self.payload)
        return self


class ActionError(ContractModel):
    code: str = Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_.-]*$")
    message: str = Field(min_length=1, max_length=4096)
    details: dict[str, Any] = Field(default_factory=dict)


class StepResult(ContractModel):
    schema_version: Literal[STEP_RESULT_SCHEMA] = STEP_RESULT_SCHEMA
    action_id: str = Field(min_length=1)
    accepted: bool
    next_observation: ObservationEnvelope | None = None
    terminated: bool = False
    truncated: bool = False
    reward: float | None = None
    info: dict[str, Any] = Field(default_factory=dict)
    error: ActionError | None = None

    @model_validator(mode="after")
    def validate_result_semantics(self) -> StepResult:
        if self.accepted and self.error is not None:
            raise ValueError("accepted StepResult cannot contain an error")
        if not self.accepted and self.error is None:
            raise ValueError("rejected StepResult must contain a structured error")
        if self.terminated and self.truncated:
            raise ValueError("StepResult cannot be both terminated and truncated")
        return self


class TimingContract(ContractModel):
    control_frequency_hz: float = Field(gt=0)
    action_horizon: int = Field(default=1, ge=1)
    step_timeout_ms: int = Field(default=1000, gt=0)
    observation_timeout_ms: int = Field(default=1000, gt=0)


class EndpointContract(ContractModel):
    schema_version: Literal[ENDPOINT_SCHEMA] = ENDPOINT_SCHEMA
    observation_spec: ObservationSpec
    action_specs: list[ActionSpec] = Field(min_length=1)
    step_result_spec: str = STEP_RESULT_SCHEMA
    timing: TimingContract
    digest: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> EndpointContract:
        kinds = [item.kind for item in self.action_specs]
        if len(kinds) != len(set(kinds)):
            raise ValueError("endpoint action specs must have unique kinds")
        if any(item.horizon > self.timing.action_horizon for item in self.action_specs):
            raise ValueError("action spec horizon cannot exceed endpoint action horizon")
        expected = self.compute_digest()
        if self.digest is None:
            self.digest = expected
        elif self.digest != expected:
            raise ValueError("endpoint contract digest does not match its contents")
        return self

    def compute_digest(self) -> str:
        return contract_digest(self.model_copy(update={"digest": None}))

    def action_spec(self, kind: str) -> ActionSpec:
        for spec in self.action_specs:
            if spec.kind == kind:
                return spec
        raise KeyError(kind)

    def validate_action(self, action: Action) -> Action:
        action = Action.model_validate(action)
        return action.validate_for(self.action_spec(action.kind))

    def validate_step_result(self, action: Action, result: StepResult) -> StepResult:
        result = StepResult.model_validate(result)
        if result.action_id != action.action_id:
            raise ValueError(
                f"StepResult action_id {result.action_id!r} does not match {action.action_id!r}"
            )
        if result.next_observation is not None:
            self.observation_spec.validate_observation(result.next_observation)
        return result


__all__ = [
    "ACTION_SCHEMA",
    "ENDPOINT_SCHEMA",
    "OBSERVATION_SCHEMA",
    "STEP_RESULT_SCHEMA",
    "Action",
    "ActionError",
    "ActionSpec",
    "BlobRef",
    "EndpointContract",
    "ObservationChannelSpec",
    "ObservationEnvelope",
    "ObservationSpec",
    "StepResult",
    "TimingContract",
    "canonical_json",
    "contract_digest",
]

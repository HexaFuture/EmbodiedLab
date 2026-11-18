"""Sim-managed Worker client with a durable, filesystem-backed command ledger."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import httpx

from .runtime_contracts import (
    RUNTIME_PROTOCOL_VERSION,
    RuntimeCommand,
    RuntimeCommandAckRequest,
    RuntimeCommandClaimRequest,
    RuntimeCommandClaimResponse,
    RuntimeCommandCompleteRequest,
    RuntimeCommandEventsRequest,
    RuntimeCommandReceipt,
    RuntimeCommandRenewRequest,
    RuntimeErrorDetail,
    RuntimeEvent,
    ViewerTunnelCredentialRequest,
    ViewerTunnelCredentialResponse,
    WorkerCapabilities,
    WorkerHeartbeatRequest,
    WorkerHeartbeatResponse,
    WorkerInventory,
    WorkerRegistrationRequest,
    WorkerRegistrationResponse,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ManagedCommandExecutor(Protocol):
    """Implementations must reconcile by command idempotency key and runtime revision."""

    def execute(self, command: RuntimeCommand) -> dict[str, Any]: ...

    def cancel(self, command: RuntimeCommand) -> None: ...


@dataclass(frozen=True)
class ManagedWorkerIdentity:
    worker_id: str
    agent_instance_id: str
    agent_generation: int


class ManagedRuntimeClient:
    """Strict v2 client. It has no SDK Controller or standalone fallback."""

    def __init__(
        self,
        base_url: str,
        bearer_token: str,
        registration_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not bearer_token or not registration_key:
            raise ValueError("managed runtime requires bearer token and registration key")
        self.http = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {bearer_token}",
                "X-EmbodiedLab-Worker-Registration-Key": registration_key,
            },
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self.http.close()

    def register(self, request: WorkerRegistrationRequest) -> WorkerRegistrationResponse:
        response = self.http.post(
            "internal/v2/runtime/workers/register", json=request.model_dump(mode="json")
        )
        response.raise_for_status()
        return WorkerRegistrationResponse.model_validate(response.json())

    def heartbeat(
        self, identity: ManagedWorkerIdentity, request: WorkerHeartbeatRequest
    ) -> WorkerHeartbeatResponse:
        self._verify_identity(identity, request.agent_instance_id, request.agent_generation)
        response = self.http.post(
            f"internal/v2/runtime/workers/{identity.worker_id}/heartbeat",
            json=request.model_dump(mode="json", exclude_none=True),
        )
        response.raise_for_status()
        return WorkerHeartbeatResponse.model_validate(response.json())

    def viewer_credentials(self, identity: ManagedWorkerIdentity) -> ViewerTunnelCredentialResponse:
        payload = ViewerTunnelCredentialRequest(
            agent_instance_id=identity.agent_instance_id,
            agent_generation=identity.agent_generation,
        )
        response = self.http.post(
            f"internal/v2/runtime/workers/{identity.worker_id}/viewer-credentials",
            json=payload.model_dump(mode="json"),
        )
        response.raise_for_status()
        return ViewerTunnelCredentialResponse.model_validate(response.json())

    def claim(
        self, identity: ManagedWorkerIdentity, *, max_wait_seconds: int = 20
    ) -> RuntimeCommand | None:
        request = RuntimeCommandClaimRequest(
            agent_instance_id=identity.agent_instance_id,
            agent_generation=identity.agent_generation,
            max_wait_seconds=max_wait_seconds,
        )
        response = self.http.post(
            f"internal/v2/runtime/workers/{identity.worker_id}/commands/claim",
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()
        claimed = RuntimeCommandClaimResponse.model_validate(response.json()).command
        if claimed is not None:
            self._verify_command(identity, claimed)
        return claimed

    def acknowledge(
        self, identity: ManagedWorkerIdentity, command: RuntimeCommand
    ) -> RuntimeCommandReceipt:
        request = RuntimeCommandAckRequest(
            **self._delivery(identity, command), acknowledged_at=_utcnow()
        )
        return self._command_write(command.command_id, "ack", request)

    def publish_events(
        self,
        identity: ManagedWorkerIdentity,
        command: RuntimeCommand,
        events: list[RuntimeEvent],
    ) -> RuntimeCommandReceipt:
        request = RuntimeCommandEventsRequest(**self._delivery(identity, command), events=events)
        return self._command_write(command.command_id, "events", request)

    def renew(
        self,
        identity: ManagedWorkerIdentity,
        command: RuntimeCommand,
        *,
        extend_seconds: int = 30,
    ) -> RuntimeCommandReceipt:
        request = RuntimeCommandRenewRequest(
            **self._delivery(identity, command), extend_seconds=extend_seconds
        )
        return self._command_write(command.command_id, "renew", request)

    def complete(
        self,
        identity: ManagedWorkerIdentity,
        command: RuntimeCommand,
        *,
        event_id: str,
        sequence: int,
        terminal_state: str,
        output: dict[str, Any] | None = None,
        error: RuntimeErrorDetail | None = None,
    ) -> RuntimeCommandReceipt:
        request = RuntimeCommandCompleteRequest(
            **self._delivery(identity, command),
            event_id=event_id,
            sequence=sequence,
            terminal_state=terminal_state,
            output=output or {},
            error=error,
            completed_at=_utcnow(),
        )
        return self._command_write(command.command_id, "complete", request)

    def _command_write(
        self,
        command_id: str,
        action: str,
        request: RuntimeCommandAckRequest
        | RuntimeCommandRenewRequest
        | RuntimeCommandEventsRequest
        | RuntimeCommandCompleteRequest,
    ) -> RuntimeCommandReceipt:
        response = self.http.post(
            f"internal/v2/runtime/commands/{command_id}/{action}",
            json=request.model_dump(mode="json", exclude_none=True),
        )
        response.raise_for_status()
        return RuntimeCommandReceipt.model_validate(response.json())

    @staticmethod
    def _delivery(identity: ManagedWorkerIdentity, command: RuntimeCommand) -> dict[str, Any]:
        ManagedRuntimeClient._verify_command(identity, command)
        return {
            "protocol_version": RUNTIME_PROTOCOL_VERSION,
            "agent_instance_id": identity.agent_instance_id,
            "agent_generation": identity.agent_generation,
            "delivery_id": command.delivery_id,
            "runtime_revision": command.runtime_revision,
        }

    @staticmethod
    def _verify_identity(
        identity: ManagedWorkerIdentity, agent_instance_id: str, agent_generation: int
    ) -> None:
        if (
            agent_instance_id != identity.agent_instance_id
            or agent_generation != identity.agent_generation
        ):
            raise ValueError("managed runtime identity mismatch")

    @staticmethod
    def _verify_command(identity: ManagedWorkerIdentity, command: RuntimeCommand) -> None:
        if command.worker_id != identity.worker_id:
            raise ValueError("runtime command targets another worker")
        if command.agent_generation != identity.agent_generation:
            raise ValueError("runtime command has stale agent generation")


class LocalCommandLedger:
    """One JSON record per idempotency key, serialized with a process-safe lock."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self._lock_path = self.root / ".ledger.lock"

    def record_delivery(self, command: RuntimeCommand) -> tuple[dict[str, Any], bool]:
        fingerprint = self._fingerprint(command)
        with self._lock():
            path = self._record_path(command.idempotency_key, command.runtime_revision)
            current = self._read(path)
            if current is not None:
                if current["fingerprint"] != fingerprint:
                    raise RuntimeError("idempotency key reused with different command")
                return current, False
            record = {
                "schema_version": "embodiedlab.runtime.local-command/v1",
                "command_id": command.command_id,
                "idempotency_key": command.idempotency_key,
                "runtime_revision": command.runtime_revision,
                "fingerprint": fingerprint,
                "state": "acknowledged",
                "last_sequence": 0,
                "output": {},
                "error": None,
                "updated_at": _utcnow().isoformat(),
            }
            self._write(path, record)
            return record, True

    def update(
        self,
        command: RuntimeCommand,
        *,
        state: str,
        last_sequence: int,
        output: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock():
            path = self._record_path(command.idempotency_key, command.runtime_revision)
            current = self._read(path)
            if current is None or current["fingerprint"] != self._fingerprint(command):
                raise RuntimeError("runtime command was not durably acknowledged")
            if last_sequence < current["last_sequence"]:
                raise RuntimeError("runtime command sequence cannot move backwards")
            current.update(
                state=state,
                last_sequence=last_sequence,
                output=output or {},
                error=error,
                updated_at=_utcnow().isoformat(),
            )
            self._write(path, current)
            return current

    def get(self, command: RuntimeCommand) -> dict[str, Any] | None:
        with self._lock():
            record = self._read(
                self._record_path(command.idempotency_key, command.runtime_revision)
            )
            if record is not None and record["fingerprint"] != self._fingerprint(command):
                raise RuntimeError("idempotency key reused with different command")
            return record

    def _record_path(self, idempotency_key: str, runtime_revision: int) -> Path:
        digest = hashlib.sha256(f"{idempotency_key}\0{runtime_revision}".encode()).hexdigest()
        return self.root / f"{digest}.json"

    @staticmethod
    def _fingerprint(command: RuntimeCommand) -> str:
        payload = command.model_dump(
            mode="json", exclude={"delivery_id", "delivery_attempt", "claim_expires_at"}
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def _lock(self):
        return _LedgerLock(self._lock_path)

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write(path: Path, record: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(record, stream, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary.exists():
                temporary.unlink()


class _LedgerLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open("a+", encoding="utf-8")
        os.chmod(self.path, 0o600)
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        assert self.stream is not None
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


class ManagedCommandRunner:
    """Execute each command revision once and replay its durable terminal result."""

    TERMINAL_STATES = {"succeeded", "failed", "cancelled"}

    def __init__(self, executor: ManagedCommandExecutor, ledger: LocalCommandLedger) -> None:
        self.executor = executor
        self.ledger = ledger

    def run(
        self, command: RuntimeCommand, *, running_sequence: int = 1
    ) -> tuple[dict[str, Any], bool]:
        if running_sequence < 1:
            raise ValueError("running sequence must be positive")
        record, created = self.ledger.record_delivery(command)
        if not created and record["state"] in self.TERMINAL_STATES:
            return record, False
        if record["state"] == "acknowledged":
            self.ledger.update(command, state="running", last_sequence=running_sequence)
        try:
            if command.command_type.value == "workload.cancel":
                self.executor.cancel(command)
                output: dict[str, Any] = {}
                state = "cancelled"
            else:
                output = self.executor.execute(command)
                state = "succeeded"
            terminal_increment = 2 if command.command_type.value == "workload.run" else 1
            return (
                self.ledger.update(
                    command,
                    state=state,
                    last_sequence=running_sequence + terminal_increment,
                    output=output,
                ),
                True,
            )
        except Exception as exc:
            error = {
                "code": type(exc).__name__,
                "message": str(exc)[:1024],
                "retryable": False,
            }
            terminal_increment = 2 if command.command_type.value == "workload.run" else 1
            return (
                self.ledger.update(
                    command,
                    state="failed",
                    last_sequence=running_sequence + terminal_increment,
                    error=error,
                ),
                True,
            )


class ManagedWorkerDaemon:
    """One-step managed command loop; scheduling remains exclusively in Sim."""

    def __init__(
        self,
        client: ManagedRuntimeClient,
        identity: ManagedWorkerIdentity,
        runner: ManagedCommandRunner,
    ) -> None:
        self.client = client
        self.identity = identity
        self.runner = runner

    def run_once(self, *, max_wait_seconds: int = 20) -> bool:
        command = self.client.claim(self.identity, max_wait_seconds=max_wait_seconds)
        if command is None:
            return False
        self.runner.ledger.record_delivery(command)
        self.client.acknowledge(self.identity, command)
        existing = self.runner.ledger.get(command)
        assert existing is not None
        running_sequence = 1
        if existing["state"] not in ManagedCommandRunner.TERMINAL_STATES:
            events = self._running_events(command)
            self.client.publish_events(self.identity, command, events)
            running_sequence = events[-1].sequence
        record, _executed = self.runner.run(command, running_sequence=running_sequence)
        terminal_state = record["state"]
        if command.command_type.value == "workload.run":
            self.client.publish_events(
                self.identity,
                command,
                [self._workload_terminal_prelude(command, record)],
            )
        error = None
        if terminal_state == "failed":
            error = RuntimeErrorDetail.model_validate(record["error"])
        self.client.complete(
            self.identity,
            command,
            event_id=f"{command.command_id}:{record['last_sequence']}",
            sequence=record["last_sequence"],
            terminal_state=terminal_state,
            output=record["output"],
            error=error,
        )
        return True

    @staticmethod
    def _workload_terminal_prelude(command: RuntimeCommand, record: dict[str, Any]) -> RuntimeEvent:
        sequence = record["last_sequence"] - 1
        now = _utcnow()
        output = record.get("output") or {}
        steps = output.get("steps")
        if record["state"] == "succeeded" and isinstance(steps, int) and steps > 0:
            return RuntimeEvent(
                event_id=f"{command.command_id}:{sequence}",
                sequence=sequence,
                event_type="progress",
                phase="collecting",
                current=steps,
                total=steps,
                unit="steps",
                message="Managed workload completed",
                occurred_at=now,
            )
        return RuntimeEvent(
            event_id=f"{command.command_id}:{sequence}",
            sequence=sequence,
            event_type="message",
            message="Managed workload reached a terminal state",
            occurred_at=now,
        )

    @staticmethod
    def _running_events(command: RuntimeCommand) -> list[RuntimeEvent]:
        now = _utcnow()
        events = [
            RuntimeEvent(
                event_id=f"{command.command_id}:1",
                sequence=1,
                event_type="state",
                state="running",
                occurred_at=now,
            )
        ]
        if command.command_type.value == "environment.start":
            events.append(
                RuntimeEvent(
                    event_id=f"{command.command_id}:2",
                    sequence=2,
                    event_type="phase",
                    phase="environment_starting",
                    message="Preparing the managed environment",
                    occurred_at=now,
                )
            )
        elif command.command_type.value == "workload.run":
            arguments = command.payload.get("execution", {}).get("arguments", {})
            steps = arguments.get("steps")
            events.append(
                RuntimeEvent(
                    event_id=f"{command.command_id}:2",
                    sequence=2,
                    event_type="progress",
                    phase="workload_running",
                    current=0,
                    total=steps if isinstance(steps, int) and steps > 0 else None,
                    unit="steps" if isinstance(steps, int) and steps > 0 else None,
                    message="Managed workload started",
                    occurred_at=now,
                )
            )
        elif command.command_type.value == "workload.cancel":
            events.append(
                RuntimeEvent(
                    event_id=f"{command.command_id}:2",
                    sequence=2,
                    event_type="phase",
                    phase="cleanup",
                    message="Cancelling the managed workload",
                    occurred_at=now,
                )
            )
        return events


def registration_request(
    *,
    agent_instance_id: str,
    sdk_version: str,
    hostname: str,
    capabilities: WorkerCapabilities,
    inventory: WorkerInventory,
    capability_digest: str,
    inventory_digest: str,
    labels: dict[str, str] | None = None,
) -> WorkerRegistrationRequest:
    return WorkerRegistrationRequest(
        agent_instance_id=agent_instance_id,
        offered_protocol_versions=[RUNTIME_PROTOCOL_VERSION],
        sdk_version=sdk_version,
        hostname=hostname,
        labels=labels or {},
        capabilities=capabilities,
        inventory=inventory,
        capability_digest=capability_digest,
        inventory_digest=inventory_digest,
    )

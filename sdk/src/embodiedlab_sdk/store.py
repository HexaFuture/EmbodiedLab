"""Persistence boundary. PostgreSQL is required outside explicit dev/test mode."""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from importlib.resources import files
from threading import RLock
from typing import Any

from .models import (
    Operation,
    OperationProgressEvent,
    OperationStatus,
    Worker,
    WorkerRole,
    utcnow,
)


class OperationEventConflict(ValueError):
    pass


def _same_client_event(first: OperationProgressEvent, second: OperationProgressEvent) -> bool:
    return first.model_dump(exclude={"created_at"}) == second.model_dump(exclude={"created_at"})

_ACTIVE_ENVIRONMENT_STATES = frozenset({"pending", "starting", "ready", "stopping"})
_FINISHED_OPERATION_STATES = frozenset(
    status.value
    for status in (
        OperationStatus.COMPLETED,
        OperationStatus.FAILED,
        OperationStatus.INTERRUPTED,
        OperationStatus.CANCELLED,
    )
)
# How long a finished Operation keeps its output in the in-memory store. Clients read the
# output as soon as the Operation finishes; after this the record stays, without its output.
OPERATION_OUTPUT_RETENTION_S = float(
    os.environ.get("EMBODIEDLAB_OPERATION_OUTPUT_RETENTION_S", "600")
)


def _worker_environment_limit(worker: Worker) -> int | None:
    """Return the declared live-environment capacity for a simulation Worker."""

    raw = worker.capacity.get("max_environments")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        return 0
    return raw


def _operation_environment_count(operation: Operation) -> int:
    environment_ids = operation.input.get("environment_ids")
    if isinstance(environment_ids, list) and environment_ids:
        return len(environment_ids)
    raw = operation.input.get("count", 1)
    return raw if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0 else 1


def _worker_can_claim(
    operation: Operation,
    worker: Worker,
    *,
    active_environment_count: int,
) -> bool:
    if operation.target_worker_id not in {None, worker.id}:
        return False
    required_role = _required_role(operation.kind)
    if required_role is not None and required_role != worker.role:
        return False
    if operation.kind != "env.start":
        return True

    requested_gpu = operation.input.get("gpu")
    worker_gpu = worker.labels.get("gpu")
    if requested_gpu is not None and worker_gpu is not None:
        if str(requested_gpu) != str(worker_gpu):
            return False

    limit = _worker_environment_limit(worker)
    if limit is not None:
        requested = _operation_environment_count(operation)
        if active_environment_count + requested > limit:
            return False
    return True


def _bind_worker_allocation(operation: Operation, worker: Worker) -> Operation:
    """Persist the GPU selected by a labeled Worker on an unpinned start."""

    if operation.kind != "env.start" or operation.input.get("gpu") is not None:
        return operation
    worker_gpu = worker.labels.get("gpu")
    if worker_gpu is None:
        return operation
    return operation.model_copy(update={"input": {**operation.input, "gpu": worker_gpu}})


class Store(ABC):
    @abstractmethod
    def put_operation(self, operation: Operation) -> Operation: ...

    @abstractmethod
    def get_operation(self, operation_id: str) -> Operation | None: ...

    @abstractmethod
    def list_operations(
        self, project_id: str | None = None, include_output: bool = True
    ) -> list[Operation]: ...

    @abstractmethod
    def claim_operation(self, worker: Worker) -> Operation | None: ...

    @abstractmethod
    def put_worker(self, worker: Worker) -> Worker: ...

    @abstractmethod
    def register_worker(self, worker: Worker) -> Worker: ...

    @abstractmethod
    def get_worker(self, worker_id: str) -> Worker | None: ...

    @abstractmethod
    def list_workers(self) -> list[Worker]: ...

    @abstractmethod
    def put_resource(
        self, kind: str, resource_id: str, value: dict[str, Any]
    ) -> dict[str, Any]: ...

    @abstractmethod
    def get_resource(self, kind: str, resource_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def list_resources(self, kind: str, project_id: str | None = None) -> list[dict[str, Any]]: ...

    @abstractmethod
    def append_log(self, operation_id: str, stream: str, message: str) -> None: ...

    @abstractmethod
    def list_logs(self, operation_id: str, after: int = 0) -> list[dict[str, Any]]: ...

    @abstractmethod
    def append_operation_event(
        self, event: OperationProgressEvent
    ) -> tuple[OperationProgressEvent, bool]: ...

    @abstractmethod
    def list_operation_events(
        self, operation_id: str, after: int = 0, limit: int = 100
    ) -> list[OperationProgressEvent]: ...

    @abstractmethod
    def audit_orphans(self) -> dict[str, int]: ...


class InMemoryStore(Store):
    """Deterministic test/development store; never selected implicitly in production.

    Every Operation record is retained, so the store keeps two side structures to stay
    fast with hundreds of thousands of them: an index of the pending records (the only
    ones a Worker can claim; a poll used to scan every record under the lock), and a
    queue of finished records whose output (images, masks) is dropped
    ``output_retention_s`` after they finished; the record itself stays, with
    ``output={}`` and ``output_included=False``.
    """

    def __init__(
        self,
        output_retention_s: float | None = None,
        clock=time.monotonic,
    ) -> None:
        self._lock = RLock()
        self._operations: dict[str, dict[str, Any]] = {}
        self._pending: dict[str, dict[str, Any]] = {}
        self._finished: deque[tuple[float, str]] = deque()
        self._output_retention_s = (
            OPERATION_OUTPUT_RETENTION_S if output_retention_s is None else output_retention_s
        )
        self._clock = clock
        self._workers: dict[str, dict[str, Any]] = {}
        self._resources: dict[str, dict[str, dict[str, Any]]] = {}
        self._logs: dict[str, list[dict[str, Any]]] = {}
        self._operation_events: dict[str, list[dict[str, Any]]] = {}

    def _store_operation(self, record: dict[str, Any]) -> None:
        """Write one record and keep the pending index and the output queue in step (under the lock)."""

        operation_id = record["id"]
        self._operations[operation_id] = record
        if record.get("status") == OperationStatus.PENDING.value:
            self._pending[operation_id] = record
        else:
            self._pending.pop(operation_id, None)
        if record.get("status") in _FINISHED_OPERATION_STATES and record.get("output"):
            self._finished.append((self._clock(), operation_id))
        self._drop_old_outputs()

    def _drop_old_outputs(self) -> None:
        """Drop the output of every Operation that finished more than the retention ago (under the lock)."""

        cutoff = self._clock() - self._output_retention_s
        while self._finished and self._finished[0][0] <= cutoff:
            _, operation_id = self._finished.popleft()
            record = self._operations.get(operation_id)
            if (
                record is not None
                and record.get("status") in _FINISHED_OPERATION_STATES
                and record.get("output")
            ):
                # Replace the record whole: list_operations holds record references outside the lock.
                self._operations[operation_id] = {**record, "output": {}, "output_included": False}

    def put_operation(self, operation: Operation) -> Operation:
        with self._lock:
            self._store_operation(operation.model_dump(mode="json"))
        return operation

    def get_operation(self, operation_id: str) -> Operation | None:
        value = self._operations.get(operation_id)
        return Operation.model_validate(deepcopy(value)) if value else None

    def list_operations(
        self, project_id: str | None = None, include_output: bool = True
    ) -> list[Operation]:
        # Snapshot under the lock: environments write Operations while a list is
        # being copied, and iterating the live dict then raises "dictionary changed
        # size during iteration".  Records are replaced whole, so holding the record
        # references is safe; the output (images, masks) is only copied when asked for.
        with self._lock:
            items = [
                item
                for item in self._operations.values()
                if project_id is None or item.get("project_id") == project_id
            ]
        values = []
        for item in items:
            if include_output:
                values.append(Operation.model_validate(deepcopy(item)))
            else:
                summary = deepcopy({key: value for key, value in item.items() if key != "output"})
                summary.update(output={}, output_included=False)
                values.append(Operation.model_validate(summary))
        return sorted(values, key=lambda item: item.created_at, reverse=True)

    def claim_operation(self, worker: Worker) -> Operation | None:
        """Atomically claim one compatible pending operation."""

        with self._lock:
            active_environment_count = sum(
                1
                for item in self._resources.get("environment", {}).values()
                if item.get("worker_id") == worker.id
                and item.get("state") in _ACTIVE_ENVIRONMENT_STATES
            )
            # Only pending records can be claimed; they are indexed, since scanning every
            # retained Operation on each Worker poll stalled the Controller.
            values = sorted(self._pending.values(), key=lambda item: item["created_at"])
            for raw in values:
                operation = Operation.model_validate(raw)
                if operation.status != OperationStatus.PENDING:
                    continue
                if not _worker_can_claim(
                    operation,
                    worker,
                    active_environment_count=active_environment_count,
                ):
                    continue
                operation = _bind_worker_allocation(operation, worker)
                claimed = operation.model_copy(
                    update={
                        "status": OperationStatus.SCHEDULED,
                        "worker_id": worker.id,
                        "updated_at": utcnow(),
                    }
                )
                self._store_operation(claimed.model_dump(mode="json"))
                return claimed
        return None

    def put_worker(self, worker: Worker) -> Worker:
        with self._lock:
            self._workers[worker.id] = worker.model_dump(mode="json")
        return worker

    def register_worker(self, worker: Worker) -> Worker:
        with self._lock:
            if worker.registration_key:
                matches = [
                    Worker.model_validate(raw)
                    for raw in self._workers.values()
                    if raw.get("registration_key") == worker.registration_key
                ]
                if len(matches) > 1:
                    raise ValueError("registration_key resolves to multiple Workers")
                if matches:
                    existing = matches[0]
                    if existing.role != worker.role:
                        raise ValueError("registration_key role cannot change")
                    worker = worker.model_copy(
                        update={"id": existing.id, "created_at": existing.created_at}
                    )
            self._workers[worker.id] = worker.model_dump(mode="json")
        return worker

    def get_worker(self, worker_id: str) -> Worker | None:
        value = self._workers.get(worker_id)
        return Worker.model_validate(deepcopy(value)) if value else None

    def list_workers(self) -> list[Worker]:
        return [Worker.model_validate(deepcopy(item)) for item in self._workers.values()]

    def put_resource(self, kind: str, resource_id: str, value: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._resources.setdefault(kind, {})[resource_id] = deepcopy(value)
        return deepcopy(value)

    def get_resource(self, kind: str, resource_id: str) -> dict[str, Any] | None:
        value = self._resources.get(kind, {}).get(resource_id)
        return deepcopy(value) if value else None

    def list_resources(self, kind: str, project_id: str | None = None) -> list[dict[str, Any]]:
        values = [deepcopy(item) for item in self._resources.get(kind, {}).values()]
        if project_id is not None:
            values = [item for item in values if item.get("project_id") == project_id]
        return values

    def append_log(self, operation_id: str, stream: str, message: str) -> None:
        with self._lock:
            entries = self._logs.setdefault(operation_id, [])
            entries.append({"offset": len(entries) + 1, "stream": stream, "message": message})

    def list_logs(self, operation_id: str, after: int = 0) -> list[dict[str, Any]]:
        return [
            deepcopy(item) for item in self._logs.get(operation_id, []) if item["offset"] > after
        ]

    def append_operation_event(
        self, event: OperationProgressEvent
    ) -> tuple[OperationProgressEvent, bool]:
        with self._lock:
            events = self._operation_events.setdefault(event.operation_id, [])
            for raw in events:
                existing = OperationProgressEvent.model_validate(raw)
                if existing.event_id == event.event_id:
                    if not _same_client_event(existing, event):
                        raise OperationEventConflict("event_id already contains different data")
                    return existing, False
                if existing.sequence == event.sequence:
                    raise OperationEventConflict("operation event sequence already exists")
            expected = events[-1]["sequence"] + 1 if events else 1
            if event.sequence != expected:
                raise OperationEventConflict(
                    f"operation event sequence gap: expected {expected}, got {event.sequence}"
                )
            events.append(event.model_dump(mode="json"))
        return event, True

    def list_operation_events(
        self, operation_id: str, after: int = 0, limit: int = 100
    ) -> list[OperationProgressEvent]:
        values = [
            OperationProgressEvent.model_validate(deepcopy(item))
            for item in self._operation_events.get(operation_id, [])
            if item["sequence"] > after
        ]
        return values[:limit]

    def audit_orphans(self) -> dict[str, int]:
        operation_ids = set(self._operations)
        return {
            "operation_event.operation": sum(
                len(events)
                for operation_id, events in self._operation_events.items()
                if operation_id not in operation_ids
            )
        }


class PostgresStore(Store):
    """Small psycopg implementation using JSONB resources and explicit migrations."""

    def __init__(self, database_url: str, *, migrate: bool = True) -> None:
        import psycopg

        self._psycopg = psycopg
        self.database_url = database_url
        if migrate:
            self.migrate()

    def _connect(self):
        return self._psycopg.connect(self.database_url)

    def migrate(self) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            migrations = files("embodiedlab_sdk").joinpath("migrations")
            for migration in sorted(migrations.iterdir(), key=lambda item: item.name):
                if migration.name.endswith(".sql") and not migration.name.endswith(".down.sql"):
                    cursor.execute(migration.read_text())

    def rollback_migration(self, migration_name: str) -> None:
        if not migration_name or any(value in migration_name for value in ("/", "\\", "..")):
            raise ValueError("invalid migration name")
        migration = files("embodiedlab_sdk").joinpath(
            "migrations", f"{migration_name}.down.sql"
        )
        if not migration.is_file():
            raise ValueError(f"rollback migration does not exist: {migration_name}")
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(migration.read_text())

    def _put_json(self, table: str, item_id: str, value: dict[str, Any]) -> None:
        payload = json.dumps(value)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {table} (id, data) VALUES (%s, %s::jsonb) "
                "ON CONFLICT (id) DO UPDATE SET data=EXCLUDED.data, updated_at=now()",
                (item_id, payload),
            )

    def _get_json(self, table: str, item_id: str) -> dict[str, Any] | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT data FROM {table} WHERE id=%s", (item_id,))
            row = cursor.fetchone()
        return row[0] if row else None

    def _list_json(self, table: str) -> list[dict[str, Any]]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT data FROM {table} ORDER BY created_at DESC")
            return [row[0] for row in cursor.fetchall()]

    def put_operation(self, operation: Operation) -> Operation:
        self._put_json("embodiedlab_operations", operation.id, operation.model_dump(mode="json"))
        return operation

    def get_operation(self, operation_id: str) -> Operation | None:
        value = self._get_json("embodiedlab_operations", operation_id)
        return Operation.model_validate(value) if value else None

    def list_operations(
        self, project_id: str | None = None, include_output: bool = True
    ) -> list[Operation]:
        values = [Operation.model_validate(item) for item in self._list_json("embodiedlab_operations")]
        return [item for item in values if project_id is None or item.project_id == project_id]

    def claim_operation(self, worker: Worker) -> Operation | None:
        """Claim with one PostgreSQL transaction and SKIP LOCKED."""

        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT count(*)
                FROM embodiedlab_resources
                WHERE kind='environment'
                  AND data->>'worker_id'=%s
                  AND data->>'state'=ANY(%s::text[])
                """,
                (worker.id, list(_ACTIVE_ENVIRONMENT_STATES)),
            )
            active_environment_count = int(cursor.fetchone()[0])
            cursor.execute(
                """
                SELECT id, data
                FROM embodiedlab_operations
                WHERE data->>'status' = 'pending'
                  AND (
                    data->>'target_worker_id' IS NULL
                    OR data->>'target_worker_id' = %s
                  )
                ORDER BY created_at ASC
                FOR UPDATE SKIP LOCKED
                """,
                (worker.id,),
            )
            for operation_id, raw in cursor.fetchall():
                operation = Operation.model_validate(raw)
                if not _worker_can_claim(
                    operation,
                    worker,
                    active_environment_count=active_environment_count,
                ):
                    continue
                operation = _bind_worker_allocation(operation, worker)
                claimed = operation.model_copy(
                    update={
                        "status": OperationStatus.SCHEDULED,
                        "worker_id": worker.id,
                        "updated_at": utcnow(),
                    }
                )
                cursor.execute(
                    """
                    UPDATE embodiedlab_operations
                    SET data=%s::jsonb, updated_at=now()
                    WHERE id=%s AND data->>'status'='pending'
                    """,
                    (
                        json.dumps(claimed.model_dump(mode="json"), default=str),
                        operation_id,
                    ),
                )
                if cursor.rowcount == 1:
                    return claimed
        return None

    def put_worker(self, worker: Worker) -> Worker:
        self._put_json("embodiedlab_workers", worker.id, worker.model_dump(mode="json"))
        return worker

    def register_worker(self, worker: Worker) -> Worker:
        if not worker.registration_key:
            return self.put_worker(worker)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"embodiedlab-worker-registration:{worker.registration_key}",),
            )
            cursor.execute(
                "SELECT data FROM embodiedlab_workers WHERE data->>'registration_key'=%s FOR UPDATE",
                (worker.registration_key,),
            )
            values = [Worker.model_validate(row[0]) for row in cursor.fetchall()]
            if len(values) > 1:
                raise ValueError("registration_key resolves to multiple Workers")
            if values:
                existing = values[0]
                if existing.role != worker.role:
                    raise ValueError("registration_key role cannot change")
                worker = worker.model_copy(
                    update={"id": existing.id, "created_at": existing.created_at}
                )
            cursor.execute(
                "INSERT INTO embodiedlab_workers (id, data) VALUES (%s, %s::jsonb) "
                "ON CONFLICT (id) DO UPDATE SET data=EXCLUDED.data, updated_at=now()",
                (worker.id, json.dumps(worker.model_dump(mode="json"), default=str)),
            )
        return worker

    def get_worker(self, worker_id: str) -> Worker | None:
        value = self._get_json("embodiedlab_workers", worker_id)
        return Worker.model_validate(value) if value else None

    def list_workers(self) -> list[Worker]:
        return [Worker.model_validate(item) for item in self._list_json("embodiedlab_workers")]

    def put_resource(self, kind: str, resource_id: str, value: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(value)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO embodiedlab_resources (kind, id, data) VALUES (%s, %s, %s::jsonb) "
                "ON CONFLICT (kind, id) DO UPDATE SET data=EXCLUDED.data, updated_at=now()",
                (kind, resource_id, payload),
            )
        return value

    def get_resource(self, kind: str, resource_id: str) -> dict[str, Any] | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT data FROM embodiedlab_resources WHERE kind=%s AND id=%s", (kind, resource_id)
            )
            row = cursor.fetchone()
        return row[0] if row else None

    def list_resources(self, kind: str, project_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT data FROM embodiedlab_resources WHERE kind=%s ORDER BY created_at DESC", (kind,)
            )
            values = [row[0] for row in cursor.fetchall()]
        return [
            item for item in values if project_id is None or item.get("project_id") == project_id
        ]

    def append_log(self, operation_id: str, stream: str, message: str) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO embodiedlab_operation_logs "
                "(operation_id, stream, message) VALUES (%s, %s, %s)",
                (operation_id, stream, message),
            )

    def list_logs(self, operation_id: str, after: int = 0) -> list[dict[str, Any]]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT sequence, stream, message, created_at FROM embodiedlab_operation_logs "
                "WHERE operation_id=%s AND sequence>%s ORDER BY sequence",
                (operation_id, after),
            )
            return [
                {"offset": row[0], "stream": row[1], "message": row[2], "created_at": row[3]}
                for row in cursor.fetchall()
            ]

    def append_operation_event(
        self, event: OperationProgressEvent
    ) -> tuple[OperationProgressEvent, bool]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM embodiedlab_operations WHERE id=%s FOR UPDATE",
                (event.operation_id,),
            )
            if cursor.fetchone() is None:
                raise OperationEventConflict("operation does not exist")
            cursor.execute(
                "SELECT data FROM embodiedlab_operation_events WHERE event_id=%s",
                (event.event_id,),
            )
            row = cursor.fetchone()
            if row is not None:
                existing = OperationProgressEvent.model_validate(row[0])
                if not _same_client_event(existing, event):
                    raise OperationEventConflict("event_id already contains different data")
                return existing, False
            cursor.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM embodiedlab_operation_events "
                "WHERE operation_id=%s",
                (event.operation_id,),
            )
            expected = cursor.fetchone()[0] + 1
            if event.sequence != expected:
                raise OperationEventConflict(
                    f"operation event sequence gap: expected {expected}, got {event.sequence}"
                )
            try:
                cursor.execute(
                    "INSERT INTO embodiedlab_operation_events "
                    "(operation_id, sequence, event_id, data) VALUES (%s, %s, %s, %s::jsonb)",
                    (
                        event.operation_id,
                        event.sequence,
                        event.event_id,
                        json.dumps(event.model_dump(mode="json")),
                    ),
                )
            except self._psycopg.errors.UniqueViolation as exc:
                raise OperationEventConflict("operation event identity already exists") from exc
        return event, True

    def list_operation_events(
        self, operation_id: str, after: int = 0, limit: int = 100
    ) -> list[OperationProgressEvent]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT data FROM embodiedlab_operation_events "
                "WHERE operation_id=%s AND sequence>%s ORDER BY sequence LIMIT %s",
                (operation_id, after, limit),
            )
            return [OperationProgressEvent.model_validate(row[0]) for row in cursor.fetchall()]

    def audit_orphans(self) -> dict[str, int]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM embodiedlab_operation_events event "
                "LEFT JOIN embodiedlab_operations operation ON operation.id=event.operation_id "
                "WHERE operation.id IS NULL"
            )
            return {"operation_event.operation": int(cursor.fetchone()[0])}


def _required_role(kind: str) -> WorkerRole | None:
    if kind.startswith(("env.", "tool.", "evaluate.")):
        return WorkerRole.SIMULATION
    if kind.startswith("service."):
        return WorkerRole.MODEL
    return None


def store_from_url(database_url: str | None, *, dev_memory: bool = False) -> Store:
    if dev_memory:
        return InMemoryStore()
    if not database_url or database_url.startswith("memory://"):
        raise RuntimeError(
            "EMBODIEDLAB_DATABASE_URL is required; use --dev-memory only for local development"
        )
    return PostgresStore(database_url)

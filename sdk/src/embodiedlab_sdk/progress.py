"""Bounded structured progress reporting for Worker-owned Operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import httpx

from .models import OperationPhase, ProgressEventType, utcnow


class OperationProgressReporter:
    def __init__(
        self,
        client: httpx.Client,
        operation_id: str,
        worker_id: str,
        agent_generation: str,
        *,
        attempt: int = 1,
        trace_id: str | None = None,
        spool_dir: str | Path | None = None,
        max_pending_events: int = 256,
    ) -> None:
        if max_pending_events < 1:
            raise ValueError("max_pending_events must be positive")
        self._client = client
        self.operation_id = operation_id
        self.worker_id = worker_id
        self.agent_generation = agent_generation
        self.attempt = attempt
        self.trace_id = trace_id
        self._max_pending_events = max_pending_events
        self._spool_path = (
            Path(spool_dir).expanduser() / f"{operation_id}.json" if spool_dir else None
        )
        self._pending = self._load_spool()
        self._sequence: int | None = None
        self._lock = Lock()

    def _load_spool(self) -> list[dict[str, Any]]:
        if self._spool_path is None or not self._spool_path.exists():
            return []
        try:
            value = json.loads(self._spool_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("progress spool is unreadable") from exc
        if not isinstance(value, dict) or value.get("operation_id") != self.operation_id:
            raise RuntimeError("progress spool operation mismatch")
        pending = value.get("pending")
        if not isinstance(pending, list) or any(not isinstance(item, dict) for item in pending):
            raise RuntimeError("progress spool payload is invalid")
        if len(pending) > self._max_pending_events:
            raise RuntimeError("progress spool exceeds configured bound")
        return pending

    def _persist_spool(self) -> None:
        if self._spool_path is None:
            return
        if not self._pending:
            self._spool_path.unlink(missing_ok=True)
            return
        self._spool_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self._spool_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"operation_id": self.operation_id, "pending": self._pending},
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(self._spool_path)

    def _load_sequence(self) -> int:
        after = 0
        while True:
            response = self._client.get(
                f"/v1/operations/{self.operation_id}/events",
                params={"after": after, "limit": 1000},
            )
            response.raise_for_status()
            entries = response.json().get("entries", [])
            if not entries:
                return after
            after = max(int(entry["sequence"]) for entry in entries)
            if len(entries) < 1000:
                return after

    def _emit(self, event_type: ProgressEventType, **values: Any) -> dict[str, Any]:
        with self._lock:
            if self._sequence is None:
                self._sequence = self._load_sequence()
                self._pending = [
                    item for item in self._pending if int(item.get("sequence", 0)) > self._sequence
                ]
                for item in self._pending:
                    item["worker_id"] = self.worker_id
                    item["agent_generation"] = self.agent_generation
                    item["event_id"] = f"event_{uuid4().hex}"
                self._persist_spool()
                self._flush_locked()
                if self._pending:
                    self._sequence = max(
                        self._sequence,
                        *(int(item["sequence"]) for item in self._pending),
                    )
            if len(self._pending) >= self._max_pending_events:
                raise RuntimeError("progress spool is full")
            payload = {
                "event_id": f"event_{uuid4().hex}",
                "sequence": self._sequence + 1,
                "event_type": event_type.value,
                "worker_id": self.worker_id,
                "agent_generation": self.agent_generation,
                "attempt": self.attempt,
                "trace_id": self.trace_id,
                "occurred_at": utcnow().isoformat(),
                **values,
            }
            self._pending.append(payload)
            self._sequence = int(payload["sequence"])
            self._persist_spool()
            delivered = self._flush_locked()
            self._sequence = max(self._sequence, int(payload["sequence"]))
            return {**payload, "delivery": "delivered" if delivered else "spooled"}

    def flush(self) -> bool:
        with self._lock:
            if not self._pending:
                return True
            if self._sequence is None:
                self._sequence = self._load_sequence()
            return self._flush_locked()

    def _flush_locked(self) -> bool:
        while self._pending:
            payload = self._pending[0]
            response = None
            for retry in range(2):
                try:
                    response = self._client.post(
                        f"/internal/v1/operations/{self.operation_id}/events",
                        json=payload,
                    )
                    response.raise_for_status()
                    break
                except httpx.TransportError:
                    if retry == 1:
                        return False
            assert response is not None
            self._sequence = int(response.json()["sequence"])
            self._pending.pop(0)
            self._persist_spool()
        return True

    def phase(self, phase: OperationPhase | str, message: str | None = None) -> dict[str, Any]:
        value = phase.value if isinstance(phase, OperationPhase) else phase
        return self._emit(ProgressEventType.PHASE, phase=value, message=message)

    def report(
        self,
        *,
        current: int,
        total: int | None = None,
        unit: str | None = None,
        phase: OperationPhase | str | None = None,
        message: str | None = None,
        metrics: dict[str, float] | None = None,
        episode_id: str | None = None,
        episode_index: int | None = None,
        step_index: int | None = None,
        sample_count: int | None = None,
    ) -> dict[str, Any]:
        phase_value = phase.value if isinstance(phase, OperationPhase) else phase
        return self._emit(
            ProgressEventType.PROGRESS,
            phase=phase_value,
            current=current,
            total=total,
            unit=unit,
            message=message,
            metrics=metrics or {},
            episode_id=episode_id,
            episode_index=episode_index,
            step_index=step_index,
            sample_count=sample_count,
        )

    def checkpoint(self, artifact_ref: str, message: str | None = None) -> dict[str, Any]:
        return self._emit(
            ProgressEventType.CHECKPOINT,
            checkpoint_ref=artifact_ref,
            message=message,
        )


@dataclass(frozen=True)
class ExecutionContext:
    progress: OperationProgressReporter

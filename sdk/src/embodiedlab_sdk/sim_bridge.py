"""Optional, failure-isolated heartbeat bridge from one SDK Worker to Sim."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SimHeartbeatClient:
    def __init__(
        self,
        base_url: str,
        sim_worker_id: str,
        api_key: str,
        registration_key: str,
        sdk_worker_id: Callable[[], str | None],
        environment_ids: Callable[[], list[str]],
        capacity: dict[str, Any],
        agent_generation: str,
        *,
        interval: float = 5.0,
        timeout: float = 3.0,
    ) -> None:
        if interval <= 0:
            raise ValueError("Sim heartbeat interval must be positive")
        self._sim_worker_id = sim_worker_id
        self._sdk_worker_id = sdk_worker_id
        self._environment_ids = environment_ids
        self._capacity = dict(capacity)
        self._agent_generation = agent_generation
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-EmbodiedLab-Worker-Registration-Key": registration_key,
            },
        )

    def heartbeat_once(self) -> None:
        sdk_worker_id = self._sdk_worker_id()
        if not sdk_worker_id:
            return
        environment_ids = sorted(set(self._environment_ids()))
        response = self._http.post(
            f"/internal/v1/workers/{self._sim_worker_id}/heartbeats",
            json={
                "sdk_worker_id": sdk_worker_id,
                "agent_generation": self._agent_generation,
                "state": "ready",
                "capacity": self._capacity,
                "environment_ids": environment_ids,
            },
        )
        response.raise_for_status()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.heartbeat_once()
            except Exception as exc:
                logger.warning(
                    "sim_worker_heartbeat_failed", extra={"error_type": type(exc).__name__}
                )
            self._stop.wait(self._interval)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="embodiedlab-sim-heartbeat", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval + 0.5))
        self._http.close()

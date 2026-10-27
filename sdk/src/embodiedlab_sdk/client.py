"""Public synchronous Python SDK for the EmbodiedLab Controller API."""

from __future__ import annotations

import time
from typing import Any

import httpx

from .models import TERMINAL_OPERATION_STATUSES, Operation
from .tool_bundle import ToolBundleArtifact
from .workflow import WorkflowBundle


class EmbodiedLabAPIError(RuntimeError):
    pass


class EmbodiedLabClient:
    def __init__(
        self,
        controller_url: str = "http://127.0.0.1:8130",
        *,
        api_key: str | None = None,
        project_id: str = "default",
        role: str = "user",
        timeout: float = 30.0,
    ) -> None:
        headers = {"X-EmbodiedLab-Role": role}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.project_id = project_id
        self.http = httpx.Client(
            base_url=controller_url.rstrip("/"), headers=headers, timeout=timeout
        )

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> EmbodiedLabClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        # A GET only reads, so it is retried when the connection drops (the Controller closes idle
        # keep-alive connections; under host load a poll can hit one as it closes). Commands are never retried.
        attempts = 5 if method.upper() == "GET" else 1
        for attempt in range(attempts):
            try:
                response = self.http.request(method, path, **kwargs)
                break
            except httpx.TransportError:
                if attempt == attempts - 1:
                    raise
                time.sleep(min(8.0, 0.5 * 2**attempt))
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise EmbodiedLabAPIError(f"{response.status_code}: {detail}")
        if not response.content:
            return None
        return response.json()

    def ready(self) -> dict[str, Any]:
        return self.request("GET", "/ready")

    def version(self) -> dict[str, Any]:
        return self.request("GET", "/version")

    def capabilities(self) -> dict[str, Any]:
        return self.request("GET", "/capabilities")

    def list_workers(self) -> list[dict[str, Any]]:
        return self.request("GET", "/v1/workers")

    def list_environments(self, *, correlation_id: str | None = None) -> list[dict[str, Any]]:
        params = {"correlation_id": correlation_id} if correlation_id else None
        return self.request("GET", "/v1/environments", params=params)

    def operation(self, operation_id: str) -> Operation:
        return Operation.model_validate(self.request("GET", f"/v1/operations/{operation_id}"))

    def wait_operation(
        self, operation_id: str, *, timeout: float | None = None, interval: float = 0.5
    ) -> Operation:
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            operation = self.operation(operation_id)
            if operation.status in TERMINAL_OPERATION_STATUSES:
                return operation
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"operation {operation_id} did not finish within {timeout}s")
            time.sleep(interval)

    def cancel_operation(self, operation_id: str) -> Operation:
        return Operation.model_validate(
            self.request("POST", f"/v1/operations/{operation_id}/cancel")
        )

    def start_environments(self, **values: Any) -> dict[str, Any]:
        return self.request(
            "POST",
            "/v1/environments",
            json={"project_id": self.project_id, **values},
        )

    def upload_workflow(self, bundle: WorkflowBundle) -> dict[str, Any]:
        return self.request(
            "PUT",
            f"/v1/workflows/{bundle.digest}",
            content=bundle.content,
            headers={
                "Content-Type": "application/octet-stream",
                "X-EmbodiedLab-Project": self.project_id,
            },
        )

    def upload_tool_bundle(self, bundle: ToolBundleArtifact) -> dict[str, Any]:
        return self.request(
            "PUT",
            f"/v1/tool-bundles/{bundle.digest}",
            content=bundle.content,
            headers={
                "Content-Type": "application/gzip",
                "X-EmbodiedLab-Project": self.project_id,
            },
        )

    def environment_command(
        self, environment_id: str, command: str, arguments: dict[str, Any] | None = None
    ) -> Operation:
        return Operation.model_validate(
            self.request(
                "POST",
                f"/v1/environments/{environment_id}/{command}",
                json={"project_id": self.project_id, "arguments": arguments or {}},
            )
        )

    def attach_tools(self, environment_id: str, digest: str) -> Operation:
        return Operation.model_validate(
            self.request(
                "POST",
                f"/v1/environments/{environment_id}/tools/attach",
                json={"project_id": self.project_id, "digest": digest},
            )
        )

    def detach_tools(
        self, environment_id: str, *, digest: str | None = None, name: str | None = None
    ) -> Operation:
        selector = {key: value for key, value in (("digest", digest), ("name", name)) if value}
        return Operation.model_validate(
            self.request(
                "POST",
                f"/v1/environments/{environment_id}/tools/detach",
                json={"project_id": self.project_id, **selector},
            )
        )

    def set_tool_visibility(self, environment_id: str, name: str, *, visible: bool) -> Operation:
        return Operation.model_validate(
            self.request(
                "POST",
                f"/v1/environments/{environment_id}/tools/visibility",
                json={"project_id": self.project_id, "name": name, "visible": visible},
            )
        )

    def call_tool(self, environment_id: str, name: str, arguments: dict[str, Any]) -> Operation:
        return Operation.model_validate(
            self.request(
                "POST",
                "/v1/tools/call",
                json={
                    "project_id": self.project_id,
                    "environment_id": environment_id,
                    "name": name,
                    "arguments": arguments,
                },
            )
        )

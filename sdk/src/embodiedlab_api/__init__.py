"""embodiedlab_api: the method-agnostic client for one EmbodiedLab environment.

Every method (a coding agent, a ReAct loop, a custom harness) reaches the
benchmark through this module and nothing else.  It talks HTTP to the
Controller's public tool interface and needs only the Python standard library:

    from embodiedlab_api import connect
    env = connect                      # reads env.json in the workspace (or EMBODIEDLAB_* variables)
    spec = env.task                    # task card, public parameters, tools, submission schema, budget
    for t in env.tools: print(t["name"], t["signature"])
    obs = env.observe                  # three RGB views; obs["images"] are PNG paths in the workspace
    state = env.call("get_robot_state")
    env.submit(estimate=..., unit=..., ...)   # once; the first submission closes the episode

Every call is appended to <workspace>/calls.jsonl and every image the tool
returns is written to <workspace>/images/ (the returned dict carries their
paths under "images").  Hidden values never travel through this module: it only
talks to the Controller, which only forwards public tool calls.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

__all__ = ["Env", "ToolError", "APIError", "connect"]
__version__ = "0.1.0"

DEFAULT_CONTROLLER = "http://127.0.0.1:18731"
TERMINAL = {"completed", "failed", "interrupted", "cancelled"}


class APIError(RuntimeError):
    """The Controller refused or could not complete a request (transport or HTTP error)."""


class ToolError(RuntimeError):
    """A public tool refused the call or failed; the message is the tool's own error."""


class _Controller:
    """Minimal HTTP client for the Controller's public endpoints (no third-party packages)."""

    def __init__(self, url: str, *, project_id: str = "default", api_key: str | None = None, timeout: float = 60):
        self.url = url.rstrip("/")
        self.project_id = project_id
        self.headers = {"X-EmbodiedLab-Role": "user", "Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.timeout = timeout

    def request(self, method: str, path: str, *, params: dict | None = None, body: dict | None = None) -> Any:
        url = self.url + path + (("?" + urllib.parse.urlencode(params)) if params else "")
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method, headers=self.headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except ValueError:
                pass
            raise APIError(f"{exc.code} {method} {path}: {detail}") from None
        except urllib.error.URLError as exc:
            raise APIError(f"cannot reach the Controller at {self.url}: {exc.reason}") from None
        return json.loads(raw) if raw else None

    def call_tool(self, environment_id: str, name: str, arguments: dict) -> dict:
        return self.request("POST", "/v1/tools/call", body={"project_id": self.project_id, "environment_id": environment_id,
                                                              "name": name, "arguments": arguments})

    def wait(self, operation_id: str, timeout: float) -> dict:
        deadline = time.time() + timeout
        delay = 0.3
        while True:
            operation = self.request("GET", f"/v1/operations/{operation_id}")
            if operation.get("status") in TERMINAL:
                return operation
            if time.time() > deadline:
                raise APIError(f"operation {operation_id} still {operation.get('status')} after {timeout:.0f} s")
            time.sleep(delay)
            delay = min(delay * 1.5, 3.0)


def _find_env_json(start: Path) -> Path | None:
    for directory in (start, *start.parents):
        candidate = directory / "env.json"
        if candidate.is_file():
            return candidate
    return None


def connect(environment_id: str | None = None, *, controller: str | None = None,
            workspace: str | os.PathLike | None = None, timeout: float = 1800) -> "Env":
    """Open the environment of the current trial.

    Resolution order: explicit arguments, then EMBODIEDLAB_ENV_ID /
    EMBODIEDLAB_CONTROLLER_URL / EMBODIEDLAB_WORKSPACE, then an env.json found
    in the current directory or one of its parents (the trial runner writes it).
    """
    environment_id = environment_id or os.environ.get("EMBODIEDLAB_ENV_ID")
    controller = controller or os.environ.get("EMBODIEDLAB_CONTROLLER_URL")
    workspace = workspace or os.environ.get("EMBODIEDLAB_WORKSPACE")
    if not (environment_id and controller and workspace):
        found = _find_env_json(Path(workspace) if workspace else Path.cwd())
        if found is None:
            raise RuntimeError("no environment: pass environment_id or run inside a trial workspace (env.json)")
        record = json.loads(found.read_text())
        environment_id = environment_id or record["environment_id"]
        controller = controller or record.get("controller") or DEFAULT_CONTROLLER
        workspace = workspace or str(found.parent)
    return Env(environment_id, controller=controller, workspace=workspace, timeout=timeout)


class Env:
    """One environment's public tools, with a call log and image files in the workspace."""

    def __init__(self, environment_id: str, *, controller: str = DEFAULT_CONTROLLER,
                 workspace: str | os.PathLike = ".", timeout: float = 1800):
        self.environment_id = environment_id
        self.controller = controller
        self.workspace = Path(workspace)
        self.images_dir = self.workspace / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._api = _Controller(controller, api_key=os.environ.get("EMBODIEDLAB_API_KEY"))
        self._log_path = self.workspace / "calls.jsonl"
        self.count = self._count_logged()
        self.last_images: list[str] = []

    # ---- discovery -------------------------------------------------------
    def tools(self) -> list[dict[str, Any]]:
        """The tools this environment exposes right now: name, signature, description."""
        tools = self._api.request("GET", "/v1/tools", params={"environment_id": self.environment_id})
        return [{k: t.get(k) for k in ("name", "signature", "description", "effect")} for t in tools]

    def task(self) -> dict[str, Any]:
        """The public task specification (task card, parameters, tools, submission schema, budget)."""
        return self.call("get_task_spec")

    # ---- the common tools ------------------------------------------------
    def observe(self) -> dict[str, Any]:
        """Three calibrated RGB views of the current state (calibration in get_task_spec's
        device_calibration.cameras; depth only through the T1 perception tools); images are saved as PNG files."""
        return self.call("get_observation")

    def state(self) -> dict[str, Any]:
        return self.call("get_robot_state")

    def state_id(self) -> str:
        return self.state()["state_id"]

    def wait(self, seconds: float) -> dict[str, Any]:
        return self.call("wait", seconds=seconds)

    def submit(self, **fields: Any) -> dict[str, Any]:
        """Submit the answer. Allowed once: the first submission closes the episode, valid or not."""
        return self.call("submit_result", **fields)

    # ---- the generic call ------------------------------------------------
    def call(self, name: str, /, **arguments: Any) -> Any:
        """Call one public tool by name with keyword arguments; returns the tool's result.

        Images returned by the tool are written to the workspace and listed under
        result["images"] (paths); the raw base64 never reaches the caller.
        """
        started = time.time()
        self.count += 1
        label = f"{self.count:04d}-{name}"
        operation = self._api.call_tool(self.environment_id, name, dict(arguments))
        operation = self._api.wait(operation["id"], self.timeout)
        output = operation.get("output") or {}
        paths: list[str] = []
        for index, item in enumerate(output.get("tool_images") or []):
            if isinstance(item, dict) and item.get("type") == "image" and item.get("data"):
                path = self.images_dir / f"{label}-{index}.png"
                path.write_bytes(base64.b64decode(item["data"]))
                paths.append(str(path))
        result = _replace_inline_images(output.get("result"), self.images_dir, label, paths)
        self.last_images = paths
        ok = operation.get("status") == "completed" and output.get("ok") is not False
        error = output.get("error") if output.get("ok") is False else operation.get("error")
        with self._log_path.open("a") as log:
            log.write(json.dumps({"label": label, "tool": name, "arguments": arguments, "ok": ok, "error": error,
                                  "state_id": output.get("state_id"), "images": paths,
                                  "wall_s": round(time.time() - started, 2), "result": _summarize(result)},
                                 default=str) + "\n")
        if not ok:
            message = error.get("message") if isinstance(error, dict) else error
            raise ToolError(f"{name}: {message or output}")
        if isinstance(result, dict) and paths and "images" not in result:
            result = dict(result, images=paths)
        return result

    def close(self) -> None:
        pass

    def __enter__(self) -> "Env":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _count_logged(self) -> int:
        if not self._log_path.is_file():
            return 0
        with self._log_path.open() as handle:
            return sum(1 for line in handle if line.strip())


def _replace_inline_images(value: Any, directory: Path, label: str, paths: list[str]) -> Any:
    """Some tools embed base64 images inside their result; write them out and keep the path."""
    if isinstance(value, list):
        return [_replace_inline_images(v, directory, label, paths) for v in value]
    if not isinstance(value, dict):
        return value
    data = value.get("data")
    if value.get("type") == "image" and isinstance(data, str) and len(data) > 256:
        path = directory / f"{label}-{len(paths)}.png"
        path.write_bytes(base64.b64decode(data))
        paths.append(str(path))
        return {k: v for k, v in value.items() if k != "data"} | {"file": str(path)}
    return {k: _replace_inline_images(v, directory, label, paths) for k, v in value.items()}



def _summarize(value: Any, limit: int = 2000) -> Any:
    text = json.dumps(value, default=str)
    return value if len(text) <= limit else {"_truncated": text[:limit]}

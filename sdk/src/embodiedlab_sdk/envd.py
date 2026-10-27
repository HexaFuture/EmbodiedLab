"""Persistent code-execution daemon running inside an environment container."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import importlib
import inspect
import io
import json
import math
import os
import struct
import sys
import traceback
import zlib
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import RLock
from types import ModuleType
from typing import Any
from uuid import uuid4

import cloudpickle
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .dynamic_tools import (
    DEFAULT_TOOL_BUNDLE_ROOT,
    AttachedBundle,
    DynamicToolError,
    attach_bundle,
    catalog_tool_names,
    detach_bundle,
)
from .tool_guard import ToolGuardError


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        if (
            getattr(value, "ndim", 0) in {2, 3}
            and str(getattr(value, "dtype", "")) == "uint8"
            and getattr(value, "size", 0) > 4096
        ):
            try:
                from PIL import Image

                image_buffer = io.BytesIO()
                Image.fromarray(value).save(image_buffer, format="PNG")
                return {
                    "type": "image",
                    "mime_type": "image/png",
                    "encoding": "base64",
                    "data": base64.b64encode(image_buffer.getvalue()).decode("ascii"),
                    "shape": list(value.shape),
                }
            except ImportError:
                pass
        converted = value.tolist()
        # Keep observations bounded; large arrays belong in artifact/image channels.
        if getattr(value, "size", 0) <= 4096:
            return converted
        return {
            "type": type(value).__name__,
            "shape": list(getattr(value, "shape", ())),
            "dtype": str(getattr(value, "dtype", "unknown")),
        }
    return {"type": type(value).__name__, "repr": repr(value)[:4096]}


def _merge_captured_stream(outer: str, inner: str) -> str:
    """Merge nested stream captures without returning the same bytes twice."""

    if not inner:
        return outer
    if not outer:
        return inner
    if inner in outer:
        return outer
    if outer in inner:
        return inner
    return outer + inner


class ExecRequest(BaseModel):
    code: str
    filename: str = "inline.py"


class RestoreRequest(BaseModel):
    state_id: str


class TruthQueryRequest(BaseModel):
    relation: str
    obj_a: str
    obj_b: str | None = None
    threshold: float | None = None


class WorkflowRunRequest(BaseModel):
    workflow_dir: str | None = None
    task: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolAttachRequest(BaseModel):
    digest: str
    content_b64: str


class ToolDetachRequest(BaseModel):
    digest: str | None = None
    name: str | None = None


class ToolVisibilityRequest(BaseModel):
    name: str
    visible: bool


def _dynamic_tool_http_error(exc: DynamicToolError | ToolGuardError) -> HTTPException:
    code = getattr(exc, "code", "invalid_tool_bundle")
    status = 404 if code in {"bundle_not_attached", "unknown_tool"} else 422
    return HTTPException(status_code=status, detail=exc.as_detail())


class EnvironmentSession:
    def __init__(self, bootstrap: str | None = None) -> None:
        self.lock = RLock()
        self.namespace: dict[str, Any] = {
            "__name__": "__embodiedlab_environment__",
            "__embodiedlab_store_image": self.store_image,
        }
        self.snapshots: dict[str, bytes] = {}
        self.state_evaluations: dict[str, dict[str, Any]] = {}
        self.images: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
        self.last_result: Any = None
        self.last_error: dict[str, Any] | None = None
        self.registry_revision = 0
        self.attached_bundles: dict[str, AttachedBundle] = {}
        self.disabled_tools: set[str] = set()
        self.tool_bundle_root = Path(
            os.getenv("EMBODIEDLAB_ENVD_TOOL_BUNDLE_DIR", DEFAULT_TOOL_BUNDLE_ROOT)
        )
        if bootstrap:
            self._bootstrap(bootstrap)
        self.save_state()

    def store_image(self, image: Any, media_type: str = "image/png") -> str:
        """Keep one bounded Agent-readable image and return its envd URL."""

        if media_type != "image/png":
            raise ValueError("envd image cache currently accepts only image/png")
        if (
            getattr(image, "ndim", None) != 3
            or getattr(image, "shape", (0, 0, 0))[-1] < 3
            or str(getattr(image, "dtype", "")) != "uint8"
        ):
            raise ValueError("cached image must be an HxWx3+ uint8 array")
        height, width = int(image.shape[0]), int(image.shape[1])
        if height > 4096 or width > 4096:
            raise ValueError("cached image dimensions must not exceed 4096x4096")

        def png_chunk(kind: bytes, payload: bytes) -> bytes:
            body = kind + payload
            return (
                struct.pack(">I", len(payload))
                + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
            )

        raw = b"".join(
            b"\x00" + image[row, :, :3].tobytes() for row in range(height)
        )
        data = (
            b"\x89PNG\r\n\x1a\n"
            + png_chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
            )
            + png_chunk(b"IDAT", zlib.compress(raw))
            + png_chunk(b"IEND", b"")
        )
        if len(data) > 8 * 1024 * 1024:
            raise ValueError("encoded cached image must not exceed 8 MiB")

        with self.lock:
            token = uuid4().hex
            self.images[token] = (data, media_type)
            while len(self.images) > 24:
                self.images.popitem(last=False)
        path = f"/candidate-images/{token}.png"
        return f"{os.getenv('EMBODIEDLAB_ENVD_URL', '').rstrip('/')}{path}"

    def image(self, token: str) -> tuple[bytes, str]:
        with self.lock:
            try:
                value = self.images[token]
            except KeyError as exc:
                raise KeyError(token) from exc
            self.images.move_to_end(token)
            return value

    def _bootstrap(self, spec: str) -> None:
        module_name, separator, function_name = spec.partition(":")
        if not separator:
            raise ValueError("EMBODIEDLAB_ENV_BOOTSTRAP must use module:function syntax")
        function = getattr(importlib.import_module(module_name), function_name)
        function(self.namespace)

    def _snapshot_namespace(self) -> bytes:
        values: dict[str, Any] = {}
        for key, value in self.namespace.items():
            if key.startswith("__") or isinstance(value, ModuleType) or callable(value):
                continue
            try:
                cloudpickle.dumps(value)
            except Exception:
                continue
            values[key] = value
        save_hook = self.namespace.get("embodiedlab_save_state")
        external = save_hook() if callable(save_hook) else None
        return cloudpickle.dumps(
            {
                "variables": values,
                "external": external,
                # Dynamic tool APIs live outside the namespace and survive a
                # restore; the snapshot records the registry so a restore can
                # be reconciled against what is currently attached.
                "registry": {
                    "revision": self.registry_revision,
                    "bundles": [bundle.digest for bundle in self.attached_bundles.values()],
                },
            }
        )

    def save_state(self) -> str:
        state_id = f"state_{uuid4().hex}"
        snapshot = self._snapshot_namespace()
        evaluation = self._evaluate_state()
        self.snapshots[state_id] = snapshot
        self.state_evaluations[state_id] = evaluation
        return state_id

    def _evaluate_state(self) -> dict[str, Any]:
        """Capture official evaluator values at the exact snapshot boundary."""

        hook = self.namespace.get("embodiedlab_evaluate_state")
        if not callable(hook):
            return {}
        value = hook()
        if not isinstance(value, Mapping):
            raise TypeError("embodiedlab_evaluate_state must return a mapping")
        task_completed = value.get("task_completed")
        if not isinstance(task_completed, bool):
            raise TypeError("embodiedlab_evaluate_state.task_completed must be a boolean")
        reward = value.get("reward")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
        ):
            raise TypeError("embodiedlab_evaluate_state.reward must be a finite number")
        sim_step = value.get("sim_step")
        if (
            isinstance(sim_step, bool)
            or not isinstance(sim_step, int)
            or sim_step < 0
        ):
            raise TypeError("embodiedlab_evaluate_state.sim_step must be a non-negative integer")
        evaluator = value.get("evaluator")
        if not isinstance(evaluator, str) or not evaluator.strip() or len(evaluator) > 128:
            raise TypeError("embodiedlab_evaluate_state.evaluator must be a bounded string")
        evaluation = {
            "reward": float(reward),
            "task_completed": task_completed,
            "sim_step": sim_step,
            "evaluator": evaluator,
        }
        report = value.get("scientific_evaluation")
        if report is not None:
            if not isinstance(report, Mapping):
                raise TypeError("scientific_evaluation must be a mapping")
            # Freeze a bounded JSON report with the exact state, not a mutable
            # reference to the evaluator's last result.
            encoded = json.dumps(dict(report), allow_nan=False)
            if len(encoded.encode("utf-8")) > 64 * 1024:
                raise ValueError("scientific_evaluation exceeds 64 KiB")
            evaluation["scientific_evaluation"] = json.loads(encoded)
        return evaluation

    def verify_state(self) -> dict[str, Any]:
        """Live, redacted stage predicates without requiring a snapshot.

        Consumes the bootstrap-injected ``embodiedlab_verify_state`` hook (RoboCasa
        stage verifier).  Intended for a trusted third-party verifier (the
        controller) to query stage-level subtask progress at any sim step; it
        exposes predicates/progress only, never object poses / qpos / contacts
        or hidden-camera truth.
        """
        hook = self.namespace.get("embodiedlab_verify_state")
        if not callable(hook):
            raise RuntimeError("environment does not expose a stage verifier")
        value = hook()
        if not isinstance(value, Mapping):
            raise TypeError("embodiedlab_verify_state must return a mapping")
        return dict(value)

    def evaluate_state(self) -> dict[str, Any]:
        """Live official evaluator result without requiring a snapshot.

        Mirrors:meth:`_evaluate_state` but reads the current simulator truth
        instead of a stored snapshot boundary, so a trusted third-party verifier
        can poll the official task completion and reward in real time.
        """
        hook = self.namespace.get("embodiedlab_evaluate_state")
        if not callable(hook):
            raise RuntimeError("environment does not expose the official evaluator")
        value = hook()
        if not isinstance(value, Mapping):
            raise TypeError("embodiedlab_evaluate_state must return a mapping")
        return dict(value)

    def truth_state(self) -> dict[str, Any]:
        """Live semantic truth (objects + per-object predicates + fixture state).

        Consumes the bootstrap-injected ``embodiedlab_truth_state`` hook -- a semantic
        view above the redacted stage predicates, without raw poses or contacts.
        """
        hook = self.namespace.get("embodiedlab_truth_state")
        if not callable(hook):
            raise RuntimeError("environment does not expose a semantic truth hook")
        value = hook()
        if not isinstance(value, Mapping):
            raise TypeError("embodiedlab_truth_state must return a mapping")
        return dict(value)

    def truth_query(self, value: TruthQueryRequest) -> bool:
        """Evaluate one semantic predicate for an object/object pair."""
        hook = self.namespace.get("embodiedlab_truth_query")
        if not callable(hook):
            raise RuntimeError("environment does not expose a semantic query hook")
        return bool(
            hook(value.relation, value.obj_a, value.obj_b, value.threshold)
        )

    def execute(self, code: str, filename: str) -> dict[str, Any]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with self.lock, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                hook = self.namespace.get("embodiedlab_execute_code")
                if callable(hook):
                    context = {
                        "TASK": self.namespace.get("TASK"),
                        "PARAMETERS": self.namespace.get("PARAMETERS", {}),
                        "INPUTS": self.namespace.get("INPUTS", {}),
                    }
                    if len(inspect.signature(hook).parameters) >= 3:
                        value = hook(code, filename, context)
                    else:
                        value = hook(code, filename)
                    if isinstance(value, Mapping):
                        if value.get("ok") is False:
                            self.last_error = dict(value.get("error") or {})
                            outer_stdout = stdout.getvalue()
                            outer_stderr = stderr.getvalue()
                            return {
                                "ok": False,
                                "stdout": _merge_captured_stream(
                                    outer_stdout, str(value.get("stdout", ""))
                                ),
                                "stderr": _merge_captured_stream(
                                    outer_stderr, str(value.get("stderr", ""))
                                ),
                                "error": self.last_error,
                                **{
                                    key: item
                                    for key, item in value.items()
                                    if key
                                    not in {"ok", "stdout", "stderr", "error", "result"}
                                },
                            }
                        self.last_result = value.get("result")
                        extra = {
                            key: item
                            for key, item in value.items()
                            if key not in {"ok", "stdout", "stderr", "error", "result"}
                        }
                        hook_stdout = str(value.get("stdout", ""))
                        hook_stderr = str(value.get("stderr", ""))
                    else:
                        self.last_result = value
                        extra = {}
                        hook_stdout = ""
                        hook_stderr = ""
                    self.namespace["RESULT"] = self.last_result
                else:
                    exec(compile(code, filename, "exec"), self.namespace, self.namespace)
                    self.last_result = self.namespace.get("RESULT")
                    extra = {}
                    hook_stdout = ""
                    hook_stderr = ""
                self.last_error = None
                state_id = self.save_state()
                outer_stdout = stdout.getvalue()
                outer_stderr = stderr.getvalue()
                return {
                    "ok": True,
                    "stdout": _merge_captured_stream(outer_stdout, hook_stdout),
                    "stderr": _merge_captured_stream(outer_stderr, hook_stderr),
                    "result": _json_value(self.last_result),
                    "state_id": state_id,
                    **_json_value(extra),
                }
            except BaseException as exc:
                self.last_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                return {
                    "ok": False,
                    "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue(),
                    "error": self.last_error,
                }

    def observe(self) -> dict[str, Any]:
        with self.lock:
            hook = self.namespace.get("embodiedlab_observe_with_images")
            if not callable(hook):
                hook = self.namespace.get("embodiedlab_observe")
            observation = hook() if callable(hook) else self.last_result
            tool_images: Any = None
            if isinstance(observation, dict) and "__embodiedlab_tool_result__" in observation:
                tool_images = observation.get("__embodiedlab_tool_images__")
                observation = observation["__embodiedlab_tool_result__"]
            payload = {
                "result": _json_value(observation),
                "error": self.last_error,
                "variables": sorted(key for key in self.namespace if not key.startswith("__")),
            }
            if tool_images:
                payload["tool_images"] = _json_value(tool_images)
            return payload

    def tools(self) -> Any:
        with self.lock:
            hook = self.namespace.get("embodiedlab_tools")
            if not callable(hook):
                return []
            catalog = _json_value(hook())
            if not isinstance(catalog, list):
                return catalog
            sources = {
                tool_name: bundle
                for bundle in self.attached_bundles.values()
                for tool_name in bundle.tool_names
            }
            visible: list[Any] = []
            for entry in catalog:
                if not isinstance(entry, dict):
                    visible.append(entry)
                    continue
                name = entry.get("name")
                if name in self.disabled_tools:
                    continue
                bundle = sources.get(name)
                if bundle is not None:
                    entry = {
                        **entry,
                        "source": "dynamic",
                        "bundle": bundle.name,
                        "digest": bundle.digest,
                    }
                visible.append(entry)
            return visible

    def tool_registry(self) -> dict[str, Any]:
        with self.lock:
            return {
                "tools": self.tools(),
                "registry_revision": self.registry_revision,
                "disabled": sorted(self.disabled_tools),
                "bundles": [bundle.metadata() for bundle in self.attached_bundles.values()],
            }

    def capture_status(self) -> dict[str, Any]:
        with self.lock:
            hook = self.namespace.get("embodiedlab_capture_status")
            if not callable(hook):
                return {
                    "ok": True,
                    "supported": False,
                    "enabled": False,
                    "finalized": False,
                    "frame_count": 0,
                }
            return {"ok": True, **_json_value(hook())}

    def finalize_capture(self) -> dict[str, Any]:
        with self.lock:
            hook = self.namespace.get("embodiedlab_finalize_capture")
            if not callable(hook):
                return {
                    "ok": False,
                    "error": {
                        "type": "CaptureUnavailable",
                        "message": "environment does not provide video capture",
                    },
                }
            try:
                return {"ok": True, **_json_value(hook())}
            except BaseException as exc:
                return {
                    "ok": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }

    def attach_tool_bundle(self, digest: str, content_b64: str) -> dict[str, Any]:
        if self.namespace.get("EMBODIEDLAB_RESTRICTED_TOOLS"):
            raise ToolGuardError(
                "restricted_environment", "dynamic tools disabled by environment",
                filename="<bundle>", line=0,
            )
        with self.lock:
            bundle = attach_bundle(
                self.namespace,
                self.attached_bundles,
                digest=digest,
                content_b64=content_b64,
                unpack_root=self.tool_bundle_root,
            )
            self.registry_revision += 1
            return {
                "ok": True,
                "attached": bundle.metadata(),
                "registry_revision": self.registry_revision,
            }

    def detach_tool_bundle(
        self, *, digest: str | None = None, name: str | None = None
    ) -> dict[str, Any]:
        with self.lock:
            bundle = detach_bundle(
                self.namespace, self.attached_bundles, digest=digest, name=name
            )
            self.disabled_tools.difference_update(bundle.tool_names)
            self.registry_revision += 1
            return {
                "ok": True,
                "detached": bundle.metadata(),
                "registry_revision": self.registry_revision,
            }

    def set_tool_visibility(self, name: str, visible: bool) -> dict[str, Any]:
        with self.lock:
            if name not in catalog_tool_names(self.namespace):
                raise DynamicToolError(
                    "unknown_tool",
                    f"tool {name!r} does not exist in this environment",
                    name=name,
                )
            was_disabled = name in self.disabled_tools
            if visible:
                self.disabled_tools.discard(name)
            else:
                self.disabled_tools.add(name)
            if (name in self.disabled_tools) != was_disabled:
                self.registry_revision += 1
            return {
                "ok": True,
                "name": name,
                "visible": visible,
                "registry_revision": self.registry_revision,
                "disabled": sorted(self.disabled_tools),
            }

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            if name in self.disabled_tools:
                self.last_error = {
                    "type": "ToolDisabledError",
                    "code": "tool_disabled",
                    "message": f"tool {name!r} is disabled in this environment session",
                }
                return {"ok": False, "error": self.last_error}
            hook = self.namespace.get("embodiedlab_call_tool_with_images")
            if not callable(hook):
                hook = self.namespace.get("embodiedlab_call_tool")
            try:
                if callable(hook):
                    result = hook(name, dict(arguments))
                else:
                    function = self.namespace.get(name)
                    if not callable(function):
                        raise KeyError(name)
                    result = function(**dict(arguments))
                # The robot bootstrap uses a private wrapper to carry trusted
                # image bytes out of the capability callback without changing
                # the model-facing function's return value.  Unwrap it here
                # and expose the media as a sibling of ``result``.
                tool_images: Any = None
                if isinstance(result, dict) and "__embodiedlab_tool_result__" in result:
                    tool_images = result.get("__embodiedlab_tool_images__")
                    result = result["__embodiedlab_tool_result__"]
                self.last_result = result
                self.namespace["RESULT"] = result
                state_id = self.save_state()
                payload = {"ok": True, "result": _json_value(result), "state_id": state_id}
                if tool_images:
                    payload["tool_images"] = _json_value(tool_images)
                return payload
            except BaseException as exc:
                self.last_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                return {"ok": False, "error": self.last_error}

    def run_workflow(
        self,
        workflow_dir: str | None = None,
        *,
        task: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.namespace.get("EMBODIEDLAB_RESTRICTED_TOOLS"):
            raise ValueError("workflows disabled by this environment")
        configured = os.getenv("EMBODIEDLAB_WORKFLOW_DIR")
        root = Path(configured or workflow_dir or "").expanduser().resolve()
        if configured and workflow_dir and root != Path(configured).expanduser().resolve():
            raise ValueError("workflow_dir does not match the mounted Workflow")
        manifest_path = root / "workflow.yaml"
        if not root.is_dir() or not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("Workflow must contain a regular workflow.yaml")
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("workflow.yaml must contain an object")
        entrypoint_name = manifest.get("entrypoint", "entrypoint.py")
        if not isinstance(entrypoint_name, str) or not entrypoint_name:
            raise ValueError("workflow entrypoint must be a non-empty path")
        entrypoint = (root / entrypoint_name).resolve()
        if root not in entrypoint.parents or not entrypoint.is_file() or entrypoint.is_symlink():
            raise ValueError("workflow entrypoint must be a regular file inside the Workflow")
        merged_parameters: dict[str, Any] = {}
        parameter_name = manifest.get("parameters")
        if isinstance(parameter_name, str) and parameter_name:
            parameter_path = (root / parameter_name).resolve()
            if (
                root not in parameter_path.parents
                or not parameter_path.is_file()
                or parameter_path.is_symlink()
            ):
                raise ValueError("workflow parameters must be a regular file inside the Workflow")
            loaded = json.loads(parameter_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("workflow parameters must contain a JSON object")
            merged_parameters.update(loaded)
        merged_parameters.update(dict(parameters or {}))
        task_value = (
            task
            or self.namespace.get("TASK")
            or os.getenv("EMBODIEDLAB_TASK")
            or os.getenv("EMBODIEDLAB_TASK_ID")
            or manifest.get("task_id")
        )
        self.namespace.update(
            TASK=task_value,
            PARAMETERS=merged_parameters,
            INPUTS={"task": task_value, "parameters": merged_parameters},
            __file__=str(entrypoint),
        )
        code = entrypoint.read_text(encoding="utf-8")
        sys.path.insert(0, str(root))
        try:
            result = self.execute(code, str(entrypoint))
        finally:
            if sys.path and sys.path[0] == str(root):
                sys.path.pop(0)
        return {
            **result,
            "workflow": {
                "directory": str(root),
                "entrypoint": entrypoint_name,
                "task_id": manifest.get("task_id"),
            },
        }

    def states(self) -> list[dict[str, Any]]:
        with self.lock:
            return [
                {"state_id": state_id, **self.state_evaluations.get(state_id, {})}
                for state_id in self.snapshots
            ]

    def restore(self, state_id: str) -> dict[str, Any]:
        if self.namespace.get("EMBODIEDLAB_RESTRICTED_TOOLS"):
            raise ValueError("state restore disabled by this environment")
        with self.lock:
            if state_id not in self.snapshots:
                raise KeyError(state_id)
            snapshot = cloudpickle.loads(self.snapshots[state_id])
            values = snapshot["variables"]
            for key, current in list(self.namespace.items()):
                if key.startswith("__") or isinstance(current, ModuleType) or callable(current):
                    continue
                try:
                    cloudpickle.dumps(current)
                except Exception:
                    continue
                self.namespace.pop(key, None)
            self.namespace.update(values)
            restore_hook = self.namespace.get("embodiedlab_restore_state")
            if callable(restore_hook) and snapshot["external"] is not None:
                restore_hook(snapshot["external"])
            self.last_result = self.namespace.get("RESULT")
            self.last_error = None
            return {
                **self.observe(),
                "registry": {
                    "revision": self.registry_revision,
                    "bundles": [
                        bundle.digest for bundle in self.attached_bundles.values()
                    ],
                    "snapshot": snapshot.get("registry"),
                },
            }


def create_env_app(session: EnvironmentSession | None = None) -> FastAPI:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embodiedlab-environment")
    state = session or executor.submit(
        EnvironmentSession, os.getenv("EMBODIEDLAB_ENV_BOOTSTRAP")
    ).result()
    app = FastAPI(title="EmbodiedLab Environment Daemon", version="0.1.0")

    async def run_in_environment_thread(function: Any, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            partial(function, *args, **kwargs),
        )

    @app.on_event("shutdown")
    def shutdown_environment_thread() -> None:
        try:
            close = state.namespace.get("embodiedlab_close")
            if callable(close):
                executor.submit(close).result()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        metadata = await run_in_environment_thread(
            lambda: _json_value(state.namespace.get("EMBODIEDLAB_ENV_METADATA", {}))
        )
        return {
            "ready": True,
            "environment_id": os.getenv("EMBODIEDLAB_ENV_ID"),
            "metadata": metadata,
        }

    @app.get("/candidate-images/{token}.png")
    async def candidate_image(token: str) -> Response:
        try:
            data, media_type = await run_in_environment_thread(state.image, token)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="candidate image not found") from exc
        return Response(
            content=data,
            media_type=media_type,
            headers={"Cache-Control": "private, no-store"},
        )

    @app.post("/exec")
    async def execute(value: ExecRequest) -> dict[str, Any]:
        return await run_in_environment_thread(state.execute, value.code, value.filename)

    @app.post("/workflow/run")
    async def run_workflow(value: WorkflowRunRequest) -> dict[str, Any]:
        try:
            return await run_in_environment_thread(
                state.run_workflow,
                value.workflow_dir,
                task=value.task,
                parameters=value.parameters,
            )
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/observe")
    async def observe() -> dict[str, Any]:
        return await run_in_environment_thread(state.observe)

    @app.get("/tools")
    async def tools() -> dict[str, Any]:
        return await run_in_environment_thread(state.tool_registry)

    @app.get("/capture/status")
    async def capture_status() -> dict[str, Any]:
        return await run_in_environment_thread(state.capture_status)

    @app.post("/capture/finalize")
    async def finalize_capture() -> dict[str, Any]:
        return await run_in_environment_thread(state.finalize_capture)

    @app.post("/tools/call")
    async def call_tool(value: ToolCallRequest) -> dict[str, Any]:
        return await run_in_environment_thread(
            state.call_tool, value.name, value.arguments
        )

    @app.post("/tools/attach")
    async def attach_tools(value: ToolAttachRequest) -> dict[str, Any]:
        try:
            return await run_in_environment_thread(
                state.attach_tool_bundle, value.digest, value.content_b64
            )
        except (DynamicToolError, ToolGuardError) as exc:
            raise _dynamic_tool_http_error(exc) from exc

    @app.post("/tools/detach")
    async def detach_tools(value: ToolDetachRequest) -> dict[str, Any]:
        try:
            return await run_in_environment_thread(
                state.detach_tool_bundle, digest=value.digest, name=value.name
            )
        except DynamicToolError as exc:
            raise _dynamic_tool_http_error(exc) from exc

    @app.post("/tools/visibility")
    async def set_tool_visibility(value: ToolVisibilityRequest) -> dict[str, Any]:
        try:
            return await run_in_environment_thread(
                state.set_tool_visibility, value.name, value.visible
            )
        except DynamicToolError as exc:
            raise _dynamic_tool_http_error(exc) from exc

    @app.get("/states")
    async def states() -> dict[str, Any]:
        return {"states": await run_in_environment_thread(state.states)}

    @app.get("/verify")
    async def verify() -> dict[str, Any]:
        try:
            return await run_in_environment_thread(state.verify_state)
        except (RuntimeError, TypeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/evaluate")
    async def evaluate() -> dict[str, Any]:
        try:
            return await run_in_environment_thread(state.evaluate_state)
        except (RuntimeError, TypeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/truth")
    async def truth() -> dict[str, Any]:
        try:
            return await run_in_environment_thread(state.truth_state)
        except (RuntimeError, TypeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/truth")
    async def truth_query(value: TruthQueryRequest) -> dict[str, Any]:
        try:
            result = await run_in_environment_thread(state.truth_query, value)
        except (RuntimeError, TypeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"result": result}

    @app.post("/restore")
    async def restore(value: RestoreRequest) -> dict[str, Any]:
        try:
            return await run_in_environment_thread(state.restore, value.state_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"state {value.state_id} not found"
            ) from exc

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="embodiedlab-envd")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    uvicorn.run(create_env_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

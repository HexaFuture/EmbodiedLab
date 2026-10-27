"""Standard OCI runtime boundary used by simulation and model adapters."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ContainerRuntimeError(RuntimeError):
    pass


# ``--mount`` only parses key=value fields, so a bind mount is made writable by
# spelling out ``readonly=false``.  The positional forms one reaches for first
# ("rw", "readwrite") are both rejected by the daemon as invalid fields, and the
# failure surfaces only once the container is already being created.  Profiles
# therefore declare the readable name and this table does the translation.
_MOUNT_MODES = {
    "readonly": "readonly",
    "ro": "readonly",
    "readwrite": "readonly=false",
    "rw": "readonly=false",
}


def _mount_mode(mode: str) -> str:
    try:
        return _MOUNT_MODES[mode]
    except KeyError:
        raise ContainerRuntimeError(
            f"unsupported mount mode {mode!r}; expected one of {sorted(_MOUNT_MODES)}"
        ) from None


@dataclass(frozen=True, slots=True)
class ContainerSpec:
    name: str
    image: str
    command: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    mounts: list[tuple[str, str, str]] = field(default_factory=list)
    ports: list[tuple[int, int]] = field(default_factory=list)
    publish_host: str | None = None
    network: str | None = None
    shm_size: str | None = None
    gpu: str | None = None
    egl: bool = False


class OCIRuntime:
    def __init__(self, binary: str | None = None) -> None:
        configured = binary or os.getenv("EMBODIEDLAB_OCI_RUNTIME")
        self.binary = configured or shutil.which("docker") or shutil.which("podman") or ""
        if not self.binary:
            raise ContainerRuntimeError("Docker or Podman is required on Worker nodes")

    def _run(
        self,
        arguments: list[str],
        *,
        input_text: str | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [self.binary, *arguments],
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if check and result.returncode != 0:
            raise ContainerRuntimeError(
                f"OCI command failed ({result.returncode}): "
                f"stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
            )
        return result

    def start(self, spec: ContainerSpec) -> str:
        arguments = ["run", "--detach", "--name", spec.name, "--pull", "missing"]
        for key, value in sorted(spec.labels.items()):
            arguments.extend(["--label", f"{key}={value}"])
        for key, value in sorted(spec.environment.items()):
            # Values are passed directly, never expanded through a shell.
            arguments.extend(["--env", f"{key}={value}"])
        for host, container, mode in spec.mounts:
            source = str(Path(host).expanduser().resolve())
            arguments.extend(
                ["--mount", f"type=bind,src={source},dst={container},{_mount_mode(mode)}"]
            )
        for host_port, container_port in spec.ports:
            published = f"{host_port}:{container_port}"
            if spec.publish_host:
                published = f"{spec.publish_host}:{published}"
            arguments.extend(["--publish", published])
        if spec.network:
            arguments.extend(["--network", spec.network])
        if spec.shm_size:
            arguments.extend(["--shm-size", spec.shm_size])
        if spec.gpu is not None:
            gpu = _normalize_gpu_selector(spec.gpu)
            if Path(self.binary).name == "docker":
                arguments.extend(["--gpus", f"device={gpu}"])
            else:
                arguments.extend(["--device", f"nvidia.com/gpu={gpu}"])
            if spec.egl:
                arguments.extend(["--env", "NVIDIA_DRIVER_CAPABILITIES=all"])
        if spec.egl:
            arguments.extend(["--env", "MUJOCO_GL=egl", "--env", "PYOPENGL_PLATFORM=egl"])
        arguments.append(spec.image)
        arguments.extend(spec.command)
        return self._run(arguments).stdout.strip()

    def exec_python(
        self, container_id: str, code: str, *, timeout: float | None = None
    ) -> dict[str, Any]:
        result = self._run(
            ["exec", "--interactive", container_id, "python", "-"],
            input_text=code,
            timeout=timeout,
        )
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}

    def exec_command(
        self,
        container_id: str,
        command: list[str],
        *,
        input_text: str | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> dict[str, Any]:
        """Run ``command`` in the container.

        ``check=False`` returns the completed process even on a non-zero exit
        so callers speaking a structured stdout protocol (``embodiedlab-env-client``
        exits 2 with ``{"ok": false, "error": {...}}`` on stdout) can surface
        the payload instead of an opaque exec failure.
        """

        result = self._run(
            ["exec", "--interactive", container_id, *command],
            input_text=input_text,
            timeout=timeout,
            check=check,
        )
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}

    def stop(self, container_id: str, *, timeout: int = 15) -> None:
        timeout_flag = "--timeout" if Path(self.binary).name == "docker" else "--time"
        try:
            self._run(
                ["stop", timeout_flag, str(timeout), container_id],
                timeout=timeout + 60,
            )
        except ContainerRuntimeError as exc:
            if "no such container" in str(exc).lower():
                return
            raise
        try:
            self._run(["rm", container_id], timeout=10)
        except ContainerRuntimeError as exc:
            if "no such container" not in str(exc).lower():
                raise

    def list_containers(
        self,
        *,
        labels: dict[str, str] | None = None,
        label_keys: list[str] | None = None,
    ) -> list[str]:
        arguments = ["ps", "--all", "--quiet"]
        for key, value in sorted((labels or {}).items()):
            arguments.extend(["--filter", f"label={key}={value}"])
        for key in sorted(label_keys or []):
            arguments.extend(["--filter", f"label={key}"])
        return [
            item.strip()
            for item in self._run(arguments, timeout=10).stdout.splitlines()
            if item.strip()
        ]

    def inspect(self, container_id: str) -> dict[str, Any]:
        import json

        value = json.loads(self._run(["inspect", container_id]).stdout)
        if not value:
            raise ContainerRuntimeError(f"container {container_id} was not found")
        return value[0]


def _normalize_gpu_selector(value: str) -> str:
    """Accept SDK CLI selectors and Docker's ``device=<id>`` form."""
    normalized = value.strip()
    if normalized.startswith("device="):
        normalized = normalized.removeprefix("device=").strip()
    if not normalized:
        raise ContainerRuntimeError("GPU selector must not be empty")
    return normalized

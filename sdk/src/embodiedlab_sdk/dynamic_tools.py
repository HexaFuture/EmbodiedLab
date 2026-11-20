"""In-process dynamic tool bundles for the environment daemon.

Design (M2): dynamic tools register inside the envd process itself — no
separate runner process.  Atomicity comes from envd's existing single-threaded
executor: attach/detach run on the same thread as every tool call, so a
registry mutation can never race an in-flight call.

A bundle becomes globally visible by wrapping its registered tools into one
ApiBase-shaped object and handing it to the bootstrap's ``embodiedlab_attach_api``
hook, which stores it under ``environment._apis``.  The bootstrap's
``embodiedlab_tools``/``embodiedlab_call_tool`` hooks recompute from ``_apis`` on every
call, so the tools appear on every path (tool list, tool calls, env exec
function bindings, workflows) without extra plumbing.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib
import inspect
import io
import re
import shutil
import sys
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import unpack_artifact
from .tool_bundle import ToolBundleManifest, load_tool_bundle_manifest, validate_tool_bundle_digest
from .tool_guard import validate_tool_bundle_sources

DEFAULT_TOOL_BUNDLE_ROOT = "/opt/embodiedlab/tool-bundles"
MAX_TOOL_BUNDLE_BYTES = 8 * 1024 * 1024
DYNAMIC_API_PREFIX = "dyn:"

_TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.-]*$")

# Bundle authors declare effects in the manifest vocabulary ("read_only" is the
# default); the robot runtime validates tool effects against its own ToolEffect
# literals, so the registered effect is canonicalized once here and every
# tools path sees the same value.
_CANONICAL_EFFECTS = {
    "read_only": "read_env",
    "read_env": "read_env",
    "local_compute": "local_compute",
    "stateful_io": "stateful_io",
    "physical": "physical",
    "unknown": "unknown",
}
_AUTO_ALLOW_EFFECTS = frozenset({"read_env", "local_compute"})


class DynamicToolError(ValueError):
    """A dynamic tool operation failed, with a machine-readable code."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = {key: value for key, value in details.items() if value is not None}

    def as_detail(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), **self.details}


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    name: str
    function: Callable[..., Any]
    description: str
    effect: str


class DynamicBundleApi:
    """ApiBase-shaped adapter exposing one bundle's registered tools.

    Duck-typed on purpose: envd must run without the robot dependency stack,
    so this class implements the surface the robot runtime consumes from
    ``environment._apis`` values (``functions`` / ``tool_approval_policies`` /
    ``tool_effects`` / ``combined_doc``) without importing ``embodiedlab_sdk.robot``.
    """

    def __init__(
        self,
        *,
        bundle_name: str,
        version: str,
        digest: str,
        tools: Mapping[str, RegisteredTool],
    ) -> None:
        self.bundle_name = bundle_name
        self.version = version
        self.digest = digest
        self._tools = dict(tools)

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {name: tool.function for name, tool in self._tools.items()}

    def tool_approval_policies(self) -> dict[str, str]:
        return {
            name: "allow" if tool.effect in _AUTO_ALLOW_EFFECTS else "confirm"
            for name, tool in self._tools.items()
        }

    def tool_effects(self) -> dict[str, str]:
        return {name: tool.effect for name, tool in self._tools.items()}

    def combined_doc(self) -> str:
        lines: list[str] = []
        for name, tool in self._tools.items():
            try:
                signature = str(inspect.signature(tool.function))
            except (TypeError, ValueError):
                signature = "(...)"
            lines.append(f"{name}{signature}")
            if tool.description:
                lines.append("  Doc:")
                lines.extend(f"    {line}" for line in tool.description.splitlines())
            lines.append("")
        return "\n".join(lines).strip()


class ToolBundleContext:
    """Registration context handed to a bundle's ``register(context)`` entrypoint.

    During registration the context only records tools; ``observe`` and
    ``call_tool`` raise so a bundle cannot execute environment actions while
    being attached.  After registration completes, registered tools may keep
    the context and use those same methods at call time to observe the
    environment or compose existing tools.
    """

    def __init__(self, namespace: Mapping[str, Any], *, bundle_name: str) -> None:
        self._namespace = namespace
        self._bundle_name = bundle_name
        self._registering = True
        self._tools: dict[str, RegisteredTool] = {}

    def register_tool(
        self,
        name: str,
        function: Callable[..., Any],
        *,
        description: str | None = None,
        effect: str = "read_only",
    ) -> None:
        if not self._registering:
            raise DynamicToolError(
                "registration_closed",
                f"bundle {self._bundle_name!r} tried to register {name!r} after "
                "registration finished",
            )
        if not isinstance(name, str) or not _TOOL_NAME_PATTERN.fullmatch(name):
            raise DynamicToolError(
                "invalid_tool_name", f"invalid tool name: {name!r}", name=str(name)
            )
        if name in self._tools:
            raise DynamicToolError(
                "duplicate_tool_name",
                f"bundle {self._bundle_name!r} registered {name!r} twice",
                name=name,
            )
        if not callable(function):
            raise DynamicToolError(
                "invalid_tool_function", f"tool {name!r} is not callable", name=name
            )
        canonical = _CANONICAL_EFFECTS.get(effect)
        if canonical is None:
            raise DynamicToolError(
                "invalid_tool_effect",
                f"tool {name!r} declared unknown effect {effect!r}",
                name=name,
            )
        doc = description if description is not None else (inspect.getdoc(function) or "")
        self._tools[name] = RegisteredTool(name, function, str(doc), canonical)

    def observe(self) -> Any:
        self._require_call_time("observe")
        hook = self._namespace.get("embodiedlab_observe")
        if not callable(hook):
            raise DynamicToolError(
                "observe_unavailable", "environment does not expose embodiedlab_observe"
            )
        return hook()

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        self._require_call_time("call_tool")
        hook = self._namespace.get("embodiedlab_call_tool")
        if not callable(hook):
            raise DynamicToolError(
                "call_tool_unavailable", "environment does not expose embodiedlab_call_tool"
            )
        return hook(name, dict(arguments or {}))

    def _require_call_time(self, operation: str) -> None:
        if self._registering:
            raise DynamicToolError(
                "registration_side_effect",
                f"tool bundle registration must be side-effect free: {operation}() "
                "is only available at tool-call time",
            )

    def _finish_registration(self) -> dict[str, RegisteredTool]:
        self._registering = False
        return dict(self._tools)


@dataclass(slots=True)
class AttachedBundle:
    name: str
    version: str
    digest: str
    api_key: str
    root: Path
    entrypoint: str
    tool_names: list[str]
    sys_path_entries: list[str]
    module_names: list[str]

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "digest": self.digest,
            "api_key": self.api_key,
            "entrypoint": self.entrypoint,
            "tools": list(self.tool_names),
        }


def catalog_tool_names(namespace: Mapping[str, Any]) -> set[str]:
    """Return every tool name the environment currently advertises."""

    hook = namespace.get("embodiedlab_tools")
    if not callable(hook):
        return {
            key
            for key, value in namespace.items()
            if callable(value) and not key.startswith("__")
        }
    names: set[str] = set()
    for entry in hook() or []:
        if isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
            names.add(entry["name"])
    return names


def attach_bundle(
    namespace: MutableMapping[str, Any],
    attached: MutableMapping[str, AttachedBundle],
    *,
    digest: str,
    content_b64: str,
    unpack_root: str | Path = DEFAULT_TOOL_BUNDLE_ROOT,
) -> AttachedBundle:
    """Verify, unpack, screen, import, and register one Common ToolBundle."""

    attach_api = namespace.get("embodiedlab_attach_api")
    if not callable(attach_api) or not callable(namespace.get("embodiedlab_detach_api")):
        raise DynamicToolError(
            "dynamic_tools_unsupported",
            "environment bootstrap does not expose embodiedlab_attach_api/embodiedlab_detach_api",
        )
    hexadecimal = _validated_digest(digest)
    if digest in attached:
        raise DynamicToolError(
            "bundle_already_attached",
            f"tool bundle {digest} is already attached",
            digest=digest,
        )
    content = _decoded_content(content_b64)
    if len(content) > MAX_TOOL_BUNDLE_BYTES:
        raise DynamicToolError(
            "bundle_too_large",
            f"tool bundle exceeds {MAX_TOOL_BUNDLE_BYTES} bytes before base64",
            digest=digest,
        )
    actual = hashlib.sha256(content).hexdigest()
    if actual != hexadecimal:
        raise DynamicToolError(
            "digest_mismatch",
            f"tool bundle content hashes to sha256:{actual}, not {digest}",
            digest=digest,
        )
    root = _materialized(content, Path(unpack_root).expanduser() / hexadecimal, digest)
    manifest = _validated_manifest(root, digest)
    bundle_name = manifest.metadata.name
    if any(bundle.name == bundle_name for bundle in attached.values()):
        raise DynamicToolError(
            "bundle_name_conflict",
            f"a tool bundle named {bundle_name!r} is already attached",
            name=bundle_name,
        )
    validate_tool_bundle_sources(root)
    module_name, _, function_name = manifest.entrypoint.partition(":")
    sys_path_entries = [
        str(path) for path in (root / "src", root / "vendor") if path.is_dir()
    ]
    modules_before = set(sys.modules)
    sys.path[:0] = sys_path_entries
    try:
        importlib.invalidate_caches()
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise DynamicToolError(
                "entrypoint_import_failed",
                f"could not import bundle module {module_name!r}: "
                f"{type(exc).__name__}: {exc}",
                digest=digest,
            ) from exc
        if not _module_within(module, root):
            raise DynamicToolError(
                "entrypoint_module_conflict",
                f"module {module_name!r} resolves outside the bundle; another "
                "package already owns that module name",
                digest=digest,
            )
        register = getattr(module, function_name, None)
        if not callable(register):
            raise DynamicToolError(
                "entrypoint_missing",
                f"bundle entrypoint {manifest.entrypoint!r} is not callable",
                digest=digest,
            )
        context = ToolBundleContext(namespace, bundle_name=bundle_name)
        try:
            register(context)
        except DynamicToolError:
            raise
        except Exception as exc:
            raise DynamicToolError(
                "registration_failed",
                f"bundle registration raised {type(exc).__name__}: {exc}",
                digest=digest,
            ) from exc
        tools = context._finish_registration()
        if not tools:
            raise DynamicToolError(
                "no_tools_registered",
                "bundle registration did not register any tool",
                digest=digest,
            )
        conflicts = sorted(set(tools) & catalog_tool_names(namespace))
        if conflicts:
            raise DynamicToolError(
                "tool_name_conflict",
                "tool names already exist in this environment: " + ", ".join(conflicts),
                conflicts=conflicts,
            )
        api = DynamicBundleApi(
            bundle_name=bundle_name,
            version=manifest.metadata.version,
            digest=digest,
            tools=tools,
        )
        api_key = f"{DYNAMIC_API_PREFIX}{bundle_name}"
        try:
            attach_api(api_key, api)
        except Exception as exc:
            raise DynamicToolError(
                "bundle_name_conflict",
                f"environment rejected api key {api_key!r}: {exc}",
                name=bundle_name,
            ) from exc
    except BaseException:
        _remove_sys_path_entries(sys_path_entries)
        _purge_bundle_modules(root, modules_before)
        raise
    bundle = AttachedBundle(
        name=bundle_name,
        version=manifest.metadata.version,
        digest=digest,
        api_key=api_key,
        root=root,
        entrypoint=manifest.entrypoint,
        tool_names=sorted(tools),
        sys_path_entries=sys_path_entries,
        module_names=_bundle_module_names(root, modules_before),
    )
    attached[digest] = bundle
    return bundle


def detach_bundle(
    namespace: MutableMapping[str, Any],
    attached: MutableMapping[str, AttachedBundle],
    *,
    digest: str | None = None,
    name: str | None = None,
) -> AttachedBundle:
    """Unregister a bundle's API and unload its sys.path/module footprint."""

    if (digest is None) == (name is None):
        raise DynamicToolError(
            "invalid_detach_request", "detach requires exactly one of digest or name"
        )
    if digest is not None:
        bundle = attached.get(digest)
    else:
        bundle = next((item for item in attached.values() if item.name == name), None)
    if bundle is None:
        raise DynamicToolError(
            "bundle_not_attached",
            f"tool bundle {digest or name!r} is not attached",
            digest=digest,
            name=name,
        )
    detach_api = namespace.get("embodiedlab_detach_api")
    if not callable(detach_api):
        raise DynamicToolError(
            "dynamic_tools_unsupported",
            "environment bootstrap does not expose embodiedlab_detach_api",
        )
    detach_api(bundle.api_key)
    _remove_sys_path_entries(bundle.sys_path_entries)
    # Tools may import bundle modules lazily after attach, so purge by origin
    # rather than trusting the module list recorded at attach time.
    _purge_bundle_modules(bundle.root, frozenset())
    attached.pop(bundle.digest, None)
    return bundle


def _validated_digest(digest: str) -> str:
    try:
        return validate_tool_bundle_digest(digest)
    except ValueError as exc:
        raise DynamicToolError("invalid_digest", str(exc)) from exc


def _decoded_content(content_b64: str) -> bytes:
    try:
        return base64.b64decode(content_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise DynamicToolError(
            "invalid_content", f"content_b64 is not valid base64: {exc}"
        ) from exc


def _materialized(content: bytes, root: Path, digest: str) -> Path:
    if root.is_dir():
        # Content-addressed: the digest already proved this tree's content.
        return root
    try:
        return unpack_artifact(io.BytesIO(content), root, label="tool bundle")
    except Exception as exc:
        shutil.rmtree(root, ignore_errors=True)
        raise DynamicToolError(
            "invalid_archive", f"tool bundle archive is invalid: {exc}", digest=digest
        ) from exc


def _validated_manifest(root: Path, digest: str) -> ToolBundleManifest:
    try:
        manifest = load_tool_bundle_manifest(root)
    except ValueError as exc:
        raise DynamicToolError("invalid_manifest", str(exc), digest=digest) from exc
    if manifest.kind != "CommonToolBundle":
        raise DynamicToolError(
            "unsupported_bundle_kind",
            f"dynamic attach requires a CommonToolBundle, got {manifest.kind!r}",
            digest=digest,
        )
    return manifest


def _module_within(module: Any, root: Path) -> bool:
    file = getattr(module, "__file__", None)
    candidates = [file] if file else list(getattr(module, "__path__", None) or [])
    resolved_root = root.resolve()
    for candidate in candidates:
        try:
            if Path(candidate).resolve().is_relative_to(resolved_root):
                return True
        except (OSError, TypeError, ValueError):
            continue
    return False


def _bundle_module_names(root: Path, modules_before: frozenset[str] | set[str]) -> list[str]:
    return sorted(
        name
        for name, module in list(sys.modules.items())
        if name not in modules_before and _module_within(module, root)
    )


def _purge_bundle_modules(root: Path, modules_before: frozenset[str] | set[str]) -> None:
    for module_name in _bundle_module_names(root, modules_before):
        sys.modules.pop(module_name, None)


def _remove_sys_path_entries(entries: list[str]) -> None:
    for entry in entries:
        try:
            sys.path.remove(entry)
        except ValueError:
            continue


__all__ = [
    "DEFAULT_TOOL_BUNDLE_ROOT",
    "DYNAMIC_API_PREFIX",
    "MAX_TOOL_BUNDLE_BYTES",
    "AttachedBundle",
    "DynamicBundleApi",
    "DynamicToolError",
    "RegisteredTool",
    "ToolBundleContext",
    "attach_bundle",
    "catalog_tool_names",
    "detach_bundle",
]

"""AST screening for dynamic tool bundle sources.

The guard defends against self-deception, not malice: a Common ToolBundle must
not silently bypass the environment boundary by importing the simulator or the
robot runtime directly, and it must not smuggle code past the review by
building it at runtime.  Everything else is deliberately allowed — tools are
working code, not sandboxed model output.

``embodiedlab_sdk.robot`` is banned wholesale (including ``utils``): carving out
read-only submodules would require a per-module audit that M2 does not need,
and a bundle can vendor pure-Python helpers instead.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

MAX_TOOL_MODULE_BYTES = 512 * 1024

BANNED_IMPORT_MODULES = (
    "ctypes",
    "embodiedlab_sdk.robot",
    "mujoco",
    "robocasa",
    "robosuite",
)
BANNED_CALLS = frozenset({"eval", "exec", "compile", "__import__"})


class ToolGuardError(ValueError):
    """A tool module violates the bundle screening rules."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        filename: str,
        line: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.filename = filename
        self.line = line

    def as_detail(self) -> dict[str, Any]:
        detail: dict[str, Any] = {
            "code": self.code,
            "message": str(self),
            "filename": self.filename,
        }
        if self.line is not None:
            detail["line"] = self.line
        return detail


def _banned_module(name: str) -> str | None:
    for banned in BANNED_IMPORT_MODULES:
        if name == banned or name.startswith(banned + "."):
            return banned
    return None


def _imported_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level:
        # Bundle-relative imports resolve inside the bundle tree, which is
        # screened file by file anyway.
        return []
    module = node.module or ""
    names = [module] if module else []
    for alias in node.names:
        if alias.name == "*":
            continue
        names.append(f"{module}.{alias.name}" if module else alias.name)
    return names


def validate_tool_module(source: str, filename: str) -> None:
    """Reject a tool module that crosses the environment boundary."""

    if len(source.encode("utf-8")) > MAX_TOOL_MODULE_BYTES:
        raise ToolGuardError(
            "module_too_large",
            f"{filename} exceeds the {MAX_TOOL_MODULE_BYTES} byte tool module limit",
            filename=filename,
        )
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise ToolGuardError(
            "syntax_error",
            f"{filename} is not valid Python: {exc.msg}",
            filename=filename,
            line=exc.lineno,
        ) from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in _imported_names(node):
                banned = _banned_module(name)
                if banned is not None:
                    raise ToolGuardError(
                        "banned_import",
                        f"{filename} imports {name!r}; tool bundles must not import "
                        f"{banned!r} — call environment capabilities through the "
                        "registration context instead",
                        filename=filename,
                        line=node.lineno,
                    )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in BANNED_CALLS
        ):
            raise ToolGuardError(
                "banned_call",
                f"{filename} calls {node.func.id}(); tool bundles must not build "
                "code at runtime",
                filename=filename,
                line=node.lineno,
            )


def validate_tool_bundle_sources(root: str | Path) -> None:
    """Screen every Python module under the bundle's src/ and vendor/ trees.

    Only these two trees enter ``sys.path`` when a bundle is attached, so they
    are the only importable code and the only trees that need screening.
    """

    bundle_root = Path(root)
    for tree_name in ("src", "vendor"):
        tree_root = bundle_root / tree_name
        if not tree_root.is_dir():
            continue
        for path in sorted(tree_root.rglob("*.py")):
            relative = path.relative_to(bundle_root).as_posix()
            validate_tool_module(path.read_text(encoding="utf-8"), relative)


__all__ = [
    "BANNED_CALLS",
    "BANNED_IMPORT_MODULES",
    "MAX_TOOL_MODULE_BYTES",
    "ToolGuardError",
    "validate_tool_bundle_sources",
    "validate_tool_module",
]

"""Install version-matched Agent Skills released with the CLI."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .catalog import load_catalog

VERSION_FILE = ".embodiedlab-sdk-version"


def source_skills_dir() -> Path:
    configured = os.getenv("EMBODIEDLAB_SKILLS_DIR")
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[2] / "skills"
        if not root.is_dir():
            root = Path(sys.prefix) / "share" / "embodiedlab-sdk" / "skills"
    if not root.is_dir():
        raise RuntimeError(
            "bundled Skills were not found; set EMBODIEDLAB_SKILLS_DIR or use the "
            "official EmbodiedLab SDK installer"
        )
    return root


def target_skills_dir(target: str | None = None) -> Path:
    if target:
        return Path(target).expanduser().resolve()
    codex_home = Path(os.getenv("CODEX_HOME", Path.home() / ".codex"))
    return codex_home / "skills"


def install_skills(*, target: str | None = None, update: bool = False) -> list[dict[str, Any]]:
    source = source_skills_dir()
    destination = target_skills_dir(target)
    destination.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for item in load_catalog("skills"):
        name = item["name"]
        src = source / name
        dst = destination / name
        if not src.is_dir():
            raise RuntimeError(f"release catalog references missing Skill {name}")
        if dst.exists():
            existing_version = (
                (dst / VERSION_FILE).read_text().strip() if (dst / VERSION_FILE).is_file() else None
            )
            if existing_version == __version__ and not update:
                results.append(
                    {
                        "name": name,
                        "version": existing_version,
                        "status": "current",
                        "path": str(dst),
                    }
                )
                continue
            if not update:
                results.append(
                    {
                        "name": name,
                        "version": existing_version,
                        "status": "update-available",
                        "path": str(dst),
                    }
                )
                continue
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        (dst / VERSION_FILE).write_text(f"{__version__}\n", encoding="utf-8")
        results.append(
            {"name": name, "version": __version__, "status": "installed", "path": str(dst)}
        )
    return results


def skill_status(*, target: str | None = None) -> list[dict[str, Any]]:
    destination = target_skills_dir(target)
    values = []
    for item in load_catalog("skills"):
        path = destination / item["name"]
        installed = (
            (path / VERSION_FILE).read_text().strip() if (path / VERSION_FILE).is_file() else None
        )
        values.append(
            {
                "name": item["name"],
                "cli_version": __version__,
                "installed_version": installed,
                "status": "current"
                if installed == __version__
                else ("missing" if installed is None else "mismatch"),
                "path": str(path),
            }
        )
    return values

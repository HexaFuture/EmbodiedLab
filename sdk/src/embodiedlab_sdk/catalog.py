"""Load immutable release catalogs shipped with EmbodiedLab SDK."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def load_catalog(name: str) -> list[dict[str, Any]]:
    value = json.loads(files("embodiedlab_sdk").joinpath(f"data/{name}.json").read_text())
    if not isinstance(value, list):
        raise RuntimeError(f"catalog {name} must be a JSON array")
    return value


def service_spec(name: str) -> dict[str, Any]:
    for item in load_catalog("services"):
        if name == item["name"] or name in item.get("aliases", []):
            return item
    raise ValueError(f"unknown model service {name}")


def environment_profile(name: str) -> dict[str, Any]:
    for item in load_catalog("environment_profiles"):
        if name == item["name"] or name in item.get("aliases", []):
            return item
    raise ValueError(f"unknown environment profile {name}")


def capabilities() -> dict[str, Any]:
    return {
        "api_version": "v1",
        "placement": {
            "target_worker_id": True,
            "stable_worker_registration": True,
            "execution_correlation": True,
        },
        "services": load_catalog("services"),
        "environment_profiles": load_catalog("environment_profiles"),
        "robot_tools": load_catalog("robot_tools"),
        "skills": load_catalog("skills"),
    }

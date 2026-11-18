"""Manifest and packaging contract for Common ToolBundles."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .artifacts import ArtifactCache, package_directory, validate_digest

TOOL_BUNDLE_MANIFEST = "tool-bundle.yaml"
TOOL_BUNDLE_SCHEMA = "embodiedlab.tool_bundle/v1alpha1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BundleMetadata(StrictModel):
    name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=128)


class BundleCompatibility(StrictModel):
    sdk: str = Field(min_length=1)
    profiles: list[str] = Field(default_factory=list)
    python_packages: dict[str, str] = Field(default_factory=dict)


class ServiceRequirement(StrictModel):
    api: str = Field(min_length=1)
    version: str = Field(min_length=1)


class BundleRequirements(StrictModel):
    observations: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    services: dict[str, ServiceRequirement] = Field(default_factory=dict)


class ToolDefinition(StrictModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]*$")
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    effect: str = "read_only"
    protected: bool = False


class EndpointContractReference(StrictModel):
    observation_spec: str = Field(min_length=1)
    action_specs: list[str] = Field(default_factory=list)
    step_result_spec: str = Field(min_length=1)
    control_frequency_hz: float | None = Field(default=None, gt=0)


class BundleProvides(StrictModel):
    tools: list[ToolDefinition] = Field(default_factory=list)
    endpoint_contract: EndpointContractReference | None = None


class ToolReplacement(StrictModel):
    provider: str = Field(min_length=1)
    version: str | None = None
    digest: str | None = None

    @field_validator("digest")
    @classmethod
    def validate_replacement_digest(cls, value: str | None) -> str | None:
        if value is not None:
            validate_digest(value, label="replacement provider")
        return value


class ToolBundleManifest(StrictModel):
    schema_version: Literal[TOOL_BUNDLE_SCHEMA]
    kind: Literal["CommonToolBundle", "EnvironmentAdapter"]
    metadata: BundleMetadata
    entrypoint: str
    compatibility: BundleCompatibility
    requires: BundleRequirements = Field(default_factory=BundleRequirements)
    provides: BundleProvides = Field(default_factory=BundleProvides)
    replaces: list[ToolReplacement] = Field(default_factory=list)

    @field_validator("entrypoint")
    @classmethod
    def validate_entrypoint(cls, value: str) -> str:
        module, separator, function = value.partition(":")
        if not separator or not module or not function:
            raise ValueError("entrypoint must use module:function syntax")
        if any(not part.isidentifier() for part in module.split(".")):
            raise ValueError("entrypoint module must use Python identifiers")
        if not function.isidentifier():
            raise ValueError("entrypoint function must be a Python identifier")
        return value


@dataclass(frozen=True, slots=True)
class ToolBundleArtifact:
    digest: str
    content: bytes
    files: tuple[str, ...]
    manifest: ToolBundleManifest


def load_tool_bundle_manifest(directory: str | Path) -> ToolBundleManifest:
    root = Path(directory).expanduser()
    manifest_path = root / TOOL_BUNDLE_MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"tool bundle must contain a regular {TOOL_BUNDLE_MANIFEST}")
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read {TOOL_BUNDLE_MANIFEST}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{TOOL_BUNDLE_MANIFEST} must contain a YAML object")
    try:
        return ToolBundleManifest.model_validate(raw)
    except ValueError as exc:
        raise ValueError(f"invalid {TOOL_BUNDLE_MANIFEST}: {exc}") from exc


def package_tool_bundle(directory: str | Path) -> ToolBundleArtifact:
    manifest = load_tool_bundle_manifest(directory)
    artifact = package_directory(directory, label="tool bundle")
    return ToolBundleArtifact(artifact.digest, artifact.content, artifact.files, manifest)


def validate_tool_bundle_digest(digest: str) -> str:
    return validate_digest(digest, label="tool bundle")


class ToolBundleCache:
    def __init__(self, root: str | Path) -> None:
        self._cache = ArtifactCache(root, namespace="tool-bundles")

    @property
    def root(self) -> Path:
        return self._cache.root

    def path(self, digest: str) -> Path | None:
        return self._cache.path(digest, label="tool bundle")

    def materialize(self, digest: str, chunks: Iterable[bytes]) -> Path:
        return self._cache.materialize(digest, chunks, label="tool bundle")

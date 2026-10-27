"""Deterministic, content-addressed artifacts shared by workflows and bundles."""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import re
import shutil
import stat
import tarfile
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

IGNORED_PARTS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
MAX_ARTIFACT_MEMBERS = 10_000
MAX_ARTIFACT_FILE_SIZE = 8 * 1024**3
MAX_ARTIFACT_TOTAL_SIZE = 32 * 1024**3
_DIGEST_PATTERN = re.compile(r"sha256:([0-9a-f]{64})\Z")


@dataclass(frozen=True, slots=True)
class PackagedArtifact:
    digest: str
    content: bytes
    files: tuple[str, ...]


def validate_digest(digest: str, *, label: str = "artifact") -> str:
    """Return the hexadecimal digest after validating its canonical form."""

    match = _DIGEST_PATTERN.fullmatch(digest)
    if match is None:
        raise ValueError(f"{label} digest must use sha256:<64 lowercase hex characters>")
    return match.group(1)


def package_directory(directory: str | Path, *, label: str = "artifact") -> PackagedArtifact:
    """Create a deterministic gzip-tar artifact from a regular-file directory."""

    source_root = Path(directory).expanduser()
    if source_root.is_symlink():
        raise ValueError(f"{label} directory must not be a symlink: {source_root}")
    root = source_root.resolve()
    if not root.is_dir():
        raise ValueError(f"{label} directory does not exist: {root}")

    entries: list[tuple[str, os.stat_result]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} must not contain symlinks: {relative}")
        if not stat.S_ISREG(metadata.st_mode):
            continue
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.suffix in IGNORED_SUFFIXES:
            continue
        entries.append((relative.as_posix(), metadata))

    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for relative, metadata in entries:
                path = root.joinpath(*PurePosixPath(relative).parts)
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    current = os.fstat(descriptor)
                    if not stat.S_ISREG(current.st_mode) or (
                        current.st_dev,
                        current.st_ino,
                        current.st_size,
                    ) != (metadata.st_dev, metadata.st_ino, metadata.st_size):
                        raise ValueError(f"{label} file changed while packaging: {relative}")
                    info = tarfile.TarInfo(relative)
                    info.size = current.st_size
                    info.mode = stat.S_IMODE(current.st_mode)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with os.fdopen(descriptor, "rb") as handle:
                        descriptor = -1
                        archive.addfile(info, handle)
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
    content = buffer.getvalue()
    return PackagedArtifact(
        digest=f"sha256:{hashlib.sha256(content).hexdigest()}",
        content=content,
        files=tuple(relative for relative, _ in entries),
    )


def _safe_member_path(member: tarfile.TarInfo, *, label: str) -> PurePosixPath:
    path = PurePosixPath(member.name)
    if (
        not member.name
        or member.name.startswith("/")
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != member.name
    ):
        raise ValueError(f"unsafe {label} archive path: {member.name!r}")
    if not (member.isfile() or member.isdir()):
        raise ValueError(f"unsupported {label} archive member: {member.name!r}")
    return path


def unpack_artifact(
    source: BinaryIO,
    destination: str | Path,
    *,
    label: str = "artifact",
    max_members: int = MAX_ARTIFACT_MEMBERS,
    max_file_size: int = MAX_ARTIFACT_FILE_SIZE,
    max_total_size: int = MAX_ARTIFACT_TOTAL_SIZE,
) -> Path:
    """Validate and unpack a gzip-tar containing only canonical files/directories."""

    if min(max_members, max_file_size, max_total_size) <= 0:
        raise ValueError("artifact archive limits must be positive")
    target_root = Path(destination)
    target_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=source, mode="r:gz") as archive:
        members: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
        kinds: dict[PurePosixPath, str] = {}
        total_size = 0
        for member_number, member in enumerate(archive, start=1):
            if member_number > max_members:
                raise ValueError(f"{label} archive exceeds member limit ({max_members})")
            path = _safe_member_path(member, label=label)
            if member.size < 0:
                raise ValueError(f"negative {label} archive member size: {member.name!r}")
            if member.isfile():
                if member.size > max_file_size:
                    raise ValueError(
                        f"{label} archive member exceeds file size limit: {member.name!r}"
                    )
                total_size += member.size
                if total_size > max_total_size:
                    raise ValueError(
                        f"{label} archive exceeds total size limit ({max_total_size} bytes)"
                    )
            if path in kinds:
                raise ValueError(f"duplicate {label} archive path: {member.name!r}")
            for parent in path.parents:
                if parent != PurePosixPath(".") and kinds.get(parent) == "file":
                    raise ValueError(
                        f"{label} archive nests content below a file: {member.name!r}"
                    )
            if member.isfile() and any(existing.is_relative_to(path) for existing in kinds):
                raise ValueError(
                    f"{label} archive file conflicts with a directory: {member.name!r}"
                )
            kinds[path] = "file" if member.isfile() else "directory"
            members.append((member, path))

        for member, path in members:
            target = target_root.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                target.chmod(stat.S_IMODE(member.mode) & 0o777)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"{label} archive file has no content: {member.name!r}")
            with extracted, target.open("xb") as handle:
                shutil.copyfileobj(extracted, handle)
            target.chmod(stat.S_IMODE(member.mode) & 0o777)
    return target_root.resolve()


class ArtifactCache:
    """Atomically materialize verified artifacts by SHA-256 digest."""

    def __init__(self, root: str | Path, *, namespace: str | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.storage_root = self.root if namespace is None else self.root / namespace
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def path(self, digest: str, *, label: str = "artifact") -> Path | None:
        hexadecimal = validate_digest(digest, label=label)
        target = self.storage_root / hexadecimal
        if target.is_dir():
            return target
        if target.exists():
            raise ValueError(f"{label} cache target is not a directory: {target}")
        return None

    def materialize(
        self,
        digest: str,
        chunks: Iterable[bytes],
        *,
        label: str = "artifact",
        max_members: int = MAX_ARTIFACT_MEMBERS,
        max_file_size: int = MAX_ARTIFACT_FILE_SIZE,
        max_total_size: int = MAX_ARTIFACT_TOTAL_SIZE,
    ) -> Path:
        hexadecimal = validate_digest(digest, label=label)
        cached = self.path(digest, label=label)
        if cached is not None:
            return cached

        archive_path: Path | None = None
        staging: Path | None = None
        try:
            descriptor, temporary = tempfile.mkstemp(
                dir=self.storage_root,
                prefix=f".{hexadecimal}.",
                suffix=".tar.gz",
            )
            archive_path = Path(temporary)
            hasher = hashlib.sha256()
            with os.fdopen(descriptor, "wb") as handle:
                for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise TypeError("artifact download chunks must be bytes")
                    hasher.update(chunk)
                    handle.write(chunk)
            if hasher.hexdigest() != hexadecimal:
                raise ValueError(f"{label} content digest mismatch")

            staging = Path(
                tempfile.mkdtemp(dir=self.storage_root, prefix=f".{hexadecimal}.unpack.")
            )
            with archive_path.open("rb") as handle:
                unpack_artifact(
                    handle,
                    staging,
                    label=label,
                    max_members=max_members,
                    max_file_size=max_file_size,
                    max_total_size=max_total_size,
                )
            target = self.storage_root / hexadecimal
            try:
                staging.rename(target)
                staging = None
            except OSError:
                if not target.is_dir():
                    raise
            return target
        finally:
            if archive_path is not None:
                archive_path.unlink(missing_ok=True)
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)

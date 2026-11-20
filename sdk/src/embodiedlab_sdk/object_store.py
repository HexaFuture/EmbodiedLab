"""Artifact storage boundary: local disk for one node, S3/MinIO for distributed mode."""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO


class ObjectStore(ABC):
    @abstractmethod
    def put(self, key: str, source: BinaryIO) -> str: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def open(self, key: str) -> BinaryIO:
        """Open an object for incremental reading; the caller owns the reader."""

    def iter_bytes(self, key: str, *, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        reader = self.open(key)
        try:
            while chunk := reader.read(chunk_size):
                yield chunk
        finally:
            reader.close()


class LocalObjectStore(ObjectStore):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents:
            raise ValueError("object key escapes storage root")
        return path

    def put(self, key: str, source: BinaryIO) -> str:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                while chunk := source.read(1024 * 1024):
                    handle.write(chunk)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return f"file://{target}"

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def open(self, key: str) -> BinaryIO:
        return self._path(key).open("rb")


class S3ObjectStore(ObjectStore):
    def __init__(self, bucket: str, *, endpoint_url: str | None = None, prefix: str = "") -> None:
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = boto3.client("s3", endpoint_url=endpoint_url)

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, source: BinaryIO) -> str:
        resolved = self._key(key)
        self.client.upload_fileobj(source, self.bucket, resolved)
        return f"s3://{self.bucket}/{resolved}"

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except self.client.exceptions.ClientError:
            return False

    def open(self, key: str) -> BinaryIO:
        return self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"]


def object_store_from_env(*, local_root: str | None = None) -> ObjectStore:
    bucket = os.getenv("EMBODIEDLAB_S3_BUCKET")
    if bucket:
        return S3ObjectStore(
            bucket,
            endpoint_url=os.getenv("EMBODIEDLAB_S3_ENDPOINT"),
            prefix=os.getenv("EMBODIEDLAB_S3_PREFIX", ""),
        )
    return LocalObjectStore(local_root or os.getenv("EMBODIEDLAB_ARTIFACT_DIR", "~/.embodiedlab/artifacts"))

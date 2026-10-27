"""Per-run instance snapshot: the instance record a run materialized, kept in the run directory.

The shared instance store (one file per instance id) is what the Worker reads when an environment starts. Instance
ids depend on the experiment, split and index only (they also seed the appearance, so they never change), which
means materializing the same seed from another tree (another spec version) overwrites the shared file. An
environment has loaded its instance by then, but an offline re-read of the store can pick up the other version.

So every runner writes the record it materialized into its run directory (`instance.json`) and offline evaluation
reads that snapshot, checked against the sealed episode (instance id, spec digest) and against its own truth hash.
Runs made before snapshots existed fall back to the shared store.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from benchmark.instances.service import Instance, materialize, read

SNAPSHOT_NAME = "instance.json"


def truth_sha256(record: dict) -> str:
    """The truth hash recomputed from a record's own fields (as the runtime checks it at environment start)."""
    truth = {"exp_id": record["exp_id"], "instance_id": record["instance_id"], "hidden": record["hidden"]}
    return hashlib.sha256(json.dumps(truth, sort_keys=True).encode()).hexdigest()


def save(instance: Instance, run_dir: str | Path) -> Path:
    """Write the record of `instance` (exactly what `materialize` writes) to <run_dir>/instance.json.

    The record is materialized into a private temporary directory, never re-read from the shared store, so a
    concurrent overwrite of the shared file cannot leak into the snapshot."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=run_dir, prefix=".instance-") as tmp:
        written = materialize(instance, tmp)
        target = run_dir / SNAPSHOT_NAME
        shutil.move(str(written), str(target))
    os.chmod(target, 0o600)
    return target


def check(record: dict, sealed_dir: str | Path | None = None) -> list[str]:
    """Problems of a snapshot: a truth hash that does not match its own hidden values, or (given the sealed
    capture) an instance id or spec digest other than the sealed episode's."""
    problems = []
    if record.get("truth_sha256") != truth_sha256(record):
        problems.append("instance snapshot truth does not match its recorded hash")
    if sealed_dir is not None:
        episode = json.loads((Path(sealed_dir) / "episode.json").read_text())
        for field in ("instance_id", "spec_digest"):
            if episode.get(field) != record.get(field):
                problems.append(f"instance snapshot {field} {record.get(field)} differs from the sealed episode's "
                                f"{episode.get(field)}")
    return problems


def for_evaluation(run_dir: str | Path, store: str | Path, instance_id: str) -> dict:
    """The instance record to evaluate a run's sealed capture against: the run's snapshot when it has one, else the
    shared store (runs made before snapshots existed).

    A snapshot whose truth hash does not match its hidden values is refused (ValueError). An instance id or spec
    digest other than the sealed episode's is left to the evaluator, which then fails `sealed_record_intact` and
    names the mismatch (the run really was made against another instance or spec version)."""
    path = Path(run_dir) / SNAPSHOT_NAME
    if not path.is_file():
        return read(store, instance_id)
    record = json.loads(path.read_text())
    corrupt = [p for p in check(record) if "truth" in p]
    if corrupt:
        raise ValueError(f"{path}: " + "; ".join(corrupt))
    return record

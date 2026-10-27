"""Instance service (layer 4): dev / test / train splits and opaque instance ids.

dev: seed = SHA-256("embodiedlab-dev/v1" | exp_id | index)   -- public, reproducible
test: seed = HMAC-SHA256(K_test, "params" | exp_id | index)     -- only with the test key
train: same as test with K_train (interface only; nothing is generated in this phase)

Hidden values are drawn by the registry from `experiment.yaml`. An index whose
draw violates the spec constraints is rejected (`InstanceRejected`); the table
builder skips it and records the skip, so each index is used at most once.

`materialize` writes the private instance record that the Bench reads through
its read-only instance mount (the management channel). Callers and Agents only
ever see `instance_id`.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

from benchmark.registry.loader import load

SPLITS = ("dev", "test", "train")
DEV_NAMESPACE = b"embodiedlab-dev/v1"
TABLE_VERSION = 1
INSTANCE_SCHEMA = "embodiedlab.instance/v1"
LAYOUT_CANDIDATES = 8


class InstanceRejected(ValueError):
    def __init__(self, exp_id, index, failed):
        super().__init__(f"{exp_id} index {index} violates {failed}")
        self.failed = failed


def key_id(key: bytes) -> str:
    return hashlib.sha256(b"embodiedlab-key-id\x1f" + key).hexdigest()[:12]


def load_key(path: str | Path) -> bytes:
    key = Path(path).read_bytes().strip()
    if len(key) < 32:
        raise ValueError("instance keys must hold at least 32 bytes")
    return key


def derivation_namespace(exp_id: str) -> str:
    """The name instances are derived under: the experiment id as first released ("<id>_v1"), so renaming an
    experiment never changes the instance, hidden values or layout behind a (split, index)."""
    return exp_id + "_v1"


def _message(label: bytes, exp_id: str, index: int) -> bytes:
    if type(index) is not int or index < 0:
        raise ValueError("index must be a non-negative integer")
    return label + b"\x1f" + derivation_namespace(exp_id).encode() + b"\x1f" + str(index).encode()


def _digest(split: str, label: bytes, exp_id: str, index: int, key: bytes | None) -> bytes:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split == "dev":
        if key is not None:
            raise ValueError("the dev split is public and takes no key")
        return hashlib.sha256(DEV_NAMESPACE + b"\x1f" + _message(label, exp_id, index)).digest()
    if key is None:
        raise ValueError(f"the {split} split requires its key")
    return hmac.new(key, _message(label, exp_id, index), hashlib.sha256).digest()


@dataclass(frozen=True)
class Instance:
    instance_id: str
    exp_id: str
    split: str
    index: int
    key_id: str | None
    hidden: dict
    spec_digest: str
    layout_candidates: list | None = None

    def truth(self) -> dict:
        return {"exp_id": self.exp_id, "instance_id": self.instance_id, "hidden": _jsonable(self.hidden)}

    def truth_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.truth(), sort_keys=True).encode()).hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def derive(exp_id: str, split: str, index: int, key: bytes | None = None) -> Instance:
    spec = load(exp_id)
    seed = int.from_bytes(_digest(split, b"params", exp_id, index, key), "big")
    hidden = spec.sample(random.Random(seed))
    failed = spec.failed_constraints(hidden)
    if failed:
        raise InstanceRejected(exp_id, index, failed)
    instance_id = "inst-" + _digest(split, b"id", exp_id, index, key).hex()[:24]
    return Instance(instance_id, exp_id, split, index, None if key is None else key_id(key),
                    _jsonable(hidden), spec.digest(), layout_candidates(spec, split, index, key))


def calibration_instance(exp_id: str, name: str, fixed: dict) -> Instance:
    """A named calibration extreme (e.g. the ends of a parameter range), outside every split.

    Sampled quantities in `fixed` take the given values (inside their support),
    derived ones are recomputed; the id comes from its own namespace and the
    layout candidates are those of dev index 0.
    """
    spec = load(exp_id)
    hidden = spec.sample(random.Random(0), fixed=fixed)
    failed = spec.failed_constraints(hidden)
    if failed:
        raise InstanceRejected(exp_id, -1, failed)
    digest = hashlib.sha256(b"embodiedlab-calib/v1\x1f" + f"{derivation_namespace(exp_id)}\x1f{name}".encode()).hexdigest()
    return Instance("inst-" + digest[:24], exp_id, "calib", 0, None, _jsonable(hidden), spec.digest(),
                    layout_candidates(spec, "dev", 0))


def layout_candidates(spec, split, index, key=None):
    """Ordered rigid placements {dx_m, dy_m, yaw_deg}; the Bench uses the first admissible one.

    Drawn from its own stream, so adding or changing jitter never alters hidden values.
    """
    jitter = spec.raw["scene"].get("layout_jitter")
    if not jitter:
        return None
    rng = random.Random(int.from_bytes(_digest(split, b"layout", spec.id, index, key), "big"))
    xy, yaw = float(jitter["xy_m"]), float(jitter["yaw_deg"])
    return [{"dx_m": rng.uniform(-xy, xy), "dy_m": rng.uniform(-xy, xy), "yaw_deg": rng.uniform(-yaw, yaw)}
            for _ in range(LAYOUT_CANDIDATES)]


def issue_table(exp_id: str, split: str, count: int, start: int = 0, key: bytes | None = None,
                segment: str = "formal") -> tuple[list[Instance], list[dict]]:
    """First `count` accepted indices from `start`; rejected indices are listed, never reused."""
    accepted, skipped, index = [], [], start
    while len(accepted) < count:
        try:
            accepted.append(derive(exp_id, split, index, key))
        except InstanceRejected as exc:
            skipped.append({"index": index, "failed_constraints": exc.failed, "segment": segment})
        index += 1
        if index - start > 100 * count + 1000:
            raise RuntimeError(f"{exp_id}: constraints reject almost every draw")
    return accepted, skipped


def materialize(instance: Instance, store: str | Path) -> Path:
    """Write the private instance record read by the Bench and the evaluator."""
    spec = load(instance.exp_id)
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": INSTANCE_SCHEMA, "instance_id": instance.instance_id,
        "exp_id": instance.exp_id, "split": instance.split, "index": instance.index,
        "key_id": instance.key_id, "table_version": TABLE_VERSION,
        "spec_digest": instance.spec_digest, "runtime": spec.runtime, "recipe": spec.recipe_name,
        "scene_params": _jsonable(spec.scene_params(instance.hidden)),
        "layout_jitter": spec.raw["scene"].get("layout_jitter"),
        "layout_candidates": instance.layout_candidates,
        "hidden": instance.hidden, "truth_sha256": instance.truth_sha256(),
        "budget": spec.budget,
        "public_task": spec.public_task(instance.hidden),
    }
    path = store / f"{instance.instance_id}.json"
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def read(store: str | Path, instance_id: str) -> dict:
    """params(instance_id): management-channel lookup for the runtime and evaluator."""
    if not instance_id.startswith("inst-") or "/" in instance_id:
        raise ValueError("malformed instance id")
    return json.loads((Path(store) / f"{instance_id}.json").read_text())

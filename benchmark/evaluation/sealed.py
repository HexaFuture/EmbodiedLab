"""Reader for a sealed episode directory written by the Bench at submission.

Layout (`embodiedlab.sealed-episode/v1`):
  episode.json     run facts: ids, ticks, budget, submission attempts, terminal apparatus state, device values
  submission.json  the first submit_result arguments exactly as received
  evidence.json    {evidence_id: record}; each record carries the SHA-256 of its payload
  evidence/        instrument payload files referenced by records (e.g. the camera views of a ruler reading)
  manifest.json    {relative path: SHA-256} for every file above
The evaluator never calls the runtime: everything it needs is in this directory
plus the instance's truth from the instance service.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

SEALED_SCHEMA = "embodiedlab.sealed-episode/v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True).encode())


@dataclass
class Sealed:
    root: Path
    episode: dict
    submission: object
    evidence: dict
    manifest: dict
    manifest_problems: list = field(default_factory=list)

    @property
    def episode_id(self) -> str:
        return self.episode["episode_id"]

    def payload_sha_ok(self, record: dict) -> bool:
        """Recompute the record's payload hash (tamper check for measurement_evidence_valid)."""
        kind = record.get("kind")
        if kind == 'centroid_photogate':
            public = record['public']
            unhashed = {k:v for k,v in public.items() if k != 'payload_sha256'}
            return (canonical_sha256(unhashed) == public['payload_sha256']
                    and sha256_bytes(public['csv'].encode()) == public['sha256'])
        if kind in ("ruler_image", "bench_view"):
            # The standard views of the state ({camera: {file, sha256}}); older records carry one image_file.
            files = ([(record["image_file"], record["image_sha256"])] if "image_file" in record else
                     [(v["file"], v["sha256"]) for v in (record.get("images") or {}).values()])
            return bool(files) and all((self.root / f).is_file() and sha256_bytes((self.root / f).read_bytes()) == h
                                       for f, h in files)
        if kind in ("optical_gate", "airtrack_recorder", "spectro_read", "titration_session", "titration_record", "incline_record",
                    "kinetic_run", "screen_capture", "thermo_log", "thermo_record", "gauge_log", "meter_log"):
            return canonical_sha256(record["public"]) == record["sha256"]
        if kind == "rectifier_scope":
            from benchmark.evaluation.rectifier import digest
            return digest(record["public"]) == record["sha256"]
        if kind in ("coupled_recorder", "tray_encoder", "torsion_photogate", "circuit_logger"):
            return sha256_bytes(record["csv"].encode()) == record["sha256"]
        return False


def load_sealed(root: str | Path) -> Sealed:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    problems = []
    for relative, digest in manifest.items():
        path = root / relative
        if not path.is_file():
            problems.append(f"missing {relative}")
        elif sha256_bytes(path.read_bytes()) != digest:
            problems.append(f"hash mismatch {relative}")
    for required in ("episode.json", "submission.json", "evidence.json"):
        if required not in manifest:
            problems.append(f"unmanifested {required}")
    episode = json.loads((root / "episode.json").read_text())
    if episode.get("schema_version") != SEALED_SCHEMA:
        problems.append(f"unsupported sealed schema {episode.get('schema_version')!r}")
    # Python's json accepts NaN/Infinity; valid_submission rejects them explicitly.
    submission = json.loads((root / "submission.json").read_text())
    evidence = json.loads((root / "evidence.json").read_text())
    return Sealed(root, episode, submission, evidence, manifest, problems)


def write_manifest(root: str | Path) -> dict:
    """Used by the Bench (and tests) to seal a directory."""
    root = Path(root)
    manifest = {str(p.relative_to(root)): sha256_bytes(p.read_bytes())
                for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest

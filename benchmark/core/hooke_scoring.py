"""Pure episode contracts; no simulator imports and no public truth endpoint."""
from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass


RANDOMIZATION_VERSION = "hooke-k/v1"
EVALUATOR = "embodiedlab-hooke-identification-v1"
WEIGHT_IDS = tuple(f"weight_{i:02d}" for i in range(1, 6))


def sample_hooke(seed: int) -> float:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    digest = hashlib.sha256(f"{RANDOMIZATION_VERSION}:{seed}".encode()).digest()
    # Conservative development range around the validated 18 N/m apparatus.
    # Public development seeds are reproducible, not a hidden benchmark split.
    return 18.0 + 4.0 * int.from_bytes(digest[:8], "big") / 2**64


def sample_configuration(seed: int, task_id: str) -> dict:
    stiffness = sample_hooke(seed)
    if task_id == "hooke_identification_v1":
        count, version = 5, RANDOMIZATION_VERSION
    elif task_id == "hooke_identification_v2":
        digest = hashlib.sha256(f"hooke-apparatus/v2:{seed}:count".encode()).digest()
        count = 3 + int.from_bytes(digest[:8], "big") % 3
        version = "hooke-apparatus/v2.1"
        k_digest = hashlib.sha256(f"{version}:{seed}:stiffness".encode()).digest()
        stiffness = 18.0 + 12.0 * int.from_bytes(k_digest[:8], "big") / 2**64
    else:
        raise ValueError("unsupported Hooke task version")
    # Existing tray supports at most five 70 mm reserved footprints. Limit
    # full-load travel to 15 cm, inside the validated ruler/guide range.
    if not 3 <= count <= 5 or count * 0.05 * 9.81 / stiffness > 0.15:
        raise ValueError("sample exceeds validated apparatus capacity or travel")
    return {"stiffness_n_m": stiffness, "weight_count": count,
            "weight_mass_kg": 0.05, "randomization_version": version}


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    frame: int
    supported: tuple[str, ...]
    stable: bool
    reading_cm: float | None
    image_sha256: str


class HookeEpisode:
    def __init__(self, seed: int, horizon: int = 18000, task_id: str = "hooke_identification_v1"):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer (30 Hz frames)")
        self.configuration = sample_configuration(seed, task_id)
        self.truth = self.configuration["stiffness_n_m"]
        self.task_id = task_id
        self.success_relative_error = 0.10 if task_id == "hooke_identification_v2" else 0.05
        self.evaluator = "embodiedlab-hooke-identification-" + task_id.rsplit("_",1)[1]
        self.weight_ids = WEIGHT_IDS[:self.configuration["weight_count"]]
        self.episode_id = uuid.uuid4().hex
        self.horizon = horizon
        self.evidence: dict[str, Evidence] = {}
        self.report = None

    def public_task(self):
        count = len(self.weight_ids)
        return {
            "schema_version": "embodiedlab.task/v2", "task_id": self.task_id,
            "episode_id": self.episode_id,
            "instruction": "Determine the unknown spring stiffness by performing a loading experiment.",
            "weights": [{"id": name, "mass_kg": 0.05} for name in self.weight_ids],
            "gravity_m_s2": 9.81, "ruler_unit": "cm",
            "measurement_requirements": {
                "target_reading_count": count + 1,
                "conditions": "Unloaded, then one stable reading after each additional weight.",
                "record": "Read and record the unloaded condition before loading; record each subsequent reading before changing the load. Save measurements.csv with evidence_id, mass_kg, reading_cm and uncertainty_cm.",
                "instrument": "Read the red pointer tip against the ruler in the camera images. Choose waiting time, repeats and fitting method yourself.",
            },
            "completion_requirements": {
                "physical": "All weights listed in this task must be released and stably supported by the tray.",
                "distinct_loaded_measurements_min": count if self.task_id.endswith("v2") else 3,
                "unloaded_measurement_required": True,
            },
            "horizon": self.horizon, "sim_step_unit": "1/30 simulated second",
            "score": "hard gates times max(0, 1 - relative parameter error)",
            "success_relative_error": self.success_relative_error,
            "status": "development; public seeds are reproducible, not hidden-test secrets",
            "submission": {
                "estimate": "positive number", "unit": "N/m",
                "model": "nonempty description", "uncertainty_statement": "nonempty description",
                "measurements": [{"evidence_id": "...", "reading_cm": 0.0, "mass_kg": 0.0}],
            },
        }

    def add_evidence(self, frame, supported, stable, reading_cm, image_sha256):
        eid = f"{self.episode_id}:{len(self.evidence) + 1}"
        self.evidence[eid] = Evidence(eid, frame, tuple(sorted(supported)), bool(stable),
                                      reading_cm, image_sha256)
        return eid

    def evaluation(self, frame):
        base = {"reward": 0.0, "task_completed": False, "sim_step": frame, "evaluator": self.evaluator}
        if self.report is not None:
            return {**base, **self.report}
        return {**base, "terminated": False}

    def submit(self, value, *, frame, all_supported, stable, released, budget_ok=True):
        if self.report is not None:
            raise ValueError("episode already terminated; submissions are immutable")
        gates = {
            "valid_submission": False, "measurement_evidence_valid": False,
            "all_weights_on_tray": bool(all_supported), "stable": bool(stable),
            "released": bool(released), "within_budget": bool(budget_ok and frame <= self.horizon),
        }
        error = None
        relative_error = None
        reading_diagnostics = []
        try:
            if not isinstance(value, dict):
                raise ValueError("submission must be an object")
            allowed = {"estimate", "unit", "model", "uncertainty_statement", "measurements"}
            if set(value) != allowed:
                raise ValueError("submission fields must match the public task schema")
            estimate = finite(value["estimate"], "estimate")
            if estimate <= 0 or value["unit"] != "N/m":
                raise ValueError("estimate must be positive with unit N/m")
            for key in ("model", "uncertainty_statement"):
                if not isinstance(value[key], str) or not 1 <= len(value[key].strip()) <= 4000:
                    raise ValueError(f"{key} must be a bounded nonempty string")
            rows = value["measurements"]
            if not isinstance(rows, list) or not 4 <= len(rows) <= 100:
                raise ValueError("provide 4 to 100 measurement rows")
            gates["valid_submission"] = True
            ids, masses = set(), set()
            for row in rows:
                if not isinstance(row, dict) or set(row) != {"evidence_id", "reading_cm", "mass_kg"}:
                    raise ValueError("measurement fields must match public schema")
                eid = row["evidence_id"]
                if not isinstance(eid, str) or eid in ids or eid not in self.evidence:
                    raise ValueError("unknown, duplicate or cross-episode evidence")
                ids.add(eid)
                evidence = self.evidence[eid]
                if not set(evidence.supported).issubset(self.weight_ids):
                    raise ValueError("evidence contains a weight outside this task")
                reading = finite(row["reading_cm"], "reading_cm")
                mass = finite(row["mass_kg"], "mass_kg")
                if not evidence.stable or evidence.reading_cm is None:
                    raise ValueError("measurement does not have a stable readable ruler")
                difference_cm = abs(reading - evidence.reading_cm)
                if self.task_id == "hooke_identification_v2":
                    reading_diagnostics.append({
                        "evidence_id": eid, "submitted_reading_cm": reading,
                        "automatic_reading_cm": evidence.reading_cm,
                        "absolute_difference_cm": difference_cm,
                        "within_legacy_tolerance": difference_cm <= 0.25,
                    })
                elif difference_cm > 0.25:
                    raise ValueError("measurement is inconsistent with its RGB evidence")
                if abs(mass - len(evidence.supported) * 0.05) > 1e-6:
                    raise ValueError("measurement load differs from physical support evidence")
                masses.add(len(evidence.supported))
            required_loads = len(self.weight_ids) if self.task_id.endswith("v2") else 3
            if 0 not in masses or len(masses - {0}) < required_loads:
                raise ValueError(f"need unloaded and {required_loads} distinct loaded conditions")
            gates["measurement_evidence_valid"] = True
            relative_error = abs(estimate - self.truth) / self.truth
        except (ValueError, KeyError, TypeError) as exc:
            error = str(exc)
        valid = all(gates.values())
        reward = max(0.0, 1.0 - relative_error) if valid and relative_error is not None else 0.0
        self.report = {
            "schema_version": "embodiedlab.evaluation/v1",
            "episode_id": self.episode_id, "evaluator": self.evaluator,
            "reward": reward, "task_completed": valid and (relative_error <= self.success_relative_error
                or math.isclose(relative_error, self.success_relative_error, rel_tol=0.0, abs_tol=1e-12)),
            "sim_step": frame, "terminated": True, "gates": gates,
            "parameter_relative_error": relative_error, "error": error,
        }
        if self.task_id == "hooke_identification_v2":
            self.report["evaluation_policy"] = "hooke-v2-reading-diagnostic-10pct/v1"
            self.report["success_relative_error"] = self.success_relative_error
            self.report["reading_diagnostics"] = reading_diagnostics
        return dict(self.report)

    def terminate(self, frame, reason):
        if self.report is None:
            self.report = {**self.evaluation(frame), "terminated": True,
                           "schema_version": "embodiedlab.evaluation/v1",
                           "episode_id": self.episode_id, "error": reason}
        return dict(self.report)

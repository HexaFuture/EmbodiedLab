"""Experiment registry: load, validate and expand `experiments/<exp_id>/experiment.yaml`.

Layer 3 of the architecture. Everything an Agent sees about an experiment
(task card, tool list, budget, submission schema) is generated here from the
declarative spec; hidden values are sampled here but only ever leave through
the instance service's private materialization.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import jsonschema
import yaml

from . import expressions

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "experiments"
KITS = REPO / "benchmark" / "kits"
SCHEMA_PATH = Path(__file__).with_name("experiment.schema.json")

TASK_SCHEMA_VERSION = "embodiedlab.task/v3"
OBS_TOOLS = ("get_task_spec", "get_observation", "get_robot_state")
CONTROL_TOOLS = ("execute_joint_trajectory", "set_gripper", "wait")
SUBMIT_TOOLS = ("submit_result",)
_HIDDEN_REF = re.compile(r"^\{hidden\.([A-Za-z_][A-Za-z0-9_]*)\}$")
_PLACEHOLDER = re.compile(r"\{([a-z][A-Za-z0-9_]*)\}")    # mixed case: units such as mL, pH, uM, Hz, V, A


class SpecError(ValueError):
    pass


@functools.cache
def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _stamp(root: Path) -> tuple:
    """Modification times of the files under root. Cache keys include it, so a long-running process (the
    Controller) sees spec and recipe edits instead of the versions it first loaded."""
    if root.is_file():
        return ((root.name, root.stat().st_mtime_ns),)
    return tuple(sorted((str(p.relative_to(root)), p.stat().st_mtime_ns) for p in root.rglob("*")
                        if p.is_file() and "__pycache__" not in p.parts))


def kit_recipes(kit: str) -> dict:
    path = KITS / kit / "recipes.yaml"
    if not path.is_file():
        raise SpecError(f"kit {kit!r} has no recipe registry at {path}")
    return _kit_recipes(kit, _stamp(path))


@functools.lru_cache(maxsize=64)
def _kit_recipes(kit: str, stamp: tuple) -> dict:
    return yaml.safe_load((KITS / kit / "recipes.yaml").read_text())["recipes"]


def gate_entries(gates: list) -> list[tuple[str, dict]]:
    """Normalise spec gate items (`name` or `{name: params}`) to (name, params)."""
    entries = []
    for item in gates:
        if isinstance(item, str):
            entries.append((item, {}))
        else:
            (name, params), = item.items()
            entries.append((name, dict(params)))
    return entries


def _flatten(prefix: str, value, out: dict):
    """Public constants become task-card placeholders (`a_b`, plus `_text` forms)."""
    if isinstance(value, dict):
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value.values()):
            out[f"{prefix}_text"] = ", ".join(f"{k}: {v}" for k, v in value.items())
        for key, item in value.items():
            _flatten(f"{prefix}_{key}" if prefix else key, item, out)
    elif isinstance(value, list):
        if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            out[f"{prefix}_text"] = f"{value[0]} to {value[1]}"
        else:
            out[f"{prefix}_text"] = ", ".join(str(v) for v in value)
        out[prefix] = value
        for index, item in enumerate(value):         # single entries, e.g. a width and a height
            if isinstance(item, (int, float, str)) and not isinstance(item, bool):
                out[f"{prefix}_{index}"] = item
    else:
        out[prefix] = value


@dataclass(frozen=True)
class ExperimentSpec:
    raw: dict
    root: Path

    # --- identity ---------------------------------------------------------
    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def tier(self) -> str:
        return self.raw["tier"]

    @property
    def kit(self) -> str:
        return self.raw["kit"]

    @property
    def recipe_name(self) -> str:
        return self.raw["scene"]["recipe"]

    @property
    def recipe(self) -> dict:
        recipe = copy.deepcopy(kit_recipes(self.kit)[self.recipe_name])
        # Experiment-specific acquisition budget cannot exceed the device cap.
        arms = self.raw.get("consumables", {}).get("logger_arms")
        logger = recipe.get("public_constants", {}).get("logger")
        if arms is not None and logger is not None:
            logger["max_arms"] = min(int(arms), logger["max_arms"])
        # Public constants that belong to an instrument (its `constants` list) are published only to experiments
        # that use that instrument (the tray encoder's go to hooke_dynamic_k, not to hooke_k).
        used = set(self.raw.get("instruments", []))
        for name, instrument in recipe.get("instruments", {}).items():
            if name not in used:
                for key in instrument.get("constants", ()):
                    recipe.get("public_constants", {}).pop(key, None)
        return recipe

    @property
    def runtime(self) -> str:
        """Bench apparatus family that builds this experiment's scene."""
        return self.recipe["runtime"]

    @property
    def gates(self) -> list[tuple[str, dict]]:
        return gate_entries(self.raw["gates"])

    @property
    def scoring(self) -> dict:
        return self.raw["scoring"]

    @property
    def budget(self) -> dict:
        return {"submissions": 1, **self.raw["budget"]}

    def gate_params(self, name: str) -> dict | None:
        for gate, params in self.gates:
            if gate == name:
                return params
        return None

    def digest(self) -> str:
        """SHA-256 over what defines the experiment: its spec, card and schema files and the parsed kit
        recipe it uses. Only its own recipe enters (canonical JSON), so adding or editing another recipe of the
        kit leaves the digest unchanged."""
        h = hashlib.sha256()
        files = [self.root / "experiment.yaml", self.root / self.raw["task_card"],
                 self.root / self.raw["submission"]["schema"]]
        for path in files:
            h.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
        recipe = json.dumps(self.recipe, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        h.update(f"recipe:{self.kit}/{self.recipe_name}".encode() + b"\0" + recipe.encode() + b"\0")
        return h.hexdigest()

    # --- hidden parameters -------------------------------------------------
    def sample(self, rng: random.Random, fixed: dict | None = None) -> dict:
        """Draw hidden values in declaration order; bounds may reference earlier values.

        fixed: values for sampled (non-derived) quantities, used for calibration
        extremes; each must lie inside its distribution's support.
        """
        fixed = dict(fixed or {})
        values: dict = {}
        for name, dist in self.raw["hidden"].items():
            kind = dist["dist"]
            if name in fixed and kind == "derived":
                raise SpecError(f"{self.id}: {name} is derived and cannot be fixed")
            if kind == "choice":
                if name in fixed and fixed[name] not in dist["values"]:
                    raise SpecError(f"{self.id}: {name}={fixed[name]} is not one of {dist['values']}")
                values[name] = fixed.pop(name) if name in fixed else rng.choice(list(dist["values"]))
            elif kind == "derived":
                values[name] = expressions.evaluate(dist["expr"], values)
            else:
                low = float(expressions.evaluate(dist["low"], values))
                high = float(expressions.evaluate(dist["high"], values))
                if not (math.isfinite(low) and math.isfinite(high) and low <= high):
                    raise SpecError(f"{self.id}: invalid bounds for {name}: [{low}, {high}]")
                if name in fixed:
                    value = fixed.pop(name)
                    if not low - 1e-9 * max(1.0, abs(low)) <= value <= high + 1e-9 * max(1.0, abs(high)):
                        raise SpecError(f"{self.id}: {name}={value} outside [{low}, {high}]")
                    values[name] = value
                elif kind == "uniform":
                    values[name] = rng.uniform(low, high)
                elif kind == "loguniform":
                    if low <= 0:
                        raise SpecError(f"{self.id}: loguniform {name} needs a positive lower bound")
                    values[name] = math.exp(rng.uniform(math.log(low), math.log(high)))
                elif kind == "int_uniform":
                    values[name] = rng.randint(int(low), int(high))
            if isinstance(values[name], list):
                values[name] = tuple(values[name])
        if fixed:
            raise SpecError(f"{self.id}: unknown hidden quantities {sorted(fixed)}")
        return values

    def failed_constraints(self, hidden: dict) -> list[str]:
        return [c for c in self.raw["constraints"] if not expressions.evaluate(c, hidden)]

    def public_hidden(self, hidden: dict) -> dict:
        return {k: hidden[k] for k, d in self.raw["hidden"].items() if d.get("public")}

    def scene_params(self, hidden: dict) -> dict:
        """Recipe parameters with `{hidden.x}` references substituted (management channel only)."""
        def resolve(value):
            if isinstance(value, str):
                match = _HIDDEN_REF.match(value)
                return hidden[match.group(1)] if match else value
            if isinstance(value, list):
                return [resolve(v) for v in value]
            if isinstance(value, dict):
                return {k: resolve(v) for k, v in value.items()}
            return value
        return resolve(copy.deepcopy(self.raw["scene"]["params"]))

    # --- public artefacts ----------------------------------------------------
    def submission_schema(self) -> dict:
        return json.loads((self.root / self.raw["submission"]["schema"]).read_text())

    def tools(self) -> list[str]:
        tools = list(OBS_TOOLS) + list(CONTROL_TOOLS) + list(self.recipe.get("controls", []))
        for instrument in self.raw["instruments"]:
            tools += [t for t in self.recipe["instruments"][instrument]["tools"] if t not in tools]
        if self.gate_params("prediction_before_holdout") is not None:
            tools.append("record_prediction")
        return tools + list(SUBMIT_TOOLS)

    def card_context(self, hidden: dict) -> dict:
        context: dict = {}
        _flatten("", self.recipe.get("public_constants", {}), context)
        context.update(self.public_hidden(hidden))
        context["sim_s"] = _number(self.raw["budget"]["sim_s"])
        context["wall_s"] = _number(self.raw["budget"]["wall_s"])
        threshold = self.scoring["threshold"]
        context["threshold"] = _number(threshold)
        context["threshold_pct"] = _number(round(threshold * 100, 6))
        for name, params in self.gates:
            for key, value in params.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    context.setdefault(key, _number(value))
                    if key.endswith("_tol"):
                        context.setdefault(key + "_pct", _number(round(value * 100, 6)))
            if name == "submission_self_consistent":
                context["self_consistency_pct"] = _number(round(params["tol"] * 100, 6))
        examples = self.submission_schema().get("examples") or [{}]
        context["submission_example"] = json.dumps(examples[0], indent=2, ensure_ascii=False)
        return context

    def render_task_card(self, hidden: dict) -> str:
        template = (self.root / self.raw["task_card"]).read_text()
        context = self.card_context(hidden)
        missing = sorted({m for m in _PLACEHOLDER.findall(template) if m not in context})
        if missing:
            raise SpecError(f"{self.id}: task card placeholders without public values: {missing}")
        return _PLACEHOLDER.sub(lambda m: str(context[m.group(1)]), template)

    def public_task(self, hidden: dict) -> dict:
        """The Agent-facing task specification; contains no hidden, seed or index values."""
        return {
            "schema_version": TASK_SCHEMA_VERSION,
            "exp_id": self.id, "task_id": self.id, "version": self.raw["version"],
            "title": self.raw["title"], "tier": self.tier, "kit": self.kit,
            "task_card": self.render_task_card(hidden),
            "tools": self.tools(),
            "affordances": list(self.raw.get("affordances", [])),
            "budget": self.budget,
            "public_parameters": self.public_hidden(hidden),
            "public_constants": copy.deepcopy(self.recipe.get("public_constants", {})),
            "submission_schema": self.submission_schema(),
            "primary": dict(self.raw["submission"]["primary"]),
        }


def _number(value):
    return int(value) if isinstance(value, float) and value.is_integer() else value



def _validate(spec: ExperimentSpec) -> None:
    raw = spec.raw
    try:
        jsonschema.validate(raw, _schema())
    except jsonschema.ValidationError as exc:
        raise SpecError(f"{spec.root.name}: {exc.message} at {list(exc.absolute_path)}") from None
    if raw["id"] != spec.root.name:
        raise SpecError(f"spec id {raw['id']!r} differs from directory {spec.root.name!r}")
    recipes = kit_recipes(raw["kit"])
    if raw["scene"]["recipe"] not in recipes:
        raise SpecError(f"{spec.id}: recipe {raw['scene']['recipe']!r} is not registered in kit {raw['kit']}")
    recipe = recipes[raw["scene"]["recipe"]]
    unknown = set(raw["scene"]["params"]) - set(recipe.get("params", {}))
    if unknown:
        raise SpecError(f"{spec.id}: recipe {spec.recipe_name} has no parameters {sorted(unknown)}")
    if raw["scene"].get("layout_jitter") and not recipe.get("layout_jitter_supported"):
        raise SpecError(f"{spec.id}: recipe {spec.recipe_name} does not support layout_jitter")
    for instrument in raw["instruments"]:
        if instrument not in recipe["instruments"]:
            raise SpecError(f"{spec.id}: recipe {spec.recipe_name} has no instrument {instrument!r}")
    defined: set[str] = set()
    for name, dist in raw["hidden"].items():
        for field in ("low", "high", "expr"):
            if isinstance(dist.get(field), str):
                undefined = expressions.names(dist[field]) - defined
                if undefined:
                    raise SpecError(f"{spec.id}: {name}.{field} references undefined {sorted(undefined)}")
        defined.add(name)
    for constraint in raw["constraints"]:
        undefined = expressions.names(constraint) - defined
        if undefined:
            raise SpecError(f"{spec.id}: constraint {constraint!r} references undefined {sorted(undefined)}")
    for value in json.dumps(raw["scene"]["params"]).split('"'):
        match = _HIDDEN_REF.match(value)
        if match and match.group(1) not in defined:
            raise SpecError(f"{spec.id}: scene parameter references undefined hidden {match.group(1)!r}")
    from benchmark.evaluation.gates import GATES
    for name, _ in spec.gates:
        if name not in GATES:
            raise SpecError(f"{spec.id}: gate {name!r} is not in the common gate library")
    targets = raw["scoring"]["target"]
    for target in [targets] if isinstance(targets, str) else targets:
        if raw["scoring"]["metric"] == "sealed_rel_error":
            if (spec.id, target) not in (("projectile_landing", "distance_m"), ("optical_relay", "magnification")):
                raise SpecError("sealed outcome targets are defined for projectile distance_m and relay magnification")
        elif target not in raw["hidden"]:
            raise SpecError(f"{spec.id}: scoring target {target!r} is not a hidden parameter")
    submission = spec.submission_schema()
    jsonschema.Draft202012Validator.check_schema(submission)
    # Examples hold placeholders ("<C in farad, a number>"), not values an agent could copy, so they are checked for
    # their fields only: every required field present and no field the schema does not define.
    for example in submission.get("examples", []):
        fields = set(example)
        if set(submission.get("required", [])) - fields or fields - set(submission.get("properties", {})):
            raise SpecError(f"{spec.id}: the submission example does not have the schema's fields")
    field = raw["scoring"].get("field", raw["submission"]["primary"]["field"])
    if field not in submission.get("properties", {}):
        raise SpecError(f"{spec.id}: scored field {field!r} is absent from the submission schema")
    if raw['scoring']['metric'] == 'composite':
        components = raw['scoring']['components']
        declared = raw['scoring']['target']
        declared = [declared] if isinstance(declared, str) else declared
        if set(c['target'] for c in components) != set(declared):
            raise SpecError('composite components must cover the declared targets')
        if len({c['field'] for c in components}) != len(components):
            raise SpecError('composite fields must be unique')
        for c in components:
            if c['field'] not in submission.get('properties', {}):
                raise SpecError('composite field absent from submission schema')
    for hook in (raw.get("hooks") or {}).values():
        if hook is not None:
            raise SpecError(f"{spec.id}: hooks are declared but no hook loader exists yet")
    # A dry sample proves the distributions, constraints and card render end to end.
    hidden = spec.sample(random.Random(0))
    spec.failed_constraints(hidden)
    spec.render_task_card(hidden)


def load(exp_id: str) -> ExperimentSpec:
    root = EXPERIMENTS / exp_id
    if not (root / "experiment.yaml").is_file():
        raise SpecError(f"unknown experiment {exp_id!r}")
    return _load(exp_id, _stamp(root), _stamp(KITS))


@functools.lru_cache(maxsize=256)
def _load(exp_id: str, stamp: tuple, kits_stamp: tuple) -> ExperimentSpec:
    root = EXPERIMENTS / exp_id
    spec = ExperimentSpec(yaml.safe_load((root / "experiment.yaml").read_text()), root)
    _validate(spec)
    return spec


def list_experiments(tier: str | None = None, kit: str | None = None) -> list[str]:
    found = []
    for path in sorted(EXPERIMENTS.glob("*/experiment.yaml")):
        spec = load(path.parent.name)
        if tier and spec.tier != tier or kit and spec.kit != kit:
            continue
        found.append(spec.id)
    return found

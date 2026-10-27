"""Process score: how far a run got, judged offline from the records the Bench exported when it stopped.

    python -m benchmark.evaluation.process --trial <trial dir> [--json]
    python -m benchmark.evaluation.process --summary <trials root> [--json]

Six levels, in order:
    L1  hands-on        the robot first really acts on this experiment's apparatus (grasp-and-move, press, push,
                        dispense...); waving in the air or touching unrelated objects does not count
    L2  set up          the apparatus first in a state from which ANY reasonable method can measure the needed quantity
                        (only the measured object and the instruments used matter)
    L3  first reading   the first valid reading in an L2 state (readout principle: a static value shown on a screen in
                        some observation; a dynamic curve recorded and exported)
    L4  enough data     the valid readings suffice in principle to determine every target quantity by ANY reasonable
                        method (minimum information only; no fixed count, order, instrument or processing). Except
                        hooke_k, pendulum_g and hooke_dynamic_k: the sets their card
                        requires (>= 4 loads; each of P1, P2, P3; >= 3 loads)
    L5  accurate        every measurement gate passes (the gates that are not tidy-up gates, valid_submission and
                        within_budget included) AND the error is within tolerance (e <= 1)
    L6  tidy            every tidy-up gate passes as well, and L5 holds (= task_completed)
level = the number of consecutive levels reached from L1. Weights: L1-L4 10 each; L5 (knowing the result) and L6
(restoring the bench), which the task requires, 30 each; out of 100. The process score is the cumulative weight of
the level reached: L0 0, L1 0.10, L2 0.20, L3 0.30, L4 0.40, L5 0.70, L6 1.00.
Monotone fallback: a run that reaches L5 while a detector missed one of L1-L4 is recorded at level 5 (6 when L6 holds
too) and flagged `monotone_fallback:<missing levels>` (also listed in the report's `monotone_fallback`) for review: a
reasonable method measured accurately reaches L5, so a missed earlier level is a detector defect.

L1-L4 are per-experiment detectors (`Ladder`, registered in benchmark/evaluation/process_rules: LADDERS) over the
exported records in captures/<env>/process (frames, instrument records, state and action logs, evidence, the
`exports` and `calls` of episode.json); they never use the Agent's words or a tool's own report of success. L5 and
L6 come from the evaluation of the sealed submission and are computed here, the same way for every experiment; an
experiment may name extra tidy-up gates (`Ladder.tidy_gates`) that count for L6 only.

Aggregation: the mean over seeds per experiment, then the mean over experiments, so that every experiment counts
equally; per-discipline means (the experiment's kit) are reported alongside as a breakdown.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

PROCESS_SCHEMA = "embodiedlab.process-record/v1"
LEVELS = ("L1", "L2", "L3", "L4", "L5", "L6")
DETECTED = LEVELS[:4]          # the per-experiment detectors' levels; L5 and L6 come from the evaluation
DISCIPLINES = ("mechanics", "circuits", "optics", "thermal", "chemistry")
SCORER = "embodiedlab-process/2"
SCHEME = "six-levels-weighted"
LEVEL_WEIGHTS = {"L1": 10, "L2": 10, "L3": 10, "L4": 10, "L5": 30, "L6": 30}   # out of 100
# The legacy milestones E1-M2 (scorer embodiedlab-process/1) map onto L1-L4 for the provisional ladders.
LEGACY_MAP = {"E1": "L1", "E2": "L2", "M1": "L3", "M2": "L4"}
REPO = Path(__file__).resolve().parents[2]
STORE = Path(os.environ.get("EMBODIEDLAB_INSTANCE_STORE",
                            str(Path(os.environ.get("EMBODIEDLAB_DATA", str(REPO / "data"))) / "instances")))


@dataclass
class Hit:
    """One level: reached or not, the first physics tick it held (None when not timed), and why."""
    reached: bool
    tick: int | None = None
    basis: str = ""
    flags: list = field(default_factory=list)

    def as_dict(self, physics_hz):
        out = {"reached": self.reached, "tick": self.tick,
               "t_s": None if self.tick is None else round(self.tick / physics_hz, 3), "basis": self.basis}
        if self.flags:
            out["flags"] = list(self.flags)
        return out


# Gates on the end state the card asks for (tidy-up): L6 (and success) requires them, L5 (the measurement) does not.
# An experiment adds its own through Ladder.tidy_gates (tidy_gates(exp_id) gives the full set).
TIDY_GATES = frozenset({"encoder_bench_restored", "thermal_bench_restored", "chem_bench_restored", "beam_bench_restored", "circuit_bench_restored", "objects_in_box", "bench_in_order", "weights_in_box", "gate_timers_reset", "coupled_bench_restored", "incline_bench_restored", "optics_bench_restored", "airtrack_bench_restored"})

@dataclass(frozen=True)
class Ladder:
    """The per-experiment part of the six levels.

    detect(run) -> {"L1": Hit, "L2": Hit, "L3": Hit, "L4": Hit} from the process record (ProcessRun); a level left
        out is not reached. Each Hit carries the first physics tick it held (the tick of the record that shows it)
        and a short basis; flags on a Hit are reported with it.
    tidy_gates: gates of this experiment that count as tidy-up (L6) rather than measurement (L5), beyond TIDY_GATES
        (e.g. a hand-empty-at-submission gate the design moves to the tidy-up list).
    provisional: the legacy E1-M2 rule mapped by LEGACY_MAP (from_legacy), until the family writes real detectors.
    note: one line for the report (what the detectors follow, e.g. " ")."""
    detect: object
    tidy_gates: frozenset = frozenset()
    provisional: bool = False
    note: str = ""


def from_legacy(rule, tidy_gates=(), note="legacy E1-M2 rule mapped E1->L1, E2->L2, M1->L3, M2->L4"):
    """A provisional Ladder from a legacy rule(run) -> {"E1", "E2", "M1", "M2": Hit}."""
    def detect(run):
        found = rule(run)
        return {LEGACY_MAP[k]: v for k, v in found.items() if k in LEGACY_MAP}
    detect.__doc__ = rule.__doc__
    detect.legacy_rule = rule
    return Ladder(detect=detect, tidy_gates=frozenset(tidy_gates), provisional=True, note=note)


def tidy_gates(exp_id=None) -> frozenset:
    """The tidy-up gates of an experiment: TIDY_GATES plus its Ladder.tidy_gates (TIDY_GATES alone without exp_id)."""
    if exp_id is None:
        return TIDY_GATES
    from benchmark.evaluation.process_rules import LADDERS
    ladder = LADDERS.get(exp_id)
    return TIDY_GATES | (ladder.tidy_gates if ladder else frozenset())


def hit(tick, basis, flags=()):
    return Hit(True, tick, basis, list(flags))


def miss(basis, flags=()):
    return Hit(False, None, basis, list(flags))


class ProcessRun:
    """The process record of one run (captures/<env>/process), read lazily."""

    def __init__(self, root, public=None):
        self.root = Path(root)
        self.episode = json.loads((self.root / "episode.json").read_text())
        # The public task the Agent was given (public_parameters, public_constants), when the caller has it.
        self.public = public or {}
        self.manifest_problems = []
        manifest_path = self.root / "manifest.json"
        if manifest_path.is_file():
            for relative, digest in json.loads(manifest_path.read_text()).items():
                path = self.root / relative
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    self.manifest_problems.append(relative)
        else:
            self.manifest_problems.append("manifest.json")

    @property
    def exp_id(self):
        return self.episode["exp_id"]

    @property
    def physics_hz(self):
        return self.episode["physics_hz"]

    @property
    def end_tick(self):
        return self.episode["end_tick"]

    @cached_property
    def evidence(self) -> dict:
        path = self.root / "evidence.json"
        return json.loads(path.read_text()) if path.is_file() else {}

    @cached_property
    def logs(self) -> dict:
        path = self.root / "bench_logs.json"
        return json.loads(path.read_text()) if path.is_file() else {}

    @property
    def terminal_state(self) -> dict:
        return self.episode.get("terminal_state") or {}

    @property
    def device(self) -> dict:
        return self.episode.get("device") or {}

    @property
    def calls(self) -> list:
        """[(tool, tick before, tick after, ok, error type)] of every call, in order."""
        return [tuple(c) for c in self.episode.get("calls") or []]

    def private_names(self):
        return sorted(p.name for p in (self.root / "private").glob("*")) if (self.root / "private").is_dir() else []

    def jsonl(self, name="physics.jsonl"):
        """Rows of a private JSON-lines log (stored gzipped), in order; empty when the log is absent."""
        path = self.root / "private" / (name + ".gz")
        if not path.is_file():
            return
        with gzip.open(path, "rt") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    @cached_property
    def frames(self) -> list:
        return list(self.jsonl("physics.jsonl"))

    def private_json(self, name):
        path = self.root / "private" / name
        return json.loads(path.read_text()) if path.is_file() else None


def find_process(trial_dir):
    found = [p for p in Path(trial_dir).glob("captures/*/process") if (p / "episode.json").is_file()]
    return found[0] if found else None


def prefix_level(hits: dict) -> int:
    """The number of consecutive levels reached from L1."""
    level = 0
    for name in LEVELS:
        if not (name in hits and hits[name].reached):
            break
        level += 1
    return level


def _within_tolerance(e) -> bool:
    import math
    return isinstance(e, (int, float)) and (e <= 1 or math.isclose(e, 1, rel_tol=0, abs_tol=1e-12))


def judge_submission(evaluation: dict | None, submitted: bool, tick, tidy: frozenset) -> tuple[Hit, Hit]:
    """(L5, L6) from the evaluator's report of the sealed submission (None when nothing was evaluated).

    L5: every gate that is not a tidy-up gate passes (valid_submission, within_budget, sealed_record_intact and the
    experiment's measurement gates) and e <= 1. L6: L5 and every tidy-up gate passes."""
    if evaluation is None:
        why = "no submission" if not submitted else "submission not evaluated"
        return miss(why), miss(why)
    failed = [k for k, v in (evaluation.get("gates") or {}).items() if not v]
    measured = [k for k in failed if k not in tidy]
    e = evaluation.get("e")
    if measured:
        l5 = miss(f"failed measurement gates: {measured}")
    elif not _within_tolerance(e):
        l5 = miss("no error (submission not scored)" if e is None else f"every measurement gate passes, e = {e:.3f} > 1")
    else:
        l5 = hit(tick, f"every measurement gate passes, e = {e:.3f} <= 1")
    tidy_failed = [k for k in failed if k in tidy]
    if not l5.reached:
        l6 = miss("L5 not reached" + (f"; failed tidy-up gates: {tidy_failed}" if tidy_failed else ""))
    elif tidy_failed:
        l6 = miss(f"failed tidy-up gates: {tidy_failed}")
    else:
        l6 = hit(tick, "every tidy-up gate passes")
    return l5, l6


def score_run(process_dir, evaluation: dict | None = None, public: dict | None = None) -> dict:
    """The six levels, the level and the process score of one run.

    evaluation: the evaluator's report of the sealed submission (evaluation.json), None when nothing was submitted.
    public: the public task the Agent was given (task_spec.json), for rules that check published values."""
    from benchmark.evaluation.process_rules import LADDERS

    run = ProcessRun(process_dir, public)
    ladder = LADDERS.get(run.exp_id)
    if ladder is None:
        raise KeyError(f"no process ladder for {run.exp_id}")
    found = dict(ladder.detect(run) or {})
    unknown = sorted(set(found) - set(DETECTED))
    if unknown:
        raise ValueError(f"{run.exp_id}: detectors returned {unknown}; only {list(DETECTED)} are per-experiment")
    hits = {name: found.get(name) or miss("not judged") for name in DETECTED}
    submitted = bool(run.episode.get("submitted"))
    tidy = TIDY_GATES | ladder.tidy_gates
    hits["L5"], hits["L6"] = judge_submission(evaluation, submitted, run.episode.get("submit_tick"), tidy)
    prefix = prefix_level(hits)
    level, flags, fallback = prefix, [], []
    if hits["L5"].reached and prefix < 5:
        # A reasonable method measured accurately reaches L5: a missed earlier level is a detector defect to review.
        fallback = [n for n in DETECTED if not hits[n].reached]
        level = 6 if hits["L6"].reached else 5
        flags.append("monotone_fallback:" + ",".join(fallback))
    else:
        beyond = [n for n in LEVELS[prefix:] if hits[n].reached]
        if beyond:
            flags.append("reached_beyond_prefix:" + ",".join(beyond))
    if evaluation is not None and bool(hits["L6"].reached) != bool(evaluation.get("task_completed")):
        flags.append(f"l6_differs_from_task_completed:{evaluation.get('task_completed')}")
    if run.manifest_problems:
        flags.append(f"process_manifest_problems:{run.manifest_problems}")
    return {
        "scorer": SCORER, "scheme": SCHEME, "exp_id": run.exp_id, "instance_id": run.episode.get("instance_id"),
        "episode_id": run.episode.get("episode_id"), "end_reason": run.episode.get("end_reason"),
        "submitted": submitted, "end_tick": run.end_tick,
        "levels": {n: hits[n].as_dict(run.physics_hz) for n in LEVELS},
        "level": level, "process_score": level_score(level), "flags": flags, "monotone_fallback": fallback,
        "ladder": {"provisional": ladder.provisional, "note": ladder.note,
                   "tidy_gates": sorted(ladder.tidy_gates)},
    }


def level_score(level: int) -> float:
    """The process score of a level: the cumulative weight of L1..L<level> out of 100 (L0 = 0, L6 = 1)."""
    return sum(LEVEL_WEIGHTS[name] for name in LEVELS[:level]) / 100


def public_task(trial_dir, instance_id, store=STORE):
    """The public task of a run: the trial workspace's task_spec.json, else the instance record's public part."""
    spec = Path(trial_dir) / "workspace" / "task_spec.json"
    if spec.is_file():
        return json.loads(spec.read_text())
    snapshot = Path(trial_dir) / "instance.json"     # the run's own instance record (benchmark.instances.snapshot)
    if snapshot.is_file():
        return json.loads(snapshot.read_text()).get("public_task")
    record = Path(store) / f"{instance_id}.json"
    return json.loads(record.read_text()).get("public_task") if record.is_file() else None


def score_trial(trial_dir, store=STORE) -> dict:
    """Score a trial directory (trial.py layout: captures/, evaluation.json, workspace/task_spec.json)."""
    trial_dir = Path(trial_dir)
    process = find_process(trial_dir)
    if process is None:
        raise FileNotFoundError(f"{trial_dir}: no process record (run made before the record existed, or not stopped)")
    evaluation_path = trial_dir / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.is_file() else None
    instance_id = json.loads((process / "episode.json").read_text()).get("instance_id")
    return score_run(process, evaluation, public_task(trial_dir, instance_id, store))


def summarize(scores: list[dict], kits: dict[str, str]) -> dict:
    """Experiment means over seeds, the mean over experiments (each counts equally), and per-discipline means.

    scores: score_run outputs; kits: exp_id -> discipline. The funnel is the share of runs at each level or above."""
    by_exp: dict[str, list] = {}
    for s in scores:
        by_exp.setdefault(s["exp_id"], []).append(s)
    experiments = {}
    for exp, rows in sorted(by_exp.items()):
        experiments[exp] = {
            "n": len(rows), "kit": kits.get(exp), "process_score": sum(r["process_score"] for r in rows) / len(rows),
            "funnel": {m: sum(r["level"] > i for r in rows) / len(rows) for i, m in enumerate(LEVELS)}}
    disciplines = {}
    for kit in DISCIPLINES:
        members = [v for v in experiments.values() if v["kit"] == kit]
        if members:
            disciplines[kit] = {
                "experiments": len(members),
                "process_score": sum(m["process_score"] for m in members) / len(members),
                "funnel": {m: sum(x["funnel"][m] for x in members) / len(members) for m in LEVELS}}
    overall = (sum(v["process_score"] for v in experiments.values()) / len(experiments)) if experiments else None
    funnel = ({m: sum(v["funnel"][m] for v in experiments.values()) / len(experiments) for m in LEVELS}
              if experiments else None)
    return {"scorer": SCORER, "process_score": overall, "funnel": funnel, "disciplines": disciplines,
            "experiments": experiments}


def kit_of(exp_id):
    from benchmark.registry.loader import load
    return load(exp_id).raw["kit"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trial", help="one trial directory")
    group.add_argument("--summary", help="a directory of trial directories")
    parser.add_argument("--store", default=str(STORE), help="instance store, for the public task of runs without a workspace")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.trial:
        report = score_trial(args.trial, args.store)
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            marks = " ".join(f"{m}{'+' if report['levels'][m]['reached'] else '-'}" for m in LEVELS)
            print(f"{report['exp_id']}: level {report['level']} ({report['process_score']:.2f})  {marks}  "
                  f"{' '.join(report['flags'])}")
        return 0
    scores, skipped = [], []
    for trial in sorted(p for p in Path(args.summary).iterdir() if p.is_dir()):
        try:
            scores.append(score_trial(trial, args.store))
        except FileNotFoundError:
            skipped.append(trial.name)
    report = summarize(scores, {s["exp_id"]: kit_of(s["exp_id"]) for s in scores})
    report["skipped_without_process_record"] = skipped
    print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else
          f"{len(scores)} runs, process score {report['process_score']}, funnel {report['funnel']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

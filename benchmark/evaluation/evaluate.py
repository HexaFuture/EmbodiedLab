"""Offline evaluation service (layer 6): sealed episode + submission + truth -> evaluation.json.

    python -m benchmark.evaluation.evaluate --sealed <dir> --instance <store>/<instance_id>.json [--out evaluation.json]

Gates run first (all declared gates, in spec order; valid_submission is always
assessed first because the other gates read submission fields), then scoring:
    e_rel = primary relative error (max over targets for max_rel_error; none for composite scores)
    e     = e_rel / threshold, or for composite scores the largest error / its own threshold
    reward = G * max(0, 1 - e);  task_completed = G and e <= 1 (G includes the tidy-up gates the card asks for)
`quantities` lists every scored quantity with the submitted value, the truth, the absolute
and the relative error, the metric and threshold it is scored with, and error / threshold.
`bench_restore` says which loose items are back where they started (descriptive; it enters no score).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from benchmark.registry.loader import load
from benchmark.evaluation import gates as gate_library
from benchmark.evaluation.sealed import load_sealed
from benchmark.evaluation.bench_restore import bench_restore

EVALUATOR = "embodiedlab-eval/1"
SCHEMA = "embodiedlab.evaluation/v2"


def _rel(submitted, truth):
    return abs(submitted - truth) / abs(truth)


def quantity(name, submitted, truth, metric, threshold, error=None):
    """One scored quantity with both error forms; `metric` says which of them is held to `threshold`."""
    submitted, truth = float(submitted), float(truth)
    absolute = abs(submitted - truth)
    relative = absolute / abs(truth) if truth else None
    if error is None:
        error = absolute if metric == "abs_error" else relative
    return {"name": name, "submitted": submitted, "truth": truth, "abs_error": absolute, "rel_error": relative,
            "metric": metric, "threshold": threshold, "normalized_error": error / threshold}


def score(spec, hidden, submission) -> dict:
    scoring = spec.scoring
    field = scoring.get("field", spec.raw["submission"]["primary"]["field"])
    threshold = scoring["threshold"]
    if spec.id == "circuit_blackbox3":
        from benchmark.evaluation.circuit_network import parameter_errors
        errors = parameter_errors(hidden, submission)
        correct = submission["topology"] == hidden["topology"]
        quantities = [{"name": "topology", "submitted": submission["topology"], "truth": hidden["topology"],
                       "correct": correct, "metric": "categorical", "threshold": None,
                       "normalized_error": 0.0 if correct else 1.0 / threshold}]
        if correct:
            quantities += [quantity(key, value, hidden[key], "rel_error", threshold, error)
                           for (key, value), error in zip(submission["parameters"].items(), errors)]
    elif scoring["metric"] == "rel_error":
        errors = [_rel(submission[field], hidden[scoring["target"]])]
        quantities = [quantity(scoring["target"], submission[field], hidden[scoring["target"]], "rel_error",
                               threshold, errors[0])]
    elif scoring["metric"] == "max_rel_error":
        values = [submission[f] for f in scoring["fields"]] if "fields" in scoring else submission[field]
        errors = [_rel(v, hidden[t]) for v, t in zip(values, scoring["target"])]
        quantities = [quantity(t, v, hidden[t], "rel_error", threshold, error)
                      for v, t, error in zip(values, scoring["target"], errors)]
    elif scoring["metric"] == "composite":
        quantities = [quantity(c["target"], submission[c["field"]], hidden[c["target"]], c["metric"], c["threshold"])
                      for c in scoring["components"]]
        # Mixed units (absolute and relative errors) have no single relative error; e is the worst error / threshold.
        return {"relative_errors": None, "e_rel": None,
                "e": max(q["normalized_error"] for q in quantities), "quantities": quantities}
    else:
        raise NotImplementedError(f"metric {scoring['metric']} is not implemented yet")
    e_rel = max(errors)
    return {"relative_errors": errors, "e_rel": e_rel, "e": e_rel / threshold, "quantities": quantities}


def diagnostics(spec, ctx, gate_details) -> dict:
    wanted = set(spec.scoring.get("diagnostics", []))
    out = {}
    if not ctx.submission_valid:
        return out
    if "reading_diagnostics" in wanted:
        rows = []
        for row in ctx.submission["measurements"]:
            record = ctx.sealed.evidence.get(row["evidence_id"]) or {}
            auto = record.get("automatic_reading_cm")
            rows.append({"evidence_id": row["evidence_id"], "submitted_reading_cm": row["reading_cm"],
                         "automatic_reading_cm": auto,
                         "absolute_difference_cm": None if auto is None else abs(row["reading_cm"] - auto)})
        diffs = [r["absolute_difference_cm"] for r in rows if r["absolute_difference_cm"] is not None]
        out["reading_diagnostics"] = rows
        out["max_reading_difference_cm"] = max(diffs) if diffs else None
        # Attribution threshold (3.7): |Agent reading - diagnostic reading| > 3 mm => perception.
        out["perception_reading_flag"] = bool(diffs) and max(diffs) > 0.3
    detail = gate_details.get("submission_self_consistent", {})
    if "fit_r2" in wanted and "fit_r2" in detail:
        out["fit_r2"] = detail["fit_r2"]
    if "regression_intercept_s2" in wanted and "intercept_s2" in detail:
        out["regression_intercept_s2"] = detail["intercept_s2"]
    if "zero_intercept_g" in wanted:
        rows = ctx.submission["measurements"]
        sxy = sum(r["L_eff_m"] * r["period_s"] ** 2 for r in rows)
        sxx = sum(r["L_eff_m"] ** 2 for r in rows)
        out["zero_intercept_g"] = 4 * math.pi ** 2 * sxx / sxy if sxy > 0 else None
    return out


def evaluate(sealed_dir, instance: dict) -> dict:
    spec = load(instance["exp_id"])
    sealed = load_sealed(sealed_dir)
    if spec.id == "projectile_landing":
        from benchmark.evaluation.projectile import evaluate_projectile
        report = evaluate_projectile(spec, sealed, instance)
        report["bench_restore"] = bench_restore(sealed.episode.get("bench_order"))
        return report
    hidden = instance["hidden"]
    ctx = gate_library.Context(spec=spec, hidden=hidden, sealed=sealed)
    results, details = {}, {}
    binding = []
    if sealed.manifest_problems:
        binding.append(f"sealed manifest: {sealed.manifest_problems}")
    if sealed.episode.get("instance_id") != instance["instance_id"]:
        binding.append("sealed episode belongs to another instance")
    if sealed.episode.get("exp_id") != spec.id:
        binding.append("sealed episode belongs to another experiment")
    if sealed.episode.get("spec_digest") != instance["spec_digest"]:
        binding.append("sealed episode was run against a different spec version")
    ordered = sorted(spec.gates, key=lambda item: item[0] != "valid_submission")
    for name, params in ordered:
        passed, detail = gate_library.GATES[name](ctx, **params)
        results[name], details[name] = bool(passed), detail
        if name == "valid_submission":
            ctx.submission_valid = bool(passed)
    if binding:
        results["sealed_record_intact"], details["sealed_record_intact"] = False, {"problems": binding}
    G = all(results.values())
    if spec.id == "rectifier_supply" and ctx.submission_valid:
        from benchmark.evaluation.rectifier import score as rectifier_score
        scored = rectifier_score(hidden, ctx.submission, sealed)
    elif spec.id == "optical_relay" and ctx.submission_valid:
        from benchmark.evaluation.optics import relay_score
        scored = relay_score(spec, hidden, ctx.submission, sealed)
    else:
        scored = score(spec, hidden, ctx.submission) if ctx.submission_valid else None
    e = scored["e"] if scored else None
    reward = max(0.0, 1.0 - e) if G and scored else 0.0
    completed = bool(G and scored and (e <= 1 or math.isclose(e, 1, rel_tol=0, abs_tol=1e-12)))
    return {
        "schema_version": SCHEMA, "evaluator": EVALUATOR,
        "exp_id": spec.id, "spec_digest": spec.digest(),
        "instance_id": instance["instance_id"], "split": instance["split"],
        "key_id": instance.get("key_id"), "table_version": instance.get("table_version"),
        "episode_id": sealed.episode.get("episode_id"),
        "gates": {name: results[name] for name in results}, "gate_details": details, "G": G,
        "threshold": spec.scoring["threshold"], "metric": spec.scoring["metric"],
        "relative_errors": scored["relative_errors"] if scored else None,
        "e_rel": scored["e_rel"] if scored else None, "e": e,
        "quantities": scored["quantities"] if scored else None,
        "reward": reward, "task_completed": completed,
        "sim_s_used": sealed.episode["submit_tick"] / sealed.episode["physics_hz"],
        "diagnostics": diagnostics(spec, ctx, details),
        "bench_restore": bench_restore(sealed.episode.get("bench_order")),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sealed", required=True)
    parser.add_argument("--instance", required=True, help="materialized instance file (trusted store)")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    report = evaluate(args.sealed, json.loads(Path(args.instance).read_text()))
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n")
    summary = {k: report[k] for k in ("exp_id", "G", "e_rel", "e", "reward", "task_completed")}
    summary["failed_gates"] = [k for k, v in report["gates"].items() if not v]
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

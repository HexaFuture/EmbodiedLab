"""Attainment of an exploration run: how far the result got against the
instance's reference value, uncapped, beside the capped score (success, e and reward do not change).

    python -m benchmark.evaluation.attainment <evaluation.json or trial dir> ... [--json]

    experiment            metric            attainment       quantity of evaluation.json
    active_filter         filter_design     a - A_search (dB) attenuation_db (submitted a), the report's attainment
                                                             reference A_search (the board search; the quantity's truth
                                                             is the success line A_ref)
    vibration_absorber    absorber_tuning   A* / A           peak_mm (submitted A, truth A*)
    fiber_coupling        fiber_coupling    eta / eta*       eta (core coupling)
    reaction_yield        reaction_yield    y / y*           yield
    truss_bridge          truss_design      H / P*           held_load_N

At the reference the attainment is 0 dB (the filter) or 1 (the others); above it, a better result than the reference.
It is read from the scored `quantities` (and, where the score names one, its attainment reference: active_filter,
whose success line and ranking reference differ), so it can be recomputed offline from evaluation.json (no hidden
values needed). Runs without a score (no valid submission) have none.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# metric -> (quantity name, kind): kind "difference_db" = submitted - truth; "ratio" = submitted / truth;
# "inverse_ratio" = truth / submitted (smaller is better)
RULES = {
    "filter_design": ("attenuation_db", "difference_db"),
    "absorber_tuning": ("peak_mm", "inverse_ratio"),
    "fiber_coupling": ("eta", "ratio"),
    "reaction_yield": ("yield", "ratio"),
    "truss_design": ("held_load_N", "ratio"),
}


def attainment(metric, quantities, reference=None):
    """{'value', 'unit', 'achieved', 'reference', 'quantity'} or None (not an exploration metric, or not scored).
    reference: the score's attainment reference when it differs from the quantity's truth (else the truth)."""
    rule = RULES.get(metric)
    if rule is None or not quantities:
        return None
    name, kind = rule
    q = next((x for x in quantities if x.get("name") == name), None)
    if q is None or q.get("submitted") is None or q.get("truth") is None:
        return None
    if kind == "difference_db" and reference is None:     # active_filter without its ranking reference
        return None
    achieved, reference = float(q["submitted"]), float(q["truth"] if reference is None else reference)
    if kind == "difference_db":
        value, unit = achieved - reference, "dB"
    elif kind == "ratio":
        value, unit = (achieved / reference if reference else None), "ratio"
    else:
        value, unit = (reference / achieved if achieved else None), "ratio"
    if value is not None and not math.isfinite(value):
        value = None if math.isnan(value) else value
    return {"value": value, "unit": unit, "achieved": achieved, "reference": reference, "quantity": name}


def of_report(report):
    """Attainment of an evaluation.json report (its own field when present, else computed from its quantities)."""
    if "attainment" in report:
        return report["attainment"]
    return attainment(report.get("metric"), report.get("quantities"))


def _report_path(path):
    p = Path(path)
    return p / "evaluation.json" if p.is_dir() else p


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="evaluation.json files or trial directories")
    parser.add_argument("--json", action="store_true", help="one JSON object per line")
    args = parser.parse_args(argv)
    for path in args.paths:
        rp = _report_path(path)
        try:
            report = json.loads(rp.read_text())
        except (OSError, ValueError) as exc:
            print(json.dumps({"path": str(path), "error": f"{type(exc).__name__}: {exc}"}))
            continue
        row = {"path": str(path), "exp_id": report.get("exp_id"), "metric": report.get("metric"),
               "task_completed": report.get("task_completed"), "attainment": attainment(report.get("metric"),
                                                                                       report.get("quantities"))}
        if args.json:
            print(json.dumps(row, ensure_ascii=False))
        else:
            a = row["attainment"]
            print(f"{row['exp_id']}\t{'-' if a is None else a['value']}\t{'' if a is None else a['unit']}\t{path}")


if __name__ == "__main__":
    main()

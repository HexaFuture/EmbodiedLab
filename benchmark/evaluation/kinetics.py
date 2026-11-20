"""kinetics_crystal_violet: evidence and self-consistency of timed absorbance runs (crystal violet + OH-).

Gates of the orders task (the card gives no rate law, the agent reports the orders m, n and k):
- kinetic_orders_correct: the reported integer orders equal the bench's rate law, rate = k [CV+]^1 [OH-]^1
  (benchmark/core/chem_containers.py react); a wrong order is a wrong categorical answer, G = 0 (as the topology of
  circuit_blackbox3). k is then scored against the hidden k_M_s, which is defined with these orders.
- kinetic_records_valid: every run a quantity names in derived_from is a cited run, and the cited points include two
  cuvettes holding both crystal violet and NaOH (in one run or in different runs; grouped by the cuvette read and its
  dispensing history six levels) whose [OH-] (from the private nominal dispensing history and the
  labels, the cuvette not spilled or overflowed) differ: the higher at least oh_margin_pct % above the lower.

The older kinetic_runs gate below (a refit of k from the rate law) is not used by the spec .
A cited run is a KIN run of this episode at a wavelength where a water BLANK was taken; every used point was read
with the same cuvette in the well (>= 2 mL, mixed, not overflowed, nothing added during the run). The run's
[OH-] and [CV] follow from that cuvette's nominal dispensing history and the labels. Refit: per run the least-
squares slope of ln A against t over the used points gives k' (s^-1); k = mean of k' / [OH-].
"""
from __future__ import annotations

import math

STATUS_PROBLEM = {"slot_empty": "no cuvette in the well", "low_volume": "less than the minimum reading volume",
                  "not_mixed": "cuvette not mixed", "no_blank_at_λ": "no BLANK at the reading wavelength",
                  "blank_ref_not_solvent": "BLANK reference was not water only", "overflowed": "cuvette overflowed"}


def _mix(history):
    volumes = {}
    for source, mL in history:
        volumes[source] = volumes.get(source, 0.0) + float(mL)
    return volumes


def slope(ts, ys):
    n = len(ts)
    mt, my = sum(ts) / n, sum(ys) / n
    stt = sum((t - mt) ** 2 for t in ts)
    if stt <= 0:
        raise ValueError("all points at the same time")
    return sum((t - mt) * (y - my) for t, y in zip(ts, ys)) / stt


def kinetic_runs(ctx, min_runs=2, min_oh_ratio=1.5, min_points=10, min_decay=0.5, min_A=0.02, composition_tol=0.01,
                 self_consistency_tol=0.03):
    from benchmark.evaluation.gates import _need_submission      # the gate library imports this module
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    sub, device = ctx.submission, ctx.sealed.episode["device"]
    lines = device["source_of"]
    problems, refits, ohs = [], [], []
    used = [r for r in sub["runs"] if r["used_in_fit"]]
    if len(used) < min_runs:
        problems.append(f"{len(used)} runs used, need at least {min_runs}")
    if len({r["evidence_id"] for r in used}) != len(used):
        problems.append("a run is cited more than once")
    min_read = device["min_read_mL"]
    for run in used:
        label = f"run {run['evidence_id']}"
        record, problem = ctx.evidence(run["evidence_id"])
        if problem or record.get("kind") != "kinetic_run":
            problems.append(f"{label}: {problem or 'not a KIN run'}")
            continue
        public = record["public"]
        if public["blank_wavelength_nm"] != public["wavelength_nm"]:
            problems.append(f"{label}: no BLANK at the run's wavelength")
        rows = {r["point"]: r for r in public["rows"]}
        private = {p["point"]: p for p in record["private_points"]}
        points = sorted(set(run["points"]))
        if len(points) != len(run["points"]):
            problems.append(f"{label}: a point is listed twice")
        if len(points) < min_points:
            problems.append(f"{label}: {len(points)} points used, need at least {min_points}")
        missing = [p for p in points if p not in rows]
        if missing:
            problems.append(f"{label}: points {missing[:5]} are not in the run")
            continue
        if not points:
            continue
        cuvettes = {private[p]["cuvette"] for p in points}
        histories = {tuple(map(tuple, private[p]["nominal_history"])) for p in points}
        if len(cuvettes) != 1 or len(histories) != 1:
            problems.append(f"{label}: the used points are not one cuvette with one composition")
            continue
        for p in points:
            status = private[p]["status"]
            if status != "ok":
                problems.append(f"{label} point {p}: {STATUS_PROBLEM.get(status, status)}")
                break
            if private[p]["spilled"] or private[p]["volume_mL"] is None or private[p]["volume_mL"] < min_read - 1e-9:
                problems.append(f"{label} point {p}: less than the minimum reading volume or spilled")
                break
        mix = _mix(private[points[0]]["nominal_history"])
        total = sum(mix.values())
        if total <= 0:
            problems.append(f"{label}: empty cuvette"); continue
        c_oh = device["naoh_label_M"] * mix.get(lines["NaOH"], 0.0) / total
        c_cv = device["c_stock_uM_label"] * mix.get(lines["dye"], 0.0) / total
        for name, claimed, value in (("c_OH_M", run["c_OH_M"], c_oh), ("c_CV_uM", run["c_CV_uM"], c_cv)):
            if abs(claimed - value) > composition_tol * value + 1e-9:
                problems.append(f"{label}: {name} {claimed} disagrees with the dispensing history ({value:.5g})")
        if c_oh <= 0:
            problems.append(f"{label}: no NaOH in the cuvette"); continue
        a = [rows[p]["A_display"] for p in points]
        t = [rows[p]["t_s"] for p in points]
        if min(a) < min_A:
            problems.append(f"{label}: a used reading is below A = {min_A} (noise floor)"); continue
        if a[-1] > (1 - min_decay) * a[0]:
            problems.append(f"{label}: the used readings decay by less than {min_decay:.0%}")
        k_prime = -slope(t, [math.log(v) for v in a])
        refits.append({"evidence_id": run["evidence_id"], "c_OH_M": c_oh, "k_prime_s": k_prime, "k_M_s": k_prime / c_oh})
        ohs.append(c_oh)
    if len(ohs) >= 2 and max(ohs) < min_oh_ratio * min(ohs) - 1e-12:
        problems.append(f"the runs' [OH-] differ by less than a factor {min_oh_ratio}")
    detail = {"refits": refits}
    if not problems and refits:
        k = sum(r["k_M_s"] for r in refits) / len(refits)
        detail["refit_k_M_s"] = k
        if k <= 0:
            problems.append("refit k is not positive")
        elif abs(sub["k_M_s"] - k) / k > self_consistency_tol:
            problems.append(f"k_M_s {sub['k_M_s']} differs from the refit {k:.5g} by more than {self_consistency_tol:.0%}")
    detail["problems"] = problems
    return not problems, detail


TRUE_ORDERS = {"order_CV": 1, "order_OH": 1}         # the bench's rate law: rate = k [CV+][OH-]


def kinetic_orders_correct(ctx):
    from benchmark.evaluation.gates import _need_submission      # the gate library imports this module
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    wrong = sorted(f for f, true in TRUE_ORDERS.items() if ctx.submission[f] != true)
    return not wrong, {"problems": [f"{f} is not the order of the reaction" for f in wrong]}


def kinetic_records_valid(ctx, oh_margin_pct=20):
    from benchmark.evaluation.gates import _need_submission
    from benchmark.evaluation.chem_gates import _lost, composition, measures_object
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    sub, exp, device = ctx.submission, ctx.spec.id, ctx.sealed.episode["device"]
    cited = [run["evidence_id"] for run in sub["runs"]]
    problems = [f"derived_from.{q}: {eid} is not a cited run" for q, ids in sub["derived_from"].items()
                for eid in ids if eid not in cited]
    levels, runs_of = {}, {}           # (cuvette, dispensing history) of the cited points -> [OH-]; -> cited runs
    for run in sub["runs"]:
        record, problem = ctx.evidence(run["evidence_id"])
        if problem is not None or record.get("kind") != "kinetic_run":
            continue                    # spectro_readings_genuine reports it
        private = {p["point"]: p for p in record["private_points"]}
        for point in run["points"]:
            p = private.get(point["point"])
            if p is None or _lost(p) or not p["nominal_history"]:
                continue
            if measures_object(exp, device, p["nominal_history"], p["cuvette"]):
                key = (p["cuvette"], tuple(tuple(h) for h in p["nominal_history"]))
                levels[key] = round(composition(exp, device, p["nominal_history"])["c_OH_M"], 9)
                runs_of.setdefault(key, set()).add(run["evidence_id"])
    factor = 1 + oh_margin_pct / 100
    pairs = [(a, b) for a in levels for b in levels if a != b and levels[b] >= factor * levels[a]]
    if not pairs:
        problems.append(f"no two cited cuvettes holding crystal violet and NaOH whose [OH-] differ by "
                        f"{oh_margin_pct}% or more")
    names = {key: f"cuvette {key[0]} #{k}" for k, key in enumerate(levels, start=1)}
    return not problems, {"problems": problems,
                          "oh_levels_M": {names[key]: [v] for key, v in levels.items()},
                          "runs": {names[key]: sorted(v) for key, v in runs_of.items()}}


# --- tidy-up: no KIN run still going when the episode ends (rule "recording stopped") ---------------
MEASURE_S, SCAN_S, KIN_INTERVAL_S, KIN_DURATION_S = 2.0, 15.0, 2.0, 300.0     # benchmark/core/chem_spectro.py


def kin_running_at_end(episode):
    """(running, problem) from the sealed bench records: the spectrophotometer keys in the bench's action log replayed
    with the instrument's rules (KIN starts a run unless a BLANK / READ / SCAN is still measuring, KIN during a run
    ends it, a run ends by itself after KIN_DURATION_S); the replayed run count must match the recorded KIN runs.
    (None, None) when the episode carries no bench action log."""
    device = episode.get("device") or {}
    actions, chemistry = device.get("actions"), device.get("chemistry") or {}
    if actions is None or "kinetic_private" not in chemistry:
        return None, None
    hz = episode.get("physics_hz", 240)
    start, busy_until, runs = None, float("-inf"), 0
    for action in sorted((a for a in actions if str(a.get("button", "")).startswith("spectro:")),
                         key=lambda a: a["tick"]):
        t, kind = action["tick"] / hz, action["button"].split(":", 1)[1]
        if start is not None and t >= start + KIN_DURATION_S - 1e-9:
            start = None
        if start is not None:
            if kind == "kin":
                start = None
            continue
        if t < busy_until - 1e-9:
            continue
        if kind == "kin":
            start, runs = t, runs + 1
        else:
            busy_until = t + (SCAN_S if kind == "scan" else MEASURE_S)
    recorded = chemistry["kinetic_private"]
    if runs != len(recorded):
        return None, f"the KIN key presses ({runs} runs) do not replay the {len(recorded)} recorded KIN runs"
    if start is not None and len(recorded[-1]["points"]) >= round(KIN_DURATION_S / KIN_INTERVAL_S):
        start = None                                                  # its full duration was taken
    return start is not None, None

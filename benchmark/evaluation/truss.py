"""truss_bridge gates and score (the pegboard; the Agent loads its own bridge).

The terminal state is the bench when the loading started (the first weight on the platform, truss_frame.LoadMonitor),
or at submission when nothing was loaded; the device record's `loading` holds the loading (held load, failure,
touches on the bridge).

- truss_tests_genuine: (kept for older episodes; not in the gate list) `tests` lists at least one
  tester run, each row citing an observation at which the display showed DONE for that run.
- truss_mass_limit: the bridge in the terminal state (the bars with both eyes on seated pins) weighs at most
  mass_limit_g (the kit's nominal masses, truss_model).
- truss_bench_tidy: in the terminal state no bar is in the tester, every intact bar outside the bridge lies in a rack
  slot, every failed bar lies in the scrap tray, every loose pin stands in the pin stand or in a board hole, and the
  robot holds nothing but a weight.
- truss_hands_off: from the loading's start nothing touched a bar or a pin of the bridge: no robot link, nothing the
  robot held, no weight (truss_frame contact view, every tick).

Score (metric truss_design): H is the held load, the largest settled pull of the rope while the bridge was intact
(the backend's reading: the platform still for 0.5 s, nothing of the robot on the platform or its weights); no loading,
H = 0. P* is the largest failure load of any bridge the kit builds on the board within the mass limit at the kinds'
nominal (hidden) properties (truss_optimum). e = max(0, 1 - H / P*) / threshold. P (the bridge's failure load at each
bar's own properties) and the loading loss 1 - H / P are diagnostics.

P* depends on the hidden properties only. It is looked up first in an optional table
experiments/truss_bridge/pstar_table.json (not distributed), then in a cache file (EMBODIEDLAB_TRUSS_PSTAR_CACHE, default
~/.cache/embodiedlab/truss_pstar.json); both are keyed by the nominal properties and the model version. Only an
instance in neither is solved (truss_optimum: deterministic node and iteration limits, no wall clock), and cached.
`python -m benchmark.evaluation.truss prewarm <instance.json> ...` fills the cache; `... table <instance.json> ...`
writes that table.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from benchmark.core import truss_model as M

MODEL_VERSION = "truss-v2.2"   # four kinds; at most one bar per pin pair
TABLE = Path(__file__).resolve().parents[2] / "experiments" / "truss_bridge" / "pstar_table.json"


def _where(ctx):
    state = ctx.sealed.episode.get("terminal_state") or {}
    return state.get("where"), state


def bridge_bars(ctx):
    """[(bar, (hole, hole))] of the bridge at submission."""
    where, _ = _where(ctx)
    return sorted((bar, tuple(w[1])) for bar, w in (where or {}).items() if w[0] == "bridge")


def bridge_load(ctx):
    """(P, governing (bar, mode) or None, mass g) of the bridge at submission."""
    device = ctx.sealed.episode["device"]
    bars, failed = device["bars"], device.get("failed") or {}
    members = bridge_bars(ctx)
    mass = sum(M.bar_mass_g(bars[b]["kind"], int(bars[b]["label"])) for b, _ in members)
    intact = [(b, pins) for b, pins in members if b not in failed]
    P, which = M.capacity([{"pins": pins, "label": int(bars[b]["label"]), "T_N": bars[b]["T_N"],
                            "P100_N": bars[b]["P100_N"], "N_N": bars[b]["N_N"]} for b, pins in intact])
    governing = None if which is None else (intact[which[0]][0], which[1])
    return float(P), governing, mass


def nominal_properties(hidden):
    return {kind: {"T_N": float(hidden[f"T_{kind}_N"]), "P100_N": float(hidden[f"P100_{kind}_N"]),
                   "N_N": float(hidden[f"N_{kind}_N"])} for kind in M.KINDS}


def _cache_path():
    return Path(os.environ.get("EMBODIEDLAB_TRUSS_PSTAR_CACHE")
                or Path.home() / ".cache" / "embodiedlab" / "truss_pstar.json")


def _cache_key(props):
    text = json.dumps({"model": MODEL_VERSION, "props": {k: {q: round(v, 9) for q, v in sorted(p.items())}
                                                          for k, p in sorted(props.items())}}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _cache_read():
    return _read(_cache_path())


def _cache_write(key, value):
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _cache_read()
        data[key] = value
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".truss_pstar.")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=0, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        pass                              # an unwritable cache only costs the solve again


def optimum(hidden):
    """{'P_N', 'design', 'certified', 'stages'} at the kinds' nominal properties (cached)."""
    props = nominal_properties(hidden)
    key = _cache_key(props)
    for source, table in (("table", _read(TABLE)), ("cache", _cache_read())):
        if key in table:
            return dict(table[key], source=source)
    q = solve(props)
    _cache_write(key, q)
    return dict(q, source="solved")


def solve(props):
    from benchmark.core import truss_optimum as O
    q = O.optimum(props)
    return {"P_N": q["P_N"], "design": [list(d) for d in q["design"] or []], "certified": q["certified"],
            "deterministic": q.get("deterministic", True), "stages": q["stages"]}


def truss_tests_genuine(ctx, peak_tol_n=0.2):
    from benchmark.evaluation.gates import _need_submission
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    tol = float(ctx.param(peak_tol_n)) + 1e-9
    problems, rows, runs = [], [], set()
    for i, row in enumerate(ctx.submission["tests"]):
        eid = row["evidence_id"]
        record, problem = ctx.evidence(eid)
        if problem is None and record.get("kind") != "bench_view":
            problem = f"{eid}: not an observation of this bench"
        if problem:
            problems.append(f"row {i}: {problem}")
            continue
        display, run = record.get("display") or {}, record.get("run") or {}
        entry = {"evidence_id": eid, "bar": row["bar"], "mode": row["mode"], "peak_N": row["peak_N"],
                 "shown": display, "run": run.get("run"), "run_bar": [run.get("kind"), run.get("label")],
                 "run_mode": run.get("mode"), "run_end": run.get("end")}
        rows.append(entry)
        if display.get("state") != "DONE" or not run:
            problems.append(f"row {i}: the tester did not show a finished run at {eid}")
            continue
        if run.get("run") in runs:
            problems.append(f"row {i}: run {run.get('run')} cited twice")
        runs.add(run.get("run"))
        kind, label = row["bar"].split()
        if run.get("kind") is None:
            problems.append(f"row {i}: no bar was mounted in that run")
        elif (run.get("kind"), int(run.get("label"))) != (kind, int(label)):
            problems.append(f"row {i}: that run tested a different bar")
        if run.get("mode") != row["mode"]:
            problems.append(f"row {i}: that run was in {run.get('mode')}, not {row['mode']}")
        shown = display.get("peak_N")
        if not isinstance(shown, (int, float)):
            problems.append(f"row {i}: the display showed no PEAK at {eid}")
        elif abs(float(row["peak_N"]) - shown) > tol:
            problems.append(f"row {i}: peak {row['peak_N']} N, the display showed {shown:.1f} N")
    if not rows:
        problems.append("no test cited")
    return not problems, {"problems": problems, "rows": rows}


def truss_mass_limit(ctx, mass_limit_g=50.0):
    where, _ = _where(ctx)
    if where is None:
        return False, {"problems": ["no record of the bench at submission"]}
    _, _, mass = bridge_load(ctx)
    limit = float(ctx.param(mass_limit_g))
    ok = mass <= limit + 1e-9
    return ok, {"mass_g": round(mass, 4), "limit_g": limit, "bars": [[b, list(p)] for b, p in bridge_bars(ctx)],
                "problems": [] if ok else [f"the bridge weighs {mass:.2f} g, more than {limit:g} g"]}


def truss_bench_tidy(ctx):
    where, state = _where(ctx)
    if not isinstance(where, dict) or not where:
        return False, {"problems": ["no record of the bench at submission"]}
    failed = (ctx.sealed.episode.get("device") or {}).get("failed") or {}
    pins = state.get("pins")
    problems = []
    for bar, (place, _) in sorted(where.items()):
        if place == "tester":
            problems.append("a bar is still in the tester")
        elif bar in failed and place != "tray":
            problems.append("a failed bar is not in the scrap tray")
        elif bar not in failed and place not in ("bridge", "rack"):
            problems.append("an intact bar outside the bridge is not in a rack slot")
    if not isinstance(pins, dict):
        problems.append("no record of the loose pins at submission")
    else:
        for pin, (place, _) in sorted(pins.items()):
            if place not in ("hole", "stand"):
                problems.append("a loose pin is neither in the pin stand nor in a board hole")
    held = [k for k in (state.get("held"), state.get("releasing")) if k is not None]
    if any(not k.startswith(("w2kg", "w05kg", "w01kg")) for k in held):
        problems.append("the robot holds something other than a weight")
    return not problems, {"problems": sorted(set(problems)), "where": where, "pins": pins, "failed": failed}


def _loading(ctx):
    return (ctx.sealed.episode.get("device") or {}).get("loading") or {}


def truss_hands_off(ctx):
    loading = _loading(ctx)
    if not loading.get("started"):
        return True, {"started": False, "touches": []}
    touches = loading.get("touches") or []
    problems = [f"{t['by']} touched {t['body']} at tick {t['tick']}" for t in touches[:20]]
    return not touches, {"started": True, "touches": touches[:50], "problems": problems}


def truss_score(spec, hidden, submission, ctx):
    threshold = float(spec.scoring["threshold"])
    best = optimum(hidden)
    P_star = float(best["P_N"])
    P, governing, mass = bridge_load(ctx)
    loading = _loading(ctx)
    H = float(loading.get("held_N") or 0.0) if loading.get("started") else 0.0
    ratio = min(1.0, H / P_star) if P_star > 0 else 0.0          # P* may be a hair low when it was not certified
    e = (1.0 - ratio) / threshold
    quantities = [
        {"name": "held_load_N", "submitted": H, "truth": P_star, "abs_error": P_star - H, "rel_error": 1 - ratio,
         "metric": "shortfall", "threshold": threshold, "normalized_error": e}]
    return {"relative_errors": None, "e_rel": None, "e": e, "quantities": quantities,
            "diagnostics": {"held_N": H, "P_N": P, "P_star_N": P_star, "H_over_P_star": H / P_star if P_star > 0 else None,
                            "P_over_P_star": P / P_star if P_star > 0 else None,
                            "loading_loss": 1 - H / P if P > 0 else None, "governing": governing,
                            "mass_g": round(mass, 4),
                            "loading": {k: loading.get(k) for k in ("started", "failed", "fail_pull_N", "peak_N",
                                                                   "placements", "capacity_N", "start_tick",
                                                                   "fail_tick")},
                            "optimum": {k: best.get(k) for k in ("design", "certified", "deterministic", "source")}}}


def _instances(paths):
    for p in paths:
        data = json.loads(Path(p).read_text())
        for inst in data if isinstance(data, list) else [data]:
            yield inst.get("instance_id", p), inst.get("hidden", inst)


def _prewarm(paths):
    """Solve and cache P* of every instance (json files or lists of instances with a 'hidden' dict)."""
    for name, hidden in _instances(paths):
        q = optimum(hidden)
        print(name, round(q["P_N"], 3), q["certified"], q["source"], flush=True)


def _table(paths):
    """Solve every instance afresh and add it to the table (sorted keys, one entry per instance)."""
    table = _read(TABLE)
    for name, hidden in _instances(paths):
        key = _cache_key(nominal_properties(hidden))
        if key not in table:
            table[key] = dict(solve(nominal_properties(hidden)), instance_id=name)
        print(name, round(table[key]["P_N"], 3), table[key]["certified"], flush=True)
        TABLE.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] in ("prewarm", "table"):
        (_prewarm if sys.argv[1] == "prewarm" else _table)(sys.argv[2:])
    else:
        sys.exit("usage: python -m benchmark.evaluation.truss prewarm|table <instance.json> ...")

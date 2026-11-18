"""vibration_absorber gates and score.

- absorber_sweep_genuine: sweep_evidence cites a sweep of this episode (kind absorber_sweep, its public record intact)
  that completed, and the absorber has not been changed since that sweep started (no configuration transition after
  its start: weights in the tray, the carrier's detent, the valve), so the bench at submission is the configuration it
  swept.
- absorber_readings_genuine: peak_reading cites an observation of this episode (get_observation's evidence_id) at
  which the analyzer's screen showed the finished sweep_evidence's PEAK (drawn, not mid-refresh), and
  peak_amplitude_mm equals that on-screen PEAK within PEAK_TOL_COUNTS counts of its last digit (0.001 mm). The
  analyzer's export holds no PEAK: the reading comes off the screen.
- absorber_configuration: the submitted configuration (total mass g, blade length mm, valve) is the configuration on the
  bench at submission (the sealed device record): the mass within mass_tol_g, the length and valve exactly; the carrier
  stands in a detent.
- absorber_bench_tidy: at submission no sweep runs, the robot holds nothing, and every weight not in the tray stands in
  a rack seat.

Score (metric absorber_tuning): A_final is the tip's peak over a sweep from rest of the configuration on the bench at
submission (absorber_model.sweep, the hidden values), A* the smallest peak over every configuration the bench allows
(absorber_model.optimum). e = max((max(A_final, A*) / A* - 1) / threshold, |peak_amplitude_mm - A_final| /
(reported_threshold A_final)).

A* depends on the hidden values only. It is looked up first in an optional table
experiments/vibration_absorber/astar_table.json (not distributed), then in a cache
file (EMBODIEDLAB_ABSORBER_ASTAR_CACHE, default ~/.cache/embodiedlab/absorber_astar.json); both are keyed by the
hidden values and the model version. Only an instance in neither is computed (the full grid, deterministic, about a
minute) and cached. `python -m benchmark.evaluation.absorber prewarm <instance.json> ...` fills the cache;
`... table <instance.json> ...` writes that table.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from benchmark.core import absorber_model as M

PEAK_COUNT_MM = 0.001              # the analyzer's PEAK: three decimals (mm)
PEAK_TOL_COUNTS = 2
TABLE = Path(__file__).resolve().parents[2] / "experiments" / "vibration_absorber" / "astar_table.json"


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _device(ctx):
    return ctx.sealed.episode.get("device") or {}


def _state(ctx):
    return ctx.sealed.episode.get("terminal_state") or {}


# --- A* ---------------------------------------------------------------------------------------------------------
def _cache_path():
    return Path(os.environ.get("EMBODIEDLAB_ABSORBER_ASTAR_CACHE")
                or Path.home() / ".cache" / "embodiedlab" / "absorber_astar.json")


def _cache_key(hidden):
    h = M.hidden_of(hidden)
    return _sha({"model": M.MODEL_VERSION, "hidden": {k: round(v, 12) for k, v in sorted(h.items())}})


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _cache_write(key, value):
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _read(path)
        data[key] = value
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".absorber_astar.")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=0, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        pass                              # an unwritable cache only costs the grid again


def solve(hidden):
    q = M.optimum(M.hidden_of(hidden))
    return {k: q[k] for k in ("A_star_mm", "mass_g", "length_mm", "valve", "edge")}


def optimum(hidden):
    """{'A_star_mm', 'mass_g', 'length_mm', 'valve', 'edge', 'source'} (table, cache or computed)."""
    key = _cache_key(hidden)
    for source, table in (("table", _read(TABLE)), ("cache", _read(_cache_path()))):
        if key in table:
            return dict(table[key], source=source)
    q = solve(hidden)
    _cache_write(key, q)
    return dict(q, source="computed")


def final_peak(hidden, configuration):
    """(peak mm, its frequency Hz) of a sweep from rest of the configuration {'mass_g', 'length_mm', 'valve'}."""
    peak, hz = M.sweep(M.hidden_of(hidden), configuration["mass_g"], configuration["length_mm"], configuration["valve"])
    return float(peak), float(hz)


# --- gates ---------------------------------------------------------------------------------------------------------
def sweep_record(ctx, evidence_id):
    """(record, problem) of a cited sweep: from this episode, kind absorber_sweep, its public record intact."""
    if not isinstance(evidence_id, str) or not evidence_id.startswith(ctx.sealed.episode_id + ":sweep:"):
        return None, f"{evidence_id}: not a sweep of this episode"
    record = ctx.sealed.evidence.get(evidence_id)
    if not isinstance(record, dict) or record.get("kind") != "absorber_sweep":
        return None, f"{evidence_id}: unknown sweep"
    public = record.get("public")
    if not isinstance(public, dict) or _sha(public) != record.get("sha256"):
        return None, f"{evidence_id}: payload hash mismatch"
    return record, None


def absorber_sweep_genuine(ctx):
    sub = ctx.submission if ctx.submission_valid else None
    if sub is None:
        return False, {"reason": "submission is not valid; gate cannot be assessed"}
    eid = sub["sweep_evidence"]
    record, problem = sweep_record(ctx, eid)
    if problem:
        return False, {"problems": [problem]}
    problems = []
    public = record["public"]
    if public.get("status") != "complete":
        problems.append(f"{eid}: the sweep did not complete ({public.get('status')})")
    device = _device(ctx)
    started = float(public.get("started_s", float("inf")))
    later = [t for t in device.get("transitions") or [] if float(t["t_s"]) > started]
    if later:
        problems.append(f"{eid}: the absorber was changed after the sweep started")
    if record.get("config") != {k: device.get("configuration", {}).get(k) for k in ("weights", "mass_g", "length_mm",
                                                                                     "valve")}:
        problems.append(f"{eid}: the bench at submission is not the configuration the sweep ran on")
    return not problems, {"problems": problems, "sweep": {k: public.get(k) for k in ("evidence_id", "status",
                                                                                    "started_s", "ended_s")},
                          "transitions_after": later}


def shown_value(cells, decimals=3):
    """The number a row of seven-segment cells shows (blank cells skipped, the point before the last `decimals`
    cells), or None (blank, Err)."""
    if not isinstance(cells, str) or not cells.strip() or not cells.replace(" ", "").isdigit():
        return None
    return int(cells.replace(" ", "")) / 10 ** decimals


def absorber_readings_genuine(ctx):
    sub = ctx.submission if ctx.submission_valid else None
    if sub is None:
        return False, {"reason": "submission is not valid; gate cannot be assessed"}
    eid, sweep = sub["peak_reading"]["evidence_id"], sub["sweep_evidence"]
    record, problem = ctx.evidence(eid)
    if problem is None and record.get("kind") != "bench_view":
        problem = f"{eid}: not an observation"
    if problem:
        return False, {"problems": [problem]}
    display = record.get("display") or {}
    drawn = (record.get("drawn") or {}).get("analyzer") or {}
    shown = shown_value(drawn.get("peak"))
    row = {"evidence_id": eid, "screen": {k: drawn.get(k) for k in ("peak", "hz", "run")},
           "sweep_shown": display.get("sweep"), "peak_shown_mm": shown}
    problems = []
    if display.get("sweep") != sweep:
        problems.append(f"{eid}: the screen did not show the result of {sweep}")
    elif display.get("state") != "done" or drawn.get("run") or shown is None:
        problems.append(f"{eid}: the screen did not show the finished sweep's PEAK")
    elif drawn.get("peak") != (record.get("analyzer") or {}).get("peak"):
        problems.append(f"{eid}: the screen had not refreshed to the sweep's PEAK")
    else:
        r = float(sub["peak_amplitude_mm"])
        if abs(r - shown) > PEAK_TOL_COUNTS * PEAK_COUNT_MM + 1e-9:
            problems.append(f"{eid}: peak_amplitude_mm {r} is not the {shown:.3f} mm PEAK the screen showed")
    return not problems, {"problems": problems, "reading": row}


def absorber_configuration(ctx, mass_tol_g=0.5):
    sub = ctx.submission if ctx.submission_valid else None
    if sub is None:
        return False, {"reason": "submission is not valid; gate cannot be assessed"}
    actual = _device(ctx).get("configuration") or {}
    stated = sub["configuration"]
    problems = []
    if actual.get("length_mm") is None:
        problems.append("the carrier does not stand in a detent")
    if actual.get("mass_g") is None or abs(float(stated["mass_g"]) - float(actual["mass_g"])) > float(ctx.param(mass_tol_g)):
        problems.append("the stated mass is not the mass in the tray")
    if actual.get("length_mm") is not None and int(stated["length_mm"]) != int(actual["length_mm"]):
        problems.append("the stated blade length is not the carrier's detent")
    if actual.get("valve") is None or int(stated["valve"]) != int(actual["valve"]):
        problems.append("the stated valve setting is not the valve's")
    return not problems, {"problems": problems, "stated": stated, "actual": actual}


def absorber_bench_tidy(ctx):
    state = _state(ctx)
    where = state.get("weight_seat")
    if not isinstance(where, dict) or not where:
        return False, {"problems": ["no record of the bench at submission"]}
    problems = []
    if state.get("running"):
        problems.append("a sweep is running")
    if state.get("held") is not None or state.get("releasing") is not None:
        problems.append("the robot holds something")
    for w, seat in sorted(where.items()):
        if seat is None or not (seat.startswith("rack:") or seat.startswith("tray:")):
            problems.append("a weight is neither in the tray nor in a rack seat")
    return not problems, {"problems": sorted(set(problems)), "weight_seat": where}


# --- score -------------------------------------------------------------------------------------------------------
def absorber_score(spec, hidden, submission, ctx):
    threshold, reported = float(spec.scoring["threshold"]), float(spec.scoring["reported_threshold"])
    best = optimum(hidden)
    A_star = float(best["A_star_mm"])
    actual = _device(ctx).get("configuration") or {}
    if actual.get("length_mm") is None or actual.get("mass_g") is None:
        A, hz = float("inf"), None
    else:
        A, hz = final_peak(hidden, actual)
    r = float(submission["peak_amplitude_mm"])
    shortfall = (max(A, A_star) / A_star - 1.0) / threshold
    prediction = abs(r - A) / (reported * A) if 0 < A < float("inf") else float("inf")
    e = max(shortfall, prediction)
    quantities = [
        {"name": "peak_mm", "submitted": A, "truth": A_star, "abs_error": A - A_star, "rel_error": A / A_star - 1.0,
         "metric": "excess", "threshold": threshold, "normalized_error": shortfall},
        {"name": "peak_amplitude_mm", "submitted": r, "truth": A, "abs_error": abs(r - A),
         "rel_error": abs(r - A) / A if 0 < A < float("inf") else None, "metric": "rel_error", "threshold": reported,
         "normalized_error": prediction}]
    return {"relative_errors": None, "e_rel": None, "e": e, "quantities": quantities,
            "diagnostics": {"A_final_mm": A, "A_final_hz": hz, "A_star_mm": A_star,
                            "A_over_A_star": A / A_star if A_star > 0 else None, "configuration": actual,
                            "optimum": {k: best.get(k) for k in ("mass_g", "length_mm", "valve", "edge", "source")}}}


# --- table and cache ---------------------------------------------------------------------------------------------
def _instances(paths):
    for p in paths:
        data = json.loads(Path(p).read_text())
        for inst in data if isinstance(data, list) else [data]:
            yield inst.get("instance_id", p), inst.get("hidden", inst)


def _prewarm(paths):
    for name, hidden in _instances(paths):
        q = optimum(hidden)
        print(name, round(q["A_star_mm"], 6), q["source"], flush=True)


def _table(paths):
    table = _read(TABLE)
    for name, hidden in _instances(paths):
        key = _cache_key(hidden)
        if key not in table:
            table[key] = dict(solve(hidden), instance_id=name)
        print(name, round(table[key]["A_star_mm"], 6), flush=True)
        TABLE.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] in ("prewarm", "table"):
        (_prewarm if sys.argv[1] == "prewarm" else _table)(sys.argv[2:])
    else:
        sys.exit("usage: python -m benchmark.evaluation.absorber prewarm|table <instance.json> ...")

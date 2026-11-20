"""fiber_coupling gates, score and process ladder.

Gates (card section 11):
- fiber_power_evidence: power_evidence cites an observation of this episode (kind bench_view), and the light path has
  not changed since it: no change of the laser, of a seat (the coupler's lens seat and cap slot, the target's slot)
  or of a cartridge's orientation after its tick, and every screw (two per mirror, the focus) at submission within
  screw_tol_deg (knob degrees) of its position then. A knob turned inside its backlash moves no screw.
- fiber_installed_as_stated: `lens` is the focal length printed on the cartridge in the coupler; the cap is not on the
  coupler; the laser is on.
- fiber_bench_tidy: at submission the robot holds nothing, every cartridge not in the coupler stands in a cartridge
  slot of the lens box, the cap stands in the box or lies elsewhere off the coupler (bench_in_order checks it is
  neither tipped over nor off the bench).

Score (metric fiber_coupling, card section 10): the bench at submission as sealed (the cartridge in the coupler and
its orientation, the five screws, the target) is recomputed from the hidden instance (fiber_model): eta_final the core
light of that light path (times the target's transmission while the target stands in the beam), P_final the power the
meter receives (core and cladding light, laser 1000 uW). eta* (the best of the five lenses at its best alignment and
focus, core light only) depends on model_seed only. e = max(max(0, 1 - eta_final / eta*) / threshold,
|power_uW - P_final| / (reported_threshold P_final)); success: eta_final >= (1 - threshold) eta* and the reported power
within reported_threshold of P_final.

eta* is looked up in an optional table experiments/fiber_coupling/eta_star_table.json (not distributed),
then in a cache file (EMBODIEDLAB_FIBER_ETA_CACHE, default ~/.cache/embodiedlab/fiber_eta_star.json), both keyed by
(MODEL_VERSION, model_seed); otherwise computed (about a second) and cached.
`python -m benchmark.evaluation.fiber table <instance.json | seed:N> ...` writes that table.

Ladder (fiber_coupling_levels; records only: the bench's actions, transitions and knob moves in bench_logs.json and
the observations in evidence.json; no hidden values, no eta*, no submission):
L1  the earliest of: POWER pressed; a holder set into or taken out of the coupler (lens seat, cap slot) or the
    target's slot; a knob turned by a degree or more.
L2  the coupling set up for the meter: the laser on, a cartridge in the coupler and the cap off.
L3  an observation in which the meter shows light coupled into the fiber (a reading above zero), at or after L2.
L4  observed readings above zero at two different alignments (exploration tier: no method prescribed):
    another cartridge or orientation in the coupler, or a screw (mirrors, focus) more than LADDER_SCREW_DEG of knob
    rotation from where it stood at the first reading; timed at the first observation of the second.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from benchmark.core import fiber_model as M

TABLE = Path(__file__).resolve().parents[2] / "experiments" / "fiber_coupling" / "eta_star_table.json"
SUCCESS_RATIO = 0.95
LENS_SEAT, CAP_SEAT, TARGET_SEAT = "coupler:lens", "coupler:cap", "bench:target"
OPTICS = (LENS_SEAT, CAP_SEAT, TARGET_SEAT)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _cache_path():
    return Path(os.environ.get("EMBODIEDLAB_FIBER_ETA_CACHE")
                or Path.home() / ".cache" / "embodiedlab" / "fiber_eta_star.json")


def _cache_key(hidden):
    return _sha({"model": M.MODEL_VERSION, "model_seed": int(hidden["model_seed"])})


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
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".fiber_eta_star.")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=0, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        pass


def solve(hidden):
    q = M.optimum(M.hidden_of(hidden))
    out = {"eta_star": q["eta_star"], "lens": q["lens"], "dz_m": q["dz"], "per_lens": q["per_lens"]}
    return json.loads(json.dumps(out, default=float))


def optimum(hidden):
    key = _cache_key(hidden)
    for source, table in (("table", _read(TABLE)), ("cache", _read(_cache_path()))):
        if key in table:
            return dict(table[key], source=source)
    q = solve(hidden)
    _cache_write(key, q)
    return dict(q, source="computed")


def error(eta, eta_star, threshold=1.0 - SUCCESS_RATIO):
    """Normalised shortfall of a coupled fraction against eta* (success: <= 1)."""
    return max(0.0, 1.0 - float(eta) / float(eta_star)) / float(threshold)


# --- the sealed bench ----------------------------------------------------------------------------------------------
def _device(ctx):
    return ctx.sealed.episode.get("device") or {}


def _state(ctx):
    return ctx.sealed.episode.get("terminal_state") or {}


def _optics(record):
    """The optical state of a record (an observation, a transition, the final snapshot): laser, seats, orientation."""
    seats = record.get("seats") or {}
    return bool(record.get("power")), tuple(seats.get(k) for k in OPTICS), bool(record.get("reversed"))


def installed_lens(device, final=None):
    """Index of the lens in the coupler at submission (None)."""
    final = final if final is not None else device.get("final") or {}
    oid = (final.get("seats") or {}).get(LENS_SEAT)
    cartridges = device.get("cartridges") or {}
    return int(cartridges[oid]) if oid in cartridges else None


def light_path(hidden, device):
    """(eta_final, P_final_uW, detail) of the bench at submission, recomputed from the hidden instance."""
    final = device.get("final") or {}
    lens = installed_lens(device, final)
    seats = final.get("seats") or {}
    detail = {"lens": lens, "reversed": bool(final.get("reversed")), "cap_on": seats.get(CAP_SEAT) is not None,
              "target_in": seats.get(TARGET_SEAT) is not None}
    if lens is None or detail["cap_on"]:
        return 0.0, 0.0, detail
    inst = M.hidden_of(hidden)
    screw = np.asarray(final["screw_deg"], float)
    px, py = M.screw_phis(inst, screw)
    dz = float(M.screw_dz(inst, lens, screw))
    if detail["reversed"]:
        dz -= 2 * float(device["lens_offsets_m"][lens])
    core, total = M.coupling(inst, lens, px, py, dz)
    T = float((final.get("light") or {}).get("target_T", 1.0)) if detail["target_in"] else 1.0
    detail.update(dz_m=dz, core=float(core), total=float(total), target_T=T,
                  sealed_core=(final.get("light") or {}).get("core"), sealed_total=(final.get("light") or {}).get("total"))
    return float(core) * T, M.P_LASER_UW * float(total) * T, detail


# --- gates ---------------------------------------------------------------------------------------------------------
def observation_record(ctx, evidence_id):
    """(record, problem) of a cited observation: one of this episode, kind bench_view."""
    if not isinstance(evidence_id, str) or not evidence_id.startswith(ctx.sealed.episode_id + ":obs:"):
        return None, f"{evidence_id}: not an observation of this episode"
    record = ctx.sealed.evidence.get(evidence_id)
    if not isinstance(record, dict) or record.get("kind") != "bench_view" or record.get("evidence_id") != evidence_id:
        return None, f"{evidence_id}: unknown observation"
    return record, None


def fiber_power_evidence(ctx, screw_tol_deg=0.05):
    sub = ctx.submission if ctx.submission_valid else None
    if sub is None:
        return False, {"reason": "submission is not valid; gate cannot be assessed"}
    eid = sub["power_evidence"]
    record, problem = observation_record(ctx, eid)
    if problem:
        return False, {"problems": [problem]}
    device = _device(ctx)
    final = device.get("final") or {}
    problems = []
    tick = int(record["tick"])
    then = _optics(record)
    changes = [t for t in device.get("transitions") or [] if int(t["tick"]) > tick and _optics(t) != then]
    if changes or _optics(final) != then:
        problems.append(f"{eid}: the laser, the cartridge, the cap or the target changed after this observation")
    tol = float(ctx.param(screw_tol_deg))
    moved = {}
    for name, a, b in zip(M.KNOBS, record.get("screw_deg") or [], final.get("screw_deg") or []):
        if abs(float(b) - float(a)) > tol:
            moved[name] = round(float(b) - float(a), 4)
    if len(record.get("screw_deg") or []) != 5 or len(final.get("screw_deg") or []) != 5:
        problems.append("no record of the adjusters")
    elif moved:
        problems.append(f"{eid}: an adjuster moved after this observation ({sorted(moved)})")
    return not problems, {"problems": problems, "observation_tick": tick, "submit_tick": final.get("tick"),
                          "meter_shown_uW": record.get("meter_uW"), "screw_moved_deg": moved,
                          "changes_after": changes[:5]}


def fiber_installed_as_stated(ctx, f_tol_mm=0.005):
    sub = ctx.submission if ctx.submission_valid else None
    if sub is None:
        return False, {"reason": "submission is not valid; gate cannot be assessed"}
    device = _device(ctx)
    final = device.get("final") or {}
    lens = installed_lens(device, final)
    text = device.get("lens_text") or []
    printed = float(text[lens]) if lens is not None and lens < len(text) else None
    problems = []
    if lens is None:
        problems.append("no cartridge in the coupler")
    elif abs(float(sub["lens"]) - printed) > float(ctx.param(f_tol_mm)):
        problems.append("the stated lens is not the cartridge in the coupler")
    if (final.get("seats") or {}).get(CAP_SEAT) is not None:
        problems.append("the cap is on the coupler")
    if not final.get("power"):
        problems.append("the laser is off")
    return not problems, {"problems": problems, "stated_mm": sub["lens"], "installed_mm": printed,
                          "reversed": bool(final.get("reversed"))}


def fiber_bench_tidy(ctx):
    state = _state(ctx)
    where = state.get("holder_seat")
    if not isinstance(where, dict) or not where:
        return False, {"problems": ["no record of the bench at submission"]}
    cartridges = _device(ctx).get("cartridges") or {}
    problems = []
    if state.get("held") is not None:
        problems.append("the robot holds something")
    for oid in sorted(cartridges):
        seat = where.get(oid)
        slot = isinstance(seat, str) and seat.startswith("box:") and seat[4:].isdigit()
        if seat != LENS_SEAT and not slot:
            problems.append(f"{oid}: neither in the coupler nor in a slot of the lens box")
    cap = (_device(ctx).get("final") or {}).get("seats", {}).get(CAP_SEAT)
    if cap is not None:
        problems.append("the cap is on the coupler")
    return not problems, {"problems": problems, "holder_seat": where}


# --- score -------------------------------------------------------------------------------------------------------
def fiber_score(spec, hidden, submission, ctx):
    threshold, reported = float(spec.scoring["threshold"]), float(spec.scoring["reported_threshold"])
    best = optimum(hidden)
    eta_star = float(best["eta_star"])
    device = _device(ctx)
    if int(device.get("model_seed", -1)) != int(hidden["model_seed"]):
        eta, P, detail = 0.0, 0.0, {"problem": "the sealed bench is another instance's"}
    else:
        eta, P, detail = light_path(hidden, device)
    r = float(submission["power_uW"])
    shortfall = error(eta, eta_star, threshold)
    prediction = abs(r - P) / (reported * P) if P > 0 else float("inf")
    e = max(shortfall, prediction)
    quantities = [
        {"name": "eta", "submitted": eta, "truth": eta_star, "abs_error": eta_star - eta,
         "rel_error": 1.0 - eta / eta_star, "metric": "shortfall", "threshold": threshold,
         "normalized_error": shortfall},
        {"name": "power_uW", "submitted": r, "truth": P, "abs_error": abs(r - P),
         "rel_error": abs(r - P) / P if P > 0 else None, "metric": "rel_error", "threshold": reported,
         "normalized_error": prediction}]
    return {"relative_errors": None, "e_rel": None, "e": e, "quantities": quantities,
            "diagnostics": {"eta_final": eta, "eta_star": eta_star, "eta_over_eta_star": eta / eta_star,
                            "P_final_uW": P, "light_path": detail,
                            "optimum": {k: best.get(k) for k in ("lens", "dz_m", "per_lens", "source")}}}


# --- process ladder ------------------------------------------------------------------------------------------------
LADDER_SCREW_DEG = 0.05             # L4: two alignments differ by more than this on a screw (fiber_power_evidence's tol)


def fiber_coupling_levels(run):
    """L1-L4 of fiber_coupling (see the module docstring)."""
    from benchmark.evaluation.process import hit, miss
    logs = run.logs or {}
    actions, transitions = logs.get("actions") or [], logs.get("transitions") or []
    acts = [(int(a["tick"]), "POWER pressed") for a in actions if a.get("action") == "power"]
    start = (transitions[0].get("seats") or {}) if transitions else {}
    for t in transitions[1:]:
        seats = t.get("seats") or {}
        if any(seats.get(k) != start.get(k) for k in OPTICS):
            acts.append((int(t["tick"]), f"holders {seats}"))
            break
    moves = logs.get("knob_moves") or []
    if moves:
        acts.append((int(moves[0]["tick"]), f"knob {moves[0]['knobs']} turned"))
    hits = {"L1": hit(*min(acts)) if acts else miss("no POWER press, no holder moved, no knob turned")}

    def after(level, tick):
        return tick if level.tick is None else max(level.tick, tick)

    ready = [int(t["tick"]) for t in transitions
             if t.get("power") and (t.get("seats") or {}).get(LENS_SEAT) and not (t.get("seats") or {}).get(CAP_SEAT)]
    hits["L2"] = (hit(after(hits["L1"], ready[0]), "laser on, a cartridge in the coupler, the cap off") if ready
                  else miss("never the laser on with a cartridge in the coupler and the cap off"))
    seen = []                                 # (tick, evidence id, cartridge, alignment) of observed readings above zero
    for eid, record in (run.evidence or {}).items():
        if record.get("kind") != "bench_view" or not float(record.get("meter_uW") or 0) > 0:
            continue
        lens = (record.get("seats") or {}).get(LENS_SEAT)
        seen.append((int(record["tick"]), eid, lens, (lens, bool(record.get("reversed")), record.get("screw_deg") or [])))
    seen.sort()
    l3 = [x for x in seen if hits["L2"].reached and x[0] >= hits["L2"].tick]
    hits["L3"] = (hit(after(hits["L2"], l3[0][0]), f"{l3[0][1]}: the meter reads {l3[0][2]} light")
                  if l3 else miss("no observation of a meter reading above zero"))
    hits["L4"] = miss("no readings above zero observed at two different alignments")

    def differ(a, b):
        return a[:2] != b[:2] or len(a[2]) != len(b[2]) or any(abs(float(u) - float(v)) > LADDER_SCREW_DEG
                                                                for u, v in zip(a[2], b[2]))
    if hits["L3"].reached:
        first = l3[0][3]
        other = next((x for x in l3 if differ(x[3], first)), None)
        if other is not None:
            hits["L4"] = hit(after(hits["L3"], other[0]), f"{other[1]}: a reading at another alignment than {l3[0][1]}")
    return hits


# --- table ---------------------------------------------------------------------------------------------------------
def _instances(args):
    for a in args:
        if a.startswith("seed:"):
            yield a, {"model_seed": int(a[5:])}
            continue
        data = json.loads(Path(a).read_text())
        for inst in data if isinstance(data, list) else [data]:
            yield inst.get("instance_id", a), inst.get("hidden", inst)


def _table(args):
    table = _read(TABLE)
    for name, hidden in _instances(args):
        key = _cache_key(hidden)
        if key not in table:
            table[key] = dict(solve(hidden), instance_id=name, model_seed=int(hidden["model_seed"]))
        print(name, round(table[key]["eta_star"], 5), table[key]["lens"], flush=True)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "table":
        _table(sys.argv[2:])
    else:
        sys.exit("usage: python -m benchmark.evaluation.fiber table <instance.json | seed:N> ...")

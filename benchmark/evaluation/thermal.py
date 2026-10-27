"""Thermal kit gates (calorimetry_specific_heat, thermistor_beta, rod_expansion).

    thermal_readings_genuine    calorimetry: every reading cites a record of the recorder of this episode (kind
                                thermo_record, as its latest export before the submission returned it) and a window
                                of it, and its T_C lies within the recorded values of its column in that window;
                                dry block: every point cites a different observation of this episode (get_observation's
                                evidence_id, kind bench_view) and its readings equal what the displays showed at that
                                observation (the latest refresh, private) within one unit of the last displayed digit
                                (thermistor: T_C the controller's BLOCK reading, R_ohm the ohmmeter on its range then,
                                OL is no reading; rod: T_C either thermometer, the probe's PROBE or the controller's
                                BLOCK (the point's optional T_source "probe" / "block" names which), indicator_mm the
                                indicator)
    thermal_measurement_valid   calorimetry: the sample entered the water (a real drop, the bench's place transitions)
                                between the water windows, after the heater window, and stays in the water through the
                                after window; no cited record has to run through the drop (six levels:
                                a record stopped before the drop and another started after it is as good); dry block:
                                at the observation of every point used in the fit the probe stands in the block's well
                                (private bench state); rod: only for a point whose T_C is the probe's reading (the rod
                                follows the block, so a BLOCK reading needs no probe in the well)
    thermal_bench_restored      tidy-up at submission: calorimetry: the sample stands in a seat (the heater well or
                                the cup collar), the stirrer is off, the recorder is not recording, the heater set
                                point is back at its minimum and the hand is empty; dry block: the probe stands in its
                                own stand, the set point is back at its minimum, the ohmmeter (thermistor) is back in
                                autorange and the hand is empty

How many points to use, and whether to correct for heat loss, is the Agent's choice; the error decides.
"""
from __future__ import annotations

EPS = 1e-6
CALOR_READINGS = {"heater": "heater_C", "water_before": "T_C", "water_after": "T_C"}
# dry block: submitted field -> (displayed values in the observation record's `shown` it may equal, last-digit unit or
# its key); rod_expansion: T_C is either thermometer, PROBE or BLOCK (six levels)
DISPLAYED = {"thermistor_beta": {"T_C": (("block_C",), 0.01), "R_ohm": (("R_ohm",), "R_resolution_ohm")},
             "rod_expansion": {"T_C": (("probe_C", "block_C"), 0.01), "indicator_mm": (("indicator_mm",), 0.001)}}
T_SOURCES = {"probe": "probe_C", "block": "block_C"}     # rod_expansion: a point's optional T_source -> its display


def _need(ctx):
    from benchmark.evaluation.gates import _need_submission
    return _need_submission(ctx)


# --- calorimetry: the recorder ------------------------------------------------------------------------------------
def _record(ctx, rid):
    """(record, problem) of a cited recorder record, as its latest export before the submission returned it."""
    record, problem = ctx.evidence(rid)
    if problem:
        return None, problem
    if record.get("kind") != "thermo_record":
        return None, f"{rid}: not a record of the recorder"
    submit = ctx.sealed.episode.get("submit_tick")
    if record.get("export_tick") is not None and submit is not None and record["export_tick"] > submit:
        return None, f"{rid}: exported after the submission"
    return record, None


def _window(record, t_from, t_to):
    """(public rows, private rows) of the record within [t_from, t_to]."""
    rows, private = record["public"]["rows"], record.get("private") or []
    keep = [k for k, r in enumerate(rows) if t_from - EPS <= r["t_s"] <= t_to + EPS]
    return [rows[k] for k in keep], [private[k] for k in keep if k < len(private)]


def _within(label, rows, field, value, problems):
    values = [r[field] for r in rows if r.get(field) is not None]
    if not values:
        problems.append(f"{label}: no {field} recorded in the window")
    elif not min(values) - EPS <= value <= max(values) + EPS:
        problems.append(f"{label}: {field} {value} lies outside the recorded values in the window "
                        f"({min(values)}-{max(values)})")


def _drops(ctx):
    """Times at which the sample entered the water (transitions into the cup)."""
    return [t["t_s"] for t in ctx.sealed.episode["device"]["transitions"] if t["place"] == "cup"]


# --- dry block: the observations ------------------------------------------------------------------------------------
def _observation(ctx, eid):
    record, problem = ctx.evidence(eid)
    if problem:
        return None, problem
    if record.get("kind") != "bench_view":
        return None, f"{eid}: not an observation of this bench"
    return record, None


def _points(ctx):
    """[(label, point, observation record or None, problem)] of the submitted points."""
    out, seen = [], set()
    for k, p in enumerate(ctx.submission["points"], start=1):
        eid = p["evidence_id"]
        record, problem = _observation(ctx, eid)
        if problem is None and eid in seen:
            problem = f"{eid}: cited twice"
        seen.add(eid)
        out.append((f"point {k} ({eid})", p, record, problem))
    return out


def _shown_as(ctx, p, field, shown):
    """(the displays of `shown` that p[field] equals within one unit of their last digit, the displays it may equal)."""
    keys, unit = DISPLAYED[ctx.spec.id][field]
    if field == "T_C" and p.get("T_source") in T_SOURCES and T_SOURCES[p["T_source"]] in keys:
        keys = (T_SOURCES[p["T_source"]],)
    tol = shown.get(unit) if isinstance(unit, str) else unit
    if not isinstance(tol, (int, float)):
        return [], keys
    return [k for k in keys if isinstance(shown.get(k), (int, float))
            and abs(p[field] - shown[k]) <= float(tol) + EPS], keys


# --- readings genuine ---------------------------------------------------------------------------------------------
def thermal_readings_genuine(ctx):
    if (invalid := _need(ctx)) is not None:
        return invalid
    sub, problems = ctx.submission, []
    if ctx.spec.id == "calorimetry_specific_heat":
        for name, field in CALOR_READINGS.items():
            r = sub["readings"][name]
            record, problem = _record(ctx, r["record_id"])
            if problem:
                problems.append(f"{name}: {problem}")
                continue
            if r["t_to_s"] < r["t_from_s"]:
                problems.append(f"{name}: the window ends before it starts")
                continue
            rows, _ = _window(record, r["t_from_s"], r["t_to_s"])
            _within(name, rows, field, r["T_C"], problems)
        return not problems, {"problems": problems}
    rows = []
    for label, p, record, problem in _points(ctx):
        if problem:
            problems.append(f"{label}: {problem}")
            continue
        shown = record.get("shown") or {}
        entry = {"evidence_id": p["evidence_id"], "shown": shown}
        for field in DISPLAYED[ctx.spec.id]:
            matched, keys = _shown_as(ctx, p, field, shown)
            values = [shown.get(k) for k in keys if isinstance(shown.get(k), (int, float))]
            if not values:
                problems.append(f"{label}: the display showed no {field} reading at that observation")
            elif not matched:
                showed = " or ".join(f"{v}" for v in values)
                problems.append(f"{label}: {field} {p[field]}, the display showed {showed}")
        rows.append(entry)
    return not problems, {"problems": problems, "points": rows}


# --- measurement valid ---------------------------------------------------------------------------------------------
def thermal_measurement_valid(ctx):
    if (invalid := _need(ctx)) is not None:
        return invalid
    sub, problems, detail = ctx.submission, [], {}
    if ctx.spec.id == "calorimetry_specific_heat":
        r = sub["readings"]
        for name in CALOR_READINGS:
            _, problem = _record(ctx, r[name]["record_id"])
            if problem:
                problems.append(f"{name}: {problem}")
        if problems:
            return False, {"problems": problems}
        detail["drops_s"] = _drops(ctx)
        # Any real drop counts, whether or not a cited record ran through it (six levels).
        between = [t for t in _drops(ctx) if r["water_before"]["t_to_s"] <= t + 1.0 and r["water_after"]["t_from_s"] >= t
                   and r["heater"]["t_to_s"] <= t + 1.0]
        if not between:
            problems.append("no drop lies between the water_before and water_after windows after the heater window")
        else:
            drop = between[-1]
            detail["drop_s"] = drop
            out = [t for t in ctx.sealed.episode["device"]["transitions"]
                   if drop < t["t_s"] <= r["water_after"]["t_to_s"] + EPS and t["place"] != "cup"]
            if out:
                problems.append(f"the sample left the water at {out[0]['t_s']:.1f} s, before the end of water_after")
        return not problems, detail | {"problems": problems}
    if not any(p["used_in_fit"] for p in sub["points"]):
        problems.append("no point is marked used_in_fit: the submission rests on no measured point")
    for label, p, record, problem in _points(ctx):
        if not p["used_in_fit"]:
            continue
        if record is None:                               # a citation that is no observation (readings_genuine says why)
            problems.append(f"{label}: {problem}")
            continue
        if ctx.spec.id == "rod_expansion":
            # The rod follows the block: a BLOCK reading needs no probe in the well, a PROBE reading does. A T_C that
            # equals neither display is readings_genuine's finding.
            matched, _ = _shown_as(ctx, p, "T_C", record.get("shown") or {})
            if "probe_C" not in matched or "block_C" in matched:
                continue
        if record.get("place") != "well":
            problems.append(f"{label}: the probe was not standing in the block's well at that observation")
    return not problems, {"problems": problems}


# --- tidy-up -------------------------------------------------------------------------------------------------------
def thermal_bench_restored(ctx, set_min_C=25.0):
    state = ctx.sealed.episode.get("terminal_state") or {}
    problems = []
    if state.get("held") is not None:
        problems.append(f"the hand holds the {state['held']}")
    if ctx.spec.id == "calorimetry_specific_heat":
        if state.get("sample_seat") not in ("heater", "cup"):
            problems.append("the sample does not stand in a seat (the heater well or the cup collar)")
        if state.get("stirring") is not False:
            problems.append("the stirrer is still running")
        if state.get("recording") is not False:
            problems.append("the recorder is still recording")
        if state.get("heater_set_C") is None or abs(state["heater_set_C"] - float(set_min_C)) > EPS:
            problems.append(f"the heater set point is {state.get('heater_set_C')} C, not back at {set_min_C:g} C")
    else:
        if state.get("probe_place") != "stand":
            problems.append("the probe is not back in its stand")
        if state.get("set_C") is None or abs(state["set_C"] - float(set_min_C)) > EPS:
            problems.append(f"the set point is {state.get('set_C')} C, not back at {set_min_C:g} C")
        if ctx.spec.id == "thermistor_beta" and state.get("auto") is not True:
            problems.append(f"the ohmmeter is not back in autorange (it holds the {state.get('range')} range)")
    keys = ("held", "sample_seat", "sample_place", "stirring", "recording", "heater_set_C", "probe_place", "set_C",
            "range", "auto")
    return not problems, {"problems": problems, "state": {k: state[k] for k in keys if k in state}}

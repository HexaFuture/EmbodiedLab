"""Process-score rules of the thermal kit: calorimetry_specific_heat, thermistor_beta, rod_expansion.

Records used (all exported at stop):
    private/physics.jsonl     the chemistry backend's 30 Hz frame rows (held object, positions xyz_mm, speed) and its
                              grasp events (grasp_attach / grasp_release / grasp_break with the object's world position)
    bench_logs.json           the bench's 1 Hz display refreshes (rows), the private state at each (private: where the
                              sample / probe is, the set point, the stirrer, the meter range, the true temperatures and,
                              calorimetry, the recorder's running record), the tick-exact place transitions and the
                              button actions
    evidence.json             dry block: the observations (kind bench_view: tick, what the displays showed, where the
                              probe stood, the true temperatures)
Places come from the object's pose as the backend reads it every tick: dry block 'well' / 'stand' / 'air', calorimeter
'heater' / 'cup' / 'air'. In a seat ('well', 'stand', 'heater') a held or tilted (> 10 deg) object reads 'air'; the
calorimeter's 'cup' is the sample's body centre inside the cup below the water line, at any tilt, held or not. A
grasp is therefore never inferred from a place; it is the physical grasp joint of the physics log, and the object
must then rise while held.
"""
from __future__ import annotations

import math

from benchmark.evaluation.process import hit, miss

LIFT_M = 0.01                 # a grasp counts once the held object rose this far (fallback: moved; seat tolerance 3-4 mm)
MOVING_M_S = 0.005            # held-object speeds below this are grip jitter (a still held probe reads ~0.002 m/s)
PROBE_SETTLED_K = 0.05        # probe within this of its steady temperature (block, plus the meter's self-heating)
ROD_SETTLED_K = 0.4           # M1: rod within this of the block
ROD_STABLE_K = 0.05           # M2: rod within this of the block (a stable indicator reading)
STABLE_ROWS, DISTINCT_K = 20, 5.0                   # M2: consecutive settled rows; two block temperatures this far apart
BASELINE_S, MIXING_S = 30.0, 60.0                   # M2: log before the drop, the sample in the water after it


# --- shared ---------------------------------------------------------------------------------------------------------
def _physics(run):
    """(frame rows, grasp events) of the episode proper: the backend settles the bench before the episode and then
    resets its clock to 0, so everything logged before the last clock reset is dropped. Frame rows get their tick."""
    step = run.physics_hz // 30
    frames, events, last = [], [], -1
    for row in run.frames:
        tick = row["frame"] * step if "frame" in row else row.get("tick")
        if tick is None:
            continue
        if tick < last:
            frames, events = [], []
        last = tick
        if "frame" in row:
            frames.append(row | {"tick": tick})
        elif row.get("event") in ("grasp_attach", "grasp_release", "grasp_break"):
            events.append(row)
    return frames, events


def _lifted(run, key):
    """The first tick at which `key` was grasped and lifted off where it stood, or None.

    A grasp is the backend's grasp joint (grasp_attach .. grasp_release / grasp_break). Lifted: a held frame shows the
    object LIFT_M above its height at the grasp (frame xyz_mm; the last frame at or before the attach, else the attach
    event's position). Records without xyz_mm (before it was logged) fall back to the held object's path: the 30 Hz
    frame speed of the bench's only movable object, integrated while held over frames where it moves at >=
    MOVING_M_S, or else the distance between its positions at attach and at release."""
    frames, events = _physics(run)
    heights = any("xyz_mm" in f for f in frames)
    step = run.physics_hz // 30
    opened = None
    for e in events + [{"event": "end", "tick": math.inf}]:
        if e["event"] == "grasp_attach" and e.get("object") == key:
            opened = e
            continue
        if opened is None or e["event"] == "grasp_attach":
            continue
        held = [f for f in frames if opened["tick"] < f["tick"] <= e["tick"] and f.get("held") == key]
        if heights:
            before = [f["xyz_mm"][key][2] for f in frames if f["tick"] <= opened["tick"] and key in f.get("xyz_mm", {})]
            base = before[-1] if before else (opened.get("object_world_m") or [0, 0, math.inf])[2] * 1000
            up = next((f for f in held if f.get("xyz_mm", {}).get(key, [0, 0, -math.inf])[2] - base >= LIFT_M * 1000),
                      None)
            if up:
                return up["tick"]
        else:
            path = 0.0
            for f in held:
                speed = f.get("max_speed_m_s") or 0.0
                path += speed * step / run.physics_hz if speed >= MOVING_M_S else 0.0
                if path >= LIFT_M:
                    return f["tick"]
            a, b = opened.get("object_world_m"), e.get("object_world_m")
            if a and b and math.dist(a, b) >= LIFT_M:
                return e["tick"]
        opened = None
    return None


def _bench(run, name):
    """A bench list from bench_logs.json, else the copy in the device record (transitions, actions)."""
    value = run.logs.get(name)
    return value if isinstance(value, list) else list(run.device.get(name) or [])


def _rows(run):
    """1 Hz display refreshes joined with the private state kept with them, each with its tick."""
    rows, private = _bench(run, "rows"), _bench(run, "private")
    return [r | p | {"tick": round(r["t_s"] * run.physics_hz)} for r, p in zip(rows, private)]


def _first_place(run, place):
    return next((t for t in _bench(run, "transitions") if t["place"] == place), None)


# --- calorimetry_specific_heat --------------------------------------------------------------------------------------
def calorimetry(run):
    """E1 the sample grasped (grasp joint) and raised >= 10 mm while held.
    E2 the sample first enters the cup (its body inside the cup below the water line, held or not; tick-exact).
    M1 the first recorded row (the recorder running, REC) with the sample in the cup after a drop that came after a
       >= 30 s baseline of the same record.
    M2 minimum data: one drop inside a record with >= 30 s of recorded readings before it and the record still running
       with the sample in the water 60 s after it (that record's row 60 s after such a drop). A repeated run (the
       sample re-heated and dropped again) counts the same way."""
    out = {}
    tick = _lifted(run, "sample")
    out["E1"] = hit(tick, "sample grasped and raised while held") if tick is not None else miss("sample never grasped and raised")
    transitions = _bench(run, "transitions")
    drops = [t for t in transitions if t["place"] == "cup"]
    out["E2"] = hit(drops[0]["tick"], "sample entered the cup") if drops else miss("sample never in the cup")
    rows = [r for r in _rows(run) if r.get("record") is not None]
    if not drops or not rows:
        return out | {"M1": miss("no drop" if not drops else "nothing recorded"), "M2": miss("no drop or no record")}
    starts = {}
    for r in rows:
        starts.setdefault(r["record"], r["t_s"])

    def recording_at(d):
        """The record that runs through drop d with >= BASELINE_S recorded before it, or None."""
        before = [r for r in rows if r["tick"] <= d["tick"]]
        if not before:
            return None
        n = before[-1]["record"]
        after = next((r for r in rows if r["tick"] > d["tick"]), None)
        if after is None or after["record"] != n or d["t_s"] - starts[n] < BASELINE_S:
            return None
        return n

    based = [(d, n) for d in drops if (n := recording_at(d)) is not None]
    first = next((r for d, n in based for r in rows if r["tick"] >= d["tick"] and r["record"] == n
                  and r["place"] == "cup"), None)
    out["M1"] = (hit(first["tick"], f"first recorded row in the cup after a >= {BASELINE_S:.0f} s recorded baseline")
                 if first else miss(f"no drop inside a record after {BASELINE_S:.0f} s of it"))
    best = None
    for d, n in based:
        left = [t for t in transitions if d["t_s"] < t["t_s"] <= d["t_s"] + MIXING_S and t["place"] != "cup"]
        done = next((r for r in rows if r["record"] == n and r["t_s"] - d["t_s"] >= MIXING_S), None)
        if not left and done is not None:
            best = (done, d)
            break
    out["M2"] = (hit(best[0]["tick"], f"drop at {best[1]['t_s']:.0f} s with {BASELINE_S:.0f} s recorded before and "
                                      f"{MIXING_S:.0f} s in the water after") if best
                 else miss(f"no drop with {BASELINE_S:.0f} s recorded before it and {MIXING_S:.0f} s recorded in the "
                           f"water after"))
    return out


# --- thermistor_beta, rod_expansion ---------------------------------------------------------------------------------
def _settled(r, rod, tol_rod):
    """A refresh row or an observation record (the reading shown: R_ohm in a row, shown.R_ohm in an observation)."""
    if r["place"] != "well":
        return False
    steady = r["block_C"] + (r.get("self_heating_K") or 0.0)
    if abs(r["probe_C"] - steady) > PROBE_SETTLED_K:
        return False
    if rod:
        return abs(r["rod_C"] - r["block_C"]) <= tol_rod
    return (r["shown"].get("R_ohm") if "shown" in r else r.get("R_ohm")) is not None


def _observations(run):
    """The observations (bench_view evidence) in time order."""
    return sorted((r for r in run.evidence.values() if r.get("kind") == "bench_view"), key=lambda r: r["tick"])


def dry_block(rod):
    def rule(run):
        """E1 the probe grasped (grasp joint) and raised >= 10 mm off its stand while held.
        E2 the probe first stands in the block's well (released, upright, seated; tick-exact).
        M1 the first observation (get_observation) with the probe in the well and within 0.05 K of its steady
           temperature (the block, plus the ohmmeter's self-heating) and a reading on the meter (rod_expansion: the rod
           within 0.4 K of the block).
        M2 minimum data: stable readings observed at two block temperatures at least 5 K apart: an observation made
           after >= 20 consecutive settled display refreshes with the probe in the well (rod_expansion: the rod within
           0.05 K of the block, a stable indicator reading), and itself settled; the tick of the second such
           observation."""
        out = {}
        tick = _lifted(run, "probe")
        out["E1"] = hit(tick, "probe grasped and raised while held") if tick is not None else miss("probe never grasped and raised")
        seated = _first_place(run, "well")
        out["E2"] = hit(seated["tick"], "probe placed in the well") if seated else miss("probe never in the well")
        rows, views = _rows(run), _observations(run)
        first = next((v for v in views if _settled(v, rod, ROD_SETTLED_K)), None)
        out["M1"] = (hit(first["tick"], "observed with the probe in the well and settled" + (", rod settled" if rod else ""))
                     if first else miss("no observation with the probe in the well and settled"))
        temps, tol = [], ROD_STABLE_K if rod else ROD_SETTLED_K
        for v in views:
            before = [r for r in rows if r["tick"] <= v["tick"]][-STABLE_ROWS:]
            if len(before) < STABLE_ROWS or not _settled(v, rod, tol) or not all(_settled(r, rod, tol) for r in before):
                continue
            if all(abs(v["block_C"] - t) >= DISTINCT_K for t in temps):
                temps.append(v["block_C"])
            if len(temps) >= 2:
                out["M2"] = hit(v["tick"], f"stable readings observed at {temps[0]:.1f} and {temps[1]:.1f} C")
                return out
        out["M2"] = miss(f"stable readings observed at {len(temps)} block temperature(s) {DISTINCT_K:.0f} K apart "
                         f"(needs 2)")
        return out
    return rule


RULES = {
    "calorimetry_specific_heat": calorimetry,
    "thermistor_beta": dry_block(rod=False),
    "rod_expansion": dry_block(rod=True),
}


# --- six levels (benchmark/evaluation/process.py) ---------------------------------------------------
# The legacy E1-M2 rules above (scorer embodiedlab-process/1) are kept for reference only: all three experiments'
# entries are replaced below by their own L1-L4 detectors (tidy-up gates: the defaults, bench_in_order and
# thermal_bench_restored).
from benchmark.evaluation.process import Ladder, from_legacy  # noqa: E402,F401

LADDERS = {exp_id: from_legacy(rule) for exp_id, rule in RULES.items()}


# --- six levels: the thermal kit's own detectors ---------------------------------------
MIXED_FRACTION = 0.05         # calorimetry L4: sample-water difference down to this share of its value at the drop
ROD_VALID_K = 0.4             # rod L3/L4: the rod within this of the block at the observation
ROD_APART_K = 5.0             # rod L4: two block temperatures this far apart (one set-point step)
NEAR_STEADY_K = 0.5           # thermistor L4 (i): the probe within this of its steady temperature (block + self-heating)
THERMISTOR_APART_K = 1.0      # thermistor L4: BLOCK readings this far apart (i) / spanned in one stay (ii)
DYNAMIC_VIEWS = 3             # thermistor L4 (ii): observations in one continuous stay in the well


def _num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _hands_on(run, key):
    """L1: the earliest of `key` grasped and raised >= 10 mm while held (the grasp joint), or any button press the
    bench logged (pressed to its switch depth; a press at the set point's limit counts too: it acted on the bench)."""
    found = []
    tick = _lifted(run, key)
    if tick is not None:
        found.append((tick, f"{key} grasped and raised while held"))
    presses = sorted(_bench(run, "actions"), key=lambda a: a["tick"])
    if presses:
        found.append((presses[0]["tick"], f"{str(presses[0].get('action')).upper()} pressed"))
    if not found:
        return miss(f"{key} never grasped and raised, no button pressed")
    return hit(*min(found))


# calorimetry_specific_heat -------------------------------------------------------------------------------------
def _calor_records(run):
    """Every recorder record as its latest export before the submission (or the end) returned it (evidence kind
    thermo_record): [(record id, rows)], each row its public reading joined with the private state logged with it
    (place, water_C, sample_C, tick) and `available`, the tick of the first export that carried it (the episode's
    `exports` list where it has one, else the record's latest export)."""
    limit = run.episode.get("submit_tick")
    limit = run.end_tick if limit is None else limit
    bench = _rows(run)
    out = []
    for rid, record in sorted(run.evidence.items()):
        if record.get("kind") != "thermo_record" or not _num(record.get("export_tick")):
            continue
        last = record["export_tick"]
        if limit is not None and last > limit:
            continue
        public = record.get("public") or {}
        rows = public.get("rows") or []
        private = record.get("private") or []
        if len(private) < len(rows) or not all("tick" in q for q in private[:len(rows)]):
            # the bench's own copy of the record's rows: the refreshes logged while it ran, in order
            private = [r for r in bench if r.get("record") == public.get("record")][:len(rows)]
            if len(private) < len(rows):
                continue
        ticks = sorted({e["tick"] for e in (run.episode.get("exports") or [])
                        if _num(e.get("tick")) and e["tick"] <= last
                        and (not e.get("evidence_ids") or rid in e["evidence_ids"])} | {last})
        joined = []
        for row, state in zip(rows, private):
            tick = state.get("tick", round(row["t_s"] * run.physics_hz))
            joined.append(row | {k: v for k, v in state.items() if k not in row} | {
                "tick": tick, "available": next(t for t in ticks if t >= tick)})
        out.append((rid, joined))
    return out


def calorimetry_levels(run):
    """L1-L4 of calorimetry_specific_heat.

    L1 the sample grasped and raised >= 10 mm while held, or STIR / UP / DN / REC pressed (the earliest).
    L2 the recorder recording while the sample stands in the heater well (released, upright) or lies in the water: the
       configuration in which the three readings (the sample's temperature from the block, the water before, the
       water after) can be recorded. Tick-exact from the REC presses and the place transitions.
    L3 the first recorded row in an L2 state that an export carried (timed at that export).
    L4 one real drop d (a transition into the cup) and, among the exported rows of any records: a heater row before d
       with the sample in the heater well and no other drop between; a water row before d with the sample out of the
       water and no other drop between; a row after d, the sample in the water throughout, at which the sample-water
       difference has fallen to 5 % of its value at d (mixing essentially complete; true bench temperatures). No
       count of records, no record through the drop, no baseline length, no correction required; timed at the latest
       of the three (at the export that carried it)."""
    out = {"L1": _hands_on(run, "sample")}
    transitions = sorted(_bench(run, "transitions"), key=lambda t: t["tick"])
    bench = _rows(run)
    place = bench[0].get("place", "heater") if bench else "heater"
    recording, l2 = False, None
    events = sorted([(a["tick"], 0, a) for a in _bench(run, "actions") if a.get("action") == "rec"]
                    + [(t["tick"], 1, t) for t in transitions], key=lambda e: (e[0], e[1]))
    for tick, kind, e in events:
        if kind == 0:
            recording = bool(e.get("recording"))
        else:
            place = e["place"]
        if recording and place in ("heater", "cup"):
            l2 = hit(tick, f"recording with the sample {'in the heater well' if place == 'heater' else 'in the water'}")
            break
    out["L2"] = l2 or miss("never recording with the sample in the heater well or in the water")
    rows = [r for _, rows in _calor_records(run) for r in rows]
    seated = [r for r in rows if r.get("place") in ("heater", "cup") and (l2 is None or r["tick"] >= l2.tick)]
    if l2 is None or not seated:
        out["L3"] = miss("no exported row recorded with the sample in the heater well or in the water")
        out["L4"] = miss("no exported row recorded in a set-up state")
        return out
    first = min(seated, key=lambda r: (r["available"], r["tick"]))
    out["L3"] = hit(first["available"], f"row at {first['t_s']:.0f} s ({first['place']}) exported")
    drops = [t["tick"] for t in transitions if t["place"] == "cup"]
    best = None
    for d in (t for t in transitions if t["place"] == "cup"):
        before_drop = [x for x in drops if x < d["tick"]]
        prior = before_drop[-1] if before_drop else -math.inf     # the previous drop: rows before it do not count
        leave = next((t["tick"] for t in transitions if t["tick"] > d["tick"] and t["place"] != "cup"), math.inf)
        gap0 = abs(d.get("sample_C", math.nan) - d.get("water_C", math.nan))
        heater = [r["available"] for r in rows if prior < r["tick"] < d["tick"] and r.get("place") == "heater"
                  and _num(r.get("heater_C"))]
        water0 = [r["available"] for r in rows if prior < r["tick"] < d["tick"] and r.get("place") != "cup"
                  and _num(r.get("T_C"))]
        mixed = [r["available"] for r in rows if d["tick"] <= r["tick"] < leave and r.get("place") == "cup"
                 and _num(r.get("T_C")) and _num(r.get("sample_C")) and _num(r.get("water_C")) and _num(gap0)
                 and abs(r["sample_C"] - r["water_C"]) <= MIXED_FRACTION * gap0]
        if heater and water0 and mixed:
            done = max(min(heater), min(water0), min(mixed))
            if best is None or done < best[0]:
                best = (done, d)
    out["L4"] = (hit(best[0], f"drop at {best[1]['t_s']:.0f} s with the block and the water recorded before it and "
                              f"the mixed water recorded after it")
                 if best else miss("no drop with the block and the water recorded before it and the water recorded "
                                   f"until the sample-water difference fell to {MIXED_FRACTION:.0%} after it"))
    return out


# rod_expansion -------------------------------------------------------------------------------------------------
def rod_levels(run):
    """L1-L4 of rod_expansion.

    L1 the probe grasped and raised >= 10 mm while held, or UP / DN pressed (the earliest).
    L2 set up from the start: the rod follows the block and the controller's BLOCK always shows the block (a probe
       in the well is another way, not needed); timed at L1, flagged state_at_start.
    V  a valid reading state at an observation: the indicator and BLOCK show a reading and the rod is within 0.4 K
       of the block (true bench state; the indicator then shows no further drift). BLOCK is always valid, so the
       thermometer the Agent read does not matter here (a PROBE reading's well condition is the gate's).
    L3 the first observation in V (one at room temperature at the start counts).
    L4 two observations in V at block temperatures >= 5 K apart. No count, order, heating or cooling, thermometer."""
    out = {"L1": _hands_on(run, "probe")}
    start = out["L1"].tick if out["L1"].reached else 0
    out["L2"] = hit(start, "set up at the start: the rod follows the block, BLOCK shows the block",
                    flags=("state_at_start",))
    valid = [v for v in _observations(run)
             if _num((v.get("shown") or {}).get("indicator_mm")) and _num((v.get("shown") or {}).get("block_C"))
             and _num(v.get("rod_C")) and _num(v.get("block_C")) and abs(v["rod_C"] - v["block_C"]) <= ROD_VALID_K]
    if not valid:
        out["L3"] = miss(f"no observation with the rod within {ROD_VALID_K} K of the block")
        out["L4"] = miss("no valid reading")
        return out
    out["L3"] = hit(max(valid[0]["tick"], start), f"observed at {valid[0]['t_s']:.0f} s with the rod at the block "
                                                   f"temperature ({valid[0]['block_C']:.1f} C)")
    for k, v in enumerate(valid):
        other = next((u for u in valid[:k] if abs(u["block_C"] - v["block_C"]) >= ROD_APART_K), None)
        if other is not None:
            out["L4"] = hit(max(v["tick"], start), f"rod at the block temperature observed at {other['block_C']:.1f} "
                                                   f"and {v['block_C']:.1f} C")
            return out
    out["L4"] = miss(f"valid readings at no two block temperatures {ROD_APART_K:.0f} K apart")
    return out


# thermistor_beta -----------------------------------------------------------------------------------------------
def thermistor_levels(run):
    """L1-L4 of thermistor_beta.

    L1 the probe grasped and raised >= 10 mm while held, or UP / DN / RNG / AUTO pressed (the earliest).
    L2 the probe first stands in the block's well (released, upright, seated; tick-exact).
    L3 the first observation with the probe in the well, a resistance (not OL) and BLOCK on the displays, and the
       refresh shown made after the probe last entered the well. No equilibrium required (when to read is the
       Agent's call; a dynamic method reads off equilibrium).
    L4 among the L3-valid observations either (i) two with BLOCK readings >= 1 K apart, the probe at each within
       0.5 K of its steady temperature (block + the meter's self-heating), or (ii) >= 3 in one continuous stay in
       the well spanning >= 1 K of BLOCK readings (B, R25 and the probe lag). No count, span, order, range, wait."""
    out = {"L1": _hands_on(run, "probe")}
    transitions = sorted(_bench(run, "transitions"), key=lambda t: t["tick"])
    seated = next((t for t in transitions if t["place"] == "well"), None)
    out["L2"] = hit(seated["tick"], "probe placed in the well") if seated else miss("probe never in the well")
    valid = []
    for v in _observations(run):
        shown = v.get("shown") or {}
        entry = next((t for t in reversed(transitions) if t["tick"] <= v["tick"]), None)
        if (v.get("place") == "well" and entry is not None and entry["place"] == "well"
                and _num(shown.get("R_ohm")) and _num(shown.get("block_C")) and _num(shown.get("t_s"))
                and shown["t_s"] >= entry["t_s"] - 1e-3):
            valid.append((v, entry["tick"]))
    if not valid:
        out["L3"] = miss("no observation of a reading refreshed with the probe in the well")
        out["L4"] = miss("no valid reading")
        return out
    out["L3"] = hit(valid[0][0]["tick"], f"reading observed with the probe in the well at {valid[0][0]['t_s']:.0f} s")

    def near(v):
        return (_num(v.get("probe_C")) and _num(v.get("block_C"))
                and abs(v["probe_C"] - v["block_C"] - (v.get("self_heating_K") or 0.0)) <= NEAR_STEADY_K)

    for k, (v, stay) in enumerate(valid):
        block = v["shown"]["block_C"]
        if near(v):
            other = next((u for u, _ in valid[:k] if near(u)
                          and abs(u["shown"]["block_C"] - block) >= THERMISTOR_APART_K), None)
            if other is not None:
                out["L4"] = hit(v["tick"], f"near-steady readings at BLOCK {other['shown']['block_C']:.2f} and "
                                           f"{block:.2f} C")
                return out
        same = [u["shown"]["block_C"] for u, s in valid[:k + 1] if s == stay]
        if len(same) >= DYNAMIC_VIEWS and max(same) - min(same) >= THERMISTOR_APART_K:
            out["L4"] = hit(v["tick"], f"{len(same)} readings in one stay in the well over BLOCK "
                                       f"{min(same):.2f}-{max(same):.2f} C")
            return out
    out["L4"] = miss(f"neither two near-steady readings {THERMISTOR_APART_K:g} K apart nor {DYNAMIC_VIEWS} readings "
                     f"spanning {THERMISTOR_APART_K:g} K in one stay in the well")
    return out


LADDERS["calorimetry_specific_heat"] = Ladder(detect=calorimetry_levels,
                                              note="six-level detectors")
LADDERS["rod_expansion"] = Ladder(detect=rod_levels, note="six-level detectors")
LADDERS["thermistor_beta"] = Ladder(detect=thermistor_levels,
                                    note="six-level detectors")

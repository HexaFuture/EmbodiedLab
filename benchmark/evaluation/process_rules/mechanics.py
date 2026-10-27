"""Process-score rules of the mechanics kit (legacy E1, E2, M1, M2; provisional L1-L4, see LADDERS at the end), over
the process record of one run.

Sources: private/physics.jsonl (the backend's frame log: 30 Hz rows with a `frame` for the Hooke, torsion and beam
benches, 30 Hz rows with a `physics_tick` for the pendulum bench (contact flags of the whole frame,
`peak_deg` the frame's largest |angle|), one row per physics tick with a `tick` or `physics_tick` for the others;
`event` rows carry a `tick`), the
evidence records (as a submission at stop would seal them), bench_logs.json (beam bench rows, private rows and
transitions) and the Bench call log (tool, tick before, tick after, ok, error). Apparatus preparation runs before the
episode clock is reset to 0; rows before the last reset and after the end tick are ignored.

M1 is the first valid reading taken in the E2 state: its tick is never earlier than E2's. Readings taken before E2
(an empty tray, an empty turntable) count toward M2 only (process-score design 2.3).

Frame log not exported: the record export (benchmark/core/instance_episodes.py, finish_process_record) stores a
frame log gzipped and drops it when the stored physics.jsonl.gz is over 50 MB; the cap applies to the compressed file,
which is 10-15x smaller than the raw log (raw per-tick sizes: pendulum ~1.6 kB/tick earlier, now ~0.1
kB/tick; projectile ~0.8, incline ~0.4, air track ~0.33, coupled ~0.33 earlier and ~0.2 since), so a full
budget stays well under it (incline ~6 MB, projectile ~30 MB stored). When private_skipped.json lists physics.jsonl,
the per-tick rules take E1 and E2 from the first record that could only exist after them (its start), flagged
`frame_log_skipped`; E1 then shares E2's tick.
"""
from __future__ import annotations

import ast
import itertools
import json
import math
from collections import deque

from benchmark.evaluation.gates import (coupled_clean_mask, coupled_modes_determined, coupled_segments, encoder_free_run,
                                       longest_ok_run, oscillation_period, photogate_period, photogate_rows)
from benchmark.evaluation.process import hit, miss

G = 9.81
CARRY_FRAMES = 15           # 0.5 s of holding or touching
CARRY_JOINT_RAD = 0.05      # the arm moved what it held


# --- shared helpers -------------------------------------------------------------------------------------------------
def _log(run):
    """(states, events): [(tick, row)] of the episode in file order, after the last clock reset, up to the end tick."""
    ft = run.physics_hz // 30

    def clock(row):
        if "event" in row:
            return row.get("tick")
        if "frame" in row:
            return row["frame"] * ft
        return row.get("tick", row.get("physics_tick"))

    rows = run.frames
    start, last = 0, None
    for i, row in enumerate(rows):
        t = None if "event" in row else clock(row)
        if t is None:
            continue
        if last is not None and t < last:
            start = i
        last = t
    states, events = [], []
    for row in rows[start:]:
        t = clock(row)
        if t is None or t > run.end_tick:
            continue
        (events if "event" in row else states).append((t, row))
    return states, events


def _by_number(evidence, kind=None):
    """Evidence records in acquisition order (the number after the last ':' of the id)."""
    items = [(eid, r) for eid, r in evidence.items() if kind is None or r.get("kind") == kind]
    return sorted(items, key=lambda item: int(str(item[0]).rsplit(":", 1)[-1]))


def _lifted(states, rise_mm=10):
    """First (tick, key) at which the held part stands >= rise_mm above its height in the first episode frame
    (frame fields `held` and `xyz_mm`, integer mm); (None, None) when never. Records without these fields: absent."""
    z0 = {k: p[2] for k, p in states[0][1]["xyz_mm"].items()}
    for t, r in states:
        key = r.get("held")
        if key is not None and key in z0 and r["xyz_mm"][key][2] - z0[key] >= rise_mm:
            return t, key
    return None, None


def _frames_skipped(run):
    """physics.jsonl was too large for the record export (listed in private_skipped.json)."""
    path = run.root / "private_skipped.json"
    try:
        return path.is_file() and "physics.jsonl" in json.loads(path.read_text())
    except (OSError, ValueError):
        return False


def _from_record(hits, tick, basis):
    """E1 and E2 inferred from a record when the frame log was not exported (see the module docstring)."""
    if tick is None:
        return
    flags = ["frame_log_skipped"]
    hits["E2"] = hit(tick, f"{basis} (frame log not exported)", flags)
    hits["E1"] = hit(tick, f"implied by E2: {basis} (frame log not exported)", flags)


def _has(states, field):
    return bool(states) and field in states[0][1]


def _carried(states, holding):
    """First tick of a run of >= CARRY_FRAMES frames in which `holding(row)` is true and the arm moved
    >= CARRY_JOINT_RAD from where the run began (the robot held something and moved it)."""
    start = None
    for i, (t, row) in enumerate(states):
        if holding(row):
            if start is None:
                start, q0 = i, row["q_rad"]
            moved = max(abs(a - b) for a, b in zip(row["q_rad"], q0))
            if i - start + 1 >= CARRY_FRAMES and moved >= CARRY_JOINT_RAD:
                return states[start][0], states[start][1]
        else:
            start = None
    return None, None


def _finish(hits, first_tick):
    """A measurable state already true at the start counts as reached (design 2.3 rule 4), flagged."""
    e2 = hits["E2"]
    if e2.reached and e2.tick is not None and e2.tick <= first_tick:
        e2.flags.append("state_at_start")
        if not hits["E1"].reached:
            hits["E1"] = hit(e2.tick, "measurable state already true at the start", ["state_at_start"])
    return {name: hits[name] for name in ("E1", "E2", "M1", "M2")}


def _m1_in_state(e2, candidates, what):
    """First (tick, basis) of candidates (in order) not earlier than E2."""
    if not e2.reached:
        return miss(f"no {what}: E2 not reached")
    for tick, basis in candidates:
        if tick is not None and tick >= e2.tick:
            return hit(tick, basis)
    return miss(f"no {what} at or after E2 (tick {e2.tick})")


# --- hooke_k -----------------------------------------------------------------------------------------------------
def _hooke_supported(rows):
    """backend.support_state over 30 frames: tray at rest, all released, weights with the support force of 50 g.
    The weight-to-deck gap of the live check needs weight poses, which the frame log does not carry."""
    z = [r["z"] for r in rows]
    stable = max(z) - min(z) < .0008 and max(abs(b - a) for a, b in zip(z, z[1:])) * 30 < .002
    released = all(r["released"] for r in rows)
    supported = []
    for key in rows[-1]["forces"]:
        forces = [r["forces"][key] for r in rows]
        if abs(sum(forces) / len(forces) - .05 * G) < .06 and min(forces) > .2:
            supported.append(key)
    return supported, stable, released


def hooke_k(run):
    """E1  a weight held and lifted >= 10 mm above its start (frame fields `held`, `xyz_mm`). Records without them:
        >= 0.5 s not released (grasp constraint or robot contact on a weight) while the arm moves >= 0.05 rad.
    E2  >= 1 weight supported on the tray, tray stable, all released, over the 30-frame support window.
    M1  first ruler image with a supported weight, stable and released, at or after E2.
    M2  stable, released ruler images cover every load 0..n (n = number of weights)."""
    states, _ = _log(run)
    rows = [r for _, r in states]
    hits = {}
    if _has(states, "xyz_mm"):
        t, key = _lifted(states)
        hits["E1"] = hit(t, f"{key} held and lifted 10 mm") if t is not None else miss("no weight held and lifted")
    else:
        t, _ = _carried(states, lambda r: not r["released"])
        hits["E1"] = hit(t, "weight held and carried") if t is not None else miss("no weight held and carried")
    hits["E2"] = miss("no weight supported, stable and released on the tray")
    for i in range(29, len(rows)):
        supported, stable, released = _hooke_supported(rows[i - 29:i + 1])
        if supported and stable and released:
            hits["E2"] = hit(states[i][0], f"supported {supported}, stable, released")
            break
    ft = run.physics_hz // 30
    images = [r for _, r in _by_number(run.evidence, "ruler_image") if r["stable"] and r["released"]]
    hits["M1"] = _m1_in_state(hits["E2"], [(r["frame"] * ft, f"ruler image, {len(r['supported'])} weight(s)")
                                           for r in images if r["supported"]], "loaded ruler image")
    n = len(run.terminal_state.get("weight_ids") or (rows[-1]["forces"] if rows else ()))
    seen = set()
    hits["M2"] = miss(f"stable released readings do not cover loads 0..{n}")
    for r in images:
        seen.add(len(r["supported"]))
        if n and seen >= set(range(n + 1)):
            hits["M2"] = hit(r["frame"] * ft, f"stable released readings cover loads 0..{n}")
            break
    if not hits["M2"].reached:
        hits["M2"].basis += f" (have {sorted(seen)})"
    return _finish(hits, ft)


# --- hooke_dynamic_k ---------------------------------------------------------------------------------------------
DYN_WINDOW_FRAMES = 75      # 2.5 s: two periods of the slowest tray (1.14 s)


def _on_tray(row):
    return frozenset(k for k, f in row["forces"].items() if f > .1 and not row["robot_contacts"][k])


def _encoder_oscillation(record):
    t, z = longest_ok_run(record["rows"], record.get("sample_status"))
    return oscillation_period(t, z), (t[-1] - t[0] if t else 0.0)


def hooke_dynamic_k(run):
    """E1  the tray first excited by the robot: tray height off its starting rest by > 1 mm, or a weight on the tray.
    E2  REC pressed on the recorder panel (frame-log `key` event with effect `start`) while the tray vibrates freely
        over the 2.5 s before: all weights released, the same load throughout, >= 2 upward crossings of the mean height
        and peak-to-peak >= 1 mm, no robot contact on the tray and the damper lever's magnet off the vane
        (`tray_contact`, `damper`). Records of the former encoder tools (frame field `brake`): the first such window.
    M1  first complete encoder record at or after E2 with an unchanged load and an oscillation of >= 1 mm.
    M2  records with an unchanged load, >= 5 periods of consecutive ok samples and >= 1 mm, for >= 2 loads."""
    states, events = _log(run)
    rows = [r for _, r in states]
    legacy = bool(rows) and "brake" in rows[0]
    hits = {"E1": miss("tray never excited"),
            "E2": miss("no free tray vibration" if legacy else "REC never pressed on a freely vibrating tray")}
    if rows:
        z0 = rows[0]["z"]
        for t, r in states:
            if abs(r["z"] - z0) > .001 or _on_tray(r):
                hits["E1"] = hit(t, "tray displaced or loaded")
                break

    def free(window):
        if not all(r["released"] for r in window) or len({_on_tray(r) for r in window}) != 1:
            return None
        if any(r.get("tray_contact") or r.get("brake") or r.get("damper") for r in window):
            return None
        z = [r["z"] for r in window]
        mean = sum(z) / len(z)
        ups = sum(1 for a, b in zip(z, z[1:]) if a < mean <= b)
        if ups >= 2 and max(z) - min(z) >= .001:
            return f"{len(_on_tray(window[-1]))} weight(s), {(max(z) - min(z)) * 1000:.1f} mm peak-to-peak"
        return None

    if legacy:                                  # arm_encoder / set_damping_brake records (earlier)
        for i in range(DYN_WINDOW_FRAMES - 1, len(rows)):
            basis = free(rows[i - DYN_WINDOW_FRAMES + 1:i + 1])
            if basis:
                hits["E2"] = hit(states[i][0], f"free vibration, {basis}")
                break
    else:
        ticks = [t for t, _ in states]
        for t, e in events:
            if e.get("event") != "key" or e.get("effect") != "start":
                continue
            i = sum(1 for s in ticks if s <= t) - 1
            basis = free(rows[i - DYN_WINDOW_FRAMES + 1:i + 1]) if i + 1 >= DYN_WINDOW_FRAMES else None
            if basis:
                hits["E2"] = hit(t, f"REC pressed ({e.get('evidence_id')}) on free vibration, {basis}")
                break
    good, readings = [], []
    for eid, r in _by_number(run.evidence, "tray_encoder"):
        if r.get("status") != "complete" or r["load_ids_at_arm"] != r["load_ids_at_end"]:
            continue
        measured, span = _encoder_oscillation(r)
        if measured is None or measured[2] < 1.0:
            continue
        readings.append((r["end_tick"], f"{eid}: oscillation {measured[2]:.1f} mm"))
        if span >= 5 * measured[0]:
            good.append((r["end_tick"], len(r["load_ids_at_arm"])))
    hits["M1"] = _m1_in_state(hits["E2"], readings, "encoder record of an oscillation")
    loads, hits["M2"] = set(), miss("fewer than two loads with 5 clean periods")
    for tick, load in good:
        loads.add(load)
        if len(loads) >= 2:
            hits["M2"] = hit(tick, f"loads {sorted(loads)} weight(s), each >= 5 clean periods")
            break
    return _finish(hits, run.physics_hz // 30)


# --- pendulum_g --------------------------------------------------------------------------------------------------
def pendulum_g(run):
    """E1  first robot contact (any part of the robot) with a pendulum after which that pendulum swings: in the 0.5 s
        after the contact ends the pendulum is untouched and its peak |angle| exceeds 0.3 deg. Tick: the last tick
        of that contact.
    E2  START pressed (bench log `button_presses`, effect `started`) on the gate of a pendulum that swings freely:
        over the 2 s before the press no contact on it and peak |angle| in [0.3, 5) deg. Records from before the
        gate buttons have no presses: the start ticks of their gates stand in for them.
    M1  first gate evidence at or after E2 that is complete, has no invalid reason and is robot-excited
        (contact_excitation). Every finished acquisition is evidence from the tick it finishes.
    M2  such a gate (8-10 cycles; the buttons time 10) for all three pendulums. Invalid gates are flagged; records
        from before the RESET button (no `gate_history` in the bench log) flag M2 unreachable when too few
        pendulums are left for a first gate.
    Contact flags per frame row: `robot_contact` (older records `probe_contact`) and `any_contact`; the angle is the
    row's `peak_deg` (the frame's largest |angle|; older per-tick rows: |angle_deg|)."""
    states, _ = _log(run)
    hz = run.physics_hz
    short, window, spacing = hz // 2, 2 * hz, max(hz // 30, 1)
    hits = {"E1": miss("no robot contact left a pendulum swinging"),
            "E2": miss("START never pressed on a freely swinging pendulum")}
    series = {}                 # pendulum -> [(tick, peak |angle| deg, any contact, robot contact)]
    for t, r in states:
        for key, p in r["pendula"].items():
            series.setdefault(key, []).append((t, p.get("peak_deg", abs(p["angle_deg"])), bool(p["any_contact"]),
                                               bool(p.get("robot_contact", p.get("probe_contact")))))
    first = None
    for key, rows in series.items():
        for i, (t, _, _, robot) in enumerate(rows):
            if not robot or (i + 1 < len(rows) and rows[i + 1][3]):        # the last tick of a robot contact
                continue
            if first is not None and t >= first[0]:
                break
            after = [row for row in rows[i + 1:] if row[0] - t <= short]
            if (after and after[-1][0] >= t + short - spacing and not any(row[2] for row in after)
                    and max(row[1] for row in after) > .3):
                first = (t, f"{key} left swinging at {max(row[1] for row in after):.2f} deg after robot contact")
                break
    if first:
        hits["E1"] = hit(*first)
    gates = [(eid, r) for eid, r in _by_number(run.evidence, "optical_gate")]
    history = run.logs.get("gate_history")              # every started acquisition, RESET ones included
    states_of_gates = run.logs.get("gate_states")       # PendulumEpisode.process_extra: current acquisition per gate
    if "button_presses" in run.logs:
        presses = [(p["tick"], p["pendulum_id"], "START pressed") for p in run.logs["button_presses"]
                   if p.get("effect") == "started"]
    else:                                                # before the buttons: gate start ticks
        started = (history if history is not None else
                   [dict(g, pendulum_id=k) for k, g in (states_of_gates or {}).items()])
        presses = [(g["start_tick"], g["pendulum_id"], "gate started") for g in started if g.get("start_tick") is not None]
        presses += [(r["start_tick"], r["pendulum_id"], "gate started") for _, r in gates
                    if r.get("start_tick") is not None]
    for tick, key, what in sorted(set(presses)):
        rows = [row for row in series.get(key, ()) if tick - window <= row[0] <= tick]
        if (rows and rows[0][0] <= tick - window + spacing and not any(row[2] for row in rows)
                and .3 <= max(row[1] for row in rows) < 5):
            hits["E2"] = hit(tick, f"{what} on {key}, swinging freely at {max(row[1] for row in rows):.2f} deg")
            break
    if not states and _frames_skipped(run):     # timing start of the first gate on a robot-excited pendulum
        starts = [(r["end_tick"] - round(r["public"]["elapsed_s"] * hz), f"{eid}: timing started on a robot-excited "
                   f"{r['pendulum_label']}") for eid, r in gates if r["contact_excitation"]
                  and isinstance(r["public"].get("elapsed_s"), (int, float)) and r.get("end_tick") is not None]
        unexported = (history if history is not None else
                      [dict(g, pendulum_id=k) for k, g in (states_of_gates or {}).items()])
        starts += [(g["start_tick"], f"gate started on robot-excited {g['pendulum_id']}") for g in unexported
                   if g.get("contact_excitation") and g.get("timing_started") and not g.get("evidence_id")]
        if starts:
            _from_record(hits, *min(starts))
    valid = [(eid, r) for eid, r in gates if r["public"]["status"] == "complete" and not r["invalid_reasons"]
             and r["contact_excitation"]]
    hits["M1"] = _m1_in_state(hits["E2"], [(r["end_tick"], f"{eid}: complete gate on {r['pendulum_label']}")
                                           for eid, r in valid], "valid gate")
    total = len(run.device.get("effective_lengths_m") or {}) or 3
    done, hits["M2"] = set(), miss(f"fewer than {total} pendulums with a valid robot-excited gate")
    for eid, r in valid:
        if 8 <= r["public"]["cycles"] <= 10:
            done.add(r["pendulum_label"])
            if len(done) >= total:
                hits["M2"] = hit(r["end_tick"], f"valid gates on {sorted(done)}")
                break
    spent = {r["pendulum_id"] for _, r in gates if r["invalid_reasons"]}
    if history is not None:
        spent |= {g["pendulum_id"] for g in history if g["invalid_reasons"] or g["status"] == "invalid"}
    elif states_of_gates is not None:
        spent |= {k for k, g in states_of_gates.items() if g["invalid_reasons"] or g["status"] == "invalid"}
    else:                                                # older records: the runtime's gate dicts, sets as repr
        for key, gate in (run.logs.get("gates") or {}).items():
            try:
                reasons = ast.literal_eval(gate.get("invalid_reasons", "set()")) if isinstance(gate, dict) else set()
            except (ValueError, SyntaxError):
                reasons = set()
            if reasons:
                spent.add(key)
    if spent:
        hits["M2"].flags.append(f"invalid_gate:{sorted(spent)}")
        if not hits["M2"].reached and total - len(spent) < 3 and history is None:
            hits["M2"].flags.append("M2_unreachable")
    return _finish(hits, 1)


# --- coupled_modes -----------------------------------------------------------------------------------------------
def _releases(states):
    """Ticks at which both brakes opened together (the release bar firing an armed recorder)."""
    out, previous = [], None
    for t, r in states:
        if previous is not None and all(previous) and not any(r["locked"]):
            out.append(t)
        previous = r["locked"]
    return out


def coupled_modes(run):
    """E1  the robot moved a catch or a hooked slider by more than 1 mm: the bench's `robot_moved` event (records
        earlier, with preparation brakes: a slider unlocked, displaced > 1 mm from its start, robot contact
        at that tick).
    E2  REC keyed on the recorder keypad (a `key` event with effect `ready`) while both sliders were hooked and at
        least one stood >= 1 mm from its encoder zero (records earlier: a successful arm_recorder call with
        both brakes locked and a slider displaced >= 1 mm).
    M1  first complete recording; its tick is the recording's end tick (`recordings` in the bench log; older records:
        the k-th recording starts at the k-th release and ends one recording length later).
    M2  two complete recordings, each >= 40 s with both channels >= 1 mm peak-to-peak and initial displacements
        linearly independent (|det| / norms >= 0.3)."""
    states, events = _log(run)
    hz = run.physics_hz
    hits = {"E1": miss("no catch or slider moved by the robot"),
            "E2": miss("REC never keyed with both sliders hooked and one displaced")}
    legacy = bool(states) and "locked" in states[0][1]       # preparation brakes and arm_recorder (earlier)
    if legacy:
        x0 = states[0][1]["x_m"]
        by_tick = {}
        for t, r in states:
            by_tick[t] = r
            if not hits["E1"].reached and r["external_contact"] and any(
                    not lock and abs(x - s) > .001 for lock, x, s in zip(r["locked"], r["x_m"], x0)):
                hits["E1"] = hit(t, "slider gripped, unlocked and displaced")
        for name, before, _, ok, _ in run.calls:
            r = by_tick.get(before)
            if name == "arm_recorder" and ok and r and all(r["locked"]) and max(
                    abs(x - s) for x, s in zip(r["x_m"], x0)) >= .001:
                hits["E2"] = hit(before, "armed with both brakes locked on a displaced slider")
                break
    for t, r in events:
        if r["event"] == "robot_moved" and not hits["E1"].reached:
            hits["E1"] = hit(t, f"a catch or a hooked slider moved by more than 1 mm (sliders {r.get('x_mm')} mm, "
                                f"catches {r.get('c_mm')} mm)")
        if (r["event"] == "key" and r.get("effect") == "ready" and not hits["E2"].reached
                and all(r.get("hooked") or [False]) and max(abs(x) for x in r.get("x_mm") or [0.0]) >= 1):
            hits["E2"] = hit(t, f"REC keyed for {r.get('ready_s')} s with both sliders hooked at {r['x_mm']} mm")
    releases = _releases(states) if legacy else []
    records = []
    for k, (eid, r) in enumerate(_by_number(run.evidence, "coupled_recorder")):
        if eid in (run.logs.get("recordings") or {}):      # CoupledEpisode.process_extra: [start, end] ticks
            end = run.logs["recordings"][eid][1]
        else:                                               # older records: the k-th release starts recording k
            end = releases[k] + round(r["rows"][-1]["t_s"] * hz) if k < len(releases) and r["rows"] else None
        records.append((end, eid, r))
    spans = run.logs.get("recordings") or {}
    if not states and _frames_skipped(run):     # the arm call before the first recording that starts displaced
        armed = [before for name, before, _, ok, _ in run.calls if name == "arm_recorder" and ok]
        starts = sorted(spans[eid][0] for _, eid, r in records if eid in spans and r["rows"]
                        and max(abs(r["rows"][0]["x1_mm"] or 0), abs(r["rows"][0]["x2_mm"] or 0)) >= 1)
        calls = [t for t in armed if starts and t <= starts[0]]
        if calls:
            _from_record(hits, calls[-1], "armed before a recording that starts displaced")
        elif starts and not armed:      # REC keyed on the keypad: the recording's start is the earliest evidence of it
            _from_record(hits, starts[0], "a recording that starts displaced (REC keyed on the keypad before it)")
    complete = [(end, eid, r) for end, eid, r in records if r["status"] == "complete" and not r["invalid_reasons"]]
    hits["M1"] = _m1_in_state(hits["E2"], [(end, f"{eid}: complete recording") for end, eid, _ in complete],
                              "complete recording")

    def usable(r):
        rows = r["rows"]
        if not rows or rows[-1]["t_s"] < 40 or rows[0]["x1_mm"] is None or rows[0]["x2_mm"] is None:
            return False
        for key in ("x1_mm", "x2_mm"):
            values = [row[key] for row in rows if row[key] is not None]
            if not values or max(values) - min(values) < 1:
                return False
        return True

    hits["M2"] = miss("no pair of complete >= 40 s recordings with independent initial displacements")
    pool = [(run.end_tick if end is None else end, eid, r) for end, eid, r in complete if usable(r)]
    pairs = []
    for (e1, a, ra), (e2, b, rb) in itertools.combinations(pool, 2):
        (a1, a2), (b1, b2) = [(x["rows"][0]["x1_mm"], x["rows"][0]["x2_mm"]) for x in (ra, rb)]
        norm = math.hypot(a1, a2) * math.hypot(b1, b2)
        if norm >= 1 and abs(a1 * b2 - a2 * b1) / norm >= .3:
            pairs.append((max(e1, e2), f"{a} and {b}: independent excitation, normalized |det| "
                                       f"{abs(a1 * b2 - a2 * b1) / norm:.2f}"))
    if pairs:
        hits["M2"] = hit(*min(pairs))
    return _finish(hits, 1)


# --- torsion_inertia, torsion_centroid -----------------------------------------------------------------------
TORSION_LOADS = {(): "empty", ("standard_disc",): "standard_disc", ("unknown_object",): "unknown_object"}
TORSION_WINDOW_FRAMES = 90  # 3 s: over half of the slowest loaded period (about 4 s)


def _resting(row, masses):
    """Objects resting on the turntable in this frame, as backend.loads judges each of its 10 frames."""
    return frozenset(k for k, f in row["turntable_force"].items()
                     if f > .5 * masses[k] * G and not row["robot_contacts"][k] and k not in (row["held"], row["releasing"]))


def _torsion_e1(states, masses, half_angle):
    """Earliest of: the turntable turned beyond the photogate flag (|angle| > flag half-angle), or an object held
    with no support from the turntable or the table for 3 frames (lifted)."""
    start, key, count = None, None, 0
    for t, r in states:
        if abs(r["angle_rad"]) > half_angle:
            return hit(t, "turntable turned past the photogate flag")
        held = r["held"]
        if held is not None and r["turntable_force"][held] + r["table_force"][held] < .1 * masses[held] * G:
            if held != key:
                start, key, count = t, held, 0
            count += 1
            if count >= 3:
                return hit(start, f"{key} lifted")
        else:
            start, key, count = None, None, 0
    return miss("turntable never turned and no object lifted")


def _free_swing(window, masses, half_angle):
    """A free torsional oscillation over the window: brake (damper lever) off, nothing held or touching an object, no
    robot contact on the turntable (`turntable_contact`, when logged), the same load throughout, |angle| beyond the
    flag half-angle and a sign change (the turntable crosses its rest angle)."""
    if any(r["brake"] or not r["released"] or r.get("turntable_contact") for r in window) or \
            len({_resting(r, masses) for r in window}) != 1:
        return False
    angles = [r["angle_rad"] for r in window]
    return max(abs(a) for a in angles) > half_angle and min(angles) < 0 < max(angles)


def _torsion_e2(states, masses, half_angle):
    """Records earlier (arm_photogate): a free torsional oscillation over 3 s (_free_swing)."""
    rows = [r for _, r in states]
    for i in range(TORSION_WINDOW_FRAMES - 1, len(rows)):
        window = rows[i - TORSION_WINDOW_FRAMES + 1:i + 1]
        if _free_swing(window, masses, half_angle):
            return hit(states[i][0], f"free oscillation, load {sorted(_resting(window[-1], masses)) or 'empty'}")
    return miss("no free torsional oscillation")


def _torsion_rec_e2(states, events, masses, half_angle):
    """REC keyed on the recorder panel (a `key` event that started a record) while the turntable swung freely:
    _free_swing over the 3 s of frames up to the press."""
    import bisect
    ticks = [t for t, _ in states]
    for t, e in events:
        if e["event"] != "key" or e.get("effect") != "start":
            continue
        end = bisect.bisect_right(ticks, t)
        if end < TORSION_WINDOW_FRAMES:
            continue
        window = [r for _, r in states[end - TORSION_WINDOW_FRAMES:end]]
        if _free_swing(window, masses, half_angle):
            return hit(t, f"REC keyed during a free oscillation, load {sorted(_resting(window[-1], masses)) or 'empty'}")
    return miss("REC never keyed while the turntable swung freely")


def _torsion_record_load(at_arm, at_end):
    if at_arm is None or at_end is None or set(at_arm) != set(at_end):
        return None
    return TORSION_LOADS.get(tuple(sorted(at_arm)))


def torsion_inertia(run):
    """E1  turntable turned past the photogate flag, or an object lifted (held, no turntable or table support).
    E2  REC keyed on the recorder panel while the turntable swung freely: released, load unchanged, crossing its rest
        angle (_torsion_rec_e2; records earlier, without the damper lever: a free oscillation, _torsion_e2).
    M1  first photogate record at or after E2 with an unchanged load and >= 3 consecutive ok transits.
    M2  the empty turntable, the standard disc and the unknown object each with a record of unchanged load and
        >= 5 complete periods of consecutive ok transits."""
    states, events = _log(run)
    masses = run.device.get("object_masses_kg") or {}
    half = run.device.get("flag_half_angle_rad") or .0055
    legacy = bool(states) and "damper" not in states[0][1]
    hits = {"E1": _torsion_e1(states, masses, half),
            "E2": _torsion_e2(states, masses, half) if legacy else _torsion_rec_e2(states, events, masses, half)}
    readings, done = [], {}
    hits["M2"] = miss("loads empty, standard_disc and unknown_object not all recorded for 5 clean periods")
    for eid, r in _by_number(run.evidence, "torsion_photogate"):
        load = _torsion_record_load(r.get("loads_at_arm"), r.get("loads_at_end"))
        measured = photogate_period(photogate_rows(r))
        if r.get("status") != "complete" or load is None or measured is None:
            continue
        readings.append((r["end_tick"], f"{eid}: {load}, {measured[1]} periods"))
        if measured[1] >= 5:
            done.setdefault(load, r["end_tick"])
            if len(done) == 3 and not hits["M2"].reached:
                hits["M2"] = hit(r["end_tick"], "empty, standard_disc and unknown_object recorded")
    hits["M1"] = _m1_in_state(hits["E2"], readings, "photogate record")
    if not hits["M2"].reached:
        hits["M2"].basis += f" (have {sorted(done)})"
    return _finish(hits, run.physics_hz // 30)


def torsion_centroid(run):
    """E1  an object lifted (held, no turntable or table support), or the turntable turned past the photogate flag.
    E2  exactly one object seated on the locating pin, not held or touched (frame field `mount` [load, hole] names an
        object; CentroidBackend.mount).
    M1  first valid record of a mounted object at or after E2 (complete, load unchanged, seated on the pin
        throughout, >= 5 clean periods: torsion.centroid_record_valid).
    M2  valid records of the empty turntable, the standard disc and the sample at >= 2 different holes.
    Records earlier (mounting gauge, no pin) have no `mount`: E2 and the M levels are not reached."""
    from benchmark.evaluation.torsion import centroid_mount, centroid_record_valid
    states, _ = _log(run)
    masses = run.device.get("object_masses_kg") or {}
    half = run.device.get("flag_half_angle_rad") or .0055
    hits = {"E1": _torsion_e1(states, masses, half), "E2": miss("never exactly one object seated on the pin")}
    for t, r in states:
        mount = r.get("mount") or [None, None]
        if mount[0] in ("standard_disc", "unknown_object"):
            hits["E2"] = hit(t, f"{mount[0]} seated on the pin (hole {mount[1]})")
            break
    readings, have, holes = [], set(), set()
    hits["M2"] = miss("need valid records of the empty turntable, the standard disc and two sample holes")
    for eid, r in _by_number(run.evidence, "centroid_photogate"):
        mount = centroid_mount(r)
        if mount is None:
            continue
        _, issues, _ = centroid_record_valid(r, mount[0])
        if issues:
            continue
        if mount[0] != "empty":
            readings.append((r["end_tick"], f"{eid}: {mount[0]} on the pin (hole {mount[1]})"))
        have.add(mount[0])
        if mount[0] == "unknown_object":
            holes.add(mount[1])
        if not hits["M2"].reached and {"empty", "standard_disc"} <= have and len(holes) >= 2:
            hits["M2"] = hit(r["end_tick"], f"empty, standard and the sample at holes {sorted(holes)}")
    hits["M1"] = _m1_in_state(hits["E2"], readings, "valid record of a mounted object")
    if not hits["M2"].reached:
        hits["M2"].basis += f" (have {sorted(have)}, holes {sorted(holes)})"
    return _finish(hits, run.physics_hz // 30)


# --- airtrack_collision ------------------------------------------------------------------------------------------
def _airtrack_valid(r, mass_a):
    """The reduction of a valid run record, as _airtrack_recorder checks it, or None."""
    from benchmark.core.airtrack.measurement import reduce_record
    try:
        return reduce_record(r, mass_a)
    except (ValueError, KeyError, TypeError):
        return None


def airtrack_collision(run):
    """E1  earliest of: a glider moved more than 1 mm by the robot (the bench's `robot_moved` event: one continuous
        robot contact) or the launcher cocked (`cocked`: the pawl dropped into a notch).
    E2  REC keyed (a `key` event with effect `ready`) while both gliders were held in their start catches and the
        launcher was cocked.
    M1  first valid collision run (measurement.reduce_record); its tick is the end of the last of its three passes.
    M2  valid collisions from three launcher notches whose joint mass solves (joint_mass: distinct incoming speeds);
        its tick is when the third of them became valid.
    Frame log not exported: E1 and E2 from the start of the first run (a release with the recorder READY)."""
    from benchmark.core.airtrack.measurement import joint_mass
    _, events = _log(run)
    hits = {"E1": miss("no glider moved by the robot and the launcher never cocked"),
            "E2": miss("REC never keyed with both gliders caught and the launcher cocked")}
    for t, r in events:
        kind = r["event"]
        if kind in ("robot_moved", "cocked") and not hits["E1"].reached:
            hits["E1"] = hit(t, "glider moved by the robot" if kind == "robot_moved"
                             else f"launcher cocked into notch {r.get('notch')}")
        if (kind == "key" and r.get("effect") == "ready" and not hits["E2"].reached
                and all(r.get("caught") or [False]) and r.get("notch") is not None):
            hits["E2"] = hit(t, f"REC keyed for {r.get('ready_s')} s with both gliders caught, notch {r['notch']}")
    if not events and _frames_skipped(run):
        starts = [ticks[0] for ticks in (run.logs.get("recordings") or {}).values()]
        if starts:
            _from_record(hits, min(starts), "a run started at a release with the recorder READY")
    mass_a = run.device.get("mass_A_kg", .2)
    valid = []
    for eid, r in _by_number(run.evidence, "airtrack_recorder"):
        fit = _airtrack_valid(r, mass_a)
        if fit is not None:
            valid.append((fit["complete_tick"], eid, fit))
    valid.sort(key=lambda v: v[0])
    hits["M1"] = _m1_in_state(hits["E2"], [(t, f"{eid}: valid collision, notch {f['notch']}") for t, eid, f in valid],
                              "valid collision")
    hits["M2"] = miss("no three valid collisions from three notches with a solvable joint mass")
    for k in range(3, len(valid) + 1):
        for trio in itertools.combinations(valid[:k], 3):
            if len({x[2]["notch"] for x in trio if x[2]["notch"] is not None}) < 3:
                continue
            try:
                joint_mass([x[2] for x in trio], mass_a)
            except ValueError:
                continue
            hits["M2"] = hit(valid[k - 1][0], f"notches {sorted(x[2]['notch'] for x in trio)}: joint mass solvable")
            break
        if hits["M2"].reached:
            break
    return _finish(hits, 1)


# --- incline_friction --------------------------------------------------------------------------------------------
def _travel(run_):
    positions = [s["position_m"] for s in run_]
    return max(positions) - min(positions)


def incline_friction(run):
    """Two sliders, A and B; frame rows carry `sliders` {name: s_m, normal_m, lateral_m, robot_contact,
    ...}, `held` (the slider's name) and `sensed` (the slider the position sensor measures).
    E1  a slider held and moved > 1 cm from its starting pose.
    E2  a slider measured by the position sensor (on its track) and free (not held, no robot contact) for 0.25 s.
    M1  first record at or after E2 with a free slide (a free run of one slider travelling >= 2 cm); its tick is its
        last sample.
    M2  for each slider one static record as the gate accepts it (a free run of that slider holding the rest, a
        >= 1 deg rise and the onset of a >= 2 cm slide) and one kinetic record (>= 10 cm free slide at a fixed angle
        >= 5 deg), four different records, each showing one slider only (incline.record_sliders)."""
    from benchmark.evaluation.incline import SLIDERS, free_runs, record_sliders
    states, _ = _log(run)
    hz = run.physics_hz
    hits = {"E1": miss("no slider held and moved"), "E2": miss("no slider free on the ramp sensor")}
    keys = ("s_m", "normal_m", "lateral_m")
    if states:
        p0 = {k: [states[0][1]["sliders"][k][x] for x in keys] for k in SLIDERS}
        for t, r in states:
            held = r["held"]
            if held in p0 and math.dist([r["sliders"][held][x] for x in keys], p0[held]) > .01:
                hits["E1"] = hit(t, f"slider {held} held and moved")
                break
    start = None
    for t, r in states:
        sensed = r.get("sensed")
        on = sensed is not None and r["held"] != sensed and not r["sliders"][sensed]["robot_contact"]
        start = (start if start is not None else t) if on else None
        if on and t - start >= hz // 4:
            hits["E2"] = hit(start, f"slider {sensed} free on the ramp sensor")
            break
    records = [(eid, r, free_runs(r)) for eid, r in _by_number(run.evidence, "incline_record")
               if len(r["private_rows"]) == len(r["public"]["samples"])]
    end = {eid: r["public"]["samples"][-1]["physics_tick"] for eid, r, _ in records}
    if not states and _frames_skipped(run):     # first 0.25 s free on the position sensor in a record
        free = [(x[0]["physics_tick"], f"{eid}: slider {k} free on the ramp sensor") for eid, _, runs in records
                for k, x in runs if x[-1]["physics_tick"] - x[0]["physics_tick"] >= hz // 4]
        if free:
            _from_record(hits, *min(free))
    hits["M1"] = _m1_in_state(hits["E2"], [(end[eid], f"{eid}: free slide") for eid, _, runs in records
                                           if any(_travel(x) >= .02 for _, x in runs)], "record with a free slide")
    shows = {(role, eid): record_sliders(r, role) for eid, r, _ in records for role in ("static", "kinetic")}
    lists = {(role, k): [eid for eid, _, _ in records if shows[(role, eid)] == [k]]
             for role in ("static", "kinetic") for k in SLIDERS}
    best = None
    for sa in lists[("static", "A")]:
        for sb in lists[("static", "B")]:
            for ka in lists[("kinetic", "A")]:
                for kb in lists[("kinetic", "B")]:
                    if len({sa, sb, ka, kb}) == 4:
                        tick = max(end[e] for e in (sa, sb, ka, kb))
                        if best is None or tick < best[0]:
                            best = (tick, sa, sb, ka, kb)
    hits["M2"] = (hit(best[0], f"static A {best[1]}, static B {best[2]}, kinetic A {best[3]}, kinetic B {best[4]}")
                  if best else miss("need a static and a kinetic record of each slider: "
                                    + ", ".join(f"{role} {k} {v}" for (role, k), v in lists.items())))
    return _finish(hits, 1)


# --- projectile_landing -----------------------------------------------------------------------------------------
def _seated(row):
    """backend.ball_in_cradle 'seated': on the rails within 3 mm of the latched plunger face, at rest, not held."""
    from benchmark.core.projectile.geometry import AXIS, ball_rest
    if row["held"] == "ball":
        return False
    offset = [b - r for b, r in zip(row["ball"], ball_rest(row["plunger_travel"]))]
    along = sum(o * a for o, a in zip(offset, AXIS))
    across = math.sqrt(max(0.0, sum(o * o for o in offset) - along * along))
    speed = math.sqrt(sum(v * v for v in row["ball_velocity"]))
    return across < .006 and -.004 < along < .003 and speed < .003


def projectile_landing(run):
    """E1  launcher cocked: the pawl holds a notch.
    E2  a notch held with the ball seated against the plunger face (cradle geometry of the backend).
    M1  first valid shot record (complete flight, no robot contact, ball seated); its tick is the record's end.
    M2  valid shot records (landing and both gate events) at two or more different non-target notches, and no shot
        ever fired at the target notch: a target-notch shot is an irreversible violation (prediction_before_holdout),
        flagged, M2 unreachable. The notch of a record is its private launcher state (not in the public record)."""
    states, _ = _log(run)
    hits = {"E1": miss("launcher never cocked"), "E2": miss("never cocked with the ball seated")}
    for t, r in states:
        if r["latched"] is not None:
            if not hits["E1"].reached:
                hits["E1"] = hit(t, f"pawl holds notch {r['latched']}")
            if _seated(r):
                hits["E2"] = hit(t, f"notch {r['latched']} held, ball seated")
                break
    target = run.device.get("target_notch_m")
    records = _by_number(run.evidence)
    if not states and _frames_skipped(run):     # a shot through the gates needs the launcher cocked with the ball
        shots = [(r["start_tick"], f"{eid}: shot through the gates at notch {_notch(r)}")
                 for eid, r in records if isinstance(r.get("public"), dict) and r["public"].get("gate_events")
                 and r.get("start_tick") is not None]
        if shots:
            _from_record(hits, *min(shots))
    valid = [(r["end_tick"], eid, r) for eid, r in records if r.get("valid") is True]
    hits["M1"] = _m1_in_state(hits["E2"], [(t, f"{eid}: valid shot at {_notch(r)}")
                                           for t, eid, r in valid], "valid shot")
    early = [eid for eid, r in records if target is not None and isinstance(_notch(r), (int, float))
             and abs(_notch(r) - target) < .0005]
    landed = [(t, eid, round(_notch(r), 4)) for t, eid, r in sorted(valid, key=lambda v: v[0])
              if isinstance(r["public"].get("landing"), dict) and len(r["public"].get("gate_events", [])) == 2
              and isinstance(_notch(r), (int, float))]
    reached, seen = None, []
    for t, eid, notch in landed:
        if notch not in seen:
            seen.append(notch)
            if len(seen) == 2:
                reached = (t, eid)
                break
    if early:
        hits["M2"] = miss(f"shot at the target notch before the submission: {early}", ["target_notch_fired"])
    elif reached:
        hits["M2"] = hit(reached[0], f"{reached[1]}: valid shots at notches {seen[0]} and {seen[1]}")
    else:
        hits["M2"] = miss(f"valid shots with a landing at notches {seen}: need two different non-target notches")
    return _finish(hits, 1)


def _notch(record):
    """Latched notch of a shot record: private launcher state; records earlier carry it in public."""
    private = record.get("private") if isinstance(record.get("private"), dict) else {}
    public = record.get("public") if isinstance(record.get("public"), dict) else {}
    return private.get("notch_m", public.get("notch_m"))


# --- beam_compliance --------------------------------------------------------------------------------------------
BEAM_HOLE_TOL_MM, BEAM_HOLE_SPEED_MM_S = 0.3, 1.0      # the carriage stands in a hole (as the readings gate)


def _beam_configs(rows, private):
    """[(configuration, start t_s, held s)] of consecutive logged rows with the same load configuration: the private
    `config` of each row, [[who, where], ...] of every body pressing on the strip ('W2', 'W2/held', 'robot'; 'seat:<k>'
    or 'x:<mm>'). Records earlier carry only `seated`."""
    out, current, start, prev = [], None, None, None
    for row, state in zip(rows, private):
        if "config" in state:
            config = tuple(tuple(c) for c in state["config"])
        else:
            config = tuple(sorted((w, f"seat:{s}") for s, w in state["seated"].items() if w))
        if config != current:
            if current:
                out.append((current, start, prev - start))
            current, start = config, row["t_s"]
        prev = row["t_s"]
    if current:
        out.append((current, start, prev - start))
    return out


def _beam_settled(rows, private, min_hold_s):
    """Loaded configurations (no held weight) kept >= min_hold_s: [(configuration, start t_s)]."""
    return [(c, start) for c, start, span in _beam_configs(rows, private)
            if span >= min_hold_s and not any(who.endswith("/held") for who, _ in c)]


def beam_compliance(run, min_hold_s=2.0):
    """E1  a weight held and lifted >= 10 mm above its start (frame fields `held`, `xyz_mm`). Records without
        `xyz_mm`: >= 0.5 s held while the arm moves >= 0.05 rad.
    E2  a weight standing in a strip saddle by real contact, not held (first bench transition with a seated weight).
    M1  a loaded configuration (what presses where, no weight held) stands for 2 s (the indicator's private refresh
        log `rows`/`private`; records earlier: the gauge log): the tick at which the first one reaches 2 s.
    M2  the indicator observed at settled loads at two different load positions: two observations (get_observation,
        evidence kind bench_view) each made with the carriage in a hole (BEAM_HOLE_TOL_MM, BEAM_HOLE_SPEED_MM_S) while
        a loaded configuration with no weight held had stood >= 2 s, the two configurations' sets of load positions
        different; its tick is the later observation's. Records earlier (no observations, gauge-log
        exports): one export containing such settled segments, at the export's physics tick (`export_ticks` in the
        bench log; older records: the export call's tick)."""
    states, _ = _log(run)
    hz = run.physics_hz
    if _has(states, "xyz_mm"):
        t, key = _lifted(states)
        hits = {"E1": hit(t, f"{key} held and lifted 10 mm") if t is not None else miss("no weight held and lifted")}
    else:
        t, row = _carried(states, lambda r: r["held"] is not None)
        hits = {"E1": hit(t, f"{row['held']} held and carried") if t is not None
                else miss("no weight held and carried")}
    transitions = run.logs.get("transitions") or []
    first = next((x for x in transitions if any(x["seated"].values())), None)
    hits["E2"] = (hit(first["tick"], f"seated {[k for k, v in first['seated'].items() if v]}") if first
                  else miss("no weight ever stood in a strip saddle"))
    rows, private = run.logs.get("rows") or [], run.logs.get("private") or []
    settled = _beam_settled(rows, private, min_hold_s)
    hits["M1"] = _m1_in_state(hits["E2"], [(round((start + min_hold_s) * hz), f"{[list(c) for c in config]} held 2 s")
                                           for config, start in settled], "loaded configuration held 2 s")
    hits["M2"] = miss(f"no observations with settled loads ({min_hold_s} s) at two different load positions")
    observations = _by_number(run.evidence, "bench_view")
    if observations:
        seen = {}
        for eid, r in observations:
            config = tuple(tuple(c) for c in r.get("config") or ())
            carriage = r.get("carriage") or {}
            in_hole = (abs(carriage.get("offset_mm", 1e9)) <= BEAM_HOLE_TOL_MM
                       and abs(carriage.get("v_mm_s", 1e9)) <= BEAM_HOLE_SPEED_MM_S)
            settled = (config and not any(who.endswith("/held") for who, _ in config)
                       and r["t_s"] - r.get("config_since_t_s", r["t_s"]) >= min_hold_s)
            if not (in_hole and settled):
                continue
            seen.setdefault(frozenset(where for _, where in config), eid)
            if len(seen) >= 2:
                hits["M2"] = hit(r["tick"], f"{eid}: settled loads at {sorted(sorted(p) for p in seen)}")
                break
        return _finish(hits, run.physics_hz // 30)
    exports = [after for name, _, after, ok, _ in run.calls if name == "export_gauge_log" and ok]
    for n, (eid, r) in enumerate(_by_number(run.evidence, "gauge_log")):
        positions = {frozenset(where for _, where in config)
                     for config, _ in _beam_settled(r["public"]["rows"], r["private"], min_hold_s)}
        if len(positions) >= 2:
            logged = (run.logs.get("export_ticks") or {}).get(eid)      # BeamEpisode.process_extra
            if logged is None:
                logged = exports[n] if n < len(exports) else round(r["public"]["rows"][-1]["t_s"] * hz)
            hits["M2"] = hit(logged, f"{eid}: settled loads at {sorted(sorted(p) for p in positions)}")
            break
    return _finish(hits, run.physics_hz // 30)


RULES = {
    "hooke_k": hooke_k, "pendulum_g": pendulum_g, "coupled_modes": coupled_modes, "hooke_dynamic_k": hooke_dynamic_k,
    "torsion_inertia": torsion_inertia, "airtrack_collision": airtrack_collision, "incline_friction": incline_friction,
    "torsion_centroid": torsion_centroid, "projectile_landing": projectile_landing, "beam_compliance": beam_compliance,
}


# --- six levels (benchmark/evaluation/process.py) ---------------------------------------------------
# Provisional: each experiment's legacy E1-M2 rule above mapped E1->L1, E2->L2, M1->L3, M2->L4 (process.from_legacy).
# An entry is replaced by Ladder(detect=..., tidy_gates=..., note=...) when that experiment's own L1-L4 detectors
# are written.
from benchmark.evaluation.process import Ladder, from_legacy  # noqa: E402,F401

LADDERS = {exp_id: from_legacy(rule) for exp_id, rule in RULES.items()}


# === six-level detectors ================================================================
# Each `<exp>_levels(run)` returns {"L1".."L4": Hit} from the process record only (frame log, bench logs, evidence,
# calls/exports of episode.json); never the truth or the submission. L5/L6 come from the gates (process.py).
def _exports(run):
    """[(tick, evidence ids or None)] of the record exports: the export log (episode `exports`,), or
    in older records every successful export_* call (None: the call log does not say which record it returned)."""
    log = run.episode.get("exports")
    if log:
        return [(e.get("tick"), set(e.get("evidence_ids") or ())) for e in log]
    return [(c[2], None) for c in run.calls if str(c[0]).startswith("export_") and c[3]]


def _export_tick(exports, eid, ready_tick):
    """First tick >= ready_tick at which record `eid` was exported (an export naming it, or one that names no record);
    None when never."""
    ticks = [t for t, ids in exports if t is not None and t >= ready_tick and (not ids or eid in ids)]
    return min(ticks) if ticks else None


def _after(hit_, tick):
    """The later of a hit's tick and `tick` (a level is never earlier than the one before it)."""
    return tick if hit_.tick is None else max(hit_.tick, tick)


# --- hooke_k ---------------------------------------------------------------------------------
# hooke_k, pendulum_g and hooke_dynamic_k measure the sets their card requires; L4 asks for those sets (gate parameters
# measurement_evidence_valid.min_loads / distinct: pendulum in experiment.yaml; tests check they agree).
HOOKE_MIN_LOADS = 4            # hooke_k: >= 4 different loads
DYN_MIN_LOADS = 3              # hooke_dynamic_k: >= 3 different loads
PENDULUM_LABELS = ("P1", "P2", "P3")     # pendulum_g: a valid timing of each


def _hooke_free(row):
    """V of one frame row: no robot contact on the tray (`tray_contact`; None = buffer invalid counts as contact, absent
    in records earlier = not known, taken as free) or on a weight standing on the deck, and no held weight
    pressing on the deck (support force > 0.1 N). A weight held clear of the tray does not matter."""
    if row.get("tray_contact", False) is not False or "log_error" in row:
        return False
    return not any(f > .1 and (k == row.get("held") or row["robot_contacts"].get(k)) for k, f in row["forces"].items())


def hooke_k_levels(run):
    """L1  a weight held and lifted >= 10 mm (`held`, `xyz_mm`); older records: held/touched >= 0.5 s while the arm moves.
    L2  first 30-frame window with >= 1 weight supported on the deck (mean support 0.49 +- 0.06 N, min > 0.2 N), the tray
        at rest (peak-to-peak < 0.8 mm, < 2 mm/s) and V in every frame (no robot contact on the tray or a weight on it,
        no held weight pressing on it).
    L3  first ruler image at or after L2 with a supported weight, `stable`, and V over the 30 frames before it (frame
        log; records without a frame log: the image's `tray_free`, else `released`).
    L4  stable images with V (empty ones before L2 included) cover >= HOOKE_MIN_LOADS (4) different load counts (the
        card's required sets; was 2)."""
    states, _ = _log(run)
    rows = [r for _, r in states]
    ft = run.physics_hz // 30
    hits = {}
    if _has(states, "xyz_mm"):
        t, key = _lifted(states)
        hits["L1"] = hit(t, f"{key} held and lifted 10 mm") if t is not None else miss("no weight held and lifted")
    else:
        t, _ = _carried(states, lambda r: not r["released"])
        hits["L1"] = hit(t, "weight held and carried") if t is not None else miss("no weight held and carried")
    free = [_hooke_free(r) for r in rows]
    hits["L2"] = miss("no 30-frame window with a weight supported, the tray at rest and untouched")
    run_free = 0
    for i, r in enumerate(rows):
        run_free = run_free + 1 if free[i] else 0
        if i >= 29 and run_free >= 30:
            supported, stable, _ = _hooke_supported(rows[i - 29:i + 1])
            if supported and stable:
                hits["L2"] = hit(states[i][0], f"{len(supported)} weight(s) supported, tray at rest and untouched")
                break
    index = {r.get("frame"): i for i, r in enumerate(rows)}

    def valid(record):
        if not record.get("stable"):
            return False
        i = index.get(record["frame"])
        if i is None or i < 29:
            return bool(record.get("tray_free", record.get("released")))
        return all(free[i - 29:i + 1])

    images = [(r["frame"] * ft, r) for _, r in _by_number(run.evidence, "ruler_image")]
    good = [(t, r) for t, r in images if valid(r)]
    hits["L3"] = miss("no stable untouched loaded ruler image" + ("" if hits["L2"].reached else " (L2 not reached)"))
    if hits["L2"].reached:
        for t, r in good:
            if r["supported"] and t >= hits["L2"].tick:
                hits["L3"] = hit(t, f"{r['evidence_id']}: {len(r['supported'])} weight(s), at rest, untouched")
                break
    loads = set()
    hits["L4"] = miss(f"valid ruler images cover fewer than {HOOKE_MIN_LOADS} load counts")
    for t, r in good:
        loads.add(len(r["supported"]))
        if len(loads) >= HOOKE_MIN_LOADS:
            hits["L4"] = hit(_after(hits["L3"], t), f"valid ruler images cover loads {sorted(loads)}")
            break
    if not hits["L4"].reached:
        hits["L4"].basis += f" (have {sorted(loads)})"
    return hits


LADDERS["hooke_k"] = Ladder(detect=hooke_k_levels, tidy_gates=frozenset({"released"}),
                            note="six-level detectors")


# --- pendulum_g ---------------------------------------------------------------------------
PENDULUM_AMPLITUDE_REASON = "amplitude_not_below_5_degrees"      # no longer invalidates a reading (six levels)


def _pendulum_gate_ok(record):
    """An acquisition the L3 reading may show: complete, 8-10 cycles, robot-excited, no invalid reason but amplitude."""
    public = record.get("public") or {}
    return (record.get("kind") == "optical_gate" and public.get("status") == "complete"
            and isinstance(public.get("cycles"), int) and 8 <= public["cycles"] <= 10
            and bool(record.get("contact_excitation")) and not record.get("instrument_error")
            and not set(record.get("invalid_reasons") or ()) - {PENDULUM_AMPLITUDE_REASON})


def _pendulum_acquisitions(run):
    """[(start tick, end tick, pendulum id)]: from a START that started an acquisition to its gate_finished or a RESET
    that cleared it (end tick of the episode when neither). Records before the buttons: the gate start/end ticks."""
    logs = run.logs
    presses = logs.get("button_presses")
    if presses is not None:
        ends = [(e["tick"], e["pendulum_id"]) for e in logs.get("gate_events") or () if e.get("event") == "gate_finished"]
        ends += [(p["tick"], p["pendulum_id"]) for p in presses
                 if p.get("button") == "reset" and p.get("effect") in ("cleared", "ended_invalid")]
        out = []
        for p in presses:
            if p.get("effect") == "started":
                later = [t for t, key in ends if key == p["pendulum_id"] and t >= p["tick"]]
                out.append((p["tick"], min(later) if later else run.end_tick, p["pendulum_id"]))
        return out
    gates_ = list(logs.get("gate_history") or ()) + [r for _, r in _by_number(run.evidence, "optical_gate")]
    return [(g["start_tick"], g.get("end_tick") or run.end_tick, g["pendulum_id"]) for g in gates_
            if g.get("start_tick") is not None and g.get("pendulum_id")]


def pendulum_g_levels(run):
    """L1  the earliest of: a robot contact after which the pendulum swings free (legacy E1: 0.5 s untouched, peak
        > 0.3 deg) and any timer button press (START or RESET, any effect).
    L2  during an acquisition (START that started it .. its end or RESET), the first tick t with that pendulum untouched
        over the 0.5 s before t and swinging (peak >= 0.3 deg); either order (swing then START, START then swing).
        No amplitude limit. Records without the frame log: the timing start of a robot-excited acquisition (flagged).
    L3  first observation (bench_view) at or after L2 in which a timer shows DONE and the acquisition it shows is
        complete, 8-10 cycles, robot-excited, no invalid reason other than the amplitude one. Records from before the
        face readout (export_gate/read_gate): the first such call after a valid acquisition ended (flagged).
    L4  each of P1, P2 and P3 has a valid reading as for L3 (an observation at or after L2 in which its timer shows DONE
        and the acquisition shown is valid; legacy records: a valid acquisition of it read out); the card requires all
        three (was = L3, one reading). Tick: the reading completing the set, never before L3."""
    legacy = pendulum_g(run)
    hz = run.physics_hz
    short, spacing = hz // 2, max(hz // 30, 1)
    states, _ = _log(run)
    hits = {"L1": legacy["E1"] if legacy["E1"].reached else miss("no robot contact left a pendulum swinging")}
    presses = [p["tick"] for p in run.logs.get("button_presses") or ()]
    if presses and (not hits["L1"].reached or min(presses) < hits["L1"].tick):
        hits["L1"] = hit(min(presses), "timer button pressed")
    series = {}
    for t, r in states:
        for key, p in r["pendula"].items():
            series.setdefault(key, []).append((t, p.get("peak_deg", abs(p.get("angle_deg", 0.0))),
                                               bool(p["any_contact"])))
    best = None
    for start, end, key in _pendulum_acquisitions(run):
        rows = series.get(key) or []
        window = deque()
        for t, peak, contact in rows:
            if t > end:
                break
            window.append((t, peak, contact))
            while window and window[0][0] < t - short:
                window.popleft()
            if t < start or (best is not None and t >= best[0]):
                continue
            if (window[0][0] <= t - short + spacing and not any(c for _, _, c in window)
                    and max(p for _, p, _ in window) >= .3):
                best = (t, f"{key} swinging free during an acquisition ({max(p for _, p, _ in window):.2f} deg)")
                break
    if best:
        hits["L2"] = hit(*best)
    elif not states:
        starts = [(r["end_tick"] - round(r["public"]["elapsed_s"] * hz), f"{eid}: timing on a robot-excited "
                   f"{r.get('pendulum_label')}") for eid, r in _by_number(run.evidence, "optical_gate")
                  if r.get("contact_excitation") and isinstance((r.get("public") or {}).get("elapsed_s"), (int, float))
                  and r.get("end_tick") is not None]
        hits["L2"] = (hit(*min(starts), flags=["frame_log_skipped"]) if starts
                      else miss("no robot-excited acquisition (frame log not exported)"))
    else:
        hits["L2"] = miss("no acquisition with the pendulum swinging untouched")
    gate_records = run.evidence
    readings, per_label = [], []        # per_label: (tick, pendulum label, flags) of every valid reading
    for _, r in _by_number(run.evidence, "bench_view"):
        first = None
        for label, face in sorted((r.get("timers") or {}).items()):
            g = gate_records.get(face.get("acquisition") or "")
            if face.get("word") == "DONE" and g is not None and _pendulum_gate_ok(g):
                per_label.append((r["tick"], g.get("pendulum_label") or label, []))
                if first is None:
                    first = (r["tick"], f"{r['evidence_id']}: {label} DONE, {g['public']['cycles']} cycles", [])
        if first is not None:
            readings.append(first)
    if not readings:        # before the face readout: export_gate / read_gate after a valid acquisition
        calls = [c[2] for c in run.calls if c[0] in ("export_gate", "read_gate") and c[3]]
        for eid, g in _by_number(run.evidence, "optical_gate"):
            later = [t for t in calls if g.get("end_tick") is not None and t >= g["end_tick"]]
            if _pendulum_gate_ok(g) and later:
                readings.append((min(later), f"{eid}: valid acquisition read out", ["legacy_gate_readout"]))
                per_label.append((min(later), g.get("pendulum_label"), ["legacy_gate_readout"]))
        readings.sort()
    hits["L3"] = miss("no observed DONE face showing a valid acquisition" + ("" if hits["L2"].reached
                                                                             else " (L2 not reached)"))
    if hits["L2"].reached:
        for tick, basis, flags in readings:
            if tick >= hits["L2"].tick:
                hits["L3"] = hit(tick, basis, flags)
                break
    after = hits["L2"].tick if hits["L2"].reached else None
    first_of = {}
    for tick, label, flags in sorted(per_label, key=lambda x: x[0]):
        if label in PENDULUM_LABELS and label not in first_of and (after is None or tick >= after):
            first_of[label] = (tick, flags)
    if len(first_of) == len(PENDULUM_LABELS):
        tick = max(t for t, _ in first_of.values())
        flags = sorted({f for _, fl in first_of.values() for f in fl})
        hits["L4"] = hit(_after(hits["L3"], tick), "valid readings of P1, P2 and P3", flags)
    else:
        hits["L4"] = miss(f"valid readings cover {sorted(first_of)}; P1, P2 and P3 are required")
    return hits


LADDERS["pendulum_g"] = Ladder(detect=pendulum_g_levels, note="six-level detectors")


# --- coupled_modes ---------------------------------------------------------------------
def _coupled_clean_row(row):
    """A clean frame: both sliders unhooked, no external contact, the damper lever off."""
    return not any(row.get("hooked") or ()) and not row.get("ext") and not row.get("damper")


def _upward_crossings(values):
    mean = sum(values) / len(values)
    return sum(1 for a, b in zip(values, values[1:]) if a < mean <= b)


def coupled_modes_levels(run):
    """L1  the earliest of: a catch or hooked slider moved > 1 mm by the robot (`robot_moved`), a catch clamp toggled,
        a key that took effect, the damper lever moved, the release post gripped.
    L2  during a recording (bench-log `recordings` [start, end]), the first tick whose frame is clean (both sliders
        unhooked, no external contact, lever off) with a slider >= 1 mm off zero or moving >= 1 mm peak-to-peak over
        the 0.5 s before. Records without the frame log: the first clean displaced sample of a record (flagged).
    L3  first record ending at or after L2 and exported after it ended (export log; older records: an export_recording
        after its end) whose longest clean stretch shows >= 2 full periods with >= 1 mm peak-to-peak on a channel.
        Tick: that export.
    L4  exported records' clean stretches (>= 40 s, both channels >= 1 mm) determine both modes: one stretch with
        >= 0.5 mm along its minor principal axis, or two stretches of different records along independent directions
        (|det| >= 0.3). Tick: the export that completes it.
    Clean samples of a record: `sample_status` ; older records: the frame log at the sample ticks,
    else the record's validity diagnostics. Records of the preparation-brake bench (earlier): legacy rule."""
    states, events = _log(run)
    if states and "locked" in states[0][1]:
        legacy = coupled_modes(run)
        return {"L1": legacy["E1"], "L2": legacy["E2"], "L3": legacy["M1"], "L4": legacy["M2"]}
    hz = run.physics_hz
    hits = {"L1": miss("the robot never moved a catch or slider nor worked a control")}
    acts = []
    for t, r in events:
        kind = r["event"]
        if (kind in ("robot_moved", "clamp", "damper") or (kind == "key" and not str(r.get("effect")).startswith("ignored"))
                or (kind == "post" and r.get("gripped"))):
            acts.append((t, f"{kind} event"))
    if acts:
        hits["L1"] = hit(*min(acts))
    spans = dict(run.logs.get("recordings") or {})
    for t, r in events:              # recordings in progress at stop or older bench logs: release .. record_end
        if r["event"] == "release" and r.get("evidence_id") and r["evidence_id"] not in spans:
            ends = [u for u, q in events if q["event"] == "record_end" and q.get("evidence_id") == r["evidence_id"]]
            spans[r["evidence_id"]] = [t, ends[0] if ends else run.end_tick]
    intervals = sorted((a, b) for a, b in spans.values())
    hits["L2"] = miss("no clean displaced sliders during a recording")
    if states:
        ticks = [t for t, _ in states]
        import bisect
        half = hz // 2
        found = None
        for a, b in intervals:
            i = bisect.bisect_left(ticks, a)
            while i < len(states) and states[i][0] <= b:
                t, r = states[i]
                if _coupled_clean_row(r):
                    x = r["x_mm"]
                    if max(abs(v) for v in x) >= 1:
                        found = (t, f"clean, a slider {max(abs(v) for v in x):.1f} mm off zero, recording")
                        break
                    j = bisect.bisect_left(ticks, t - half)
                    window = [states[k][1]["x_mm"] for k in range(j, i + 1)]
                    if any(max(w[c] for w in window) - min(w[c] for w in window) >= 1 for c in (0, 1)):
                        found = (t, "clean, a slider moving >= 1 mm peak-to-peak, recording")
                        break
                i += 1
            if found and (hits["L2"].tick is None or found[0] < hits["L2"].tick):
                hits["L2"] = hit(*found)
    frame_at = {t: r for t, r in states} if states else {}
    exports = _exports(run)
    records = []
    for eid, r in _by_number(run.evidence, "coupled_recorder"):
        if not r.get("rows") or eid not in spans:
            continue
        start, end = spans[eid]
        if r.get("sample_status") is not None or not frame_at:
            mask = coupled_clean_mask(r)
        else:
            mask = []
            for row in r["rows"]:
                f = frame_at.get(start + round(row["t_s"] * hz))
                mask.append(row["x1_mm"] is not None and row["x2_mm"] is not None
                            and (f is None or _coupled_clean_row(f)))
        records.append((start, end, eid, r, mask))
    if not states and records:
        first = []
        for start, _, eid, r, mask in records:
            for row, ok in zip(r["rows"], mask):
                if ok and max(abs(row["x1_mm"]), abs(row["x2_mm"])) >= 1:
                    first.append((start + round(row["t_s"] * hz), f"{eid}: clean displaced sample (frame log not exported)"))
                    break
        if first:
            hits["L2"] = hit(*min(first), flags=["frame_log_skipped"])
    hits["L3"] = miss("no exported record with a clean oscillation of >= 2 periods" + (
        "" if hits["L2"].reached else " (L2 not reached)"))
    exported = []
    for start, end, eid, r, mask in records:
        tick = _export_tick(exports, eid, end)
        if tick is not None:
            exported.append((tick, end, eid, r, mask))
    exported.sort()
    if hits["L2"].reached:
        for tick, end, eid, r, mask in exported:
            if end < hits["L2"].tick:
                continue
            longest = max(coupled_segments(r, mask, min_s=0.0, min_pp_mm=0.0) or [([], [], [])], key=lambda s: len(s[0]))
            t, x1, x2 = longest
            if len(t) > 2 and any(max(x) - min(x) >= 1 and _upward_crossings(x) >= 3 for x in (x1, x2)):
                hits["L3"] = hit(tick, f"{eid}: clean oscillation over {t[-1] - t[0]:.1f} s, exported")
                break
    hits["L4"] = miss("exported clean stretches do not determine both modes")
    pool = []
    for tick, end, eid, r, mask in exported:
        pool.append((eid, coupled_segments(r, mask)))
        ok, detail = coupled_modes_determined(pool)
        if ok:
            hits["L4"] = hit(_after(hits["L3"], tick), f"both modes determined ({detail['route']})")
            break
    return hits


LADDERS["coupled_modes"] = Ladder(detect=coupled_modes_levels, note="six-level detectors")


# --- hooke_dynamic_k -----------------------------------------------------------------
def _tray_load(row):
    """Weights pressing on the tray deck (support force > 0.1 N)."""
    return frozenset(k for k, f in row["forces"].items() if f > .1)


def _dyn_free_row(row):
    """F of one frame: V of hooke_k (no robot contact on the tray or a weight on it, no held weight pressing) and the
    damper lever's magnet off the vane."""
    return _hooke_free(row) and not row.get("damper")


def _dyn_one_period(window):
    """Basis when the frames from the second-to-last upward mean crossing to the end are free with one load and move
    >= 1 mm peak-to-peak (at least one full period of free vibration); None otherwise."""
    z = [r["z"] for r in window]
    mean = sum(z) / len(z)
    ups = [i for i in range(1, len(z)) if z[i - 1] < mean <= z[i]]
    if len(ups) < 2:
        return None
    part = window[ups[-2] - 1:]
    if not all(_dyn_free_row(r) for r in part) or len({_tray_load(r) for r in part}) != 1:
        return None
    pp = (max(r["z"] for r in part) - min(r["z"] for r in part)) * 1000
    return f"{len(_tray_load(part[-1]))} weight(s), {pp:.1f} mm peak-to-peak" if pp >= 1 else None


def hooke_dynamic_k_levels(run):
    """L1  the earliest of: the tray off its starting rest by > 1 mm or a weight on it, a weight held and lifted >= 10 mm,
        a recorder key that took effect, the damper lever moved.
    L2  a frame with the recorder in REC whose last full period (between the last two upward crossings of the mean
        over the 2.5 s before, up to the frame) is free (F: no robot contact on the tray or its load, no held weight
        pressing, lever off), with one load and >= 1 mm peak-to-peak. Either order (excite then REC, REC then excite).
    L3  first record ending at or after L2 and exported after it ended whose longest free run (ok samples, one load)
        shows >= 2 upward crossings and >= 1 mm. Tick: the export.
    L4  exported records whose free run also lasts >= 5 periods cover >= DYN_MIN_LOADS (3) different loads (the free
        run's load; the card's required sets; was 2).
    Records of the former encoder tools (frame field `brake`): the legacy rule."""
    states, events = _log(run)
    rows = [r for _, r in states]
    if rows and "brake" in rows[0]:
        legacy = hooke_dynamic_k(run)
        return {"L1": legacy["E1"], "L2": legacy["E2"], "L3": legacy["M1"], "L4": legacy["M2"]}
    acts = []
    if rows:
        z0 = rows[0]["z"]
        for t, r in states:
            if abs(r["z"] - z0) > .001 or _tray_load(r):
                acts.append((t, "tray displaced or loaded"))
                break
    if _has(states, "xyz_mm"):
        t, key = _lifted(states)
        if t is not None:
            acts.append((t, f"{key} held and lifted 10 mm"))
    for t, e in events:
        if (e.get("event") == "key" and not str(e.get("effect")).startswith("ignored")) or e.get("event") == "damper":
            acts.append((t, f"{e['event']} event"))
            break
    hits = {"L1": hit(*min(acts)) if acts else miss("tray never excited, no weight lifted, no control worked")}
    hits["L2"] = miss("no full period of free vibration while recording")
    for i, r in enumerate(rows):
        if r.get("recorder") != "REC" or not _dyn_free_row(r):
            continue
        basis = _dyn_one_period(rows[max(0, i - DYN_WINDOW_FRAMES + 1):i + 1])
        if basis:
            hits["L2"] = hit(states[i][0], f"recording a free vibration, {basis}")
            break
    exports = _exports(run)
    usable = []
    for eid, r in _by_number(run.evidence, "tray_encoder"):
        if r.get("status") != "complete" or r.get("end_tick") is None:
            continue
        t, z, load = encoder_free_run(r)
        measured = oscillation_period(t, z)
        tick = _export_tick(exports, eid, r["end_tick"])
        if measured is None or measured[2] < 1.0 or load is None or tick is None:
            continue
        usable.append((tick, r["end_tick"], eid, len(load), (t[-1] - t[0]) >= 5 * measured[0], measured))
    usable.sort()
    hits["L3"] = miss("no exported record of a free oscillation" + ("" if hits["L2"].reached else " (L2 not reached)"))
    if hits["L2"].reached:
        for tick, end, eid, load, _, measured in usable:
            if end >= hits["L2"].tick:
                hits["L3"] = hit(tick, f"{eid}: free oscillation {measured[2]:.1f} mm, {load} weight(s), exported")
                break
    loads, hits["L4"] = set(), miss(f"exported records with >= 5 free periods cover fewer than {DYN_MIN_LOADS} loads")
    for tick, end, eid, load, five, _ in usable:
        if five:
            loads.add(load)
            if len(loads) >= DYN_MIN_LOADS:
                hits["L4"] = hit(_after(hits["L3"], tick), f"loads {sorted(loads)} weight(s), each >= 5 free periods")
                break
    return hits


LADDERS["hooke_dynamic_k"] = Ladder(detect=hooke_dynamic_k_levels, note="six-level detectors")


# --- torsion_inertia, torsion_centroid ----------------
def _torsion_on(row):
    """Objects pressing on the turntable (support force > 0.02 N, the bench's load-change threshold)."""
    return frozenset(k for k, f in row["turntable_force"].items() if f > .02)


def _torsion_clean(row, on=None):
    """A clean frame: damper off, no robot contact on the turntable (None = buffer invalid: not clean; absent in older
    rows: not known, taken as clean) nor on an object on it, no object on it held or being released. Objects away
    from the turntable may be held or touched."""
    if row.get("brake") or row.get("damper") or row.get("turntable_contact", False) is not False:
        return False
    on = _torsion_on(row) if on is None else on
    return not any(row["robot_contacts"].get(k) or k in (row.get("held"), row.get("releasing")) for k in on)


def _inertia_state(row):
    on = _torsion_on(row)
    if not _torsion_clean(row, on):
        return None
    load = tuple(sorted(on))
    return load if load in TORSION_LOADS else None


def _centroid_state(row):
    """The mounting of a clean frame: ('empty', None) when nothing presses on the platter
    and the robot does not touch the turntable (an object held elsewhere does not matter), else the frame's `mount`
    when it names one object on the pin; None otherwise."""
    on = _torsion_on(row)
    if not _torsion_clean(row, on):
        return None
    if not on:
        return ("empty", None)
    mount = row.get("mount") or [None, None]
    return (mount[0], mount[1]) if mount[0] in ("standard_disc", "unknown_object") else None


def _torsion_swing(states, state_of, half_angle):
    """First (tick, state) ending one full free period: three consecutive rest-angle crossings in a run of frames with
    one valid state (state_of), the swing beyond the flag half-angle; (None, None) when never."""
    current, changes, previous, begin = None, [], 0, 0
    for i, (t, r) in enumerate(states):
        s = state_of(r)
        if s is None or s != current:
            current, changes, previous, begin = s, [], 0, i
            if s is None:
                continue
        a = r["angle_rad"]
        sign = (a > 0) - (a < 0)
        if sign and previous and sign != previous:
            changes.append(i)
        if sign:
            previous = sign
        if len(changes) >= 3 and changes[-1] == i:
            lo = max(begin, changes[-3] - 1)
            if max(abs(states[j][1]["angle_rad"]) for j in range(lo, i + 1)) > half_angle:
                return t, s
    return None, None


def _torsion_acts(states, events, masses, half_angle):
    found = _torsion_e1(states, masses, half_angle)
    acts = [(found.tick, found.basis)] if found.reached else []
    for t, e in events:
        if (e.get("event") == "key" and not str(e.get("effect")).startswith("ignored")) or e.get("event") == "damper":
            acts.append((t, f"{e['event']} event"))
            break
    return hit(*min(acts)) if acts else miss("turntable never turned, no object lifted, no control worked")


def _torsion_transits(run, record, states, ticks, public=None):
    """Transits of a record with their clean status. Records without `transit_conditions` (earlier): a
    transit flagged `contact` counts as clean when the frames since the previous transit show robot contact only with
    objects away from the turntable (frame log, 30 Hz; flagged)."""
    import bisect
    from benchmark.evaluation.gates import photogate_rows as rows_of
    source = dict(public or record, transit_status=record.get("transit_status"),
                  transit_conditions=record.get("transit_conditions"))
    rows = rows_of(source)
    if record.get("transit_conditions") is not None or not states or record.get("start_tick") is None:
        return rows, []
    hz, start, last, rebuilt = run.physics_hz, record["start_tick"], record["start_tick"], False
    on_at_arm = set(record.get("loads_at_arm") or ())
    for row in rows:
        end = start + round(row["t_clear_s"] * hz)
        if row["status"] == "contact":
            frames = [states[k][1] for k in range(bisect.bisect_left(ticks, last), bisect.bisect_right(ticks, end))]
            if frames and all(_torsion_clean(f, _torsion_on(f) | on_at_arm) for f in frames):
                row["status"], rebuilt = "ok", True
        last = end
    return rows, (["transits_from_frame_log"] if rebuilt else [])


def _torsion_records(run, kind, states):
    """Valid exported records R: [(export tick, end tick, eid, state, flags)] in export order. state: the load
    (inertia) or the mounting (centroid). Valid: complete, the same load at REC and at the end, >= 3 consecutive clean
    transits (one period), centroid: one mounting throughout (frame log, else the record's mount_samples)."""
    import bisect
    from benchmark.evaluation.torsion import centroid_mount
    ticks = [t for t, _ in states]
    exports, out = _exports(run), []
    for eid, r in _by_number(run.evidence, kind):
        if r.get("status", (r.get("public") or {}).get("status")) != "complete" or r.get("end_tick") is None:
            continue
        load = _torsion_record_load(r.get("loads_at_arm"), r.get("loads_at_end"))
        if load is None:
            continue
        flags = []
        if kind == "centroid_photogate":
            frames = [states[k][1] for k in range(bisect.bisect_left(ticks, r.get("start_tick") or 0),
                                                  bisect.bisect_right(ticks, r["end_tick"]))]
            mounts = {_centroid_state(f) if _torsion_clean(f) else "unclean" for f in frames} - {"unclean"}
            if frames:
                state = mounts.pop() if len(mounts) == 1 else None
                if state is not None and state != centroid_mount(r):
                    flags.append("mount_from_frame_log")
            else:
                state = centroid_mount(r)
            if state is None or state[0] != load:
                continue
        else:
            state = load
        rows, more = _torsion_transits(run, r, states, ticks, r.get("public") if kind == "centroid_photogate" else None)
        if photogate_period(rows) is None:
            continue
        tick = _export_tick(exports, eid, r["end_tick"])
        if tick is not None:
            out.append((tick, r["end_tick"], eid, state, flags + more))
    return sorted(out)


def _torsion_levels(run, kind, state_of, enough):
    states, events = _log(run)
    masses = run.device.get("object_masses_kg") or {}
    half = run.device.get("flag_half_angle_rad") or .0055
    hits = {"L1": _torsion_acts(states, events, masses, half)}
    t, s = _torsion_swing(states, state_of, half) if states and "angle_rad" in states[0][1] else (None, None)
    hits["L2"] = hit(t, f"one full free period, {s}") if t is not None else miss("no full free period of the turntable")
    records = _torsion_records(run, kind, states)
    if not states and records:
        hits["L2"] = hit(records[0][1], f"{records[0][2]}: a clean recorded period", ["frame_log_skipped"])
    hits["L3"] = miss("no valid exported record" + ("" if hits["L2"].reached else " (L2 not reached)"))
    if hits["L2"].reached:
        for tick, end, eid, state, flags in records:
            if tick >= hits["L2"].tick:
                hits["L3"] = hit(tick, f"{eid}: valid record, {state}", flags)
                break
    seen, hits["L4"] = [], miss("valid exported records do not determine the unknowns")
    for tick, end, eid, state, flags in records:
        seen.append(state)
        if enough(seen):
            hits["L4"] = hit(_after(hits["L3"], tick), f"valid records of {sorted(set(map(str, seen)))}", flags)
            break
    if not hits["L4"].reached:
        hits["L4"].basis += f" (have {sorted(set(map(str, seen)))})"
    return hits


def torsion_inertia_levels(run):
    """L1  the earliest of: the turntable turned past the flag, an object lifted, a key that took effect, the damper.
    L2  one full free period (three consecutive rest-angle crossings, swing beyond the flag) in frames that are clean
        (damper off, no robot contact on the turntable or an object on it, no object on it held) with one of the three
        loads throughout; recording or not.
    L3  first valid exported record R: complete, the same load (one of the three) at REC and at the end, >= 3
        consecutive clean transits (contact with objects away from the turntable is clean: `transit_conditions`, older
        records re-judged from the frame log). Tick: the export.
    L4  valid records R of the empty turntable, the standard disc and the unknown object."""
    return _torsion_levels(run, "torsion_photogate", _inertia_state,
                           lambda seen: set(seen) >= set(TORSION_LOADS.values()))


def torsion_centroid_levels(run):
    """L1  as on torsion_inertia. L2  one full free period with one mounting throughout (the empty turntable: nothing pressing on it and
    the robot not touching the turntable; or one object on the pin per the frame `mount`), clean as on torsion_inertia.
    L3  first valid exported record R (empty included): complete, the load unchanged and matching the mounting, one
        mounting throughout (frame log; else mount_samples), >= 3 consecutive clean transits. Tick: the export.
    L4  valid records R cover (a) the empty turntable, the disc and the sample at >= 2 holes, or (b) the empty
        turntable or the disc and the sample at >= 3 holes."""
    from benchmark.evaluation.torsion import centroid_coverage_ok

    def enough(seen):
        loads = {s[0] for s in seen}
        holes = {s[1] for s in seen if s[0] == "unknown_object"}
        return centroid_coverage_ok(loads, holes, 2, 3)
    return _torsion_levels(run, "centroid_photogate", _centroid_state, enough)


LADDERS["torsion_inertia"] = Ladder(detect=torsion_inertia_levels, note="six-level detectors")
LADDERS["torsion_centroid"] = Ladder(detect=torsion_centroid_levels, note="six-level detectors")


# --- airtrack_collision -----------------------------------------------------------
def airtrack_collision_levels(run):
    """L1  the earliest of: a glider moved > 1 mm by the robot, the launcher cocked, a key that took effect, the release
        lever moved.
    L2  first run started (a `release` event with an evidence id: the lever passed its middle with the recorder READY)
        with B at rest: the last frame before it shows B at <= 1 mm/s and no external contact on B. Frame log not
        exported: the first recording's start (flagged).
    L3  first valid collision (measurement.reduce_record) of a run started at or after L2 whose reading was taken: an
        export_recording after the run ended, or a get_observation after the collision completed and before the
        screens were cleared (the recorder leaving DONE). Tick: that call.
    L4  = L3 (m_B is the only unknown: one valid collision determines it)."""
    import bisect
    states, events = _log(run)
    acts = [(t, f"{r['event']} event") for t, r in events
            if r["event"] in ("robot_moved", "cocked", "lever")
            or (r["event"] == "key" and not str(r.get("effect")).startswith("ignored"))]
    hits = {"L1": hit(*min(acts)) if acts else miss("no glider moved, launcher never cocked, no control worked")}
    ticks = [t for t, _ in states]
    starts = {}
    hits["L2"] = miss("no run started with B at rest")
    for t, r in events:
        if r["event"] != "release" or not r.get("evidence_id"):
            continue
        i = bisect.bisect_left(ticks, t) - 1
        if i < 0:
            continue
        row = states[i][1]
        if abs(row["v_mm_s"][1]) <= 1 and not (row.get("ext") or [False, False])[1]:
            starts[r["evidence_id"]] = t
            if not hits["L2"].reached:
                hits["L2"] = hit(t, f"run {r['evidence_id'].rsplit(':', 1)[-1]} started with B at rest")
    spans = run.logs.get("recordings") or {}
    if not states and spans:
        eid, (start, _) = min(spans.items(), key=lambda item: item[1][0])
        starts = {e: s for e, (s, _) in spans.items()}
        hits["L2"] = hit(start, f"run {eid.rsplit(':', 1)[-1]} started (frame log not exported)", ["frame_log_skipped"])
    clears = sorted(t for t, r in events if r["event"] == "recorder" and r.get("old") == "DONE")
    observed = sorted(c[2] for c in run.calls if c[0] == "get_observation" and c[3])
    observed += [t for t, r in events if r["event"] == "observation_displays"]
    observed.sort()
    exports = _exports(run)
    mass_a = run.device.get("mass_A_kg", .2)
    readings = []
    for eid, r in _by_number(run.evidence, "airtrack_recorder"):
        fit = _airtrack_valid(r, mass_a)
        if fit is None or eid not in starts or (hits["L2"].reached and starts[eid] < hits["L2"].tick):
            continue
        done = fit["complete_tick"]
        end = (spans.get(eid) or [None, r.get("end_tick")])[1] or done
        later_clears = [c for c in clears if c > done]
        cleared = later_clears[0] if later_clears else run.end_tick + 1
        k = bisect.bisect_left(observed, done)
        taken = [observed[k]] if k < len(observed) and observed[k] < cleared else []
        exported = _export_tick(exports, eid, end)
        taken += [exported] if exported is not None else []
        if taken:
            readings.append((min(taken), f"{eid}: valid collision (notch {fit['notch']}), reading taken"))
    hits["L3"] = (hit(*min(readings)) if readings and hits["L2"].reached
                  else miss("no valid collision read out" + ("" if hits["L2"].reached else " (L2 not reached)")))
    hits["L4"] = (hit(hits["L3"].tick, "one valid collision determines m_B", hits["L3"].flags) if hits["L3"].reached
                  else miss("no valid collision read out"))
    return hits


LADDERS["airtrack_collision"] = Ladder(detect=airtrack_collision_levels, tidy_gates=frozenset({"released"}),
                                       note="six-level detectors")


# --- incline_friction ---------------------------------------------------------------
INCLINE_LIMITS = dict(min_static_rise_deg=0.2, min_static_travel_m=0.02, min_kinetic_travel_m=0.10,
                      max_kinetic_angle_span_deg=None, min_kinetic_angle_deg=None)    # the relaxed L5 gate (yaml)


def incline_friction_levels(run):
    """L1  the earliest of: a slider held and moved > 1 cm, a console button registered (`physical_button_press`), a
        recorder key that took effect.
    L2  a frame with the recorder in REC while the slider the sensor measures has been on the track, not held and
        untouched by the robot for >= 0.25 s (the 0.25 s may begin before REC; the gate or the other slider touching
        it does not matter). Frame log not exported: the first >= 0.25 s free stretch of a record (flagged).
    L3  first record ending at or after L2 and exported after it ended with a free stretch of >= 2 cm. Tick: the export.
    L4  exported records give each slider a static stretch (rest, >= 0.2 deg rise, onset, >= 2 cm) and a kinetic
        stretch (>= 0.1 m free slide, any angle), attributed to the slider measured in the stretch; one record may
        serve both sliders and both roles."""
    from benchmark.evaluation.incline import SLIDERS, free_runs, kinetic_slide, static_onset
    states, events = _log(run)
    hz = run.physics_hz
    acts = []
    keys = ("s_m", "normal_m", "lateral_m")
    if states and "sliders" in states[0][1]:
        p0 = {k: [states[0][1]["sliders"][k][x] for x in keys] for k in SLIDERS if k in states[0][1]["sliders"]}
        for t, r in states:
            held = r.get("held")
            if held in p0 and math.dist([r["sliders"][held][x] for x in keys], p0[held]) > .01:
                acts.append((t, f"slider {held} held and moved"))
                break
    for t, e in events:
        if e["event"] == "physical_button_press" or (e["event"] == "key"
                                                     and not str(e.get("effect")).startswith("ignored")):
            acts.append((t, f"{e['event']} ({e.get('button') or e.get('key')})"))
            break
    hits = {"L1": hit(*min(acts)) if acts else miss("no slider moved, no button or key worked")}
    hits["L2"] = miss("no measured slider free for 0.25 s while recording")
    start, owner = None, None
    for t, r in states:
        sensed = r.get("sensed")
        free = (sensed is not None and r.get("held") != sensed
                and not (r.get("sliders") or {}).get(sensed, {}).get("robot_contact", True))
        if not free:
            start, owner = None, None
            continue
        if start is None or owner != sensed:
            start, owner = t, sensed
        if r.get("recorder") == "REC" and t - start >= hz // 4:
            hits["L2"] = hit(t, f"recording slider {sensed}, free on the track")
            break
    spans = run.logs.get("recordings") or {}
    exports = _exports(run)
    records = []
    for eid, r in _by_number(run.evidence, "incline_record"):
        if len(r["private_rows"]) != len(r["public"]["samples"]) or not r["public"]["samples"]:
            continue
        end = (spans.get(eid) or [None, r["public"]["samples"][-1]["physics_tick"]])[1]
        tick = _export_tick(exports, eid, end)
        if tick is not None:
            records.append((tick, end, eid, free_runs(r)))
    records.sort()
    if not states and records:
        free = [(x[0]["physics_tick"], f"{eid}: slider {k} free on the track") for _, _, eid, runs in records
                for k, x in runs if x[-1]["physics_tick"] - x[0]["physics_tick"] >= hz // 4]
        if free:
            hits["L2"] = hit(*min(free), flags=["frame_log_skipped"])
    hits["L3"] = miss("no exported record with a free slide of >= 2 cm" + (
        "" if hits["L2"].reached else " (L2 not reached)"))
    if hits["L2"].reached:
        for tick, end, eid, runs in records:
            if end >= hits["L2"].tick and any(_travel(x) >= .02 for _, x in runs):
                hits["L3"] = hit(tick, f"{eid}: free slide, exported")
                break
    have, lim = set(), INCLINE_LIMITS
    hits["L4"] = miss("exported records do not give each slider a static and a kinetic stretch")
    for tick, end, eid, runs in records:
        for slider, x in runs:
            if static_onset(x, lim["min_static_rise_deg"], lim["min_static_travel_m"]):
                have.add(("static", slider))
            if kinetic_slide(x, lim["min_kinetic_travel_m"], lim["max_kinetic_angle_span_deg"],
                             lim["min_kinetic_angle_deg"]):
                have.add(("kinetic", slider))
        if have >= {(role, k) for role in ("static", "kinetic") for k in SLIDERS}:
            hits["L4"] = hit(_after(hits["L3"], tick), "static and kinetic stretches of both sliders")
            break
    if not hits["L4"].reached:
        hits["L4"].basis += f" (have {sorted(have)})"
    return hits


LADDERS["incline_friction"] = Ladder(detect=incline_friction_levels, note="six-level detectors")


# --- projectile_landing ----------------------------------------------------------
PROJECTILE_CARRY_S, PROJECTILE_CARRY_RAD, PROJECTILE_MOVED_M = 0.5, 0.05, 0.01


def _projectile_carried(states, hz):
    """First tick of a run >= 0.5 s in which the robot holds the ball or the cup and the arm moved >= 0.05 rad."""
    start = None
    for t, row in states:
        joints = row.get("robot_joints")
        if row.get("held") in ("ball", "cup") and joints:
            if start is None:
                start, q0 = t, joints
            moved = max(abs(a - b) for a, b in zip(joints, q0))
            if t - start >= PROJECTILE_CARRY_S * hz and moved >= PROJECTILE_CARRY_RAD:
                return start, row["held"]
        else:
            start = None
    return None, None


def projectile_landing_levels(run):
    """L1  the earliest of: the pawl holds a notch; the ball or the cup held >= 0.5 s while the arm moved >= 0.05 rad;
        the ball moved > 1 cm from where it started; a shot registered (its trigger tick).
    L2  first frame with the pawl at a NON-target notch and the ball seated against the plunger face (backend cradle
        geometry, `_seated`). Frame log not exported: the trigger of the first non-target shot through the gates.
    L3  first shot triggered at or after L2 that is valid from the trigger to the curtain (protocol.shot_valid), at a
        non-target notch, with a landing and both gate edges, and exported after its record closed (trigger + 4 s);
        tick: that export.
    L4  = L3 when no shot was ever fired at the target notch (the spring constant is the only unknown: one notch
        determines it by a model, two by interpolation; the least wins). A target-notch shot makes L4 unreachable
        (flag target_notch_fired; the run fails prediction_before_holdout anyway)."""
    from benchmark.core.projectile.protocol import shot_valid
    states, _ = _log(run)
    hz = run.physics_hz
    target = run.device.get("target_notch_m")

    def target_notch(notch):
        return target is not None and isinstance(notch, (int, float)) and abs(notch - target) < .0005

    records = [(eid, r) for eid, r in _by_number(run.evidence) if isinstance(r, dict)]
    acts = []
    latched = next(((t, f"pawl holds notch {r['latched']}") for t, r in states if r.get("latched") is not None), None)
    if latched:
        acts.append(latched)
    t, held = _projectile_carried(states, hz)
    if t is not None:
        acts.append((t, f"{held} held and carried"))
    if states and states[0][1].get("ball") is not None:
        b0 = states[0][1]["ball"]
        moved = next((t for t, r in states if r.get("ball") is not None
                      and math.dist(r["ball"], b0) > PROJECTILE_MOVED_M), None)
        if moved is not None:
            acts.append((moved, "ball moved > 1 cm"))
    shots = [(r["start_tick"], f"{eid}: shot registered") for eid, r in records if type(r.get("start_tick")) is int]
    if shots:
        acts.append(min(shots))
    hits = {"L1": hit(*min(acts)) if acts else miss("launcher never cocked, ball and cup never handled, no shot")}
    hits["L2"] = miss("never cocked at a non-target notch with the ball seated")
    for t, r in states:
        if r.get("latched") is not None and not target_notch(r["latched"]) and _seated(r):
            hits["L2"] = hit(t, f"notch {r['latched']} held, ball seated")
            break
    if not states:
        gated = [(r["start_tick"], f"{eid}: shot through the gates at notch {_notch(r)}") for eid, r in records
                 if isinstance(r.get("public"), dict) and r["public"].get("gate_events")
                 and type(r.get("start_tick")) is int and not target_notch(_notch(r))]
        if gated:
            hits["L2"] = hit(*min(gated), ["frame_log_skipped"])
            if not hits["L1"].reached:
                hits["L1"] = hit(min(gated)[0], "implied by L2 (frame log not exported)", ["frame_log_skipped"])
    exports = _exports(run)
    readings = []
    for eid, r in records:
        public = r.get("public") if isinstance(r.get("public"), dict) else {}
        events = public.get("gate_events") or []
        if not (shot_valid(r) and isinstance(public.get("landing"), dict) and len(events) == 2
                and isinstance(_notch(r), (int, float)) and not target_notch(_notch(r))
                and type(r.get("start_tick")) is int and type(r.get("end_tick")) is int):
            continue
        if not hits["L2"].reached or r["start_tick"] < hits["L2"].tick:
            continue
        tick = _export_tick(exports, eid, r["end_tick"])
        if tick is not None:
            flags = [] if r.get("valid") is True else ["valid_to_the_curtain"]
            readings.append((tick, f"{eid}: valid shot at notch {_notch(r)}, exported", flags))
    hits["L3"] = (hit(*min(readings)) if readings
                  else miss("no valid non-target shot exported" + ("" if hits["L2"].reached else " (L2 not reached)")))
    early = [eid for eid, r in records if target_notch(_notch(r))]
    if early:
        hits["L4"] = miss(f"shot at the target notch before the submission: {early}", ["target_notch_fired"])
    elif hits["L3"].reached:
        hits["L4"] = hit(hits["L3"].tick, "one valid non-target shot determines the spring", hits["L3"].flags)
    else:
        hits["L4"] = miss("no valid non-target shot exported")
    return hits


LADDERS["projectile_landing"] = Ladder(detect=projectile_landing_levels, note="six-level detectors")


# --- beam_compliance ----------------------------------------------------------------
BEAM_SETTLE_S = 1.2           # ~5 lag constants (0.25 s): the residual is below 1 %
BEAM_MIN_DET = 0.01           # normalized |det| of two difference vectors: excludes only numerically parallel ones
BEAM_LENGTH_M = 0.4           # geometric scale of the coefficient vectors (L^2, L^3/3): makes the det dimensionless


def _beam_config_at(transitions, tick):
    """(configuration, since t_s, hole_mm) standing at `tick` from the bench transitions (before the first one: the
    empty start, since 0 s, hole unknown: None)."""
    before = [x for x in transitions if x["tick"] <= tick]
    if not before:
        return [], 0.0, None
    config, hole = before[-1].get("config") or [], before[-1].get("hole_mm")
    since = before[-1]["t_s"]
    for x in reversed(before):
        if (x.get("config") or []) != config:
            return config, since, hole
        since = x["t_s"]
    return config, (since if config else 0.0), hole


def _beam_loaded(config):
    return any(who != "robot" and not who.endswith("/held") for who, _ in config)


def _beam_clean(config):
    return not any(who == "robot" or who.endswith("/held") for who, _ in config)


def _beam_coefficients(config, forces, x_m, seats_m, masses):
    """c = (sum F a x, sum F g(x, a) / 6) of a load configuration read at x (y = c1 / k_theta + c2 / EI), scaled by the
    strip length (L^2, L^3 / 3) to be dimensionless. F: the recorded force of the entry, else the
    weight's nominal m g; a: the saddle's seat distance or the recorded x:<mm>. None when an entry is unknown."""
    c1 = c2 = 0.0
    for who, where in config:
        F = (forces or {}).get(who)
        if F is None:
            if who not in masses:
                return None
            F = masses[who] * G
        if where.startswith("seat:"):
            a = seats_m.get(where[5:])
        elif where.startswith("x:"):
            a = float(where[2:]) / 1000
        else:
            a = None
        if a is None:
            return None
        g = x_m * x_m * (3 * a - x_m) if x_m <= a else a * a * (3 * x_m - a)
        c1 += F * a * x_m
        c2 += F * g / 6
    return c1 / BEAM_LENGTH_M ** 2, 3 * c2 / BEAM_LENGTH_M ** 3


def beam_compliance_levels(run):
    """L1  the earliest of: a weight held and lifted >= 10 mm; ZERO pressed (device actions); the carriage pushed
        >= 1 mm from where it started (private refresh rows; transitions to another hole as a fallback).
    L2  first bench transition with a weight pressing on the strip (seat:<k> or x:<mm>, not held), no robot on the
        strip and the carriage in a hole, the state standing >= 1.2 s (a weight bouncing off a saddle is no set-up).
    L3  first observation (bench_view) in a valid reading state V with a load: the carriage in a hole (<= 0.3 mm,
        <= 1 mm/s), no robot and no held weight on the strip, the configuration standing >= 1.2 s, the LCD showing.
    L4  the V states (observations, loaded or empty, and the ZERO presses: reading 0.000 there, when the carriage stood
        in a hole under a configuration standing >= 1.2 s) give difference vectors c(s) - c(s') of two states at the
        same hole between the same two ZERO presses, c = (sum F a x, sum F g(x, a) / 6); L4 when two of them span the
        plane (normalized |det| >= 0.01). Nominal masses, seat distances and hole positions only; never EI or k_theta."""
    states, _ = _log(run)
    hz = run.physics_hz
    logs, device = run.logs, run.device
    transitions = logs.get("transitions") or device.get("transitions") or []
    actions = [a for a in (logs.get("actions") or device.get("actions") or []) if a.get("action") == "zero"]
    seats_m = {str(k): float(v) for k, v in (device.get("seats_m") or {}).items()}
    masses = {k: float(v) for k, v in (device.get("masses_kg") or {}).items()}
    holes_mm = [round(float(v) * 1000) for v in (device.get("holes_m") or {}).values()]
    acts = []
    if _has(states, "xyz_mm"):
        t, key = _lifted(states)
        if t is not None:
            acts.append((t, f"{key} held and lifted 10 mm"))
    if actions:
        acts.append((actions[0]["tick"], "ZERO pressed"))
    rows, private = logs.get("rows") or [], logs.get("private") or []
    pushed = None
    if private and "carriage_mm" in private[0]:
        x0 = private[0]["carriage_mm"]
        pushed = next((round(r["t_s"] * hz) for r, p in zip(rows, private) if abs(p["carriage_mm"] - x0) >= 1.0), None)
    elif transitions:
        start = transitions[0].get("hole_mm")
        pushed = next((x["tick"] for x in transitions
                       if x.get("hole_mm") is not None and start is not None and x["hole_mm"] != start), None)
    if pushed is not None:
        acts.append((pushed, "carriage pushed 1 mm"))
    hits = {"L1": hit(*min(acts)) if acts else miss("no weight lifted, no ZERO, carriage never pushed")}
    def setup(x):
        config = x.get("config") or []
        return _beam_loaded(config) and _beam_clean(config) and x.get("hole_mm") is not None

    first, start = None, None                     # the state must stand BEAM_SETTLE_S (a bounce is no set-up)
    for x in transitions + [{"tick": run.end_tick + 1, "config": [], "hole_mm": None}]:
        if setup(x):
            start = start or x
        elif start is not None:
            if x["tick"] - start["tick"] >= BEAM_SETTLE_S * hz:
                first = start
                break
            start = None
    hits["L2"] = (hit(first["tick"], f"{first['config']} on the strip, carriage in hole {first['hole_mm']}") if first
                  else miss("never a weight standing on the strip (1.2 s) with no robot on it and the carriage in a hole"))
    zero_ticks = sorted(a["tick"] for a in actions)
    valid = []                                     # (tick, label, config, forces, hole_mm, segment)
    for eid, r in _by_number(run.evidence, "bench_view"):
        carriage = r.get("carriage") or {}
        config = [list(c) for c in r.get("config") or []]
        settled_since = (_beam_config_at(transitions, r["tick"])[1] if transitions
                         else r.get("config_since_t_s", r["t_s"]))
        if not (abs(carriage.get("offset_mm", 1e9)) <= BEAM_HOLE_TOL_MM
                and abs(carriage.get("v_mm_s", 1e9)) <= BEAM_HOLE_SPEED_MM_S and _beam_clean(config)
                and r["t_s"] - settled_since >= BEAM_SETTLE_S - 1e-9
                and isinstance(r.get("display_mm"), (int, float))):
            continue
        segment = sum(1 for z in zero_ticks if z <= r["tick"])
        valid.append((r["tick"], eid, config, r.get("config_F_N"), carriage.get("nearest_hole_mm"), segment))
    for n, a in enumerate(sorted(actions, key=lambda a: a["tick"])):
        config, since, hole = _beam_config_at(transitions, a["tick"])
        if "hole_mm" in a:                         # the ZERO record says where it was pressed
            hole = a["hole_mm"]
        elif not any(x["tick"] <= a["tick"] for x in transitions):
            near = [h for h in holes_mm if abs(h - a.get("carriage_mm", 1e9)) <= BEAM_HOLE_TOL_MM]
            hole = near[0] if near else None
        if (hole is not None and _beam_clean(config) and a["tick"] / hz - since >= BEAM_SETTLE_S - 1e-9):
            segment = sum(1 for z in zero_ticks if z <= a["tick"])
            valid.append((a["tick"], f"ZERO {n + 1}", a.get("config", config), a.get("config_F_N"), hole, segment))
    valid.sort(key=lambda v: v[0])
    loaded = [(t, label) for t, label, config, *_ in valid if _beam_loaded(config) and not label.startswith("ZERO")]
    hits["L3"] = (hit(loaded[0][0], f"{loaded[0][1]}: loaded, settled, carriage in a hole, LCD read") if loaded
                  else miss("no observation of a settled load with the carriage in a hole"))
    if hits["L3"].reached and hits["L2"].reached:
        hits["L3"].tick = _after(hits["L2"], hits["L3"].tick)
    groups, vectors = {}, []
    hits["L4"] = miss("no two independent same-hole differences")
    for t, label, config, forces, hole, segment in valid:
        c = _beam_coefficients(config, forces, hole / 1000, seats_m, masses) if hole is not None else None
        if c is None:
            continue
        found = None
        for prev in groups.get((hole, segment), []):
            d = (c[0] - prev[0], c[1] - prev[1])
            norm = math.hypot(*d)
            if norm < 1e-9:
                continue
            for v in vectors:
                if abs(v[0] * d[1] - v[1] * d[0]) / (math.hypot(*v) * norm) >= BEAM_MIN_DET:
                    found = (t, label)
                    break
            vectors.append(d)
            if found:
                break
        groups.setdefault((hole, segment), []).append(c)
        if found:
            tick = _after(hits["L3"], t) if hits["L3"].reached else t
            hits["L4"] = hit(tick, f"{label}: two independent same-hole differences")
            break
    return hits


LADDERS["beam_compliance"] = Ladder(detect=beam_compliance_levels, note="six-level detectors")

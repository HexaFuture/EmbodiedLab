"""Circuit board bench (resistor_blackbox): gates and process milestones.

Evidence (sealed by the Bench, see benchmark/core/circuit_grid/registered.py):
    `<episode>:obs:<n>`     kind screen_capture: one get_observation; public {tick, image hashes}, private `state` =
                            the bench at that tick: supply, placements, every instrument's topology flags and true
                            value, `displays` (what every display showed: value in SI units or None for OL, count
                            size, the tick of the conversion shown), the tick of the last change (`last_change_tick`)
    `<episode>:logger:<n>`  kind circuit_logger: a finished logger record (CSV hash-sealed), its start and end tick
                            and `states`: the bench state snapshots in force while it ran

Submission entries: {"evidence_id", "readings": {instrument: value in V or A}, "used_in_fit"} for an observation;
{"evidence_id", "t0_s", "t1_s", "readings": {"CH1" / "CH2": window mean in V}, "used_in_fit"} for a logger record.

Gates:
    readings_genuine        every cited reading is what the instrument showed: within `counts` counts of the display
                            at that observation (the display must show a number), or within `logger_mV` mV of the
                            window mean of the record's samples (the window inside the record, >= 2 samples, all ok)
    measurement_valid       every entry used in the fit was taken in a measuring state: supply ON, at least i_min_A
                            through A-B, every block on the board seated, no change that counts
                            (below) for steady_s before the reading (and every cited display converted after the last
                            such change), every cited instrument seated, and at least one of them across A-B or in
                            series with it
    circuit_bench_restored  tidy-up at submission: supply OFF, every block in a tray seat for its kind, the logger
                            not recording, the robot holding nothing

What a measurement looks at, shared by the circuit-board experiments (circuit_grid_variants.py,
rectifier.py): only the blocks on the board. The hand is not looked at: a reading taken while the hand holds a block
is valid; a block taken off the board is a change that counts. A block counts as on the board
from a bench state in which it is seated on the board until one in which it stands in a tray seat (or is seated on the
board again). Blocks in the tray are not looked at: a block tipped over or lifted in its well, knocked out of it,
picked up from it or put back into it neither makes a reading invalid nor counts as a change; bench_in_order /
circuit_bench_restored judge them. The changes that count: a block seated on the board or leaving it, a lever
throw, the switch, the supply, an ammeter range change, an LED sweep start; their times come from the bench's
complete state log (the process record next to the sealed record, hash-checked) - without it every recorded change
counts, as the sealed observations carry only the time of the last change, not what changed.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from bisect import bisect_right
from pathlib import Path

I_MIN_A = 1.0e-6
STEADY_S = 0.2
SENSITIVITY_MIN = 0.1
DISPLAYED = ("V1", "V2", "A", "CH1")          # instruments with a display (CH1: the logger panel's live reading)
CHANNELS = ("CH1", "CH2")


# --- shared ---------------------------------------------------------------------------------------------------------
def _measurements(ctx):
    entries = ctx.submission.get("measurements") if isinstance(ctx.submission, dict) else None
    return entries if isinstance(entries, list) else []


def logger_samples(record):
    """[(t_s, status, ch1_V, ch2_V)] of a finished record (from its hash-checked CSV)."""
    rows = csv.DictReader(io.StringIO(record["csv"]))
    return [(float(r["t_s"]), r["status"], float(r["ch1_V"]), float(r["ch2_V"])) for r in rows]


# --- what a measurement looks at (module docstring) -------------------------------------------------
def board_history(states):
    """{block: ([tick], [seated on the board])} of the bench states in which each block was seated (on the board or in
    a tray seat), in tick order."""
    out = {}
    for s in sorted((s for s in states if isinstance(s, dict) and isinstance(s.get("placement"), dict)),
                    key=lambda s: int(s["tick"])):
        for block, place in s["placement"].items():
            if place == "loose":
                continue
            ticks, on_board = out.setdefault(block, ([], []))
            ticks.append(int(s["tick"]))
            on_board.append(str(place).startswith("board"))
    return out


def on_board(block, place, history, tick):
    """Whether a block with this placement at `tick` is on the board: seated there, or loose and last seated (at or
    before `tick`) on the board."""
    if place is None:
        return False
    if place != "loose":
        return str(place).startswith("board")
    ticks, seated_on_board = history.get(block, ((), ()))
    k = bisect_right(ticks, int(tick)) - 1
    return k >= 0 and seated_on_board[k]


def unseated_on_board(state, history):
    """The blocks loose in this bench state that are on the board."""
    tick = int(state["tick"])
    return sorted(b for b, p in (state.get("placement") or {}).items()
                  if p == "loose" and on_board(b, p, history, tick))


def evidence_states(records):
    """The bench states of observation and logger records."""
    out = []
    for record in records:
        if record.get("kind") == "screen_capture" and isinstance(record.get("state"), dict):
            out.append(record["state"])
        elif record.get("kind") in ("circuit_logger",) and isinstance(record.get("states"), list):
            out += record["states"]
    return out


def process_state_log(sealed):
    """The bench's complete state log from the process record next to a sealed record (captures/<env>/process,
    hash-checked, same episode), or None."""
    try:
        root = Path(sealed.root).parent / "process"
        manifest = json.loads((root / "manifest.json").read_text())
        data = (root / "bench_logs.json").read_bytes()
        if manifest.get("bench_logs.json") != hashlib.sha256(data).hexdigest():
            return None
        if json.loads((root / "episode.json").read_text()).get("episode_id") != sealed.episode.get("episode_id"):
            return None
        log = json.loads(data).get("state_log")
    except (OSError, ValueError, AttributeError, TypeError):
        return None
    return log if isinstance(log, list) and log else None


class BenchChanges:
    """The bench changes that count for a measurement (module docstring), from the bench's complete state log; with
    no log every recorded change counts. observation and snapshots give the bench states with the time of the last
    change that counts in place of the last recorded change."""

    def __init__(self, log=None, history=None):
        log = sorted((s for s in log or [] if isinstance(s, dict) and "change_tick" in s), key=lambda s: int(s["tick"]))
        self.history = board_history(log) if history is None else history
        self.known, self.counted = set(), []
        prev = None
        for s in log:
            if int(s["change_tick"]) == int(s["tick"]):
                self.known.add(int(s["tick"]))
                if prev is None or self._counts(prev, s):
                    self.counted.append(int(s["tick"]))
            prev = s

    def _counts(self, prev, s):
        before, after = prev.get("placement") or {}, s.get("placement") or {}
        for token in str(s.get("reason") or "").split(","):
            if token == "held":
                blocks = [b for b in {prev.get("held"), s.get("held")} if b]
            elif token in after:
                blocks = [token]
            else:
                return True                        # start, supply, switch, lever, range, sweep, or unknown
            for b in blocks:
                if on_board(b, before.get(b), self.history, prev["tick"]) or \
                        on_board(b, after.get(b), self.history, s["tick"]):
                    return True
        return False

    def last(self, change_tick):
        """The last change that counts at or before a recorded change (the recorded one when the log lacks it)."""
        change_tick = int(change_tick)
        if change_tick not in self.known:
            return change_tick
        k = bisect_right(self.counted, change_tick) - 1
        return self.counted[k] if k >= 0 else change_tick

    def observation(self, state):
        return dict(state, last_change_tick=self.last(state["last_change_tick"]))

    def snapshots(self, states):
        return [dict(s, change_tick=self.last(s["change_tick"])) for s in states]


def sealed_view(ctx):
    """(board history, BenchChanges) for the gates: the process state log and the sealed evidence."""
    log = process_state_log(ctx.sealed) or []
    records = [r for r, problem in map(ctx.evidence, ctx.sealed.evidence) if not problem]
    history = board_history(log + evidence_states(records))
    return history, BenchChanges(log, history)


def run_view(run):
    """(board history, BenchChanges) for the process milestones: the run's state log and evidence."""
    log = (run.logs or {}).get("state_log") or []
    history = board_history(log + evidence_states((run.evidence or {}).values()))
    return history, BenchChanges(log, history)


# --- what an instrument measures (benchmark/core/circuit_grid/roles.py) ------------
# Q: the instruments that measure the object under test: across it, in series with it, across a resistor block of
# printed value in series with it, across it and an ammeter in series with it, or across the driving supply while a
# resistor block of printed value is in series with it (the voltage-difference method). A reading by any instrument
# seated on the board counts while some instrument on the board measures the object (circuit-board design 
# "a reading elsewhere in the circuit counts while the device itself is being measured").
Q_TAGS = ("across", "series", "across_known", "across_ammeter")


class Roles:
    """Roles of bench states: the state's own `roles` (benches), else those of the state-log entry in
    force at its tick (the entry's own or computed by roles.from_state from its netlist), else None (no process
    record: the callers fall back to the bench's topology flags)."""

    def __init__(self, exp_id, log=None, box_elements=None):
        self.exp_id, self.box_elements = exp_id, box_elements
        self.log = sorted((s for s in log or [] if isinstance(s, dict) and "netlist" in s and "tick" in s),
                          key=lambda s: int(s["tick"]))
        self.ticks = [int(s["tick"]) for s in self.log]
        self.cache = {}

    def at(self, state):
        own = state.get("roles") if isinstance(state, dict) else None
        if isinstance(own, dict) and "objects" in own:
            return own
        k = bisect_right(self.ticks, int(state.get("tick", -1))) - 1 if isinstance(state, dict) else -1
        if k < 0:
            return None
        if k not in self.cache:
            entry = self.log[k]
            found = entry.get("roles")
            if not (isinstance(found, dict) and "objects" in found):
                try:
                    from benchmark.core.circuit_grid.roles import from_state
                    found = from_state(entry, self.exp_id, self.box_elements)
                except Exception:                                       # noqa: BLE001
                    found = None
            self.cache[k] = found
        return self.cache[k]


def q_instruments(roles, obj=None):
    """{instrument: tags} of the instruments measuring an object (all objects when obj is None) per Q_TAGS; across the
    supply only while a resistor block of printed value is in series with the object."""
    out = {}
    for name, o in ((roles or {}).get("objects") or {}).items():
        if obj is not None and name != obj:
            continue
        for inst, tags in (o.get("instruments") or {}).items():
            mine = [t for t in tags if t in Q_TAGS or (t == "across_supply" and o.get("known_series"))]
            if mine:
                out.setdefault(inst, [])
                out[inst] += [t for t in mine if t not in out[inst]]
    return out


def ref_arms(roles):
    """resistor_blackbox: the resistor blocks of printed value in a closed circuit with the supply while the box is in
    no closed circuit (the reference arm of a substitution), {block: instruments across it or in series with it}."""
    objects = (roles or {}).get("objects") or {}
    if (objects.get("box") or {}).get("loops"):
        return {}
    out = {}
    for name, o in objects.items():
        if name == "box" or not o.get("supply_loop"):
            continue
        insts = [i for i, tags in (o.get("instruments") or {}).items() if "across" in tags or "series" in tags]
        if insts:
            out[name] = insts
    return out


def rb_state_problems(state, roles, *, board, i_min_a=I_MIN_A):
    """resistor_blackbox measuring state (six-level design): the supply ON, the box in a closed
    circuit with the supply (or, for a substitution, a resistor block of printed value in its place), every block on
    the board seated. Without roles (no process record) the old current floor through A-B stands in for the circuit."""
    problems = []
    supply = state.get("supply")
    on = supply.get("on") if isinstance(supply, dict) else state.get("supply_on")
    if not on:
        problems.append("supply OFF")
    if roles is None:
        if abs(float(state.get("i_ab_a") or 0.0)) < i_min_a:
            problems.append(f"current through A-B {abs(float(state.get('i_ab_a') or 0.0)):.3g} A < {i_min_a} A")
    elif not ((roles["objects"].get("box") or {}).get("supply_loop") or ref_arms(roles)):
        problems.append("neither the box nor a resistor block of printed value in its place is in a closed circuit "
                        "with the supply")
    unseated = unseated_on_board(state, board)
    if unseated:
        problems.append(f"blocks on the board not seated: {unseated}")
    return problems


def rb_instrument_problems(state, names, roles):
    """Cited instruments seated on the board, and an instrument on the board (cited or not) measuring the box (Q) or
    the reference arm in its place (six-level design)."""
    problems = []
    flags = state.get("instruments") or {}
    for name in names:
        if not (flags.get(name) or {}).get("seated"):
            problems.append(f"{name} not seated on the board")
    if roles is None:
        ok = any(f.get("seated") and (f.get("across_ab") or f.get("series_ab")) for f in flags.values()
                 if isinstance(f, dict))
    else:
        ok = bool(q_instruments(roles, "box")) or bool(ref_arms(roles))
    if not ok:
        problems.append("no instrument on the board measures the box (across A-B, in series, across a resistor block "
                        "in series with it, across it and the ammeter, or across the supply)")
    return problems


def rb_observation_problems(state, names, physics_hz, *, board, roles, i_min_a=I_MIN_A, steady_s=STEADY_S):
    problems = rb_state_problems(state, roles, board=board, i_min_a=i_min_a)
    tick, change = int(state["tick"]), int(state["last_change_tick"])
    if tick - change < round(steady_s * physics_hz):
        problems.append(f"last change {(tick - change) / physics_hz:.3f} s before the observation < {steady_s} s")
    for name in names:
        shown = (state.get("displays") or {}).get(name) or {}
        if shown.get("sample_tick") is None or shown["sample_tick"] <= change:
            problems.append(f"{name} display not updated since the last change")
    return problems + rb_instrument_problems(state, names, roles)


def rb_window_problems(states, t0_tick, t1_tick, names, physics_hz, *, board, roles, i_min_a=I_MIN_A,
                       steady_s=STEADY_S):
    found = window_states(states, t0_tick, t1_tick)
    if not found:
        return ["no bench state for the window"]
    problems = []
    if len({s["change_tick"] for s in found}) != 1:
        problems.append("the bench changed during the window")
    change = found[0]["change_tick"]
    if t0_tick - change < round(steady_s * physics_hz):
        problems.append(f"window starts {(t0_tick - change) / physics_hz:.3f} s after the last change < {steady_s} s")
    for s in found:
        r = roles.at(s)
        for p in rb_state_problems(s, r, board=board, i_min_a=i_min_a) + rb_instrument_problems(s, names, r):
            if p not in problems:
                problems.append(p)
    return problems


def state_problems(state, *, board, i_min_a=I_MIN_A):
    """Why a bench state (an observation's `state` or a logger state snapshot) is not a measuring state; board =
    board_history of the episode (only the blocks on the board are looked at, not the hand)."""
    problems = []
    supply = state.get("supply")
    on = supply.get("on") if isinstance(supply, dict) else state.get("supply_on")
    if not on:
        problems.append("supply OFF")
    if abs(float(state.get("i_ab_a") or 0.0)) < i_min_a:
        problems.append(f"current through A-B {abs(float(state.get('i_ab_a') or 0.0)):.3g} A < {i_min_a} A")
    unseated = unseated_on_board(state, board)
    if unseated:
        problems.append(f"blocks on the board not seated: {unseated}")
    return problems


def instrument_problems(state, names):
    """Cited instruments must be seated on the board, at least one across A-B or in series with it."""
    problems = []
    flags = state.get("instruments") or {}
    for name in names:
        if not (flags.get(name) or {}).get("seated"):
            problems.append(f"{name} not seated on the board")
    if not any((flags.get(n) or {}).get("across_ab") or (flags.get(n) or {}).get("series_ab") for n in names):
        problems.append("no cited instrument across A-B or in series with it")
    return problems


def observation_problems(state, names, physics_hz, *, board, i_min_a=I_MIN_A, steady_s=STEADY_S):
    problems = state_problems(state, board=board, i_min_a=i_min_a)
    tick, change = int(state["tick"]), int(state["last_change_tick"])
    if tick - change < round(steady_s * physics_hz):
        problems.append(f"last change {(tick - change) / physics_hz:.3f} s before the observation < {steady_s} s")
    for name in names:
        shown = (state.get("displays") or {}).get(name) or {}
        if shown.get("sample_tick") is None or shown["sample_tick"] <= change:
            problems.append(f"{name} display not updated since the last change")
    return problems + instrument_problems(state, names)


def window_states(states, t0_tick, t1_tick):
    """The state snapshots in force during [t0_tick, t1_tick] (the one before t0 included)."""
    before = [s for s in states if s["tick"] <= t0_tick]
    inside = [s for s in states if t0_tick < s["tick"] <= t1_tick]
    return before[-1:] + inside


def window_problems(states, t0_tick, t1_tick, names, physics_hz, *, board, i_min_a=I_MIN_A, steady_s=STEADY_S):
    found = window_states(states, t0_tick, t1_tick)
    if not found:
        return ["no bench state for the window"]
    problems = []
    if len({s["change_tick"] for s in found}) != 1:
        problems.append("the bench changed during the window")
    change = found[0]["change_tick"]
    if t0_tick - change < round(steady_s * physics_hz):
        problems.append(f"window starts {(t0_tick - change) / physics_hz:.3f} s after the last change < {steady_s} s")
    for s in found:
        for p in state_problems(s, board=board, i_min_a=i_min_a) + instrument_problems(s, names):
            if p not in problems:
                problems.append(p)
    return problems


# --- gates ----------------------------------------------------------------------------------------------------------
def readings_genuine(ctx, counts=2, logger_mV=2.0):
    entries = _measurements(ctx)
    if not any(isinstance(m, dict) and m.get("used_in_fit") is True for m in entries):
        return False, {"reason": "no measurement used in the fit"}
    problems, checked = [], 0
    for k, m in enumerate(entries):
        record, problem = ctx.evidence(m.get("evidence_id"))
        readings = m.get("readings")
        if problem:
            problems.append(f"#{k}: {problem}")
            continue
        if not isinstance(readings, dict) or not readings:
            problems.append(f"#{k}: no readings")
            continue
        if record.get("kind") == "screen_capture" and isinstance(record.get("state"), dict):
            if "t0_s" in m or "t1_s" in m:
                problems.append(f"#{k}: an observation has no time window")
            displays = record["state"].get("displays") or {}
            for name, value in readings.items():
                shown = displays.get(name) if name in DISPLAYED else None
                if shown is None:
                    problems.append(f"#{k}: {name} has no display")
                elif shown.get("value") is None or shown.get("sample_tick") is None:
                    problems.append(f"#{k}: {name} showed no number")
                elif abs(float(value) - shown["value"]) > counts * shown["count_si"] * (1 + 1e-9):
                    problems.append(f"#{k}: {name} = {value} differs from the display by more than {counts} counts")
                checked += 1
        elif record.get("kind") == "circuit_logger":
            t0, t1 = m.get("t0_s"), m.get("t1_s")
            if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (t0, t1)) or not t0 < t1:
                problems.append(f"#{k}: a logger entry needs t0_s < t1_s")
                continue
            if t0 < record["t_start_s"] - 1e-9 or t1 > record["t_end_s"] + 1e-9:
                problems.append(f"#{k}: window outside the record")
                continue
            rows = [r for r in logger_samples(record) if t0 - 1e-9 <= r[0] <= t1 + 1e-9]
            if len(rows) < 2 or any(r[1] != "ok" for r in rows):
                problems.append(f"#{k}: window needs >= 2 samples, all ok")
                continue
            for name, value in readings.items():
                if name not in CHANNELS:
                    problems.append(f"#{k}: {name} is not a logger channel")
                    continue
                mean = sum(r[2 if name == "CH1" else 3] for r in rows) / len(rows)
                if abs(float(value) - mean) > logger_mV / 1000 * (1 + 1e-9):
                    problems.append(f"#{k}: {name} = {value} differs from the window mean {mean:.4f} V by more "
                                    f"than {logger_mV} mV")
                checked += 1
        else:
            problems.append(f"#{k}: {m.get('evidence_id')} is neither an observation nor a logger record")
    return not problems, {"readings_checked": checked, "problems": problems[:20]}


def measurement_valid(ctx, i_min_uA=I_MIN_A * 1e6, steady_s=STEADY_S):
    """Gate measurement_valid (module docstring), relaxed by the six-level design: the measuring
    state is the box (or a resistor block of printed value in its place) in a closed circuit with the supply, and any
    instrument on the board (cited or not) measuring it (Q). i_min_uA applies only without the process record."""
    i_min_A = float(i_min_uA) * 1e-6
    hz = ctx.sealed.episode["physics_hz"]
    used = [(k, m) for k, m in enumerate(_measurements(ctx)) if isinstance(m, dict) and m.get("used_in_fit") is True]
    if not used:
        return False, {"reason": "no measurement used in the fit"}
    board, changes = sealed_view(ctx)
    roles = Roles("resistor_blackbox", process_state_log(ctx.sealed))
    problems = []
    for k, m in used:
        record, problem = ctx.evidence(m.get("evidence_id"))
        if problem:
            problems.append(f"#{k}: {problem}")
            continue
        names = sorted(m.get("readings") or {}) if isinstance(m.get("readings"), dict) else []
        if not names:
            problems.append(f"#{k}: no readings")
            continue
        if record.get("kind") == "screen_capture" and isinstance(record.get("state"), dict):
            state = changes.observation(record["state"])
            found = rb_observation_problems(state, names, hz, board=board, roles=roles.at(state),
                                            i_min_a=i_min_A, steady_s=steady_s)
        elif record.get("kind") == "circuit_logger" and isinstance(record.get("states"), list):
            try:
                t0, t1 = round(float(m["t0_s"]) * hz), round(float(m["t1_s"]) * hz)
            except (KeyError, TypeError, ValueError):
                problems.append(f"#{k}: a logger entry needs t0_s and t1_s")
                continue
            found = rb_window_problems(changes.snapshots(record["states"]), t0, t1, names, hz, board=board,
                                       roles=roles, i_min_a=i_min_A, steady_s=steady_s)
        else:
            found = ["not an observation or a logger record"]
        problems += [f"#{k}: {p}" for p in found]
    return not problems, {"entries_used": len(used), "problems": problems[:20]}


def circuit_bench_restored(ctx, max_capacitor_v=0.1):
    state = ctx.sealed.episode.get("terminal_state") or {}
    placement = state.get("placement") or {}
    away = sorted(b for b, p in placement.items() if not (isinstance(p, list) and p[:1] == ["tray"] and p[2:3] == [True]))
    problems = []
    if state.get("supply_on") is not False:
        problems.append("supply not OFF")
    if not placement or away or state.get("all_in_tray") is not True:
        problems.append(f"blocks not in a tray seat for their kind: {away}")
    if state.get("logger_recording") is not False:
        problems.append("logger still recording")
    if state.get("held"):
        problems.append(f"robot holding {state['held']}")
    if state.get("box_capacitor_v") is not None and abs(float(state["box_capacitor_v"])) > float(max_capacitor_v):
        problems.append(f"box capacitor at {float(state['box_capacitor_v']):.3f} V > {max_capacitor_v} V")
    return not problems, {"problems": problems}


# --- process milestones (benchmark/evaluation/process_rules/circuits.py) -------------------------------------------
def _seated_valid_instruments(state, physics_hz):
    """Instruments at an observation whose reading would be a valid measurement on its own or with others."""
    out = []
    change = int(state["last_change_tick"])
    for name in DISPLAYED:
        shown = (state.get("displays") or {}).get(name) or {}
        flags = (state.get("instruments") or {}).get(name) or {}
        if flags.get("seated") and shown.get("value") is not None and (shown.get("sample_tick") or -1) > change:
            out.append(name)
    return out


def _record_windows(record_rows, states, physics_hz):
    """(state, ok sample ticks) per bench state of a record that lasts >= STEADY_S before its samples."""
    out = []
    ticks = [round(t * physics_hz) for t, status, _, _ in record_rows if status == "ok"]
    if not ticks:
        return out
    by_change = {}
    for s in window_states(states, ticks[0], ticks[-1]):
        by_change.setdefault(s["change_tick"], s)
    for change, s in by_change.items():
        later = [x["tick"] for x in states if x["change_tick"] != change and x["tick"] > change]
        end = min(later) if later else math.inf
        inside = [t for t in ticks if change + round(STEADY_S * physics_hz) <= t < end]
        out.append((s, inside))
    return out


def process_hits(run):
    from benchmark.evaluation.process import hit, miss
    hz = run.physics_hz
    logs = run.logs
    actions = logs.get("actions") or []
    states = logs.get("state_log") or []
    board, changes = run_view(run)
    timed = changes.snapshots(states)
    hits = {}
    first = next((a for a in actions if a.get("event") == "placement" and str(a.get("to", "")).startswith("board:")),
                 None)
    hits["E1"] = (hit(first["tick"], f"{first['block']} seated at {first['to']}") if first
                  else miss("no block seated on the board"))
    e2 = next((s for s in states if s["supply"]["on"] and abs(s["i_ab_a"]) >= I_MIN_A and any(
        f.get("seated") and (f.get("across_ab") or f.get("series_ab")) for f in s["instruments"].values())), None)
    hits["E2"] = (hit(e2["tick"], "supply ON, current through A-B, an instrument across A-B or in series")
                  if e2 else miss("never a closed circuit with current through A-B and an instrument connected"))
    m1, m2 = [], []
    for eid, record in (run.evidence or {}).items():
        if record.get("kind") == "screen_capture" and isinstance(record.get("state"), dict):
            state = changes.observation(record["state"])
            names = _seated_valid_instruments(state, hz)
            good = [n for n in names if not observation_problems(state, [n], hz, board=board)]
            if not good:
                continue
            m1.append((state["tick"], f"{eid}: {good}"))
            if len(names) >= 2 and not observation_problems(state, names, hz, board=board) and \
                    state.get("r_sensitivity", 0) >= SENSITIVITY_MIN:
                m2.append((state["tick"], f"{eid}: {names}, sensitivity to R {state['r_sensitivity']:.3f}"))
    records = [(eid, r, logger_samples(r), r["end_tick"], []) for eid, r in (run.evidence or {}).items()
               if r.get("kind") == "circuit_logger"]
    open_record = logs.get("logger_open")
    if isinstance(open_record, dict) and open_record.get("rows"):
        rows = [(r["t_s"], r["status"], r["ch1_V"], r["ch2_V"]) for r in open_record["rows"]]
        records.append((open_record["evidence_id"], open_record, rows, open_record["last_tick"], ["open_record"]))
    for eid, record, rows, end, flags in records:
        for state, inside in _record_windows(rows, timed, hz):
            if len(inside) < 3:
                continue
            live = [c for c in CHANNELS if (state["instruments"].get(c) or {}).get("seated")]
            if not live or state_problems(state, board=board) or instrument_problems(state, live):
                continue
            m1.append((end, f"{eid}: {len(inside)} samples with {live}", flags))
            if len(live) == 2 and state.get("r_sensitivity", 0) >= SENSITIVITY_MIN:
                m2.append((end, f"{eid}: both channels, sensitivity to R {state['r_sensitivity']:.3f}", flags))
    best = min(m1, key=lambda x: x[0]) if m1 else None
    hits["M1"] = hit(best[0], best[1], best[2] if len(best) > 2 else ()) if best else miss("no valid reading")
    best = min(m2, key=lambda x: x[0]) if m2 else None
    hits["M2"] = (hit(best[0], best[1], best[2] if len(best) > 2 else ()) if best
                  else miss("no valid reading set that determines R at one state"))
    return hits

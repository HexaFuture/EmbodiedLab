"""Process rules of the circuit kit: rc_capacitance, led_planck, circuit_blackbox3, battery_emf
(one panel of three slider levers with detents, a two-channel logger) and rectifier_supply (CAP and MODE levers, a
servo-driven electronic load, a scope). There is nothing to grasp.

Sources in the process record:
    frames        private/physics.jsonl: every 8th tick (30 Hz) whether the hand touches each lever
    selector log  bench_logs.json `actions` (every detent change) and `state_log` (the switch state at every change;
                  on the panel also the capacitor voltage); the rectifier logs each change of CAP, LOAD or MODE
    records       evidence.json: completed logger acquisitions (CSV, start and end tick) or scope records, and
                  bench_logs.json `logger_open` / `scope_open`: the one still recording at stop (samples so far)

E1  the first lever pushed out of its detent with the hand on it (a contact frame of that lever within 2 frames of
    the tick it left the detent) that comes to rest in a different detent. A lever that springs back into its own
    detent or stays between detents, a lever moved without contact, and the rectifier's LOAD (moved by the
    set_load_resistance servo) do not count.
E2  the first tick of a measurable selector state (per experiment), after the hold where one is stated.
M1  the first completed record with >= 3 in-range samples taken in a measurable state (after its hold); the tick is
    the record's end. The rectifier: >= 5 ticks (one source period) with MODE ON and CAP in a detent, not over range.
M2  per experiment; the tick is the end of the record that completes the set.
An open record (recording at stop) is used like a completed one up to its last sample: its samples are real
readings, so it counts for M1, and for M2 only what it had already recorded (a run cut off at stop counts only if its
recorded part meets the length checks). Such a hit is timed at the last recorded tick and flagged `open_record`.
A process record written before these fields existed simply has no open record.
A state measurable at tick 0 reaches E1 (tick 0) and E2 with the flag `start_state` (rc_capacitance starts
in A-B CHARGE in about 1/9 of the instances). No circuit experiment has an irreversible
protocol violation.

resistor_blackbox runs on the circuit board (benchmark/core/circuit_grid: blocks placed on a grid board, meters read
from the camera images, a logger with a keypad). Its milestones (benchmark/evaluation/circuit_grid.process_hits),
from bench_logs.json `actions` (every seat and board event) and `state_log` (the bench at every change: supply,
placements, current through A-B, each instrument seated / across A-B / in series), the observation captures and
logger records in evidence.json and `logger_open`:
E1  the first block seated on the board (both pins in the sockets of one edge, released).
E2  the first state with the supply ON, >= 1 uA through A-B and an instrument seated across A-B or in series.
M1  the first valid reading: an observation in a measuring state (gate measurement_valid) where an instrument
    across A-B or in series shows a number converted after the last change; or a logger record with >= 3 ok
    samples of a seated channel in such a state, >= 0.2 s after the change (timed at the record end).
M2  the first observation or record in a measuring state where >= 2 instruments read the circuit and their ratio
    responds to R (|d ln(y_i / y_j) / d ln R| >= 0.1, from the bench solution): the readings at that one state
    determine R.
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right

from benchmark.evaluation.gates import _line_fit, logger_rows
from benchmark.evaluation.process import hit, miss

CONTACT_FRAMES = 2              # a contact frame within this many 30 Hz frames of the tick the lever left its detent
MIN_SAMPLES = 3                 # in-range samples that make a reading
SOURCE_V, LOGGER_OHM = 5.0, 1.0e7
DRIVEN = ("CHARGE", "DISCHARGE")


def _label(value):
    return "transition" if value is None else str(value)


def _state(switch, names):
    return ", ".join(f"{n} {_label(v)}" for n, v in zip(names, switch))


def _at(record, basis):
    """A milestone met by `record`, at its end (an open record: its last recorded tick)."""
    return hit(record["end_tick"], basis, ["open_record"] if record.get("status") == "open" else [])


def _open(run, name):
    """The record still recording at stop (bench_logs.json `name`), timed at its last recorded tick; None if absent."""
    record = run.logs.get(name)
    if not isinstance(record, dict) or not (record.get("rows") or record.get("samples")):
        return None
    return dict(record, end_tick=record["last_tick"])


def _bench_log(run, name, private):
    """A bench list from bench_logs.json, else the copy the backend wrote at close."""
    if name in run.logs:
        return run.logs[name]
    if private.endswith(".jsonl"):
        return list(run.jsonl(private))
    return run.private_json(private) or []


# --- E1: a lever pushed by the hand into another detent ---------------------------------------------------------------
def _contacts(run):
    """{lever: sorted ticks of episode frames with the hand touching it} (frames before the bench existed skipped)."""
    step, ticks = run.physics_hz // 30, {}
    for row in run.frames:
        if "frame" not in row or row.get("switch") is None:
            continue
        for name, touching in (row.get("robot_contacts") or {}).items():
            if touching:
                ticks.setdefault(name, []).append(row["frame"] * step)
    return ticks


def _pushes(moves):
    """(tick, lever, from, to) of each lever that left a detent and came to rest in a different detent."""
    out = []
    for i, (tick, name, old, new) in enumerate(moves):
        if old == "transition" or new == old:
            continue
        landing = new if new != "transition" else next(
            (n for _, s, _, n in moves[i + 1:] if s == name and n != "transition"), None)
        if landing is not None and landing != old:
            out.append((tick, name, old, landing))
    return out


def _e1(run, moves):
    """The first push (see _pushes) with a contact frame of that lever within CONTACT_FRAMES of leaving the detent."""
    if not run.frames:
        return miss("no frame log (private/physics.jsonl)")
    contacts, window = _contacts(run), CONTACT_FRAMES * (run.physics_hz // 30)
    pushes = _pushes(moves)
    for tick, name, old, landing in pushes:
        touched = contacts.get(name, [])
        i = bisect_left(touched, tick - window)
        if i < len(touched) and touched[i] <= tick + window:
            return hit(tick, f"{name} pushed by the hand from {old} to {landing}")
    return miss(f"{len(pushes)} lever change(s), none with the hand on the lever" if pushes
                else "no lever moved to another detent")


# --- logger panels (rc_capacitance, led_planck, circuit_blackbox3, battery_emf) -------------------
class _Panel:
    """Selector intervals [(start, end, switch, extra)] from the state log, the measurable ones (held >= hold ticks),
    and each record's runs of consecutive in-range samples within one interval [(interval, rows)]."""

    def __init__(self, run, measurable, hold_s, names):
        self.run, self.names = run, names
        self.hold = round(hold_s * run.physics_hz)
        log = _bench_log(run, "state_log", "circuit_state.jsonl")
        self.intervals = [(e["tick"], log[i + 1]["tick"] if i + 1 < len(log) else run.end_tick, tuple(e["switch"]),
                           e.get("vc_V")) for i, e in enumerate(log)]
        self.ok = [measurable(sw, x) and end - start >= self.hold for start, end, sw, x in self.intervals]
        starts = [i[0] for i in self.intervals]
        records = [r for r in run.evidence.values()
                   if r.get("kind") == "circuit_logger" and r.get("status") == "complete"]
        self.records = []
        records = sorted(records, key=lambda r: r["end_tick"]) + [r for r in (_open(run, "logger_open"),) if r]
        for record in records:
            runs, key = [], None
            for row in _logger_rows(record):
                row["tick"] = round(row["t_s"] * run.physics_hz)
                k = bisect_right(starts, row["tick"]) - 1 if row["status"] == "ok" else None
                if k is not None and k != key:
                    runs.append((k, []))
                if k is not None:
                    runs[-1][1].append(row)
                key = k
            self.records.append((record, runs))

    def moves(self):
        return [(a["tick"], a["selector"], _label(a["old"]), _label(a["new"]))
                for a in _bench_log(self.run, "actions", "circuit_actions.json")]

    def measured(self, k, rows):
        """The rows of a run taken in a measurable state after its hold."""
        return [r for r in rows if self.ok[k] and r["tick"] >= self.intervals[k][0] + self.hold]

    def e2(self):
        for k, (start, end, switch, _) in enumerate(self.intervals):
            if self.ok[k]:
                return hit(start + self.hold, f"{_state(switch, self.names)} from tick {start}",
                           ["start_state"] if start == 0 else [])
        return miss("never in a measurable state" + (f" held {self.hold} ticks" if self.hold else ""))

    def m1(self):
        for record, runs in self.records:
            n = sum(len(self.measured(k, rows)) for k, rows in runs)
            if n >= MIN_SAMPLES:
                return _at(record, f"{record['evidence_id']}: {n} in-range samples in the measurable state")
        return miss(f"{len(self.records)} record(s), none with {MIN_SAMPLES} samples in the measurable state")


PANEL = ("PAIR", "RS", "MODE")


def _logger_rows(record):
    """Samples of a completed record from its hash-checked CSV; of an open record from its rows (labels as text)."""
    if "csv" in record:
        return logger_rows(record)
    return [{"t_s": float(r["t_s"]), "status": r["status"], "ch1_V": float(r["ch1_V"]), "ch2_V": float(r["ch2_V"]),
             "pair": str(r["pair"]), "rs": str(r["rs"]), "mode": str(r["mode"])} for r in record["rows"]]


def _logger_hits(run, measurable, hold_s, protocol, names=PANEL):
    panel = _Panel(run, measurable, hold_s, names)
    e2 = panel.e2()
    if e2.reached and "start_state" in e2.flags:
        e1 = hit(0, "measurable at the start, no push needed", ["start_state"])
    else:
        e1 = _e1(run, panel.moves())
    return {"E1": e1, "E2": e2, "M1": panel.m1(), "M2": protocol(run, panel)}


def _duration(rows):
    return rows[-1]["t_s"] - rows[0]["t_s"]


def rc_capacitance(run):
    """E2: A-B, R_s in a detent, CHARGE or DISCHARGE, the capacitor voltage at entry >= 0.5 V from that mode's final
    value (5 V R_in/(R_in+R_s) or 0), i.e. a step. M2: one run of one record in such a state that starts within 2 tau
    of the step, lasts >= 2 tau and has >= 10 samples per tau (tau: the device's A-B time constant of that R_s)."""
    def measurable(sw, vc):
        pair, rs, mode = sw
        if pair != "A-B" or rs is None or mode not in DRIVEN or vc is None:
            return False
        final = SOURCE_V * LOGGER_OHM / (LOGGER_OHM + float(rs)) if mode == "CHARGE" else 0.0
        return abs(vc - final) >= 0.5

    def protocol(run, panel):
        taus = run.device.get("tau_ab_charge_s") or {}
        for record, runs in panel.records:
            for k, rows in runs:
                rows = panel.measured(k, rows)
                tau = taus.get(_label(panel.intervals[k][2][1]))
                if len(rows) < 2 or not tau:
                    continue
                lag = (rows[0]["tick"] - panel.intervals[k][0]) / run.physics_hz
                if lag <= 2 * tau and _duration(rows) >= 2 * tau and record["sample_hz"] * tau >= 10:
                    return _at(record, f"{record['evidence_id']}: {_duration(rows) / tau:.1f} tau of the "
                               f"{panel.intervals[k][2][2]} step from {lag / tau:.2f} tau, "
                               f"{record['sample_hz'] * tau:.0f} samples per tau")
        return miss("no step window of >= 2 tau at >= 10 samples per tau starting within 2 tau of the step")

    return _logger_hits(run, measurable, 0.0, protocol)


def resistor_blackbox(run):
    """The circuit board: milestones from benchmark/evaluation/circuit_grid.process_hits (module docstring)."""
    from benchmark.evaluation.circuit_grid import process_hits
    return process_hits(run)


def led_planck(run):
    """E2: SWEEP with an LED selected and R_s 100 or 330. M2: sweeps of >= 3 LEDs of printed wavelength and X; a sweep
    is one run of one record in SWEEP with that LED whose current CH2/R_s (nominal) rises from <= 0.1 mA (below
    turn-on) to >= 3 mA."""
    def measurable(sw, _):
        return sw[2] == "SWEEP" and sw[0] is not None and sw[1] in (100, 330)

    def protocol(run, panel):
        known = set(run.device.get("known_nm") or ("L1", "L2", "L3", "L4"))
        swept = set()
        for record, runs in panel.records:
            for k, rows in runs:
                led, rs, mode = panel.intervals[k][2]
                if mode != "SWEEP" or led is None or rs is None:
                    continue
                currents = [r["ch2_V"] / float(rs) for r in rows]
                if min(currents) <= 1e-4 and max(currents) >= 3e-3:
                    swept.add(led)
            if "X" in swept and len(swept & known) >= 3:
                return _at(record, f"full sweeps of {sorted(swept)}")
        return miss(f"full sweeps of {sorted(swept)}; need X and 3 of {sorted(known)}")

    return _logger_hits(run, measurable, 0.0, protocol, ("LED", "RS", "SUPPLY"))


def circuit_blackbox3(run):
    """E2: a terminal pair and R_s in detents, CHARGE or DISCHARGE, held 0.2 s. M2: each of A-B, B-C, A-C has a run of
    one record in such a driven state with >= 30 samples over >= 5 s (the network_evidence segment, for all pairs)."""
    def measurable(sw, _):
        return sw[0] is not None and sw[1] is not None and sw[2] in DRIVEN

    def protocol(run, panel):
        pairs = set()
        for record, runs in panel.records:
            for k, rows in runs:
                pair, rs, mode = panel.intervals[k][2]
                if measurable((pair, rs, mode), None) and len(rows) >= 30 and _duration(rows) >= 5.0 - 1e-6:
                    pairs.add(pair)
            if pairs >= {"A-B", "B-C", "A-C"}:
                return _at(record, "driven segments >= 5 s on A-B, B-C and A-C")
        return miss(f"driven segments >= 5 s on {sorted(pairs)}; need all three pairs")

    return _logger_hits(run, measurable, 0.2, protocol)


def battery_emf(run):
    """E2: switch ON with the load and the shunt in detents. M2: >= 3 different loads, each with a run of >= 3
    in-range samples in ON, all at one shunt."""
    def measurable(sw, _):
        return sw[2] == "ON" and sw[0] is not None and sw[1] is not None

    def protocol(run, panel):
        loads = {}
        for record, runs in panel.records:
            for k, rows in runs:
                load, shunt, mode = panel.intervals[k][2]
                if measurable((load, shunt, mode), None) and len(rows) >= MIN_SAMPLES:
                    loads.setdefault(_label(shunt), set()).add(_label(load))
            best = max(loads.items(), key=lambda kv: len(kv[1]), default=(None, set()))
            if len(best[1]) >= 3:
                return _at(record, f"loads {sorted(best[1])} ohm at shunt {best[0]} ohm")
        return miss(f"loads recorded per shunt: { {k: sorted(v) for k, v in loads.items()} }; need 3 at one shunt")

    return _logger_hits(run, measurable, 0.0, protocol, ("LOAD", "SHUNT", "SWITCH"))


# --- rectifier_supply ----------------------------------------------------------------------------------------------
MODULES = ("C100", "C220", "C470")
MEAN_MIN_V, RIPPLE_MAX_V = 8.5, 0.8
PERIOD_SAMPLES = 240            # one 50 Hz source period at 12 kHz
LOAD_STEADY = 0.005             # load within 0.5 % across a steady segment
LOAD_FINAL_TOL = 0.02           # the device's final-configuration tolerance
SETTLED = 0.01                  # first and last analysed period means within 1 %
OUTPUT = 2                      # scope column output_voltage_V


def _segments(record):
    """(cap, load ohm, samples) of each run of ticks with MODE ON, CAP in a detent and a steady load."""
    controls, samples = record["controls"], record["samples"]
    per_tick = len(samples) // len(controls) if controls else 0
    out, current = [], None
    for i, c in enumerate(controls):
        on = c["mode"] == "ON" and c["cap"] is not None
        if on and current and current[0] == c["cap"] and abs(c["load_ohm"] / current[1] - 1) <= LOAD_STEADY:
            current[3] = i
            continue
        current = [c["cap"], c["load_ohm"], i, i] if on else None
        if current:
            out.append(current)
    return [(cap, load, samples[a * per_tick:(b + 1) * per_tick]) for cap, load, a, b in out]


def _steady(samples):
    """(mean V, ripple V) over the last whole source periods (2-10) when the first and last of them agree in mean."""
    n = min(10, len(samples) // PERIOD_SAMPLES)
    if n < 2:
        return None
    v = [s[OUTPUT] for s in samples[-n * PERIOD_SAMPLES:]]
    first, last = sum(v[:PERIOD_SAMPLES]) / PERIOD_SAMPLES, sum(v[-PERIOD_SAMPLES:]) / PERIOD_SAMPLES
    if abs(last - first) > SETTLED * max(abs(last), 1e-6):
        return None
    return sum(v) / len(v), max(v) - min(v)


def rectifier_supply(run):
    """E1: CAP or MODE pushed by the hand into another detent (LOAD is moved by the set_load_resistance servo and
    never counts). E2: MODE ON with CAP in a detent. M1: the first completed scope record, not over range, with >= 5
    ticks in that state. M2: for each module a steady record segment that meets the supply requirement (mean >= 8.5 V,
    ripple <= 0.8 V) and one that does not (the two sides of its load limit), and a steady segment at the task load
    (within 2 %) that meets it (the final configuration). Steady: MODE ON, one CAP, the load within 0.5 %, >= 2 whole
    source periods, the first and last analysed periods' means within 1 %; measured over the last <= 10 periods."""
    actions = _bench_log(run, "actions", "circuit_actions.json")
    initial = actions[0]["old"] if actions else (run.terminal_state.get("switch") or [None, None, None])
    changes = [(0, tuple(initial))] + [(a["tick"], tuple(a["new"])) for a in actions]
    names = ("CAP", "LOAD", "MODE")

    moves = [(a["tick"], name, _label(a["old"][i]), _label(a["new"][i]))
             for a in actions for i, name in ((0, "CAP"), (2, "MODE")) if a["old"][i] != a["new"][i]]
    e2 = miss("MODE never ON with CAP in a detent")
    for start, switch in changes:
        if switch[2] == "ON" and switch[0] is not None:
            e2 = hit(start, f"MODE ON, CAP {switch[0]}", ["start_state"] if start == 0 else [])
            break
    e1 = hit(0, "measurable at the start", ["start_state"]) if e2.reached and e2.tick == 0 else _e1(run, moves)

    records = sorted((v["public"] for v in run.evidence.values() if v.get("kind") == "rectifier_scope"
                      and v["public"].get("end_tick") is not None),
                     key=lambda r: r["end_tick"]) + [r for r in (_open(run, "scope_open"),) if r]
    m1 = miss(f"{len(records)} scope record(s), none in range with 5 ticks in MODE ON")
    for record in records:
        on = sum(c["mode"] == "ON" and c["cap"] is not None for c in record["controls"])
        if not record["overrange"] and on >= 5:
            m1 = _at(record, f"{record['evidence_id']}: {on} ticks with MODE ON")
            break

    final = run.device.get("final_load_ohm")
    sides, final_ok = {m: set() for m in MODULES}, None
    m2 = None
    for record in records:
        if record["overrange"]:
            continue
        for cap, load, samples in _segments(record):
            measured = _steady(samples)
            if measured is None:
                continue
            meets = measured[0] >= MEAN_MIN_V and measured[1] <= RIPPLE_MAX_V
            sides[cap].add(meets)
            if final and meets and abs(load / final - 1) <= LOAD_FINAL_TOL:
                final_ok = final_ok or (cap, load)
        if all(len(s) == 2 for s in sides.values()) and final_ok:
            m2 = _at(record, f"each module steady on both sides of its limit; final {final_ok[0]} at "
                     f"{final_ok[1]:.0f} ohm meets the requirement")
            break
    if m2 is None:
        bracketed = [m for m in MODULES if len(sides[m]) == 2]
        m2 = miss(f"modules bracketed: {bracketed}; final configuration at {final} ohm "
                  f"{'recorded' if final_ok else 'not recorded meeting the requirement'}")
    return {"E1": e1, "E2": e2, "M1": m1, "M2": m2}


RULES = {"rc_capacitance": rc_capacitance, "resistor_blackbox": resistor_blackbox, "led_planck": led_planck,
         "circuit_blackbox3": circuit_blackbox3, "battery_emf": battery_emf, "rectifier_supply": rectifier_supply}

# Experiments moved onto the circuit board (battery_emf, ...): their milestones replace the panel rules above
# (benchmark/evaluation/circuit_grid_variants.py).
from benchmark.evaluation.circuit_grid_variants import PROCESS_RULES as _GRID_RULES  # noqa: E402
RULES.update(_GRID_RULES)


# --- six levels (benchmark/evaluation/process.py) ---------------------------------------------------
# Provisional: each experiment's legacy E1-M2 rule above mapped E1->L1, E2->L2, M1->L3, M2->L4 (process.from_legacy).
# An entry is replaced by Ladder(detect=..., tidy_gates=..., note=...) when that experiment's own L1-L4 detectors
# are written.
from benchmark.evaluation.process import Ladder, from_legacy  # noqa: E402,F401

LADDERS = {exp_id: from_legacy(rule) for exp_id, rule in RULES.items()}


# --- six levels of the circuit-board experiments ---------------------------------
# Shared by resistor_blackbox, rc_capacitance, led_planck, circuit_blackbox3, battery_emf
# and rectifier_supply. Sources: bench_logs.json `actions` and `state_log` (with each entry's `roles`, or the
# same computed from its netlist: benchmark/core/circuit_grid/roles.py), the observation and logger records in
# evidence.json, `logger_open`, episode.json `device` (box_elements, rectifier final load). Never the truth or the
# submission. Only changes on the board count (circuit_grid.BenchChanges); the hand is not looked at.
#
# L1  the earliest of: a block seated on the board after the hand held it; a block grasped that, once let go, does not
#     rest in its starting tray seat (timed at the grasp); the supply, the switch, the changeover lever (after the
#     start), a logger key with an effect, the LED SWEEP key with an effect. A block knocked without a grasp, moving
#     in the air, the ammeter's own auto-range do not count.
# L2  per experiment (the measured object in a state some reasonable method measures from, with an instrument on it).
# L3  the first valid reading in an L2 state: an observation where an instrument measuring the object (Q,
#     circuit_grid.Q_TAGS) shows a number converted after the last change on the board, in a measuring state, the
#     experiment's settling time after that change; or a logger record (finished, or still recording at stop) with
#     >= 2 ok samples of a seated channel measuring the object in such a state (timed at the record's end).
# L4  per experiment (enough data in principle for every target by some reasonable method).
from benchmark.evaluation import circuit_grid as _CG  # noqa: E402
from benchmark.evaluation import circuit_grid_variants as _CV  # noqa: E402

GRID_STEADY_S = {"resistor_blackbox": 0.2, "battery_emf": 0.2, "led_planck": 0.2, "rc_capacitance": 0.0,
                 "circuit_blackbox3": 0.0}       # the gates' steady_s (experiment.yaml)
STEP_MIN_V, STEP_CHANGE_V = 0.5, 0.010           # a step starts >= 0.5 V from its final value; readings span 10 mV
LED_PAIR_RATIO, LED_MATCH = 2.0, 0.10            # led_planck L4 (alpha) current ratio, (beta) matched currents
LOAD_STEP = 0.05                                 # battery_emf L4: cell currents >= 5 % apart


def grid_l1(run):
    """L1 of the circuit-board experiments (see above)."""
    actions = (run.logs or {}).get("actions") or []
    found = []
    grasps = {}
    for k, a in enumerate(actions):
        event, tick = a.get("event"), a.get("tick", 0)
        if event == "held" and a.get("to"):
            block = a["to"]
            grasps.setdefault(block, []).append(tick)
            rest = None
            for b in actions[k + 1:]:
                if b.get("event") == "held" and b.get("to") == block:
                    break                                         # grasped again: where it rested before counts
                if b.get("event") == "placement" and b.get("block") == block:
                    rest = b.get("to")
            if rest is None or rest != f"tray:{block}":
                found.append((tick, f"{block} grasped and moved (rests at {rest or 'loose'})"))
        elif event == "placement" and str(a.get("to", "")).startswith("board:") and \
                any(t <= tick for t in grasps.get(a.get("block"), [])):
            found.append((tick, f"{a['block']} seated at {a['to']}"))
        elif event in ("supply", "switch"):
            found.append((tick, f"{event} {'ON' if a.get('on') or a.get('closed') else 'OFF'}"))
        elif event == "changeover" and tick > 0:
            found.append((tick, f"{a.get('block')} lever to {a.get('to')}"))
        elif event == "logger_key" and not str(a.get("effect", "")).startswith(("ignored", "error")):
            found.append((tick, f"logger key {a.get('key')} ({a.get('effect')})"))
        elif event == "sweep_key" and a.get("effect") not in (None, "output_off"):
            found.append((tick, f"SWEEP key ({a.get('effect')})"))
    if not found:
        return miss("no block grasped and moved, no key or control used")
    tick, basis = min(found, key=lambda x: x[0])
    return hit(tick, basis)


class _Reading:
    """One valid reading: an observation, or the samples of one logger record in one state-log entry."""

    def __init__(self, tick, eid, kind, state, roles, names, q, samples=(), flags=()):
        self.tick, self.eid, self.kind, self.state, self.roles = tick, eid, kind, state, roles
        self.names, self.q, self.samples, self.flags = list(names), dict(q), list(samples), list(flags)

    @property
    def times(self):
        return self.samples if self.kind == "record" else [self.state["tick"]]


def _box_elements(run):
    return (run.device or {}).get("box_elements")


def _grid_check(exp, state, roles, board):
    """(problems, {instrument: tags} measuring the object) of a bench state (the gates' measuring state)."""
    if exp == "resistor_blackbox":
        problems = _CG.rb_state_problems(state, roles, board=board) + _CG.rb_instrument_problems(state, [], roles)
        q = _CG.q_instruments(roles, "box") if roles is not None and (roles["objects"].get("box") or {}).get(
            "supply_loop") else {}
        if roles is not None:
            for _, insts in _CG.ref_arms(roles).items():
                for i in insts:
                    q.setdefault(i, ["reference"])
        elif not problems:
            q = {k: ["on_dut"] for k, f in (state.get("instruments") or {}).items()
                 if isinstance(f, dict) and f.get("seated") and (f.get("across_ab") or f.get("series_ab"))}
        return problems, q
    return _CV.measuring_problems(exp, state, roles, [], board=board), _CV.measured_by(exp, state, roles)


def grid_readings(run, exp):
    """Every valid reading of a run (see L3 above), in tick order."""
    hz = run.physics_hz
    steady = round(GRID_STEADY_S.get(exp, 0.2) * hz)
    log = (run.logs or {}).get("state_log") or []
    board, changes = _CG.run_view(run)
    roles = _CG.Roles(exp, log, _box_elements(run))
    out = []
    for eid, record in (run.evidence or {}).items():
        if record.get("kind") != "screen_capture" or not isinstance(record.get("state"), dict):
            continue
        state = changes.observation(record["state"])
        change = int(state["last_change_tick"])
        if int(state["tick"]) - change < steady:
            continue
        r = roles.at(state)
        problems, q = _grid_check(exp, state, r, board)
        if problems:
            continue
        names = [n for n in _CG.DISPLAYED
                 if ((state.get("instruments") or {}).get(n) or {}).get("seated")
                 and ((state.get("displays") or {}).get(n) or {}).get("value") is not None
                 and (((state.get("displays") or {}).get(n) or {}).get("sample_tick") or -1) > change]
        mine = {n: t for n, t in q.items() if n in names}
        if mine:
            out.append(_Reading(int(state["tick"]), eid, "obs", state, r, names, mine))
    timed = changes.snapshots(log)
    ticks = [int(s["tick"]) for s in timed]
    for eid, record, rows, end, flags in _CV._records(run):
        ok = [round(t * hz) for t, status, _, _ in rows if status == "ok"]
        groups = {}
        for t in ok:
            k = bisect_right(ticks, t) - 1
            if k < 0 or t < timed[k]["change_tick"] + steady:
                continue
            groups.setdefault(timed[k]["change_tick"], {}).setdefault(k, []).append(t)
        for _, parts in groups.items():
            if sum(len(v) for v in parts.values()) < 2:
                continue
            for k, samples in sorted(parts.items()):
                state = timed[k]
                r = roles.at(state)
                problems, q = _grid_check(exp, state, r, board)
                live = [c for c in _CG.CHANNELS if ((state.get("instruments") or {}).get(c) or {}).get("seated")]
                mine = {n: t for n, t in q.items() if n in live}
                if problems or not mine:
                    continue
                out.append(_Reading(int(end), eid, "record", state, r, live, mine, samples, flags))
    return sorted(out, key=lambda x: (x.tick, x.kind))


def _first_l2(run, exp, test):
    log = (run.logs or {}).get("state_log") or []
    roles = _CG.Roles(exp, log, _box_elements(run))
    for s in log:
        why = test(s, roles.at(s))
        if why:
            return hit(int(s["tick"]), why)
    return None


def _l3(readings, l2):
    if not l2.reached:
        return miss("not set up (L2)")
    if not readings:
        return miss("no valid reading of an instrument measuring the object")
    r = readings[0]
    return hit(max(r.tick, l2.tick), f"{r.eid}: {sorted(r.q)} ({r.kind})", r.flags)


def _after(level, tick, basis, flags=()):
    """A level that holds from `tick`, never before the level below it."""
    return hit(max(tick, level.tick or 0) if level.reached else tick, basis, flags)


def _ladder_hits(run, exp, l2_test, l4):
    hits = {"L1": grid_l1(run)}
    l2 = _first_l2(run, exp, l2_test)
    hits["L2"] = l2 or miss("never the object under test in a measurable state with an instrument on it")
    readings = grid_readings(run, exp)
    hits["L3"] = _l3(readings, hits["L2"])
    found = l4(run, readings)
    if found is None or not found.reached:
        hits["L4"] = found or miss("not enough data")
    else:
        hits["L4"] = _after(hits["L3"], found.tick, found.basis, found.flags) if hits["L3"].reached else found
    return hits


# --- battery_emf ------------------------------------------------------------------------------------------------
def _battery_l2(state, roles):
    q = _CV.measured_by("battery_emf", state, roles)
    if not q:
        return None
    i = abs(float(state.get("i_dut_a") or 0.0))
    return f"{sorted(q)} on the cell, cell current {i * 1e3:.3g} mA" + (" (open circuit)" if i < 1e-6 else "")


def _battery_l4(run, readings):
    seen = []
    for r in readings:
        i = abs(float(r.state.get("i_dut_a") or 0.0))
        other = next((x for x in seen if abs(i - x[1]) >= LOAD_STEP * max(i, x[1])), None)
        if other:
            return hit(r.tick, f"{other[0]} at {other[1] * 1e3:.3g} mA and {r.eid} at {i * 1e3:.3g} mA", r.flags)
        seen.append((r.eid, i))
    return miss(f"{len(readings)} valid reading(s), not at two cell currents >= {100 * LOAD_STEP:.0f} % apart "
                f"(open circuit counts as one)")


def battery_emf_levels(run):
    """L1-L4 of battery_emf). L2: an instrument on the board measuring the cell
    (across it, in series, across a shunt or another resistor block of printed value in series with it), loaded or
    open. L4: valid readings at two bench states whose cell currents differ by >= 5 % (open circuit is one)."""
    return _ladder_hits(run, "battery_emf", _battery_l2, _battery_l4)


# --- rc_capacitance and the capacitor of circuit_blackbox3 ------------------------------------------------
def _box_l2(exp):
    def test(state, roles):
        q = _CV.measured_by(exp, state, roles)
        if not q:
            return None
        if abs(float(state.get("i_dut_a") or 0.0)) >= 1e-6:
            return f"current through the box, {sorted(q)} on it"
        if _CV.closed_or_charged(exp, state, roles):
            return f"the box in a closed circuit (or its capacitor charged), {sorted(q)} on it"
        return None
    return test


def _cap_at(state, tick, hz):
    cap = state.get("cap") or {}
    v, vf, tau = cap.get("v"), cap.get("v_final"), cap.get("tau_s")
    if v is None or vf is None:
        return None
    if not tau:
        return float(v)
    return float(vf) + (float(v) - float(vf)) * math.exp(-max(0, tick - int(state["tick"])) / hz / float(tau))


def _steps(run, readings):
    """{change tick: (start V, final V, [(tick, capacitor V)])} of the readings in steps: bench states (counted
    changes) that began with the box capacitor >= STEP_MIN_V from its final value; plus {change tick: [(tick, V)]} of
    every reading (for the start values before a step)."""
    hz = run.physics_hz
    log = (run.logs or {}).get("state_log") or []
    _, changes = _CG.run_view(run)
    starts = {}
    for s in changes.snapshots(log):
        if int(s["tick"]) == int(s["change_tick"]) and s.get("cap"):
            starts[int(s["change_tick"])] = s
    points, steps = {}, {"counted": list(changes.counted)}
    for r in readings:
        c = int(r.state.get("change_tick", r.state.get("last_change_tick", -1)))
        for t in r.times:
            v = _cap_at(r.state, t, hz) if r.kind == "record" else (r.state.get("cap") or {}).get("v")
            if v is not None:
                points.setdefault(c, []).append((int(t), float(v)))
        s = starts.get(c)
        if s and abs(float(s["cap"]["v"]) - float(s["cap"]["v_final"])) >= STEP_MIN_V:
            steps[c] = (float(s["cap"]["v"]), float(s["cap"]["v_final"]))
    return steps, points


def _step_hit(run, readings):
    """L4 of a capacitor step: in one step >= 3 valid readings at different times whose
    capacitor voltages span >= 10 mV; or 2 such readings and a reading of an end point (the settled final value in
    the step, or the state just before it)."""
    steps, points = _steps(run, readings)
    counted = steps.pop("counted")
    for c, (v0, vf) in sorted(steps.items()):
        pts = sorted(set(points.get(c) or []))
        times = sorted({t for t, _ in pts})
        if len(times) < 2:
            continue
        span = max(v for _, v in pts) - min(v for _, v in pts)
        end = [t for t, v in pts if abs(v - vf) <= max(0.001, 0.002 * abs(v0 - vf))]
        prior = [x for x in counted if x < c]
        before = prior[-1] if prior and prior[-1] in points else None   # a reading of the state just before the step
        if len(times) >= 3 and span >= STEP_CHANGE_V:
            # the tick at which the third distinct time with the span became available: the readings' own ticks
            tick = max(r.tick for r in readings if int(r.state.get("change_tick", r.state.get("last_change_tick",
                                                                                                -1))) == c)
            return hit(tick, f"step from {v0:.3f} V toward {vf:.3f} V: {len(times)} reading times, "
                             f"{span * 1e3:.0f} mV")
        if span >= STEP_CHANGE_V and (end or before is not None):
            tick = max(r.tick for r in readings if int(r.state.get("change_tick", r.state.get("last_change_tick",
                                                                                                -1))) == c)
            return hit(tick, f"step from {v0:.3f} V toward {vf:.3f} V: 2 reading times and "
                             f"{'its settled value' if end else 'the value before it'}")
    return None


def _rc_l4(run, readings):
    return _step_hit(run, readings) or miss(
        "no step (a state that began >= 0.5 V from its final value) with 3 valid readings at different times "
        "spanning 10 mV (or 2 and an end point)")


def rc_capacitance_levels(run):
    """L1-L4 of rc_capacitance). L2: an instrument measuring the box (Q) while
    current flows through it, it is in a closed circuit, or its capacitor is charged. L4: see _step_hit."""
    return _ladder_hits(run, "rc_capacitance", _box_l2("rc_capacitance"), _rc_l4)


# --- resistor_blackbox -------------------------------------------------------------------------------------------
def _circuit_key(state, log_entry=None):
    """The board circuit without the instruments, independent of how it is wired: the partition of the elements' pins
    and the supply's straps into nodes (wires, the closed switch and the changeover's throw merged as roles.py does), and
    the supply on or off. A wire added only to move a voltmeter leaves it the same."""
    from benchmark.core.circuit_grid.roles import SUPPLY_NODES, _Nodes
    src = log_entry or state
    nodes = _Nodes(src.get("placement"), src.get("switch_closed"), src.get("throws"))
    insts = set(_CG.DISPLAYED) | set(_CG.CHANNELS) | {"V2", "A"}
    groups = {}
    for e in src.get("netlist") or []:
        if str(e[0]) in insts:
            continue
        for pin, n in enumerate((e[1], e[2])):
            groups.setdefault(nodes.find(str(n)), set()).add((str(e[0]), pin))
    for pin, n in enumerate(SUPPLY_NODES):
        groups.setdefault(nodes.find(n), set()).add(("__supply", pin))
    return (frozenset(frozenset(g) for g in groups.values()), bool((src.get("supply") or {}).get("on")))


def _entry_at(log, tick):
    ticks = [int(s["tick"]) for s in log]
    k = bisect_right(ticks, int(tick)) - 1
    return log[k] if k >= 0 else None


def _rb_quantities(r):
    """({'V', 'I'} the reading determines of the box, or {'Iref'} of a reference arm)."""
    roles = r.roles
    if roles is None:
        flags = r.state.get("instruments") or {}
        out = set()
        for n in r.names:
            f = flags.get(n) or {}
            if f.get("across_ab"):
                out.add("V")
            if f.get("series_ab"):
                out.add("I")
        return out
    box = roles["objects"].get("box") or {}
    tags = {n: set(t) for n, t in (box.get("instruments") or {}).items() if n in r.names}
    out = set()
    if any("across" in t for t in tags.values()) or \
            ("A" in tags and "series" in tags["A"] and any("across_ammeter" in t for t in tags.values())):
        out.add("V")
    if ("A" in tags and "series" in tags["A"]) or any("across_known" in t for t in tags.values()) or \
            (box.get("known_series") and any("across_supply" in t for t in tags.values())):
        out.add("I")
    if "A" in tags and "series" in tags["A"]:
        out.add("Ibox")
    for arm, insts in _CG.ref_arms(roles).items():
        if "A" in insts and "A" in r.names:
            out.add("Iref")
    return out


def _rb_l4(run, readings):
    log = (run.logs or {}).get("state_log") or []
    per_circuit, box_i, ref_i = {}, [], []
    for r in readings:
        got = _rb_quantities(r)
        entry = _entry_at(log, r.state["tick"]) or r.state
        if {"V", "I"} <= got:
            return hit(r.tick, f"{r.eid}: the box voltage and current in one reading ({sorted(r.names)})", r.flags)
        key = _circuit_key(r.state, entry)
        have = per_circuit.setdefault(key, set())
        have |= got & {"V", "I"}
        if {"V", "I"} <= have:
            return hit(r.tick, f"{r.eid}: the box voltage and current read one after the other in one circuit",
                       r.flags)
        rng = entry.get("ammeter_range")
        if "Ibox" in got:
            box_i.append(rng)
        if "Iref" in got:
            ref_i.append(rng)
        if any(a == b for a in box_i for b in ref_i):
            return hit(r.tick, f"{r.eid}: substitution, the box loop and a resistor block in its place read on one "
                               f"ammeter range", r.flags)
    return miss(f"{len(readings)} valid reading(s); no box voltage and current (together, one after the other in "
                f"one circuit, or by substitution)")


def _rb_l2(state, roles):
    on = (state.get("supply") or {}).get("on")
    if not on:
        return None
    if roles is None:
        if abs(float(state.get("i_ab_a") or 0)) >= 1e-6 and any(
                f.get("seated") and (f.get("across_ab") or f.get("series_ab"))
                for f in (state.get("instruments") or {}).values()):
            return "supply ON, current through A-B, an instrument across A-B or in series"
        return None
    box = roles["objects"].get("box") or {}
    q = _CG.q_instruments(roles, "box")
    if box.get("supply_loop") and q:
        return f"supply ON, the box in a closed circuit with it, {sorted(q)} measuring it"
    arms = _CG.ref_arms(roles)
    if arms:
        return f"supply ON, reference arm {sorted(arms)} in place of the box"
    return None


def resistor_blackbox_levels(run):
    """L1-L4 of resistor_blackbox). L2: supply ON and the box in a closed
    circuit with it and an instrument measuring it (Q), or a resistor block of printed value in its place with an
    instrument on it (substitution). L4: the box voltage (across A-B, or across it and the ammeter) and current
    (ammeter in series, across a resistor block in series, or the supply voltage with such a block) in one reading,
    one after the other in one circuit (only instruments moved), or by substitution on one ammeter range."""
    return _ladder_hits(run, "resistor_blackbox", _rb_l2, _rb_l4)


# --- led_planck -------------------------------------------------------------------------------------------------
def _led_l2(state, roles):
    currents = state.get("dut_currents_a") or {}
    for led, i in sorted(currents.items()):
        if abs(float(i)) < 1e-6:
            continue
        if roles is None:
            flags = state.get("instruments") or {}
            if any(led in (f.get("v_of") or []) or led in (f.get("i_of") or []) or f.get("on_dut")
                   for f in flags.values() if isinstance(f, dict) and f.get("seated")):
                return f"{led} carries {abs(i) * 1e3:.3g} mA with an instrument on it"
            continue
        q = _CG.q_instruments(roles, led)
        if q:
            return f"{led} carries {abs(i) * 1e3:.3g} mA, {sorted(q)} measuring it"
    return None


def _led_points(readings):
    out = []
    for r in readings:
        for led, i in sorted((r.state.get("dut_currents_a") or {}).items()):
            i = abs(float(i))
            if i < _CV.LED_CONDUCT_A:
                continue
            v_ok, i_ok = _CV.led_vi(r.roles, r.state, r.names, led)
            if v_ok and i_ok:
                out.append((r.tick, r.eid, led, i, r.flags))
    return sorted(out, key=lambda x: x[0])


def _led_l4(run, readings):
    points = _led_points(readings)
    per = {}
    for tick, eid, led, i, flags in points:
        per.setdefault(led, []).append(i)
        if "X" not in per:
            continue
        known = [k for k in per if k != "X"]
        alpha = [k for k in known if max(per[k]) >= LED_PAIR_RATIO * min(per[k])]
        if max(per["X"]) >= LED_PAIR_RATIO * min(per["X"]) and alpha:
            return hit(tick, f"I-V data of X and {alpha} (each >= 2 conducting readings, currents x{LED_PAIR_RATIO:g})",
                       flags)
        for k in known:
            if any(abs(a - b) <= LED_MATCH * max(a, b) for a in per["X"] for b in per[k]):
                return hit(tick, f"X and {k} read at matched currents (within {100 * LED_MATCH:.0f} %)", flags)
    summary = {k: f"{len(v)} x {min(v) * 1e3:.3g}-{max(v) * 1e3:.3g} mA" for k, v in sorted(per.items())}
    return miss(f"conducting readings with voltage and current: {summary or 'none'}; need X and one LED of printed "
                f"wavelength with I-V data (2 readings x{LED_PAIR_RATIO:g} apart each) or at matched currents")


def led_planck_levels(run):
    """L1-L4 of led_planck). L2: a seated LED carries >= 1 uA with an instrument
    measuring it. L4: X and at least one LED of printed wavelength each conducting (>= 0.5 mA) with voltage and current
    determined (circuit_grid_variants.led_vi), either (alpha) each with 2 readings whose currents differ x2, or (beta)
    X and that LED at currents within 10 %."""
    return _ladder_hits(run, "led_planck", _led_l2, _led_l4)


# --- circuit_blackbox3 ------------------------------------------------------------------------------------------
BB3_I_MIN_A = 1e-6              # a reading determines the box current only while >= 1 uA flows into the box


def _bb3_info(r):
    """(driven pairs, pairs whose voltage is read, current determined) of the box in one reading. The current: the
    ammeter in series, a voltmeter or channel across a printed-value resistor in series, or that resistor alone (from
    the published supply set point), each only while the bench's current into the box at the reading is >= 1 uA (supply
    ON or the capacitor discharging): with nothing flowing a 0 V / 0 A pair determines no resistor."""
    box = ((r.roles or {}).get("objects") or {}).get("box") or {}
    tags = {n: set(t) for n, t in (box.get("instruments") or {}).items() if n in r.names}
    pairs = box.get("pairs") or {}
    read = {pairs[n] for n, t in tags.items() if "across" in t and n in pairs}
    flowing = abs(float(r.state.get("i_dut_a") or 0.0)) >= BB3_I_MIN_A
    current = flowing and (("A" in tags and "series" in tags["A"]) or any("across_known" in t for t in tags.values())
                           or bool(box.get("known_series")))
    return set(box.get("loops") or []), read, current


def _bb3_l4(run, readings):
    elements = list(_box_elements(run) or [])
    n_r = sum(1 for e in elements if e.startswith("R"))
    has_c = any(e.startswith("C") for e in elements)
    log = (run.logs or {}).get("state_log") or []
    vi_pairs, probe, done = set(), False, {}
    per_circuit = {}
    watched = {}
    for r in readings:
        driven, read, current = _bb3_info(r)
        entry = _entry_at(log, r.state["tick"]) or r.state
        key = _circuit_key(r.state, entry)
        have = per_circuit.setdefault(key, [set(), set(), False])
        have[0] |= driven
        have[1] |= read
        have[2] = have[2] or current
        if have[2]:
            vi_pairs |= have[0] & have[1]
            if any(p not in have[0] for p in have[1]) and have[0]:
                probe = True
        c = int(r.state.get("change_tick", r.state.get("last_change_tick", -1)))
        for p in read:
            watched.setdefault(p, {}).setdefault(c, set()).update(r.times)
        if "R" not in done and n_r and (len(vi_pairs) >= n_r or probe):
            done["R"] = r.tick
        if "T" not in done and not has_c:
            seen = {p for p, by in watched.items() if any(len(ts) >= 2 for ts in by.values())}
            if "A-B" in seen or {"A-C", "B-C"} <= seen:
                done["T"] = r.tick
    if has_c:
        step = _step_hit(run, readings)
        if step:
            done["C"] = step.tick
            done["T"] = step.tick
    need = (["R"] if n_r else []) + (["C"] if has_c else []) + ["T"]
    if all(k in done for k in need):
        return hit(max(done[k] for k in need), f"resistors ({n_r}) determined by {sorted(vi_pairs) or 'a probe'}"
                   + ("; a capacitor step" if has_c else "; the topology by time-resolved readings"))
    return miss(f"needed {need}, reached {sorted(done)}: voltage and current on pairs {sorted(vi_pairs)}, third-terminal "
                f"probe {probe}")


def circuit_blackbox3_levels(run):
    """L1-L4 of circuit_blackbox3). L2 as rc_capacitance's on any terminal
    pair. L4: every box resistor determined (voltage and current on as many driven terminal pairs as resistors, read
    together or one after the other in one circuit, or a driven pair with its current and the third terminal's
    voltage); the capacitor (if any) by a step as rc_capacitance's; the topology told apart (by that step, or for a
    box without a capacitor by readings at 2 times in one state on A-B or on both A-C and B-C)."""
    return _ladder_hits(run, "circuit_blackbox3", _box_l2("circuit_blackbox3"), _bb3_l4)


# --- rectifier_supply -------------------------------------------------------------------------------------------
RECT_SETTLE_S = 1.0             # the voltmeter's mean (delta): >= 1 s after the last board change (the final gate's)


def _rect_meter(state):
    return [n for n in ("V1", "V2", "CH1", "CH2")
            if ((state.get("instruments") or {}).get(n) or {}).get("across_output")]


def _rect_l2_state(state):
    out = state.get("output") or {}
    caps = list(out.get("caps_across") or [])
    if not _rect_meter(state) or out.get("load_ohm") is None or out.get("shorted") or not caps:
        return None
    on = (state.get("supply") or {}).get("on") if isinstance(state.get("supply"), dict) else state.get("supply_on")
    if on:
        return f"source ON, {caps} and {out['load_ohm']:.0f} ohm across the output with an instrument on it"
    volts = state.get("capacitor_v") or {}
    if any(abs(float(volts.get(c) or 0.0)) >= STEP_MIN_V for c in caps):
        return f"source OFF, charged {caps} discharging into {out['load_ohm']:.0f} ohm with an instrument on it"
    return None


def rectifier_supply_levels(run):
    """L1-L4 of rectifier_supply).
    L2: the source ON with a capacitor and a load across the output (not joined) and an instrument across it; or the
    source OFF with a charged capacitor (>= 0.5 V) and a load across the output and an instrument across it (a
    discharge). L3: in such a state a logger record (finished or still recording at stop) with >= 2 ok samples of a
    channel across the output, or an observation of a voltmeter across the output converted after the last change.
    L4: each of C100, C220, C470 by one of (alpha) a steady segment on each side of its limit, (beta) one steady,
    valid segment at any load, (gamma) a discharge record: source OFF, that capacitor alone with a
    load across the output, >= 3 ok samples of a channel across it spanning 10 mV, (delta) the source ON, that
    capacitor alone with a load, an observation of a voltmeter across the output converted >= 1 s after the last
    change; and a steady segment at the task load (820 ohm +-3 %) meeting the requirement (as the legacy M2)."""
    from benchmark.evaluation.rectifier import MEAN_MIN_V, RIPPLE_MAX_V
    hz = run.physics_hz
    hits = {"L1": grid_l1(run)}
    log = (run.logs or {}).get("state_log") or []
    l2 = next((hit(int(s["tick"]), why) for s in log for why in (_rect_l2_state(s),) if why), None)
    hits["L2"] = l2 or miss("never a capacitor and a load across the output with an instrument on it (source ON, or "
                            "OFF with the capacitor charged)")
    board, changes = _CG.run_view(run)
    timed = changes.snapshots(log)
    ticks = [int(s["tick"]) for s in timed]
    l3 = []
    events = []                     # (tick, module or "final", how)
    for eid, record, rows, end, flags in _CV._records(run):
        ok = [(round(t * hz), ch1, ch2) for t, status, ch1, ch2 in rows if status == "ok"]
        groups = {}
        for t, ch1, ch2 in ok:
            k = bisect_right(ticks, t) - 1
            if k >= 0:
                groups.setdefault(timed[k]["change_tick"], (timed[k], []))[1].append((t, ch1, ch2))
        for _, (state, samples) in groups.items():
            if len(samples) >= 2 and _rect_l2_state(state) and \
                    any(((state.get("instruments") or {}).get(c) or {}).get("across_output") for c in _CG.CHANNELS):
                l3.append((int(end), f"{eid}: {len(samples)} samples of the output", flags))
            out = state.get("output") or {}
            caps = list(out.get("caps_across") or [])
            on = (state.get("supply") or {}).get("on")
            chans = [c for c in _CG.CHANNELS if ((state.get("instruments") or {}).get(c) or {}).get("across_output")]
            if not on and len(caps) == 1 and list(out.get("caps_on_board") or []) == caps and \
                    out.get("load_ohm") is not None and not out.get("shorted") and chans and len(samples) >= 3:
                col = 1 if chans[0] == "CH1" else 2
                v = [x[col] for x in samples]
                if max(v) - min(v) >= STEP_CHANGE_V:
                    events.append((int(end), caps[0], "a discharge record"))
    for eid, record in (run.evidence or {}).items():
        if record.get("kind") != "screen_capture" or not isinstance(record.get("state"), dict):
            continue
        state = changes.observation(record["state"])
        change = int(state["last_change_tick"])
        meters = [n for n in ("V1", "V2") if ((state.get("instruments") or {}).get(n) or {}).get("across_output")
                  and ((state.get("displays") or {}).get(n) or {}).get("value") is not None
                  and (((state.get("displays") or {}).get(n) or {}).get("sample_tick") or -1) > change]
        if not meters:
            continue
        if _rect_l2_state(dict(state, supply={"on": state.get("supply_on")})):
            l3.append((int(state["tick"]), f"{eid}: {meters} across the output", []))
        out = state.get("output") or {}
        caps = list(out.get("caps_across") or [])
        settled = round(RECT_SETTLE_S * hz)
        if state.get("supply_on") and len(caps) == 1 and list(out.get("caps_on_board") or []) == caps and \
                out.get("load_ohm") is not None and not out.get("shorted") and int(state["tick"]) - change >= settled \
                and any((((state.get("displays") or {}).get(n) or {}).get("sample_tick") or -1) >= change + settled
                        for n in meters):
            events.append((int(state["tick"]), caps[0], "a voltmeter mean"))
    if hits["L2"].reached and l3:
        t, basis, flags = min(l3, key=lambda x: x[0])
        hits["L3"] = hit(max(t, hits["L2"].tick), basis, flags)
    else:
        hits["L3"] = miss("no reading of the output in such a state" if hits["L2"].reached else "not set up (L2)")
    # (alpha), (beta) and the final segment: the legacy rule's segments (circuit_grid_variants.rectifier_supply)
    device = run.device or {}
    source_hz = float(device.get("source_hz") or 50.0)
    final = float(device.get("final_load_ohm", 820.0))
    tol = float(device.get("final_load_tolerance", 0.03))
    sides = {m: set() for m in _CV.RECT_MODULES}
    for eid, record, rows, end, flags in _CV._records(run):
        states = log or record.get("states") or []
        rate = float(record.get("sample_hz") or record.get("rate_hz") or 0.0)
        if not rate and len(rows) >= 2 and rows[1][0] > rows[0][0]:
            rate = 1.0 / (rows[1][0] - rows[0][0])
        if not states or not rate:
            continue
        period = round(rate / source_hz)
        for cap, load, v in _CV._rect_segments(rows, states, hz):
            measured = _CV._rect_steady(v, period)
            if measured is None or cap not in sides:
                continue
            meets = measured[0] >= MEAN_MIN_V and measured[1] <= RIPPLE_MAX_V
            sides[cap].add(meets)
            if len(sides[cap]) == 2:
                events.append((int(end), cap, "bracketed"))
            if meets and abs(load / final - 1) <= tol:
                events.append((int(end), "final", f"{cap} at {load:.0f} ohm"))
        for cap, load, v in _CV._rect_segments(rows, states, hz, changes):
            measured = _CV._rect_steady(v, period)
            if measured is None or cap not in sides:
                continue
            events.append((int(end), cap, "a steady segment"))
            if measured[0] >= MEAN_MIN_V and measured[1] <= RIPPLE_MAX_V and abs(load / final - 1) <= tol:
                events.append((int(end), "final", f"{cap} at {load:.0f} ohm"))
    covered, l4 = {}, None
    for t, what, how in sorted(events, key=lambda x: x[0]):
        covered.setdefault(what, (t, how))
        if all(k in covered for k in _CV.RECT_MODULES + ("final",)):
            l4 = hit(t, "; ".join(f"{k}: {covered[k][1]}" for k in _CV.RECT_MODULES + ("final",)))
            break
    if l4 is not None and hits["L3"].reached:
        l4 = _after(hits["L3"], l4.tick, l4.basis)
    hits["L4"] = l4 or miss(f"covered {sorted(covered)}; need each of {list(_CV.RECT_MODULES)} (both sides, a steady "
                            f"segment, a discharge record or a settled voltmeter mean) and the task load meeting the "
                            f"requirement")
    return hits


TIDY_NOTE = "six-level detectors"
for _exp, _detect in (("battery_emf", battery_emf_levels), ("rc_capacitance", rc_capacitance_levels),
                      ("resistor_blackbox", resistor_blackbox_levels), ("led_planck", led_planck_levels),
                      ("circuit_blackbox3", circuit_blackbox3_levels), ("rectifier_supply", rectifier_supply_levels)):
    LADDERS[_exp] = Ladder(detect=_detect, note=TIDY_NOTE.format(exp=_exp))

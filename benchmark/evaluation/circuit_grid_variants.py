"""Circuit-board experiments beyond resistor_blackbox (battery_emf, rc_capacitance, circuit_blackbox3, led_planck,
rectifier_supply): the measuring
state gate and the process milestones. The benches are benchmark/core/circuit_grid/variants.py (NetBench) with the
profiles battery.py, rc.py and network.py; the evidence is the same as resistor_blackbox's (observation captures with
the bench state, logger records with the state snapshots in force while they ran; benchmark/evaluation/circuit_grid.py)
with a device under test (DUT) in place of the box resistor A-B:
    i_dut_a       the largest current entering one DUT terminal from the board (battery_emf: the cell current; the
                  black box: its largest terminal current), at the observation or when that bench state began
    instruments   per instrument: seated, across (pins on two DUT terminals), series (removing it breaks the external
                  path between two DUT terminals), on_dut (across or series)
    cap           (a box with a capacitor) its voltage, current, Thevenin resistance, time constant and final voltage
                  when that state began
The gates readings_genuine and circuit_bench_restored are resistor_blackbox's.

Gate grid_measurement_valid: every entry used in the fit was taken in a measuring state: at least i_min_uA through the
DUT (at the observation; for a logger window in every bench state the window covers, when that state began), every
block on the board seated, no change that counts (a block seated on the board or leaving it, lever,
switch, supply, ammeter range, sweep start) for steady_s before (and every cited display converted after the last change), every cited instrument seated, and an
instrument (cited or not) seated across the DUT or in series with it: a reading elsewhere in the circuit, e.g. across a
shunt in the loop, counts while the DUT itself is being measured.
Only the blocks on the board are looked at (not the hand: a reading taken while the hand holds a block is valid,
), and only changes on the board (with lever, switch, supply,
range, sweep) count as changes; blocks in the tray are left to the tidy-up gates (definitions in
benchmark/evaluation/circuit_grid.py, "What a measurement looks at").

Process milestones (benchmark/evaluation/process_rules/circuits.py takes them from here):
E1  the first block seated on the board.
E2  the first bench state with >= 1 uA through the DUT and an instrument seated across it or in series with it.
M1  the first valid reading: an observation in a measuring state where an instrument on the DUT shows a number
    converted after the last change, or a logger record with >= 3 ok samples of a seated channel in such a state
    (>= 0.2 s after the change; timed at the record end).
M2  per experiment:
    battery_emf        valid readings at two bench states whose cell currents differ by >= 5 % (two loads)
    rc_capacitance     a logger record of a step: a state that began with the box capacitor >= 0.5 V from its final
                       voltage, a seated channel, samples starting within 2 tau of the step, lasting >= 2 tau, >= 10
                       samples per tau (tau from the bench solution of that state)
    circuit_blackbox3  valid readings that together pass current through every element of the box network, and if it
                       has a capacitor also a record of a step as for rc_capacitance
    led_planck         I-V data across conduction on X and on 2 LEDs of printed wavelength: for each, at least 3
                       valid readings while it conducts (>= 0.5 mA, the card's glow current), with its voltage and
                       its current both measured, spanning a current ratio of at least 10. A
                       reading is an observation as M1 or a bench-state entry of a logger record (during a SWEEP an
                       entry every 0.1 s; benchmark/core/circuit_grid/led.py), timed at the record end. The device
                       under test of E2 is every seated LED.
Gate led_x_measured (led_planck): at least one entry used in the fit shows X conducting (>= conduct_mA through X at the
observation, or in every bench-state entry of the window) with its voltage and its current both measured by the
instruments cited in that entry: voltage by a voltmeter or channel across X (v_of), current by an ammeter in series with
X or a voltmeter or channel across a resistor block in series with X (i_of).
rectifier_supply has its own milestones and the gate rectifier_grid_protocol (see rectifier_supply below).
"""
from __future__ import annotations

import math
from bisect import bisect_right

from benchmark.evaluation.circuit_grid import (CHANNELS, DISPLAYED, I_MIN_A, STEADY_S, BenchChanges, Roles,
                                               _measurements, logger_samples, process_state_log, q_instruments,
                                               run_view, sealed_view, unseated_on_board, window_states)

LOAD_STEP = 0.05          # two loads: cell currents differ by this fraction
STEP_V = 0.5
MIN_SAMPLES = 3


# --- measuring state ----------------------------------------------------------------------------------------------
def state_problems(state, *, board, i_min_a=I_MIN_A):
    """Why a bench state is not a measuring state; board = board_history of the episode (only the blocks on the board
    are looked at, not the hand; circuit_grid.py)."""
    problems = []
    i = abs(float(state.get("i_dut_a") or 0.0))
    if i < i_min_a:
        problems.append(f"current through the device under test {i:.3g} A < {i_min_a} A")
    unseated = unseated_on_board(state, board)
    if unseated:
        problems.append(f"blocks on the board not seated: {unseated}")
    return problems


def instrument_problems(state, names):
    problems = []
    flags = state.get("instruments") or {}
    for name in names:
        if not (flags.get(name) or {}).get("seated"):
            problems.append(f"{name} not seated on the board")
    if not any(f.get("seated") and f.get("on_dut") for f in flags.values() if isinstance(f, dict)):
        problems.append("no instrument seated across the device under test or in series with it")
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


# --- measuring state, six-level design ---------------------------------------------------------------
# No current floor any more (a zero-current reading, the open-circuit voltage of the cell, the settled end of a step
# are readings). In its place:
#   rc_capacitance, circuit_blackbox3  the box in a closed circuit (an external path between two of its terminals that
#                   is not through a voltmeter or channel input), or its capacitor charged (>= 0.5 V) with only
#                   high-impedance inputs on it (the start value V0 of a discharge)
#   battery_emf, led_planck  nothing beyond an instrument measuring the object
# and for all: an instrument on the board (cited or not) measuring the object (Q, circuit_grid.Q_TAGS: across it, in
# series, across a resistor block of printed value in series with it, across it and the ammeter, or across the supply
# with such a block in series), every block on the board seated, every cited instrument seated. A logger window may
# contain one throw of the changeover (or the switch) or one supply press on rc_capacitance and circuit_blackbox3
# (the step itself; nothing else on the board changes), and during an LED sweep only the state entries inside the
# window are looked at. Without the process record (roles) the bench's topology flags and the old current floor stand
# in.
STEP_EXPS = ("rc_capacitance", "circuit_blackbox3")
LEVER_TOKENS = {"changeover", "supply", "switch", "held"}


def _cap_charged(state):
    return abs(float((state.get("cap") or {}).get("v") or 0.0)) >= STEP_V


def closed_or_charged(exp, state, roles, i_min_a=I_MIN_A):
    """rc_capacitance / circuit_blackbox3: the box in a closed circuit or its capacitor charged (see above)."""
    if roles is None:
        return abs(float(state.get("i_dut_a") or 0.0)) >= i_min_a or _cap_charged(state)
    box = (roles.get("objects") or {}).get("box") or {}
    return bool(box.get("loops")) or _cap_charged(state)


def measured_by(exp, state, roles):
    """{instrument: tags} of the instruments on the board measuring the object (Q); without roles the bench's flags."""
    if roles is None:
        return {k: ["on_dut"] for k, f in (state.get("instruments") or {}).items()
                if isinstance(f, dict) and f.get("seated") and f.get("on_dut")}
    return q_instruments(roles)


def measuring_problems(exp, state, roles, names, *, board, closed=True, i_min_a=I_MIN_A):
    """Why a bench state is not a measuring state for readings of `names` (see above)."""
    problems = []
    unseated = unseated_on_board(state, board)
    if unseated:
        problems.append(f"blocks on the board not seated: {unseated}")
    flags = state.get("instruments") or {}
    for name in names:
        if not (flags.get(name) or {}).get("seated"):
            problems.append(f"{name} not seated on the board")
    if not measured_by(exp, state, roles):
        problems.append("no instrument on the board measures the device under test (across it, in series, across a "
                        "resistor block in series with it, across it and the ammeter, or across the supply)")
    if closed and exp in STEP_EXPS and not closed_or_charged(exp, state, roles, i_min_a):
        problems.append("the box is in no closed circuit and its capacitor is not charged")
    return problems


def grid_observation_problems(exp, state, names, physics_hz, *, board, roles, steady_s=STEADY_S, i_min_a=I_MIN_A):
    problems = measuring_problems(exp, state, roles, names, board=board, i_min_a=i_min_a)
    tick, change = int(state["tick"]), int(state["last_change_tick"])
    if tick - change < round(steady_s * physics_hz):
        problems.append(f"last change {(tick - change) / physics_hz:.3f} s before the observation < {steady_s} s")
    for name in names:
        shown = (state.get("displays") or {}).get(name) or {}
        if shown.get("sample_tick") is None or shown["sample_tick"] <= change:
            problems.append(f"{name} display not updated since the last change")
    return problems


def in_window(states, t0_tick, t1_tick):
    """The bench states in force during [t0, t1]; the one before t0 left out when an entry of the same bench state
    (an LED sweep's 0.1 s entries) lies inside the window."""
    found = window_states(states, t0_tick, t1_tick)
    if len(found) > 1 and found[0]["tick"] <= t0_tick and found[1]["change_tick"] == found[0]["change_tick"]:
        found = found[1:]
    return found


def _board_part(state):
    return {b: p for b, p in (state.get("placement") or {}).items() if str(p).startswith("board")}


def lever_step(found):
    """(ok, index of the first state after the step) when the change-tick groups of a window differ only by one
    throw of the changeover (to the open middle and on to the other throw counts as one), one switch change or one
    supply press: the board's blocks in the same places throughout; else (False, None). One group: (True, 0)."""
    groups = []
    for k, s in enumerate(found):
        if not groups or groups[-1][0] != s["change_tick"]:
            groups.append((s["change_tick"], k))
    if len(groups) == 1:
        return True, 0
    events = 0
    for (_, a), (_, b) in zip(groups, groups[1:]):
        prev, nxt = found[a], found[b]
        tokens = set(str(nxt.get("reason") or "").split(","))
        if _board_part(prev) != _board_part(nxt):
            return False, None
        extra = tokens - LEVER_TOKENS - set(nxt.get("placement") or {})
        if extra:
            return False, None
        if "supply" in tokens or "switch" in tokens:
            events += 1
        if "changeover" in tokens and (nxt.get("throws") or {}).get("CO") is not None:
            events += 1                                # landing on a throw ends one flip (the open middle does not)
    if events > 1:
        return False, None
    return True, groups[-1][1]


def grid_window_problems(exp, states, t0_tick, t1_tick, names, physics_hz, *, board, roles, steady_s=STEADY_S,
                         i_min_a=I_MIN_A):
    found = in_window(states, t0_tick, t1_tick)
    if not found:
        return ["no bench state for the window"]
    problems = []
    step_at = 0
    if len({s["change_tick"] for s in found}) != 1:
        ok, step_at = lever_step(found) if exp in STEP_EXPS else (False, None)
        if not ok:
            problems.append("the bench changed during the window" + (
                " (beyond one throw of the changeover or one supply press)" if exp in STEP_EXPS else ""))
            step_at = 0
    change = found[0]["change_tick"]
    if t0_tick - change < round(steady_s * physics_hz):
        problems.append(f"window starts {(t0_tick - change) / physics_hz:.3f} s after the last change < {steady_s} s")
    for k, s in enumerate(found):
        for p in measuring_problems(exp, s, roles.at(s), names, board=board, closed=k >= step_at, i_min_a=i_min_a):
            if p not in problems:
                problems.append(p)
    return problems


def grid_measurement_valid(ctx, i_min_uA=I_MIN_A * 1e6, steady_s=STEADY_S):
    """Gate grid_measurement_valid, six-level design (see "measuring state" above). i_min_uA is kept
    in the specs and applies only without the process record."""
    i_min_A = float(i_min_uA) * 1e-6
    exp = ctx.spec.id
    hz = ctx.sealed.episode["physics_hz"]
    used = [(k, m) for k, m in enumerate(_measurements(ctx)) if isinstance(m, dict) and m.get("used_in_fit") is True]
    if not used:
        return False, {"reason": "no measurement used in the fit"}
    board, changes = sealed_view(ctx)
    roles = Roles(exp, process_state_log(ctx.sealed), (ctx.sealed.episode.get("device") or {}).get("box_elements"))
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
            found = grid_observation_problems(exp, state, names, hz, board=board, roles=roles.at(state),
                                              steady_s=steady_s, i_min_a=i_min_A)
        elif record.get("kind") == "circuit_logger" and isinstance(record.get("states"), list):
            try:
                t0, t1 = round(float(m["t0_s"]) * hz), round(float(m["t1_s"]) * hz)
            except (KeyError, TypeError, ValueError):
                problems.append(f"#{k}: a logger entry needs t0_s and t1_s")
                continue
            found = grid_window_problems(exp, changes.snapshots(record["states"]), t0, t1, names, hz, board=board,
                                         roles=roles, steady_s=steady_s, i_min_a=i_min_A)
        else:
            found = ["not an observation or a logger record"]
        problems += [f"#{k}: {p}" for p in found]
    return not problems, {"entries_used": len(used), "problems": problems[:20]}


# --- process milestones -----------------------------------------------------------------------------------------------
def _valid_observations(run):
    """(tick, eid, state, instruments on the DUT showing a valid number) of the observations in a measuring state."""
    hz = run.physics_hz
    board, changes = run_view(run)
    out = []
    for eid, record in (run.evidence or {}).items():
        if record.get("kind") != "screen_capture" or not isinstance(record.get("state"), dict):
            continue
        state = changes.observation(record["state"])
        change = int(state["last_change_tick"])
        names = []
        for name in DISPLAYED:
            shown = (state.get("displays") or {}).get(name) or {}
            flags = (state.get("instruments") or {}).get(name) or {}
            if flags.get("seated") and shown.get("value") is not None and (shown.get("sample_tick") or -1) > change:
                names.append(name)
        good = [n for n in names if not observation_problems(state, [n], hz, board=board)
                or not observation_problems(state, names, hz, board=board)]
        if good:
            out.append((state["tick"], eid, state, good))
    return sorted(out, key=lambda x: x[0])


def _records(run):
    """(eid, record, rows (t_s, status, ch1, ch2), end tick, flags) of the finished records and the open one."""
    out = [(eid, r, logger_samples(r), r["end_tick"], []) for eid, r in (run.evidence or {}).items()
           if r.get("kind") == "circuit_logger"]
    open_record = (run.logs or {}).get("logger_open")
    if isinstance(open_record, dict) and open_record.get("rows"):
        rows = [(r["t_s"], r["status"], r["ch1_V"], r["ch2_V"]) for r in open_record["rows"]]
        out.append((open_record["evidence_id"], open_record, rows, open_record["last_tick"], ["open_record"]))
    return sorted(out, key=lambda x: x[3])


def _record_runs(rows, states, physics_hz):
    """(state, ok sample ticks) per bench state of a record, the samples >= STEADY_S after that state's change."""
    out = []
    ticks = [round(t * physics_hz) for t, status, _, _ in rows if status == "ok"]
    if not ticks:
        return out
    by_change = {}
    for s in window_states(states, ticks[0], ticks[-1]):
        by_change.setdefault(s["change_tick"], s)
    for change, s in by_change.items():
        later = [x["tick"] for x in states if x["change_tick"] != change and x["tick"] > change]
        end = min(later) if later else math.inf
        inside = [t for t in ticks if change <= t < end]
        out.append((s, inside))
    return out


def _valid_record_runs(run):
    """(end tick, eid, state, steady sample ticks, all sample ticks, flags) of record runs in a measuring state."""
    hz = run.physics_hz
    board, changes = run_view(run)
    states = changes.snapshots((run.logs or {}).get("state_log") or [])
    out = []
    for eid, record, rows, end, flags in _records(run):
        for state, inside in _record_runs(rows, states, hz):
            steady = [t for t in inside if t >= state["change_tick"] + round(STEADY_S * hz)]
            live = [c for c in CHANNELS if (state["instruments"].get(c) or {}).get("seated")]
            if not live or state_problems(state, board=board):
                continue
            out.append((end, eid, state, steady, inside, flags, live, record))
    return out


def _common(run):
    from benchmark.evaluation.process import hit, miss
    actions = (run.logs or {}).get("actions") or []
    states = (run.logs or {}).get("state_log") or []
    hits = {}
    first = next((a for a in actions if a.get("event") == "placement" and str(a.get("to", "")).startswith("board:")),
                 None)
    hits["E1"] = (hit(first["tick"], f"{first['block']} seated at {first['to']}") if first
                  else miss("no block seated on the board"))
    e2 = next((s for s in states if abs(s.get("i_dut_a") or 0.0) >= I_MIN_A and any(
        f.get("seated") and f.get("on_dut") for f in s["instruments"].values())), None)
    hits["E2"] = (hit(e2["tick"], "current through the device under test with an instrument across it or in series")
                  if e2 else miss("never current through the device under test with an instrument connected to it"))
    m1 = [(t, f"{eid}: {names}", []) for t, eid, _, names in _valid_observations(run)]
    for end, eid, state, steady, _, flags, live, _ in _valid_record_runs(run):
        if len(steady) >= MIN_SAMPLES and not instrument_problems(state, live):
            m1.append((end, f"{eid}: {len(steady)} samples with {live}", flags))
    best = min(m1, key=lambda x: x[0]) if m1 else None
    hits["M1"] = hit(best[0], best[1], best[2]) if best else miss("no valid reading")
    return hits


def battery_emf(run):
    from benchmark.evaluation.process import hit, miss
    hits = _common(run)
    readings = [(t, eid, abs(state["i_dut_a"]), []) for t, eid, state, _ in _valid_observations(run)]
    for end, eid, state, steady, _, flags, live, _ in _valid_record_runs(run):
        if len(steady) >= MIN_SAMPLES and not instrument_problems(state, live):
            readings.append((end, eid, abs(state["i_dut_a"]), flags))
    readings.sort(key=lambda x: x[0])
    m2 = None
    for k, (t, eid, i, flags) in enumerate(readings):
        other = next((r for r in readings[:k] if abs(i - r[2]) >= LOAD_STEP * max(i, r[2])), None)
        if other:
            m2 = hit(t, f"{other[1]} at {1000 * other[2]:.2f} mA and {eid} at {1000 * i:.2f} mA", flags)
            break
    hits["M2"] = m2 or miss(f"{len(readings)} valid reading(s), not at two loads (cell currents >= "
                            f"{100 * LOAD_STEP:.0f} % apart)")
    return hits


def _step_records(run):
    """(end tick, basis, flags) of logger records of a capacitor step (see the module docstring, M2)."""
    hz = run.physics_hz
    out = []
    for end, eid, state, steady, inside, flags, live, record in _valid_record_runs(run):
        cap = state.get("cap") or {}
        tau = cap.get("tau_s")
        if not tau or cap.get("v") is None or cap.get("v_final") is None:
            continue
        if abs(cap["v"] - cap["v_final"]) < STEP_V or len(inside) < 2:
            continue
        lag = (inside[0] - state["change_tick"]) / hz
        span = (inside[-1] - inside[0]) / hz
        rate = record.get("sample_hz") or 0
        if lag <= 2 * tau and span >= 2 * tau and rate * tau >= 10:
            out.append((end, f"{eid}: {span / tau:.1f} tau of a step from {cap['v']:.3f} V toward "
                             f"{cap['v_final']:.3f} V, from {lag / tau:.2f} tau, {rate * tau:.0f} samples per tau",
                        flags))
    return out


def rc_capacitance(run):
    from benchmark.evaluation.process import hit, miss
    hits = _common(run)
    steps = _step_records(run)
    hits["M2"] = (hit(*min(steps, key=lambda x: x[0])) if steps else
                  miss("no logger record of a capacitor step (>= 2 tau from within 2 tau of the step, >= 10 samples "
                       "per tau)"))
    return hits


def _run_currents(state, inside, physics_hz):
    """The state with dut_currents_a = the larger magnitude of each element's current at the start and at the last
    sample of a record run (with a capacitor every current relaxes exponentially from the state's start toward its
    settled value, so the extremes are at the ends)."""
    cap = state.get("cap") or {}
    final, tau = cap.get("dut_currents_final_a"), cap.get("tau_s")
    if not final or not tau or not inside:
        return state
    decay = math.exp(-(inside[-1] - state["change_tick"]) / physics_hz / tau)
    start = state.get("dut_currents_a") or {}
    end = {k: final.get(k, 0.0) + (i - final.get(k, 0.0)) * decay for k, i in start.items()}
    return dict(state, dut_currents_a={k: max(abs(start[k]), abs(end[k])) for k in start})


def circuit_blackbox3(run):
    from benchmark.evaluation.process import hit, miss
    hits = _common(run)
    elements = set((run.device or {}).get("box_elements") or [])
    covered, done = set(), None
    events = [(t, eid, state) for t, eid, state, _ in _valid_observations(run)]
    events += [(end, eid, _run_currents(state, inside, run.physics_hz))
               for end, eid, state, steady, inside, _, live, _ in _valid_record_runs(run)
               if len(steady) >= MIN_SAMPLES and not instrument_problems(state, live)]
    for t, eid, state in sorted(events, key=lambda x: x[0]):
        covered |= {k for k, i in (state.get("dut_currents_a") or {}).items() if abs(i) >= I_MIN_A}
        if elements and covered >= elements:
            done = (t, eid)
            break
    steps = _step_records(run)
    needs_step = any(e.startswith("C") for e in elements)
    if done is None:
        hits["M2"] = miss(f"current through {sorted(covered)} of {sorted(elements)} in valid readings")
    elif needs_step and not steps:
        hits["M2"] = miss("every element carried current, but no logger record of a capacitor step")
    else:
        tick = max(done[0], min(s[0] for s in steps)) if needs_step else done[0]
        hits["M2"] = hit(tick, f"every element of {sorted(elements)} carried current by {done[1]}"
                         + (" and a capacitor step was recorded" if needs_step else ""))
    return hits


# --- led_planck (benchmark/core/circuit_grid/led.py) -----------------------------------------------------------------
LED_MIN_KNOWN = 2           # X and two LEDs of printed wavelength
LED_CONDUCT_A = 0.5e-3      # conducting: at least the glow current (public as glow_ma)
LED_MIN_READINGS = 3        # I-V data across conduction: at least 3 valid readings while conducting,
LED_CURRENT_SPAN = 10.0     # spanning a current ratio of at least 10


def _led_current(state, led):
    return abs(float((state.get("dut_currents_a") or {}).get(led) or 0.0))


def _led_measured(state, names, led):
    """The instruments `names` measure both the LED's voltage and its current in this bench state."""
    flags = state.get("instruments") or {}
    v = any(led in ((flags.get(n) or {}).get("v_of") or []) for n in names)
    i = any(led in ((flags.get(n) or {}).get("i_of") or []) for n in names)
    return v and i


def led_vi(roles, state, names, led):
    """(voltage determined, current determined) of one LED by the instruments `names` read in one bench state
    (six-level design). Voltage: a voltmeter or channel across the LED; across the LED and the
    ammeter in series with it, the ammeter read too (its drop is published); or across the driving supply together with
    one across a resistor block of printed value in series with the LED (V_LED = V_S - V_R). Current: the ammeter in
    series with it, or a voltmeter or channel across a resistor block of printed value in series with it. Without
    roles: the bench's v_of / i_of flags."""
    names = set(names)
    if roles is None:
        flags = state.get("instruments") or {}
        return (any(led in ((flags.get(n) or {}).get("v_of") or []) for n in names),
                any(led in ((flags.get(n) or {}).get("i_of") or []) for n in names))
    o = ((roles.get("objects") or {}).get(led)) or {}
    tags = {n: set(t) for n, t in (o.get("instruments") or {}).items() if n in names}
    known = any("across_known" in t for t in tags.values())
    amm = any("series" in t for n, t in tags.items() if n == "A")
    v = any("across" in t for t in tags.values()) or \
        (amm and any("across_ammeter" in t for t in tags.values())) or \
        (known and bool(o.get("known_series")) and any("across_supply" in t for t in tags.values()))
    return v, amm or known


def led_x_measured(ctx, conduct_mA=LED_CONDUCT_A * 1e3):
    """X itself measured while conducting (gate led_x_measured), six-level design: the entries used
    in the fit that cite one evidence record are taken together (an observation's voltmeter and ammeter written as two
    entries), X's voltage and current determined as in led_vi, and a logger window needs one state entry inside it with
    X at >= conduct_mA (the window of a sweep may include the part below conduction)."""
    conduct = float(conduct_mA) * 1e-3
    hz = ctx.sealed.episode["physics_hz"]
    used = [(k, m) for k, m in enumerate(_measurements(ctx)) if isinstance(m, dict) and m.get("used_in_fit") is True]
    roles = Roles("led_planck", process_state_log(ctx.sealed))
    cited = {}
    for _, m in used:
        if isinstance(m.get("readings"), dict):
            cited.setdefault(m.get("evidence_id"), set()).update(m["readings"])
    tried = []
    for k, m in used:
        record, problem = ctx.evidence(m.get("evidence_id"))
        if problem:
            tried.append(f"#{k}: {problem}")
            continue
        names = sorted(cited.get(m.get("evidence_id")) or ())
        if record.get("kind") == "screen_capture" and isinstance(record.get("state"), dict):
            states = [record["state"]]
        elif record.get("kind") == "circuit_logger" and isinstance(record.get("states"), list):
            try:
                states = in_window(record["states"], round(float(m["t0_s"]) * hz), round(float(m["t1_s"]) * hz))
            except (KeyError, TypeError, ValueError):
                tried.append(f"#{k}: a logger entry needs t0_s and t1_s")
                continue
        else:
            tried.append(f"#{k}: not an observation or a logger record")
            continue
        if not states:
            tried.append(f"#{k}: no bench state")
            continue
        best = None
        for st in states:
            v, i = led_vi(roles.at(st), st, names, "X")
            if _led_current(st, "X") >= conduct and v and i:
                best = st
                break
        if best is not None:
            return True, {"entry": k, "evidence_id": m.get("evidence_id"), "x_current_a": _led_current(best, "X"),
                          "instruments": names}
        i_x = max(_led_current(st, "X") for st in states)
        vi = [led_vi(roles.at(st), st, names, "X") for st in states]
        tried.append(f"#{k}: X carries up to {i_x * 1e3:.3g} mA; {names} determine its voltage: "
                     f"{any(v for v, _ in vi)}, its current: {any(i for _, i in vi)}")
    return False, {"reason": f"no entry used in the fit shows X conducting (>= {conduct_mA} mA) with its voltage and "
                             f"its current both determined by the cited instruments", "entries": tried[:20]}


def _led_readings(run):
    """(tick, evidence_id, led, current) of valid readings of an LED's voltage and current while it conducts."""
    hz = run.physics_hz
    out = []
    for t, eid, state, names in _valid_observations(run):
        for led in (state.get("dut_currents_a") or {}):
            i = _led_current(state, led)
            if i >= LED_CONDUCT_A and _led_measured(state, names, led):
                out.append((t, eid, led, i))
    board, changes = run_view(run)
    states = changes.snapshots((run.logs or {}).get("state_log") or [])
    for eid, record, rows, end, flags in _records(run):
        ticks = [round(t * hz) for t, status, _, _ in rows if status == "ok"]
        if len(ticks) < MIN_SAMPLES:
            continue
        for st in window_states(states, ticks[0], ticks[-1]):
            if max(st["tick"], ticks[0]) < st["change_tick"] + round(STEADY_S * hz):
                continue
            live = [c for c in CHANNELS if (st["instruments"].get(c) or {}).get("seated")]
            if not live or state_problems(st, board=board) or instrument_problems(st, live):
                continue
            for led in (st.get("dut_currents_a") or {}):
                i = _led_current(st, led)
                if i >= LED_CONDUCT_A and _led_measured(st, live, led):
                    out.append((end, eid, led, i))
    return sorted(out, key=lambda x: x[0])


def led_planck(run):
    """M2: I-V data across conduction on X and LED_MIN_KNOWN LEDs of printed wavelength (module docstring). M1 also
    counts the first such reading (a record that starts before a sweep reaches conduction)."""
    from benchmark.evaluation.process import hit, miss
    hits = _common(run)
    readings = _led_readings(run)
    if readings and (not hits["M1"].reached or readings[0][0] < hits["M1"].tick):
        t, eid, led, i = readings[0]
        hits["M1"] = hit(t, f"{eid}: {led} at {i * 1e3:.3g} mA, voltage and current measured")
    per, done = {}, None

    def across(led):
        currents = per.get(led) or []
        return len(currents) >= LED_MIN_READINGS and max(currents) >= LED_CURRENT_SPAN * min(currents)

    for t, eid, led, i in readings:
        per.setdefault(led, []).append(i)
        ok = sorted(k for k in per if across(k))
        if "X" in ok and len(ok) - 1 >= LED_MIN_KNOWN:
            done = hit(t, f"I-V data across conduction on {ok} by {eid}")
            break
    summary = {k: f"{len(v)} readings, {min(v) * 1e3:.3g}-{max(v) * 1e3:.3g} mA" for k, v in sorted(per.items())}
    hits["M2"] = done or miss(f"I-V data while conducting: {summary or 'none'}; need X and {LED_MIN_KNOWN} LEDs of "
                              f"printed wavelength, each with {LED_MIN_READINGS} readings spanning a current ratio of "
                              f"{LED_CURRENT_SPAN:g}")
    return hits


# --- rectifier_supply (benchmark/core/circuit_grid/rectifier.py) ------------------------------------------------------
RECT_MIN_S = 0.1


def _rect_ready(state):
    out = state.get("output") or {}
    return bool((state.get("supply") or {}).get("on")) and bool(out.get("caps_across")) and \
        out.get("load_ohm") is not None and not out.get("shorted")


def _rect_records(run):
    """(end tick, eid, capacitors across the output) of finished logger records with a channel across the output
    while the source was ON with a capacitor and a load across it, unchanged for >= RECT_MIN_S of the record. Only
    changes that count (circuit_grid.py) split a record: consecutive bench states with the same last
    counted change form one stretch, so a block tipped in the tray, or the hand taking or holding a block that is not
    on the board, does not cut it; the hand is not looked at."""
    hz = run.physics_hz
    _, changes = run_view(run)
    out = []
    for eid, record in (run.evidence or {}).items():
        if record.get("kind") != "circuit_logger" or not isinstance(record.get("states"), list):
            continue
        start, end = record["start_tick"], record["end_tick"]
        stretches = []                                   # [first state]
        for s in changes.snapshots(record["states"]):
            if not stretches or stretches[-1][0]["change_tick"] != s["change_tick"]:
                stretches.append([s])
        for k, (s,) in enumerate(stretches):
            lo = max(s["tick"], start)
            hi = min(stretches[k + 1][0]["tick"], end) if k + 1 < len(stretches) else end
            live = any((s["instruments"].get(c) or {}).get("across_output") for c in CHANNELS)
            if hi - lo >= RECT_MIN_S * hz and live and _rect_ready(s):
                out.append((end, eid, tuple(s["output"]["caps_across"])))
    return sorted(out)


RECT_MODULES = ("C100", "C220", "C470")
RECT_LOAD_STEADY = 0.005        # the load within 0.5 % across a steady segment
RECT_SETTLED = 0.01             # the first and last analysed periods' means within 1 %
RECT_PERIODS = (2, 10)          # whole source periods analysed: at least 2, the last <= 10
RECT_COLUMN = {"CH1": 2, "CH2": 3}


def _rect_steady_key(state):
    """(module, load ohm, channel) of a state that can hold a steady segment: the source ON, exactly one capacitor
    block on the board and it seated across the output, the output terminals not joined, a load across the output and
    a logger channel across it (CH1 first); else None."""
    out = state.get("output") or {}
    caps = list(out.get("caps_across") or [])
    if not (state.get("supply") or {}).get("on") or out.get("shorted") or out.get("load_ohm") is None or \
            len(caps) != 1 or list(out.get("caps_on_board") or []) != caps:
        return None
    channels = [c for c in CHANNELS if ((state.get("instruments") or {}).get(c) or {}).get("across_output")]
    return (caps[0], float(out["load_ohm"]), channels[0]) if channels else None


def _rect_segments(rows, states, physics_hz, changes=None):
    """(module, load ohm, output voltages) of each run of consecutive samples of a record whose bench states all
    hold one module, one channel and a load within RECT_LOAD_STEADY of the run's first (_rect_steady_key); a sample
    at t_s belongs to the physics tick ceil(t_s * physics_hz). Runs with a sample not ok are dropped. With changes
    (BenchChanges) only valid runs: no change that counts (circuit_grid.py) inside the run; a change
    in the tray does not cut it and the hand is not looked at."""
    if changes is not None:
        states = changes.snapshots(states)
    out, current = [], None
    ticks = [s["tick"] for s in states]
    for t, status, ch1, ch2 in rows:
        k = bisect_right(ticks, math.ceil(t * physics_hz - 1e-9)) - 1
        key = _rect_steady_key(states[k]) if k >= 0 else None
        if key and changes is not None:
            key = key + (states[k]["change_tick"],)
        if key and current and current[0] == key[0] and current[2] == key[2] and current[4] == key[3:] and \
                abs(key[1] / current[1] - 1) <= RECT_LOAD_STEADY:
            current[3].append((status, ch1 if key[2] == "CH1" else ch2))
            continue
        current = [key[0], key[1], key[2], [(status, ch1 if key[2] == "CH1" else ch2)], key[3:]] if key else None
        if current:
            out.append(current)
    return [(cap, load, [v for _, v in samples]) for cap, load, _, samples, _ in out
            if all(s == "ok" for s, _ in samples)]


def _rect_steady(v, period):
    """(mean V, ripple V) over the last whole source periods (RECT_PERIODS) of a segment when the first and last of
    them agree in mean within RECT_SETTLED; else None."""
    n = min(RECT_PERIODS[1], len(v) // period) if period > 0 else 0
    if n < RECT_PERIODS[0]:
        return None
    v = v[-n * period:]
    first, last = sum(v[:period]) / period, sum(v[-period:]) / period
    if abs(last - first) > RECT_SETTLED * max(abs(last), 1e-6):
        return None
    return sum(v) / len(v), max(v) - min(v)


def rectifier_supply(run):
    """E1 the first block seated; E2 the first state with the source ON and a capacitor and a load across the
    output; M1 the first logger record of the output in such a state (a channel across the output, >= 0.1 s
    unchanged).
    M2 is reached by either of (: M2 does not prescribe the method; whether R_min is right is judged by
    the gates and e):
    (a) bracketed (design): for EACH of C100, C220 and C470 a steady record
    segment on each side of that module's minimum load R_min, and a steady segment at the task load that meets the
    supply requirement (the final configuration);
    (b) recorded: for EACH of C100, C220 and C470 at least one steady, valid record segment (any load), and a steady,
    valid segment at the task load that meets the supply requirement. Valid: no change that counts on the board
    during the segment (circuit_grid.py; a change in the tray does not cut it, the hand is not looked at).
    The two sides of R_min are defined by the requirement: a segment that
    meets it (mean >= 8.5 V and ripple <= 0.8 V) is at or above R_min, one that does not meet it (the mean or the
    ripple fails) is below R_min; a module is bracketed when it has a steady segment of each. The task load is the
    final-configuration load of the gate rectifier_grid_protocol: the device's final_load_ohm within its
    final_load_tolerance (the recipe's 820 ohm block, +-3 %; accepted deviation), with any module.
    A steady segment is a run of samples of one logger record (finished, or still recording at stop) during which
    the bench state log shows the source ON, exactly one capacitor block on the board and it seated across the
    output, the output terminals not joined, the load across the output within 0.5 % of the run's first, and the
    same logger channel across the output (CH1 first); every sample ok; at least 2 whole source periods; the first
    and last analysed periods' means within 1 %; the mean and the peak-to-peak ripple measured over the last <= 10
    whole source periods. Timed at the end of the record that completes (a) or (b)."""
    from benchmark.evaluation.process import hit, miss
    actions = (run.logs or {}).get("actions") or []
    states = (run.logs or {}).get("state_log") or []
    hits = {}
    first = next((a for a in actions if a.get("event") == "placement" and str(a.get("to", "")).startswith("board:")),
                 None)
    hits["E1"] = (hit(first["tick"], f"{first['block']} seated at {first['to']}") if first
                  else miss("no block seated on the board"))
    e2 = next((s for s in states if _rect_ready(s)), None)
    hits["E2"] = (hit(e2["tick"], f"source ON with {e2['output']['caps_across']} and "
                                  f"{e2['output']['load_ohm']:.0f} ohm across the output") if e2
                  else miss("never the source ON with a capacitor and a load across the output"))
    records = _rect_records(run)
    hits["M1"] = (hit(records[0][0], f"{records[0][1]}: the output with {list(records[0][2])}") if records
                  else miss("no logger record of the output with a capacitor and a load"))
    from benchmark.evaluation.rectifier import MEAN_MIN_V, RIPPLE_MAX_V
    device = run.device or {}
    hz = run.physics_hz
    source_hz = float(device.get("source_hz") or 50.0)
    final = float(device.get("final_load_ohm", 820.0))            # as the gate rectifier_grid_protocol
    tol = float(device.get("final_load_tolerance", 0.03))
    state_log = (run.logs or {}).get("state_log") or []
    _, changes = run_view(run)
    sides, final_ok, done = {m: set() for m in RECT_MODULES}, None, None
    recorded, recorded_final = set(), None                       # (b): valid steady segments
    for eid, record, rows, end, flags in _records(run):
        states = state_log or record.get("states") or []
        rate = float(record.get("sample_hz") or record.get("rate_hz") or 0.0)
        if not rate and len(rows) >= 2 and rows[1][0] > rows[0][0]:
            rate = 1.0 / (rows[1][0] - rows[0][0])
        if not states or not rate:
            continue
        period = round(rate / source_hz)
        for cap, load, v in _rect_segments(rows, states, hz):
            measured = _rect_steady(v, period)
            if measured is None or cap not in sides:
                continue
            meets = measured[0] >= MEAN_MIN_V and measured[1] <= RIPPLE_MAX_V
            sides[cap].add(meets)
            if meets and abs(load / final - 1) <= tol:
                final_ok = final_ok or (cap, load)
        for cap, load, v in _rect_segments(rows, states, hz, changes):
            measured = _rect_steady(v, period)
            if measured is None or cap not in sides:
                continue
            recorded.add(cap)
            if measured[0] >= MEAN_MIN_V and measured[1] <= RIPPLE_MAX_V and abs(load / final - 1) <= tol:
                recorded_final = recorded_final or (cap, load)
        if all(len(s) == 2 for s in sides.values()) and final_ok:
            done = hit(end, f"each module steady on both sides of its limit; final {final_ok[0]} at "
                            f"{final_ok[1]:.0f} ohm meets the requirement (by {eid})", flags)
            break
        if recorded >= set(RECT_MODULES) and recorded_final:
            done = hit(end, f"each module recorded steadily; final {recorded_final[0]} at {recorded_final[1]:.0f} ohm "
                            f"meets the requirement (by {eid})", flags)
            break
    covered = {m: sorted("meets" if s else "fails" for s in sides[m]) for m in RECT_MODULES}
    missing = [m for m in RECT_MODULES if m not in recorded]
    hits["M2"] = done or miss(f"steady segments per module {covered}; need every module recorded steadily and validly "
                              f"(missing: {missing or 'none'}) or on both sides of its limit (meets and fails), and "
                              f"the final configuration at {final:g} ohm "
                              f"{'recorded' if (recorded_final or final_ok) else 'not recorded meeting the requirement'}")
    return hits


def rect_final_record(evidence, eid, physics_hz):
    """(record, ok sample rows, bench states in force during the samples) of the logger record cited as
    final_configuration.evidence_id, or (None, [], []) when it is not a finished logger record with states."""
    record = (evidence or {}).get(eid) if isinstance(eid, str) else None
    if not record or record.get("kind") != "circuit_logger" or not isinstance(record.get("states"), list):
        return None, [], []
    rows = [r for r in logger_samples(record) if r[1] == "ok"]
    if not rows:
        return record, [], []
    return record, rows, window_states(record["states"], round(rows[0][0] * physics_hz), round(rows[-1][0] * physics_hz))


def rectifier_grid_protocol(ctx, settle_s=1.0, min_steady_s=0.2, min_rate_khz=8):
    """The final supply is shown by the logger record cited as final_configuration.evidence_id: a finished record of this episode, nothing changed on the board during it (one bench state, an
    ammeter range change included; changes off the board, e.g. in the tray, do not count: circuit_grid.py), and in that state the source ON, exactly one capacitor block on the board, seated
    across the output and the one named in final_configuration.module_id, the output terminals not joined, the
    resistance across the output (capacitors out, meters in) within the task tolerance of the task load, and a logger
    channel across the output; recorded at >= min_rate_khz; the samples taken >= settle_s after that state began
    last >= min_steady_s, and on a channel across the output their mean and peak-to-peak meet the supply requirements.
    final_configuration.load_ohm is the task load, and the cited evidence is this episode's (logger records or
    observations). The device's post-submission check (validation_problem, final_problems, post_submit) is not
    used: the bench is tidied before the submission (circuit_bench_restored)."""
    from benchmark.evaluation.gates import _need_submission      # the gate library imports this module
    from benchmark.evaluation.rectifier import MEAN_MIN_V, RIPPLE_MAX_V
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    device = ctx.sealed.episode.get("device") or {}
    hz = float(ctx.sealed.episode["physics_hz"])
    load_ohm = float(device.get("final_load_ohm", 820.0))
    tol = float(device.get("final_load_tolerance", 0.03))
    problems, detail = [], {}
    for eid in ctx.submission.get("evidence_ids") or []:
        record, problem = ctx.evidence(eid)
        if problem or record.get("kind") not in ("circuit_logger", "screen_capture"):
            problems.append(problem or f"{eid}: not a logger record or an observation")
    config = ctx.submission.get("final_configuration") or {}
    module = config.get("module_id")
    declared = config.get("load_ohm")
    if not isinstance(declared, (int, float)) or isinstance(declared, bool) or \
            abs(float(declared) / load_ohm - 1) > tol:
        problems.append("final_configuration.load_ohm is not the task load")
    eid = config.get("evidence_id")
    record, problem = ctx.evidence(eid)
    if problem or record.get("kind") != "circuit_logger" or not isinstance(record.get("states"), list):
        problems.append(f"final record: {problem or f'{eid}: not a logger record'}")
        return not problems, {"problems": problems}
    rate = float(record.get("sample_hz") or 0)
    if rate < min_rate_khz * 1000 - 1e-9:
        problems.append(f"final record at {rate / 1000:g} kHz, below {min_rate_khz} kHz")
    _, rows, found = rect_final_record(ctx.sealed.evidence, eid, hz)
    if not rows:
        problems.append("final record has no samples")
        return not problems, {"problems": problems}
    found = BenchChanges(process_state_log(ctx.sealed)).snapshots(found)     # only board changes count
    if not found or len({s["change_tick"] for s in found}) != 1:
        problems.append("the board changed during the final record")
        return not problems, {"problems": problems}
    state = found[0]
    out = state.get("output") or {}
    if not (state.get("supply") or {}).get("on"):
        problems.append("the source was OFF during the final record")
    if out.get("shorted"):
        problems.append("the output terminals were joined")
    if out.get("caps_on_board") != [module]:
        problems.append(f"capacitor blocks on the board: {out.get('caps_on_board')}, declared {module}")
    elif out.get("caps_across") != [module]:
        problems.append(f"{module} was not seated across the output")
    load = out.get("load_ohm")
    if load is None or abs(float(load) / load_ohm - 1) > tol:
        problems.append(f"load across the output {load} ohm, not within {tol:.0%} of {load_ohm:g} ohm")
    channels = [k for k in CHANNELS if ((state.get("instruments") or {}).get(k) or {}).get("across_output")]
    if not channels:
        problems.append("no logger channel across the output")
    begin = state["change_tick"] / hz + settle_s
    steady = [r for r in rows if r[0] >= begin - 1e-9]
    length = len(steady) / rate if rate else 0.0
    detail.update(evidence_id=eid, sample_hz=rate, steady_s=round(length, 6), channels=channels)
    if length < min_steady_s - 1e-9:
        problems.append(f"{length:.3f} s of the final record lies {settle_s} s or more after the last change; "
                        f"need {min_steady_s} s")
    elif channels:
        import numpy as np
        best = None
        for name in channels:
            v = np.array([r[2] if name == CHANNELS[0] else r[3] for r in steady])
            mean, ripple = float(v.mean()), float(np.ptp(v))
            detail[name] = {"mean_voltage_V": mean, "ripple_pp_V": ripple}
            ok = mean >= MEAN_MIN_V and ripple <= RIPPLE_MAX_V
            best = best or ok
        if not best:
            problems.append(f"final supply {detail} does not meet {MEAN_MIN_V} V / {RIPPLE_MAX_V} V")
    return not problems, {"problems": problems, **detail}


PROCESS_RULES = {"battery_emf": battery_emf, "rc_capacitance": rc_capacitance, "circuit_blackbox3": circuit_blackbox3,
                 "led_planck": led_planck, "rectifier_supply": rectifier_supply}

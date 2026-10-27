"""Common gate library with mechanics-kit extensions.

Each gate is `gate(ctx, **params) -> (passed, detail)`. Gates read only the
sealed episode, the submission and the instance truth held in `ctx`; they never
call the runtime. Physical conditions come from the Bench's private audit that
was sealed with each evidence record, never from Agent-reported values.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import jsonschema

from benchmark.registry import expressions

GATES: dict = {}


def gate(name):
    def register(function):
        GATES[name] = function
        return function
    return register


@dataclass
class Context:
    spec: object              # benchmark.registry.loader.ExperimentSpec
    hidden: dict              # instance truth (trusted)
    sealed: object            # benchmark.evaluation.sealed.Sealed
    submission_valid: bool = False

    @property
    def submission(self):
        return self.sealed.submission

    def param(self, value):
        """Gate parameters may be expressions over hidden values, e.g. "n_weights+1"."""
        return expressions.evaluate(value, self.hidden)

    def evidence(self, evidence_id):
        """Return (record, problem). Rejects unknown, foreign-episode and tampered evidence."""
        if not isinstance(evidence_id, str):
            return None, "evidence_id is not a string"
        if not evidence_id.startswith(self.sealed.episode_id + ":"):
            return None, f"{evidence_id}: not from this episode"
        record = self.sealed.evidence.get(evidence_id)
        if record is None:
            return None, f"{evidence_id}: unknown evidence"
        if not self.sealed.payload_sha_ok(record):
            return None, f"{evidence_id}: payload hash mismatch"
        return record, None


def _finite_numbers(value) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_numbers(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite_numbers(v) for v in value)
    return True


def _need_submission(ctx):
    if not ctx.submission_valid:
        return False, {"reason": "submission is not valid; gate cannot be assessed"}
    return None


# --- submission & budget -------------------------------------------------------
@gate("valid_submission")
def valid_submission(ctx: Context, ascending=()):
    episode = ctx.sealed.episode
    if episode.get("submission_attempts") != 1:
        return False, {"reason": f"expected exactly one submission, saw {episode.get('submission_attempts')}"}
    try:
        jsonschema.validate(ctx.submission, ctx.spec.submission_schema())
    except jsonschema.ValidationError as exc:
        return False, {"reason": f"schema: {exc.message}", "path": list(exc.absolute_path)}
    if not _finite_numbers(ctx.submission):
        return False, {"reason": "non-finite number in submission"}
    for field in ascending:
        values = ctx.submission[field]
        if any(b <= a for a, b in zip(values, values[1:])):
            return False, {"reason": f"{field} must be strictly ascending"}
    return True, {}


@gate("within_budget")
def within_budget(ctx: Context):
    episode = ctx.sealed.episode
    budget_ticks = round(ctx.spec.budget["sim_s"] * episode["physics_hz"])
    used = episode["submit_tick"]
    ok = used <= min(budget_ticks, episode["horizon_ticks"])
    detail = {"sim_s_used": used / episode["physics_hz"], "sim_s_budget": ctx.spec.budget["sim_s"]}
    if ctx.spec.id == "circuit_blackbox3" and episode.get("runtime") == "circuit":   # the old panel
        wall = episode.get("wall_s_used")
        arms = episode.get("logger_arms_used")
        ok = (ok and type(wall) in (int, float) and math.isfinite(wall)
              and 0 <= wall <= ctx.spec.budget["wall_s"]
              and type(arms) is int and 0 <= arms <= ctx.spec.raw["consumables"]["logger_arms"])
        detail.update(wall_s_used=wall, logger_arms_used=arms)
    return ok, detail


# --- apparatus terminal state (mechanics kit) --------------------------------
@gate("all_weights_on_tray")
def all_weights_on_tray(ctx: Context):
    state = ctx.sealed.episode["terminal_state"]
    missing = sorted(set(state["weight_ids"]) - set(state["supported"]))
    return not missing, {"not_on_tray": missing}


@gate("weights_in_box")
def weights_in_box(ctx: Context):
    """Every weight seated in some seat of the weight box (any free seat; terminal_state weight_box_seats, by the
    public threshold: within 20 mm horizontally and vertically of a seated weight, tilted at most 10 degrees, not held)."""
    state = ctx.sealed.episode["terminal_state"]
    seats = state.get("weight_box_seats") or {}
    missing = sorted(set(state["weight_ids"]) - set(seats.values()))
    return not missing, {"not_in_box": missing, "seats": seats}


@gate("encoder_bench_restored")
def encoder_bench_restored(ctx: Context):
    """hooke_dynamic_k tray-encoder bench: at submission the recorder is not recording and the damper lever is at OFF, as the Bench
    sealed it (terminal_state damper and recorder, HookeBackend.bench_state)."""
    state = ctx.sealed.episode.get("terminal_state") or {}
    damper, recorder = state.get("damper"), state.get("recorder")
    if not (isinstance(damper, dict) and isinstance(recorder, dict)):
        return False, {"problems": ["no record of the recorder panel and damper lever at submission"]}
    problems = []
    if damper.get("state") != "OFF":
        problems.append(f"damper lever {damper.get('state')}, not OFF")
    if recorder.get("ready") or recorder.get("recording"):
        problems.append(f"recorder {recorder.get('state')}")
    return not problems, {"problems": problems, "damper": damper, "recorder": recorder}


@gate("released")
def released(ctx: Context):
    return bool(ctx.sealed.episode["terminal_state"]["released"]), {}


@gate("stable")
def stable(ctx: Context):
    return bool(ctx.sealed.episode["terminal_state"]["stable"]), {}


@gate("gate_timers_reset")
def gate_timers_reset(ctx: Context):
    """Pendulum bench: at submission every gate timer is idle (reset, display at zero), as the Bench sealed it."""
    timers = (ctx.sealed.episode.get("terminal_state") or {}).get("gate_timers")
    if not isinstance(timers, dict) or not timers:
        return False, {"problems": ["no record of the gate timers at submission"]}
    not_reset = {label: status for label, status in sorted(timers.items()) if status != "idle"}
    return not not_reset, {"not_reset": not_reset}


@gate("coupled_bench_restored")
def coupled_bench_restored(ctx: Context, position_tol_mm=2.0, max_speed_mm_s=5.0):
    """Coupled bench: at submission both catches are clamped at their zero marks, both sliders hooked and at rest at
    their encoder zero (within position_tol_mm, at most max_speed_mm_s), the damper lever at OFF, the recorder
    neither READY (REC keyed, record not yet started) nor recording and the robot holding nothing (as on airtrack_collision), as the
    Bench sealed it (terminal_state of CoupledBackend.bench_state). Records sealed before the hand check existed
    (older records) carry no `released` field and are not checked for it."""
    state = ctx.sealed.episode.get("terminal_state") or {}
    catches, sliders = state.get("catches"), state.get("sliders")
    damper, recorder = state.get("damper"), state.get("recorder")
    if not (isinstance(catches, list) and isinstance(sliders, list) and len(catches) == len(sliders) == 2
            and isinstance(damper, dict) and isinstance(recorder, dict)):
        return False, {"problems": ["no record of the coupled bench state at submission"]}
    tol, speed = float(ctx.param(position_tol_mm)), float(ctx.param(max_speed_mm_s))
    problems = []
    for name, catch in zip(("CH1", "CH2"), catches):
        if not catch.get("clamped"):
            problems.append(f"{name} catch not clamped")
        if not abs(catch.get("offset_mm", math.inf)) <= tol:
            problems.append(f"{name} catch {catch.get('offset_mm')} mm from its zero mark")
    for name, slider in zip(("CH1", "CH2"), sliders):
        if not slider.get("hooked"):
            problems.append(f"{name} slider not hooked")
        if not abs(slider.get("offset_mm", math.inf)) <= tol:
            problems.append(f"{name} slider {slider.get('offset_mm')} mm from its encoder zero")
        if not abs(slider.get("speed_mm_s", math.inf)) <= speed:
            problems.append(f"{name} slider moving at {slider.get('speed_mm_s')} mm/s")
    if damper.get("state") != "OFF":
        problems.append(f"damper lever {damper.get('state')}, not OFF")
    if recorder.get("ready") or recorder.get("recording"):
        problems.append(f"recorder {recorder.get('state')}")
    if state.get("released") is False:
        problems.append("the robot still holds a catch, the release post or the damper lever")
    return not problems, {"problems": problems, "terminal_state": state}


@gate("airtrack_bench_restored")
def airtrack_bench_restored(ctx: Context, position_tol_mm=1.0, max_speed_mm_s=5.0):
    """airtrack_collision air track: at submission both gliders are held in their start catches (on their marks within
    position_tol_mm, at most max_speed_mm_s), the launcher is uncocked, the release lever is at HOLD, the recorder is
    neither READY nor recording and the robot holds nothing, as the Bench sealed it (AirtrackBackend.bench_state)."""
    state = ctx.sealed.episode.get("terminal_state") or {}
    gliders, launcher = state.get("gliders"), state.get("launcher")
    lever, recorder = state.get("lever"), state.get("recorder")
    if not (isinstance(gliders, list) and len(gliders) == 2 and isinstance(launcher, dict)
            and isinstance(lever, dict) and isinstance(recorder, dict)):
        return False, {"problems": ["no record of the air-track bench state at submission"]}
    tol, speed = float(ctx.param(position_tol_mm)), float(ctx.param(max_speed_mm_s))
    problems = []
    for glider in gliders:
        name = glider.get("glider")
        if not glider.get("caught"):
            problems.append(f"glider {name} not held in its start catch")
        if not abs(glider.get("offset_mm", math.inf)) <= tol:
            problems.append(f"glider {name} {glider.get('offset_mm')} mm from its start mark")
        if not abs(glider.get("speed_mm_s", math.inf)) <= speed:
            problems.append(f"glider {name} moving at {glider.get('speed_mm_s')} mm/s")
    if launcher.get("cocked"):
        problems.append(f"launcher cocked (notch {launcher.get('notch')})")
    if lever.get("state") != "HOLD":
        problems.append(f"release lever {lever.get('state')}, not HOLD")
    if recorder.get("ready") or recorder.get("recording"):
        problems.append(f"recorder {recorder.get('state')}")
    if not state.get("released"):
        problems.append("the robot still holds a handle or the lever post")
    return not problems, {"problems": problems, "terminal_state": state}


# --- evidence ----------------------------------------------------------------------
@gate("measurement_evidence_valid")
def measurement_evidence_valid(ctx: Context, instrument, min_points, require=(), **params):
    blocked = _need_submission(ctx)
    if blocked:
        return blocked
    checker = INSTRUMENT_CHECKS[instrument]
    return checker(ctx, int(ctx.param(min_points)), set(require), **params)


def _ruler_image(ctx: Context, min_points, require, min_loads=2):
    # distinct_loads: the used_in_fit readings cover >= min_loads different loads (hooke_k: 4)
    rows = ctx.submission["measurements"]
    problems, seen, used = [], set(), []
    mass = ctx.sealed.episode["device"]["weight_mass_kg"]
    for i, row in enumerate(rows):
        record, problem = ctx.evidence(row["evidence_id"])
        if problem or record.get("kind") != "ruler_image":
            problems.append(f"row {i}: {problem or 'not ruler evidence'}")
            continue
        if row["evidence_id"] in seen:
            problems.append(f"row {i}: duplicate evidence {row['evidence_id']}")
        seen.add(row["evidence_id"])
        if not row["used_in_fit"]:
            continue
        load = len(record["supported"])
        if "stable" in require and not record["stable"]:
            problems.append(f"row {i}: tray not at rest when observed")
        # no_contact (six levels): over the last 1 s the robot touched neither the tray nor a weight on it
        # and no held weight pressed on them (`tray_free`, recorded); a weight held clear of the tray
        # does not matter. Older records: `released` (no weight held or touched at all).
        if "no_contact" in require and not record.get("tray_free", record["released"]):
            problems.append(f"row {i}: robot touched the tray or a weight on it when observed")
        if "load_matches" in require and abs(row["mass_kg"] - load * mass) > 1e-6:
            problems.append(f"row {i}: mass_kg {row['mass_kg']} but {load} weight(s) on tray")
        used.append(load)
    n = ctx.hidden.get("n_weights")
    if "loads_cover_0_to_n" in require:
        missing = sorted(set(range(n + 1)) - set(used))
        if missing:
            problems.append(f"no used_in_fit reading with {missing} weight(s) on the tray")
    min_loads = int(ctx.param(min_loads))
    if "distinct_loads" in require and len(set(used)) < min_loads:
        problems.append(f"used_in_fit readings cover {len(set(used))} load(s); at least {min_loads} different loads needed")
    if len(used) < min_points:
        problems.append(f"{len(used)} used_in_fit readings < {min_points}")
    return not problems, {"problems": problems, "used_points": len(used)}


def _optical_gate(ctx: Context, min_points, require, distinct=None, period_tol=0.005):
    rows = ctx.submission["measurements"]
    problems, pendulums = [], []
    lengths = ctx.sealed.episode["device"]["effective_lengths_m"]
    if len(rows) != min_points:
        problems.append(f"{len(rows)} measurements, expected {min_points}")
    if len({r["evidence_id"] for r in rows}) != len(rows):
        problems.append("the same evidence is cited more than once")
    for i, row in enumerate(rows):
        record, problem = ctx.evidence(row["evidence_id"])
        if problem or record.get("kind") != "optical_gate":
            problems.append(f"row {i}: {problem or 'not optical gate evidence'}")
            continue
        public, reasons = record["public"], set(record["invalid_reasons"])
        actual = record["pendulum_label"]          # P label of the pendulum that physically blocked this gate
        pendulums.append(actual)
        if "complete" in require and public["status"] != "complete":
            problems.append(f"row {i}: acquisition status {public['status']}")
        if "cycles_in_8_10" in require and not 8 <= public["cycles"] <= 10:
            problems.append(f"row {i}: {public['cycles']} cycles")
        if "amplitude_lt_5_deg" in require and "amplitude_not_below_5_degrees" in reasons:
            problems.append(f"row {i}: amplitude reached 5 degrees")
        if "no_contact_during_timing" in require and "contact_during_timing" in reasons:
            problems.append(f"row {i}: contact during timing")
        other = reasons - {"amplitude_not_below_5_degrees", "contact_during_timing"}
        if other:
            problems.append(f"row {i}: instrument invalid ({', '.join(sorted(other))})")
        if "excited_by_robot" in require and not record["contact_excitation"]:
            problems.append(f"row {i}: pendulum was not set swinging by the robot")
        if "identity_matches" in require and row["pendulum_id"] != actual:
            problems.append(f"row {i}: labelled {row['pendulum_id']} but evidence is from {actual}")
        if "length_matches" in require and abs(row["L_eff_m"] - lengths[actual]) > 1e-4:
            problems.append(f"row {i}: L_eff_m {row['L_eff_m']} differs from {actual} ({lengths[actual]})")
        if "period_matches_evidence" in require and public["cycles"]:
            period = public["elapsed_s"] / public["cycles"]
            if abs(row["period_s"] - period) / period > period_tol:
                problems.append(f"row {i}: period_s {row['period_s']} differs from evidence {period:.5f}")
    if distinct == "pendulum" and len(set(pendulums)) != min_points:
        problems.append(f"evidence covers pendulums {sorted(set(pendulums))}, need {min_points} distinct")
    return not problems, {"problems": problems, "pendulums": pendulums}


def _timer_display(ctx: Context, min_points, require, distinct=None, period_tol=0.005, time_tol=0.001):
    """pendulum_g (no export_gate): the timers are read on their faces in the camera images.

    Each measurement cites an observation of this episode (get_observation's evidence_id, kind bench_view) at which
    the timer of its pendulum showed DONE (shown_done); its time_s equals the TIME that face showed within time_tol
    (one unit of the last digit) and its cycles the CYCLES shown (reading_matches_display); period_s agrees with
    them within period_tol (period_matches_evidence). The acquisition that face showed, the Bench's private record
    of it (kind optical_gate), must be complete with 8-10 cycles, the swing below 5 degrees from START to the end,
    started by the robot, with no contact during timing and no instrument fault, as before."""
    rows = ctx.submission["measurements"]
    problems, pendulums, detail = [], [], []
    lengths = ctx.sealed.episode["device"]["effective_lengths_m"]
    tol = float(ctx.param(time_tol)) + 1e-9
    # six levels: at least min_points rows (was exactly), repeats allowed; distinct="pendulum":
    # the rows cover min_points different pendulums (pendulum_g: P1, P2 and P3)
    if len(rows) < min_points:
        problems.append(f"{len(rows)} measurements, expected at least {min_points}")
    for i, row in enumerate(rows):
        label, eid = row["pendulum_id"], row["evidence_id"]
        record, problem = ctx.evidence(eid)
        if problem is None and record.get("kind") != "bench_view":
            problem = f"{eid}: not an observation of this bench"
        if problem:
            problems.append(f"row {i}: {problem}")
            continue
        face = (record.get("timers") or {}).get(label)
        if not isinstance(face, dict):
            problems.append(f"row {i}: no record of the {label} timer at {eid}")
            continue
        pendulums.append(label)
        entry = {"evidence_id": eid, "pendulum_id": label, "time_s": row["time_s"], "cycles": row["cycles"],
                 "shown": {k: face.get(k) for k in ("word", "time_s", "cycles")}, "acquisition": face.get("acquisition")}
        detail.append(entry)
        if "shown_done" in require and face.get("word") != "DONE":
            problems.append(f"row {i}: the {label} timer showed {face.get('word')} at {eid}, not DONE")
            continue
        shown_t, shown_n = face.get("time_s"), face.get("cycles")
        if not isinstance(shown_t, (int, float)) or not isinstance(shown_n, int):
            problems.append(f"row {i}: the {label} timer showed no reading at {eid}")
            continue
        if "reading_matches_display" in require:
            if not abs(row["time_s"] - shown_t) <= tol:
                problems.append(f"row {i}: time_s {row['time_s']} s, the {label} timer showed {shown_t:.3f} s at {eid}")
            if row["cycles"] != shown_n:
                problems.append(f"row {i}: cycles {row['cycles']}, the {label} timer showed {shown_n:02d} at {eid}")
        if "period_matches_evidence" in require and shown_n:
            period = shown_t / shown_n
            if abs(row["period_s"] - period) / period > period_tol:
                problems.append(f"row {i}: period_s {row['period_s']} differs from the reading {period:.5f}")
        gate, problem = ctx.evidence(face.get("acquisition"))
        if problem is None and gate.get("kind") != "optical_gate":
            problem = "not a gate acquisition"
        if problem:
            problems.append(f"row {i}: no intact record of the acquisition the {label} timer showed ({problem})")
            continue
        public, reasons = gate["public"], set(gate["invalid_reasons"])
        actual = gate["pendulum_label"]
        if "identity_matches" in require and actual != label:
            problems.append(f"row {i}: labelled {label} but the acquisition is from {actual}")
        if "complete" in require and public["status"] != "complete":
            problems.append(f"row {i}: acquisition status {public['status']}")
        if "cycles_in_8_10" in require and not 8 <= public["cycles"] <= 10:
            problems.append(f"row {i}: {public['cycles']} cycles")
        if "amplitude_lt_5_deg" in require and "amplitude_not_below_5_degrees" in reasons:
            problems.append(f"row {i}: amplitude reached 5 degrees")
        if "no_contact_during_timing" in require and "contact_during_timing" in reasons:
            problems.append(f"row {i}: contact during timing")
        other = reasons - {"amplitude_not_below_5_degrees", "contact_during_timing"}
        if other:
            problems.append(f"row {i}: instrument invalid ({', '.join(sorted(other))})")
        if "excited_by_robot" in require and not gate["contact_excitation"]:
            problems.append(f"row {i}: pendulum was not set swinging by the robot")
        if "length_matches" in require and abs(row["L_eff_m"] - lengths[label]) > 1e-4:
            problems.append(f"row {i}: L_eff_m {row['L_eff_m']} differs from {label} ({lengths[label]})")
    if distinct == "pendulum" and len(set(pendulums)) != min_points:
        problems.append(f"readings cover pendulums {sorted(set(pendulums))}, need {min_points} distinct")
    return not problems, {"problems": problems, "pendulums": pendulums, "rows": detail}


# Coupled recorder, six levels: a record counts by its clean stretches, not as a whole.
COUPLED_SEGMENT_REASONS = {"external_contact_during_recording", "brake_engaged_during_recording", "over_range_or_nonfinite"}


def coupled_clean_mask(record):
    """Per-sample clean flags of a coupled recording (both sliders unhooked, no external contact, lever off, in range):
    `sample_status` == 'ok' (sealed); older records: samples outside every validity_diagnostics
    interval [first_t_s, last_t_s] widened by one sample (what happened in between is not recorded)."""
    rows = record["rows"]
    status = record.get("sample_status")
    present = [r["x1_mm"] is not None and r["x2_mm"] is not None for r in rows]
    if status is not None and len(status) == len(rows):
        return [ok and s == "ok" for ok, s in zip(present, status)]
    margin = 1.0 / float(record.get("sample_hz") or 120)
    diagnostics = record.get("validity_diagnostics") or {}
    if any(not isinstance(diagnostics.get(reason), dict) for reason in
           set(record.get("invalid_reasons") or ()) & COUPLED_SEGMENT_REASONS):
        return [False] * len(rows)          # a reason without its interval: nothing is known to be clean
    spans = [(d["first_t_s"] - margin, d["last_t_s"] + margin) for d in diagnostics.values()
             if isinstance(d, dict) and "first_t_s" in d]
    return [ok and not any(a <= r["t_s"] <= b for a, b in spans) for ok, r in zip(present, rows)]


def coupled_segments(record, mask=None, min_s=40.0, min_pp_mm=1.0):
    """Clean segments [(t, x1, x2)] (runs of consecutive clean samples) lasting >= min_s with both channels moving
    >= min_pp_mm peak-to-peak within the segment. mask: per-sample clean flags (default coupled_clean_mask)."""
    rows = record["rows"]
    mask = coupled_clean_mask(record) if mask is None else mask
    out, run = [], []
    for row, ok in zip(rows + [None], list(mask) + [False]):
        if ok and row is not None:
            run.append(row)
            continue
        if run and run[-1]["t_s"] - run[0]["t_s"] >= min_s - 1e-6:
            x1, x2 = [r["x1_mm"] for r in run], [r["x2_mm"] for r in run]
            if max(x1) - min(x1) >= min_pp_mm and max(x2) - min(x2) >= min_pp_mm:
                out.append(([r["t_s"] for r in run], x1, x2))
        run = []
    return out


def coupled_axes(x1, x2):
    """Principal-component axes of a segment's (x1, x2) motion: (major unit vector, peak-to-peak along the minor axis)."""
    m1, m2 = sum(x1) / len(x1), sum(x2) / len(x2)
    a = sum((u - m1) ** 2 for u in x1)
    b = sum((v - m2) ** 2 for v in x2)
    c = sum((u - m1) * (v - m2) for u, v in zip(x1, x2))
    theta = 0.5 * math.atan2(2 * c, a - b)
    major = (math.cos(theta), math.sin(theta))
    minor = [-(u - m1) * major[1] + (v - m2) * major[0] for u, v in zip(x1, x2)]
    return major, max(minor) - min(minor)


def coupled_modes_determined(segments_by_record, min_minor_pp_mm=0.5, min_abs_det=0.3):
    """Both modes measurable L4/L5): (a) one segment with >= min_minor_pp_mm along its minor
    principal axis (both modes present), or (b) two segments from different records whose major axes are linearly
    independent (|det| >= min_abs_det). Returns (ok, detail)."""
    axes = []
    for eid, segments in segments_by_record:
        for t, x1, x2 in segments:
            major, minor_pp = coupled_axes(x1, x2)
            if minor_pp >= min_minor_pp_mm:
                return True, {"route": "two_modes_in_one_segment", "evidence_id": eid, "minor_axis_pp_mm": minor_pp}
            axes.append((eid, major))
    best = 0.0
    for (ea, a), (eb, b) in itertools.combinations(axes, 2):
        if ea != eb:
            best = max(best, abs(a[0] * b[1] - a[1] * b[0]))
    if best >= min_abs_det:
        return True, {"route": "independent_major_axes", "normalized_abs_det": best}
    return False, {"reason": "no clean segment shows both modes and no two records move along independent directions",
                   "segments": sum(len(s) for _, s in segments_by_record), "best_abs_det": best}


def _coupled_recorder(ctx: Context, min_points, require):
    ids = ctx.submission["measurement_evidence_ids"]
    problems = []
    if len(ids) < min_points or len(set(ids)) != len(ids):      # six levels: at least min_points (was exactly)
        problems.append(f"need {min_points} distinct recordings")
    segmented = "clean_segment_s_ge_40" in require
    for eid in ids:
        record, problem = ctx.evidence(eid)
        if problem or record.get("kind") != "coupled_recorder":
            problems.append(problem or f"{eid}: not a coupled recording")
            continue
        reasons = set(record["invalid_reasons"])
        if "complete" in require and record["status"] != "complete" and not (
                segmented and reasons <= COUPLED_SEGMENT_REASONS):
            problems.append(f"{eid}: status {record['status']}")
        if "no_hold" in require and "external_contact_during_recording" in reasons:
            problems.append(f"{eid}: contact or grip during recording")
        if "brakes_open" in require and "brake_engaged_during_recording" in reasons:
            problems.append(f"{eid}: a brake was engaged during recording")
        other = reasons - {"external_contact_during_recording", "brake_engaged_during_recording"}
        if segmented:
            other -= COUPLED_SEGMENT_REASONS
        if other:
            problems.append(f"{eid}: instrument invalid ({', '.join(sorted(other))})")
        rows = record["rows"]
        if segmented and not coupled_segments(record):
            problems.append(f"{eid}: no clean stretch of >= 40 s with both channels moving >= 1 mm")
        if "free_record_s_ge_40" in require and (not rows or rows[-1]["t_s"] < 40):
            problems.append(f"{eid}: shorter than 40 s")
        if "both_channels_pp_mm_ge_1" in require and rows:
            for key in ("x1_mm", "x2_mm"):
                values = [r[key] for r in rows if r[key] is not None]
                if not values or max(values) - min(values) < 1:
                    problems.append(f"{eid}: {key} moved less than 1 mm")
    return not problems, {"problems": problems}


def longest_ok_run(rows, status=None):
    """Longest run of consecutive `ok` samples as (t_s list, z_mm list). status: the per-sample status sealed in the
    private record (tray encoder `sample_status`); older records carry it in each row."""
    best, current = [], []
    for i, row in enumerate(rows):
        if (row["status"] if status is None else status[i]) == "ok" and row["z_mm"] is not None:
            current.append(row)
            if len(current) > len(best):
                best = current
        else:
            current = []
    return [r["t_s"] for r in best], [r["z_mm"] for r in best]


def encoder_free_run(record):
    """hooke_dynamic_k tray encoder: the longest run of consecutive `ok` samples with one load, as (t_s list, z_mm list, load ids).
    The load per sample is `sample_load_ids` (sealed); older records: the load at REC when it equals the
    load at the end, else None (unknown: the load changed somewhere in the recording)."""
    rows, status = record["rows"], record.get("sample_status")
    per_sample = record.get("sample_load_ids")
    if per_sample is None or len(per_sample) != len(rows):
        t, z = longest_ok_run(rows, status)
        at_arm, at_end = record.get("load_ids_at_arm"), record.get("load_ids_at_end")
        return t, z, (sorted(at_arm) if at_arm is not None and at_end == at_arm else None)
    best, current = ([], None), ([], None)
    for i, row in enumerate(rows):
        ok = (row["status"] if status is None else status[i]) == "ok" and row["z_mm"] is not None
        load = tuple(sorted(per_sample[i] or ()))
        if ok and current[0] and current[1] == load:
            current[0].append(row)
        elif ok:
            current = ([row], load)
        else:
            current = ([], None)
        if len(current[0]) > len(best[0]):
            best = current
    return [r["t_s"] for r in best[0]], [r["z_mm"] for r in best[0]], (list(best[1]) if best[0] else None)


def oscillation_period(t, z):
    """Evaluator's own period of a recorded oscillation: mean spacing of upward mean crossings.

    Returns (period_s, complete_periods, peak_to_peak_mm) or None when fewer than two crossings.
    """
    if len(t) < 4:
        return None
    mean = sum(z) / len(z)
    crossings = []
    for i in range(1, len(z)):
        a, b = z[i - 1] - mean, z[i] - mean
        if a < 0 <= b:
            crossings.append(t[i - 1] + (t[i] - t[i - 1]) * (-a) / (b - a))
    if len(crossings) < 2:
        return None
    period = (crossings[-1] - crossings[0]) / (len(crossings) - 1)
    return period, len(crossings) - 1, max(z) - min(z)


def _tray_encoder(ctx: Context, min_points, require, period_tol=0.01, min_pp_mm=1.0, min_loads=2):
    # distinct_loads: the used rows cover >= min_loads different loads (hooke_dynamic_k: 3)
    rows = [r for r in ctx.submission["measurements"] if r["used_in_fit"]]
    mass = ctx.sealed.episode["device"]["weight_mass_kg"]
    problems, loads, periods = [], set(), []
    if len(rows) < min_points:
        problems.append(f"{len(rows)} measurements used in the fit, need at least {min_points}")
    if len({r["evidence_id"] for r in rows}) != len(rows):
        problems.append("the same recording is cited more than once")
    for i, row in enumerate(rows):
        record, problem = ctx.evidence(row["evidence_id"])
        if problem or record.get("kind") != "tray_encoder":
            problems.append(f"row {i}: {problem or 'not tray encoder evidence'}")
            continue
        if record["status"] != "complete":
            problems.append(f"row {i}: acquisition status {record['status']}")
        # Six levels: the load is the set of weights on the tray throughout the free run the period is
        # measured on (per-sample `sample_load_ids`); a change at other times of the recording does not matter.
        t, z, load = encoder_free_run(record)
        if "load_matches" in require:
            if load is None:
                problems.append(f"row {i}: the load changed during the recording")
            elif abs(row["load_kg"] - len(load) * mass) > 1e-6:
                problems.append(f"row {i}: load_kg {row['load_kg']} but {len(load)} weights rested on the tray")
        loads.add(round(len(load if load is not None else record["load_ids_at_arm"]) * mass, 6))
        measured = oscillation_period(t, z)
        if measured is None:
            problems.append(f"row {i}: no oscillation in the longest run of ok samples")
            continue
        period, _, pp = measured
        periods.append(round(period, 6))
        span = t[-1] - t[0]
        if "ok_cycles_ge_5" in require:
            if span < 5 * period:
                problems.append(f"row {i}: {span:.2f} s of consecutive ok samples is fewer than 5 periods")
            if pp < min_pp_mm:
                problems.append(f"row {i}: peak-to-peak {pp:.1f} mm below {min_pp_mm} mm")
        if "period_matches_evidence" in require and abs(row["period_s"] - period) / period > period_tol:
            problems.append(f"row {i}: period_s {row['period_s']} differs from the recording ({period:.5f} s)")
    min_loads = int(ctx.param(min_loads))
    if "distinct_loads" in require and len(loads) < min_loads:
        problems.append(f"measurements cover {len(loads)} different load(s), fewer than {min_loads}")
    return not problems, {"problems": problems, "loads_kg": sorted(loads), "evidence_periods_s": periods}


TORSION_LOADS = {"empty": (), "standard_disc": ("standard_disc",), "unknown_object": ("unknown_object",)}
TORSION_CLEAN_CONDITIONS = {"contact_off_turntable_object"}


def photogate_rows(record):
    """Transits parsed from the hash-checked CSV (the sealed payload), not from the convenience row list. Since the CSV holds only the times; each transit's bench condition is the private `transit_status` sealed
    with the evidence (older records: the CSV `status` column)."""
    import csv
    import io
    private = record.get("transit_status")
    rows = list(csv.DictReader(io.StringIO(record["csv"])))
    if "status" not in (rows[0] if rows else {}) and (not isinstance(private, list) or len(private) != len(rows)):
        private = ["unknown"] * len(rows)          # no condition known: no transit counts as clean
    conditions = record.get("transit_conditions")
    if isinstance(conditions, list) and isinstance(private, list) and len(conditions) == len(private) == len(rows):
        # Six levels: robot contact with an object away from the turntable does not make a transit unclean
        # (`transit_conditions`, sealed older records keep their transit_status).
        private = ["ok" if set(c) <= TORSION_CLEAN_CONDITIONS else (s if s != "ok" else "contact")
                   for c, s in zip(conditions, private)]
    return [{"transit_id": int(r["transit_id"]), "t_block_s": float(r["t_block_s"]),
             "t_clear_s": float(r["t_clear_s"]), "status": r["status"] if "status" in r else private[i]}
            for i, r in enumerate(rows)]


def photogate_period(rows):
    """Evaluator's own period from the longest run of consecutive `ok` transits.

    The flag crosses the beam twice per period, so transit midpoints two apart are one
    period apart whatever the rest offset. Returns (period_s, complete_periods) or None.
    """
    best, current = [], []
    for row in rows:
        if row["status"] == "ok":
            current.append((row["t_block_s"] + row["t_clear_s"]) / 2)
            if len(current) > len(best):
                best = current
        else:
            current = []
    if len(best) < 3:
        return None
    steps = [best[i + 2] - best[i] for i in range(len(best) - 2)]
    return sum(steps) / len(steps), (len(best) - 1) // 2


def _torsion_photogate(ctx: Context, min_points, require, period_tol=0.005, centring_mm=2.0,
                       repeat_tol=0.02, max_tilt_deg=2.0):
    rows = [r for r in ctx.submission["measurements"] if r["used_in_fit"]]
    problems, by_load, placement = [], {}, {}
    if len(rows) < min_points:
        problems.append(f"{len(rows)} measurements used, need at least {min_points}")
    if len({r["evidence_id"] for r in rows}) != len(rows):
        problems.append("the same recording is cited more than once")
    if "loads_complete" in require and set(r["load"] for r in rows) != set(TORSION_LOADS):
        problems.append("measurements must cover the empty turntable, the standard disc and the unknown object")
    for i, row in enumerate(rows):
        record, problem = ctx.evidence(row["evidence_id"])
        if problem or record.get("kind") != "torsion_photogate":
            problems.append(f"row {i}: {problem or 'not photogate evidence'}")
            continue
        if record["status"] != "complete":
            problems.append(f"row {i}: acquisition status {record['status']}")
        expected = set(TORSION_LOADS[row["load"]])
        at_arm, at_end = record["loads_at_arm"] or {}, record["loads_at_end"] or {}
        if "load_matches" in require and not (set(at_arm) == set(at_end) == expected):
            problems.append(f"row {i}: load {row['load']!r} but {sorted(at_arm)} rested on the turntable "
                            f"at the start and {sorted(at_end)} at the end")
        for key in expected:
            audits = [a for a in (at_arm.get(key), at_end.get(key)) if a is not None]
            if audits:      # placement accuracy is the Agent's own operating error: reported, gated only on request
                worst = placement.setdefault(row["load"], {}).setdefault(key, {"eccentricity_mm": 0.0, "tilt_deg": 0.0})
                for field in worst:
                    worst[field] = round(max([worst[field]] + [a[field] for a in audits]), 3)
            if "centred" in require and any(a["eccentricity_mm"] > centring_mm or a["tilt_deg"] > max_tilt_deg
                                            for a in audits):
                problems.append(f"row {i}: {key} not centred on the turntable axis")
        measured = photogate_period(photogate_rows(record))
        if measured is None:
            problems.append(f"row {i}: no oscillation in the longest run of ok transits")
            continue
        period, cycles = measured
        by_load.setdefault(row["load"], []).append(period)
        if "ok_cycles_ge_5" in require and cycles < 5:
            problems.append(f"row {i}: {cycles} complete periods of consecutive ok transits, need 5")
        if "period_matches_evidence" in require and abs(row["period_s"] - period) / period > period_tol:
            problems.append(f"row {i}: period_s {row['period_s']} differs from the recording ({period:.5f} s)")
    if "repeat_consistent" in require:
        for load, periods in by_load.items():
            if len(periods) > 1 and max(periods) / min(periods) - 1 > repeat_tol:
                problems.append(f"repeated {load} recordings differ by more than {repeat_tol:.0%}")
    return not problems, {"problems": problems,
                          "evidence_periods_s": {k: [round(p, 6) for p in v] for k, v in by_load.items()},
                          "placement": placement}


def logger_rows(record):
    """Samples parsed from the hash-checked CSV (the sealed payload), not from the convenience row list."""
    import csv
    import io
    return [{"t_s": float(r["t_s"]), "status": r["status"], "ch1_V": float(r["ch1_V"]), "ch2_V": float(r["ch2_V"]),
             "pair": r["pair"], "rs": r["rs"], "mode": r["mode"]}
            for r in csv.DictReader(io.StringIO(record["csv"]))]


def _fit_window(ctx, fit):
    """(rows of the cited window, record, problem) for one submitted fit."""
    record, problem = ctx.evidence(fit["evidence_id"])
    if problem or record.get("kind") != "circuit_logger":
        return None, None, problem or "not logger evidence"
    if record["status"] != "complete":
        return None, None, f"acquisition status {record['status']}"
    rows = [r for r in logger_rows(record) if fit["t0_s"] - 1e-6 <= r["t_s"] <= fit["t1_s"] + 1e-6]
    if len(rows) < 3:
        return None, None, "fewer than 3 samples in the window"
    return rows, record, None


def _circuit_logger(ctx: Context, min_points, require, pair="A-B", min_taus=2.0, min_samples_per_tau=10,
                    entries="fits", mode=None, min_window_s=1.0, max_slope_v_per_s=0.002, settle_s=0.2):
    """Cited logger windows (`fits` in rc_capacitance, `measurements` in resistor_blackbox). Each lies in one ok selector state with the
    required pair, the declared rs and mode (`mode` fixed by the spec, else the entry's own); rc_capacitance adds the time
    constant coverage, resistor_blackbox (`steady`) the window length, the settling time after the first sample of the cited
    run in its recording and the CH1 drift."""
    cited = [f for f in ctx.submission[entries] if f["used_in_fit"]]
    problems, windows = [], []
    if len(cited) < min_points:
        problems.append(f"{len(cited)} {entries} used, need at least {min_points}")
    taus = ctx.sealed.episode["device"].get("tau_ab_charge_s") or {}
    for i, entry in enumerate(cited):
        rows, record, problem = _fit_window(ctx, entry)
        if problem:
            problems.append(f"{entries} {i}: {problem}")
            continue
        rs = str(entry.get("rs_ohm", ctx.submission.get("rs_ohm")))
        want_mode = mode or entry.get("mode")
        states = {(r["status"], r["pair"], r["rs"], r["mode"]) for r in rows}
        duration = rows[-1]["t_s"] - rows[0]["t_s"]
        window = {"evidence_id": entry["evidence_id"], "samples": len(rows), "duration_s": round(duration, 6)}
        windows.append(window)
        if "single_segment" in require and (len(states) != 1 or next(iter(states))[0] != "ok"
                                            or next(iter(states))[3] != want_mode):
            problems.append(f"{entries} {i}: the window crosses a selector change, holds non-ok samples "
                            f"or is not in {want_mode}")
        if "pair_matches" in require and any(r["pair"] != pair for r in rows):
            problems.append(f"{entries} {i}: pair is not {pair}")
        if "rs_matches" in require and any(r["rs"] != rs for r in rows):
            problems.append(f"{entries} {i}: rs differs from the declared rs_ohm {rs}")
        if {"covers_tau", "samples_per_tau"} & require:
            tau = taus[rs]
            window |= {"duration_taus": round(duration / tau, 3), "samples_per_tau": round(record["sample_hz"] * tau, 2)}
            if "covers_tau" in require and duration < min_taus * tau:
                problems.append(f"{entries} {i}: window covers {duration / tau:.2f} time constants, need {min_taus}")
            if "samples_per_tau" in require and record["sample_hz"] * tau < min_samples_per_tau:
                problems.append(f"{entries} {i}: {record['sample_hz'] * tau:.1f} samples per time constant, "
                                f"need {min_samples_per_tau}")
        if "steady" in require:
            settled = _run_start_before(logger_rows(record), rows[0])
            slope = _line_fit([r["t_s"] for r in rows], [r["ch1_V"] for r in rows])[0]
            window |= {"settled_s": round(rows[0]["t_s"] - settled, 6), "ch1_slope_V_per_s": slope}
            if duration < min_window_s:
                problems.append(f"{entries} {i}: window lasts {duration:.3f} s, need {min_window_s}")
            if rows[0]["t_s"] - settled < settle_s - 1e-9:
                problems.append(f"{entries} {i}: window starts {rows[0]['t_s'] - settled:.3f} s into its run, "
                                f"need {settle_s}")
            if abs(slope) > max_slope_v_per_s:
                problems.append(f"{entries} {i}: CH1 drifts {slope:.4g} V/s, limit {max_slope_v_per_s}")
    return not problems, {"problems": problems, "windows": windows}


def _run_start_before(all_rows, first):
    """t_s of the first sample of the run of one ok selector state that contains `first` in its recording."""
    key = (first["status"], first["pair"], first["rs"], first["mode"])
    index = next(i for i, r in enumerate(all_rows) if abs(r["t_s"] - first["t_s"]) < 1e-9)
    while index > 0 and (all_rows[index - 1]["status"], all_rows[index - 1]["pair"], all_rows[index - 1]["rs"],
                         all_rows[index - 1]["mode"]) == key:
        index -= 1
    return all_rows[index]["t_s"]


INSTRUMENT_CHECKS = {"ruler_image": _ruler_image, "optical_gate": _optical_gate,
                     "coupled_recorder": _coupled_recorder, "tray_encoder": _tray_encoder,
                     "torsion_photogate": _torsion_photogate, "circuit_logger": _circuit_logger,
                     "timer_display": _timer_display}


@gate("independent_excitation")
def independent_excitation(ctx: Context, min_abs_det, min_minor_pp_mm=None):
    """coupled_modes. With min_minor_pp_mm (six levels): the cited records' clean segments (>= 40 s, both
    channels >= 1 mm) determine both modes (coupled_modes_determined: one segment with both modes, or two records along
    independent directions). Without it (before): exactly two records whose first samples are independent."""
    blocked = _need_submission(ctx)
    if blocked:
        return blocked
    if min_minor_pp_mm is not None:
        cited = []
        for eid in dict.fromkeys(ctx.submission["measurement_evidence_ids"]):
            record, problem = ctx.evidence(eid)
            if problem or record.get("kind") != "coupled_recorder":
                return False, {"reason": problem or f"{eid}: not a coupled recording"}
            cited.append((eid, coupled_segments(record)))
        return coupled_modes_determined(cited, float(min_minor_pp_mm), float(min_abs_det))
    vectors = []
    for eid in ctx.submission["measurement_evidence_ids"]:
        record, problem = ctx.evidence(eid)
        if problem or record.get("kind") != "coupled_recorder" or not record["rows"]:
            return False, {"reason": problem or f"{eid}: no usable recording"}
        first = record["rows"][0]
        if first["x1_mm"] is None or first["x2_mm"] is None:
            return False, {"reason": f"{eid}: first sample invalid"}
        vectors.append((first["x1_mm"], first["x2_mm"]))
    if len(vectors) != 2:
        return False, {"reason": "two recordings required"}
    (a1, a2), (b1, b2) = vectors
    norm = math.hypot(a1, a2) * math.hypot(b1, b2)
    if norm < 1:
        return False, {"reason": "initial displacements are too small", "initial_mm": vectors}
    det = abs(a1 * b2 - a2 * b1) / norm
    return det >= float(min_abs_det), {"normalized_abs_det": det, "initial_mm": vectors}


# --- self-consistency ---------------------------------------------------------------
def _line_fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        raise ValueError("refit needs at least two distinct abscissae")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys))
    return slope, intercept, (1 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def refit_linear_hooke(ctx: Context):
    """m g = k (x - x0): fit pointer position (m) against force (N); k = 1/|slope|."""
    g = ctx.sealed.episode["device"]["gravity_for_conversion_m_s2"]
    rows = [r for r in ctx.submission["measurements"] if r["used_in_fit"]]
    slope, intercept, r2 = _line_fit([r["mass_kg"] * g for r in rows], [r["reading_cm"] / 100 for r in rows])
    if slope == 0:
        raise ValueError("readings do not change with load")
    return 1 / abs(slope), {"fit_r2": r2}


def refit_linear_with_intercept_T2_vs_L(ctx: Context):
    rows = ctx.submission["measurements"]
    slope, intercept, r2 = _line_fit([r["L_eff_m"] for r in rows], [r["period_s"] ** 2 for r in rows])
    if slope <= 0:
        raise ValueError("T^2 does not increase with L")
    return 4 * math.pi ** 2 / slope, {"fit_r2": r2, "intercept_s2": intercept}


def refit_pendulum_g(ctx: Context):
    """pendulum_g (six levels): any reasonable reduction of the cited rows. With >= 2 different L_eff:
    the T^2-L fit with an intercept, the fit through the origin, or the mean of 4 pi^2 L / T^2 per row; with one L_eff:
    g = 4 pi^2 L / mean(T)^2. The candidate nearest the submitted value is returned."""
    rows = ctx.submission["measurements"]
    if not rows:
        raise ValueError("no measurements")
    lengths, periods = [r["L_eff_m"] for r in rows], [r["period_s"] for r in rows]
    if min(periods) <= 0:
        raise ValueError("non-positive period")
    candidates, detail = {}, {}
    if len(set(lengths)) >= 2:
        slope, intercept, r2 = _line_fit(lengths, [t ** 2 for t in periods])
        if slope > 0:
            candidates["intercept_fit"] = 4 * math.pi ** 2 / slope
            detail.update(fit_r2=r2, intercept_s2=intercept)
        sxy = sum(l * t ** 2 for l, t in zip(lengths, periods))
        candidates["origin_fit"] = 4 * math.pi ** 2 * sum(l * l for l in lengths) / sxy
        candidates["mean_per_row"] = sum(4 * math.pi ** 2 * l / t ** 2 for l, t in zip(lengths, periods)) / len(rows)
    else:
        mean_t = sum(periods) / len(periods)
        candidates["single_length"] = 4 * math.pi ** 2 * lengths[0] / mean_t ** 2
    submitted = ctx.submission["g_m_per_s2"]
    name, value = min(candidates.items(), key=lambda item: abs(submitted - item[1]) / item[1])
    return value, {"refit_method": name, "refit_candidates": candidates, **detail}


def refit_linear_with_intercept_T2_vs_m(ctx: Context):
    """T^2 = (4 pi^2 / k) m + c over the used_in_fit rows; the intercept absorbs tray and spring mass."""
    rows = [r for r in ctx.submission["measurements"] if r["used_in_fit"]]
    slope, intercept, r2 = _line_fit([r["load_kg"] for r in rows], [r["period_s"] ** 2 for r in rows])
    if slope <= 0:
        raise ValueError("T^2 does not increase with load")
    return 4 * math.pi ** 2 / slope, {"fit_r2": r2, "intercept_s2": intercept}


def refit_torsion_inertia_ratio(ctx: Context):
    """I_u = I_std (T2^2 - T0^2) / (T1^2 - T0^2) from the mean submitted period of each load."""
    rows = [r for r in ctx.submission["measurements"] if r["used_in_fit"]]
    mean = {}
    for load in TORSION_LOADS:
        periods = [r["period_s"] for r in rows if r["load"] == load]
        if not periods:
            raise ValueError(f"no {load} measurement")
        mean[load] = sum(periods) / len(periods)
    t0, t1, t2 = (mean[k] ** 2 for k in ("empty", "standard_disc", "unknown_object"))
    if t1 <= t0 or t2 <= t0:
        raise ValueError("loaded periods must exceed the empty period")
    standard = ctx.sealed.episode["device"]["standard_disc_inertia_kgm2"]
    return standard * (t2 - t0) / (t1 - t0), {"mean_periods_s": mean}


def step_time_constant(t, v):
    """Evaluator's own first-order fit V = V_inf - (V_inf - V0) exp(-(t - t0)/tau): golden-section search over
    log tau with V_inf and V0 solved linearly for each tau. Returns tau (s)."""
    import numpy as np
    t, v = np.asarray(t, float) - t[0], np.asarray(v, float)

    def cost(log_tau):
        e = np.exp(-t / np.exp(log_tau))
        design = np.c_[1 - e, e]
        coef, *_ = np.linalg.lstsq(design, v, rcond=None)
        return float(np.sum((design @ coef - v) ** 2))

    lo, hi = math.log(max(t[-1], 1e-3) / 200), math.log(max(t[-1], 1e-3) * 20)
    golden = (math.sqrt(5) - 1) / 2
    a, b = hi - golden * (hi - lo), lo + golden * (hi - lo)
    fa, fb = cost(a), cost(b)
    while hi - lo > 1e-10:
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - golden * (hi - lo)
            fa = cost(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + golden * (hi - lo)
            fb = cost(b)
    return math.exp((lo + hi) / 2)


def refit_exp_step_capacitance(ctx: Context):
    """tau from each cited window; C = tau / R_s and C = tau (1/R_s + 1/R_in), each averaged over the windows.
    Either may match (the card allows the simpler model): the one nearer the submitted value is returned."""
    fits = [f for f in ctx.submission["fits"] if f["used_in_fit"]]
    rs = float(ctx.submission["rs_ohm"])
    r_in = ctx.sealed.episode["device"]["logger_input_ohm"]
    taus = []
    for fit in fits:
        rows, _, problem = _fit_window(ctx, fit)
        if problem:
            raise ValueError(problem)
        taus.append(step_time_constant([r["t_s"] for r in rows], [r["ch1_V"] for r in rows]))
    if not taus:
        raise ValueError("no fit windows")
    plain = sum(taus) / len(taus) / rs
    loaded = sum(taus) / len(taus) * (1 / rs + 1 / r_in)
    submitted = ctx.submission["C_F"]
    value = min((plain, loaded), key=lambda c: abs(submitted - c) / c)
    return value, {"refit_tau_s": [round(t, 6) for t in taus], "C_plain_F": plain, "C_with_input_resistance_F": loaded}


def refit_divider(ctx: Context):
    """Mean CH1, CH2 of each cited window; R = R_s V1/V2 and R = V1/(V2/R_s - V1/R_in), each averaged over the
    windows. Either may match (the card allows the simpler model): the one nearer the submitted value is returned."""
    r_in = ctx.sealed.episode["device"]["logger_input_ohm"]
    plain, loaded, means = [], [], []
    for entry in [m for m in ctx.submission["measurements"] if m["used_in_fit"]]:
        rows, _, problem = _fit_window(ctx, entry)
        if problem:
            raise ValueError(problem)
        rs = float(entry["rs_ohm"])
        v1 = sum(r["ch1_V"] for r in rows) / len(rows)
        v2 = sum(r["ch2_V"] for r in rows) / len(rows)
        if v2 <= 0 or v2 / rs - v1 / r_in <= 0:
            raise ValueError(f"{entry['evidence_id']}: no current through R_s in the window")
        plain.append(rs * v1 / v2)
        loaded.append(v1 / (v2 / rs - v1 / r_in))
        means.append({"ch1_mean_V": round(v1, 6), "ch2_mean_V": round(v2, 6)})
    if not plain:
        raise ValueError("no measurement windows")
    plain_r, loaded_r = sum(plain) / len(plain), sum(loaded) / len(loaded)
    submitted = ctx.submission["R_ohm"]
    value = min((plain_r, loaded_r), key=lambda r: abs(submitted - r) / r)
    return value, {"window_means": means, "R_plain_ohm": plain_r, "R_with_input_resistance_ohm": loaded_r}


REFITS = {"linear_hooke": refit_linear_hooke, "torsion_inertia_ratio": refit_torsion_inertia_ratio,
          "exp_step_capacitance": refit_exp_step_capacitance, "divider": refit_divider,
          "linear_with_intercept_T2_vs_L": refit_linear_with_intercept_T2_vs_L,
          "linear_with_intercept_T2_vs_m": refit_linear_with_intercept_T2_vs_m, "pendulum_g": refit_pendulum_g}


def _airtrack_recorder(ctx, min_points, require):
    """airtrack_collision air track: every row is a valid run of this episode, reduced with the private glider labels of its passes
    (measurement.reduce_record); rows agree with the reduction; joint mass solvable (three launcher notches and distinct
    incoming speeds only when required)."""
    from benchmark.core.airtrack.measurement import ROW_TOLERANCE_M_S, joint_mass, reduce_record
    problems, fits, notches = [], [], []
    mass_a = ctx.sealed.episode['device']['mass_A_kg']
    rows, cited = [], {}
    for r in ctx.submission['measurements']:
        # Six levels: a row repeated verbatim adds no data and counts once; the same collision cited with
        # other values is still a conflict.
        if r['evidence_id'] in cited:
            if cited[r['evidence_id']] != r:
                problems.append('duplicate collision evidence')
            continue
        cited[r['evidence_id']] = r
        rows.append(r)
    if len(rows) < min_points:
        problems.append('too few collisions')
    for i, row in enumerate(rows):
        record, problem = ctx.evidence(row['evidence_id'])
        if problem or record.get('kind') != 'airtrack_recorder':
            problems.append(f'row {i}: {problem or "not collision evidence"}')
            continue
        try:
            fit = reduce_record(record, mass_a)
        except (ValueError, KeyError, TypeError) as exc:
            problems.append(f'row {i}: invalid collision: {exc}')
            continue
        for key in ('v_A_before_m_s', 'v_A_after_m_s', 'v_B_after_m_s'):
            if not abs(row[key] - fit[key]) <= ROW_TOLERANCE_M_S:
                problems.append(f'row {i}: {key} inconsistent with the gate record')
        fits.append(fit)
        notches.append(fit['notch'])
    # Six levels: no three-notch rule and no pairwise speed differences unless required (`three_notches`,
    # `distinct_speeds`); one valid collision determines m_B.
    if 'three_notches' in require and len({n for n in notches if n is not None}) < 3:
        problems.append('three different launcher notches required')
    try:
        joint_mass(fits, mass_a, min_rows=max(1, min_points),
                   min_speed_gap=0.025 if 'distinct_speeds' in require else None)
    except ValueError as exc:
        problems.append(str(exc))
    return not problems, {'problems': problems, 'collisions': len(fits)}


def refit_collision_momentum(ctx):
    from benchmark.core.airtrack.measurement import joint_mass
    fit=joint_mass(ctx.submission['measurements'],ctx.sealed.episode['device']['mass_A_kg'],min_rows=1,min_speed_gap=None)
    return fit['mass_B_kg'],{'momentum_rms_kg_m_s':fit['momentum_rms_kg_m_s']}


INSTRUMENT_CHECKS['airtrack_recorder']=_airtrack_recorder
REFITS['collision_momentum']=refit_collision_momentum


@gate("submission_self_consistent")
def submission_self_consistent(ctx: Context, refit, tol, enforce=True):
    """enforce=False (per experiment, six levels): the refit is reported only and the gate always passes
    (an experiment whose design dropped the rule while its card still quotes the tolerance)."""
    blocked = _need_submission(ctx)
    if blocked:
        return blocked
    try:
        value, detail = REFITS[refit](ctx)
    except (ValueError, ZeroDivisionError) as exc:
        return (not enforce), {"reason": str(exc), **({} if enforce else {"enforced": False})}
    submitted = ctx.submission[ctx.spec.raw["submission"]["primary"]["field"]]
    difference = abs(submitted - value) / value
    if not enforce:
        return True, {"enforced": False, "refit_value": value, "relative_difference": difference, **detail}
    return difference <= tol, {"refit_value": value, "relative_difference": difference, **detail}


# --- chemistry: spectrophotometer READ rows (spectro_basic) ---------------------------------------------------
# Private read status -> the `require` condition it violates.
SPECTRO_STATUS_REQUIRE = {"slot_empty": "volume_ge_2mL", "low_volume": "volume_ge_2mL", "not_mixed": "mixed",
                          "no_blank_at_λ": "blanked_same_lambda", "blank_ref_not_solvent": "blank_ref_solvent",
                          "overflowed": "status_ok"}


def nominal_mix(history):
    """{source: nominal mL} of one cuvette from its sealed dispensing history [(source, nominal mL), ...]."""
    volumes = {}
    for source, mL in history:
        volumes[source] = volumes.get(source, 0.0) + float(mL)
    return volumes


def _spectro_entries(ctx):
    """(calibration points, unknown readings) marked used_in_fit, each with its sealed record or a problem."""
    points = [p for p in ctx.submission["calibration"]["points"] if p["used_in_fit"]]
    unknowns = [u for u in ctx.submission["unknown_measurements"] if u["used_in_fit"]]
    return points, unknowns


def _spectro_logged(ctx, entry, role):
    """Evaluator values of one cited reading from the sealed record: (A, c_uM or dilution factor, record, problem)."""
    record, problem = ctx.evidence(entry["evidence_id"])
    if problem or record.get("kind") != "spectro_read":
        return None, None, None, problem or "not a spectrophotometer reading"
    mix = nominal_mix(record["nominal_history"])
    total = sum(mix.values())
    if total <= 0:
        return None, None, record, "the cuvette in the well was empty"
    if role == "calibration":
        if mix.get("unknown", 0.0) > 0:
            return None, None, record, "calibration cuvette holds unknown solution"
        value = ctx.sealed.episode["device"]["c_stock_uM_label"] * mix.get("stock", 0.0) / total
    else:
        if mix.get("stock", 0.0) > 0 or mix.get("unknown", 0.0) <= 0:
            return None, None, record, "unknown reading is not unknown solution diluted with water"
        value = total / mix["unknown"]
    return float(record["public"]["A_display"]), value, record, None


def _spectro_read(ctx: Context, min_points, require, min_unknown=1, composition_tol=0.01, wavelength_tol_nm=1):
    """Cited READ rows: calibration points (>= min_points, from as many different cuvettes) and unknown readings
    (>= min_unknown), each taken at the submitted wavelength after a BLANK there on a water-only cuvette, on a
    cuvette holding >= 2 mL, mixed, not overflowed (private read status), with the logged A and a composition that
    matches the cuvette's nominal dispensing history."""
    points, unknowns = _spectro_entries(ctx)
    problems, rows = [], []
    if len(points) < min_points:
        problems.append(f"{len(points)} calibration points used, need at least {min_points}")
    if len(unknowns) < int(ctx.param(min_unknown)):
        problems.append(f"{len(unknowns)} unknown readings used, need at least {min_unknown}")
    cited = [e["evidence_id"] for e in points + unknowns]
    if len(set(cited)) != len(cited):
        problems.append("a reading is cited more than once")
    wavelength = ctx.submission["wavelength_nm"]
    if "wavelength_public" in require and abs(wavelength - ctx.hidden["lambda_max_nm"]) > wavelength_tol_nm:
        problems.append(f"wavelength {wavelength} nm is more than {wavelength_tol_nm} nm from the published "
                        f"{ctx.hidden['lambda_max_nm']} nm")
    cuvettes = set()
    for role, entries in (("calibration", points), ("unknown", unknowns)):
        for i, entry in enumerate(entries):
            label = f"{role} {i}"
            a, value, record, problem = _spectro_logged(ctx, entry, role)
            if record is None:
                problems.append(f"{label}: {problem}")
                continue
            public = record["public"]
            rows.append({"evidence_id": entry["evidence_id"], "cuvette": record["cuvette"], "status": record["status"]})
            if problem and "composition_matches_log" in require:
                problems.append(f"{label}: {problem}")
            status = record["status"]
            if record.get("spilled") and "status_ok" in require:
                problems.append(f"{label}: the cuvette had been spilled before this reading")
            if status != "ok":
                violated = SPECTRO_STATUS_REQUIRE.get(status, "status_ok")
                if violated in require or "status_ok" in require:
                    problems.append(f"{label}: reading fails {violated}")
            if public["wavelength_nm"] != wavelength:
                problems.append(f"{label}: read at {public['wavelength_nm']} nm, submitted {wavelength} nm")
            if "blanked_same_lambda" in require and public["blank_wavelength_nm"] != public["wavelength_nm"]:
                problems.append(f"{label}: no BLANK at the reading wavelength")
            if "volume_ge_2mL" in require and (record["volume_mL"] is None
                                               or record["volume_mL"] < ctx.sealed.episode["device"]["min_read_mL"] - 1e-9):
                problems.append(f"{label}: cuvette held less than the minimum reading volume")
            if "A_matches_evidence" in require and abs(entry["A"] - public["A_display"]) > 5e-4:
                problems.append(f"{label}: A {entry['A']} differs from the logged {public['A_display']}")
            if problem is None and "composition_matches_log" in require:
                claimed = entry["c_uM"] if role == "calibration" else entry["dilution_factor"]
                if abs(claimed - value) > composition_tol * value + 1e-9:
                    field = "c_uM" if role == "calibration" else "dilution_factor"
                    problems.append(f"{label}: {field} {claimed} disagrees with the dispensing history "
                                    f"by more than {composition_tol:.0%}")
            if role == "calibration":
                cuvettes.add(record["cuvette"])
    if len(points) >= min_points and len(cuvettes) < min_points:
        problems.append(f"calibration points come from {len(cuvettes)} cuvettes, need {min_points}")
    return not problems, {"problems": problems, "readings": rows}


def refit_linear_calibration(ctx: Context):
    """Least-squares A = slope c + intercept through the used calibration points (logged A, c from the dispensing
    history); c_u = mean over the used unknown readings of (A - intercept) / slope x dilution factor."""
    points, unknowns = _spectro_entries(ctx)
    xs, ys, estimates = [], [], []
    for entry in points:
        a, c, _, problem = _spectro_logged(ctx, entry, "calibration")
        if problem:
            raise ValueError(f"{entry['evidence_id']}: {problem}")
        xs.append(c); ys.append(a)
    slope, intercept, r2 = _line_fit(xs, ys)
    if slope <= 0:
        raise ValueError("calibration slope is not positive")
    for entry in unknowns:
        a, factor, _, problem = _spectro_logged(ctx, entry, "unknown")
        if problem:
            raise ValueError(f"{entry['evidence_id']}: {problem}")
        estimates.append((a - intercept) / slope * factor)
    if not estimates:
        raise ValueError("no unknown readings")
    return sum(estimates) / len(estimates), {"slope_per_uM": slope, "intercept": intercept, "r2": r2,
                                             "unknown_estimates_uM": estimates}


INSTRUMENT_CHECKS["spectro_read"] = _spectro_read
REFITS["linear_calibration"] = refit_linear_calibration


# --- chemistry: titration recorder sessions (titration_basic) ----------------------------------------------------
TITRATION_EQUIVALENCE_PH = 7.0


def _titration_used(ctx):
    return [t for t in ctx.submission["titrations"] if t["used_in_fit"]]


def _titration_logged(ctx, entry):
    """Logged values of one cited titration: ([(V_nominal_mL, pH_display, row, private row)] sorted by volume,
    record, problems). A point that is not a row of the cited session is a problem and is left out."""
    record, problem = ctx.evidence(entry["evidence_id"])
    if problem or record.get("kind") != "titration_session":
        return None, None, [problem or "not a titration session"]
    rows = {r["sample_id"]: r for r in record["public"]["rows"]}
    private = {p["sample_id"]: p for p in record["private_rows"]}
    points, problems = [], []
    for p in entry["points"]:
        row = rows.get(p["sample_id"])
        if row is None:
            problems.append(f"sample {p['sample_id']} is not a row of the cited session")
            continue
        points.append((float(row["V_nominal_mL"]), float(row["pH_display"]), row, private[p["sample_id"]], p))
    points.sort(key=lambda item: (item[0], item[2]["sample_id"]))
    return points, record, problems


def _titration_endpoint(points):
    """(V_eq, index of the steepest step) of logged points sorted by volume: the line through the two points of
    the steepest step at pH 7, their midpoint when that step does not cross pH 7."""
    best = None
    for i, (a, b) in enumerate(zip(points, points[1:])):
        dv = b[0] - a[0]
        if dv <= 1e-9:
            raise ValueError("two cited points at the same titrant volume")
        slope = (b[1] - a[1]) / dv
        if best is None or slope > best[0]:
            best = (slope, i)
    if best is None:
        raise ValueError("fewer than two cited points")
    a, b = points[best[1]], points[best[1] + 1]
    if a[1] < TITRATION_EQUIVALENCE_PH < b[1]:
        return a[0] + (TITRATION_EQUIVALENCE_PH - a[1]) / (b[1] - a[1]) * (b[0] - a[0]), best[1]
    return 0.5 * (a[0] + b[0]), best[1]


def _titration_recorder(ctx: Context, min_points, require, bracket_low_pH=5, bracket_high_pH=9, bracket_max_gap_mL=0.10):
    """Cited titrations (>= min_points used): each cites a recorder session of its beaker; every cited point is a
    row of that session with the stability mark, the electrode in the liquid and a counter that equals the titrant
    delivered into the beaker, and carries the logged volume and pH; the last point below the low pH and the first above the high pH are at most the gap apart."""
    used = _titration_used(ctx)
    problems, sessions = [], []
    if len(used) < min_points:
        problems.append(f"{len(used)} titrations used, need at least {min_points}")
    cited = [t["evidence_id"] for t in used]
    if len(set(cited)) != len(cited):
        problems.append("a session is cited more than once")
    for i, entry in enumerate(used):
        label = f"titration {i}"
        points, record, found = _titration_logged(ctx, entry)
        problems += [f"{label}: {p}" for p in found]
        if record is None:
            continue
        sessions.append({"evidence_id": entry["evidence_id"], "beaker": record["public"]["beaker"], "points": len(points)})
        if "beaker_matches" in require and entry["beaker"] != record["public"]["beaker"]:
            problems.append(f"{label}: beaker {entry['beaker']} but the session is on {record['public']['beaker']}")
        for v, ph, row, private, claimed in points:
            where = f"{label} sample {row['sample_id']}"
            if "points_match_log" in require and (abs(claimed["V_mL"] - v) > 5e-3 or abs(claimed["pH"] - ph) > 5e-3):
                problems.append(f"{where}: ({claimed['V_mL']} mL, pH {claimed['pH']}) differs from the logged ({v} mL, pH {ph})")
            if "stable" in require and not row["stable"]:
                problems.append(f"{where}: reading without the stability mark")
            if "immersed" in require and not private["immersed"]:
                problems.append(f"{where}: electrode not in the liquid")
            if "counter_matches" in require and (private["overflowed"] or private["titrant_nominal_mL"] is None
                                                 or abs(private["titrant_nominal_mL"] - v) > 5e-3):
                problems.append(f"{where}: the counter does not equal the titrant delivered into the beaker")
        if "endpoint_bracketed" in require:
            low, high, gap = (float(ctx.param(x)) for x in (bracket_low_pH, bracket_high_pH, bracket_max_gap_mL))
            below = [p[0] for p in points if p[1] < low]
            above = [p[0] for p in points if p[1] > high]
            if not below or not above or not 1e-9 < min(above) - max(below) <= gap + 1e-9:
                problems.append(f"{label}: the cited points do not bracket the endpoint within {gap} mL "
                                f"(one below pH {low}, one above pH {high})")
    return not problems, {"problems": problems, "sessions": sessions}


def refit_titration_strong(ctx: Context):
    """c = c_titrant(label) x V_eq / V_sample(nominal) per used titration from its cited rows (logged volume and
    pH), V_eq from the steepest step; mean over the used titrations."""
    device = ctx.sealed.episode["device"]
    estimates, endpoints = [], []
    for entry in _titration_used(ctx):
        points, record, problems = _titration_logged(ctx, entry)
        if record is None or problems:
            raise ValueError(f"{entry['evidence_id']}: {(problems or ['no record'])[0]}")
        v_eq, _ = _titration_endpoint(points)
        endpoints.append(v_eq)
        estimates.append(device["titrant_label_M"] * v_eq / device["sample_nominal_mL"])
    if not estimates:
        raise ValueError("no titration used")
    return sum(estimates) / len(estimates), {"V_eq_mL": endpoints, "estimates_M": estimates}


INSTRUMENT_CHECKS["titration_recorder"] = _titration_recorder
REFITS["titration_strong"] = refit_titration_strong


# circuit_blackbox3 uses raw evidence provenance and truth scoring, never a reference refit gate.
from benchmark.evaluation.circuit_network import network_evidence, topology_correct
GATES["network_evidence"] = network_evidence
GATES["topology_correct"] = topology_correct


# Torsion bench: the storage-box tidy-up gate and the torsion_centroid pin-mounting gates.
from benchmark.evaluation.torsion import centroid_measurement_valid, centroid_readings_genuine, objects_in_box
GATES["objects_in_box"] = objects_in_box
GATES["centroid_measurement_valid"] = centroid_measurement_valid
GATES["centroid_readings_genuine"] = centroid_readings_genuine


@gate("bench_in_order")
def bench_in_order(ctx, max_tilt_deg=45.0, max_below_table_m=0.10, exempt=(), ignore_tilt=()):
    """The bench at submission, as the Bench sealed it: no loose item tipped over (the axis that pointed up at the
    start more than max_tilt_deg from vertical), none off the bench (more than max_below_table_m below the table
    top), none held by the robot. `exempt` lists prim-path patterns of items not checked at all, `ignore_tilt` those
    whose orientation means nothing (a ball)."""
    from fnmatch import fnmatch
    record = ctx.sealed.episode.get("bench_order")
    if not isinstance(record, dict) or not isinstance(record.get("items"), dict):
        return False, {"problems": ["no record of the bench at submission"],
                       "error": record.get("error") if isinstance(record, dict) else None}
    problems = []
    for path, item in sorted(record["items"].items()):
        if any(fnmatch(path, pattern) for pattern in exempt):
            continue
        name = path.rsplit("/", 1)[-1]
        if item["held"]:
            problems.append(f"{name} is held by the robot")
        if item["below_table_top_m"] > max_below_table_m:
            problems.append(f"{name} is off the bench ({item['below_table_top_m']:.2f} m below the table top)")
        if item["tilt_deg"] > max_tilt_deg and not any(fnmatch(path, pattern) for pattern in ignore_tilt):
            problems.append(f"{name} is tipped over ({item['tilt_deg']:.0f} deg from upright)")
    return not problems, {"problems": problems, "items": sorted(record["items"])}


# Incline friction: raw-record evidence only; the coefficients are scored against the truth.
from benchmark.evaluation.incline import incline_bench_restored, incline_records
GATES["incline_records"] = incline_records
GATES["incline_bench_restored"] = incline_bench_restored


def projectile_protocol(ctx):
    """projectile_landing is evaluated as a sealed physical continuation by its offline adapter."""
    from benchmark.evaluation.projectile import evaluate_projectile
    report = evaluate_projectile(ctx.spec, ctx.sealed, {"exp_id": ctx.spec.id,
        "instance_id": ctx.sealed.episode.get("instance_id"), "spec_digest": ctx.spec.digest(),
        "hidden": ctx.hidden})
    return report["G"], {"gates": report["gates"]}

GATES["projectile_protocol"] = projectile_protocol


# Weak-acid titration (titration_weak_acid): KHP standardization, sample curve and their refit.
from benchmark.evaluation.titration_weak import weak_acid_titration
GATES["weak_acid_titration"] = weak_acid_titration


# Rectified supply: device-checked protocol and the waveform sealed after the submission.
from benchmark.evaluation.rectifier import rectifier_protocol
GATES["rectifier_protocol"] = rectifier_protocol


# Two-dye mixture (spectro_mixture): READ evidence and the Beer-Lambert two-absorber refit.
from benchmark.evaluation.spectro_mixture import spectro_mixture
GATES["spectro_mixture"] = spectro_mixture


# Crystal violet kinetics: timed absorbance runs and their ln A - t refit.
from benchmark.evaluation.kinetics import kinetic_orders_correct, kinetic_records_valid, kinetic_runs
GATES["kinetic_runs"] = kinetic_runs
GATES["kinetic_orders_correct"] = kinetic_orders_correct        # orders task of 
GATES["kinetic_records_valid"] = kinetic_records_valid


# LED thresholds (led_planck): sweep evidence and the calibration line.
from benchmark.evaluation.led import led_curves
GATES["led_curves"] = led_curves


# Optics kit: the Bessel method (lens_focal).
from benchmark.evaluation.optics import bessel_focus
GATES["bessel_focus"] = bessel_focus
from benchmark.evaluation.optics import grating_orders
GATES["grating_orders"] = grating_orders
from benchmark.evaluation.optics import slit_fringes
GATES["slit_fringes"] = slit_fringes
from benchmark.evaluation.optics import plate_shift
GATES["plate_shift"] = plate_shift
from benchmark.evaluation.optics import relay_image
GATES["relay_image"] = relay_image
# The hand-slid optics bench (lens_focal, grating_wavelength).
from benchmark.evaluation.optics import optics_readings_genuine, optics_measurement_valid, optics_bench_restored
GATES["optics_readings_genuine"] = optics_readings_genuine
GATES["optics_measurement_valid"] = optics_measurement_valid
GATES["optics_bench_restored"] = optics_bench_restored


# Thermal kit (calorimetry_specific_heat, thermistor_beta, rod_expansion): readings genuine, measurement valid,
# the bench tidied up.
from benchmark.evaluation.thermal import thermal_bench_restored, thermal_measurement_valid, thermal_readings_genuine
GATES["thermal_readings_genuine"] = thermal_readings_genuine
GATES["thermal_measurement_valid"] = thermal_measurement_valid
GATES["thermal_bench_restored"] = thermal_bench_restored


# Cantilever with a compliant clamp (beam_compliance): readings as the LCD showed them at the cited observations, the
# carriage in the named hole; the bench tidied up.
from benchmark.evaluation.beam import beam_bench_restored, beam_readings_genuine
GATES["beam_readings_genuine"] = beam_readings_genuine
GATES["beam_bench_restored"] = beam_bench_restored


# Cell EMF and internal resistance (battery_emf): loaded points and the U-I line.
from benchmark.evaluation.battery import battery_lines
GATES["battery_lines"] = battery_lines


# Unknown dye (dye_absorptivity): standards from two labelled stocks at the chosen peak, Beer-Lambert slope.
from benchmark.evaluation.dye import dye_curve
GATES["dye_curve"] = dye_curve


# Circuit board (resistor_blackbox): readings from the displays and logger records, the measuring state, tidy-up.
from benchmark.evaluation.circuit_grid import circuit_bench_restored, measurement_valid, readings_genuine
GATES["readings_genuine"] = readings_genuine
GATES["measurement_valid"] = measurement_valid
GATES["circuit_bench_restored"] = circuit_bench_restored
# The other circuit-board experiments (battery_emf, ...): the measuring state with a device under test.
from benchmark.evaluation.circuit_grid_variants import grid_measurement_valid
GATES["grid_measurement_valid"] = grid_measurement_valid
from benchmark.evaluation.circuit_grid_variants import rectifier_grid_protocol
GATES["rectifier_grid_protocol"] = rectifier_grid_protocol
from benchmark.evaluation.circuit_grid_variants import led_x_measured
GATES["led_x_measured"] = led_x_measured


# Chemistry kit (spectrophotometry and titration): readings genuine, measurement valid, bench restored.
from benchmark.evaluation.chem_gates import (chem_bench_restored, spectro_measurement_valid, spectro_readings_genuine,
                                             titration_measurement_valid, titration_readings_genuine)
GATES["spectro_readings_genuine"] = spectro_readings_genuine
GATES["spectro_measurement_valid"] = spectro_measurement_valid
GATES["titration_readings_genuine"] = titration_readings_genuine
GATES["titration_measurement_valid"] = titration_measurement_valid
GATES["chem_bench_restored"] = chem_bench_restored

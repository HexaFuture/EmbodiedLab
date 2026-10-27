"""incline_friction: raw-record evidence gate and the tidy-up gate (the bench back as found). Answers are scored
against the truth only; nothing here refits the submitted coefficients.

Two sliders, A and B. The position sensor measures only the slider on its track nearest the lower end;
each record sample keeps a private row {slider, free}: the slider measured in that tick and whether it was free (not
held, not touched by the robot, the gate or the other slider). Attribution: a cited record belongs to the slider whose
free stretch in it shows the cited measurement (static onset or kinetic slide). A stretch is unbroken samples of ONE
measured slider, free and with a position: when the sensor's reading passes from one slider to the other (the lower
one is lifted off or placed on the deck), the stretch ends. A record in which stretches of both sliders show the
measurement is ambiguous and is not evidence; a record cited for a slider whose stretch is not the one showing the
measurement is not evidence for that slider.

Six levels: a record whose stretches show the measurement for both sliders counts for each of them (the
entry names the slider; only that slider's stretches are looked at), and a record may be cited once per slider and
role; the kinetic slide needs no fixed or minimum angle when those limits are None (the angle is in every sample)."""
import math

SLIDERS = ("A", "B")


def free_runs(record):
    """[(slider, samples)]: stretches of consecutive samples in which the same slider was measured (on the sensor's
    track), free and with a position (private audit rows, one per public sample)."""
    samples, private = record["public"]["samples"], record["private_rows"]
    runs, current, owner = [], [], None
    for sample, row in zip(samples, private):
        slider = row.get("slider")
        ok = sample["position_m"] is not None and row.get("free") and slider is not None
        if current and (not ok or slider != owner):
            runs.append((owner, current))
            current = []
        if ok:
            current.append(sample)
            owner = slider
    if current:
        runs.append((owner, current))
    return runs


def _travel(run):
    positions = [s["position_m"] for s in run]
    return max(positions) - min(positions)


def static_onset(run, min_rise_deg=1.0, min_travel_m=0.02, onset_m=1e-3):
    """True when a free run contains the onset of sliding on the rising ramp: the slider rests at the start of the run
    while the angle rises by at least min_rise_deg, then moves more than onset_m from where it rested and travels at
    least min_travel_m. A run that begins after the slider has started to move (e.g. once it left the gate) has no
    onset in it."""
    if _travel(run) < min_travel_m:
        return False
    onset = next((i for i, s in enumerate(run) if abs(s["position_m"] - run[0]["position_m"]) > onset_m), None)
    return bool(onset) and run[onset]["angle_deg"] - min(s["angle_deg"] for s in run[:onset + 1]) >= min_rise_deg


def kinetic_slide(run, min_travel_m=0.10, max_angle_span_deg=0.05, min_angle_deg=5.0):
    """True when a free run holds a slide: every sample after the slider has moved 1 mm from the run's start counts;
    they span at least min_travel_m, their angles stay within max_angle_span_deg of each other and at or above
    min_angle_deg (either limit None: not checked)."""
    moving = [s for s in run if abs(s["position_m"] - run[0]["position_m"]) > 1e-3]
    if not (moving and _travel(moving) >= min_travel_m):
        return False
    angles = [s["angle_deg"] for s in moving]
    return ((max_angle_span_deg is None or max(angles) - min(angles) <= max_angle_span_deg + 1e-9)
            and (min_angle_deg is None or min(angles) >= min_angle_deg))


def record_sliders(record, role, **limits):
    """The sliders a record shows the measurement `role` ('static' or 'kinetic') for, from its free runs."""
    if role == "static":
        test = lambda run: static_onset(run, limits.get("min_static_rise_deg", 1.0), limits.get("min_static_travel_m", 0.02))
    else:
        test = lambda run: kinetic_slide(run, limits.get("min_kinetic_travel_m", 0.10),
                                         limits.get("max_kinetic_angle_span_deg", 0.05), limits.get("min_kinetic_angle_deg", 5.0))
    return sorted({slider for slider, run in free_runs(record) if test(run)})


def incline_records(ctx, min_static_rise_deg=1.0, min_static_travel_m=0.02, min_kinetic_travel_m=0.10,
                    max_kinetic_angle_span_deg=0.05, min_kinetic_angle_deg=5.0):
    """Cited records are recorder exports of this episode, each cited once with the slider it measured. For each
    slider at least one static record (the free slider starting to move while the angle was being raised:
    static_onset, free from its rest through the onset) and one kinetic record (the free slider sliding at a fixed
    angle). A record counts for the slider its qualifying stretch belongs to; one with qualifying stretches of both
    sliders is ambiguous."""
    from benchmark.evaluation.gates import _need_submission      # the gate library imports this module
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    limits = dict(min_static_rise_deg=min_static_rise_deg, min_static_travel_m=min_static_travel_m,
                  min_kinetic_travel_m=min_kinetic_travel_m, max_kinetic_angle_span_deg=max_kinetic_angle_span_deg,
                  min_kinetic_angle_deg=min_kinetic_angle_deg)
    problems, used = [], []
    entries = [("static", e) for e in ctx.submission["static_evidence"]] + \
              [("kinetic", e) for e in ctx.submission["kinetic_evidence"]]
    cited = [(e["evidence_id"], role, e["slider"]) for role, e in entries]
    if len(set(cited)) != len(cited):
        problems.append("a record is cited more than once for the same slider and role")
    accepted = {(role, slider): 0 for role in ("static", "kinetic") for slider in SLIDERS}
    for role, entry in entries:
        eid, claimed = entry["evidence_id"], entry["slider"]
        record, problem = ctx.evidence(eid)
        if problem or record.get("kind") != "incline_record":
            problems.append(f"{role}: {problem or 'not an incline record'}")
            continue
        samples = record["public"]["samples"]
        if len(record["private_rows"]) != len(samples):
            problems.append(f"{role} {eid}: audit rows do not match the record")
            continue
        shown = record_sliders(record, role, **limits)
        what = ("a free slider starting to move while the ramp was being raised (one free stretch must contain the "
                "rest, the rise and the onset of sliding)" if role == "static" else
                f"a free slide of at least {min_kinetic_travel_m} m"
                + (" at a fixed angle" if max_kinetic_angle_span_deg is not None else ""))
        if not shown:
            problems.append(f"{role} {eid}: no {what}")
        elif claimed not in shown:
            problems.append(f"{role} {eid}: cited for slider {claimed}, but the record does not show {what} of slider {claimed}")
        else:
            accepted[(role, claimed)] += 1
        used.append({"evidence_id": eid, "role": role, "slider": claimed, "samples": len(samples),
                     "free_runs": len(free_runs(record)), "shows": shown})
    for (role, slider), count in accepted.items():
        if not count:
            problems.append(f"no {role} record of slider {slider}")
    return not problems, {"problems": problems, "records": used}


def incline_bench_restored(ctx, ramp_level_deg=0.1, ramp_still_deg_s=0.05, block_tilt_deg=5.0, block_gap_m=0.003):
    """Tidy-up gate: at submission the ramp is level (within ramp_level_deg) and still (at most ramp_still_deg_s), the
    gate is back in place (commanded in and fully down), the recorder is not recording, and each slider lies flat on a
    loading pad, either pad (plate tilted at most block_tilt_deg, its underside within block_gap_m of the pad top, its
    centre over the pad) and is not held, as the Bench sealed it (terminal_state of FrictionBackend.bench_state)."""
    state = ctx.sealed.episode.get("terminal_state") or {}
    ramp, gate, recorder, sliders = (state.get(k) for k in ("ramp", "gate", "recorder", "sliders"))
    if not all(isinstance(x, dict) for x in (ramp, gate, recorder, sliders)) or \
            not all(isinstance(sliders.get(k), dict) for k in SLIDERS):
        return False, {"problems": ["no record of the incline bench at submission"]}
    level, still = float(ctx.param(ramp_level_deg)), float(ctx.param(ramp_still_deg_s))
    tilt, gap = float(ctx.param(block_tilt_deg)), float(ctx.param(block_gap_m))
    problems = []
    if not abs(ramp.get("angle_deg", math.inf)) <= level:
        problems.append(f"ramp at {ramp.get('angle_deg')} deg, not level")
    if not abs(ramp.get("rate_deg_s", math.inf)) <= still:
        problems.append(f"ramp turning at {ramp.get('rate_deg_s')} deg/s")
    if not gate.get("in_place"):
        problems.append(f"gate not in place ({gate.get('target')}, {gate.get('position_m')} m)")
    if recorder.get("recording") or recorder.get("ready"):
        problems.append(f"recorder {recorder.get('state')}")
    for name in SLIDERS:
        slider = sliders[name]
        if slider.get("held"):
            problems.append(f"slider {name} held by the robot")
        offset, half = slider.get("pad_offset_m") or [math.inf, math.inf], slider.get("pad_half_m") or [0.0, 0.0]
        if not all(abs(o) <= h for o, h in zip(offset, half)):
            problems.append(f"slider {name} centre {offset} m from the nearest loading pad's centre, not over a pad")
        if not abs(slider.get("gap_above_pad_m", math.inf)) <= gap:
            problems.append(f"slider {name} {slider.get('gap_above_pad_m')} m above the pad top, not lying on a pad")
        if not slider.get("tilt_deg", math.inf) <= tilt:
            problems.append(f"slider {name} tilted {slider.get('tilt_deg')} deg, not flat")
    return not problems, {"problems": problems, "terminal_state": state}

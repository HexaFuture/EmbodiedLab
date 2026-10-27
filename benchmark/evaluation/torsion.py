"""Torsion bench gates (torsion_inertia, torsion_centroid): the storage-box tidy-up gate and the
torsion_centroid pin-mounting evidence gates.

The photogate export holds only the transits; each transit's bench condition (contact, brake, load_changing, ok) and,
for torsion_centroid, the mounting on the pin at 30 Hz are the Bench's private records sealed with the evidence.
"""
from __future__ import annotations

from benchmark.evaluation.gates import TORSION_LOADS, _need_submission, photogate_period, photogate_rows

TORSION_OBJECTS = ("standard_disc", "unknown_object")


def objects_in_box(ctx):
    """Torsion bench at submission, as the Bench sealed it (terminal_state of TorsionBackend.bench_state): both objects
    seated in storage box seats (any seat; the public rule: within 20 mm horizontally and vertically of an object
    standing on the seat floor, tilted at most 10 degrees, not held), no photogate record running, the damper lever at
    OFF and the robot holding nothing."""
    state = ctx.sealed.episode.get("terminal_state") or {}
    seats, damper, recorder = state.get("storage_box_seats"), state.get("damper"), state.get("recorder")
    if not (isinstance(seats, dict) and isinstance(damper, dict) and isinstance(recorder, dict)):
        return False, {"problems": ["no record of the torsion bench at submission"]}
    problems = [f"{key} is not in a storage box seat" for key in TORSION_OBJECTS if key not in seats.values()]
    if recorder.get("recording") or recorder.get("photogate_recording") or state.get("photogate_active"):
        problems.append("a photogate record is running")
    if damper.get("state") != "OFF":
        problems.append(f"damper lever {damper.get('state')}, not OFF")
    if state.get("held") is not None:
        problems.append(f"the robot holds {state['held']}")
    return not problems, {"problems": problems, "seats": seats, "damper": damper.get("state"),
                          "recorder": recorder.get("state")}


# --- torsion_centroid ---------------------------------------------------------------------------------------------
def centroid_rows(record):
    """Transits of a centroid record (public CSV) with their private bench condition."""
    return photogate_rows(dict(record["public"], transit_status=record.get("transit_status"),
                               transit_conditions=record.get("transit_conditions")))


def centroid_mount(record):
    """The mounting on the pin during a record, from the private 30 Hz trace: (load, hole) when every sample agrees,
    else None. ('empty', None) is the empty turntable; the disc's centre bore is hole 0."""
    samples = record.get("mount_samples") or []
    mounts = {(s[1], s[2]) for s in samples}
    if len(samples) < 2 or len(mounts) != 1:
        return None
    (load, hole), = mounts
    return None if load is None else (load, hole)


def _expected(row):
    if row["load"] == "empty":
        return ("empty", None)
    if row["load"] == "standard_disc":
        return ("standard_disc", 0)
    return ("unknown_object", row.get("hole"))


def centroid_record_valid(record, load, min_cycles=5):
    """(measured period or None, problems) of one record under the measurement-valid conditions for `load`: complete,
    the load the same at the start and the end, the object seated on the pin throughout (empty: nothing on the platter)
    and >= min_cycles complete periods of consecutive clean transits."""
    public, problems = record["public"], []
    if record.get("status", public.get("status")) != "complete":      # private 
        problems.append("record not complete")
    expected = set(TORSION_LOADS[load])
    if set(record.get("loads_at_arm") or {}) != expected or set(record.get("loads_at_end") or {}) != expected:
        problems.append(f"the turntable load was not {load!r} at the start and the end")
    mount = centroid_mount(record)
    if mount is None or mount[0] != load:
        problems.append("the turntable was not empty throughout the record" if load == "empty" else
                        f"{load} was not seated alone on the pin throughout the record")
    measured = photogate_period(centroid_rows(record))
    if measured is None or measured[1] < min_cycles:
        problems.append(f"fewer than {min_cycles} complete periods of clean transits")
        measured = None
    return (measured[0] if measured else None), problems, mount


def centroid_coverage_ok(loads, holes, min_holes=2, alternative_holes=None):
    """The valid records determine the four unknowns: (a) the empty turntable, the standard disc and the sample at
    >= min_holes holes, or (b) with alternative_holes (six levels): the empty turntable or the disc and the
    sample at >= alternative_holes holes."""
    if {"empty", "standard_disc"} <= set(loads) and len(holes) >= min_holes:
        return True
    return (alternative_holes is not None and bool({"empty", "standard_disc"} & set(loads))
            and len(holes) >= alternative_holes)


def centroid_measurement_valid(ctx, min_cycles=5, min_holes=2, alternative_holes=None):
    """Every row used: complete record, load unchanged, the object seated on the pin throughout (the empty turntable:
    nothing on the platter), >= min_cycles clean periods; the rows cover the empty turntable, the standard disc and the
    sample at >= min_holes different holes (or, with alternative_holes, one of the empty turntable and the disc and the
    sample at >= alternative_holes holes), each citing a different record."""
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    rows = [r for r in ctx.submission["measurements"] if r["used_in_fit"]]
    problems, holes, loads = [], set(), set()
    if len({r["evidence_id"] for r in rows}) != len(rows):
        problems.append("the same record is cited more than once")
    for i, row in enumerate(rows):
        record, problem = ctx.evidence(row["evidence_id"])
        if problem or record.get("kind") != "centroid_photogate":
            problems.append(f"row {i}: {problem or 'not torsion_centroid photogate evidence'}")
            continue
        _, issues, mount = centroid_record_valid(record, row["load"], min_cycles)
        problems += [f"row {i}: {issue}" for issue in issues]
        if not issues:
            loads.add(row["load"])
            if row["load"] == "unknown_object":
                holes.add(mount[1])
    if not centroid_coverage_ok(loads, holes, min_holes, alternative_holes):
        alternative = (f", or one of them and the sample at {alternative_holes} or more holes"
                       if alternative_holes is not None else "")
        problems.append(f"valid records must cover the empty turntable, the standard disc and the sample at "
                        f"{min_holes} or more holes{alternative} (have {sorted(loads)}, holes {sorted(holes)})")
    return not problems, {"problems": problems, "holes": sorted(holes)}


def centroid_readings_genuine(ctx, period_tol=0.005):
    """Every row used names what was on the pin during its record (the sample's hole number, `hole` null for the disc
    and the empty turntable, the empty turntable recorded empty), and its period agrees with the evaluator's period of
    that record (longest run of clean transits) within period_tol."""
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    rows = [r for r in ctx.submission["measurements"] if r["used_in_fit"]]
    problems, periods = [], {}
    for i, row in enumerate(rows):
        record, problem = ctx.evidence(row["evidence_id"])
        if problem or record.get("kind") != "centroid_photogate":
            problems.append(f"row {i}: {problem or 'not torsion_centroid photogate evidence'}")
            continue
        if row["load"] != "unknown_object" and row.get("hole") is not None:
            problems.append(f"row {i}: hole given for {row['load']}")
        mount = centroid_mount(record)
        if mount != _expected(row):
            problems.append(f"row {i}: {row['load']} at hole {row.get('hole')} submitted, the record shows "
                            f"{mount[0] + ' at hole ' + str(mount[1]) if mount else 'no single mounting'}")
        measured = photogate_period(centroid_rows(record))
        if measured is None:
            problems.append(f"row {i}: no oscillation in the longest run of clean transits")
            continue
        periods[row["evidence_id"]] = round(measured[0], 6)
        if abs(row["period_s"] / measured[0] - 1) > period_tol:
            problems.append(f"row {i}: period_s {row['period_s']} differs from the record ({measured[0]:.5f} s)")
    return not problems, {"problems": problems, "evidence_periods_s": periods}

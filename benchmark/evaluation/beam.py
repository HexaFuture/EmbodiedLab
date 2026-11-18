"""beam_compliance gates.

- beam_readings_genuine: every row of `readings` cites a different observation of this episode (get_observation's
  evidence_id, kind bench_view) and the hole it names; at that observation the carriage stood in that hole (its index
  within hole_tol_mm of the hole, moving slower than max_speed_mm_s: a reading taken between holes or while the
  carriage moves is invalid) and the submitted reading equals what the LCD showed then (the latest refresh, private)
  within tol_mm (the display's last digit). How the strip was loaded and where the indicator was read is the Agent's
  choice and is scored only through e.
- beam_bench_restored (tidy-up): at submission all three weights stand in rack collars, nothing presses on the strip,
  the carriage stands in a hole (any hole) and the robot holds nothing, as the Bench sealed it (terminal_state of
  BeamBackend.seat_audit).
"""
from __future__ import annotations

HOLES_MM = (50, 100, 150, 200, 250, 300, 350, 400)


def _in_hole(carriage, hole_mm, tol_mm, speed_mm_s):
    """None when the carriage {'x_mm', 'v_mm_s'} stands in hole `hole_mm`, else what is wrong."""
    x, v = carriage.get("x_mm"), carriage.get("v_mm_s")
    if not isinstance(x, (int, float)) or not isinstance(v, (int, float)):
        return "no record of the carriage"
    if abs(x - hole_mm) > tol_mm:
        return f"the carriage's index was at {x:.2f} mm, not in hole {hole_mm:g}"
    if abs(v) > speed_mm_s:
        return f"the carriage was moving ({v:.2f} mm/s)"
    return None


def beam_readings_genuine(ctx, tol_mm=0.001, hole_tol_mm=0.3, max_speed_mm_s=1.0):
    from benchmark.evaluation.gates import _need_submission
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    tol = float(ctx.param(tol_mm)) + 1e-9
    hole_tol, speed = float(ctx.param(hole_tol_mm)), float(ctx.param(max_speed_mm_s))
    problems, rows, seen = [], [], set()
    for i, row in enumerate(ctx.submission["readings"]):
        eid = row["evidence_id"]
        record, problem = ctx.evidence(eid)
        if problem is None and record.get("kind") != "bench_view":
            problem = f"{eid}: not an observation of this bench"
        if problem is None and eid in seen:
            problem = f"{eid}: cited twice"
        seen.add(eid)
        if problem:
            problems.append(f"row {i}: {problem}")
            continue
        carriage = record.get("carriage") or {}
        shown = record.get("display_mm")
        entry = {"evidence_id": eid, "hole_mm": row["hole_mm"], "reading_mm": row["reading_mm"], "shown_mm": shown,
                 "carriage_mm": carriage.get("x_mm"), "carriage_mm_s": carriage.get("v_mm_s"),
                 "config": record.get("config"), "held": record.get("held")}
        where = _in_hole(carriage, float(row["hole_mm"]), hole_tol, speed)
        if where:
            problems.append(f"row {i}: {where}")
        if not isinstance(shown, (int, float)):
            problems.append(f"row {i}: the LCD showed no reading at {eid}")
        elif abs(row["reading_mm"] - shown) > tol:
            problems.append(f"row {i}: reading {row['reading_mm']} mm, the LCD showed {shown:.3f} mm")
        rows.append(entry)
    return not problems, {"problems": problems, "rows": rows}


def beam_bench_restored(ctx, max_load_n=0.01, hole_tol_mm=0.3, max_speed_mm_s=1.0):
    state = ctx.sealed.episode.get("terminal_state") or {}
    seats, strip = state.get("weight_seat"), state.get("strip")
    if not isinstance(seats, dict) or not seats or not isinstance(strip, dict):
        return False, {"problems": ["no record of the beam bench at submission"]}
    problems = []
    for weight, seat in sorted(seats.items()):
        if not (isinstance(seat, str) and seat.startswith("rack:")):
            problems.append(f"{weight} is not in a rack collar ({seat or 'loose'})")
    load, touching = strip.get("load_n"), strip.get("touching") or []
    if touching or not (isinstance(load, (int, float)) and abs(load) <= float(ctx.param(max_load_n))):
        problems.append(f"the strip is loaded ({load} N by {touching})")
    carriage = state.get("carriage") or {}
    holes = ctx.spec.raw.get("scene", {}).get("params", {}).get("holes_mm") or HOLES_MM
    x = carriage.get("x_mm")
    hole = min(holes, key=lambda h: abs(h - x)) if isinstance(x, (int, float)) else None
    where = (_in_hole(carriage, hole, float(ctx.param(hole_tol_mm)), float(ctx.param(max_speed_mm_s)))
             if hole is not None else "no record of the carriage")
    if where:
        problems.append(f"the indicator carriage is not in a hole: {where}")
    if state.get("held") is not None or state.get("releasing") is not None:
        problems.append(f"the robot holds {state.get('held') or state.get('releasing')}")
    return not problems, {"problems": problems, "weight_seat": seats, "strip": strip, "carriage": state.get("carriage")}

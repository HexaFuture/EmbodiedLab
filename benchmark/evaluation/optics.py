"""Optics kit gates.

Hand-slid bench (lens_focal, grating_wavelength, plate_index, slit_width, optical_relay; evidence = get_observation
records `bench_view`, bound to the physical state at the observation):
- optics_readings_genuine: every cited measurement is an observation of this episode, cited once; every measurement
  used in the fit (every row for plate_index and optical_relay) has its rider positions within position_tol_mm of the
  riders' true index marks at that observation, and each submitted screen position within spot_tol_mm of the true
  one on the screen scale: the first-order spot centres (grating_wavelength), every listed dark fringe of its stated
  order, a fringe on the card at least 1 mm inside its edge at that observation, shown or not (slit_width). Rows not
  used in the fit are not read-checked (six levels, the error decides);
- optics_measurement_valid: at every observation a measurement used in the fit cites, the element sits on rider A
  and the source is on, and the screen shows what the measurement needs: a sharp F image (blur radius at most
  max_blur_mm, lens_focal), both first-order spots on the screen (grating_wavelength) or a dark fringe on each side
  of the central maximum (both first-order dark fringes, slit_width); plate_index: every cited
  row, in group no_plate with the block on no rider and in group plate with the block on rider B, the lens on A, the
  light on and the image sharp, each group with a row (where the lens stood is the Agent's choice:
  lens_match_mm is no longer applied);
  optical_relay: the one cited observation (the submission is its row) with L1 on A, L2 on B, the light on and the
  image sharp (relay_score scores its true magnification);
- optics_bench_restored (tidy-up): at submission the source is off, every holder stands upright in a seat (on a
  rider or on the stand) and the robot holds nothing.
How many observations and which method is the Agent's choice; the error decides.

Keypad bench (records earlier: slit_width, plate_index, optical_relay; screen captures `screen_capture`): slit_fringes, plate_shift,
relay_image and relay_score below. bessel_focus and grating_orders are the keypad versions of lens_focal and
grating_wavelength (kept for their records).
"""
from __future__ import annotations

HAND_ELEMENT = {"lens_focal": "L", "grating_wavelength": "G", "plate_index": "L", "slit_width": "SL",
                "optical_relay": "L1"}
HAND_SECOND = {"optical_relay": ("B", "L2")}             # a second element on its rider at every cited observation
HAND_SINGLE = ("optical_relay",)                         # the submission itself is the one cited row
HAND_GROUPS = {"plate_index": ("no_plate", "plate")}     # rows listed per group; every row is used
HAND_READINGS = {"lens_focal": {"lens_mm": "A", "screen_mm": "S"}, "plate_index": {"lens_mm": "A", "screen_mm": "S"},
                 "grating_wavelength": {"grating_mm": "A", "screen_mm": "S"},
                 "slit_width": {"slit_mm": "A", "screen_mm": "S"},
                 "optical_relay": {"lens_A_mm": "A", "lens_B_mm": "B", "screen_mm": "S"}}
HAND_SPOTS = {"grating_wavelength": {"x_minus1_mm": "-1", "x_plus1_mm": "1"}}
SCREEN_HALF_MM = 60.0
SPOT_MARGIN_MM = 1.8                     # three spot sigmas: a spot this close to the edge is cut by it
FRINGE_MARGIN_MM = 1.0                   # a dark fringe this close to the card's edge is not on the card (optics_hand)


def _hand_rows(ctx):
    """[(index, row, record, problem)] of the cited measurements (bench_view evidence of this episode)."""
    rows, seen = [], set()
    groups = HAND_GROUPS.get(ctx.spec.id)
    if ctx.spec.id in HAND_SINGLE:
        cited = [(0, dict(ctx.submission, used_in_fit=True))]
    elif groups:
        cited = [(f"{g} {k}", dict(row, group=g, used_in_fit=True))
                 for g in groups for k, row in enumerate(ctx.submission[g]["measurements"])]
    else:
        cited = list(enumerate(ctx.submission["measurements"]))
    for i, row in cited:
        record, problem = ctx.evidence(row["evidence_id"])
        if problem is None and record.get("kind") != "bench_view":
            problem = f"{row['evidence_id']}: not an observation of this bench"
        if problem is None and row["evidence_id"] in seen:
            problem = f"{row['evidence_id']}: cited twice"
        seen.add(row["evidence_id"])
        rows.append((i, row, record, problem))
    return rows


def optics_readings_genuine(ctx, position_tol_mm=1.0, spot_tol_mm=0.5):
    from benchmark.evaluation.gates import _need_submission
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    exp = ctx.spec.id
    tol, spot_tol = float(ctx.param(position_tol_mm)), float(ctx.param(spot_tol_mm))
    problems, detail = [], []
    for i, row, record, problem in _hand_rows(ctx):
        if problem:
            problems.append(f"row {i}: {problem}")
            continue
        entry = {"evidence_id": row["evidence_id"]}
        if not row.get("used_in_fit"):
            entry["checked"] = "not used in the fit: evidence only"
            detail.append(entry)
            continue
        for field, axis in HAND_READINGS[exp].items():
            true = record["positions_mm"][axis]
            entry[field] = {"read": row[field], "true": round(true, 3)}
            if not abs(row[field] - true) <= tol:
                problems.append(f"row {i}: {field} {row[field]} mm, rider {axis} stood at {true:.2f} mm")
        spots = (record.get("optics") or {}).get("spots_mm") or {}
        for field, order in HAND_SPOTS.get(exp, {}).items():
            true = spots.get(order)
            entry[field] = {"read": row[field], "true": true}
            if true is None:
                problems.append(f"row {i}: no order {order} spot on the screen at this observation")
            elif not abs(row[field] - true) <= spot_tol:
                problems.append(f"row {i}: {field} {row[field]} mm, the order {order} spot was at {true:.2f} mm")
        if exp == "slit_width":
            problems += _fringe_readings(i, row, _fringes_on_card(ctx, record), entry, spot_tol)
        detail.append(entry)
    return not problems, {"problems": problems, "rows": detail}


def _fringes_on_card(ctx, record):
    """slit_width: {order: screen mm} of every geometric dark fringe at least FRINGE_MARGIN_MM inside the card's edge
    at an observation, shown or not (the 5 % side-lobe rule of the measurement-valid gate is not applied to readings,
). Records keep them (optics.dark_fringes_all_mm); for older ones they follow from
    the recorded slit-screen distance."""
    optics = record.get("optics") or {}
    edge = SCREEN_HALF_MM - FRINGE_MARGIN_MM
    if "dark_fringes_all_mm" in optics:
        on_card = optics["dark_fringes_all_mm"]
    elif not record.get("source_on") or optics.get("slit_screen_mm") is None:
        on_card = {}
    else:
        import sys
        from pathlib import Path
        core = str(Path(__file__).resolve().parents[1] / "core")
        if core not in sys.path:
            sys.path.insert(0, core)
        import optics_model as M
        device = ctx.sealed.episode["device"]
        slit = next(v for v in device["lenses"].values() if v.get("kind") == "slit")
        on_card = {str(k): round(1000 * x, 4) for k, x in M.slit_dark_fringes(
            float(device["laser_nm"]) * 1e-9, slit["width_m"], optics["slit_screen_mm"] / 1000)}
    return {k: x for k, x in on_card.items() if abs(x) <= edge}


def _fringe_readings(i, row, true, entry, tol):
    """slit_width: each listed dark fringe (order, screen mm) is one on the card at that observation (`true`, from
    _fringes_on_card), read within tol."""
    problems, seen, shown = [], set(), []
    for f in row["dark_fringes"]:
        order, x = int(f["order"]), f["x_mm"]
        at = true.get(str(order))
        shown.append({"order": order, "read": x, "true": at})
        if order in seen:
            problems.append(f"row {i}: dark fringe order {order} listed twice")
        seen.add(order)
        if at is None:
            problems.append(f"row {i}: no dark fringe of order {order} on the card at this observation")
        elif not abs(x - at) <= tol:
            problems.append(f"row {i}: dark fringe {order} read at {x} mm, it was at {at:.2f} mm")
    entry["dark_fringes"] = shown
    return problems


def optics_measurement_valid(ctx, max_blur_mm=1.0, lens_match_mm=1.0):
    from benchmark.evaluation.gates import _need_submission
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    exp = ctx.spec.id
    element = HAND_ELEMENT[exp]
    max_blur = float(ctx.param(max_blur_mm))
    problems, detail, used = [], [], 0
    for i, row, record, problem in _hand_rows(ctx):
        if not row.get("used_in_fit"):
            continue
        used += 1
        if problem:
            problems.append(f"row {i}: {problem}")
            continue
        optics = record.get("optics") or {}
        entry = {"evidence_id": row["evidence_id"], "source_on": record.get("source_on"),
                 "on_A": record.get("seated", {}).get("A")}
        if record.get("seated", {}).get("A") != element:
            problems.append(f"row {i}: {element} is not seated on rider A")
        if exp in HAND_SECOND:
            axis, second = HAND_SECOND[exp]
            entry["on_" + axis] = record.get("seated", {}).get(axis)
            if record.get("seated", {}).get(axis) != second:
                problems.append(f"row {i}: {second} is not seated on rider {axis}")
        if not record.get("source_on"):
            problems.append(f"row {i}: the source is off")
        if exp == "plate_index":
            seated = record.get("seated") or {}
            entry.update(group=row["group"], on_B=seated.get("B"), lens_A_mm=record["positions_mm"]["A"])
            block = [a for a, e in seated.items() if e == "P"]
            if row["group"] == "no_plate" and block:
                problems.append(f"row {i}: the block is on rider {block[0]}")
            if row["group"] == "plate" and seated.get("B") != "P":
                problems.append(f"row {i}: the block is not seated on rider B")
        if exp in ("lens_focal", "plate_index", "optical_relay"):
            blur = optics.get("blur_mm")
            entry["blur_mm"] = blur
            if blur is None or blur > max_blur:
                problems.append(f"row {i}: the image is not sharp (blur radius {blur} mm)")
        elif exp == "slit_width":
            fringes = optics.get("dark_fringes_mm") or {}
            entry["dark_fringes_shown"] = len(fringes)
            if not ("-1" in fringes and "1" in fringes):
                problems.append(f"row {i}: the screen does not show a dark fringe on each side of the central maximum")
        else:
            spots = optics.get("spots_mm") or {}
            entry["spots_mm"] = spots
            if not all(o in spots and abs(spots[o]) <= SCREEN_HALF_MM - SPOT_MARGIN_MM for o in ("-1", "1")):
                problems.append(f"row {i}: the two first-order spots are not both on the screen")
        detail.append(entry)
    if exp == "plate_index":
        problems += _plate_rows(ctx, detail, float(ctx.param(lens_match_mm)))
    if used == 0:
        problems.append("no measurement is used in the fit")
    return not problems, {"problems": problems, "rows": detail}


def _plate_rows(ctx, detail, tol):
    """plate_index: each group has a row. Where the lens stood in each row is the Agent's choice (six levels: several lens positions, or the lens moved with the screen fixed, are reasonable methods; the error
    decides), so `tol` (lens_match_mm) is no longer applied."""
    problems = []
    for g in HAND_GROUPS["plate_index"]:
        if not ctx.submission[g]["measurements"]:
            problems.append(f"group {g} has no measurement")
    return problems


def optics_bench_restored(ctx, max_tilt_deg=10.0):
    state = ctx.sealed.episode.get("terminal_state") or {}
    seats = state.get("lens_seat")
    if not isinstance(seats, dict) or "source_on" not in state:
        return False, {"problems": ["no record of the optics bench at submission"]}
    problems = []
    if state.get("source_on"):
        problems.append("the source is still on")
    for name, seat in sorted(seats.items()):
        tilt = (state.get("tilt_deg") or {}).get(name)
        if seat is None:
            problems.append(f"holder {name} is not in a seat")
        elif tilt is not None and tilt > max_tilt_deg:
            problems.append(f"holder {name} tilted {tilt} deg")
    if state.get("held") is not None:
        problems.append(f"the robot holds {state.get('held')}")
    return not problems, {"problems": problems, "source_on": state.get("source_on"), "seats": seats}



def bessel_focus(ctx, max_blur_mm=0.5, self_consistency_tol=0.005):
    """lens_focal (keypad bench, earlier): evidence and self-consistency of the Bessel (conjugate) method.
    
    Two cited screen captures of this episode, lens L in carriage A, the screen at the submitted position, the lens at
    the submitted positions (as recorded, 0.1 mm), each focused: the blur radius the true optics give for the
    recorded state is at most `max_blur_mm`. The submitted f agrees with (D^2 - d^2) / (4 D) from the submitted D and
    lens positions.
    """
    from benchmark.evaluation.gates import _need_submission
    import sys
    from pathlib import Path
    core = str(Path(__file__).resolve().parents[1] / "core")
    if core not in sys.path:
        sys.path.insert(0, core)
    import optics_model as M
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    sub, device = ctx.submission, ctx.sealed.episode["device"]
    lens = device["lenses"]["L"]
    problems, detail = [], {"captures": []}
    if len({s["evidence_id"] for s in sub["sharp"]}) != 2:
        problems.append("the two sharp images must be two different captures")
    for s in sub["sharp"]:
        record, problem = ctx.evidence(s["evidence_id"])
        if problem or record.get("kind") != "screen_capture":
            problems.append(f"{s['evidence_id']}: {problem or 'not a screen capture'}"); continue
        pos = record["public"]["positions_mm"]
        private = record["private"]
        if private["seated"].get("A") != "L":
            problems.append(f"{s['evidence_id']}: lens L is not in carriage A")
            continue
        if abs(pos["A"] - s["lens_mm"]) > 0.05 or abs(pos["S"] - sub["screen_mm"]) > 0.05:
            problems.append(f"{s['evidence_id']}: recorded positions A {pos['A']}, S {pos['S']} differ from the submission")
        lenses, screen = private["image_state"]
        optic = [(x / 1000, device["lenses"][name]["f_m"]) for x, name in lenses if 0 < x < screen]
        m, blur = M.magnification_and_blur(optic, screen / 1000, lens["aperture_m"])
        blur_mm = None if blur is None else blur * 1000
        detail["captures"].append({"evidence_id": s["evidence_id"], "blur_mm": blur_mm, "magnification": m})
        if blur_mm is None or blur_mm > max_blur_mm:
            problems.append(f"{s['evidence_id']}: the image is not focused (blur {blur_mm} mm)")
    if not problems:
        D = sub["screen_mm"]
        d = abs(sub["sharp"][0]["lens_mm"] - sub["sharp"][1]["lens_mm"])
        f = (D * D - d * d) / (4 * D)
        detail["refit_f_mm"] = f
        if f <= 0 or abs(sub["f_mm"] - f) / f > self_consistency_tol:
            problems.append(f"f_mm {sub['f_mm']} differs from (D^2 - d^2)/(4D) = {f:.3f} mm")
    detail["problems"] = problems
    return not problems, detail


def grating_orders(ctx, min_captures=3, min_span_mm=40, position_tol_mm=0.5, self_consistency_tol=0.005):
    """grating_wavelength: cited captures with the grating in carriage A at the recorded positions, the reported
    first-order half-separation within position_tol_mm of the true spots, and lambda from the least-squares slope of
    x1 against (screen - grating)."""
    import math
    import sys
    from pathlib import Path
    from benchmark.evaluation.gates import _need_submission
    core = str(Path(__file__).resolve().parents[1] / "core")
    if core not in sys.path:
        sys.path.insert(0, core)
    import optics_model as M
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    sub, device = ctx.submission, ctx.sealed.episode["device"]
    grating = device["lenses"]["G"]
    problems, rows = [], []
    if len({m["evidence_id"] for m in sub["measurements"]}) != len(sub["measurements"]):
        problems.append("a capture is cited twice")
    for m in sub["measurements"]:
        record, problem = ctx.evidence(m["evidence_id"])
        if problem or record.get("kind") != "screen_capture":
            problems.append(f"{m['evidence_id']}: {problem or 'not a screen capture'}"); continue
        pos, private = record["public"]["positions_mm"], record["private"]
        if private["seated"].get("A") != "G":
            problems.append(f"{m['evidence_id']}: the grating is not in carriage A"); continue
        if abs(pos["A"] - m["grating_mm"]) > 0.05 or abs(pos["S"] - m["screen_mm"]) > 0.05:
            problems.append(f"{m['evidence_id']}: recorded positions A {pos['A']}, S {pos['S']} differ from the entry")
        (elements, screen) = private["image_state"]
        x_g = next(x for x, name in elements if name == "G")
        spots = dict(M.grating_spots(device["laser_nm"] * 1e-9, grating["lines_per_mm"], (screen - x_g) / 1000))
        true_x1 = 1000 * (spots[1] - spots[-1]) / 2
        if abs(m["x1_mm"] - true_x1) > position_tol_mm:
            problems.append(f"{m['evidence_id']}: x1 {m['x1_mm']} mm is not where the first orders were ({true_x1:.2f} mm)")
        rows.append((m["screen_mm"] - m["grating_mm"], m["x1_mm"]))
    detail = {"problems": problems}
    if len(rows) < min_captures:
        problems.append(f"{len(rows)} captures, need {min_captures}")
    elif max(r[0] for r in rows) - min(r[0] for r in rows) < min_span_mm:
        problems.append(f"the grating-screen distances span less than {min_span_mm} mm")
    if not problems:
        n = len(rows)
        mx, my = sum(r[0] for r in rows) / n, sum(r[1] for r in rows) / n
        slope = sum((x - mx) * (y - my) for x, y in rows) / sum((x - mx) ** 2 for x, _ in rows)
        lam = 1e6 / grating["lines_per_mm"] * math.sin(math.atan(slope))
        detail.update(refit_tan_theta1=slope, refit_lambda_nm=lam)
        if abs(sub["lambda_nm"] - lam) / lam > self_consistency_tol:
            problems.append(f"lambda_nm {sub['lambda_nm']} differs from the refit {lam:.2f} nm")
    return not problems, detail


def slit_fringes(ctx, min_captures=3, min_span_mm=100, spacing_tol=0.05, self_consistency_tol=0.005):
    """slit_width: cited captures with the slit in carriage A at the recorded positions, each reported dark-fringe
    spacing within spacing_tol of the true lambda L / a at that capture, and a from the least-squares slope of the
    spacing against (screen - slit): slope = lambda / a."""
    from benchmark.evaluation.gates import _need_submission
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    sub, device = ctx.submission, ctx.sealed.episode["device"]
    slit = device["lenses"]["SL"]
    lam_mm = device["laser_nm"] * 1e-6
    problems, rows = [], []
    if len({m["evidence_id"] for m in sub["measurements"]}) != len(sub["measurements"]):
        problems.append("a capture is cited twice")
    for m in sub["measurements"]:
        record, problem = ctx.evidence(m["evidence_id"])
        if problem or record.get("kind") != "screen_capture":
            problems.append(f"{m['evidence_id']}: {problem or 'not a screen capture'}"); continue
        pos, private = record["public"]["positions_mm"], record["private"]
        if private["seated"].get("A") != "SL":
            problems.append(f"{m['evidence_id']}: the slit is not in carriage A"); continue
        if abs(pos["A"] - m["slit_mm"]) > 0.05 or abs(pos["S"] - m["screen_mm"]) > 0.05:
            problems.append(f"{m['evidence_id']}: recorded positions A {pos['A']}, S {pos['S']} differ from the entry")
        (elements, screen) = private["image_state"]
        x_s = next(x for x, name in elements if name == "SL")
        true_spacing = lam_mm * (screen - x_s) / (slit["width_m"] * 1000)
        if abs(m["spacing_mm"] - true_spacing) > spacing_tol * true_spacing:
            problems.append(f"{m['evidence_id']}: spacing {m['spacing_mm']} mm is not the fringes' ({true_spacing:.3f} mm)")
        rows.append((m["screen_mm"] - m["slit_mm"], m["spacing_mm"]))
    detail = {"problems": problems}
    if len(rows) < min_captures:
        problems.append(f"{len(rows)} captures, need {min_captures}")
    elif max(r[0] for r in rows) - min(r[0] for r in rows) < min_span_mm:
        problems.append(f"the slit-screen distances span less than {min_span_mm} mm")
    if not problems:
        n = len(rows)
        mx, my = sum(r[0] for r in rows) / n, sum(r[1] for r in rows) / n
        slope = sum((x - mx) * (y - my) for x, y in rows) / sum((x - mx) ** 2 for x, _ in rows)
        if slope <= 0:
            problems.append("the spacing does not grow with the distance")
        else:
            a_mm = lam_mm / slope
            detail.update(refit_slope=slope, refit_slit_width_mm=a_mm)
            if abs(sub["slit_width_mm"] - a_mm) / a_mm > self_consistency_tol:
                problems.append(f"slit_width_mm {sub['slit_width_mm']} differs from the refit {a_mm:.5f} mm")
    return not problems, detail


def plate_shift(ctx, min_captures=5, focus_tol_mm=4.0, self_consistency_tol=0.005):
    """plate_index: two groups of cited captures with lens L in carriage A at one position, the plate out of the
    beam ('without') and in carriage B ('with'); each group's focus lies within its captures' screen positions and
    within focus_tol_mm of the sharp image (a fabrication check: the plate moves it 19-27 mm); n agrees with
    1 / (1 - (with - without) / t)."""
    import sys
    from pathlib import Path
    from benchmark.evaluation.gates import _need_submission
    core = str(Path(__file__).resolve().parents[1] / "core")
    if core not in sys.path:
        sys.path.insert(0, core)
    import optics_model as M
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    sub, device = ctx.submission, ctx.sealed.episode["device"]
    lenses = device["lenses"]
    plate = next(k for k, v in lenses.items() if v["kind"] == "plate")
    t_mm = lenses[plate]["thickness_m"] * 1000
    problems, lens_x, detail = [], set(), {"groups": {}}
    cited = [e for group in ("without", "with") for e in sub[group]["evidence_ids"]]
    if len(set(cited)) != len(cited):
        problems.append("a capture is cited twice")
    for group in ("without", "with"):
        screens, truth = [], None
        for eid in sub[group]["evidence_ids"]:
            record, problem = ctx.evidence(eid)
            if problem or record.get("kind") != "screen_capture":
                problems.append(f"{eid}: {problem or 'not a screen capture'}"); continue
            seated = record["private"]["seated"]
            if seated.get("A") != "L":
                problems.append(f"{eid}: lens L is not in carriage A"); continue
            if (seated.get("B") == plate) != (group == "with") or (group == "without" and plate in seated.values()):
                problems.append(f"{eid}: the plate is {'not ' if group == 'with' else ''}in carriage B"); continue
            elements, screen = record["private"]["image_state"]
            lens_x.add(round(next(x for x, name in elements if name == "L"), 2))
            truth = 1000 * M.focus_position_m([(x / 1000, lenses[name]) for x, name in elements if 0 < x < screen])
            screens.append(record["public"]["positions_mm"]["S"])
        focus = sub[group]["focus_mm"]
        detail["groups"][group] = {"captures": len(screens), "true_focus_mm": truth}
        if len(screens) < min_captures:
            problems.append(f"{group}: {len(screens)} captures, need {min_captures}")
        elif not min(screens) <= focus <= max(screens):
            problems.append(f"{group}: focus {focus} mm lies outside the cited screen positions")
        elif truth is not None and abs(focus - truth) > focus_tol_mm:
            problems.append(f"{group}: focus {focus} mm is not where the image was sharp")
    if len(lens_x) > 1:
        problems.append("the lens moved between the captures")
    if not problems:
        shift = sub["with"]["focus_mm"] - sub["without"]["focus_mm"]
        if not 0 < shift < t_mm:
            problems.append(f"the focus shift {shift:.2f} mm is not between 0 and the plate's thickness")
        else:
            n = 1.0 / (1.0 - shift / t_mm)
            detail.update(refit_shift_mm=shift, refit_n=n)
            if abs(sub["n"] - n) / n > self_consistency_tol:
                problems.append(f"n {sub['n']} differs from 1 / (1 - shift / t) = {n:.4f}")
    detail["problems"] = problems
    return not problems, detail


def relay_image(ctx, max_blur_mm=1.0):
    """optical_relay: the cited capture of this episode shows both lenses in the carriages (one in A, one in B) and
    a focused image (blur no larger than max_blur_mm); its magnification is scored by relay_score."""
    import sys
    from pathlib import Path
    from benchmark.evaluation.gates import _need_submission
    core = str(Path(__file__).resolve().parents[1] / "core")
    if core not in sys.path:
        sys.path.insert(0, core)
    import optics_model as M
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    sub, device = ctx.submission, ctx.sealed.episode["device"]
    problems, detail = [], {}
    record, problem = ctx.evidence(sub["evidence_id"])
    if problem or record.get("kind") != "screen_capture":
        problems.append(f"{sub['evidence_id']}: {problem or 'not a screen capture'}")
    else:
        seated = record["private"]["seated"]
        if sorted(v for v in (seated.get("A"), seated.get("B")) if v) != ["L1", "L2"]:
            problems.append(f"{sub['evidence_id']}: the carriages hold {seated}, not both lenses")
        lenses, screen = record["private"]["image_state"]
        optic = [(x / 1000, device["lenses"][name]["f_m"]) for x, name in lenses if 0 < x < screen]
        first = device["lenses"][lenses[0][1]] if lenses else None
        m, blur = M.magnification_and_blur(optic, screen / 1000, first["aperture_m"]) if first else (None, None)
        blur_mm = None if blur is None else blur * 1000
        detail.update(magnification=m, blur_mm=blur_mm)
        if blur_mm is None or blur_mm > max_blur_mm:
            problems.append(f"{sub['evidence_id']}: the image is not focused (blur {blur_mm} mm)")
    detail["problems"] = problems
    return not problems, detail


def relay_magnification(sealed, evidence_id):
    """True (signed) magnification of a cited observation's (hand-slid bench) or capture's (keypad bench) recorded
    optical state, or None."""
    import sys
    from pathlib import Path
    core = str(Path(__file__).resolve().parents[1] / "core")
    if core not in sys.path:
        sys.path.insert(0, core)
    import optics_model as M
    record = sealed.evidence.get(evidence_id)
    if record and record.get("kind") == "bench_view":          # hand-slid bench: the optics of that observation
        m = (record.get("optics") or {}).get("magnification")
        return None if m is None or not record.get("source_on") else float(m)
    if not record or record.get("kind") != "screen_capture":
        return None
    device = sealed.episode["device"]
    lenses, screen = record["private"]["image_state"]
    optic = [(x / 1000, device["lenses"][name]["f_m"]) for x, name in lenses if 0 < x < screen]
    if not optic:
        return None
    return M.magnification_and_blur(optic, screen / 1000, device["lenses"][lenses[0][1]]["aperture_m"])[0]


def relay_score(spec, hidden, submission, sealed):
    """e_rel = |m_true - M| / M for the cited capture (an inverted image scores as missing the target)."""
    from benchmark.evaluation.evaluate import quantity
    target = float(hidden[spec.scoring["target"]])
    threshold = spec.scoring["threshold"]
    m = relay_magnification(sealed, submission["evidence_id"])
    e_rel = 1e9 if m is None else abs(m - target) / target
    # the submitted value is the magnification of the cited capture, read from the recorded optical state
    q = (quantity("magnification", m, target, "rel_error", threshold, e_rel) if m is not None else
         {"name": "magnification", "submitted": None, "truth": target, "abs_error": None, "rel_error": None,
          "metric": "rel_error", "threshold": threshold, "normalized_error": e_rel / threshold})
    return {"relative_errors": [e_rel], "e_rel": e_rel, "e": e_rel / threshold, "quantities": [q]}

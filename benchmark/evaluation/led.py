"""led_planck: evidence and self-consistency of the LED threshold calibration.

Each used LED cites a logger record of this episode in which that LED was selected with the supply sweeping and
the current (CH2 / R_s nominal) reached `min_current_A`; its submitted threshold voltage lies within the LED
voltages recorded there (the method is the Agent's own). At least `min_known` LEDs of printed wavelength and X are
used. The submitted wavelength agrees with the Agent's own calibration: a least-squares line V_th = slope / lambda
+ intercept through the used known LEDs, inverted at X's threshold.
"""
from __future__ import annotations


def _record_rows(ctx, eid):
    record, problem = ctx.evidence(eid)
    if problem or record.get("kind") != "circuit_logger":
        return None, problem or "not a logger record"
    return record["rows"] if "rows" in record else record["public"]["rows"], None


def led_curves(ctx, min_known=3, min_current_A=0.003, self_consistency_tol=0.005):
    from benchmark.evaluation.gates import _need_submission      # the gate library imports this module
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    sub, device = ctx.submission, ctx.sealed.episode["device"]
    known_nm = {k: float(v) for k, v in device["known_nm"].items()}
    problems, used = [], [e for e in sub["leds"] if e["used_in_fit"]]
    if len({e["led"] for e in used}) != len(used):
        problems.append("an LED is used twice")
    if "X" not in {e["led"] for e in used}:
        problems.append("LED X is not used")
    if len([e for e in used if e["led"] in known_nm]) < min_known:
        problems.append(f"fewer than {min_known} LEDs of printed wavelength used")
    for e in used:
        label = f"{e['led']} ({e['evidence_id']})"
        rows, problem = _record_rows(ctx, e["evidence_id"])
        if rows is None:
            problems.append(f"{label}: {problem}"); continue
        sweep = [r for r in rows if r["pair"] == e["led"] and r["mode"] == "SWEEP" and r["status"] == "ok"
                 and r["rs"] not in ("transition", None)]
        if not sweep:
            problems.append(f"{label}: no sweep of this LED in the record"); continue
        currents = [r["ch2_V"] / float(r["rs"]) for r in sweep]
        if max(currents) < min_current_A:
            problems.append(f"{label}: the sweep reaches {max(currents) * 1e3:.2f} mA, need {min_current_A * 1e3:g} mA")
        volts = [r["ch1_V"] for r in sweep]
        if not min(volts) - 1e-9 <= e["V_th_V"] <= max(volts) + 1e-9:
            problems.append(f"{label}: V_th {e['V_th_V']} V lies outside the recorded LED voltages")
    detail = {"problems": problems}
    xs = [(1.0 / known_nm[e["led"]], e["V_th_V"]) for e in used if e["led"] in known_nm]
    x_entry = next((e for e in used if e["led"] == "X"), None)
    if not problems and len(xs) >= 2 and x_entry is not None:
        n = len(xs)
        mx, my = sum(x for x, _ in xs) / n, sum(y for _, y in xs) / n
        sxx = sum((x - mx) ** 2 for x, _ in xs)
        slope = sum((x - mx) * (y - my) for x, y in xs) / sxx
        intercept = my - slope * mx
        if x_entry["V_th_V"] - intercept <= 0 or slope <= 0:
            problems.append("the calibration line cannot be inverted at X")
        else:
            lam = slope / (x_entry["V_th_V"] - intercept)
            detail.update(refit_slope_V_nm=slope, refit_intercept_V=intercept, refit_lambda_x_nm=lam)
            if abs(sub["lambda_x_nm"] - lam) / lam > self_consistency_tol:
                problems.append(f"lambda_x_nm {sub['lambda_x_nm']} differs from the calibration {lam:.2f} nm by more "
                                f"than {self_consistency_tol:.1%}")
    return not problems, detail

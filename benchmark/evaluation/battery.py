"""battery_emf: evidence and self-consistency of the U-I line.

Each used point cites a logger record of this episode with rows at that load and shunt with the switch ON; its
U_V lies within the recorded CH1 values and its I_A within the recorded CH2 / R_sh (nominal) values there. At least
`min_loads` different loads are used. The submitted E and r agree with the least-squares line U = E - r I through
the used points.
"""
from __future__ import annotations


def _record_rows(ctx, eid):
    record, problem = ctx.evidence(eid)
    if problem or record.get("kind") != "circuit_logger":
        return None, problem or "not a logger record"
    return record["rows"] if "rows" in record else record["public"]["rows"], None


def line(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


def battery_lines(ctx, min_loads=3, emf_tol=0.002, r_tol=0.01):
    from benchmark.evaluation.gates import _need_submission      # the gate library imports this module
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    sub = ctx.submission
    problems, used = [], [p for p in sub["points"] if p["used_in_fit"]]
    if len({p["load"] for p in used}) < min_loads:
        problems.append(f"fewer than {min_loads} different loads used")
    for p in used:
        label = f"load {p['load']} ohm ({p['evidence_id']})"
        rows, problem = _record_rows(ctx, p["evidence_id"])
        if rows is None:
            problems.append(f"{label}: {problem}"); continue
        on = [r for r in rows if str(r["pair"]) == p["load"] and r["mode"] == "ON" and r["status"] == "ok"
              and r["rs"] not in ("transition", None) and abs(float(r["rs"]) - p["shunt_ohm"]) < 1e-9]
        if not on:
            problems.append(f"{label}: no rows at this load and shunt with the switch ON"); continue
        volts, amps = [r["ch1_V"] for r in on], [r["ch2_V"] / p["shunt_ohm"] for r in on]
        if not min(volts) - 1e-9 <= p["U_V"] <= max(volts) + 1e-9:
            problems.append(f"{label}: U {p['U_V']} V lies outside the recorded terminal voltages")
        if not min(amps) - 1e-9 <= p["I_A"] <= max(amps) + 1e-9:
            problems.append(f"{label}: I {p['I_A']} A lies outside the recorded currents")
    detail = {"problems": problems}
    if not problems:
        fit = line([p["I_A"] for p in used], [p["U_V"] for p in used])
        if fit is None or fit[0] >= 0:
            problems.append("the U-I line has no negative slope")
        else:
            emf, r = fit[1], -fit[0]
            detail.update(refit_E_V=emf, refit_r_ohm=r)
            if abs(sub["E_V"] - emf) / emf > emf_tol:
                problems.append(f"E_V {sub['E_V']} differs from the line's intercept {emf:.4f} V by more than {emf_tol:.1%}")
            if abs(sub["r_ohm"] - r) / r > r_tol:
                problems.append(f"r_ohm {sub['r_ohm']} differs from the line's slope {r:.4f} ohm by more than {r_tol:.0%}")
    return not problems, detail

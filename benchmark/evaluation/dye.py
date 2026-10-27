"""dye_absorptivity: evidence and self-consistency of a Beer-Lambert line for an unknown dye.

Every cited READ is checked as in spectro_basic (logged A, BLANK on water at the reading wavelength, >= 2 mL, mixed, not
overflowed); a point's concentration comes from the cuvette's nominal dispensing history and the two stocks' label
values. All points are read at the submitted lambda_max_nm, which lies within wavelength_tol_nm of the dye's
absorption maximum. The refit: least-squares A = k c + b through the used points; eps = k x 1e6 / path (L mol^-1 cm^-1).
Scans never count as evidence.
"""
from __future__ import annotations

STATUS_PROBLEM = {"slot_empty": "no cuvette in the well", "low_volume": "less than the minimum reading volume",
                  "not_mixed": "cuvette not mixed", "no_blank_at_λ": "no BLANK at the reading wavelength",
                  "blank_ref_not_solvent": "BLANK reference was not water only", "overflowed": "cuvette overflowed"}


def _mix(history):
    volumes = {}
    for source, mL in history:
        volumes[source] = volumes.get(source, 0.0) + float(mL)
    return volumes


def dye_curve(ctx, min_points=3, composition_tol=0.01, wavelength_tol_nm=3, self_consistency_tol=0.02):
    from benchmark.evaluation.gates import _line_fit, _need_submission
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    sub, device = ctx.submission, ctx.sealed.episode["device"]
    labels = device["c_label_uM"]
    nm = int(sub["lambda_max_nm"])
    problems, xs, ys, cuvettes = [], [], [], set()
    if abs(nm - float(ctx.hidden["lambda_max_nm"])) > wavelength_tol_nm:
        problems.append(f"lambda_max_nm {nm} is more than {wavelength_tol_nm} nm from the dye's absorption maximum")
    used = [p for p in sub["calibration"]["points"] if p["used_in_fit"]]
    if len({p["evidence_id"] for p in used}) != len(used):
        problems.append("a reading is cited more than once")
    for i, p in enumerate(used):
        label = f"point {i} ({p['evidence_id']})"
        record, problem = ctx.evidence(p["evidence_id"])
        if problem or record.get("kind") != "spectro_read":
            problems.append(f"{label}: {problem or 'not a spectrophotometer reading'}"); continue
        public = record["public"]
        if record["status"] != "ok":
            problems.append(f"{label}: {STATUS_PROBLEM.get(record['status'], record['status'])}")
        if record.get("spilled"):
            problems.append(f"{label}: the cuvette had been spilled before this reading")
        if public["wavelength_nm"] != nm or public["blank_wavelength_nm"] != nm:
            problems.append(f"{label}: not read at {nm} nm after a BLANK there")
        if abs(p["A"] - public["A_display"]) > 5e-4:
            problems.append(f"{label}: A {p['A']} differs from the logged {public['A_display']}")
        mix = _mix(record["nominal_history"])
        total = sum(mix.values())
        if total <= 0 or set(mix) - {"stock", "water", "unknown"}:
            problems.append(f"{label}: the cuvette holds no standard"); continue
        c = sum(labels[s] * mix.get(s, 0.0) for s in ("stock", "unknown")) / total
        if abs(p["c_uM"] - c) > composition_tol * c + 1e-9:
            problems.append(f"{label}: c_uM {p['c_uM']} disagrees with the dispensing history ({c:.4f} uM)")
        xs.append(c); ys.append(float(public["A_display"])); cuvettes.add(record["cuvette"])
    if len(cuvettes) < min_points:
        problems.append(f"points come from {len(cuvettes)} cuvettes, need {min_points}")
    if xs and len(set(xs)) < 2:
        problems.append("all points share one concentration; the line cannot be refit")
    detail = {"problems": problems}
    if not problems:
        slope, intercept, r2 = _line_fit(xs, ys)
        eps = slope * 1e6
        detail.update(refit_slope_per_uM=slope, refit_intercept=intercept, r2=r2, refit_eps=eps)
        if slope <= 0 or abs(sub["epsilon_L_per_mol_cm"] - eps) / eps > self_consistency_tol:
            problems.append(f"epsilon {sub['epsilon_L_per_mol_cm']} differs from the refit {eps:.0f} by more than "
                            f"{self_consistency_tol:.0%}")
    return not problems, detail

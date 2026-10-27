"""spectro_mixture: evidence and self-consistency of a two-dye mixture analysis.

Every cited READ is checked as in spectro_basic (logged A, BLANK on water at the reading wavelength, >= 2 mL, mixed, not
overflowed) and its composition against the cuvette's nominal dispensing history: a calibration point's
(c_X, c_Y) from the two stock lines and their label values, a mixture reading's dilution factor from the
handed-out mixture cuvette. The refit is Beer-Lambert for two absorbers: at each wavelength a least-squares
A = k_X c_X + k_Y c_Y + b through the used calibration points (a dye with no non-zero point there gets k = 0),
then the mixture's dilution-corrected readings, intercepts removed, solved for (c_X, c_Y) over the wavelengths.
Scans never count as evidence.
"""
from __future__ import annotations

import numpy as np

STATUS_PROBLEM = {"slot_empty": "no cuvette in the well", "low_volume": "less than the minimum reading volume",
                  "not_mixed": "cuvette not mixed", "no_blank_at_λ": "no BLANK at the reading wavelength",
                  "blank_ref_not_solvent": "BLANK reference was not water only", "overflowed": "cuvette overflowed"}
DYES = ("X", "Y")


def _mix(history):
    volumes = {}
    for source, mL in history:
        volumes[source] = volumes.get(source, 0.0) + float(mL)
    return volumes


def logged(ctx, entry, role):
    """(wavelength, A, composition, record, problem): composition is (c_X, c_Y) in uM for a calibration point and
    the dilution factor for a mixture reading, both from the sealed dispensing history."""
    record, problem = ctx.evidence(entry["evidence_id"])
    if problem or record.get("kind") != "spectro_read":
        return None, None, None, None, problem or "not a spectrophotometer reading"
    device = ctx.sealed.episode["device"]
    lines, mixture = device["source_of"], device["mixture"]
    mix = _mix(record["nominal_history"])
    total = sum(mix.values())
    public = record["public"]
    wavelength, a = int(public["wavelength_nm"]), float(public["A_display"])
    if total <= 0:
        return wavelength, a, None, record, "the cuvette in the well was empty"
    if role == "calibration":
        if mix.get(mixture["label"], 0.0) > 0:
            return wavelength, a, None, record, "calibration cuvette holds the mixture"
        value = tuple(device["c_stock_uM_label"][d] * mix.get(lines[d], 0.0) / total for d in DYES)
    else:
        if record["cuvette"] != mixture["cuvette"] or any(mix.get(lines[d], 0.0) > 0 for d in DYES):
            return wavelength, a, None, record, "mixture reading is not the handed-out mixture (diluted with water at most)"
        value = total / mix[mixture["label"]]
    return wavelength, a, value, record, None


def _fit(points):
    """Least squares A = k_X c_X + k_Y c_Y + b at one wavelength: ({dye: k}, b, rms). Raises if undetermined."""
    active = [i for i in range(2) if any(abs(c[i]) > 0 for c, _ in points)]
    if not active:
        raise ValueError("no calibration point holds dye")
    X = np.array([[c[i] for i in active] + [1.0] for c, _ in points])
    y = np.array([a for _, a in points])
    if len(points) < X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("calibration points do not determine the sensitivities and the intercept")
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    k = {d: 0.0 for d in DYES}
    for j, i in enumerate(active):
        k[DYES[i]] = float(coef[j])
    rms = float(np.sqrt(np.mean((X @ coef - y) ** 2)))
    return k, float(coef[-1]), rms


def refit(points_by_nm, mixture_by_nm):
    """(c_X, c_Y), detail from {nm: [((c_X, c_Y), A)]} and {nm: [(A, dilution)]}."""
    rows, rhs, fits, diagonal = [], [], {}, {}
    for nm in sorted(mixture_by_nm):
        k, b, rms = _fit(points_by_nm[nm])
        y = float(np.mean([(a - b) * f for a, f in mixture_by_nm[nm]]))
        rows.append([k["X"], k["Y"]]); rhs.append(y)
        fits[str(nm)] = {"k_X_per_uM": k["X"], "k_Y_per_uM": k["Y"], "intercept": b, "rms": rms, "mixture_A_corrected": y}
    K = np.array(rows)
    if np.linalg.matrix_rank(K) < 2:
        raise ValueError("the sensitivity matrix over the wavelengths is singular")
    c, *_ = np.linalg.lstsq(K, np.array(rhs), rcond=None)
    if len(rows) == 2:
        # Diagnostic: the answer if the cross sensitivities were ignored (each dye from the wavelength where it
        # absorbs most).
        for i, d in enumerate(DYES):
            j = int(np.argmax(K[:, i]))
            diagonal[d] = rhs[j] / K[j, i]
    return (float(c[0]), float(c[1])), {"fits": fits, "condition_number": float(np.linalg.cond(K / np.abs(K).max(axis=0))),
                                         "diagonal_only_uM": diagonal}


def spectro_mixture(ctx, min_points_per_wavelength=3, min_wavelengths=2, composition_tol=0.01, self_consistency_tol=0.03):
    from benchmark.evaluation.gates import _need_submission      # the gate library imports this module
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    sub = ctx.submission
    problems, points_by_nm, mixture_by_nm = [], {}, {}
    wavelengths = set(int(w) for w in sub["wavelengths_nm"])
    if len(wavelengths) != len(sub["wavelengths_nm"]):
        problems.append("a wavelength is listed twice")
    used = [("calibration", e) for e in sub["calibration_points"] if e["used_in_fit"]] + \
           [("mixture", e) for e in sub["mixture_measurements"] if e["used_in_fit"]]
    cited = [e["evidence_id"] for _, e in used]
    if len(set(cited)) != len(cited):
        problems.append("a reading is cited more than once")
    min_read = ctx.sealed.episode["device"]["min_read_mL"]
    for n, (role, entry) in enumerate(used):
        label = f"{role} {entry['evidence_id']}"
        nm, a, value, record, problem = logged(ctx, entry, role)
        if record is None:
            problems.append(f"{label}: {problem}"); continue
        if problem:
            problems.append(f"{label}: {problem}"); continue
        if record["status"] != "ok":
            problems.append(f"{label}: {STATUS_PROBLEM.get(record['status'], record['status'])}")
        if record.get("spilled"):
            problems.append(f"{label}: the cuvette had been spilled")
        if record["volume_mL"] is None or record["volume_mL"] < min_read - 1e-9:
            problems.append(f"{label}: less than the minimum reading volume")
        if record["public"]["blank_wavelength_nm"] != nm:
            problems.append(f"{label}: no BLANK at the reading wavelength")
        if nm not in wavelengths:
            problems.append(f"{label}: read at {nm} nm, not one of the submitted wavelengths")
        if abs(entry["A"] - a) > 5e-4:
            problems.append(f"{label}: A {entry['A']} differs from the logged {a}")
        if role == "calibration":
            claimed = (entry["c_X_uM"], entry["c_Y_uM"])
            for d, c, v in zip(DYES, claimed, value):
                if abs(c - v) > composition_tol * v + 1e-6:
                    problems.append(f"{label}: c_{d}_uM {c} disagrees with the dispensing history ({v:.4g})")
            points_by_nm.setdefault(nm, []).append((value, a))
        else:
            if abs(entry["dilution_factor"] - value) > composition_tol * value:
                problems.append(f"{label}: dilution_factor {entry['dilution_factor']} disagrees with the cuvette ({value:.4g})")
            mixture_by_nm.setdefault(nm, []).append((a, value))
    if len(wavelengths) < min_wavelengths:
        problems.append(f"{len(wavelengths)} wavelengths, need at least {min_wavelengths}")
    for nm in sorted(wavelengths):
        if len(points_by_nm.get(nm, [])) < min_points_per_wavelength:
            problems.append(f"{nm} nm: {len(points_by_nm.get(nm, []))} calibration points, need {min_points_per_wavelength}")
        if not mixture_by_nm.get(nm):
            problems.append(f"{nm} nm: no mixture reading")
    detail = {"problems": problems}
    if not problems:
        try:
            (cx, cy), fit = refit(points_by_nm, mixture_by_nm)
            detail.update(fit, refit_uM={"X": cx, "Y": cy})
            for d, value, submitted in (("X", cx, sub["concentration_X_uM"]), ("Y", cy, sub["concentration_Y_uM"])):
                if value <= 0:
                    problems.append(f"refit c_{d} is not positive")
                elif abs(submitted - value) / value > self_consistency_tol:
                    problems.append(f"concentration_{d}_uM {submitted} differs from the refit {value:.4g} by more than "
                                    f"{self_consistency_tol:.0%}")
        except ValueError as exc:
            problems.append(str(exc))
    return not problems, detail

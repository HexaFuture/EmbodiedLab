"""titration_weak_acid: evidence and self-consistency of a KHP standardization and a weak-acid titration.

Each cited session is checked as in titration_basic (rows of the session, stability mark, electrode in the liquid, counter
equal to the titrant delivered into that beaker, logged volume and pH). The standardization cites the KHP beaker's
session with the label mass; the sample cites an acid beaker's session. In both, the steepest step between
consecutive cited points spans at most `endpoint_gap_mL`, and the cited points around it rise at least `min_jump_pH`
within `endpoint_gap_mL` (the jump may straddle two fine additions; the equivalence pH differs between KHP and the
acids); the sample's half-equivalence volume must lie
between two cited points at most `half_gap_fraction` x V_eq apart. The refit uses only the cited points: endpoint =
midpoint of the steepest step, c_NaOH = m / (M_KHP V_eq,KHP), c_HA = c_NaOH V_eq / V_sample (nominal), pKa = pH
interpolated at V_eq / 2.
"""
from __future__ import annotations

KHP_G_PER_MOL = 204.22


def endpoint(points):
    """(V_eq, i): midpoint of the steepest step of [(V, pH, ...)] sorted by volume."""
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
    return 0.5 * (a[0] + b[0]), best[1]


def jump_around(points, i, gap):
    """Largest pH rise between cited points at most `gap` mL apart that span the step i -> i + 1."""
    best = points[i + 1][1] - points[i][1]
    for j in range(i, -1, -1):
        for k in range(i + 1, len(points)):
            if points[k][0] - points[j][0] > gap + 1e-9:
                break
            best = max(best, points[k][1] - points[j][1])
    return best


def ph_at(points, volume):
    """Linear interpolation of the pH at a volume between the two cited points around it: (pH, gap mL)."""
    below = [p for p in points if p[0] <= volume]
    above = [p for p in points if p[0] >= volume]
    if not below or not above:
        raise ValueError(f"no cited points on both sides of {volume:.3f} mL")
    a, b = below[-1], above[0]
    if b[0] - a[0] <= 1e-9:
        return a[1], 0.0
    return a[1] + (volume - a[0]) / (b[0] - a[0]) * (b[1] - a[1]), b[0] - a[0]


def _session(ctx, entry, label, problems, require):
    from benchmark.evaluation.gates import _titration_logged
    points, record, found = _titration_logged(ctx, entry)
    problems += [f"{label}: {p}" for p in found]
    if record is None:
        return None, None
    if entry["beaker"] != record["public"]["beaker"]:
        problems.append(f"{label}: beaker {entry['beaker']} but the session is on {record['public']['beaker']}")
    for v, ph, row, private, claimed in points:
        where = f"{label} sample {row['sample_id']}"
        if abs(claimed["V_mL"] - v) > 5e-3 or abs(claimed["pH"] - ph) > 5e-3:
            problems.append(f"{where}: ({claimed['V_mL']} mL, pH {claimed['pH']}) differs from the logged ({v} mL, pH {ph})")
        if not row["stable"]:
            problems.append(f"{where}: reading without the stability mark")
        if not private["immersed"]:
            problems.append(f"{where}: electrode not in the liquid")
        if private["overflowed"] or private["titrant_nominal_mL"] is None or abs(private["titrant_nominal_mL"] - v) > 5e-3:
            problems.append(f"{where}: the counter does not equal the titrant delivered into the beaker")
    gap, jump = require
    try:
        _, i = endpoint(points)
        (va, pa, *_), (vb, pb, *_) = points[i], points[i + 1]
        if vb - va > gap + 1e-9 or jump_around(points, i, gap) < jump:
            problems.append(f"{label}: the steepest step ({va} to {vb} mL, pH {pa} to {pb}) is not part of a jump of "
                            f"at least {jump} pH within {gap} mL")
    except ValueError as exc:
        problems.append(f"{label}: {exc}")
    return [(v, ph) for v, ph, *_ in points], record


def weak_acid_titration(ctx, endpoint_gap_mL=0.05, min_jump_pH=1.0, half_gap_fraction=0.25,
                        c_tol=0.01, pka_tol=0.05):
    from benchmark.evaluation.gates import _need_submission      # the gate library imports this module
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    sub, device = ctx.submission, ctx.sealed.episode["device"]
    problems, detail = [], {}
    std, smp = sub["standardization"], sub["sample_titration"]
    if std["evidence_id"] == smp["evidence_id"]:
        problems.append("the standardization and the sample cite the same session")
    if std["beaker"] != device["khp_beaker"]:
        problems.append(f"standardization on {std['beaker']}, the KHP is in {device['khp_beaker']}")
    if abs(std["khp_mass_g"] - device["khp_mass_g"]) > 5e-5:
        problems.append(f"khp_mass_g {std['khp_mass_g']} is not the label mass {device['khp_mass_g']}")
    if smp["beaker"] not in device["sample_beakers"]:
        problems.append(f"sample titration on {smp['beaker']}, not a sample beaker")
    if abs(smp["V_sample_mL"] - device["sample_nominal_mL"]) > 1e-9:
        problems.append(f"V_sample_mL {smp['V_sample_mL']} is not the nominal {device['sample_nominal_mL']} mL")
    bracket = (float(ctx.param(endpoint_gap_mL)), float(ctx.param(min_jump_pH)))
    k_points, _ = _session(ctx, std, "standardization", problems, bracket)
    s_points, _ = _session(ctx, smp, "sample", problems, bracket)
    if k_points is not None and s_points is not None and not problems:
        try:
            v_k, _ = endpoint(k_points)
            v_s, _ = endpoint(s_points)
            c_naoh = device["khp_mass_g"] / KHP_G_PER_MOL / (v_k * 1e-3)
            c_ha = c_naoh * v_s / device["sample_nominal_mL"]
            pka, gap = ph_at(s_points, v_s / 2)
            if gap > half_gap_fraction * v_s + 1e-9:
                problems.append(f"the cited points around the half-equivalence volume are {gap:.2f} mL apart "
                                f"(at most {half_gap_fraction:.0%} of V_eq)")
            detail.update(refit={"V_eq_KHP_mL": v_k, "c_NaOH_M": c_naoh, "V_eq_mL": v_s, "c_HA_M": c_ha, "pKa": pka})
            if abs(sub["c_HA_M"] - c_ha) / c_ha > c_tol:
                problems.append(f"c_HA_M {sub['c_HA_M']} differs from the refit {c_ha:.5f} by more than {c_tol:.0%}")
            if abs(sub["pKa"] - pka) > pka_tol:
                problems.append(f"pKa {sub['pKa']} differs from the refit {pka:.3f} by more than {pka_tol}")
        except ValueError as exc:
            problems.append(str(exc))
    detail["problems"] = problems
    return not problems, detail

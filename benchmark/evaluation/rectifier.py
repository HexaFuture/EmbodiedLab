"""rectifier_supply: sealed-truth scoring and the protocol gate. Nothing here refits the submitted values.

The minimum load of each capacitor module is the periodic steady state of the published circuit with the hidden
component values (adaptive DOP853 and a bracketed Shockley current solve, independent of the device's fixed-step
kernel). The final mean voltage and ripple are scored against the periodic steady state of the final supply shown by
the logger record cited as final_configuration.evidence_id: its capacitor block's hidden capacitance, the hidden diode
factor and the resistance across the output in that record's bench state (: the bench is tidied before
the submission, so the device's post-submission waveform is not used).
"""
from __future__ import annotations

import functools
import hashlib
import json
import math

import numpy as np


MODULES = ("C100", "C220", "C470")
NOMINAL_F = (1.0e-4, 2.2e-4, 4.7e-4)
FACTORS = ("c100_factor", "c220_factor", "c470_factor")
MEAN_MIN_V, RIPPLE_MAX_V = 8.5, 0.8
LOAD_TOL, MEAN_TOL, RIPPLE_ABS_V, RIPPLE_REL = 0.01, 0.005, 0.01, 0.01


def digest(value):
    """Canonical hash of a scope record (the device's rule)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@functools.lru_cache(maxsize=4096)
def steady(c, n, r):
    """(mean V, peak-to-peak ripple V) of the output in periodic steady state: the fixed point of the map from the
    output voltage at the start of a rectified period (0.01 s) to that at its end, by the secant method on single
    periods (the map is increasing and contracting)."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
    a = 2 * n * 8.617333262145e-5 * 298

    def rhs(t, y):
        drive = abs(12 * math.sin(2 * math.pi * 50 * t)) - y[0]
        current = brentq(lambda i: 25 * i + a * math.log1p(i / 1e-8) - drive, 0, drive / 25, xtol=1e-14) if drive > 0 else 0.0
        return [(current - y[0] / r) / c]

    def period(v):
        sol = solve_ivp(rhs, (0, 0.01), [v], method="DOP853", rtol=1e-10, atol=1e-11, max_step=1 / 48000, dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        return float(sol.y[0, -1]) - v, sol

    x0, x1 = 11.0, 8.0
    f0 = period(x0)[0]
    for _ in range(60):
        f1, sol = period(x1)
        if abs(f1) < 1e-10:
            samples = sol.sol(np.arange(1, 481) / 48000)[0]
            return float(samples.mean()), float(np.ptp(samples))
        x0, f0, x1 = x1, f1, x1 - f1 * (x1 - x0) / (f1 - f0)
    raise RuntimeError("no periodic steady state")


@functools.lru_cache(maxsize=256)
def minimum_load(c, n):
    """Smallest load resistance (ohm) at which the module meets both requirements."""
    from scipy.optimize import brentq

    def residual(r):
        mean, ripple = steady(c, n, r)
        return max(MEAN_MIN_V - mean, ripple - RIPPLE_MAX_V)

    return float(brentq(residual, 150.0, 3300.0, xtol=0.005))


def truth_limits(hidden):
    return {m: minimum_load(c * float(hidden[k]), float(hidden["diode_n"])) for m, c, k in zip(MODULES, NOMINAL_F, FACTORS)}


def final_supply(hidden, submission, sealed):
    """(mean V, ripple V) in steady state of the configuration in the cited final record (one bench state: no change
    on the board during it, changes in the tray do not count (circuit_grid.py); exactly one capacitor block on the
    board and across the output, a load across it), or None."""
    from benchmark.evaluation.circuit_grid import BenchChanges, process_state_log
    from benchmark.evaluation.circuit_grid_variants import rect_final_record
    config = submission.get("final_configuration") or {}
    _, rows, found = rect_final_record(sealed.evidence, config.get("evidence_id"), float(sealed.episode["physics_hz"]))
    found = BenchChanges(process_state_log(sealed)).snapshots(found)     # only board changes count
    if not rows or not found or len({s["change_tick"] for s in found}) != 1:
        return None
    out = found[0].get("output") or {}
    caps = out.get("caps_across") or []
    if out.get("shorted") or out.get("load_ohm") is None or len(caps) != 1 or out.get("caps_on_board") != caps \
            or caps[0] not in MODULES:
        return None
    k = MODULES.index(caps[0])
    return steady(NOMINAL_F[k] * float(hidden[FACTORS[k]]), float(hidden["diode_n"]), float(out["load_ohm"]))


def score(hidden, submission, sealed):
    """Normalised errors: each minimum load / 1 %, final mean / 0.5 %, final ripple / max(0.01 V, 1 %)."""
    from benchmark.evaluation.evaluate import quantity
    limits = truth_limits(hidden)
    quantities = []
    for row in submission["load_limits"]:
        error = abs(row["minimum_load_ohm"] / limits[row["module_id"]] - 1)
        quantities.append(quantity(f"minimum_load_ohm.{row['module_id']}", row["minimum_load_ohm"],
                                   limits[row["module_id"]], "rel_error", LOAD_TOL, error))
    predicted = submission["final_measurement"]
    final = final_supply(hidden, submission, sealed)
    if final:
        # the truth of the final measurement is the steady output of the final supply the cited record shows
        mean, ripple = final
        error = abs(predicted["mean_voltage_V"] / mean - 1) if mean > 0 else math.inf
        quantities.append(quantity("final_measurement.mean_voltage_V", predicted["mean_voltage_V"], mean,
                                   "rel_error", MEAN_TOL, error))
        tolerance = max(RIPPLE_ABS_V, RIPPLE_REL * ripple)
        quantities.append(quantity("final_measurement.ripple_pp_V", predicted["ripple_pp_V"], ripple,
                                   "abs_error", tolerance))
    else:
        quantities.append({"name": "final_measurement", "submitted": None, "truth": None, "abs_error": None,
                           "rel_error": None, "metric": "rel_error", "threshold": MEAN_TOL,
                           "normalized_error": math.inf})
    return {"relative_errors": None, "e_rel": None, "e": max(q["normalized_error"] for q in quantities),
            "quantities": quantities}


def rectifier_protocol(ctx):
    """The device accepted the submission (schema, own-episode complete scope records, final configuration within
    2 % of the task load), the cited records are intact, and the configuration actually left running meets the supply
    requirements in the waveform recorded after the submission."""
    from benchmark.evaluation.gates import _need_submission      # the gate library imports this module
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    device = ctx.sealed.episode["device"]
    problems = []
    if device.get("validation_problem"):
        problems.append(f"device rejected the submission: {device['validation_problem']}")
    for eid in ctx.submission["evidence_ids"]:
        record, problem = ctx.evidence(eid)
        if problem or record.get("kind") != "rectifier_scope":
            problems.append(problem or f"{eid}: not a scope record")
    post = device.get("post_submit")
    if not post:
        problems.append("no waveform was recorded after the submission")
    else:
        if not post["configuration_valid"]:
            problems.append("the final configuration was not held during the verification")
        if post["mean_voltage_V"] < MEAN_MIN_V or post["ripple_pp_V"] > RIPPLE_MAX_V:
            problems.append(f"final supply {post['mean_voltage_V']:.3f} V, ripple {post['ripple_pp_V']:.3f} V "
                            f"does not meet {MEAN_MIN_V} V / {RIPPLE_MAX_V} V")
    return not problems, {"problems": problems, "post_submit": post}

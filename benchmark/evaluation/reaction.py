"""reaction_yield gates and score.

- reaction_product: the product cuvette the submission names is one of the bench's cuvettes, not the P standard, did
  not spill or overflow, holds what `conditions` says (its nominal dispensing history: S STOCK, R STOCK and BUFFER each
  within dose_tol_ml of the stated volume, any QUENCH, and nothing else), got s_ml of S STOCK (the batch the task fixes), stood in a block well at some time and, whenever it did,
  the block's set point was conditions.block_set_C (the private heat log, one row a second); at submission it stands
  in a rack well.
- reaction_readings_genuine: every cited reading is an absorbance the spectrophotometer screen showed from a READ at a
  cited observation (chem_gates.screen_reading), of the cuvette the row names; at least one is of the product cuvette
  and at least one of the P standard while it still held only the standard.

Score (metric reaction_yield): y is the product cuvette's yield at submission, n(P) / n(S added) from the bench's
private compositions (S added counted in S atoms, reaction_model.s_moles); y* the largest attainable yield of the
instance (reaction_model.optimum: under the protocol this bench and robot attain).
e = max((1 - min(y / y*, 1)) / threshold, |predicted_yield - y| / (predicted_threshold y)): y above y* counts as y*.
"""
from __future__ import annotations

from benchmark.core import reaction_model as RM

LINES = {"S_mL": "stock", "R_mL": "unknown", "buffer_mL": "water"}


def _device(ctx):
    return ctx.sealed.episode.get("device") or {}


def product_yield(ctx, cuvette):
    """(yield, moles by species) of a cuvette at submission (private compositions)."""
    layers = ((_device(ctx).get("chemistry") or {}).get("compositions") or {}).get(str(cuvette)) or \
        ((_device(ctx).get("chemistry") or {}).get("compositions") or {}).get(cuvette) or []
    moles = {}
    for _, v, comp in layers:
        for k, c in comp.items():
            moles[k] = moles.get(k, 0.0) + c * v * 1e-3
    s_added = RM.s_moles(moles)
    return (moles.get(RM.P, 0.0) / s_added if s_added > 0 else 0.0), moles


def _history(ctx, cuvette):
    containers = (_device(ctx).get("chemistry") or {}).get("containers") or {}
    return containers.get(str(cuvette)) or containers.get(cuvette)


def reaction_product(ctx, dose_tol_ml=0.05, s_ml=1.0):
    from benchmark.evaluation.gates import _need_submission
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    sub, device = ctx.submission, _device(ctx)
    cid, cond = int(sub["product_cuvette"]), sub["conditions"]
    record = _history(ctx, cid)
    if record is None:
        return False, {"problems": [f"there is no cuvette {cid}"]}
    problems = []
    if cid == int(device["p_standard"]["cuvette"]):
        problems.append("the product cuvette is the P standard")
    if record.get("spilled") or record.get("overflowed"):
        problems.append(f"cuvette {cid} spilled or overflowed")
    dosed = {}
    for source, ml in record.get("nominal") or []:
        dosed[source] = dosed.get(source, 0.0) + float(ml)
    tol = float(ctx.param(dose_tol_ml))
    for field, line in LINES.items():
        if abs(dosed.pop(line, 0.0) - float(cond[field])) > tol:
            problems.append(f"cuvette {cid} did not get {cond[field]} mL from the {field[:-3]} line")
    dosed.pop("quench", None)                      # the quench is the agent's to use, in any amount
    if dosed:
        problems.append(f"cuvette {cid} holds other liquids: {sorted(dosed)}")
    if abs(float(cond["S_mL"]) - float(ctx.param(s_ml))) > tol:
        problems.append(f"the product must get {ctx.param(s_ml)} mL of S STOCK, not {cond['S_mL']} mL")
    rows = [r for r in (device.get("chemistry") or {}).get("heat_log") or []
            if cid in {int(v) for v in (r.get("wells") or {}).values() if v is not None}]
    if not rows:
        problems.append(f"cuvette {cid} never stood in a block well")
    sets = sorted({int(r["set_C"]) for r in rows})
    if any(s != int(cond["block_set_C"]) for s in sets):
        problems.append(f"the block was set to {sets} C while cuvette {cid} stood in it, not {cond['block_set_C']} C")
    seat = ((ctx.sealed.episode.get("terminal_state") or {}).get("cuvette_seat") or {})
    where = seat.get(str(cid), seat.get(cid))
    if not (isinstance(where, str) and where.startswith("rack:")):
        problems.append(f"cuvette {cid} is not standing in a rack well ({where or 'not seated'})")
    return not problems, {"problems": problems, "dosed_mL": record.get("nominal"), "block_sets_C": sets}


def reaction_readings_genuine(ctx):
    from benchmark.evaluation.chem_gates import screen_reading
    from benchmark.evaluation.gates import _need_submission
    if (invalid := _need_submission(ctx)) is not None:
        return invalid
    sub, device = ctx.submission, _device(ctx)
    product, standard = int(sub["product_cuvette"]), int(device["p_standard"]["cuvette"])
    problems, rows, seen = [], [], set()
    for i, reading in enumerate(sub["readings"]):
        row, problem = screen_reading(ctx, reading["evidence_id"], reading["A"], reading.get("wavelength_nm"))
        if problem:
            problems.append(f"reading {i}: {problem}")
            continue
        private = row.get("read") or {}
        if private.get("cuvette") != int(reading["cuvette"]):
            problems.append(f"reading {i}: that READ measured cuvette {private.get('cuvette')}, not {reading['cuvette']}")
            continue
        if private.get("cuvette") == standard and [s for s, _ in private.get("nominal_history") or []] != ["P STD"]:
            problems.append(f"reading {i}: the P standard had been added to before that READ")
            continue
        seen.add(private.get("cuvette"))
        rows.append({"evidence_id": reading["evidence_id"], "cuvette": private.get("cuvette"), "A": row["A_shown"],
                     "wavelength_nm": row.get("wavelength_nm")})
    if product not in seen:
        problems.append("no cited reading of the product cuvette")
    if standard not in seen:
        problems.append("no cited reading of the P standard")
    return not problems, {"problems": problems, "rows": rows}


def reaction_score(spec, hidden, submission, ctx):
    threshold, predicted = float(spec.scoring["threshold"]), float(spec.scoring["predicted_threshold"])
    labels = {"c_S_uM": float(hidden["c_S_uM"]), "c_R_uM": float(hidden["c_R_uM"])}
    y_star, set_star, recipe_star, pull_star, _ = RM.optimum({k: float(hidden[k]) for k in RM.HIDDEN}, labels)
    y, moles = product_yield(ctx, int(submission["product_cuvette"]))
    yhat = float(submission["predicted_yield"])
    ratio = min(y / y_star, 1.0)                    # clipped: a yield above y* counts as y*
    shortfall = (1.0 - ratio) / threshold
    prediction = abs(yhat - y) / (predicted * y) if y > 0 else float("inf")
    e = max(shortfall, prediction)
    quantities = [
        {"name": "yield", "submitted": y, "truth": y_star, "abs_error": max(0.0, y_star - y), "rel_error": 1.0 - ratio,
         "metric": "shortfall", "threshold": threshold, "normalized_error": shortfall},
        {"name": "predicted_yield", "submitted": yhat, "truth": y, "abs_error": abs(yhat - y),
         "rel_error": abs(yhat - y) / y if y > 0 else None, "metric": "rel_error", "threshold": predicted,
         "normalized_error": prediction}]
    return {"relative_errors": None, "e_rel": None, "e": e, "quantities": quantities,
            "diagnostics": {"y": y, "y_star": y_star, "y_over_y_star": y / y_star,
                            "optimum": {"block_set_C": set_star, "S_mL": recipe_star[0], "R_mL": recipe_star[1],
                                        "buffer_mL": recipe_star[2], "out_of_block_s_after_mixing": pull_star},
                            "conditions": submission["conditions"],
                            "product_moles": {k: moles.get(k, 0.0) for k in RM.SPECIES}}}

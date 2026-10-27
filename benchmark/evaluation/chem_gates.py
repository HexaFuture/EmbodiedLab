"""Chemistry kit gates: readings genuine, measurement valid, bench restored.

Spectrophotometry (spectro_basic, spectro_mixture, kinetics_crystal_violet, dye_absorptivity):
- spectro_readings_genuine: every cited reading is an absorbance the screen showed at a cited observation of this
  episode (get_observation's evidence_id, kind bench_view; READ values come from the screen only) and
  produced by a READ, its A equals the shown value within one unit of the last digit (screen_reading; on the kinetics
  bench a cited point of a KIN run and its logged A_display), and every composition the submission states for the
  cuvette read
  (dye from each stock, dilution factor of the unknown or of the mixture, [OH-] and [CV]) agrees within
  composition_tol with that cuvette's nominal dispensing history and the label values; a cuvette that had spilled or
  overflowed is not described by its history and cannot be cited.
  (six levels) a cited reading must also be of a cuvette holding at least the minimum reading volume
  (2.0 mL; private status not slot_empty or low_volume): with less liquid the beam passes above it.
- spectro_measurement_valid: the cited data measured the object of the task (the unknown, the mixture, a cuvette
  holding both crystal violet and NaOH, the dye stocks).
How the data were taken (wavelengths, blanks, mixing, how many points) is not checked: a poor choice gives a wrong
answer, which the score sees.

Titration (titration_basic, titration_weak_acid):
- titration_readings_genuine: every cited row exists in the cited record of this episode with the cited pH and
  counter volume; titration_weak_acid: a stated KHP mass equals the label value. (six levels) a row
  logged between a titrant key's registration and its delivery (DELIVERY_S later; the counter steps when the liquid has
  left the dispenser, and the Agent cannot see the exact moment of registration) may carry the counter value before or
  after that delivery.
- titration_measurement_valid: the cited rows include a titration of a sample beaker (titration_basic S1-S3,
  titration_weak_acid S2 or S3) with the electrode in its liquid.

chem_bench_restored (tidy-up, both benches): spectrophotometry: every cuvette standing in a rack well (any well), the
cell well and the station seat empty, no KIN run going (replayed from the bench's key log), the robot
holding nothing; titration: the three carriers in rack wells, the
electrode on its park seat, the stirrer seat empty, no record running, the robot holding nothing.

Gates read only the sealed episode (with its private records) and the submission.
"""
from __future__ import annotations

A_TOL = 5e-4            # the display shows 3 decimals
SCREEN_A_TOL = 1e-3 + 1e-9   # a reading from the screen: within one unit of the last shown digit
PH_TOL = 5e-3           # the meter shows 0.01 pH
ML_TOL = 5e-3           # the counter shows 0.01 mL: a cited V_mL is the counter screen's reading
KHP_TOL_G = 5e-5
DELIVERY_S = 0.5        # benchmark/core/chem_titration.py: a titrant key's press to the liquid in the beaker
MIN_READ_ML = 2.0       # benchmark/core/chem_spectro.py MIN_VOLUME_ML: the beam passes above less liquid
NO_VOLUME = ("slot_empty", "low_volume")


def _need(ctx):
    from benchmark.evaluation.gates import _need_submission       # the gate library imports this module
    return _need_submission(ctx)


def _mix(history):
    volumes = {}
    for source, mL in history or ():
        volumes[source] = volumes.get(source, 0.0) + float(mL)
    return volumes


def _agrees(claimed, value, tol):
    return abs(float(claimed) - value) <= tol * abs(value) + 1e-9


# --- spectrophotometry ---------------------------------------------------------------------------------------------
def composition(exp, device, history):
    """{field: value} the nominal dispensing history implies for a cuvette; a dilution factor only when the cuvette
    holds that liquid."""
    mix = _mix(history)
    total = sum(mix.values())

    def share(source, label):
        return float(label) * mix.get(source, 0.0) / total if total > 0 else 0.0

    if exp == "spectro_basic":
        out = {"c_from_stock_uM": share("stock", device["c_stock_uM_label"])}
        if mix.get("unknown", 0.0) > 0:
            out["unknown_dilution_factor"] = total / mix["unknown"]
        return out
    if exp == "spectro_mixture":
        labels, lines, mixture = device["c_stock_uM_label"], device["source_of"], device["mixture"]
        out = {f"c_{d}_from_stock_uM": share(lines[d], labels[d]) for d in ("X", "Y")}
        if mix.get(mixture["label"], 0.0) > 0:
            out["mixture_dilution_factor"] = total / mix[mixture["label"]]
        return out
    if exp == "dye_absorptivity":
        labels = device["c_label_uM"]
        return {"c_uM": share("stock", labels["stock"]) + share("unknown", labels["unknown"])}
    if exp == "kinetics_crystal_violet":
        lines = device["source_of"]
        return {"c_CV_uM": share(lines["dye"], device["c_stock_uM_label"]),
                "c_OH_M": share(lines["NaOH"], device["naoh_label_M"])}
    raise ValueError(f"no composition rule for {exp}")


def _lost(private):
    """The cuvette read had spilled or overflowed: its dispensing history no longer describes what it holds."""
    return bool(private.get("spilled")) or private.get("status") == "overflowed"


def screen_reading(ctx, evidence_id, A, wavelength_nm=None, kinds=("read",)):
    """One absorbance read on the spectrophotometer screen at a cited observation: (row, problem).

    evidence_id must be an observation of this episode (get_observation's evidence_id, kind bench_view). At it the
    screen showed an absorbance in the row of wavelength_nm (the dual-wavelength screen: required; one wavelength row:
    the row shown, and wavelength_nm, when given, must be the wavelength it showed) whose value came from a source of
    one of `kinds` ('read', 'blank', 'kin'), and A equals the shown value within one unit of its last digit. row: the
    sealed screen row (wavelength_nm, A, A_shown, source, and 'read' (the READ's private row with its public one) or
    'kin' (the KIN point's)); None with the problem when the reading fails."""
    record, problem = ctx.evidence(evidence_id)
    if problem is None and record.get("kind") != "bench_view":
        problem = f"{evidence_id}: not an observation"
    if problem is None and not isinstance(record.get("rows"), list):
        problem = f"{evidence_id}: no record of the spectrophotometer screen"
    if problem:
        return None, problem
    rows = [r for r in record["rows"] if r.get("A_shown") is not None]
    if record.get("dual"):
        if wavelength_nm is None:
            return None, f"{evidence_id}: the reading does not name the wavelength row it was read in"
        rows = [r for r in rows if r.get("wavelength_nm_shown") == int(wavelength_nm)]
    elif wavelength_nm is not None:
        rows = [r for r in rows if r.get("wavelength_nm_shown") == int(wavelength_nm)]
    if not rows:
        at = "" if wavelength_nm is None else f" at {int(wavelength_nm)} nm"
        return None, f"{evidence_id}: the screen showed no absorbance{at} at this observation"
    row = rows[-1]
    kind = (row.get("source") or {}).get("kind")
    if kind not in kinds:
        what = {"blank": "a BLANK", "kin": "a KIN point", "read": "a READ"}.get(kind, "nothing measured")
        return None, f"{evidence_id}: the absorbance shown came from {what}, not from {' or '.join(kinds)}"
    if not isinstance(A, (int, float)) or isinstance(A, bool) or not abs(float(A) - row["A_shown"]) <= SCREEN_A_TOL:
        return None, f"{evidence_id}: A {A} is not the {row['A_shown']:.3f} the screen showed"
    return row, None


def cited_spectro(ctx):
    """One dict per cited reading: label, record (None when the citation is bad), problem, history, cuvette, claims,
    cited A, logged A."""
    exp, sub, out = ctx.spec.id, ctx.submission, []

    def bad(label, problem):
        out.append({"label": label, "record": None, "problem": problem})

    if exp == "kinetics_crystal_violet":
        for run in sub["runs"]:
            record, problem = ctx.evidence(run["evidence_id"])
            if problem is None and record.get("kind") != "kinetic_run":
                problem = f"{run['evidence_id']}: not a KIN run"
            if problem:
                bad(run["evidence_id"], problem)
                continue
            rows = {r["point"]: r for r in record["public"]["rows"]}
            private = {p["point"]: p for p in record["private_points"]}
            claims = {"c_OH_M": run["c_OH_M"], "c_CV_uM": run["c_CV_uM"]}
            for point in run["points"]:
                label = f"{run['evidence_id']} point {point['point']}"
                if point["point"] not in rows or point["point"] not in private:
                    bad(label, f"{label}: no such point in the run")
                    continue
                p = private[point["point"]]
                out.append({"label": label, "record": record, "problem": None, "history": p["nominal_history"],
                            "cuvette": p["cuvette"], "claims": claims, "A": point["A"], "lost": _lost(p),
                            "A_logged": rows[point["point"]]["A_display"], "status": p.get("status"),
                            "volume_mL": p.get("volume_mL"), "volume_logged": "volume_mL" in p})
        return out
    seen = set()
    for entry in sub["readings"]:
        nm = entry.get("wavelength_nm")
        label = entry["evidence_id"] if nm is None else f"{entry['evidence_id']} at {nm} nm"
        if (entry["evidence_id"], nm) in seen:
            bad(label, f"{label}: cited twice")
            continue
        seen.add((entry["evidence_id"], nm))
        row, problem = screen_reading(ctx, entry["evidence_id"], entry["A"], nm)
        if problem:
            bad(label, problem)
            continue
        record = row["read"]
        out.append({"label": label, "record": record, "problem": None,
                    "history": record["nominal_history"], "cuvette": record["cuvette"], "lost": _lost(record),
                    "claims": {k: v for k, v in entry.items() if k not in ("evidence_id", "A", "wavelength_nm")},
                    "A": entry["A"], "A_logged": row["A_shown"], "A_tol": SCREEN_A_TOL,
                    "status": record.get("status"), "volume_mL": record.get("volume_mL"),
                    "volume_logged": "volume_mL" in record})
    return out


def _too_little(c, min_read):
    """The cited reading's cuvette held less than the minimum reading volume (six levels); a record
    without the private volume (older than the volume log) is not judged on it."""
    if not c.get("volume_logged"):
        return False
    volume = c.get("volume_mL")
    return c.get("status") in NO_VOLUME or volume is None or float(volume) < min_read - 1e-9


def spectro_readings_genuine(ctx, composition_tol=0.01):
    if (blocked := _need(ctx)) is not None:
        return blocked
    exp, device = ctx.spec.id, ctx.sealed.episode["device"]
    min_read = float(device.get("min_read_mL", MIN_READ_ML))
    problems, checked = [], []
    for c in cited_spectro(ctx):
        if c["record"] is None:
            problems.append(c["problem"])
            continue
        label, cuvette, claims = c["label"], c["cuvette"], c["claims"]
        checked.append({"cited": label, "cuvette": cuvette, "A_logged": c["A_logged"]})
        if c["lost"]:
            problems.append(f"{label}: cuvette {cuvette} had spilled or overflowed; its dispensing history no longer "
                            f"describes it")
        if _too_little(c, min_read):
            problems.append(f"{label}: the well held no cuvette with at least {min_read:g} mL (the beam passes above "
                            f"less liquid)")
        if abs(float(c["A"]) - float(c["A_logged"])) > c.get("A_tol", A_TOL):
            problems.append(f"{label}: A {c['A']} is not the logged {c['A_logged']}")
        truth = composition(exp, device, c["history"])
        for field in sorted(set(truth) | set(claims)):
            if field not in claims:
                problems.append(f"{label}: cuvette {cuvette} holds what {field} describes; the reading does not state it")
            elif field not in truth:
                problems.append(f"{label}: {field} given, but cuvette {cuvette} holds nothing it describes")
            elif not _agrees(claims[field], truth[field], composition_tol):
                problems.append(f"{label}: {field} {claims[field]} disagrees with the dispensing history of cuvette "
                                f"{cuvette} ({truth[field]:.5g})")
    return not problems, {"problems": problems, "readings": checked}


def measures_object(exp, device, history, cuvette):
    mix = _mix(history)
    if exp == "spectro_basic":
        return mix.get("unknown", 0.0) > 0
    if exp == "spectro_mixture":
        mixture = device["mixture"]
        return cuvette == int(mixture["cuvette"]) and mix.get(mixture["label"], 0.0) > 0
    if exp == "kinetics_crystal_violet":
        lines = device["source_of"]
        return mix.get(lines["dye"], 0.0) > 0 and mix.get(lines["NaOH"], 0.0) > 0
    if exp == "dye_absorptivity":
        return mix.get("stock", 0.0) > 0 or mix.get("unknown", 0.0) > 0
    raise ValueError(exp)


OBJECT = {"spectro_basic": "a cuvette holding the unknown solution",
          "spectro_mixture": "the mixture cuvette holding the mixture",
          "kinetics_crystal_violet": "a KIN run of a cuvette holding both crystal violet and NaOH",
          "dye_absorptivity": "a cuvette holding a dye stock or a dilution of it"}


def spectro_measurement_valid(ctx):
    if (blocked := _need(ctx)) is not None:
        return blocked
    exp, device = ctx.spec.id, ctx.sealed.episode["device"]
    hits = [c["label"] for c in cited_spectro(ctx)
            if c["record"] is not None and c["history"] and measures_object(exp, device, c["history"], c["cuvette"])]
    if not hits:
        return False, {"problems": [f"no cited reading of {OBJECT[exp]}"]}
    return True, {"problems": [], "object_readings": hits}


# --- titration -----------------------------------------------------------------------------------------------------
def counter_windows(episode):
    """[(first tick, last tick, counter values)] per titrant key press (six levels): from the key's
    registration to its delivery DELIVERY_S later the counter screen may show its value before or after that delivery
    (it steps when the liquid has left the dispenser). Replayed from the sealed bench records (key log, state log:
    deliveries and ZERO); empty without them."""
    device = episode.get("device") or {}
    actions, log = device.get("actions"), (device.get("chemistry") or {}).get("state_log")
    if not isinstance(actions, list) or not isinstance(log, list):
        return []
    span = round(DELIVERY_S * int(episode.get("physics_hz") or 240))
    steps, value = [], 0.0
    for event in sorted(log, key=lambda e: e["tick"]):
        if event.get("event") == "counter_zero":
            value = 0.0
        elif event.get("event") == "titrant" and event.get("dispensed"):
            value = round(value + float(event["nominal_mL"]), 2)
        else:
            continue
        steps.append((event["tick"], value))
    windows = []
    for action in actions:
        if not str(action.get("button", "")).startswith("titrant:"):
            continue
        start = action["tick"]
        before = next((v for t, v in reversed(steps) if t < start), 0.0)
        windows.append((start, start + span, {before} | {v for t, v in steps if start <= t <= start + span}))
    return windows


def cited_rows(ctx):
    """One dict per cited row: label, record (None when bad), public row, private row, problem. A row's pH is its
    logged pH_display; its V_mL is the counter value at that row, kept privately with the row (the pH log has no volume
    column records sealed before carry it in the public row); a row logged within a titrant key's
    delivery window may carry the counter value before or after that delivery (counter_windows)."""
    out = []
    episode = ctx.sealed.episode
    windows = counter_windows(episode)
    hz = int(episode.get("physics_hz") or 240)
    for cited in ctx.submission["records"]:
        record, problem = ctx.evidence(cited["evidence_id"])
        if problem is None and record.get("kind") != "titration_record":
            problem = f"{cited['evidence_id']}: not a titration record"
        if problem:
            out.append({"label": cited["evidence_id"], "record": None, "problem": problem})
            continue
        rows = {r["sample_id"]: r for r in record["public"]["rows"]}
        private = {p["sample_id"]: p for p in record["private_rows"]}
        for row in cited["rows"]:
            label = f"{cited['evidence_id']} sample {row['sample_id']}"
            if row["sample_id"] not in rows:
                out.append({"label": label, "record": None, "problem": f"{label}: no such row in the record"})
                continue
            public = rows[row["sample_id"]]
            counter = (private.get(row["sample_id"]) or {}).get("counter_mL", public.get("V_counter_mL"))
            problem = None
            tick = round(float(public["t_s"]) * hz)
            accepted = ({float(counter)} if counter is not None else set()) | {
                v for start, end, values in windows if start <= tick <= end for v in values}
            if abs(float(row["pH"]) - float(public["pH_display"])) > PH_TOL:
                problem = f"{label}: pH {row['pH']} is not the logged {public['pH_display']}"
            elif not any(abs(float(row["V_mL"]) - v) <= ML_TOL for v in accepted):
                problem = f"{label}: V {row['V_mL']} mL is not the counter reading at that row ({counter} mL)"
            out.append({"label": label, "record": record, "public": public,
                        "private": private.get(row["sample_id"]) or {}, "problem": problem})
    return out


def titration_readings_genuine(ctx):
    if (blocked := _need(ctx)) is not None:
        return blocked
    rows = cited_rows(ctx)
    problems = [r["problem"] for r in rows if r["problem"]]
    device = ctx.sealed.episode["device"]
    stated = ctx.submission.get("khp_mass_g")
    if stated is not None and "khp_mass_g" in device and abs(float(stated) - float(device["khp_mass_g"])) > KHP_TOL_G:
        problems.append(f"khp_mass_g {stated} is not the label value")
    return not problems, {"problems": problems, "rows_cited": len(rows)}


def titration_measurement_valid(ctx):
    if (blocked := _need(ctx)) is not None:
        return blocked
    device = ctx.sealed.episode["device"]
    samples = set(device.get("sample_beakers") or ("S1", "S2", "S3"))
    hits = sorted({r["private"].get("beaker") for r in cited_rows(ctx)
                   if r["record"] is not None and r["private"].get("immersed") and r["private"].get("beaker") in samples})
    if not hits:
        return False, {"problems": [f"no cited row of a titration of {' or '.join(sorted(samples))} with the electrode "
                                    f"in the liquid"]}
    return True, {"problems": [], "sample_beakers": hits}


# --- tidy-up -------------------------------------------------------------------------------------------------------
def chem_bench_restored(ctx):
    state = ctx.sealed.episode.get("terminal_state") or {}
    problems = []
    if isinstance(state.get("cuvette_seat"), dict):
        for cid, seat in sorted(state["cuvette_seat"].items(), key=lambda kv: int(kv[0])):
            if not (isinstance(seat, str) and seat.startswith("rack:")):
                problems.append(f"cuvette {cid} is not standing in a rack well ({seat or 'not seated'})")
        seats = state.get("seats") or {}
        for seat, name in (("spectro", "the cell well"), ("station", "the station seat")):
            if seats.get(seat) is not None:
                problems.append(f"{name} holds cuvette {seats[seat]}")
        from benchmark.evaluation.kinetics import kin_running_at_end       # no KIN run still going
        running, problem = kin_running_at_end(ctx.sealed.episode)
        if problem:
            problems.append(problem)
        elif running:
            problems.append("a KIN run is still going")
    elif isinstance(state.get("carrier_seat"), dict):
        for label, seat in sorted(state["carrier_seat"].items()):
            if not (isinstance(seat, str) and seat.startswith("rack:")):
                problems.append(f"carrier {label} is not in a rack well ({seat or 'not seated'})")
        if (state.get("seats") or {}).get("stirrer") is not None:
            problems.append(f"the stirrer seat holds {state['seats']['stirrer']}")
        if state.get("electrode") != "park":
            problems.append(f"the electrode is not on its park seat ({state.get('electrode')})")
        if state.get("recording") is not False:
            problems.append("a record is still running" if state.get("recording") else "no record of the recorder state")
    else:
        return False, {"problems": ["no record of the bench at submission"]}
    if state.get("held") is not None:
        problems.append(f"the robot holds {state['held']}")
    return not problems, {"problems": problems}

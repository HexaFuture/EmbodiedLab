"""Process-score rules of the chemistry kit: spectro_basic, titration_basic, spectro_mixture, titration_weak_acid,
kinetics_crystal_violet, dye_absorptivity.

Records used (all exported at stop):
    private/physics.jsonl     the backend's 30 Hz frame rows (held object, the station seat each cuvette or carrier sits
                              on, the arm's joint angles) and its grasp events (grasp_attach / grasp_release /
                              grasp_break)
    bench_logs.json           the bench's private state_log: every dispense (into which cuvette, actual and overflowed
                              mL), MIX completion, spill, BLANK / READ / SCAN / KIN completion; titration: carrier on
                              the stirrer, every titrant delivery; spectro_scans: the SCAN spectra (rebuild on)
    evidence.json             the spectrophotometer READ rows and KIN runs with their private status (cuvette, its
                              dispensing history, volume, mixed / blanked / overflowed); the observations (bench_view)
                              with what the screen showed (a READ counts for M1/M2 once an
                              observation showed its absorbance, at that observation's tick), the titration REC records
                              with their private rows (electrode place, beaker, immersed, titrant delivered into it)
    episode.json device       the public label values (stock concentrations, the mixture cuvette, KHP and sample beakers)
A grasp is the physical grasp joint of the physics log; it is effective once the grasped object has risen LIFT_M while
the joint holds: its logged position (frame xyz_mm, when logged), else the hand height from the logged joint
angles (Franka kinematics; older records). Cuvette contents between READs are replayed from the state log with the
bench's own rules (a cuvette is mixed when all its layers have one composition).

M2 is the minimum valid data set, without count thresholds: spectro_basic a known dilution and the
unknown at one wavelength; spectro_mixture dilutions of each pure dye and the mixture at two or more wavelengths;
kinetics_crystal_violet (orders task of) complete decay runs at two known [OH-] levels, the higher at
least 20 % above the lower; dye_absorptivity readings around the maximum (a SCAN
or point-by-point READs) and a known-concentration READ within 3 nm of it.
"""
from __future__ import annotations

import math

import numpy as np

from benchmark.evaluation.process import hit, miss
from benchmark.registry.expressions import dye_lambda_max

LIFT_M = 0.010                  # the hand (and the latched object) rose this far: lifted off its seat (seat tol 3 mm)
MIN_READ_ML = 2.0               # the spectrophotometer beam passes above less liquid
A_TOL = 1e-9
# Fallback without the public task: the dyes whose maxima spectro_basic / spectro_mixture may publish (the specs' choices).
CANDIDATE_DYES = ("tartrazine", "allura_red", "crystal_violet", "brilliant_blue_fcf")
DYE_PEAK_TOL_NM = 3             # dye_absorptivity: a known-concentration READ within 3 nm of the located maximum
BRACKET_NM = 10                 # dye_absorptivity: point-by-point READs bracket the maximum from at most this far
SCAN_A_RANGE = (0.1, 2.5)       # a survey scan locates the maximum when its peak is clear of the baseline and of stray light
PEAK_TOL_NM = 1                 # spectro_basic gate: the reading wavelength within 1 nm of the published maximum
DYE_SOURCES = {"spectro_basic": ("stock", "unknown"), "spectro_mixture": ("stock", "unknown", "mixture"),
               "dye_absorptivity": ("stock", "unknown")}
NAOH = "unknown"                # kinetics line names: stock = CV, water, unknown = NaOH
KIN = dict(min_points=10, min_decay=0.5, min_A=0.02, oh_margin=0.20)   # oh_margin: the kinetic_records_valid gate
TITRATION = dict(equivalence_pH=7.0, max_gap_mL=0.10)
WEAK = dict(below_pH=7.0, above_pH=10.0)       # a weak-acid or KHP record spans its endpoint


# --- physics log ----------------------------------------------------------------------------------------------------
def _physics(run):
    """(frame rows with their tick, grasp events) of the episode proper: the backend settles the bench for 40 frames
    and then restarts its clock at 0, so everything before the last clock reset is dropped."""
    step = run.physics_hz // 30
    frames, events, last = [], [], -1
    for row in run.frames:
        tick = row["frame"] * step if "frame" in row else row.get("tick")
        if tick is None:
            continue
        if tick < last:
            frames, events = [], []
        last = tick
        if "frame" in row:
            frames.append(row | {"tick": tick})
        elif row.get("event") in ("grasp_attach", "grasp_release", "grasp_break"):
            events.append(row)
    return frames, events


# Franka Panda, modified DH (a, d, alpha) of joints 1-7 and the flange (panda_hand).
_DH = ((0.0, 0.333, 0.0), (0.0, 0.0, -math.pi / 2), (0.0, 0.316, math.pi / 2), (0.0825, 0.0, math.pi / 2),
       (-0.0825, 0.384, -math.pi / 2), (0.0, 0.0, math.pi / 2), (0.088, 0.0, math.pi / 2), (0.0, 0.107, 0.0))


def hand_position(q):
    """panda_hand origin in the robot base frame from the seven joint angles."""
    T = np.eye(4)
    for (a, d, alpha), theta in zip(_DH, list(q)[:7] + [0.0]):
        ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(theta), math.sin(theta)
        T = T @ np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])
    return T[:3, 3]


def _height_m(frame, key):
    """Height of a held object in a frame: its logged position (xyz_mm), else the hand's from the joint angles (the
    latched object moves with the hand). Records before xyz_mm have only the joint angles."""
    xyz = (frame.get("xyz_mm") or {}).get(str(key))
    return xyz[2] / 1000.0 if xyz else hand_position(frame["q_rad"])[2]


def _lifted(run, exclude=()):
    """(tick, object): the first time a grasped object (any but `exclude`) was lifted, i.e. it rose LIFT_M above its
    height at the grasp while the grasp joint held; (None, None) when never."""
    frames, events = _physics(run)
    if not frames:
        return None, None
    best = (None, None)
    for k, e in enumerate(events):
        if e["event"] != "grasp_attach" or e.get("object") in exclude:
            continue
        key = e.get("object")
        end = next((x["tick"] for x in events[k + 1:] if x.get("object") == key), math.inf)
        before = [f for f in frames if f["tick"] <= e["tick"]]
        start = before[-1] if before else frames[0]
        use_xyz = str(key) in (start.get("xyz_mm") or {})
        z0 = _height_m(start, key) if use_xyz else hand_position(start["q_rad"])[2]
        for f in frames:
            if not (e["tick"] < f["tick"] <= end and f.get("held") == key):
                continue
            z = _height_m(f, key) if use_xyz else hand_position(f["q_rad"])[2]
            if z - z0 >= LIFT_M:
                if best[0] is None or f["tick"] < best[0]:
                    best = (f["tick"], key)
                break
    return best


def _e1(run, what, exclude=()):
    tick, key = _lifted(run, exclude)
    return (hit(tick, f"{what} {key} grasped and lifted {LIFT_M * 1000:.0f} mm") if tick is not None
            else miss(f"no {what} grasped and lifted"))


# --- cuvette contents replayed from the state log -------------------------------------------------------------------
def _same(a, b):
    return all(abs(a.get(k, 0.0) - b.get(k, 0.0)) <= 1e-9 for k in set(a) | set(b))


class _Contents:
    """A cuvette as the bench keeps it: layers (source fractions, mL) in order of addition; overflow and spill marks."""

    def __init__(self):
        self.layers, self.overflowed, self.spilled = [], False, False

    @property
    def volume(self):
        return sum(v for _, v in self.layers)

    def fractions(self):
        total = self.volume
        out = {}
        for comp, v in self.layers:
            for k, f in comp.items():
                out[k] = out.get(k, 0.0) + f * v / total
        return out

    @property
    def mixed(self):
        return all(_same(self.layers[0][0], c) for c, _ in self.layers[1:])

    def add(self, source, mL, lost=0.0):
        kept = mL - lost
        if kept > 0:
            comp = {source: 1.0}
            if self.layers and _same(self.layers[-1][0], comp):
                self.layers[-1] = (self.layers[-1][0], self.layers[-1][1] + kept)
            else:
                self.layers.append((comp, kept))
        if lost > 0:
            self.overflowed = True

    def apply(self, event):
        kind = event["event"]
        if kind == "dispense" and event.get("dispensed") and event.get("into") is not None:
            self.add(event["source"], float(event["actual_mL"]), float(event.get("overflow_mL") or 0.0))
        elif kind == "mixed" and self.layers:
            self.layers = [(self.fractions(), self.volume)]
        elif kind == "spill":
            self.layers, self.spilled = [], True


def _state_log(run):
    log = run.logs.get("state_log")
    if not isinstance(log, list):
        log = ((run.device.get("chemistry") or {}).get("state_log")) or []
    return sorted(log, key=lambda e: e["tick"])


def _container(event):
    return event.get("into") if event["event"] == "dispense" else event.get("container")


def _replay(run):
    """Yield (tick, {cuvette: _Contents}) after each state-log event; the start state (the handed-out mixture of
    spectro_mixture) first, at tick 0."""
    cuvettes = {}
    mixture = run.device.get("mixture")
    if mixture:
        cuvettes[int(mixture["cuvette"])] = c = _Contents()
        c.add(mixture["label"], float(mixture["volume_mL"]))
    yield 0, cuvettes
    for event in _state_log(run):
        cid = _container(event)
        if cid is None or event["event"] not in ("dispense", "mixed", "spill"):
            continue
        cuvettes.setdefault(int(cid), _Contents()).apply(event)
        yield event["tick"], cuvettes


def _measurable(c):
    return c is not None and c.volume >= MIN_READ_ML - A_TOL and not c.overflowed and c.mixed


def _e2_slot(run, reads):
    """E2 of the spectrophotometer experiments: a cuvette in the sample well holding >= 2 mL, not overflowed, mixed.
    The well is read from the 30 Hz frames, the contents from the state-log replay; a READ whose private status says
    the same (ok, or failing only its BLANK) also counts, at its tick."""
    frames, _ = _physics(run)
    snapshots = [(tick, {cid for cid, c in current.items() if _measurable(c)}) for tick, current in _replay(run)]
    found, i = None, 0
    for f in frames:
        while i + 1 < len(snapshots) and snapshots[i + 1][0] <= f["tick"]:
            i += 1
        cid = (f.get("seats") or {}).get("spectro")
        if cid is not None and int(cid) in snapshots[i][1]:
            found = hit(f["tick"], f"cuvette {cid} in the sample well: >= {MIN_READ_ML} mL, mixed, not overflowed")
            break
    read = next((r for r in reads if r["status"] in ("ok", "no_blank_at_λ", "blank_ref_not_solvent")), None)
    if read is not None and (found is None or read["tick"] < found.tick):
        found = hit(read["tick"], f"READ {read['sample_id']} of cuvette {read['cuvette']} measurable (private status)",
                    ["e2_from_read_status"])
    return found or miss("no cuvette in the sample well with >= 2 mL, mixed and not overflowed")


# --- spectrophotometer records --------------------------------------------------------------------------------------
def _seen(run):
    """{sample_id: tick of the first observation whose screen showed that READ's absorbance} (bench_view records)."""
    seen = {}
    for record in run.evidence.values():
        if record.get("kind") != "bench_view":
            continue
        for row in record.get("rows") or ():
            source = row.get("source") or {}
            if source.get("kind") == "read" and row.get("A_shown") is not None:
                k = source["sample_id"]
                seen[k] = min(seen.get(k, record["tick"]), record["tick"])
    return seen


def _reads(run, seen=False):
    """READ rows in order, each its private record with tick, wavelength, A, blank wavelength. seen: since READ values
    reach the Agent only on the screen (records with device.screen_readings), only READs whose absorbance an
    observation showed, each at the tick of the first such observation (read_tick: when it was taken)."""
    out = []
    shown = _seen(run) if seen and run.device.get("screen_readings") else None
    for record in run.evidence.values():
        if record.get("kind") != "spectro_read":
            continue
        public = record["public"]
        tick = round(public["t_s"] * run.physics_hz)
        if shown is not None:
            if record["sample_id"] not in shown:
                continue
            tick = max(tick, shown[record["sample_id"]])
        out.append(dict(record, tick=tick, read_tick=round(public["t_s"] * run.physics_hz), nm=int(public["wavelength_nm"]),
                        A=float(public["A_display"]), blank_nm=public["blank_wavelength_nm"]))
    return sorted(out, key=lambda r: r["sample_id"])


def _mix(history):
    volumes = {}
    for source, mL in history:
        volumes[source] = volumes.get(source, 0.0) + float(mL)
    return volumes


def _valid(run, r):
    """A READ that counts: private status ok (>= 2 mL, mixed, not overflowed, BLANK on water at this wavelength),
    not spilled."""
    min_read = run.device.get("min_read_mL", MIN_READ_ML)
    return (r["status"] == "ok" and not r.get("spilled") and r.get("volume_mL") is not None
            and r["volume_mL"] >= min_read - A_TOL and r["blank_nm"] == r["nm"])


def _m1_read(run, sources):
    for r in sorted(_reads(run, seen=True), key=lambda r: r["tick"]):
        if _valid(run, r) and any(_mix(r["nominal_history"]).get(s, 0.0) > 0 for s in sources):
            return hit(r["tick"], f"READ {r['sample_id']}: cuvette {r['cuvette']} with dye, status ok at {r['nm']} nm")
    return miss("no READ with status ok of a cuvette holding dye")


def _published_peaks(run, keys):
    """({name: nm} of the published maxima, flags): the public parameters `keys` of the run's task, else every
    candidate dye's maximum, flagged."""
    params = (run.public or {}).get("public_parameters") or {}
    if all(params.get(k) is not None for k in keys):
        return {k: int(round(float(params[k]))) for k in keys}, []
    return {name: dye_lambda_max(name) for name in CANDIDATE_DYES}, ["published_peaks_unavailable"]


def _first(items, judge):
    """The first tick by which items (sorted by tick) satisfy judge(prefix) -> (ok, note); judge is monotone in the
    prefix (a data set, once present, stays present). (tick, note) or (None, note of the full set)."""
    ok, note = judge(items)
    if not ok:
        return None, note
    lo, hi = 1, len(items)
    while lo < hi:
        mid = (lo + hi) // 2
        if judge(items[:mid])[0]:
            hi = mid
        else:
            lo = mid + 1
    return items[lo - 1]["tick"], judge(items[:lo])[1]


def _m2(items, judge, basis, flags=()):
    tick, note = _first(items, judge)
    return hit(tick, f"{basis}: {note}", flags) if tick is not None else miss(f"{basis} not met: {note}", flags)


def _calibration_set(rows, concentration):
    """(cuvettes, distinct non-zero concentrations) of calibration READs; concentration(history) -> c or None."""
    cuvettes, levels = set(), set()
    for r in rows:
        c = concentration(_mix(r["nominal_history"]))
        if c is None:
            continue
        cuvettes.add(r["cuvette"])
        if c > 0:
            levels.add(round(c, 6))
    return cuvettes, levels


# --- spectro_basic ----------------------------------------------------------------------------------------------------
def _basic_m2(run, reads):
    """At one wavelength: a READ (status ok) of a cuvette with a known dye concentration (`STOCK`, with or without
    water, no `UNKNOWN`) and a READ (status ok) of a cuvette holding `UNKNOWN`."""
    def judge(prefix):
        for nm in sorted({r["nm"] for r in prefix}):
            rows = [(r, _mix(r["nominal_history"])) for r in prefix if r["nm"] == nm]
            known = [r for r, mix in rows if mix.get("stock", 0.0) > 0 and mix.get("unknown", 0.0) == 0]
            unknown = [r for r, mix in rows if mix.get("unknown", 0.0) > 0]
            if known and unknown:
                return True, (f"{nm} nm: READ {known[0]['sample_id']} of a known dilution, READ "
                              f"{unknown[0]['sample_id']} of the unknown")
        return False, "no wavelength with READs of a known dilution and of the unknown"

    return _m2([r for r in reads if _valid(run, r)], judge, "known dilution and unknown at one wavelength")


def spectro_basic(run):
    reads = _reads(run)
    return {"E1": _e1(run, "cuvette"), "E2": _e2_slot(run, reads),
            "M1": _m1_read(run, DYE_SOURCES["spectro_basic"]), "M2": _basic_m2(run, _by_tick(_reads(run, seen=True)))}


def _by_tick(reads):
    return sorted(reads, key=lambda r: r["tick"])


# --- spectro_mixture --------------------------------------------------------------------------------------------------
def _mixture_m2(run, reads):
    """At two or more wavelengths: READs (status ok) of a cuvette holding dye X from its stock and no dye Y, of one
    holding dye Y and no dye X, and of the mixture cuvette (the mixture, with or without water)."""
    device = run.device
    lines, mixture = device.get("source_of") or {}, device.get("mixture")
    if not mixture or not lines:
        return miss("device record lacks the stock lines or the mixture cuvette")

    def role(r):
        mix = _mix(r["nominal_history"])
        if mix.get(mixture["label"], 0.0) > 0:
            clean = not any(mix.get(lines[d], 0.0) > 0 for d in ("X", "Y"))
            return "mixture" if clean and int(r["cuvette"]) == int(mixture["cuvette"]) else None
        x, y = mix.get(lines["X"], 0.0) > 0, mix.get(lines["Y"], 0.0) > 0
        return "X" if x and not y else "Y" if y and not x else None

    def judge(prefix):
        roles = {}
        for r in prefix:
            kind = role(r)
            if kind:
                roles.setdefault(r["nm"], set()).add(kind)
        done = sorted(nm for nm, kinds in roles.items() if kinds >= {"X", "Y", "mixture"})
        return len(done) >= 2, f"wavelengths with READs of dye X, dye Y and the mixture: {done}"

    return _m2([r for r in reads if _valid(run, r)], judge, "pure-dye dilutions and the mixture at two wavelengths")


def spectro_mixture(run):
    reads = _reads(run)
    return {"E1": _e1(run, "cuvette"), "E2": _e2_slot(run, reads),
            "M1": _m1_read(run, DYE_SOURCES["spectro_mixture"]), "M2": _mixture_m2(run, _by_tick(_reads(run, seen=True)))}


# --- dye_absorptivity -------------------------------------------------------------------------------------------------
def _scan_peak(scan):
    """Wavelength of the absorption maximum of one SCAN spectrum (bench_logs spectro_scans) or None: a cuvette in a
    valid condition with its peak absorbance in SCAN_A_RANGE, the vertex of a parabola through the 5 nm moving average
    around its maximum."""
    nm, a = np.asarray(scan["wavelength_nm"], float), np.asarray(scan["A"], float)
    if scan.get("condition") != "ok" or len(a) < 11 or not SCAN_A_RANGE[0] <= a.max() <= SCAN_A_RANGE[1]:
        return None
    smooth = np.convolve(a, np.ones(5) / 5, mode="same")
    i = int(np.clip(np.argmax(smooth[2:-2]) + 2, 5, len(a) - 6))
    coef = np.polyfit(nm[i - 5:i + 6] - nm[i], smooth[i - 5:i + 6], 2)
    return float(nm[i] + (-coef[1] / (2 * coef[0]) if coef[0] < 0 else 0.0))


def scan_peak(logs):
    """(median maximum over the usable SCAN spectra or None, note)."""
    scans = logs.get("spectro_scans")
    if scans is None:
        return None, "scans not exported (older record)"
    peaks = [p for p in (_scan_peak(s) for s in scans) if p is not None]
    if not peaks:
        return None, f"no usable scan among {len(scans)}"
    return float(np.median(peaks)), f"maximum {np.median(peaks):.1f} nm from {len(peaks)} scans"


def _dye_m2(run, reads):
    """Readings around the absorption maximum (a usable SCAN of a dye cuvette, or READs (status ok) of one cuvette at
    three wavelengths, the middle one highest and its neighbours at most BRACKET_NM away) and a READ (status ok) of a
    cuvette with a known dye concentration within 3 nm of that maximum."""
    def dyed(r):
        mix = _mix(r["nominal_history"])
        return mix.get("stock", 0.0) > 0 or mix.get("unknown", 0.0) > 0

    items = [dict(r, kind="read") for r in reads if _valid(run, r) and dyed(r)]
    for scan in run.logs.get("spectro_scans") or []:
        peak = _scan_peak(scan)
        if peak is not None:
            items.append({"kind": "scan", "tick": round(float(scan["t_s"]) * run.physics_hz), "peak": peak,
                          "source": f"SCAN {scan['scan_id']}"})
    items.sort(key=lambda item: item["tick"])

    def judge(prefix):
        rows = [r for r in prefix if r["kind"] == "read"]
        maxima = [(p["peak"], p["source"]) for p in prefix if p["kind"] == "scan"]
        curves = {}
        for r in rows:
            curves.setdefault((r["cuvette"], tuple(map(tuple, r["nominal_history"]))), {})[r["nm"]] = r["A"]
        for (cuvette, _), curve in curves.items():
            nms = sorted(curve)
            for lo, mid, hi in zip(nms, nms[1:], nms[2:]):
                if hi - lo <= 2 * BRACKET_NM and curve[mid] >= max(curve[lo], curve[hi]):
                    maxima.append((float(mid), f"READs of cuvette {cuvette} at {lo}/{mid}/{hi} nm"))
        for peak, source in maxima:
            near = [r for r in rows if abs(r["nm"] - peak) <= DYE_PEAK_TOL_NM]
            if near:
                return True, f"maximum {peak:.1f} nm ({source}), READ {near[0]['sample_id']} at {near[0]['nm']} nm"
        return False, f"{len(maxima)} located maxima, no known-concentration READ within {DYE_PEAK_TOL_NM} nm of one"

    return _m2(items, judge, "readings around the maximum and a known-concentration READ there")


def dye_absorptivity(run):
    reads = _reads(run)
    return {"E1": _e1(run, "cuvette"), "E2": _e2_slot(run, reads),
            "M1": _m1_read(run, DYE_SOURCES["dye_absorptivity"]), "M2": _dye_m2(run, _by_tick(_reads(run, seen=True)))}


# --- kinetics_crystal_violet ------------------------------------------------------------------------------------------
def _kin_e2(run):
    """A cuvette holding dye and NaOH, mixed (the reaction runs through the whole cuvette), not overflowed."""
    for tick, cuvettes in _replay(run):
        for cid, c in cuvettes.items():
            parts = c.fractions() if c.volume > 0 else {}
            if parts.get("stock", 0) > 0 and parts.get(NAOH, 0) > 0 and c.mixed and not c.overflowed:
                return hit(tick, f"cuvette {cid}: dye and NaOH mixed")
    return miss("no cuvette with dye and NaOH mixed")


def _kin_points(run):
    """KIN points in order: (run, public row, private point, wavelength, blank wavelength, tick)."""
    out = []
    for record in run.evidence.values():
        if record.get("kind") != "kinetic_run":
            continue
        public = record["public"]
        private = {p["point"]: p for p in record.get("private_points") or []}
        for row in public["rows"]:
            p = private.get(row["point"])
            if p is not None:
                out.append(dict(p, run=public["run"], A=float(row["A_display"]), nm=public["wavelength_nm"],
                                blank_nm=public["blank_wavelength_nm"], tick=round(row["t_s"] * run.physics_hz)))
    return sorted(out, key=lambda p: p["tick"])


def _kin_ok(run, p):
    min_read = run.device.get("min_read_mL", MIN_READ_ML)
    return (p["status"] == "ok" and not p.get("spilled") and p.get("volume_mL") is not None
            and p["volume_mL"] >= min_read - A_TOL and p["blank_nm"] == p["nm"])


def _kin_m1(run, points):
    for p in points:
        mix = _mix(p["nominal_history"])
        if _kin_ok(run, p) and mix.get("stock", 0) > 0 and mix.get(NAOH, 0) > 0:
            return hit(p["tick"], f"KIN run {p['run']} point {p['point']}: cuvette {p['cuvette']} with dye and NaOH, ok")
    return miss("no KIN point with status ok of a cuvette holding dye and NaOH")


def _kin_m2(run, points):
    """Complete decay runs at two [OH-] levels (orders task of): two KIN runs, each with points (status ok,
    BLANK at the run's wavelength) of one cuvette with one dispensing history holding dye and NaOH (a known [OH-]),
    the absorbance falling from its first point (A >= 0.02) to at most half of it, the higher [OH-] (NaOH share of the
    nominal volumes) at least oh_margin above the lower."""
    def judge(prefix):
        groups = {}
        for p in prefix:
            mix = _mix(p["nominal_history"])
            if _kin_ok(run, p) and mix.get("stock", 0) > 0 and mix.get(NAOH, 0) > 0:
                share = mix[NAOH] / sum(mix.values())
                groups.setdefault((p["run"], p["cuvette"], tuple(map(tuple, p["nominal_history"])), share),
                                  []).append(p["A"])
        done = {}
        for (n, cuvette, _, share), a in groups.items():
            if a[0] >= KIN["min_A"] and min(a) <= (1 - KIN["min_decay"]) * a[0]:
                done.setdefault(n, []).append((share, cuvette))
        for n1, d1 in done.items():
            for n2, d2 in done.items():
                for s1, c1 in d1:
                    for s2, c2 in d2:
                        if n1 != n2 and s2 >= (1 + KIN["oh_margin"]) * s1:
                            return True, (f"KIN runs {n1} (cuvette {c1}) and {n2} (cuvette {c2}): [OH-] x "
                                          f"{s2 / s1:.2f}")
        return False, (f"complete decay runs of dye + NaOH cuvettes: {sorted(done)}; no two at [OH-] "
                       f"{KIN['oh_margin']:.0%} apart")

    return _m2(points, judge, "complete decay runs at two [OH-] levels")


def kinetics_crystal_violet(run):
    points = _kin_points(run)
    return {"E1": _e1(run, "cuvette"), "E2": _kin_e2(run), "M1": _kin_m1(run, points),
            "M2": _kin_m2(run, points)}


# --- titration --------------------------------------------------------------------------------------------------------
def _titration_rows(run):
    """REC record rows in order with their private state: public row + private row (electrode, beaker it stood in,
    immersed, titrant delivered into that beaker) + record id + tick."""
    out = []
    for record in run.evidence.values():
        if record.get("kind") != "titration_record":
            continue
        private = {p["sample_id"]: p for p in record.get("private_rows") or []}
        for row in record["public"]["rows"]:
            out.append(dict(private.get(row["sample_id"], {}), **row, session=record["evidence_id"],
                            tick=round(row["t_s"] * run.physics_hz)))
    return sorted(out, key=lambda r: r["sample_id"])


def _titration_valid(r):
    """A row that measures a beaker: the electrode in its liquid in the holder over the beaker on the stirrer, the
    beaker not overflowed (volumes: the titrant delivered into the beaker, as recorded privately)."""
    return bool(r.get("immersed") and r.get("beaker") and not r.get("overflowed")
                and r.get("titrant_nominal_mL") is not None)


def _titration_e2(rows):
    r = next((r for r in rows if r.get("immersed") and r.get("beaker")), None)
    return (hit(r["tick"], f"electrode immersed in {r['beaker']} (record row {r['sample_id']})") if r
            else miss("no record row with the electrode immersed in a beaker on the stirrer"))


def _titration_m1(rows):
    r = next((r for r in rows if _titration_valid(r) and r["stable"] and (r.get("titrant_nominal_mL") or 0) > 0), None)
    return (hit(r["tick"], f"stable reading in {r['beaker']} after {r['titrant_nominal_mL']} mL titrant") if r
            else miss("no recorded stable reading of a beaker that received titrant"))


def _spans(rows, low, high):
    """(V below, V above) of one record's rows of one beaker: the last titrant volume with pH < low before the first
    with pH > high, or None."""
    below = [float(r["titrant_nominal_mL"]) for r in rows if float(r["pH_display"]) < low]
    above = [float(r["titrant_nominal_mL"]) for r in rows if float(r["pH_display"]) > high]
    if not below or not above or min(above) <= max(below):
        return None
    return max(below), min(above)


def _by_record(prefix, beakers):
    groups = {}
    for r in prefix:
        if _titration_valid(r) and r["beaker"] in beakers:
            groups.setdefault((r["session"], r["beaker"]), []).append(r)
    return groups


def _strong_m2(rows, samples=("S1", "S2", "S3")):
    """One record of a sample beaker spanning the endpoint: rows below and above pH 7."""
    eq = TITRATION["equivalence_pH"]

    def judge(prefix):
        done = sorted(f"{s} ({b})" for (s, b), rs in _by_record(prefix, samples).items() if _spans(rs, eq, eq))
        return bool(done), f"records spanning pH {eq}: {done}"

    return _m2(rows, judge, "a sample titration record across the endpoint")


def _weak_m2(run, rows):
    """A record of a sample beaker spanning its endpoint (rows below pH 7 and above pH 10) with the buffer region
    recorded (a row between 25 % and 75 % of the endpoint volume), and a record of the KHP beaker spanning its
    endpoint."""
    device = run.device
    khp, samples = device.get("khp_beaker"), set(device.get("sample_beakers") or ())
    low, high = WEAK["below_pH"], WEAK["above_pH"]

    def judge(prefix):
        groups = _by_record(prefix, samples | {khp})
        good = set()
        for (_, beaker), rs in groups.items():
            span = _spans(rs, low, high)
            if span is None:
                continue
            if beaker == khp:
                good.add(beaker)
                continue
            v_end = sum(span) / 2
            if any(0.25 * v_end <= float(r["titrant_nominal_mL"]) <= 0.75 * v_end for r in rs):
                good.add(beaker)
        return khp in good and bool(good & samples), f"records across the endpoint: {sorted(good)}"

    return _m2(rows, judge, "KHP and sample titration records")


def titration_basic(run):
    rows = _titration_rows(run)
    return {"E1": _e1(run, "carrier", exclude=("electrode",)), "E2": _titration_e2(rows), "M1": _titration_m1(rows),
            "M2": _strong_m2(rows, tuple(run.device.get("sample_beakers") or ("S1", "S2", "S3")))}


def titration_weak_acid(run):
    rows = _titration_rows(run)
    return {"E1": _e1(run, "carrier", exclude=("electrode",)), "E2": _titration_e2(rows), "M1": _titration_m1(rows),
            "M2": _weak_m2(run, rows)}


RULES = {"spectro_basic": spectro_basic, "titration_basic": titration_basic, "spectro_mixture": spectro_mixture,
         "titration_weak_acid": titration_weak_acid, "kinetics_crystal_violet": kinetics_crystal_violet,
         "dye_absorptivity": dye_absorptivity}


# --- six levels (benchmark/evaluation/process.py) ---------------------------------------------------
# Provisional: each experiment's legacy E1-M2 rule above mapped E1->L1, E2->L2, M1->L3, M2->L4 (process.from_legacy).
# An entry is replaced by Ladder(detect=..., tidy_gates=..., note=...) when that experiment's own L1-L4 detectors
# are written.
from benchmark.evaluation.process import Ladder, from_legacy  # noqa: E402,F401

LADDERS = {exp_id: from_legacy(rule) for exp_id, rule in RULES.items()}


# --- six levels: the chemistry detectors of) ----------------------------------------------
# Common to the spectrophotometer experiments (spectro_basic, spectro_mixture, kinetics_crystal_violet,
# dye_absorptivity): a valid reading is one whose private status is not slot_empty / low_volume / overflowed, of a
# cuvette never spilled holding >= 2 mL. BLANK, its reference and mixing are not looked at (: the gates
# do not check them; a poor choice shows in the error). A READ counts once an observation's screen showed it, a KIN
# point or a SCAN once an export call after it returned it (readout principle). The titration benches: a
# valid row has the electrode in the liquid of a sample beaker (titration_basic S1-S3; titration_weak_acid the KHP
# beaker or a sample beaker) that has neither overflowed nor spilled; STABLE and titrant are not required; it counts
# once an export_titration_log call after it returned it. Buffers are not the measured object.
from itertools import combinations  # noqa: E402

INVALID_STATUS = ("slot_empty", "low_volume", "overflowed")
RANK_TOL = 1e-6                    # compositions: volume fractions; distinct dispensings differ by >= 1e-3
PEAK_BRACKET_NM = 30               # dye_absorptivity: point-by-point READs bracket a maximum from at most this far on each side
SCAN_MIN_PEAK_A = 0.1              # dye_absorptivity: a SCAN locates a maximum when its peak clears the baseline (no upper cap)
KIN_SIX = dict(min_points=3, min_decay=0.10, shape_decay=0.30, min_A=0.02, oh_margin=0.20, same_oh=0.01,
               cv_margin=0.20)
TITRATION_SIX = dict(min_volumes=3, min_pH_span=1.0)
# Titration bench geometry (benchmark/core/titration_apparatus.py): the electrode's seat disc stands on the bridge
# collar BRIDGE_TOP_Z = 34 mm above its carrier's origin, within ELECTRODE_XY_TOL_M = 6 mm of its axis (frame
# positions are whole millimetres: 1 mm added); seat height tolerance 3 mm (chem_backend.SEAT_Z_TOL_M).
ELECTRODE_ABOVE_CARRIER_MM, ELECTRODE_XY_MM, ELECTRODE_Z_MM = 34, 7, 4


def _actions(run):
    actions = run.logs.get("actions")
    if not isinstance(actions, list):
        actions = run.device.get("actions") or []
    return sorted(actions, key=lambda a: a["tick"])


def _hands_on(run, what):
    """L1: the earliest of an object of the bench grasped and lifted LIFT_M and a key registered on the bench."""
    tick, key = _lifted(run)
    found = hit(tick, f"{what} {key} grasped and lifted {LIFT_M * 1000:.0f} mm") if tick is not None else None
    action = next(iter(_actions(run)), None)
    if action is not None and (found is None or action["tick"] < found.tick):
        found = hit(action["tick"], f"key {action['button']} registered")
    return found or miss(f"no {what} grasped and lifted, no key registered")


def _export_ticks(run, name):
    return sorted(c[2] for c in run.calls if c and c[0] == name and (len(c) < 4 or c[3]))


def _after(ticks, tick):
    """The first export tick at or after `tick` (the export returned what was recorded by then), or None."""
    return next((t for t in ticks if t >= tick), None)


def _snapshots(run):
    """[(tick, {cuvette: (fractions, volume mL, overflowed, spilled)})] after each state-log event (replay)."""
    out = []
    for tick, cuvettes in _replay(run):
        out.append((tick, {cid: (c.fractions() if c.volume > 0 else {}, c.volume, c.overflowed, c.spilled)
                           for cid, c in cuvettes.items()}))
    return out


def _at(snaps, tick):
    current = snaps[0][1] if snaps else {}
    for t, state in snaps:
        if t > tick:
            break
        current = state
    return current


def _readable(state, wanted):
    fractions, volume, overflowed, spilled = state
    return volume >= MIN_READ_ML - A_TOL and not overflowed and not spilled and wanted(fractions)


def _v(run, r):
    """A valid reading (READ row or KIN point): the cuvette there, >= 2 mL, not overflowed, never spilled."""
    min_read = run.device.get("min_read_mL", MIN_READ_ML)
    return (r.get("status") not in INVALID_STATUS and not r.get("spilled") and r.get("volume_mL") is not None
            and r["volume_mL"] >= min_read - A_TOL)


def _fractions(history):
    mix = _mix(history)
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()} if total > 0 else {}


def _set_up_in_well(run, wanted, what, readings=()):
    """L2 of the spectrophotometer benches: a cuvette holding `what` (wanted(fractions)), >= 2 mL, not overflowed or
    spilled, standing in the sample well (30 Hz frames + state-log replay); a valid reading of such a cuvette (its
    private record, at the tick it was taken) also counts."""
    frames, _ = _physics(run)
    snaps = _snapshots(run)
    found, i = None, 0
    for f in frames:
        while i + 1 < len(snaps) and snaps[i + 1][0] <= f["tick"]:
            i += 1
        cid = (f.get("seats") or {}).get("spectro")
        state = snaps[i][1].get(int(cid)) if cid is not None and snaps else None
        if state is not None and _readable(state, wanted):
            found = hit(f["tick"], f"cuvette {cid} with {what} in the sample well (>= {MIN_READ_ML} mL)")
            break
    reading = next((r for r in sorted(readings, key=lambda r: r["read_tick"])
                    if _v(run, r) and wanted(_fractions(r["nominal_history"]))), None)
    if reading is not None and (found is None or reading["read_tick"] < found.tick):
        found = hit(reading["read_tick"], f"reading of cuvette {reading['cuvette']} with {what} (private record)",
                    ["l2_from_reading"])
    return found or miss(f"no cuvette with {what} and >= {MIN_READ_ML} mL in the sample well")


def _first_read(run, reads, wanted, what):
    """L3: the first valid READ of a cuvette holding `what`, at the observation that showed it."""
    r = next((r for r in reads if _v(run, r) and wanted(_fractions(r["nominal_history"]))), None)
    return (hit(r["tick"], f"READ {r['sample_id']} of cuvette {r['cuvette']} ({what}) at {r['nm']} nm, shown")
            if r is not None else miss(f"no valid READ of a cuvette with {what} shown on the screen"))


# --- spectro_basic ------------------------------------------------------------------------------------------------
def _has_basic(f):
    return f.get("stock", 0) > 0 or f.get("unknown", 0) > 0


def _basic_l4(run, reads):
    """Two linearly independent compositions (stock fraction, unknown fraction) among the shown valid READs at one
    wavelength: two states fix the absorptivity and the unknown's concentration (external standard, standard
    addition...; dilutions of the unknown alone do not)."""
    def judge(prefix):
        by_nm = {}
        for r in prefix:
            f = _fractions(r["nominal_history"])
            vec = (f.get("stock", 0.0), f.get("unknown", 0.0))
            if vec != (0.0, 0.0):
                by_nm.setdefault(r["nm"], []).append((vec, r))
        for nm, rows in sorted(by_nm.items()):
            for (a, ra), (b, rb) in combinations(rows, 2):
                if abs(a[0] * b[1] - a[1] * b[0]) > RANK_TOL:
                    return True, (f"{nm} nm: READs {ra['sample_id']} (cuvette {ra['cuvette']}) and {rb['sample_id']} "
                                  f"(cuvette {rb['cuvette']}) of independent compositions")
        return False, "no wavelength with READs of two independent compositions"

    return _m2([r for r in reads if _v(run, r)], judge, "two independent compositions at one wavelength")


def spectro_basic_levels(run):
    """L1-L4 of spectro_basic."""
    shown = _by_tick(_reads(run, seen=True))
    return {"L1": _hands_on(run, "cuvette"),
            "L2": _set_up_in_well(run, _has_basic, "STOCK or UNKNOWN", _reads(run)),
            "L3": _first_read(run, shown, _has_basic, "STOCK or UNKNOWN"),
            "L4": _basic_l4(run, shown)}


# --- spectro_mixture ----------------------------------------------------------------------------------------------
def _mixture_lines(run):
    lines, mixture = run.device.get("source_of") or {"X": "stock", "Y": "unknown"}, run.device.get("mixture") or {}
    return lines["X"], lines["Y"], mixture.get("label", "mixture")


def _mixture_l4(run, reads):
    """At two or more wavelengths the compositions (dye X from its stock, dye Y from its stock, mixture share) of the
    shown valid READs span three dimensions (pure standards, standard addition, synthetic mixtures...)."""
    x, y, m = _mixture_lines(run)

    def judge(prefix):
        by_nm = {}
        for r in prefix:
            f = _fractions(r["nominal_history"])
            by_nm.setdefault(r["nm"], []).append((f.get(x, 0.0), f.get(y, 0.0), f.get(m, 0.0)))
        done = sorted(nm for nm, rows in by_nm.items()
                      if len(rows) >= 3 and np.linalg.matrix_rank(np.array(rows), tol=RANK_TOL) == 3)
        return len(done) >= 2, f"wavelengths with compositions of rank 3: {done}"

    return _m2([r for r in reads if _v(run, r)], judge, "compositions of rank 3 at two wavelengths")


def spectro_mixture_levels(run):
    """L1-L4 of spectro_mixture."""
    x, y, m = _mixture_lines(run)

    def wanted(f):
        return f.get(x, 0) > 0 or f.get(y, 0) > 0 or f.get(m, 0) > 0

    shown = _by_tick(_reads(run, seen=True))
    return {"L1": _hands_on(run, "cuvette"),
            "L2": _set_up_in_well(run, wanted, "dye X, dye Y or the mixture", _reads(run)),
            "L3": _first_read(run, shown, wanted, "dye X, dye Y or the mixture"),
            "L4": _mixture_l4(run, shown)}


# --- dye_absorptivity ---------------------------------------------------------------------------------------------
def _scan_maxima(scan):
    """Wavelengths of the maximum of one SCAN spectrum: the vertex of a parabola through the 5-point moving average
    around its maximum, and the raw maximum. No upper bound on the peak absorbance."""
    nm, a = np.asarray(scan["wavelength_nm"], float), np.asarray(scan["A"], float)
    if len(a) < 11 or a.max() < SCAN_MIN_PEAK_A:
        return []
    raw = float(nm[int(np.argmax(a))])
    smooth = np.convolve(a, np.ones(5) / 5, mode="same")
    i = int(np.clip(np.argmax(smooth[2:-2]) + 2, 5, len(a) - 6))
    coef = np.polyfit(nm[i - 5:i + 6] - nm[i], smooth[i - 5:i + 6], 2)
    vertex = float(nm[i] + (-coef[1] / (2 * coef[0]) if coef[0] < 0 else 0.0))
    return [vertex, raw]


def _dye_l4(run, reads):
    """A located maximum (an exported SCAN of a cuvette holding dye, peak A >= 0.1; or valid READs of one cuvette at
    three wavelengths, the middle one highest and each neighbour at most 30 nm from it) and a shown valid READ of a
    cuvette holding dye (a known concentration: both stocks are labelled) within 3 nm of it."""
    snaps = _snapshots(run)
    exports = _export_ticks(run, "export_spectro_scans")
    items = [dict(r, kind="read") for r in reads if _v(run, r) and _has_basic(_fractions(r["nominal_history"]))]
    for scan in run.logs.get("spectro_scans") or []:
        if scan.get("condition") in INVALID_STATUS or scan.get("condition") is None or scan.get("cuvette") is None:
            continue
        taken = round(float(scan["t_s"]) * run.physics_hz)
        state = _at(snaps, taken).get(int(scan["cuvette"]))
        exported = _after(exports, taken)
        maxima = _scan_maxima(scan)
        if state is None or not _readable(state, _has_basic) or exported is None or not maxima:
            continue
        items.append({"kind": "scan", "tick": exported, "maxima": maxima, "source": f"SCAN {scan['scan_id']}"})
    items.sort(key=lambda item: item["tick"])

    def judge(prefix):
        rows = [r for r in prefix if r["kind"] == "read"]
        maxima = [(p, s["source"]) for s in prefix if s["kind"] == "scan" for p in s["maxima"]]
        curves = {}
        for r in rows:
            curves.setdefault((r["cuvette"], tuple(map(tuple, r["nominal_history"]))), {})[r["nm"]] = r["A"]
        for (cuvette, _), curve in curves.items():
            nms = sorted(curve)
            for k, mid in enumerate(nms):
                lo = [n for n in nms[:k] if mid - n <= PEAK_BRACKET_NM and curve[n] <= curve[mid]]
                hi = [n for n in nms[k + 1:] if n - mid <= PEAK_BRACKET_NM and curve[n] <= curve[mid]]
                if lo and hi:
                    a, b = lo[-1], hi[0]
                    xs, ys = np.array([a, mid, b], float), np.array([curve[a], curve[mid], curve[b]], float)
                    coef = np.polyfit(xs - mid, ys, 2)
                    vertex = mid + (-coef[1] / (2 * coef[0]) if coef[0] < 0 else 0.0)
                    source = f"READs of cuvette {cuvette} at {a}/{mid}/{b} nm"
                    maxima += [(float(mid), source), (float(vertex), source)]
        for peak, source in maxima:
            near = [r for r in rows if abs(r["nm"] - peak) <= DYE_PEAK_TOL_NM]
            if near:
                return True, f"maximum {peak:.1f} nm ({source}), READ {near[0]['sample_id']} at {near[0]['nm']} nm"
        return False, f"{len(maxima)} located maxima, no READ of a dye cuvette within {DYE_PEAK_TOL_NM} nm of one"

    return _m2(items, judge, "a located maximum and a known-concentration READ there")


def dye_absorptivity_levels(run):
    """L1-L4 of dye_absorptivity."""
    shown = _by_tick(_reads(run, seen=True))
    return {"L1": _hands_on(run, "cuvette"),
            "L2": _set_up_in_well(run, _has_basic, "dye", _reads(run)),
            "L3": _first_read(run, shown, _has_basic, "dye"),
            "L4": _dye_l4(run, shown)}


# --- kinetics_crystal_violet --------------------------------------------------------------------------------------
def _reacting(f):
    return f.get("stock", 0) > 0 and f.get(NAOH, 0) > 0


def _kin_l4(run, points):
    """Exported valid KIN points grouped by cuvette and dispensing history (not by run): two groups at [OH-] 20 %
    apart, each with >= 3 points (first A >= 0.02) falling >= 10 % below its first point (n and k); and the order in
    crystal violet: one such group falling below 70 % of its first point (the curve's shape), or two groups at one
    [OH-] (within 1 %) and [CV] 20 % apart (initial rates)."""
    exports = _export_ticks(run, "export_spectro_kinetics")
    items = []
    for p in points:
        exported = _after(exports, p["tick"])
        f = _fractions(p["nominal_history"])
        if exported is not None and _v(run, p) and _reacting(f):
            items.append(dict(p, read_tick=p["tick"], tick=exported, oh=f[NAOH], cv=f["stock"]))
    items.sort(key=lambda p: (p["tick"], p["read_tick"]))
    k = KIN_SIX

    def judge(prefix):
        groups = {}
        for p in prefix:
            groups.setdefault((p["cuvette"], tuple(map(tuple, p["nominal_history"]))), []).append(p)
        decayed = []
        for (cuvette, _), ps in groups.items():
            ps = sorted(ps, key=lambda p: p["read_tick"])
            a0, low = ps[0]["A"], min(p["A"] for p in ps)
            if len(ps) >= k["min_points"] and a0 >= k["min_A"] and low <= (1 - k["min_decay"]) * a0:
                decayed.append({"cuvette": cuvette, "oh": ps[0]["oh"], "cv": ps[0]["cv"],
                                "shape": low <= (1 - k["shape_decay"]) * a0})
        n_ok = next(((a, b) for a, b in combinations(decayed, 2)
                     if max(a["oh"], b["oh"]) >= (1 + k["oh_margin"]) * min(a["oh"], b["oh"])), None)
        m_ok = next((g for g in decayed if g["shape"]), None) or next(
            ((a, b) for a, b in combinations(decayed, 2)
             if abs(a["oh"] - b["oh"]) <= k["same_oh"] * max(a["oh"], b["oh"])
             and max(a["cv"], b["cv"]) >= (1 + k["cv_margin"]) * min(a["cv"], b["cv"])), None)
        note = (f"decaying groups (cuvette, [OH-] share): "
                f"{[(g['cuvette'], round(g['oh'], 4)) for g in decayed]}; [OH-] pair {'yes' if n_ok else 'no'}; "
                f"order in CV {'yes' if m_ok else 'no'}")
        return bool(n_ok and m_ok), note

    return _m2(items, judge, "decays at two [OH-] levels and the order in crystal violet")


def kinetics_crystal_violet_levels(run):
    """L1-L4 of kinetics_crystal_violet."""
    points = _kin_points(run)
    exports = _export_ticks(run, "export_spectro_kinetics")
    readings = [dict(p, read_tick=p["tick"]) for p in points]
    first = None
    for p in points:
        exported = _after(exports, p["tick"])
        if exported is not None and _v(run, p) and _reacting(_fractions(p["nominal_history"])) and \
                (first is None or exported < first[0]):
            first = (exported, p)
    l3 = (hit(first[0], f"KIN run {first[1]['run']} point {first[1]['point']} of cuvette {first[1]['cuvette']} "
                        f"(CV + NaOH), exported") if first else miss("no valid KIN point of a CV + NaOH cuvette exported"))
    return {"L1": _hands_on(run, "cuvette"),
            "L2": _set_up_in_well(run, _reacting, "crystal violet and NaOH", readings),
            "L3": l3, "L4": _kin_l4(run, points)}


# --- titration_basic, titration_weak_acid ---------------------------------------------------------------------
def _titration_beakers(run, weak):
    samples = list(run.device.get("sample_beakers") or ("S1", "S2", "S3"))
    khp = run.device.get("khp_beaker")
    return ([khp] if weak and khp else []) + [b for b in samples if b != khp]


def _titration_log(run):
    log = run.logs.get("state_log")
    if not isinstance(log, list):
        log = (run.device.get("chemistry") or {}).get("state_log") or []
    return sorted(log, key=lambda e: e["tick"])


def _spilled_at(run):
    out = {}
    for e in _titration_log(run):
        if e.get("event") == "spill" and e.get("beaker") is not None:
            out.setdefault(e["beaker"], e["tick"])
    return out


def _v_row(r, beakers, spills):
    return (bool(r.get("immersed")) and r.get("beaker") in beakers and not r.get("overflowed")
            and not spills.get(r["beaker"], math.inf) <= r["tick"])


def _titration_set_up(run, rows, beakers, spills):
    """L2: a sample beaker on the stirrer with the electrode in its liquid (in its carrier's holder, not held). From
    the bench's electrode events (records), else from the REC rows' private state and the 30 Hz frames
    (electrode on its carrier's collar: no tilt in the frames, flagged)."""
    found = None
    events = [e for e in _titration_log(run) if e.get("event") == "electrode"]
    for e in events:
        if e.get("place") == "holder" and e.get("immersed") and e.get("beaker") in beakers \
                and not spills.get(e["beaker"], math.inf) <= e["tick"]:
            found = hit(e["tick"], f"electrode in the liquid of {e['beaker']} on the stirrer (bench record)")
            break
    row = next((r for r in rows if _v_row(r, beakers, spills)), None)
    if row is not None and (found is None or row["tick"] < found.tick):
        found = hit(row["tick"], f"electrode immersed in {row['beaker']} (record row {row['sample_id']})")
    if events:
        return found or miss("the electrode never stood in the liquid of a sample beaker on the stirrer")
    frames, _ = _physics(run)
    for f in frames:
        if found is not None and f["tick"] >= found.tick:
            break
        carrier = (f.get("seats") or {}).get("stirrer")
        xyz = f.get("xyz_mm") or {}
        if carrier not in beakers or f.get("held") in ("electrode", carrier) or "electrode" not in xyz \
                or carrier not in xyz or spills.get(carrier, math.inf) <= f["tick"]:
            continue
        e, c = xyz["electrode"], xyz[carrier]
        if math.hypot(e[0] - c[0], e[1] - c[1]) <= ELECTRODE_XY_MM and \
                abs(e[2] - c[2] - ELECTRODE_ABOVE_CARRIER_MM) <= ELECTRODE_Z_MM:
            found = hit(f["tick"], f"electrode on the collar of {carrier} on the stirrer (frames)",
                        ["l2_from_frames_without_tilt"])
            break
    return found or miss("the electrode never stood in the liquid of a sample beaker on the stirrer")


def _screen_states(run, beakers, spills):
    """pH the Agent saw on the screen in an observation (logs screen_views, records), with the
    electrode in the liquid of a sample beaker (same validity as a REC row), as data items at the observation's tick
    (no export needed). Process levels only: the submission's evidence stays the cited REC rows (L5 gates)."""
    out = []
    for v in run.logs.get("screen_views") or []:
        r = {"tick": v["tick"], "row_tick": v["tick"], "beaker": v.get("beaker"), "immersed": v.get("immersed"),
             "overflowed": v.get("overflowed"), "titrant_nominal_mL": v.get("titrant_nominal_mL"),
             "pH_display": v.get("pH"), "source": f"screen at the observation at tick {v['tick']}"}
        if _v_row(r, beakers, spills) and r["pH_display"] is not None and r["titrant_nominal_mL"] is not None:
            out.append(r)
    return out


def _titration_q(rows, beaker):
    """The data of one beaker reach its endpoint region: >= 3 titrant volumes, displayed pH spanning >= 1.0."""
    mine = [r for r in rows if r["beaker"] == beaker and r.get("titrant_nominal_mL") is not None]
    volumes = {round(float(r["titrant_nominal_mL"]), 2) for r in mine}
    ph = [float(r["pH_display"]) for r in mine]
    t = TITRATION_SIX
    ok = len(volumes) >= t["min_volumes"] and bool(ph) and max(ph) - min(ph) >= t["min_pH_span"] - 1e-9
    return ok, f"{beaker}: {len(volumes)} volumes, pH span {max(ph) - min(ph) if ph else 0:.2f}"


def titration_levels(run, weak):
    """L1-L4 of titration_basic / titration_weak_acid."""
    beakers = _titration_beakers(run, weak)
    spills = _spilled_at(run)
    rows = sorted(_titration_rows(run), key=lambda r: r["tick"])
    exports = _export_ticks(run, "export_titration_log")
    data = []
    for r in rows:
        exported = _after(exports, r["tick"])
        if exported is not None and _v_row(r, beakers, spills):
            data.append(dict(r, row_tick=r["tick"], tick=exported, source=f"record row {r['sample_id']}"))
    # L3/L4 accept exported REC rows and pH seen on the screen in observations (screen_views, at the observation's
    # tick); the submission's evidence stays the cited REC rows (L5 gates). .
    items = sorted(data + _screen_states(run, beakers, spills), key=lambda r: (r["tick"], r["row_tick"]))
    first = items[0] if items else None
    if first is None:
        l3 = miss("no exported row and no observed screen with the electrode in a sample beaker's liquid")
    elif "sample_id" in first:
        l3 = hit(first["tick"], f"row {first['sample_id']} (electrode in {first['beaker']}) exported")
    else:
        l3 = hit(first["tick"], f"pH {first['pH_display']} seen on the screen (electrode in {first['beaker']})")
    khp = beakers[0] if weak else None
    samples = [b for b in beakers if b != khp]

    def judge(prefix):
        notes, good = [], set()
        for b in beakers:
            ok, note = _titration_q(prefix, b)
            notes.append(note)
            if ok:
                good.add(b)
        done = bool(good & set(samples)) and (khp is None or khp in good)
        return done, "; ".join(notes)

    what = ("the KHP beaker and a sample beaker each" if weak else "a sample beaker") + \
        " with >= 3 titrant volumes and pH spanning >= 1.0"
    return {"L1": _hands_on(run, "carrier or electrode"),
            "L2": _titration_set_up(run, rows, beakers, spills),
            "L3": l3, "L4": _m2(items, judge, what)}


def titration_basic_levels(run):
    """L1-L4 of titration_basic."""
    return titration_levels(run, weak=False)


def titration_weak_acid_levels(run):
    """L1-L4 of titration_weak_acid."""
    return titration_levels(run, weak=True)


_SIX = {"spectro_basic": spectro_basic_levels, "spectro_mixture": spectro_mixture_levels,
        "dye_absorptivity": dye_absorptivity_levels, "kinetics_crystal_violet": kinetics_crystal_violet_levels,
        "titration_basic": titration_basic_levels, "titration_weak_acid": titration_weak_acid_levels}
for _exp, _detect in _SIX.items():
    LADDERS[_exp] = Ladder(detect=_detect, note="six-level detectors")

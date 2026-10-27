"""Process rules of the optics kit: lens_focal, grating_wavelength, optical_relay, slit_width,
plate_index. The hand-slid bench (lens_focal and grating_wavelength the other three since hand_* rules, HandBench) reads the bench's state_log, rider trace and bench_view observations; the
keypad bench (the rest of this docstring) the stage controller's records.

Sources: the optics bench's state_log (bench_logs.json: every accepted carriage move, 'go' or 'jog', with its tick and
target; 'seats' when the element on a lens carriage changes), its screen captures (evidence.json: the public positions
to 0.1 mm, the private seats and the optical state the image was computed for), the elements' optics (episode.json
device) and the backend's frame log (private/physics.jsonl: held object and joint angles at 30 Hz). Between records the
carriage positions are rebuilt from the start positions (private/configuration.json) and the accepted moves at the
controller's constant speed, and checked against every capture (flag timeline_mismatch).

An optical state is measurable when it can be recorded: every carriage stands still (capture_screen works), or a
capture (a sweep's) was taken in it. A state already true at the bench's first tick counts, flagged true_at_start.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from benchmark.evaluation.process import hit, miss

_CORE = str(Path(__file__).resolve().parents[2] / "core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)
import optics_model as M                                              # noqa: E402
from optics_bench import MOTOR_MM_S                                   # noqa: E402

LIFT_M = 0.02            # a grasped holder counts as taken off its stand once the hand has risen this far with it
SAME_MM = 0.05           # recorded positions (0.1 mm) that count as the same
START = 1                # the bench's first tick (its first 'seats' entry)


# --- the rail as the records show it -----------------------------------------------------------------------------
class Rail:
    def __init__(self, run):
        self.run, self.hz = run, run.physics_hz
        self.elements = (run.device or {}).get("lenses") or {}
        self.laser_m = float((run.device or {}).get("laser_nm") or 0.0) * 1e-9
        config = (run.private_json("configuration.json") or {}).get("bench") or {}
        self.initial = {a: float(v) for a, v in (config.get("initial_mm") or {}).items()}
        self.screen_axis = config.get("screen_axis", "S")
        log = run.logs.get("state_log") or []
        self.seat_log = [(e["tick"], e["seated"]) for e in log if e.get("event") == "seats"]
        self.moves, self.segments = [], {a: [(0, p, p)] for a, p in self.initial.items()}
        self.step = step = MOTOR_MM_S / self.hz
        for e in log:
            if e.get("event") not in ("go", "jog") or e.get("axis") not in self.segments:
                continue
            segs = self.segments[e["axis"]]
            if abs(e["target_mm"] - segs[-1][2]) <= 1e-9:
                continue                                     # accepted, but the carriage has nowhere to go
            segs.append((e["tick"], self._at(segs[-1], e["tick"] - 1, step), float(e["target_mm"])))
            self.moves.append(e)
        self.captures = [self._capture(r) for r in sorted(
            (r for r in run.evidence.values() if r.get("kind") == "screen_capture"), key=lambda r: r["public"]["capture"])]
        self.flags = []
        if self.initial and any(max(abs(c["rebuilt"][a] - c["positions"][a]) for a in c["positions"]) > SAME_MM + 1e-6
                                for c in self.captures):
            self.flags.append("timeline_mismatch")

    @staticmethod
    def _at(seg, tick, step):
        """Position after `tick` on a segment (press tick, position then, target); the press tick already moves."""
        t0, p0, target = seg
        d = target - p0
        return p0 + math.copysign(min(max(tick - t0 + 1, 0) * step, abs(d)), d) if d else p0

    def _arrival(self, seg):
        t0, p0, target = seg
        return t0 - 1 if target == p0 else t0 + math.ceil(abs(target - p0) / self.step - 1e-9) - 1

    def _segment(self, axis, tick):
        return [s for s in self.segments[axis] if s[0] <= tick][-1]

    def positions(self, tick):
        return {a: self._at(self._segment(a, tick), tick, self.step) for a in self.segments}

    def stopped(self, tick):
        return all(self._arrival(self._segment(a, tick)) <= tick for a in self.segments)

    def seated(self, tick):
        rows = [s for t, s in self.seat_log if t <= max(tick, START)]
        return dict(rows[-1]) if rows else {}

    def _state(self, tick, positions, seated, capture=None):
        elements = sorted((round(positions[a] + self.elements[n]["offset_mm"], 4), n)
                          for a, n in seated.items() if n and a in positions and n in self.elements)
        return {"tick": tick, "positions": positions, "seated": seated, "elements": elements,
                "screen": positions.get(self.screen_axis), "capture": capture}

    def _capture(self, record):
        public, private = record["public"], record["private"]
        tick = round(public["t_s"] * self.hz)
        elements, screen = private["image_state"]
        return {"tick": tick, "positions": public["positions_mm"], "seated": private["seated"],
                "elements": [tuple(e) for e in elements], "screen": screen, "capture": record["evidence_id"],
                "rebuilt": self.positions(tick) if self.initial else {}}

    def still_states(self):
        """The rebuilt state at every tick the stopped rail's state can have changed (stops and seat changes)."""
        if not self.initial:
            return []
        ticks = {START} | {t for t, _ in self.seat_log} | {self._arrival(s) for segs in self.segments.values()
                                                           for s in segs[1:]}
        return [self._state(t, self.positions(t), self.seated(t)) for t in sorted(ticks) if self.stopped(t)]

    def measurable(self):
        return sorted(self.still_states() + self.captures, key=lambda s: (s["tick"], s["capture"] is not None))

    # optics of a state
    def on(self, state, axis):
        return state["seated"].get(axis)

    def x(self, state, name):
        return next((x for x, n in state["elements"] if n == name), None)

    def blur_and_m(self, state):
        """(blur radius mm, magnification) of the object image, as the gates compute it; (None, None) without a lens."""
        lenses = [(x / 1000, self.elements[n]["f_m"]) for x, n in state["elements"]
                  if 0 < x < state["screen"] and self.elements[n]["kind"] == "lens"]
        if not lenses:
            return None, None
        m, blur = M.magnification_and_blur(lenses, state["screen"] / 1000,
                                           self.elements[state["elements"][0][1]]["aperture_m"])
        return blur * 1000, m

    def focus_mm(self, state):
        elements = [(x / 1000, self.elements[n]) for x, n in state["elements"] if 0 < x < state["screen"]]
        return 1000 * M.focus_position_m(elements)


def first(states, test, basis, *, rail=None):
    """Hit at the first state passing `test`; true at the bench's first tick is flagged."""
    for s in states:
        if test(s):
            flags = (["true_at_start"] if s["tick"] <= START and s["capture"] is None else []) + (rail.flags if rail else [])
            return hit(s["tick"], basis + (f" ({s['capture']})" if s["capture"] else ""), flags)
    return miss("no " + basis)


# --- E1 --------------------------------------------------------------------------------------------------------
def carriage_moved(rail):
    """E1 of the keypad experiments: the first accepted carriage move (a 'go' or jog the bench accepted that gives the
    carriage a new target; the robot pressed the keys, so the move is physical)."""
    if not rail.moves:
        return miss("no accepted carriage move")
    e = rail.moves[0]
    return hit(e["tick"], f"carriage {e['axis']} sent to {e['target_mm']} mm ({e['event']})")


_DH = ((0, .333, 0), (0, 0, -math.pi / 2), (0, .316, math.pi / 2), (.0825, 0, math.pi / 2),
       (-.0825, .384, -math.pi / 2), (0, 0, math.pi / 2), (.088, 0, math.pi / 2))


def flange_z(q):
    """Height of the Panda flange (the hand's origin) above the robot base, from the joint angles (Craig DH)."""
    t = np.eye(4)
    for (a, d, alpha), theta in zip(_DH, q):
        ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(theta), math.sin(theta)
        t = t @ np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])
    return float((t @ np.array([0, 0, .107, 1]))[2])


def episode_frames(run):
    """physics.jsonl frame rows of the episode (the settling before tick 0 is logged too; its frames restart)."""
    rows, last = [], None
    for row in run.frames:
        if "frame" not in row:
            continue
        if last is not None and row["frame"] <= last:
            rows = []
        rows.append(row)
        last = row["frame"]
    return rows


def lifted(run, objects):
    """E1 of the pick experiments: one of `objects` held by the gripper and raised LIFT_M, from its height at the
    first frame of that hold. The height is the object's own (frame xyz_mm); in records without it, the hand's,
    from the joint angles (the held object moves rigidly with the hand)."""
    frame_ticks = run.physics_hz // 30
    base = None
    for row in episode_frames(run):
        held = row.get("held")
        if held in objects and held in (row.get("xyz_mm") or {}):
            z, source = row["xyz_mm"][held][2] / 1000, "object"
        elif held in objects and row.get("q_rad"):
            z, source = flange_z(row["q_rad"]), "hand"
        else:
            base = None
            continue
        if base is None or base[:2] != (held, source):
            base = (held, source, z)
        elif z - base[2] >= LIFT_M:
            return hit(row["frame"] * frame_ticks, f"{held} held and raised {1000 * (z - base[2]):.0f} mm ({source} height)",
                       [] if source == "object" else ["lift_from_joint_angles"])
    return miss(f"none of {sorted(objects)} held and raised {1000 * LIFT_M:.0f} mm")


def seated_from_log(rail, test, basis):
    """First seat change of the bench log after which `test(seated)` holds."""
    for tick, seated in rail.seat_log:
        if test(seated):
            return hit(tick, basis, ["true_at_start"] if tick <= START else [])
    return miss("never " + basis)


def first_capture(rail, test, basis):
    return first(rail.captures, test, basis, rail=rail)


def until(rail, enough, basis):
    """M2: the tick of the capture after which the captures so far hold the protocol's data set."""
    for k in range(1, len(rail.captures) + 1):
        if enough(rail.captures[:k]):
            c = rail.captures[k - 1]
            return hit(c["tick"], f"{basis} (complete at {c['capture']})", rail.flags)
    return miss(f"not {basis} ({len(rail.captures)} captures)", rail.flags)


def span(values):
    return max(values) - min(values) if values else 0.0


# --- lens_focal ----------------------------------------------------------------------------------------------
MAX_BLUR_MM = 1.0


def lens_focal(run):
    """E1 carriage moved; E2 lens L on A and a sharp image (blur <= 1 mm) in a measurable state; M1 the first
    capture in that state; M2 two sharp captures at one screen position with the lens at the two conjugate positions
    (lens positions >= 1 mm apart, one image enlarged and one reduced)."""
    rail = Rail(run)

    def sharp(s):
        blur, _ = rail.blur_and_m(s) if rail.on(s, "A") == "L" else (None, None)
        return blur is not None and blur <= MAX_BLUR_MM

    def bessel_pair(captures):
        good = [(c["positions"], abs(rail.blur_and_m(c)[1])) for c in captures if sharp(c)]
        return any(abs(p["S"] - q["S"]) <= SAME_MM and abs(p["A"] - q["A"]) >= 1.0 and (m - 1) * (n - 1) < 0
                   for i, (p, m) in enumerate(good) for q, n in good[:i])

    return {"E1": carriage_moved(rail),
            "E2": first(rail.measurable(), sharp, f"lens L on A, image blur <= {MAX_BLUR_MM} mm", rail=rail),
            "M1": first_capture(rail, sharp, f"capture with lens L on A, blur <= {MAX_BLUR_MM} mm"),
            "M2": until(rail, bessel_pair, "two sharp captures at one screen position, lens at both conjugates")}


# --- grating_wavelength --------------------------------------------------------------------------------------
def grating_wavelength(run):
    """E1 carriage moved; E2 grating G on A with both first-order spots on the screen (centres at least 3 spot sigma
    inside its edge) in a measurable state; M1 the first capture in that state; M2 >= 3 such captures whose
    grating-screen distances (recorded positions) span >= 40 mm."""
    rail = Rail(run)
    edge = M.SCREEN_M / 2 - 3 * M.SPOT_SIGMA_M

    def orders(s):
        x = rail.x(s, "G") if rail.on(s, "A") == "G" else None
        if x is None or not 0 < x < s["screen"]:
            return False
        lines = rail.elements["G"]["lines_per_mm"]
        spots = dict(M.grating_spots(rail.laser_m, lines, (s["screen"] - x) / 1000, max_order=1))
        return all(m in spots and abs(spots[m]) <= edge for m in (-1, 1))

    def enough(captures):
        d = [c["positions"]["S"] - c["positions"]["A"] for c in captures if orders(c)]
        return len(d) >= 3 and span(d) >= 40.0

    return {"E1": carriage_moved(rail),
            "E2": first(rail.measurable(), orders, "grating on A, both first orders on the screen", rail=rail),
            "M1": first_capture(rail, orders, "capture with both first orders on the screen"),
            "M2": until(rail, enough, ">= 3 captures with both first orders, distances spanning >= 40 mm")}


# --- optical_relay -------------------------------------------------------------------------------------------
RELAY_BLUR_MM = 0.25


def optical_relay(run):
    """E1 lens L1 or L2 grasped and raised off its stand; E2 the bench log has the two lenses on carriages A and B;
    M1 the first capture with both lenses in place; M2 a capture with both lenses in place and blur <= 0.25 mm."""
    rail = Rail(run)
    both = lambda seated: sorted(v for v in (seated.get("A"), seated.get("B")) if v) == ["L1", "L2"]  # noqa: E731

    def focused(c):
        blur, _ = rail.blur_and_m(c) if both(c["seated"]) else (None, None)
        return blur is not None and blur <= RELAY_BLUR_MM

    return {"E1": lifted(run, {"L1", "L2"}),
            "E2": seated_from_log(rail, both, "both lenses on carriages A and B"),
            "M1": first_capture(rail, lambda c: both(c["seated"]), "capture with both lenses on A and B"),
            "M2": until(rail, lambda cs: any(focused(c) for c in cs),
                        f"a capture with both lenses in place, blur <= {RELAY_BLUR_MM} mm")}


# --- slit_width ----------------------------------------------------------------------------------------------
SLIT_MOVED_MM = 1.0


def slit_width(run):
    """E1 carriage moved; E2 slit SL on A with the slit-screen distance changed by >= 1 mm from the start, in a
    measurable state; M1 the first capture in that state; M2 >= 3 captures with SL on A whose slit-screen distances
    (recorded positions, the start distance included) span >= 100 mm."""
    rail = Rail(run)
    d0 = rail.initial.get("S", 0.0) - rail.initial.get("A", 0.0)

    def moved(s):
        return rail.on(s, "A") == "SL" and abs(s["positions"]["S"] - s["positions"]["A"] - d0) >= SLIT_MOVED_MM

    def enough(captures):
        d = [c["positions"]["S"] - c["positions"]["A"] for c in captures if rail.on(c, "A") == "SL"]
        return len(d) >= 3 and span(d) >= 100.0

    return {"E1": carriage_moved(rail),
            "E2": first(rail.measurable(), moved, "slit on A at a new slit-screen distance", rail=rail),
            "M1": first_capture(rail, moved, "capture with the slit on A at a new distance"),
            "M2": until(rail, enough, ">= 3 slit captures spanning >= 100 mm")}


# --- plate_index ---------------------------------------------------------------------------------------------
FOCUS_TOL_MM = 4.0


def plate_index(run):
    """E1 plate P grasped and raised off its stand; E2 the bench log has P on carriage B with lens L on A; M1 the
    first capture in that state with the screen within 4 mm of the true focus; M2 at one lens position >= 5 captures
    without the plate (it on neither carriage) and >= 5 with it on B, each group's screen positions straddling its
    true focus. Captures without the plate taken before E1 count toward M2."""
    rail = Rail(run)
    plate = next((k for k, v in rail.elements.items() if v["kind"] == "plate"), "P")
    with_plate = lambda seated: seated.get("A") == "L" and seated.get("B") == plate  # noqa: E731
    without = lambda seated: seated.get("A") == "L" and plate not in seated.values()  # noqa: E731

    def near_focus(c):
        return with_plate(c["seated"]) and abs(c["screen"] - rail.focus_mm(c)) <= FOCUS_TOL_MM

    def enough(captures):
        groups = {}
        for c in captures:
            for name, test in (("with", with_plate), ("without", without)):
                if test(c["seated"]):
                    groups.setdefault((round(rail.x(c, "L"), 2), name), []).append(c)
        for (lens_x, name), cs in groups.items():
            other = groups.get((lens_x, "without" if name == "with" else "with"), [])
            if all(len(g) >= 5 and min(c["screen"] for c in g) <= rail.focus_mm(g[0]) <= max(c["screen"] for c in g)
                   for g in (cs, other)):
                return True
        return False

    return {"E1": lifted(run, {plate}),
            "E2": seated_from_log(rail, with_plate, "plate on B with lens L on A"),
            "M1": first_capture(rail, near_focus, f"capture with the plate in, screen within {FOCUS_TOL_MM} mm of focus"),
            "M2": until(rail, enough, ">= 5 captures without and >= 5 with the plate at one lens position, each "
                                      "straddling its focus")}


# --- the hand-slid bench (lens_focal, grating_wavelength plate_index) ---------------------------
# Sources: bench_logs.json (state_log: 'power' switches and 'seats' changes with their ticks; rider_trace: rider
# positions at 30 Hz whenever one moved >= 0.05 mm; start_mm) and the observations (evidence.json 'bench_view': tick,
# true rider positions, seats, source, and what the screen showed).
HAND_MOVED_MM = 1.0       # a rider counts as moved by the robot once it is this far from its start position


class HandBench:
    def __init__(self, run):
        logs = run.logs
        self.log = logs.get("state_log") or []
        self.trace = logs.get("rider_trace") or []
        self.start = {a: float(v) for a, v in (logs.get("start_mm") or {}).items()}
        self.observations = sorted((r for r in run.evidence.values() if r.get("kind") == "bench_view"),
                                   key=lambda r: r["tick"])

    def power_on(self, tick):
        on = False
        for e in self.log:
            if e.get("event") == "power" and e["tick"] <= tick:
                on = bool(e["on"])
        return on

    def seated(self, tick):
        rows = [e["seated"] for e in self.log if e.get("event") == "seats" and e["tick"] <= tick]
        return rows[-1] if rows else {}

    def first_moved(self):
        for row in self.trace:
            for a, p in sorted(row["positions_mm"].items()):
                if a in self.start and abs(p - self.start[a]) >= HAND_MOVED_MM:
                    return row["tick"], f"rider {a} slid {p - self.start[a]:+.1f} mm"
        return None

    def first_power(self):
        e = next((e for e in self.log if e.get("event") == "power" and e.get("on")), None)
        return None if e is None else (e["tick"], "source switched on")


def _hand_e1(bench):
    found = [x for x in (bench.first_moved(), bench.first_power()) if x is not None]
    if not found:
        return miss("no rider moved and the source never switched on")
    tick, basis = min(found)
    return hit(tick, basis)


def _hand_e2(bench, element, test, basis):
    """First tick at which the source is on, `element` is seated on A and `test(tick)` holds (checked at every
    power, seat and trace event)."""
    ticks = sorted({e["tick"] for e in bench.log if e.get("event") in ("power", "seats")} |
                   {r["tick"] for r in bench.trace})
    for t in ticks:
        if bench.power_on(t) and bench.seated(t).get("A") == element and test(t):
            return hit(t, basis)
    return miss("never " + basis)


def _hand_m(bench, valid, basis1, basis2):
    """M1 the first valid observation; M2 the first one that is enough for the result, which on these benches is the
    same observation (: one valid observation is enough; no second distance is required)."""
    ok = [r for r in bench.observations if valid(r)]
    if not ok:
        return miss("no " + basis1), miss("no " + basis2)
    return (hit(ok[0]["tick"], f"{basis1} ({ok[0]['evidence_id']})"),
            hit(ok[0]["tick"], f"{basis2} ({ok[0]['evidence_id']})"))


def hand_lens_focal(run):
    """E1 a rider slid >= 1 mm or the light box switched on; E2 the light box on with lens L on A (an image of the F
    on the screen); M1 the first observation with L on A, the light on and the image sharp (blur <= 1 mm, the
    measurement-valid condition); M2 one such observation (a sharp image is enough for a result)."""
    bench = HandBench(run)

    def sharp(r):
        blur = (r.get("optics") or {}).get("blur_mm")
        return (r.get("source_on") and (r.get("seated") or {}).get("A") == "L" and blur is not None
                and blur <= MAX_BLUR_MM)

    m1, m2 = _hand_m(bench, sharp, f"observation with a sharp image (blur <= {MAX_BLUR_MM} mm)",
                     "a sharp image, enough for a result")
    return {"E1": _hand_e1(bench),
            "E2": _hand_e2(bench, "L", lambda t: True, "light box on with lens L on rider A"),
            "M1": m1, "M2": m2}


def hand_grating_wavelength(run):
    """E1 a rider slid >= 1 mm or the laser switched on; E2 the laser on with grating G on A and a first-order spot on
    the screen; M1 the first observation with G on A, the laser on and both first-order spots on the screen (the
    measurement-valid condition); M2 one such observation (both first-order spots readable is enough for a
    result)."""
    bench = HandBench(run)
    edge = M.SCREEN_M * 1000 / 2 - 3 * M.SPOT_SIGMA_M * 1000
    edge_m = M.SCREEN_M / 2 - 3 * M.SPOT_SIGMA_M
    elements = (run.device or {}).get("lenses") or {}
    laser_m = float((run.device or {}).get("laser_nm") or 0.0) * 1e-9

    def at(t):
        rows = [r["positions_mm"] for r in bench.trace if r["tick"] <= t]
        return rows[-1] if rows else bench.start

    def first_order(t):
        p, g = at(t), elements.get("G") or {}
        d = (p.get("S", 0.0) - p.get("A", 0.0) - g.get("offset_mm", 0.0)) / 1000
        if d <= 0 or not g:
            return False
        spots = dict(M.grating_spots(laser_m, g["lines_per_mm"], d, max_order=1))
        return any(m in spots and abs(spots[m]) <= edge_m for m in (-1, 1))

    def valid(r):
        spots = (r.get("optics") or {}).get("spots_mm") or {}
        return (r.get("source_on") and (r.get("seated") or {}).get("A") == "G"
                and all(o in spots and abs(spots[o]) <= edge for o in ("-1", "1")))

    m1, m2 = _hand_m(bench, valid, "observation with both first-order spots on the screen",
                     "both first-order spots readable, enough for a result")
    return {"E1": _hand_e1(bench),
            "E2": _hand_e2(bench, "G", first_order, "laser on with grating G on rider A and a first-order spot on the screen"),
            "M1": m1, "M2": m2}


def hand_plate_index(run):
    """E1 a rider slid >= 1 mm or the light box switched on; E2 the light box on with lens L on rider A; M1 the first
    observation with L on A, the block on no rider, the light on and the image sharp (blur <= 1 mm); M2 also an
    observation with the block on rider B, the lens where it stood for such a no-block observation (within 1 mm), the
    light on and the image sharp (both groups of the measurement)."""
    bench = HandBench(run)

    def sharp(r):
        blur = (r.get("optics") or {}).get("blur_mm")
        return (r.get("source_on") and (r.get("seated") or {}).get("A") == "L" and blur is not None
                and blur <= MAX_BLUR_MM)

    def no_block(r):
        return sharp(r) and "P" not in (r.get("seated") or {}).values()

    def block(r):
        return sharp(r) and (r.get("seated") or {}).get("B") == "P"

    first = next((r for r in bench.observations if no_block(r)), None)
    m1 = (hit(first["tick"], f"sharp image without the block ({first['evidence_id']})") if first
          else miss("no sharp image without the block"))
    m2 = miss("not a sharp image with the block on B at the lens position of a sharp image without it")
    for r in bench.observations:
        if block(r) and any(abs(q["positions_mm"]["A"] - r["positions_mm"]["A"]) <= HAND_LENS_MATCH_MM
                            for q in bench.observations if no_block(q) and q["tick"] < r["tick"]):
            m2 = hit(r["tick"], f"sharp image with the block on B, lens unmoved ({r['evidence_id']})")
            break
    return {"E1": _hand_e1(bench),
            "E2": _hand_e2(bench, "L", lambda t: True, "light box on with lens L on rider A"),
            "M1": m1, "M2": m2}


HAND_LENS_MATCH_MM = 1.0


def hand_optical_relay(run):
    """E1 a rider slid >= 1 mm or the light box switched on; E2 the light box on with L1 seated on rider A and L2 on
    rider B; M1 the first observation in that state; M2 an observation in that state with a sharp upright image (blur
    <= 0.25 mm, magnification > 0), the measurement-valid condition."""
    bench = HandBench(run)

    def both(r):
        seated = r.get("seated") or {}
        return r.get("source_on") and seated.get("A") == "L1" and seated.get("B") == "L2"

    def sharp_upright(r):
        o = r.get("optics") or {}
        return (both(r) and o.get("blur_mm") is not None and o["blur_mm"] <= RELAY_BLUR_MM
                and (o.get("magnification") or 0) > 0)

    first = next((r for r in bench.observations if both(r)), None)
    good = next((r for r in bench.observations if sharp_upright(r)), None)
    return {"E1": _hand_e1(bench),
            "E2": _hand_e2(bench, "L1", lambda t: bench.seated(t).get("B") == "L2",
                           "light box on with L1 on rider A and L2 on rider B"),
            "M1": (hit(first["tick"], f"observation with L1 on A and L2 on B, light on ({first['evidence_id']})")
                   if first else miss("no observation with L1 on A, L2 on B and the light on")),
            "M2": (hit(good["tick"], f"sharp upright image with both lenses ({good['evidence_id']})") if good
                   else miss(f"no sharp (blur <= {RELAY_BLUR_MM} mm) upright image with both lenses in place"))}


def hand_slit_width(run):
    """E1 a rider slid >= 1 mm or the laser switched on; E2 the laser on with slit SL on rider A and both first-order
    dark fringes on the screen; M1 the first observation with SL on A, the laser on and a dark fringe on each side of
    the central maximum (the measurement-valid condition); M2 the same observation: the -1 and +1 dark fringes readable
    on both sides are enough for a result."""
    bench = HandBench(run)
    elements = (run.device or {}).get("lenses") or {}
    laser_m = float((run.device or {}).get("laser_nm") or 0.0) * 1e-9
    edge_m = M.SCREEN_M / 2 - 0.001

    def at(t):
        rows = [r["positions_mm"] for r in bench.trace if r["tick"] <= t]
        return rows[-1] if rows else bench.start

    def first_fringes(t):
        p, slit = at(t), elements.get("SL") or {}
        d = (p.get("S", 0.0) - p.get("A", 0.0) - slit.get("offset_mm", 0.0)) / 1000
        if d <= 0 or not slit:
            return False
        fringes = dict(M.slit_dark_fringes(laser_m, slit["width_m"], d))
        return all(k in fringes and abs(fringes[k]) <= edge_m for k in (-1, 1))

    def valid(r):
        f = (r.get("optics") or {}).get("dark_fringes_mm") or {}
        return r.get("source_on") and (r.get("seated") or {}).get("A") == "SL" and "-1" in f and "1" in f

    ok = [r for r in bench.observations if valid(r)]
    return {"E1": _hand_e1(bench),
            "E2": _hand_e2(bench, "SL", first_fringes, "laser on with slit SL on rider A and both first dark fringes "
                                                         "on the screen"),
            "M1": (hit(ok[0]["tick"], f"observation with a dark fringe on each side ({ok[0]['evidence_id']})") if ok
                   else miss("no observation with a dark fringe on each side of the central maximum")),
            "M2": (hit(ok[0]["tick"], f"the -1 and +1 dark fringes readable ({ok[0]['evidence_id']})") if ok
                   else miss("no observation with the -1 and +1 dark fringes readable"))}


def _by_drive(hand, keypad):
    def rule(run):
        return (hand if (run.device or {}).get("drive") == "hand" else keypad)(run)
    rule.__doc__ = f"hand-slid bench: {hand.__doc__}\nkeypad bench: {keypad.__doc__}"
    return rule


RULES = {"lens_focal": _by_drive(hand_lens_focal, lens_focal),
         "grating_wavelength": _by_drive(hand_grating_wavelength, grating_wavelength),
         "optical_relay": _by_drive(hand_optical_relay, optical_relay),
         "slit_width": _by_drive(hand_slit_width, slit_width), "plate_index": _by_drive(hand_plate_index, plate_index)}


# --- six levels (benchmark/evaluation/process.py) ---------------------------------------------------
# The legacy E1-M2 rules above stay for the keypad bench's records (earlier/13), mapped E1->L1 ... M2->L4.
# The hand-slid bench (every run since) has its own L1-L4 detectors below, after:
#   L1 hands-on   the earliest of: a rider slid >= 1 mm from its start (rider_trace), the source switched on (state_log
#                 'power'), a holder of this bench's elements grasped and raised >= 20 mm (physics.jsonl held, xyz_mm);
#   L2 set up     the first state from which any reasonable method can measure (only the element, the riders and the
#                 source matter);
#   L3            the first observation (bench_view) in such a state;
#   L4            the observations so far hold the minimum data for the targets (any method).
# L2 is the earliest of three records of the state: the physical state rebuilt from the bench's records (state_log
# 'power'/'seats' and the 30 Hz rider_trace, its screen from the bench's own computation, optics_hand.optics), the
# bench's screen_log (each screen recomputation, records), and an observation in that state.
from benchmark.evaluation.process import LEGACY_MAP, Ladder, from_legacy  # noqa: E402,F401

SPOT_EDGE_MM = M.SCREEN_M * 1000 / 2 - 3 * M.SPOT_SIGMA_M * 1000     # a first-order spot wholly on the card (1.8 mm)


class HandScreen:
    """The hand-slid bench's states through a run, with what the screen shows in each (the bench's own optics)."""

    def __init__(self, run):
        from optics_hand import HandOpticsBench
        device = run.device or {}
        self.run, self.bench = run, HandBench(run)
        self.lenses = device.get("lenses") or {}
        self.model = HandOpticsBench.screen_model(self.lenses, device.get("source", "object"), device.get("laser_nm"))
        config = (run.private_json("configuration.json") or {}).get("bench") or {}
        self.screen_axis = config.get("screen_axis", "S")
        self.screen_log = run.logs.get("screen_log") or []

    def optics(self, on, seated, positions):
        elements = tuple(sorted((round(positions[a] + self.lenses[n].get("offset_mm", 0.0), 4), n)
                                for a, n in seated.items() if n and a in positions and n in self.lenses))
        return self.model.optics((bool(on), elements, round(positions[self.screen_axis], 4)))

    def rebuilt(self):
        """(tick, source on, seats, rider positions) after each power or seat event and each rider trace row."""
        events = sorted([(e["tick"], 0, e) for e in self.bench.log if e.get("event") in ("power", "seats")] +
                        [(r["tick"], 1, r) for r in self.bench.trace], key=lambda x: (x[0], x[1]))
        on, seated, positions = False, {}, dict(self.bench.start)
        for k, (tick, kind, e) in enumerate(events):
            if kind == 1:
                positions = dict(positions, **{a: float(p) for a, p in e["positions_mm"].items()})
            elif e["event"] == "power":
                on = bool(e["on"])
            else:
                seated = dict(e["seated"])
            if k + 1 == len(events) or events[k + 1][0] != tick:
                yield tick, on, seated, positions

    def first_state(self, pre, ok, basis):
        """L2: the earliest tick at which the source and seats pass pre(on, seated) and the screen passes ok(optics),
        over the rebuilt states, the screen log and the observations."""
        found = []
        for tick, on, seated, positions in self.rebuilt():
            if self.screen_axis in positions and pre(on, seated) and ok(self.optics(on, seated, positions)):
                found.append((tick, f"{basis} (rebuilt from the bench's records)"))
                break
        for row in self.screen_log:
            if pre(row["source_on"], self.bench.seated(row["tick"])) and ok(row.get("optics") or {}):
                found.append((row["tick"], f"{basis} (screen log)"))
                break
        obs = next((r for r in self.bench.observations
                    if pre(r.get("source_on"), r.get("seated") or {}) and ok(r.get("optics") or {})), None)
        if obs is not None:
            found.append((obs["tick"], f"{basis} ({obs['evidence_id']})"))
        if not found:
            return miss("never " + basis)
        return hit(*min(found))


def hand_l1(run):
    """L1: the earliest of a rider slid >= 1 mm, the source switched on, a holder of the bench's elements grasped and
    raised >= 20 mm."""
    bench = HandBench(run)
    found = [x for x in (bench.first_moved(), bench.first_power()) if x is not None]
    lift = lifted(run, set((run.device or {}).get("lenses") or {}))
    if lift.reached:
        found.append((lift.tick, lift.basis))
    if not found:
        return miss("no rider slid, the source never switched on and no holder was raised")
    return hit(*min(found))


def _first_observation(screen, test, basis):
    r = next((r for r in screen.bench.observations if test(r)), None)
    return hit(r["tick"], f"{basis} ({r['evidence_id']})") if r else miss("no " + basis)


def _single_element_levels(run, element, ok, state_basis, obs_basis):
    """lens_focal, grating_wavelength, slit_width: L2 the first state with the source on, `element` on rider A and the screen passing ok(optics);
    L3 the first observation in that state; L4 the same observation (one valid observation determines the target,
)."""
    screen = HandScreen(run)
    pre = lambda on, seated: bool(on) and seated.get("A") == element  # noqa: E731
    l3 = _first_observation(screen, lambda r: pre(r.get("source_on"), r.get("seated") or {}) and ok(r.get("optics") or {}),
                            obs_basis)
    l4 = hit(l3.tick, "one valid observation is enough: " + l3.basis) if l3.reached else miss(l3.basis)
    return {"L1": hand_l1(run), "L2": screen.first_state(pre, ok, state_basis), "L3": l3, "L4": l4}


def six_lens_focal(run):
    """lens_focal: S = light box on, lens L on A, the F image sharp (blur <= 1 mm). L2 first S; L3 the
    first observation in S; L4 the same (one sharp image gives a (u, v) pair)."""
    def sharp(o):
        return o.get("blur_mm") is not None and o["blur_mm"] <= MAX_BLUR_MM
    return _single_element_levels(run, "L", sharp, f"light on, lens L on A, image sharp (blur <= {MAX_BLUR_MM} mm)",
                                  "observation with a sharp image")


def six_grating_wavelength(run):
    """grating_wavelength: S = laser on, grating G on A, both first-order spots wholly on the screen
    (centres >= 1.8 mm inside the edge). L2 first S; L3 the first observation in S; L4 the same."""
    def both(o):
        spots = o.get("spots_mm") or {}
        return all(k in spots and abs(spots[k]) <= SPOT_EDGE_MM for k in ("-1", "1"))
    return _single_element_levels(run, "G", both, "laser on, grating G on A, both first-order spots on the screen",
                                  "observation with both first-order spots on the screen")


def six_slit_width(run):
    """slit_width: S = laser on, slit SL on A, the -1 and +1 dark fringes showing on the screen (the
    bench's own rule: >= 1 mm inside the edge, the side lobe beyond lit >= 5 %). L2 first S; L3 the first observation
    in S; L4 the same (the +-1 fringes at one distance give a)."""
    def both(o):
        f = o.get("dark_fringes_mm") or {}
        return "-1" in f and "1" in f
    return _single_element_levels(run, "SL", both, "laser on, slit SL on A, the -1 and +1 dark fringes on the screen",
                                  "observation with the -1 and +1 dark fringes on the screen")


def six_optical_relay(run):
    """optical_relay: L2 the light box on with at least one lens seated on rider A or B; L3 the first
    observation with (i) the light on, one lens alone on a rider and its image sharp (blur <= 1 mm: a single-lens focal
    length reading), or (ii) the light on, L1 on A and L2 on B; L4 an observation with L1 on A, L2 on B, the light on
    and a sharp (blur <= 0.25 mm) upright (m > 0) image, the target setup itself."""
    screen = HandScreen(run)
    lenses = {"L1", "L2"}

    def on_riders(seated):
        return [n for a, n in seated.items() if a in ("A", "B") and n in lenses]

    pre = lambda on, seated: bool(on) and bool(on_riders(seated))  # noqa: E731
    both = lambda r: (r.get("source_on") and (r.get("seated") or {}).get("A") == "L1"  # noqa: E731
                      and (r.get("seated") or {}).get("B") == "L2")

    def single_sharp(r):
        blur = (r.get("optics") or {}).get("blur_mm")
        return (r.get("source_on") and len(on_riders(r.get("seated") or {})) == 1 and blur is not None
                and blur <= MAX_BLUR_MM)

    def target(r):
        o = r.get("optics") or {}
        return (both(r) and o.get("blur_mm") is not None and o["blur_mm"] <= RELAY_BLUR_MM
                and (o.get("magnification") or 0) > 0)

    return {"L1": hand_l1(run),
            "L2": screen.first_state(pre, lambda o: True, "light box on with a lens seated on rider A or B"),
            "L3": _first_observation(screen, lambda r: single_sharp(r) or both(r),
                                     "observation of one lens' sharp image, or with L1 on A and L2 on B, light on"),
            "L4": _first_observation(screen, target, f"observation with L1 on A, L2 on B and a sharp (blur <= "
                                                     f"{RELAY_BLUR_MM} mm) upright image")}


PLATE_SAME_LENS_MM = 1.0       # L4 (i): a pair at one lens position
PLATE_TWO_LENS_MM = 2.0        # L4 (ii): two lens positions without the block determine f and the lens offset


def six_plate_index(run):
    """plate_index: V = an observation with the light on, lens L on A and the image sharp (blur <= 1
    mm), without the block (on no rider) or with it on rider B. L2 the light on with L on A; L3 the first V, either
    kind; L4 (in any order) (i) a V without and a V with the block, the lens within 1 mm, or (ii) two V without the
    block with the lens >= 2 mm apart and a V with the block anywhere."""
    screen = HandScreen(run)
    plate = next((k for k, v in screen.lenses.items() if v.get("kind") == "plate"), "P")

    def sharp(r):
        blur = (r.get("optics") or {}).get("blur_mm")
        return (r.get("source_on") and (r.get("seated") or {}).get("A") == "L" and blur is not None
                and blur <= MAX_BLUR_MM)

    def kind(r):
        seated = r.get("seated") or {}
        if not sharp(r):
            return None
        if plate not in seated.values():
            return "without"
        return "with" if seated.get("B") == plate else None

    l4, seen = miss("no sharp images with and without the block that determine the shift"), []
    for r in screen.bench.observations:
        k = kind(r)
        if k is None:
            continue
        seen.append((k, r["positions_mm"]["A"]))
        without = [a for g, a in seen if g == "without"]
        with_ = [a for g, a in seen if g == "with"]
        if any(abs(a - b) <= PLATE_SAME_LENS_MM for a in without for b in with_):
            l4 = hit(r["tick"], f"sharp images without and with the block at one lens position ({r['evidence_id']})")
            break
        if with_ and span(without) >= PLATE_TWO_LENS_MM:
            l4 = hit(r["tick"], f"two lens positions without the block and one with it ({r['evidence_id']})")
            break
    return {"L1": hand_l1(run),
            "L2": screen.first_state(lambda on, seated: bool(on) and seated.get("A") == "L", lambda o: True,
                                     "light box on with lens L on rider A"),
            "L3": _first_observation(screen, lambda r: kind(r) is not None,
                                     f"sharp image (blur <= {MAX_BLUR_MM} mm), with or without the block"),
            "L4": l4}


def _six(exp_id, hand):
    legacy = RULES[exp_id]

    def detect(run):
        if (run.device or {}).get("drive") != "hand":            # keypad bench records: the legacy rule, mapped
            return {LEGACY_MAP[k]: v for k, v in legacy(run).items() if k in LEGACY_MAP}
        return hand(run)
    detect.__doc__ = hand.__doc__
    return Ladder(detect=detect, note="six-level detectors")


LADDERS = {"lens_focal": _six("lens_focal", six_lens_focal),
           "grating_wavelength": _six("grating_wavelength", six_grating_wavelength),
           "optical_relay": _six("optical_relay", six_optical_relay),
           "slit_width": _six("slit_width", six_slit_width),
           "plate_index": _six("plate_index", six_plate_index)}

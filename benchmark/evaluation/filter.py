"""active_filter gates and score.

The circuit F is the board at submission (the sealed device record: every seated block and the nodes its pins are on;
the ground rail's pads joined by the printed copper; the analyzer's source at the IN pad, its ground on the rail; the
probe where it sits), solved with the instance's true part values (filter_model.draw(model_seed)).

- filter_sweep_genuine: sweep_evidence cites a sweep of this episode (kind filter_sweep, its public record intact) that
  completed, covered the whole passband and the whole stopband (START <= 100 Hz, STOP >= 20 kHz), and ran on the board
  as submitted (no seat or unseat on the board after it started, the same board signature: the sealed transitions and
  the sweep's private start record); its probe (the pads in the sweep's private record, not in the public one) has its
  - pin on the ground node, its + pin on neither the ground node nor the IN node.
- filter_circuit_as_stated: the submitted circuit (block, pads) is the board at submission: every seated block, nothing
  else; two-pin parts by their two pads, the probe as (+, -), op-amp modules as (IN+, IN-, OUT). The two copies of one
  value are one part and every wire is a wire (they look alike), so either id of a pair names the part.
- filter_passband: F's passband ripple (the largest gain deviation over 100 Hz - f_p from the gain at 100 Hz) is at
  most ripple_db.
- filter_bench_tidy: at submission no sweep runs, the robot holds nothing, and every block not on the board is in a
  tray well of its own kind.

Score (metric filter_design): a = F's stopband attenuation, the gain at 100 Hz minus the highest gain over the stopband
f_s = 2 kHz to 20 kHz (filter_model.evaluate). Two reference values:
- A_ref (the success line): the best a of the standard design, two unity-gain Sallen-Key stages of the kit's pieces,
  either module order (filter_model.optimum: a deterministic enumeration with branch and bound, scored on the
  stopband). e = max(max(0, A_ref - a) / threshold, |stopband_attenuation_db - a| / reported_threshold); success
  when e <= 1.
- A_search (the ranking reference): the best a found by a search over every circuit the kit's pieces, wires and modules
  can build on the board (benchmark.evaluation.filter_search, with its fixed default settings). The attainment is
  a - A_search. It never enters e or success.

Both depend on model_seed only. They are looked up in an optional table experiments/active_filter/astar_table.json,
then in a cache file (EMBODIEDLAB_FILTER_ASTAR_CACHE, default ~/.cache/embodiedlab/filter_astar.json), both keyed by
(MODEL_VERSION, model_seed); otherwise they are computed and cached: A_ref in about ten seconds, the search in about
30 minutes on 48 cores (longer on fewer; the result does not depend on the number of cores). To compute them ahead of
scoring:

    python -m benchmark.evaluation.filter prewarm <instance.json> ...
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from benchmark.core import filter_model as M

TABLE = Path(__file__).resolve().parents[2] / "experiments" / "active_filter" / "astar_table.json"
COLS, ROWS = 7, 6
IN_NODE, GROUND_NODE = "1,3", "1,6"
RAIL = [(f"{c},{ROWS}", f"{c + 1},{ROWS}") for c in range(1, COLS)]


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _device(ctx):
    return ctx.sealed.episode.get("device") or {}


def _state(ctx):
    return ctx.sealed.episode.get("terminal_state") or {}


def kind_of(block):
    if block in ("OA1", "OA2"):
        return "opamp"
    if block == "PROBE":
        return "probe"
    return {"R": "resistor", "C": "capacitor", "W": "wire"}.get(block[:1])


# --- the circuit at submission -----------------------------------------------------------------------------------
def board_of(ctx):
    board = _device(ctx).get("board")
    return {b: list(n) for b, n in board.items()} if isinstance(board, dict) else None


def circuit(hidden, board):
    """filter_model.Circuit of a board {block: [node names]} with the instance's true values."""
    inst = M.hidden_of(hidden)
    values = M.piece_values(inst)
    amps = {"OA1": inst["amps"][0], "OA2": inst["amps"][1]}
    els = [("W", a, b) for a, b in RAIL]
    probe = None
    for block in M.piece_ids() + [f"W{k}" for k in range(1, 9)] + ["PROBE", "OA1", "OA2"]:
        if block not in board:
            continue
        n = board[block]
        kind = kind_of(block)
        if kind in ("resistor", "capacitor"):
            els.append(("R" if kind == "resistor" else "C", n[0], n[1], values[block]))
        elif kind == "wire":
            els.append(("W", n[0], n[1]))
        elif kind == "opamp":
            els.append(("OA", n[0], n[1], n[2], amps[block]))
        elif kind == "probe":
            probe = (n[0], n[1])
    return M.Circuit(els, GROUND_NODE, source=IN_NODE, probe=probe)


def response_of(hidden, board):
    """(ripple dB, stopband attenuation dB) of the board's circuit, from gains as the analyzer shows them."""
    rip, att = M.evaluate(circuit(hidden, board))
    return float(rip), float(att)


def no_response(hidden, board):
    """The probe shows nothing in the passband (gain at 100 Hz at the analyzer's floor, e.g. the probe not seated)."""
    return M.silent(circuit(hidden, board).response(M.FREQS[:1]))


def nets(board):
    """{node: net root} with the rail copper and the seated wires joining nodes."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in RAIL + [tuple(n) for blk, n in board.items() if kind_of(blk) == "wire"]:
        parent[find(a)] = find(b)
    return find


# --- A* ------------------------------------------------------------------------------------------------------------
def _cache_path():
    return Path(os.environ.get("EMBODIEDLAB_FILTER_ASTAR_CACHE")
                or Path.home() / ".cache" / "embodiedlab" / "filter_astar.json")


def _cache_key(hidden):
    return _sha({"model": M.MODEL_VERSION, "model_seed": int(hidden["model_seed"])})


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _cache_write(key, value):
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _read(path)
        data[key] = value
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".filter_astar.")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=0, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        pass                              # an unwritable cache only costs the computation again


def solve(hidden):
    """The reference entry of an instance: A_ref, the best two-stage unity-gain Sallen-Key design (either module order),
    and A_search, the board search's best layout (filter_search with its default settings)."""
    from benchmark.evaluation import filter_search as FS
    seed = int(hidden["model_seed"])
    print(f"active_filter: no stored reference values for model_seed {seed}; running the board search (about 30 min on "
          f"48 cores, longer on fewer). `python -m benchmark.evaluation.filter prewarm <instance.json>` runs it ahead "
          f"of scoring.", file=sys.stderr, flush=True)
    return FS.reference(seed)


def optimum(hidden):
    """The reference entry of an instance: {'A_ref_db', 'ref_ripple_db', 'ref_design' (the success line's standard
    design), 'A_search_db', 'search_ripple_db', 'layout' (the search's best board), 'search', ..., 'source'} (table,
    cache or computed)."""
    key = _cache_key(hidden)
    for source, table in (("table", _read(TABLE)), ("cache", _read(_cache_path()))):
        if table.get(key, {}).get("A_search_db") is not None:
            return dict(table[key], source=source)
    q = solve(hidden)
    _cache_write(key, q)
    return dict(q, source="computed")


# --- gates ---------------------------------------------------------------------------------------------------------
def sweep_record(ctx, evidence_id):
    if not isinstance(evidence_id, str) or not evidence_id.startswith(ctx.sealed.episode_id + ":sweep:"):
        return None, f"{evidence_id}: not a sweep of this episode"
    record = ctx.sealed.evidence.get(evidence_id)
    if not isinstance(record, dict) or record.get("kind") != "filter_sweep":
        return None, f"{evidence_id}: unknown sweep"
    public = record.get("public")
    if not isinstance(public, dict) or _sha(public) != record.get("sha256"):
        return None, f"{evidence_id}: payload hash mismatch"
    return record, None


def filter_sweep_genuine(ctx):
    sub = ctx.submission if ctx.submission_valid else None
    if sub is None:
        return False, {"reason": "submission is not valid; gate cannot be assessed"}
    eid = sub["sweep_evidence"]
    record, problem = sweep_record(ctx, eid)
    if problem:
        return False, {"problems": [problem]}
    public = record["public"]
    problems = []
    if public.get("status") != "complete":
        problems.append(f"{eid}: the sweep did not complete ({public.get('status')})")
    if not (float(public.get("start_hz", 1e9)) <= M.F_LO and float(public.get("stop_hz", 0)) >= M.FS_HI):
        problems.append(f"{eid}: the sweep does not cover {M.F_LO:g} Hz to {M.FS_HI:g} Hz")
    device = _device(ctx)
    started = float(public.get("started_s", float("inf")))
    later = [t for t in device.get("transitions") or [] if float(t["t_s"]) > started]
    if later or record.get("signature") != device.get("signature"):
        problems.append(f"{eid}: the board was changed after the sweep started")
    board = board_of(ctx) or {}
    probe = record.get("probe")                 # sealed private record: the probe's pads when the sweep started
    if not isinstance(probe, dict):
        problems.append(f"{eid}: the probe was not on the board")
    else:
        find = nets(board)
        if find(probe["minus"]) != find(GROUND_NODE):
            problems.append(f"{eid}: the probe's - pin is not on ground")
        if find(probe["plus"]) in (find(GROUND_NODE), find(IN_NODE)):
            problems.append(f"{eid}: the probe's + pin is on ground or on the analyzer output")
    return not problems, {"problems": problems, "transitions_after": later, "probe": probe,
                          "sweep": {k: public.get(k) for k in ("sweep", "status", "started_s", "ended_s", "start_hz",
                                                               "stop_hz", "points")}}


def part_of(block):
    """What a block is as far as the circuit goes: the two copies of a value are one part (R4k7), every wire is W."""
    kind = kind_of(block)
    if kind in ("resistor", "capacitor"):
        return block.rsplit("_", 1)[0]
    return "W" if kind == "wire" else block


def _canonical(block, pads):
    kind = kind_of(block)
    return (part_of(block), tuple(pads) if kind in ("probe", "opamp") else tuple(sorted(pads)))


def filter_circuit_as_stated(ctx):
    sub = ctx.submission if ctx.submission_valid else None
    if sub is None:
        return False, {"reason": "submission is not valid; gate cannot be assessed"}
    board = board_of(ctx)
    if board is None:
        return False, {"problems": ["no record of the board at submission"]}
    from collections import Counter
    stated = Counter(_canonical(c["block"], c["pads"]) for c in sub["circuit"])
    actual = Counter(_canonical(b, n) for b, n in board.items())
    problems = []
    if len({c["block"] for c in sub["circuit"]}) != len(sub["circuit"]):
        problems.append("a block is stated twice")
    missing = sorted((actual - stated).elements())
    extra = sorted((stated - actual).elements())
    if missing:
        problems.append("seated blocks not stated as they sit: " + ", ".join(b for b, _ in missing))
    if extra:
        problems.append("stated blocks not on the board as stated: " + ", ".join(b for b, _ in extra))
    return not problems, {"problems": problems, "actual": [[b, list(p)] for b, p in sorted(actual.elements())]}


def filter_passband(ctx, ripple_db=0.5):
    board = board_of(ctx)
    if board is None:
        return False, {"problems": ["no record of the board at submission"]}
    rip, att = response_of(ctx.hidden, board)
    limit = float(ctx.param(ripple_db))
    if no_response(ctx.hidden, board):
        return False, {"problems": [f"no signal at the probe: the gain at {M.F_LO:g} Hz reads at the analyzer's floor "
                                    f"({M.FLOOR_DB:g} dB)"], "no_response": True, "limit_db": limit}
    return rip <= limit, {"ripple_db": rip, "attenuation_db": att, "limit_db": limit}


def filter_bench_tidy(ctx):
    state = _state(ctx)
    placement = state.get("placement")
    if not isinstance(placement, dict) or not placement:
        return False, {"problems": ["no record of the bench at submission"]}
    problems = []
    if state.get("sweep_running"):
        problems.append("a sweep is running")
    if state.get("held") is not None:
        problems.append("the robot holds something")
    for block, place in sorted(placement.items()):
        if place[0] == "board":
            continue
        if not (place[0] == "tray" and place[2]):
            problems.append(f"{block} is neither on the board nor in a tray well of its kind")
    return not problems, {"problems": problems}


# --- score -------------------------------------------------------------------------------------------------------
def filter_score(spec, hidden, submission, ctx):
    threshold, reported = float(spec.scoring["threshold"]), float(spec.scoring["reported_threshold"])
    best = optimum(hidden)
    A_ref = float(best["A_ref_db"])
    A_search = None if best.get("A_search_db") is None else float(best["A_search_db"])
    board = board_of(ctx) or {}
    rip, a = response_of(hidden, board)
    dead = no_response(hidden, board)               # a dead circuit reads flat at the floor: a = 0, e = A_ref / threshold
    r = float(submission["stopband_attenuation_db"])
    shortfall = max(0.0, A_ref - a) / threshold
    prediction = abs(r - a) / reported
    e = max(shortfall, prediction)
    quantities = [
        {"name": "attenuation_db", "submitted": a, "truth": A_ref, "abs_error": A_ref - a, "rel_error": None,
         "metric": "shortfall_db", "threshold": threshold, "normalized_error": shortfall},
        {"name": "stopband_attenuation_db", "submitted": r, "truth": a, "abs_error": abs(r - a), "rel_error": None,
         "metric": "abs_error_db", "threshold": reported, "normalized_error": prediction}]
    return {"relative_errors": None, "e_rel": None, "e": e, "quantities": quantities,
            "attainment_reference": A_search,          # the attainment is a - A_search (None without it); not in e
            "diagnostics": {"ripple_db": rip, "attenuation_db": a, "no_response": dead, "A_ref_db": A_ref,
                            "A_search_db": A_search,
                            "reported_ripple_db": submission.get("passband_ripple_db"),
                            "reference": {k: best.get(k) for k in ("ref_ripple_db", "ref_design", "search_ripple_db",
                                                                   "layout", "search", "source")}}}


# --- cache -------------------------------------------------------------------------------------------------------
def _prewarm(paths):
    for path in paths:
        hidden = json.loads(Path(path).read_text())["hidden"]
        q = optimum(hidden)
        print(path, round(q["A_ref_db"], 6), round(q["A_search_db"], 6), q["source"], flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "prewarm":
        _prewarm(sys.argv[2:])
    else:
        sys.exit("usage: python -m benchmark.evaluation.filter prewarm <instance.json> ...")

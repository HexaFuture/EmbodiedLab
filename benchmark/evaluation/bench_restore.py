"""Bench restore: which loose items are back where they stood when the episode started. Descriptive only.

It is not a gate and enters neither G, the reward nor the process score, and the task cards do not ask for it; it
shows whether an agent puts things back of its own accord. Input is a bench_order record with `items` (the bench at
the end) and `start_items` (before the first tool call), as the sealed episode.json carries it at submission and the
process record (bench_logs.json, bench_order: start_items and at_stop) at stop. An item is home when it is within
HOME_DXY_M horizontally and HOME_DZ_M vertically of its start position, its start up axis is within HOME_TILT_DEG of
vertical, the robot does not hold it and it is not off the bench (OFF_BENCH_M below the table top, as in the
bench_in_order gate). Experiments without loose items give n_items = 0 and all_home = None (not applicable).
"""
from __future__ import annotations

import math

HOME_DXY_M = 0.02
HOME_DZ_M = 0.02
HOME_TILT_DEG = 10.0
OFF_BENCH_M = 0.10
THRESHOLDS = {"dxy_m": HOME_DXY_M, "abs_dz_m": HOME_DZ_M, "tilt_deg": HOME_TILT_DEG, "off_bench_below_table_m": OFF_BENCH_M}


def bench_restore(record, items=None) -> dict:
    """{n_items, n_home, all_home, items: {path: {dxy_m, dz_m, tilt_deg, held, off_bench, home}}, thresholds}.

    `record` is a bench_order record; `items` overrides record['items'] (the process record keeps the end state
    under at_stop). Without a start record every count is None and `error` says why."""
    out = {"n_items": None, "n_home": None, "all_home": None, "items": {}, "thresholds": dict(THRESHOLDS)}
    if not isinstance(record, dict):
        return dict(out, error="no bench_order record")
    start, end = record.get("start_items"), items if items is not None else record.get("items")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return dict(out, error=record.get("error") or "no start or end snapshot of the bench")
    for path in sorted(start):
        if path not in end:
            continue
        a, b = start[path], end[path]
        dxy = math.hypot(b["xyz_m"][0] - a["xyz_m"][0], b["xyz_m"][1] - a["xyz_m"][1])
        dz = b["xyz_m"][2] - a["xyz_m"][2]
        off = b["below_table_top_m"] > OFF_BENCH_M
        held = bool(b["held"])
        home = dxy <= HOME_DXY_M and abs(dz) <= HOME_DZ_M and b["tilt_deg"] <= HOME_TILT_DEG and not held and not off
        out["items"][path] = {"dxy_m": round(dxy, 4), "dz_m": round(dz, 4), "tilt_deg": b["tilt_deg"],
                              "held": held, "off_bench": off, "home": home}
    n = len(out["items"])
    n_home = sum(item["home"] for item in out["items"].values())
    return dict(out, n_items=n, n_home=n_home, all_home=(n_home == n) if n else None)


def from_process_logs(logs) -> dict:
    """bench_restore at stop, from the process record's bench_logs (for runs that did not submit)."""
    record = (logs or {}).get("bench_order")
    if not isinstance(record, dict):
        return bench_restore(None)
    at_stop = record.get("at_stop") or {}
    return bench_restore(record, items=at_stop.get("items") if isinstance(at_stop, dict) else None)

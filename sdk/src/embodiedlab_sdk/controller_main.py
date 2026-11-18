"""Controller process entry point."""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import os
import threading
import time

import uvicorn
from uvicorn.config import LOGGING_CONFIG

from .controller import create_app
from .external_services import load_external_services
from .object_store import object_store_from_env
from .security import api_keys_from_env
from .store import store_from_url

LOGGER = logging.getLogger("embodiedlab_sdk.controller")
# Full collections that pause the Controller at least this long are logged.
GC_PAUSE_LOG_SECONDS = 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embodiedlab-controller")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8130)
    parser.add_argument("--database-url", default=os.getenv("EMBODIEDLAB_DATABASE_URL"))
    parser.add_argument("--dev-memory", action="store_true")
    parser.add_argument("--artifact-dir")
    parser.add_argument(
        "--external-services-file", default=os.getenv("EMBODIEDLAB_EXTERNAL_SERVICES_FILE")
    )
    parser.add_argument("--external-services-only", action="store_true")
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--worker-stale-after",
        type=float,
        default=float(os.getenv("EMBODIEDLAB_WORKER_STALE_AFTER", "30")),
        help="mark Workers offline after this many seconds without a heartbeat",
    )
    parser.add_argument(
        "--gc-freeze-interval",
        type=float,
        default=float(os.getenv("EMBODIEDLAB_GC_FREEZE_INTERVAL", "60")),
        help="move surviving objects out of the garbage collector's reach every this many "
        "seconds (0 disables), so full collections do not walk the whole retained store",
    )
    return parser


def log_config() -> dict:
    """uvicorn's logging with a UTC timestamp on every line, access lines included."""

    config = copy.deepcopy(LOGGING_CONFIG)
    stamp = "%(asctime)s.%(msecs)03dZ "
    for formatter in config["formatters"].values():
        formatter["fmt"] = stamp + formatter["fmt"]
        formatter["datefmt"] = "%Y-%m-%dT%H:%M:%S"
    config["loggers"]["embodiedlab_sdk"] = {
        "handlers": ["default"],
        "level": "INFO",
        "propagate": False,
    }
    return config


def _log_long_gc_pauses() -> None:
    started: dict[int, float] = {}

    def callback(phase: str, info: dict) -> None:
        generation = info.get("generation")
        if generation != 2:
            return
        if phase == "start":
            started[2] = time.perf_counter()
            return
        began = started.pop(2, None)
        if began is None:
            return
        pause = time.perf_counter() - began
        if pause >= GC_PAUSE_LOG_SECONDS:
            LOGGER.warning(
                "full garbage collection paused the Controller %.2fs (collected=%s, frozen=%d)",
                pause,
                info.get("collected"),
                gc.get_freeze_count(),
            )

    gc.callbacks.append(callback)


def start_gc_freezer(
    interval: float, stop: threading.Event | None = None
) -> threading.Thread | None:
    """Freeze the long-lived heap periodically.

    The dev-memory store keeps every Operation as nested Python objects; a full collection walks all
    of them while holding the GIL, which stalls every request. Frozen objects are still freed by
    reference counting; only cycles among them are no longer collected.
    """

    if interval <= 0:
        return None

    stopped = stop or threading.Event()

    def run() -> None:
        while not stopped.wait(interval):
            gc.freeze()

    thread = threading.Thread(target=run, name="gc-freezer", daemon=True)
    thread.start()
    return thread


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.Formatter.converter = time.gmtime
    store = store_from_url(args.database_url, dev_memory=args.dev_memory)
    external_services = (
        load_external_services(args.external_services_file) if args.external_services_file else []
    )
    app = create_app(
        store,
        object_store_from_env(local_root=args.artifact_dir),
        api_keys_from_env(),
        external_services,
        managed_services_enabled=not args.external_services_only,
        worker_stale_after_seconds=args.worker_stale_after,
    )
    _log_long_gc_pauses()
    gc.freeze()
    start_gc_freezer(args.gc_freeze_interval)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        log_config=log_config(),
    )


if __name__ == "__main__":
    main()

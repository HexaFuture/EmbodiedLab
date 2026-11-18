"""Container-local client used by Workers to call the persistent environment daemon."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import httpx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embodiedlab-env-client")
    parser.add_argument(
        "command",
        choices=(
            "ready",
            "exec",
            "run-workflow",
            "observe",
            "states",
            "verify",
            "evaluate",
            "truth",
            "restore",
            "tools",
            "tool",
            "capture-status",
            "capture",
            "tool-attach",
            "tool-detach",
            "tool-visibility",
        ),
    )
    parser.add_argument("value", nargs="?")
    parser.add_argument("--filename", default="inline.py")
    parser.add_argument("--workflow-dir")
    parser.add_argument("--task")
    parser.add_argument("--name")
    parser.add_argument("--url", default=os.getenv("EMBODIEDLAB_ENVD_URL", "http://127.0.0.1:8765"))
    args = parser.parse_args(argv)
    timeout = float(os.getenv("EMBODIEDLAB_ENVD_TIMEOUT", "86400"))
    with httpx.Client(base_url=args.url, timeout=timeout) as client:
        if args.command == "ready":
            response = client.get("/ready")
        elif args.command == "exec":
            response = client.post(
                "/exec", json={"code": sys.stdin.read(), "filename": args.filename}
            )
        elif args.command == "run-workflow":
            response = client.post(
                "/workflow/run",
                json={
                    "workflow_dir": args.workflow_dir,
                    "task": args.task,
                    "parameters": json.loads(sys.stdin.read() or "{}"),
                },
            )
        elif args.command == "observe":
            response = client.get("/observe")
        elif args.command == "states":
            response = client.get("/states")
        elif args.command == "verify":
            response = client.get("/verify")
        elif args.command == "evaluate":
            response = client.get("/evaluate")
        elif args.command == "truth":
            payload = sys.stdin.read()
            if payload.strip():
                response = client.post("/truth", json=json.loads(payload))
            else:
                response = client.get("/truth")
        elif args.command == "tools":
            response = client.get("/tools")
        elif args.command == "capture-status":
            response = client.get("/capture/status")
        elif args.command == "capture":
            response = client.post("/capture/finalize")
        elif args.command == "tool":
            response = client.post(
                "/tools/call",
                json={"name": args.name, "arguments": json.loads(sys.stdin.read() or "{}")},
            )
        elif args.command == "tool-attach":
            response = client.post("/tools/attach", json=json.loads(sys.stdin.read()))
        elif args.command == "tool-detach":
            response = client.post("/tools/detach", json=json.loads(sys.stdin.read() or "{}"))
        elif args.command == "tool-visibility":
            response = client.post("/tools/visibility", json=json.loads(sys.stdin.read()))
        else:
            response = client.post("/restore", json={"state_id": args.value})
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            # envd rejections (422/404) carry a machine-readable ``detail``
            # (code/message/filename/line).  A traceback here would reduce that
            # to an OCI exec failure string, so re-emit the body on the same
            # ``{"ok": false, "error": {...}}`` channel the callers already
            # parse, keeping the exit-code convention below.
            print(
                json.dumps(
                    {"ok": False, "error": _structured_error(response)},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 2
        value: dict[str, Any] = response.json()
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return 0 if value.get("ok", True) else 2


def _structured_error(response: httpx.Response) -> dict[str, Any]:
    """The error body of a non-2xx envd response, as a dict in every case."""

    try:
        body = response.json()
    except ValueError:
        return {"message": response.text, "status_code": response.status_code}
    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, dict):
        return detail
    return {"message": str(detail), "status_code": response.status_code}


if __name__ == "__main__":
    raise SystemExit(main())

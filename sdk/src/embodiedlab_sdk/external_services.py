"""Discovery and health checks for model services managed outside EmbodiedLab SDK."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .catalog import service_spec
from .models import utcnow
from .store import Store


def load_external_services(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate an external-service manifest."""

    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("external service manifest must be a JSON array")

    services: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"external service entry {index} must be an object")
        if not isinstance(raw.get("name"), str) or not raw["name"]:
            raise ValueError(f"external service entry {index} requires name")
        spec = service_spec(raw["name"])
        name = spec["name"]
        if name in names:
            raise ValueError(f"duplicate external service {name}")

        endpoint = raw.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError(f"external service {name} requires endpoint")
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https", "tcp", "zmq"}:
            raise ValueError(f"external service {name} has unsupported endpoint scheme")
        if not parsed.hostname or parsed.port is None:
            raise ValueError(f"external service {name} endpoint requires host and port")

        health_path = raw.get("health_path")
        if health_path is not None:
            if parsed.scheme not in {"http", "https"}:
                raise ValueError(f"external service {name} health_path requires HTTP")
            if not isinstance(health_path, str) or not health_path.startswith("/"):
                raise ValueError(f"external service {name} health_path must start with /")

        timeout = raw.get("timeout", 1.0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"external service {name} timeout must be positive")

        names.add(name)
        services.append(
            {
                **raw,
                "name": name,
                "endpoint": endpoint.rstrip("/"),
                "protocol": raw.get("protocol", spec["protocol"]),
                "timeout": float(timeout),
            }
        )
    return services


def probe_external_service(service: dict[str, Any]) -> dict[str, Any]:
    """Perform a bounded HTTP or TCP readiness check."""

    endpoint = service["endpoint"]
    parsed = urlparse(endpoint)
    timeout = float(service.get("timeout", 1.0))
    checked_at = utcnow().isoformat()
    try:
        health_path = service.get("health_path")
        if health_path:
            request = Request(f"{endpoint}{health_path}", method="GET")
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                response.read(1)
                status_code = response.status
            return {
                "ready": 200 <= status_code < 300,
                "status_code": status_code,
                "checked_at": checked_at,
            }

        with socket.create_connection((parsed.hostname, parsed.port), timeout=timeout):
            pass
        return {"ready": True, "checked_at": checked_at}
    except HTTPError as exc:
        return {
            "ready": False,
            "status_code": exc.code,
            "error": str(exc),
            "checked_at": checked_at,
        }
    except (OSError, URLError, TimeoutError) as exc:
        return {"ready": False, "error": str(exc), "checked_at": checked_at}


class ExternalServiceRegistry:
    """Materialize externally managed endpoints in the Controller store."""

    def __init__(self, store: Store, services: list[dict[str, Any]]) -> None:
        self.store = store
        self.services = services
        self.names = {item["name"] for item in services}

    def refresh(self) -> None:
        for service in self.services:
            health = probe_external_service(service)
            ready = bool(health["ready"])
            name = service["name"]
            self.store.put_resource(
                "service",
                name,
                {
                    "id": name,
                    "name": name,
                    "project_id": service.get("project_id", "default"),
                    "source": "external",
                    "managed": False,
                    "replicas": 1,
                    "available_replicas": 1 if ready else 0,
                    "instances": [
                        {
                            "id": f"external_{name}_0",
                            "endpoint": service["endpoint"],
                            "protocol": service["protocol"],
                            "state": "ready" if ready else "unavailable",
                        }
                    ],
                    "state": "running" if ready else "unavailable",
                    "health": health,
                    "updated_at": health["checked_at"],
                },
            )

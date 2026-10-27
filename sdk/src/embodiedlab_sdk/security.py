"""Trusted-intranet API-key identities for project and operator separation."""

from __future__ import annotations

import json
import os
from typing import Literal

from pydantic import BaseModel


class APIIdentity(BaseModel):
    project_id: str
    role: Literal["user", "operator", "gateway", "worker"] = "user"
    subject: str = "api-key"
    worker_registration_key: str | None = None


def api_keys_from_env() -> dict[str, APIIdentity]:
    raw = os.getenv("EMBODIEDLAB_API_KEYS_JSON", "").strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("EMBODIEDLAB_API_KEYS_JSON must be an object keyed by API key")
    return {key: APIIdentity.model_validate(identity) for key, identity in value.items()}

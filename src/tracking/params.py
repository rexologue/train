from __future__ import annotations

import json
from typing import Any

from config import stable_hash


SENSITIVE_KEY_PARTS = ("password", "passwd", "secret", "token", "username", "credential", "auth")
MAX_PARAM_VALUE_LENGTH = 500


def config_hash(raw_config: dict[str, Any]) -> str:
    """Return the deterministic hash logged for an effective config."""

    return stable_hash(raw_config)


def flatten_config_params(raw_config: dict[str, Any]) -> dict[str, str]:
    """Flatten config values into MLflow-safe params while dropping secret-like keys."""

    params = flatten_mapping(raw_config)
    params["config_hash"] = config_hash(raw_config)
    return params


def flatten_mapping(value: Any, prefix: str = "") -> dict[str, str]:
    """Flatten JSON-like mappings into dotted-string MLflow params."""

    flattened: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in sorted(value.items()):
            child_key = str(key)
            path = f"{prefix}.{child_key}" if prefix else child_key
            if is_sensitive_path(path):
                continue
            flattened.update(flatten_mapping(item, path))
        return flattened

    rendered = render_param_value(value)
    if rendered is not None and prefix:
        flattened[prefix] = rendered
    return flattened


def is_sensitive_path(path: str) -> bool:
    """Return whether a config path should not be emitted to tracking."""

    lowered = path.lower()
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def render_param_value(value: Any) -> str | None:
    """Render one scalar/list value for MLflow params."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | str):
        rendered = str(value)
    else:
        try:
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except TypeError:
            rendered = str(value)
    if len(rendered) > MAX_PARAM_VALUE_LENGTH:
        return rendered[: MAX_PARAM_VALUE_LENGTH - 3] + "..."
    return rendered

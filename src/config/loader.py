from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schema import Config, ConfigError


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge YAML mappings recursively while treating `extends` as metadata."""

    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and resolve a local `extends` chain."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"Config {path} must contain a YAML mapping")
    extends = data.get("extends")
    if extends:
        base_path = Path(path).parent / str(extends)
        base = _load_yaml(base_path)
        data = _deep_merge(base, data)
    return data


def load_config(path: str | Path) -> Config:
    """Load a YAML config file and return the validated Config object."""

    return Config.from_dict(_load_yaml(path))

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import ConfigError, TrainingConfig, validate_config


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


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and resolve a local `extends` chain."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only in minimal envs.
        raise ConfigError("PyYAML is required to load project YAML configs") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"Config {path} must contain a YAML mapping")
    extends = data.get("extends")
    if extends:
        base_path = Path(path).parent / str(extends)
        base = load_yaml(base_path)
        data = _deep_merge(base, data)
    return data


def load_config(path: str | Path) -> TrainingConfig:
    """Load and validate a training/preprocessing YAML config."""

    return validate_config(load_yaml(path))

"""Typed configuration loading."""

from config.loader import load_config
from config.schema import Config, ConfigError


__all__ = [
    "Config",
    "ConfigError",
    "load_config",
]

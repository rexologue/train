from __future__ import annotations

from config import load_config
from preprocessing.io import resolve_split_paths


def test_resolve_split_paths_skips_missing_optional_test():
    config = load_config("configs/config.example.yaml")
    assert resolve_split_paths(config, ["test"]) == []

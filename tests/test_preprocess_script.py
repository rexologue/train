from __future__ import annotations

from qwen35_tuning.config.loader import load_config
from scripts.preprocess import resolve_split_work


def test_resolve_split_work_skips_missing_optional_test(tmp_path):
    config = load_config("configs/smoke.yaml")
    work = resolve_split_work(config, ["test"], tmp_path)
    assert work == []
    assert (tmp_path / "test_preprocess_manifest.json").exists()

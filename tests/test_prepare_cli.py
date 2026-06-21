from __future__ import annotations

from types import SimpleNamespace
import sys

import prepare


def test_prepare_cli_reuses_valid_cache_by_default(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_resolve_model_source(config, *, tracking_uri=None):
        calls["tracking_uri"] = tracking_uri
        return SimpleNamespace(
            effective_model_id=str(config.model.cache_dir),
            ref="models:/fixture@champion",
            pulled=False,
            used_local=True,
        )

    def fake_prepare_pretokenized_splits(config, splits, *, model_source=None, force_refresh=False):
        calls["splits"] = splits
        calls["model_source"] = model_source
        calls["force_refresh"] = force_refresh
        return []

    monkeypatch.setattr(sys, "argv", ["estadel-prepare", "--config", "configs/config.example.yaml"])
    monkeypatch.setattr(prepare, "resolve_model_source", fake_resolve_model_source)
    monkeypatch.setattr(prepare, "prepare_pretokenized_splits", fake_prepare_pretokenized_splits)

    prepare.main()

    assert calls["splits"] == ["train", "valid", "test"]
    assert calls["force_refresh"] is False
    assert calls["tracking_uri"]


def test_prepare_cli_force_rebuilds_pretokenized_cache(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_resolve_model_source(config, *, tracking_uri=None):
        del tracking_uri
        return SimpleNamespace(
            effective_model_id=str(config.model.cache_dir),
            ref="models:/fixture@champion",
            pulled=False,
            used_local=True,
        )

    def fake_prepare_pretokenized_splits(config, splits, *, model_source=None, force_refresh=False):
        del config, splits, model_source
        calls["force_refresh"] = force_refresh
        return []

    monkeypatch.setattr(sys, "argv", ["estadel-prepare", "--config", "configs/config.example.yaml", "--force"])
    monkeypatch.setattr(prepare, "resolve_model_source", fake_resolve_model_source)
    monkeypatch.setattr(prepare, "prepare_pretokenized_splits", fake_prepare_pretokenized_splits)

    prepare.main()

    assert calls["force_refresh"] is True

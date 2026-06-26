from __future__ import annotations

from types import SimpleNamespace
import sys

import pytest

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

    def fake_prepare_pretokenized_splits(
        config,
        splits,
        *,
        model_source=None,
        force_refresh=False,
        num_workers=None,
        worker_chunk_size=None,
    ):
        calls["splits"] = splits
        calls["model_source"] = model_source
        calls["force_refresh"] = force_refresh
        calls["num_workers"] = num_workers
        calls["worker_chunk_size"] = worker_chunk_size
        return []

    monkeypatch.setattr(sys, "argv", ["sft-dpo-prepare", "--config", "configs/config.example.yaml"])
    monkeypatch.setattr(prepare, "resolve_model_source", fake_resolve_model_source)
    monkeypatch.setattr(prepare, "prepare_pretokenized_splits", fake_prepare_pretokenized_splits)

    prepare.main()

    assert calls["splits"] == ["train", "valid", "test"]
    assert calls["force_refresh"] is False
    assert calls["num_workers"] is None
    assert calls["worker_chunk_size"] is None
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

    def fake_prepare_pretokenized_splits(
        config,
        splits,
        *,
        model_source=None,
        force_refresh=False,
        num_workers=None,
        worker_chunk_size=None,
    ):
        del config, splits, model_source
        calls["force_refresh"] = force_refresh
        calls["num_workers"] = num_workers
        calls["worker_chunk_size"] = worker_chunk_size
        return []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sft-dpo-prepare",
            "--config",
            "configs/config.example.yaml",
            "--force",
            "--workers",
            "4",
            "--worker-chunk-size",
            "128",
        ],
    )
    monkeypatch.setattr(prepare, "resolve_model_source", fake_resolve_model_source)
    monkeypatch.setattr(prepare, "prepare_pretokenized_splits", fake_prepare_pretokenized_splits)

    prepare.main()

    assert calls["force_refresh"] is True
    assert calls["num_workers"] == 4
    assert calls["worker_chunk_size"] == 128


def test_prepare_cli_rejects_non_positive_workers(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["sft-dpo-prepare", "--config", "configs/config.example.yaml", "--workers", "0"],
    )

    with pytest.raises(SystemExit):
        prepare.main()

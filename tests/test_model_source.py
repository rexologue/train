from __future__ import annotations

import json

import pytest
from modelctl import core as modelctl_core

from config import load_config
from config.model_source import effective_model_id
from tracking.model_source import _pull_model, registry_metadata_path, resolve_model_source, validate_model_payload


def _registry_config(tmp_path):
    config = load_config("configs/config.example.yaml")
    config.raw["model"].update(
        {
            "name": "qwen35",
            "alias": "champion",
            "cache_dir": str(tmp_path / "models" / "qwen35"),
            "checks": {
                "verify_local_hash": True,
                "verify_remote_ref": False,
                "require_registry_metadata": True,
            },
        }
    )
    return config


def test_registry_model_source_uses_non_empty_local_dir_without_remote_calls(tmp_path, monkeypatch):
    config = _registry_config(tmp_path)
    local_dir = tmp_path / "models" / "qwen35"
    local_dir.mkdir(parents=True)
    (local_dir / "config.json").write_text("{}", encoding="utf-8")
    (local_dir / "model.safetensors").write_text("weights", encoding="utf-8")
    registry_metadata_path(local_dir).write_text(
        json.dumps({"resolved_version": "7", "source_dir_hash": "sha256:local"}, sort_keys=True),
        encoding="utf-8",
    )

    def fail_info(*args, **kwargs):
        raise AssertionError("remote info should not be called")

    def fail_pull(*args, **kwargs):
        raise AssertionError("pull should not be called")

    monkeypatch.setattr("tracking.model_source._get_model_info", fail_info)
    monkeypatch.setattr("tracking.model_source._pull_model", fail_pull)
    monkeypatch.setattr("tracking.model_source._hash_directory", lambda path: "sha256:local")

    resolution = resolve_model_source(config, tracking_uri="http://mlflow:5000")

    assert resolution.used_local is True
    assert resolution.pulled is False
    assert resolution.verified_local_hash is True
    assert resolution.resolved_version == "7"
    assert resolution.effective_model_id == str(local_dir.resolve())


def test_tokenizer_always_uses_resolved_registry_model_dir(tmp_path):
    config = load_config("configs/config.example.yaml")
    local_dir = tmp_path / "registry_model"
    config.raw["model"]["resolved_model_id"] = str(local_dir)

    assert effective_model_id(config) == str(local_dir)


def test_registry_model_source_pulls_when_local_dir_is_empty(tmp_path, monkeypatch):
    config = _registry_config(tmp_path)
    calls = []

    def fake_info(ref, tracking_uri):
        calls.append(("info", ref, tracking_uri))
        return {"version": "9", "source_dir_hash": "sha256:pulled"}

    def fake_pull(ref, output_dir, tracking_uri):
        calls.append(("pull", ref, tracking_uri))
        output_dir.mkdir(parents=True)
        (output_dir / "config.json").write_text("{}", encoding="utf-8")
        (output_dir / "model.safetensors").write_text("weights", encoding="utf-8")

    monkeypatch.setattr("tracking.model_source._get_model_info", fake_info)
    monkeypatch.setattr("tracking.model_source._pull_model", fake_pull)
    monkeypatch.setattr("tracking.model_source._hash_directory", lambda path: "sha256:pulled")

    resolution = resolve_model_source(config, tracking_uri="http://mlflow:5000")

    local_dir = tmp_path / "models" / "qwen35"
    assert resolution.pulled is True
    assert resolution.used_local is False
    assert resolution.resolved_version == "9"
    assert (local_dir / "config.json").exists()
    sidecar = json.loads(registry_metadata_path(local_dir).read_text(encoding="utf-8"))
    assert sidecar["source_dir_hash"] == "sha256:pulled"
    assert calls == [
        ("info", "models:/qwen35@champion", "http://mlflow:5000"),
        ("pull", "models:/qwen35@champion", "http://mlflow:5000"),
    ]


def test_registry_model_source_rejects_local_hash_mismatch(tmp_path, monkeypatch):
    config = _registry_config(tmp_path)
    local_dir = tmp_path / "models" / "qwen35"
    local_dir.mkdir(parents=True)
    (local_dir / "config.json").write_text("{}", encoding="utf-8")
    (local_dir / "model.safetensors").write_text("weights", encoding="utf-8")
    registry_metadata_path(local_dir).write_text(
        json.dumps({"resolved_version": "7", "source_dir_hash": "sha256:expected"}, sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setattr("tracking.model_source._hash_directory", lambda path: "sha256:actual")

    with pytest.raises(ValueError, match="local model hash mismatch"):
        resolve_model_source(config, tracking_uri="http://mlflow:5000")


def test_registry_model_source_requests_payload_only_pull(tmp_path, monkeypatch):
    calls = []

    def fake_pull_model(ref, output_dir, *, payload_only, tracking_uri):
        calls.append((ref, output_dir, payload_only, tracking_uri))

    monkeypatch.setattr(modelctl_core, "pull_model", fake_pull_model)

    output_dir = tmp_path / "model"
    _pull_model("models:/qwen35@champion", output_dir, "http://mlflow:5000")

    assert calls == [
        ("models:/qwen35@champion", output_dir, True, "http://mlflow:5000"),
    ]


def test_model_payload_rejects_missing_indexed_weight_shards(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.weight": "model-00001-of-00002.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing weight shards"):
        validate_model_payload(model_dir)

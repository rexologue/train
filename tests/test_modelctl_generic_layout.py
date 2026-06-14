from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from modelctl import core


def test_log_generic_model_writes_direct_artifact_store_layout(tmp_path, monkeypatch):
    source_dir = tmp_path / "adapter"
    source_dir.mkdir()
    (source_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    logged_artifacts = []

    def fake_log_artifacts(local_dir, artifact_path):
        local_dir = Path(local_dir)
        logged_artifacts.append(
            {
                "artifact_path": artifact_path,
                "files": sorted(path.relative_to(local_dir).as_posix() for path in local_dir.rglob("*") if path.is_file()),
                "manifest": (
                    json.loads((local_dir / "manifest.json").read_text(encoding="utf-8"))
                    if (local_dir / "manifest.json").exists()
                    else None
                ),
            }
        )

    monkeypatch.setattr(core.mlflow, "log_artifacts", fake_log_artifacts)
    monkeypatch.setattr(core.mlflow, "active_run", lambda: SimpleNamespace(info=SimpleNamespace(run_id="run-123")))

    result = core.log_generic_model(
        source_dir,
        {
            "created_by": "modelctl",
            "kind": "generic",
            "model_name": "estadel-llm",
            "source_dir_hash": "sha256:abc",
        },
        {"artifact.kind": "peft_adapter_checkpoint"},
        {"training.global_step": 100},
    )

    assert result["layout"] == "modelctl_generic_direct"
    assert result["model_uri"] == "runs:/run-123/model"
    assert logged_artifacts[0]["artifact_path"] == "model"
    assert logged_artifacts[0]["files"] == [
        "MLmodel",
        "manifest.json",
        "metadata/general_tags.json",
        "metadata/training_tags.json",
    ]
    assert logged_artifacts[0]["manifest"]["source_dir_hash"] == "sha256:abc"
    assert logged_artifacts[1]["artifact_path"] == "model/payload"
    assert logged_artifacts[1]["files"] == ["adapter_config.json"]


def test_choose_pull_source_supports_direct_and_legacy_generic_layouts(tmp_path):
    direct_package = tmp_path / "direct"
    direct_payload = direct_package / "payload"
    direct_payload.mkdir(parents=True)

    legacy_package = tmp_path / "legacy"
    legacy_payload = legacy_package / "artifacts" / "package" / "payload"
    legacy_payload.mkdir(parents=True)
    (legacy_package / "artifacts" / "package" / "manifest.json").write_text(
        json.dumps({"created_by": "modelctl", "kind": "generic"}),
        encoding="utf-8",
    )

    assert core.choose_pull_source(direct_package, payload_only=True) == direct_payload
    assert core.choose_pull_source(legacy_package, payload_only=True) == legacy_payload
    assert core.choose_pull_source(direct_package, payload_only=False) == direct_package


def test_generic_payload_artifact_uris_prefer_direct_layout():
    assert core.generic_payload_artifact_uris("runs:/run-123/model") == [
        "runs:/run-123/model/payload",
        "runs:/run-123/model/artifacts/payload",
    ]

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from registry.modelctl_client import (
    ModelctlClient,
    ModelctlCommandFailure,
    ModelctlRegisterRequest,
    parse_modelctl_json_stdout,
)
from registry.package import build_candidate_registration_request
from registry.selection import CandidateWindowSelector, CheckpointCandidate, RegistrationDecision
from registry.tags import validate_training_aliases
from conftest import example_config


def test_candidate_window_selector_registers_best_checkpoint_after_window(tmp_path) -> None:
    selector = CandidateWindowSelector(
        register_every_n_checkpoints=3,
        selection_metric="eval/bfcl/accuracy",
        greater_is_better=True,
        candidate_alias_template="candidate-{candidate_index:06d}",
        rolling_candidate_alias="candidate-latest",
    )

    assert selector.observe_checkpoint(
        checkpoint_path=tmp_path / "step-000100",
        checkpoint_index=1,
        global_step=100,
        metrics={"eval/bfcl/accuracy": 0.70},
    ) is None
    assert selector.observe_checkpoint(
        checkpoint_path=tmp_path / "step-000200",
        checkpoint_index=2,
        global_step=200,
        metrics={"eval/bfcl/accuracy": 0.90},
    ) is None
    decision = selector.observe_checkpoint(
        checkpoint_path=tmp_path / "step-000300",
        checkpoint_index=3,
        global_step=300,
        metrics={"eval/bfcl/accuracy": 0.80},
    )

    assert decision is not None
    assert decision.checkpoint.global_step == 200
    assert decision.candidate_index == 1
    assert decision.aliases == ["candidate-000001", "candidate-latest"]


def test_build_candidate_registration_request_writes_modelctl_tags(tmp_path) -> None:
    config = example_config()
    checkpoint = CheckpointCandidate(
        path=tmp_path / "step-000100",
        checkpoint_index=1,
        global_step=100,
        metric_value=0.91,
        metrics={"eval/loss": 0.91},
    )
    decision = RegistrationDecision(
        checkpoint=checkpoint,
        candidate_index=1,
        aliases=["candidate-000001", "candidate-latest"],
    )

    request = build_candidate_registration_request(config, decision)

    assert request.model_name == config.project.name
    assert request.source_dir == checkpoint.path / "adapter"
    assert request.aliases == ("candidate-000001", "candidate-latest")
    assert request.general_tags_json is not None
    assert request.training_tags_json is not None
    general_tags = json.loads(request.general_tags_json.read_text(encoding="utf-8"))
    training_tags = json.loads(request.training_tags_json.read_text(encoding="utf-8"))
    assert general_tags["artifact.kind"] == "peft_adapter_checkpoint"
    assert training_tags["training.registry_role"] == "candidate"
    assert training_tags["training.selection_metric"] == "eval/loss"


def test_modelctl_client_builds_register_command_and_rejects_production_aliases() -> None:
    client = ModelctlClient(tracking_uri="http://mlflow.example", executable="modelctl")
    request = ModelctlRegisterRequest(
        model_name="dialog-model",
        source_dir="checkpoint/adapter",
        aliases=("candidate-000001", "candidate-latest"),
        general_tags_json="general.json",
        training_tags_json="training.json",
    )

    args = []
    aliases = list(request.aliases)
    validate_training_aliases(aliases)
    args.extend(["register", str(request.source_dir), request.model_name])
    for alias in aliases:
        args.extend(["--alias", alias])
    args.extend(["--general-tags-json", str(request.general_tags_json)])
    args.extend(["--training-tags-json", str(request.training_tags_json)])

    command = client._command(args)

    assert command[:3] == ["modelctl", "register", "checkpoint/adapter"]
    assert "candidate-000001" in command
    assert command[-2:] == ["--tracking-uri", "http://mlflow.example"]
    with pytest.raises(ValueError, match="production aliases"):
        validate_training_aliases(["candidate-000001", "champion"])


def test_modelctl_client_parses_json_and_surfaces_failures(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        assert command == ["modelctl", "info", "models:/source@champion"]
        assert kwargs["capture_output"] is True
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"name": "source", "version": "3", "aliases": ["champion"], "payload_hash": "sha256:x"}),
            stderr="",
        )

    monkeypatch.setattr("registry.modelctl_client.subprocess.run", fake_run)

    info = ModelctlClient().info("models:/source@champion")

    assert info.name == "source"
    assert info.version == "3"
    assert info.payload_hash == "sha256:x"

    def failing_run(command, **kwargs):
        del command, kwargs
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr("registry.modelctl_client.subprocess.run", failing_run)

    with pytest.raises(ModelctlCommandFailure, match="boom"):
        ModelctlClient().info("models:/source@missing")


def test_modelctl_client_parses_trailing_json_after_progress_logs() -> None:
    stdout = "\n".join(
        [
            "[modelctl] hashing payload directory: checkpoint/adapter",
            "[modelctl] registered model version: name=dialog-model, version=7",
            '{"name": "dialog-model", "version": "7", "aliases": ["candidate-latest"]}',
        ]
    )

    payload = parse_modelctl_json_stdout(stdout, ["modelctl", "register"])

    assert payload == {"name": "dialog-model", "version": "7", "aliases": ["candidate-latest"]}

from __future__ import annotations

import json

from config import load_config
from registry.package import build_candidate_registration_args
from registry.selection import CandidateWindowSelector
from train import restore_registry_selector
from trainer.state import TrainerState


def test_candidate_window_selector_registers_best_checkpoint_after_window(tmp_path):
    selector = CandidateWindowSelector(
        register_every_n_checkpoints=3,
        selection_metric="eval/bfcl/accuracy",
        greater_is_better=True,
        candidate_alias_template="candidate-{candidate_index:06d}",
        rolling_candidate_alias="candidate-latest",
    )

    assert (
        selector.observe_checkpoint(
            checkpoint_path=tmp_path / "step-000100",
            checkpoint_index=1,
            global_step=100,
            metrics={"eval/bfcl/accuracy": 0.70},
        )
        is None
    )
    assert (
        selector.observe_checkpoint(
            checkpoint_path=tmp_path / "step-000200",
            checkpoint_index=2,
            global_step=200,
            metrics={"eval/bfcl/accuracy": 0.90},
        )
        is None
    )
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

    next_decision = selector.observe_checkpoint(
        checkpoint_path=tmp_path / "step-000400",
        checkpoint_index=4,
        global_step=400,
        metrics={"eval/bfcl/accuracy": 0.95},
    )

    assert next_decision is None
    assert selector.next_candidate_index == 2


def test_build_candidate_registration_args_uses_adapter_checkpoint_package(tmp_path):
    config = load_config("configs/config.example.yaml")
    config.raw["registry"]["selection"] = {"metric": "eval/bfcl/accuracy", "mode": "max"}
    selector = CandidateWindowSelector.from_config(config)
    decision = selector.observe_checkpoint(
        checkpoint_path=tmp_path / "step-000100",
        checkpoint_index=1,
        global_step=100,
        metrics={"eval/bfcl/accuracy": 0.91},
    )

    assert decision is None

    for index in range(2, int(config.section("registry")["register_every_n_checkpoints"]) + 1):
        decision = selector.observe_checkpoint(
            checkpoint_path=tmp_path / f"step-{index:06d}",
            checkpoint_index=index,
            global_step=index * 100,
            metrics={"eval/bfcl/accuracy": 0.80 + index / 100},
        )

    assert decision is not None
    args = build_candidate_registration_args(config, decision)

    assert args[:2] == ["modelctl", "register"]
    assert str(decision.checkpoint.path / "adapter") in args
    assert "candidate-000001" in args
    assert "candidate-latest" in args
    assert "champion" not in args
    assert "baseline" not in args
    assert "--training-tags-json" in args
    assert "--general-tags-json" in args

    general_tags_path = args[args.index("--general-tags-json") + 1]
    training_tags_path = args[args.index("--training-tags-json") + 1]
    tag_dir = decision.checkpoint.path.parent / "modelctl_tags" / decision.checkpoint.path.name
    general_tags = json.loads((tag_dir / "general_tags.json").read_text(encoding="utf-8"))
    training_tags = json.loads((tag_dir / "training_tags.json").read_text(encoding="utf-8"))

    assert general_tags_path == str(tag_dir / "general_tags.json")
    assert training_tags_path == str(tag_dir / "training_tags.json")
    assert general_tags["artifact.kind"] == "peft_adapter_checkpoint"
    assert training_tags["training.registry_role"] == "candidate"


def test_restore_registry_selector_preserves_incomplete_window(tmp_path):
    config = load_config("configs/config.example.yaml")
    config.raw["registry"]["selection"] = {"metric": "eval/bfcl/accuracy", "mode": "max"}
    config.raw["project"]["output_dir"] = str(tmp_path.parent)
    checkpoint_root = config.checkpoint_dir
    checkpoint_root.mkdir()
    for index in range(1, 5):
        checkpoint = checkpoint_root / f"step-{index * 10:06d}"
        checkpoint.mkdir()
        (checkpoint / "manifest.json").write_text(
            json.dumps(
                {
                    "global_step": index * 10,
                    "checkpoint_index": index,
                    "metrics": {"eval/bfcl/accuracy": index / 10},
                }
            ),
            encoding="utf-8",
        )

    selector = restore_registry_selector(config, TrainerState(global_step=40, checkpoint_index=4))
    decision = selector.observe_checkpoint(
        checkpoint_path=checkpoint_root / "step-000050",
        checkpoint_index=5,
        global_step=50,
        metrics={"eval/bfcl/accuracy": 0.1},
    )

    assert decision is not None
    assert decision.candidate_index == 1
    assert decision.checkpoint.checkpoint_index == 4

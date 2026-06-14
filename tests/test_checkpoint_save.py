from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from checkpointing.load import (
    adapter_dir,
    find_latest_checkpoint,
    load_trainer_state,
    prune_old_checkpoints,
    resolve_resume_checkpoint,
    validate_resume_checkpoint,
)
from checkpointing.load import load_training_state_without_model
from checkpointing.save import save_adapter_checkpoint, save_training_state_without_model
from config import load_config
from trainer.state import TrainerState


class DummyAdapterModel:
    def save_pretrained(self, path, **kwargs):
        del kwargs
        path.mkdir(parents=True, exist_ok=False)
        (path / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
        (path / "adapter_model.safetensors").write_text("adapter", encoding="utf-8")


class FakeAccelerator:
    is_main_process = True

    def wait_for_everyone(self):
        return None

    def unwrap_model(self, model):
        return model

    def get_state_dict(self, model):
        del model
        return {}

    def save(self, obj, path):
        del obj
        Path(path).write_text("saved", encoding="utf-8")

    def save_state(self, path):
        state_dir = Path(path)
        state_dir.mkdir(parents=True, exist_ok=False)
        (state_dir / "state.txt").write_text("accelerate", encoding="utf-8")


def save_dummy_checkpoint(tmp_path, state: TrainerState, **kwargs):
    return save_adapter_checkpoint(
        root_dir=tmp_path,
        model=DummyAdapterModel(),
        state=state,
        accelerator=FakeAccelerator(),
        **kwargs,
    )


def test_save_adapter_checkpoint_writes_atomic_package(tmp_path):
    state = TrainerState(global_step=12, validation_index=3, checkpoint_index=2, consumed_batches=48)

    checkpoint = save_dummy_checkpoint(
        tmp_path,
        state,
        metrics={"eval/bfcl/accuracy": 0.75},
        config_hashes={"config": "sha256:test"},
    )

    assert checkpoint == tmp_path / "step-000012"
    assert (checkpoint / "adapter" / "adapter_config.json").exists()
    assert (checkpoint / "accelerate_state" / "state.txt").exists()
    assert not (tmp_path / "step-000012.tmp").exists()

    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["adapter_path"] == "adapter"
    assert manifest["global_step"] == 12
    assert manifest["metrics"]["eval/bfcl/accuracy"] == 0.75

    checksums = json.loads((checkpoint / "checksums.json").read_text(encoding="utf-8"))
    assert "adapter/adapter_config.json" in checksums
    assert "manifest.json" in checksums


def test_save_adapter_checkpoint_refuses_to_overwrite_final_dir(tmp_path):
    state = TrainerState(global_step=12)
    save_dummy_checkpoint(tmp_path, state)

    with pytest.raises(FileExistsError, match="checkpoint already exists"):
        save_dummy_checkpoint(tmp_path, state)


def test_find_latest_checkpoint_and_load_state(tmp_path):
    first = save_dummy_checkpoint(tmp_path, TrainerState(global_step=4, validation_index=1, checkpoint_index=1))
    second = save_dummy_checkpoint(tmp_path, TrainerState(global_step=8, validation_index=2, checkpoint_index=2))

    assert find_latest_checkpoint(tmp_path) == second
    state = load_trainer_state(first)
    assert state.global_step == 4
    assert adapter_dir(second) == second / "adapter"


def test_find_latest_checkpoint_ignores_tmp_checkpoint_dirs(tmp_path):
    tmp_checkpoint = tmp_path / "step-000020.tmp"
    tmp_checkpoint.mkdir()
    (tmp_checkpoint / "manifest.json").write_text("{}", encoding="utf-8")

    assert find_latest_checkpoint(tmp_path) is None

    final = save_dummy_checkpoint(tmp_path, TrainerState(global_step=8))

    assert find_latest_checkpoint(tmp_path) == final


def test_prune_old_checkpoints_keeps_latest_and_protected(tmp_path):
    first = save_dummy_checkpoint(tmp_path, TrainerState(global_step=4))
    second = save_dummy_checkpoint(tmp_path, TrainerState(global_step=8))
    third = save_dummy_checkpoint(tmp_path, TrainerState(global_step=12))
    tmp_checkpoint = tmp_path / "step-000016.tmp"
    tmp_checkpoint.mkdir()

    deleted = prune_old_checkpoints(tmp_path, 1, protected_paths={first})

    assert deleted == [second]
    assert first.exists()
    assert not second.exists()
    assert third.exists()
    assert tmp_checkpoint.exists()


def test_validate_resume_checkpoint_enforces_strict_hashes(tmp_path):
    checkpoint_hashes = {
        "config": "sha256:config-a",
        "dataset": "sha256:dataset-a",
        "template": "sha256:template-a",
    }
    checkpoint = save_dummy_checkpoint(
        tmp_path,
        TrainerState(global_step=12),
        config_hashes=checkpoint_hashes,
    )
    config = load_config("configs/config.example.yaml")
    config.raw["checkpointing"]["resume"]["strict_config"] = True
    config.raw["checkpointing"]["resume"]["strict_dataset_hash"] = True
    config.raw["checkpointing"]["resume"]["strict_template_hash"] = True

    validate_resume_checkpoint(config, checkpoint, dict(checkpoint_hashes))

    current_hashes = dict(checkpoint_hashes)
    current_hashes["dataset"] = "sha256:dataset-b"
    with pytest.raises(ValueError, match="resume hash mismatch"):
        validate_resume_checkpoint(config, checkpoint, current_hashes)


def test_resume_checkpoint_is_resolved_from_project_output_dir(tmp_path):
    config = load_config("configs/config.example.yaml")
    config.raw["project"]["output_dir"] = str(tmp_path)
    checkpoint = save_dummy_checkpoint(config.checkpoint_dir, TrainerState(global_step=12))

    assert resolve_resume_checkpoint(config) == checkpoint


def test_training_state_save_excludes_model_weights_and_restores_optimizer(tmp_path):
    from accelerate.utils import DistributedType

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.NO,
        process_index=0,
        num_processes=1,
        step=7,
        scaler=None,
        project_configuration=SimpleNamespace(save_on_each_node=False),
        _schedulers=[scheduler],
        _dataloaders=[],
    )

    save_training_state_without_model(
        accelerator=accelerator,
        model=torch.nn.Linear(1, 1),
        optimizer=optimizer,
        output_dir=tmp_path,
    )

    assert not list(tmp_path.glob("model*"))
    assert (tmp_path / "optimizer.bin").exists()
    optimizer.param_groups[0]["lr"] = 0.5
    accelerator.step = 0

    load_training_state_without_model(
        accelerator=accelerator,
        model=torch.nn.Linear(1, 1),
        optimizer=optimizer,
        input_dir=tmp_path,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert accelerator.step == 7

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from checkpointing.checksums import directory_checksums
from trainer.state import TrainerState

from .atomic import atomic_checkpoint_dir, commit_atomic_checkpoint_dir, reset_tmp_checkpoint_dir


def checkpoint_dir_name(global_step: int) -> str:
    return f"step-{int(global_step):06d}"


def save_adapter_checkpoint(
    *,
    root_dir: str | Path,
    model: Any,
    optimizer: Any | None = None,
    state: TrainerState,
    metrics: dict[str, Any] | None = None,
    accelerator: Any,
    config_hashes: dict[str, str] | None = None,
) -> Path:
    """Atomically save an adapter-only Accelerate/FSDP checkpoint package."""

    final_dir = Path(root_dir) / checkpoint_dir_name(state.global_step)
    tmp_dir, final_dir = atomic_checkpoint_dir(final_dir)
    is_main_process = bool(accelerator.is_main_process)
    if is_main_process:
        tmp_dir = reset_tmp_checkpoint_dir(tmp_dir)
    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    adapter_dir = tmp_dir / "adapter"
    if not hasattr(unwrapped_model, "save_pretrained"):
        raise TypeError("adapter checkpointing requires model.save_pretrained")
    unwrapped_model.save_pretrained(
        adapter_dir,
        is_main_process=is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )

    accelerator.wait_for_everyone()

    save_training_state_without_model(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer if optimizer is not None else getattr(accelerator, "_optimizers", [None])[0],
        output_dir=tmp_dir / "accelerate_state",
    )
    accelerator.wait_for_everyone()

    if not is_main_process:
        accelerator.wait_for_everyone()
        return final_dir

    manifest = {
        "format_version": 1,
        "global_step": state.global_step,
        "validation_index": state.validation_index,
        "checkpoint_index": state.checkpoint_index,
        "consumed_batches": state.consumed_batches,
        "adapter_path": "adapter",
        "metrics": metrics or {},
        "config_hashes": config_hashes or {},
    }
    _write_json(tmp_dir / "trainer_state.json", state.to_dict())
    _write_json(tmp_dir / "metrics.json", metrics or {})
    _write_json(tmp_dir / "manifest.json", manifest)
    _write_json(tmp_dir / "checksums.json", directory_checksums(tmp_dir))

    committed = commit_atomic_checkpoint_dir(tmp_dir, final_dir)
    accelerator.wait_for_everyone()
    return committed


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def save_training_state_without_model(*, accelerator: Any, model: Any, optimizer: Any, output_dir: str | Path) -> None:
    """Save optimizer/scheduler/RNG state without duplicating the fixed base model."""

    if optimizer is None or not hasattr(accelerator, "distributed_type"):
        accelerator.save_state(str(output_dir))
        return

    from accelerate.checkpointing import save_accelerator_state
    from accelerate.utils import DistributedType

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    optimizers = [optimizer]
    if accelerator.distributed_type == DistributedType.FSDP:
        from accelerate.utils.fsdp_utils import save_fsdp_optimizer

        save_fsdp_optimizer(accelerator.state.fsdp_plugin, accelerator, optimizer, model, str(output_path), 0)
        optimizers = []

    save_accelerator_state(
        str(output_path),
        model_states=[],
        optimizers=optimizers,
        schedulers=list(getattr(accelerator, "_schedulers", [])),
        dataloaders=list(getattr(accelerator, "_dataloaders", [])),
        process_index=int(accelerator.process_index),
        step=int(accelerator.step),
        scaler=getattr(accelerator, "scaler", None),
        save_on_each_node=bool(accelerator.project_configuration.save_on_each_node),
        safe_serialization=True,
    )

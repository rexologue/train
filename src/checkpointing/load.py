from __future__ import annotations

import json
from pathlib import Path
import re
import shutil

from accelerate.checkpointing import load_accelerator_state
from accelerate.utils import DistributedType
from accelerate.utils.fsdp_utils import load_fsdp_optimizer

from checkpointing.atomic import checkpoint_is_ready
from checkpointing.checksums import verify_directory_checksums
from preprocessing.io import cache_root, manifest_path
from trainer.state import TrainerState
from utils.hashing import file_sha256, sha256_text, stable_hash


def assert_resume_hashes_match(expected: dict[str, str], actual: dict[str, str]) -> None:
    mismatched = {key: (expected.get(key), actual.get(key)) for key in expected if expected.get(key) != actual.get(key)}
    if mismatched:
        raise ValueError(f"resume hash mismatch: {mismatched}")


def find_latest_checkpoint(root_dir: str | Path) -> Path | None:
    root = Path(root_dir)
    if not root.exists():
        return None
    candidates = list_checkpoints(root)
    if not candidates:
        return None
    return max(candidates, key=_checkpoint_step)


def list_checkpoints(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir)
    if not root.exists():
        return []
    return sorted((path for path in root.glob("step-*") if _is_final_checkpoint_dir(path)), key=_checkpoint_step)


def prune_old_checkpoints(
    root_dir: str | Path,
    save_total_limit: int | None,
    *,
    protected_paths: set[str | Path] | None = None,
) -> list[Path]:
    if save_total_limit is None:
        return []
    limit = int(save_total_limit)
    if limit <= 0:
        raise ValueError("checkpointing.save_total_limit must be a positive integer")

    checkpoints = list_checkpoints(root_dir)
    protected = {_canonical_path(path) for path in protected_paths or set()}
    keepers = {_canonical_path(path) for path in checkpoints[-limit:]}
    deleted: list[Path] = []
    for checkpoint in checkpoints:
        canonical = _canonical_path(checkpoint)
        if canonical in keepers or canonical in protected:
            continue
        shutil.rmtree(checkpoint)
        deleted.append(checkpoint)
    return deleted


def resolve_resume_checkpoint(config) -> Path | None:
    resume = config.checkpointing.resume
    if not resume.enabled:
        return None
    checkpoint = find_latest_checkpoint(config.checkpoint_dir)
    if checkpoint is not None:
        verify_checkpoint_integrity(checkpoint)
    return checkpoint


def verify_checkpoint_integrity(checkpoint_dir: str | Path) -> None:
    """Fail before resume if a checkpoint is incomplete or corrupt.

    Resume must never start from a partially written or bit-rotted checkpoint:
    a corrupt optimizer/adapter shard would silently continue training from
    garbage. The READY marker proves the save completed; the checksum manifest
    proves every saved file is byte-identical.
    """

    path = Path(checkpoint_dir)
    if not checkpoint_is_ready(path):
        raise ValueError(f"refusing to resume from checkpoint without READY marker: {path}")
    checksums_path = path / "checksums.json"
    if not checksums_path.exists():
        raise ValueError(f"checkpoint is missing checksums.json: {path}")
    expected = json.loads(checksums_path.read_text(encoding="utf-8"))
    problems = verify_directory_checksums(path, expected)
    if problems:
        raise ValueError(f"checkpoint integrity check failed for {path}: {problems[:10]}")


def build_resume_hashes(config, *, tokenizer=None, model_source=None) -> dict[str, str]:
    hashes = {
        "dataset": file_sha256(manifest_path(cache_root(config))),
        "data_contract": build_data_contract_hash(config),
        "training_contract": build_training_contract_hash(config, model_source=model_source),
    }
    if tokenizer is not None:
        hashes["template"] = sha256_text(getattr(tokenizer, "chat_template", "") or "")
    return hashes


def validate_resume_checkpoint(config, checkpoint_dir: str | Path, current_hashes: dict[str, str]) -> None:
    resume = config.checkpointing.resume
    expected: dict[str, str] = {}
    if resume.strict_config:
        expected["training_contract"] = _require_current_hash(current_hashes, "training_contract")
    if resume.strict_dataset_hash:
        expected["dataset"] = _require_current_hash(current_hashes, "dataset")
        expected["data_contract"] = _require_current_hash(current_hashes, "data_contract")
    if resume.strict_template_hash:
        expected["template"] = _require_current_hash(current_hashes, "template")
    if not expected:
        return
    manifest = load_checkpoint_manifest(checkpoint_dir)
    actual = manifest.get("config_hashes")
    if not isinstance(actual, dict):
        actual = {}
    assert_resume_hashes_match(expected, actual)


def build_data_contract_hash(config) -> str:
    """Hash data and sampler settings that define batch order and membership."""

    return stable_hash(
        {
            "pretokenized_manifest": file_sha256(manifest_path(cache_root(config))),
            "sampler": {
                "batch_size": config.training.per_device_train_batch_size,
                "drop_last": config.training.drop_last,
                "seed": config.project.seed,
            },
        }
    )


def build_training_contract_hash(config, *, model_source=None) -> str:
    """Hash training-critical settings separately from run metadata."""

    return stable_hash(
        {
            "model": {
                "name": config.model.name,
                "alias": config.model.alias,
                "cache_dir": str(config.model.cache_dir),
                "precision": config.model.precision,
                "attn_implementation": config.model.attn_implementation,
                "experts_implementation": config.model.experts_implementation,
                "gradient_checkpointing": config.model.gradient_checkpointing,
                "freeze_router": config.model.freeze_router,
                "expected_payload_hash": model_source_payload_hash(model_source),
            },
            "tokenizer": config.to_dict()["tokenizer"],
            "lora": config.to_dict()["lora"],
            "loss_routing": config.to_dict()["loss_routing"],
            "training": config.to_dict()["training"],
            "distributed": config.to_dict()["distributed"],
        }
    )


def model_source_payload_hash(model_source) -> str | None:
    """Return the verified model payload hash used for resume checks."""

    if model_source is None:
        return None
    expected = getattr(model_source, "expected_payload_hash", None)
    if expected:
        return str(expected)
    source = getattr(model_source, "source_dir_hash", None)
    if source:
        return str(source)
    local = getattr(model_source, "local_payload_hash", None)
    if local:
        return str(local)
    return None


def load_trainer_state(checkpoint_dir: str | Path) -> TrainerState:
    path = Path(checkpoint_dir) / "trainer_state.json"
    if not path.exists():
        raise FileNotFoundError(f"missing trainer_state.json in checkpoint: {checkpoint_dir}")
    return TrainerState.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_training_state_without_model(*, accelerator, model, optimizer, input_dir: str | Path) -> None:
    """Restore optimizer/scheduler/RNG state after base model + adapter loading."""

    if not hasattr(accelerator, "distributed_type"):
        accelerator.load_state(str(input_dir))
        return

    optimizers = [optimizer]
    if accelerator.distributed_type == DistributedType.FSDP:
        load_fsdp_optimizer(accelerator.state.fsdp_plugin, accelerator, optimizer, model, str(input_dir), 0)
        optimizers = []

    overrides = load_accelerator_state(
        str(input_dir),
        models=[],
        optimizers=optimizers,
        schedulers=list(getattr(accelerator, "_schedulers", [])),
        dataloaders=list(getattr(accelerator, "_dataloaders", [])),
        process_index=int(accelerator.process_index),
        scaler=getattr(accelerator, "scaler", None),
        map_location="on_device" if getattr(accelerator, "num_processes", 1) > 1 else "cpu",
    )
    if "step" in overrides:
        accelerator.step = overrides["step"]


def load_checkpoint_manifest(checkpoint_dir: str | Path) -> dict:
    path = Path(checkpoint_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing manifest.json in checkpoint: {checkpoint_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def adapter_dir(checkpoint_dir: str | Path) -> Path:
    manifest = load_checkpoint_manifest(checkpoint_dir)
    return Path(checkpoint_dir) / str(manifest.get("adapter_path", "adapter"))


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("step-"))
    except ValueError:
        return -1


def _is_final_checkpoint_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and re.fullmatch(r"step-\d+", path.name) is not None
        and (path / "manifest.json").exists()
        and checkpoint_is_ready(path)
    )


def _canonical_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _require_current_hash(current_hashes: dict[str, str], key: str) -> str:
    value = current_hashes.get(key)
    if not value:
        raise ValueError(f"strict resume requires current {key} hash")
    return value

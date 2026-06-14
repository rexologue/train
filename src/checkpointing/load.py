from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from config.hashing import file_sha256, sha256_text, stable_hash
from preprocessing.io import cache_root, manifest_path
from trainer.state import TrainerState


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
    resume = config.section("checkpointing").get("resume")
    if not isinstance(resume, dict) or not resume.get("enabled", False):
        return None
    raw_path = resume.get("path")
    if raw_path in (None, "", "auto"):
        return find_latest_checkpoint(config.section("checkpointing")["root_dir"])
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"explicit resume checkpoint does not exist: {path}")
    return path


def build_resume_hashes(config, *, tokenizer=None) -> dict[str, str]:
    hashes = {
        "config": stable_hash(config.raw),
        "dataset": file_sha256(manifest_path(cache_root(config))),
    }
    if tokenizer is not None:
        hashes["template"] = sha256_text(getattr(tokenizer, "chat_template", "") or "")
    return hashes


def validate_resume_checkpoint(config, checkpoint_dir: str | Path, current_hashes: dict[str, str]) -> None:
    resume = config.section("checkpointing").get("resume")
    if not isinstance(resume, dict):
        return
    expected: dict[str, str] = {}
    if bool(resume.get("strict_config", False)):
        expected["config"] = _require_current_hash(current_hashes, "config")
    if bool(resume.get("strict_dataset_hash", False)):
        expected["dataset"] = _require_current_hash(current_hashes, "dataset")
    if bool(resume.get("strict_template_hash", False)):
        expected["template"] = _require_current_hash(current_hashes, "template")
    if not expected:
        return
    manifest = load_checkpoint_manifest(checkpoint_dir)
    actual = manifest.get("config_hashes")
    if not isinstance(actual, dict):
        actual = {}
    assert_resume_hashes_match(expected, actual)


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

    from accelerate.checkpointing import load_accelerator_state
    from accelerate.utils import DistributedType

    optimizers = [optimizer]
    if accelerator.distributed_type == DistributedType.FSDP:
        from accelerate.utils.fsdp_utils import load_fsdp_optimizer

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
    )


def _canonical_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _require_current_hash(current_hashes: dict[str, str], key: str) -> str:
    value = current_hashes.get(key)
    if not value:
        raise ValueError(f"strict resume requires current {key} hash")
    return value

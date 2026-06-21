from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.dataloaders import DataLoaderBundle, SplitDataLoader
from data.ref_logprobs import (
    RefLogprobCacheState,
    build_ref_logprob_cache_signature,
    dpo_row_count,
    load_and_apply_ref_logprob_cache,
    ref_logprob_cache_dir,
    write_ref_logprob_split_cache,
)
from losses.dpo import sequence_logps
from utils.logging import get_logger


def ensure_ref_logprob_cache(
    *,
    config: Any,
    dataloaders: DataLoaderBundle,
    accelerator: Any,
    model_source: Any | None,
) -> RefLogprobCacheState:
    """Ensure configured DPO reference-logprob cache exists and is attached."""

    logger = get_logger(__name__)
    is_main_process = bool(getattr(accelerator, "is_main_process", True))
    reference = config.loss_routing.dpo.reference
    dpo_rows = dpo_row_count(dataloaders)
    if dpo_rows == 0:
        signature = build_ref_logprob_cache_signature(config, model_source)
        return RefLogprobCacheState(
            cache_dir=ref_logprob_cache_dir(config, signature),
            signature=signature,
            applied_rows=0,
            missing_rows=0,
            complete=True,
        )

    if not reference.cache_enabled:
        raise ValueError("DPO rows require loss_routing.dpo.reference.cache_enabled=true")

    state = load_and_apply_ref_logprob_cache(config, dataloaders, model_source=model_source)
    if state.complete and not reference.cache_refresh:
        if is_main_process:
            logger.info("DPO ref-logprob cache reused: rows=%s path=%s", state.applied_rows, state.cache_dir)
        return state

    if is_main_process:
        estimate = estimate_ref_logprob_work(
            dataloaders,
            num_processes=int(getattr(accelerator, "num_processes", 1)),
        )
        logger.info(
            "DPO ref-logprob cache sizing: rows=%s chosen_tokens=%s rejected_tokens=%s total_tokens=%s "
            "batch_size=%s num_processes=%s forward_batches_per_rank=%s total_forward_calls=%s cache_required=%s",
            estimate["rows"],
            estimate["chosen_tokens"],
            estimate["rejected_tokens"],
            estimate["total_tokens"],
            estimate["batch_size"],
            estimate["num_processes"],
            estimate["forward_batches_per_rank"],
            estimate["total_forward_calls"],
            reference.cache_required,
        )
        logger.info(
            "DPO ref-logprob cache precompute required: complete=%s missing=%s refresh=%s path=%s",
            state.complete,
            state.missing_rows,
            reference.cache_refresh,
            state.cache_dir,
        )
    started_at = time.monotonic()
    compute_ref_logprob_cache(
        config=config,
        dataloaders=dataloaders,
        accelerator=accelerator,
        cache_dir=state.cache_dir,
        total_rows=dpo_rows,
        refresh=bool(reference.cache_refresh),
    )
    if is_main_process:
        elapsed = max(time.monotonic() - started_at, 1e-9)
        estimate = estimate_ref_logprob_work(
            dataloaders,
            num_processes=int(getattr(accelerator, "num_processes", 1)),
        )
        logger.info(
            "DPO ref-logprob cache precompute finished: rows=%s elapsed_seconds=%.2f rows_per_second=%.4f "
            "tokens_per_second=%.2f path=%s",
            estimate["rows"],
            elapsed,
            estimate["rows"] / elapsed,
            estimate["total_tokens"] / elapsed,
            state.cache_dir,
        )
    state = load_and_apply_ref_logprob_cache(config, dataloaders, model_source=model_source)
    if state.missing_rows:
        raise ValueError(f"DPO reference-logprob cache remains incomplete: missing rows={state.missing_rows}")
    return state


def estimate_ref_logprob_work(dataloaders: DataLoaderBundle, *, num_processes: int) -> dict[str, int]:
    """Estimate DPO reference precompute size before loading the reference model."""

    rows = 0
    chosen_tokens = 0
    rejected_tokens = 0
    forward_batches_per_rank = 0
    batch_size = 1
    processes = max(int(num_processes), 1)
    for split_loader in dataloaders.splits.values():
        split_rows = [row for row in split_loader.dataset.rows if row.get("loss_kind") == "dpo_target"]
        if not split_rows:
            continue
        batch_size = max(int(split_loader.summary["batch_size"]), 1)
        rows += len(split_rows)
        chosen_tokens += sum(int(row.get("chosen_length") or len(row.get("chosen_input_ids") or [])) for row in split_rows)
        rejected_tokens += sum(int(row.get("rejected_length") or len(row.get("rejected_input_ids") or [])) for row in split_rows)
        balanced_rows_per_rank = math.ceil(len(split_rows) / processes)
        forward_batches_per_rank += math.ceil(balanced_rows_per_rank / batch_size) * 2
    return {
        "rows": rows,
        "chosen_tokens": chosen_tokens,
        "rejected_tokens": rejected_tokens,
        "total_tokens": chosen_tokens + rejected_tokens,
        "batch_size": batch_size,
        "num_processes": processes,
        "forward_batches_per_rank": forward_batches_per_rank,
        "total_forward_calls": forward_batches_per_rank * processes,
    }


def compute_ref_logprob_cache(
    *,
    config: Any,
    dataloaders: DataLoaderBundle,
    accelerator: Any,
    cache_dir: Path,
    total_rows: int,
    refresh: bool,
) -> None:
    """Compute DPO reference logprobs through a temporary base-model FSDP instance."""

    logger = get_logger(__name__)
    is_main_process = bool(getattr(accelerator, "is_main_process", True))
    if is_main_process:
        if refresh:
            reset_cache_dir(cache_dir)
        else:
            cache_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    progress_config = getattr(config, "progress", None)
    progress_enabled = bool(getattr(progress_config, "enabled", True))
    model = None
    try:
        model = build_base_reference_policy(config)
        model = accelerator.prepare(model)
        for split_loader in dataloaders.splits.values():
            if not split_has_dpo(split_loader):
                continue
            if is_main_process:
                logger.info("DPO ref-logprob precompute started: split=%s path=%s", split_loader.split, cache_dir)
            local_path = compute_split_ref_logprobs(
                config=config,
                split_loader=split_loader,
                model=model,
                accelerator=accelerator,
                cache_dir=cache_dir,
                progress_enabled=progress_enabled,
            )
            accelerator.wait_for_everyone()
            if is_main_process:
                rows = merge_rank_jsonl(cache_dir, split_loader.split)
                validate_split_rows(split_loader, rows)
                write_ref_logprob_split_cache(cache_dir, split_loader, rows)
                logger.info(
                    "DPO ref-logprob cache ready: split=%s rows=%s path=%s",
                    split_loader.split,
                    len(rows),
                    cache_dir,
                )
            accelerator.wait_for_everyone()
            if is_main_process:
                cleanup_rank_file(local_path)
            accelerator.wait_for_everyone()
    finally:
        if model is not None:
            release_base_reference_policy(model, accelerator)


def compute_split_ref_logprobs(
    *,
    config: Any,
    split_loader: SplitDataLoader,
    model: Any,
    accelerator: Any,
    cache_dir: Path,
    progress_enabled: bool,
) -> Path:
    """Compute local-rank reference logprobs for one split and write JSONL rows."""

    rank = int(getattr(accelerator, "process_index", 0))
    rank_dir = cache_dir / f"{split_loader.split}.rank_rows"
    rank_dir.mkdir(parents=True, exist_ok=True)
    output_path = rank_dir / f"rank-{rank:05d}.jsonl"
    process_index = int(getattr(accelerator, "process_index", 0))
    num_processes = int(getattr(accelerator, "num_processes", 1))
    batch_size = int(split_loader.summary["batch_size"])
    local_rows = balanced_rank_rows(
        [row for row in split_loader.dataset.rows if row.get("loss_kind") == "dpo_target"],
        process_index=process_index,
        num_processes=max(num_processes, 1),
    )
    local_prefix_rows, local_prefix_row_count = valid_rank_prefix_rows(output_path, local_rows)
    local_completed_batches = local_prefix_row_count // max(batch_size, 1)
    completed_batches = synchronized_min_int(local_completed_batches, accelerator)
    skip_row_count = min(completed_batches * max(batch_size, 1), len(local_rows))
    prefix_rows = valid_rank_rows_for_prefix(output_path, local_rows[:skip_row_count])
    remaining_rows = local_rows[skip_row_count:]
    dpo_loader = DataLoader(
        remaining_rows,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=split_loader.dataloader.collate_fn,
    )
    logger = get_logger(__name__)
    is_main_process = bool(getattr(accelerator, "is_main_process", True))
    local_owner_total = sum(1 for row in local_rows if bool(row.get("_ref_logprob_cache_owner", True)))
    resumed_owner_rows = len(prefix_rows)
    if is_main_process:
        logger.info(
            "DPO ref-logprob rank progress: split=%s rank=%s resumed_rows=%s local_rows=%s "
            "local_owner_rows=%s skipped_batches=%s output=%s",
            split_loader.split,
            rank,
            resumed_owner_rows,
            len(local_rows),
            local_owner_total,
            completed_batches,
            output_path,
        )
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    progress = reference_progress(
        local_owner_total,
        initial=resumed_owner_rows,
        enabled=progress_enabled,
        main_process=is_main_process,
    )
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            for row in prefix_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            with torch.no_grad():
                for batch in dpo_loader:
                    batch = move_batch_to_device(batch, getattr(accelerator, "device", None))
                    chosen_logps = sequence_logps(
                        model,
                        input_ids=batch["chosen_input_ids"],
                        attention_mask=batch.get("chosen_attention_mask"),
                        labels=batch["chosen_labels"],
                        ignore_index=config.ignore_index,
                    )
                    rejected_logps = sequence_logps(
                        model,
                        input_ids=batch["rejected_input_ids"],
                        attention_mask=batch.get("rejected_attention_mask"),
                        labels=batch["rejected_labels"],
                        ignore_index=config.ignore_index,
                    )
                    written = write_batch_rows(handle, batch, chosen_logps, rejected_logps)
                    handle.flush()
                    progress.update(written)
    finally:
        progress.close()
        if was_training and hasattr(model, "train"):
            model.train()
    return output_path


def build_base_reference_policy(config: Any) -> Any:
    """Load the frozen base model used as the DPO reference policy."""

    from trainer.modeling import (
        cast_floating_parameters,
        disable_model_kv_cache,
        load_base_model,
        precision_to_dtype,
        validate_model_runtime_requirements,
    )

    validate_model_runtime_requirements(config)
    model = load_base_model(config)
    disable_model_kv_cache(model)
    cast_floating_parameters(model, precision_to_dtype(config.model.precision))
    if hasattr(model, "eval"):
        model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def release_base_reference_policy(model: Any, accelerator: Any) -> None:
    """Release the temporary reference model before the train model is built."""

    if hasattr(accelerator, "free_memory"):
        try:
            accelerator.free_memory(model)
        except TypeError:
            accelerator.free_memory()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reference_progress(total_rows: int, *, initial: int = 0, enabled: bool, main_process: bool) -> Any:
    """Return a progress bar for the reference precompute stage."""

    return tqdm(
        total=total_rows,
        initial=initial,
        desc="reference",
        dynamic_ncols=True,
        disable=not (enabled and main_process),
    )


def build_dpo_dataloader(split_loader: SplitDataLoader, *, process_index: int, num_processes: int) -> DataLoader:
    """Build an unshuffled balanced DPO-only DataLoader for cache precomputation."""

    rows = [row for row in split_loader.dataset.rows if row.get("loss_kind") == "dpo_target"]
    local_rows = balanced_rank_rows(rows, process_index=int(process_index), num_processes=max(int(num_processes), 1))
    return DataLoader(
        local_rows,
        batch_size=int(split_loader.summary["batch_size"]),
        shuffle=False,
        collate_fn=split_loader.dataloader.collate_fn,
    )


def synchronized_min_int(value: int, accelerator: Any) -> int:
    """Return the minimum integer across ranks when distributed is initialized."""

    if not dist.is_available() or not dist.is_initialized():
        return int(value)
    device = getattr(accelerator, "device", None)
    tensor = torch.tensor([int(value)], dtype=torch.long, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return int(tensor.item())


def load_rank_rows(path: Path) -> dict[int, dict[str, Any]]:
    """Load rank-local JSONL rows by row_index, ignoring incomplete trailing lines."""

    if not path.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                rows[int(row["row_index"])] = row
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def rank_row_matches(cached: dict[str, Any], row: dict[str, Any]) -> bool:
    """Return whether one rank-local row matches the current rendered DPO row."""

    return (
        str(cached.get("chosen_render_hash")) == str(row.get("chosen_render_hash"))
        and str(cached.get("rejected_render_hash")) == str(row.get("rejected_render_hash"))
    )


def valid_rank_prefix_rows(path: Path, local_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Return valid cached owner rows and local-row prefix length safe to skip."""

    cached_rows = load_rank_rows(path)
    prefix_rows: list[dict[str, Any]] = []
    prefix_row_count = 0
    for row in local_rows:
        if not bool(row.get("_ref_logprob_cache_owner", True)):
            prefix_row_count += 1
            continue
        cached = cached_rows.get(int(row["row_index"]))
        if cached is None or not rank_row_matches(cached, row):
            break
        prefix_rows.append(cached)
        prefix_row_count += 1
    return prefix_rows, prefix_row_count


def valid_rank_rows_for_prefix(path: Path, prefix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return cached owner rows for an already synchronized local-row prefix."""

    cached_rows = load_rank_rows(path)
    output: list[dict[str, Any]] = []
    for row in prefix_rows:
        if not bool(row.get("_ref_logprob_cache_owner", True)):
            continue
        cached = cached_rows.get(int(row["row_index"]))
        if cached is None or not rank_row_matches(cached, row):
            break
        output.append(cached)
    return output


def balanced_rank_rows(rows: list[dict[str, Any]], *, process_index: int, num_processes: int) -> list[dict[str, Any]]:
    """Return rank rows padded so every rank executes the same number of forwards."""

    if not rows:
        return []
    row_count = len(rows)
    steps = (row_count + num_processes - 1) // num_processes
    balanced: list[dict[str, Any]] = []
    for step in range(steps):
        global_index = step * num_processes + process_index
        source_index = global_index if global_index < row_count else row_count - 1
        row = dict(rows[source_index])
        row["_ref_logprob_cache_owner"] = global_index < row_count
        balanced.append(row)
    return balanced


def move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    """Move tensor batch fields to the accelerator device."""

    if device is None:
        return batch
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def write_batch_rows(handle: Any, batch: dict[str, Any], chosen_logps: Any, rejected_logps: Any) -> int:
    """Write one computed cache batch as JSONL rows."""

    chosen_values = chosen_logps.detach().float().cpu().tolist()
    rejected_values = rejected_logps.detach().float().cpu().tolist()
    owners = batch.get("ref_logprob_cache_owner")
    written = 0
    for index, row_index in enumerate(batch["row_index"]):
        if owners is not None and not bool(owners[index]):
            continue
        row = {
            "sample_id": str(batch["sample_id"][index]),
            "row_index": int(row_index),
            "chosen_render_hash": str(batch["chosen_render_hash"][index]),
            "rejected_render_hash": str(batch["rejected_render_hash"][index]),
            "chosen_ref_logp": float(chosen_values[index]),
            "rejected_ref_logp": float(rejected_values[index]),
        }
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        written += 1
    return written


def merge_rank_jsonl(cache_dir: Path, split: str) -> list[dict[str, Any]]:
    """Merge all rank-local JSONL files for one split."""

    rank_dir = cache_dir / f"{split}.rank_rows"
    rows: list[dict[str, Any]] = []
    for path in sorted(rank_dir.glob("rank-*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    rows.sort(key=lambda row: int(row["row_index"]))
    return rows


def validate_split_rows(split_loader: SplitDataLoader, rows: list[dict[str, Any]]) -> None:
    """Validate that cache rows cover every DPO row exactly once."""

    expected = sorted(int(row["row_index"]) for row in split_loader.dataset.rows if row.get("loss_kind") == "dpo_target")
    actual = sorted(int(row["row_index"]) for row in rows)
    if actual != expected:
        raise ValueError(f"ref-logprob cache row mismatch for {split_loader.split}: expected={len(expected)} actual={len(actual)}")


def split_has_dpo(split_loader: SplitDataLoader) -> bool:
    """Return whether a split contains DPO rows."""

    return any(row.get("loss_kind") == "dpo_target" for row in split_loader.dataset.rows)


def reset_cache_dir(cache_dir: Path) -> None:
    """Reset one generated reference-logprob cache directory."""

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=False)


def cleanup_rank_file(path: Path) -> None:
    """Remove rank-local temporary JSONL files after merge."""

    rank_dir = path.parent
    if rank_dir.exists():
        shutil.rmtree(rank_dir)

from __future__ import annotations

import argparse
import math
from typing import Any

import torch
from accelerate.utils import gather_object

from config import load_config
from data.collators import pad_sequences
from data.pretokenized_dataset import PretokenizedDataset
from data.ref_cache import (
    load_reusable_entries,
    read_cache_manifest,
    ref_cache_paths,
    reference_signature,
    write_ref_logp_cache,
)
from losses.dpo import peft_adapter_disabled, sequence_logps
from preprocessing.io import load_pretokenized_split_results
from tracking.model_source import (
    load_model_source_resolution_from_cache,
    resolve_model_source,
)
from trainer.distributed import create_accelerator, prepare_with_accelerator
from trainer.modeling import build_training_objects
from utils.logging import configure_logging, get_logger
from utils.seed import set_seed


DPO_LOSS_KIND = "dpo_target"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    """Precompute DPO reference completion logps into the optional ref cache.

    Run after `prepare` and before `train`, with the same launcher as training
    so the (possibly sharded) model loads identically:

        accelerate launch --use_fsdp src/precompute_ref.py --config <cfg>

    The cache is keyed by per-sequence render hash, so re-running only computes
    rows whose rendering changed. Training consumes the cache automatically when
    present and falls back to the on-the-fly reference otherwise.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=None,
        help="sequences per reference forward (default: training.per_device_train_batch_size)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute every reference logp, ignoring any reusable cache entries",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.project.seed)

    runtime = create_accelerator(config)
    accelerator = runtime.accelerator
    is_main = bool(getattr(accelerator, "is_main_process", True))
    configure_logging(is_main_process=is_main)
    logger = get_logger("precompute_ref")
    logger.info("loaded config: project=%s run_name=%s", config.project.name, config.project.run_name)

    # Only the main process talks to the registry; it writes the local sidecar
    # that worker processes read after the barrier, mirroring train.py.
    if is_main:
        model_source = resolve_model_source(config, tracking_uri=config.mlflow.tracking_uri)
        logger.info(
            "registry model resolved: effective_model_id=%s ref=%s pulled=%s used_local=%s",
            model_source.effective_model_id,
            model_source.ref,
            model_source.pulled,
            model_source.used_local,
        )
    accelerator.wait_for_everyone()
    if not is_main:
        model_source = load_model_source_resolution_from_cache(config)

    signature = reference_signature(config, model_source)
    logger.info("reference signature: %s", signature)

    results = load_pretokenized_split_results(config, ["train", "valid", "test"])
    sequences = collect_dpo_sequences(results)
    logger.info("collected unique DPO completion sequences: %s", len(sequences))
    if not sequences:
        logger.info("no dpo_target sequences found; nothing to precompute")
        return

    reusable = {} if args.force else load_reusable_entries(config, expected_signature=signature)
    missing = [seq for seq in sequences if seq["render_hash"] not in reusable]
    logger.info(
        "reference cache state: reusable=%s missing=%s force=%s",
        len(reusable),
        len(missing),
        args.force,
    )

    if not missing:
        if is_main:
            _refresh_cache_if_needed(config, sequences, reusable, signature, model_source, results)
        accelerator.wait_for_everyone()
        logger.info("reference cache already complete; no forward passes needed")
        return

    objects = build_training_objects(config, total_steps=1)
    model = prepare_with_accelerator(runtime, objects.model)[0]
    if hasattr(model, "eval"):
        model.eval()

    pad_token_id = _resolve_pad_token_id(objects.tokenizer)
    batch_size = int(args.batch_size or config.training.per_device_train_batch_size)
    logger.info("computing reference logps: missing=%s batch_size=%s", len(missing), batch_size)

    local_entries = compute_reference_logps(
        model,
        accelerator,
        missing,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        ignore_index=int(config.ignore_index),
    )

    gathered = gather_object([local_entries])
    merged = dict(reusable)
    for entry_map in gathered:
        merged.update(entry_map)

    # Keep only entries for sequences still present in the current data, so the
    # cache cannot grow unbounded across data revisions.
    current_hashes = {seq["render_hash"] for seq in sequences}
    merged = {render_hash: logp for render_hash, logp in merged.items() if render_hash in current_hashes}

    if is_main:
        missing_after = [seq["render_hash"] for seq in sequences if seq["render_hash"] not in merged]
        if missing_after:
            raise RuntimeError(
                f"reference precompute incomplete: {len(missing_after)} sequences still missing after gather; "
                f"sample={missing_after[:5]}"
            )
        splits = [result.split for result in results]
        write_ref_logp_cache(
            config,
            entries=merged,
            signature=signature,
            model_source=model_source,
            splits=splits,
        )
        logger.info("reference precompute complete: total_entries=%s", len(merged))
    accelerator.wait_for_everyone()


def collect_dpo_sequences(results: list[Any]) -> list[dict[str, Any]]:
    """Collect unique chosen/rejected completion sequences across splits.

    Each DPO row contributes two sequences (chosen, rejected), each addressed by
    its own render hash. Deduplicating by render hash means a sequence shared
    across splits or rows is scored once, and an unchanged sequence keeps its
    hash so a re-run reuses it.
    """

    logger = get_logger("precompute_ref")
    by_hash: dict[str, dict[str, Any]] = {}
    skipped_without_hash = 0

    for result in results:
        dataset = PretokenizedDataset.from_parquet(result.pretok_path, split=result.split)
        for row in dataset.rows:
            if row.get("loss_kind") != DPO_LOSS_KIND:
                continue
            for side in ("chosen", "rejected"):
                render_hash = row.get(f"{side}_render_hash")
                if not isinstance(render_hash, str):
                    skipped_without_hash += 1
                    continue
                if render_hash in by_hash:
                    continue
                by_hash[render_hash] = {
                    "render_hash": render_hash,
                    "input_ids": row[f"{side}_input_ids"],
                    "attention_mask": row[f"{side}_attention_mask"],
                    "labels": row[f"{side}_labels"],
                }

    if skipped_without_hash:
        logger.warning(
            "skipped %s DPO completion sequences with no render hash; they cannot be cached "
            "and will use the on-the-fly reference at train time",
            skipped_without_hash,
        )

    # Deterministic order so rank sharding is identical across processes.
    return [by_hash[render_hash] for render_hash in sorted(by_hash)]


def compute_reference_logps(
    model: Any,
    accelerator: Any,
    sequences: list[dict[str, Any]],
    *,
    batch_size: int,
    pad_token_id: int,
    ignore_index: int,
) -> dict[str, float]:
    """Score reference logps for this rank's shard with even cross-rank batches.

    Sequences are sharded round-robin across ranks. Because FSDP forward passes
    are collective, every rank must run the same number of forward calls or the
    fast ranks deadlock waiting on the parameter all-gather. We therefore pad the
    per-rank batch count up to the global maximum with dummy forwards whose
    results are discarded.
    """

    world_size = max(int(getattr(accelerator, "num_processes", 1)), 1)
    rank = int(getattr(accelerator, "process_index", 0))
    device = getattr(accelerator, "device", torch.device("cpu"))

    local = sequences[rank::world_size]
    local_num_batches = math.ceil(len(local) / batch_size) if local else 0
    global_num_batches = _global_max(accelerator, local_num_batches, device)

    logger = get_logger("precompute_ref")
    logger.info(
        "rank %s/%s scoring local_sequences=%s local_batches=%s global_batches=%s",
        rank,
        world_size,
        len(local),
        local_num_batches,
        global_num_batches,
    )

    # A non-empty fallback keeps exhausted ranks participating in the collective
    # forward. It is always a real sequence, so any logp it produces is correct;
    # it is simply not recorded by this rank.
    dummy = local[0] if local else sequences[0]
    entries: dict[str, float] = {}

    for batch_index in range(global_num_batches):
        start = batch_index * batch_size
        chunk = local[start : start + batch_size]
        is_real = bool(chunk)
        if not is_real:
            chunk = [dummy]

        logps = _score_batch(
            model,
            accelerator,
            chunk,
            device=device,
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
        )
        if is_real:
            for sequence, logp in zip(chunk, logps):
                entries[sequence["render_hash"]] = float(logp)

    return entries


def _score_batch(
    model: Any,
    accelerator: Any,
    chunk: list[dict[str, Any]],
    *,
    device: Any,
    pad_token_id: int,
    ignore_index: int,
) -> list[float]:
    input_ids = torch.tensor(
        pad_sequences([list(seq["input_ids"]) for seq in chunk], pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.tensor(
        pad_sequences([list(seq["attention_mask"]) for seq in chunk], 0),
        dtype=torch.long,
        device=device,
    )
    labels = torch.tensor(
        pad_sequences([list(seq["labels"]) for seq in chunk], ignore_index),
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad(), peft_adapter_disabled(model, accelerator=accelerator):
        logps = sequence_logps(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            ignore_index=ignore_index,
        )
    return [float(value) for value in logps.detach().float().cpu().tolist()]


def _global_max(accelerator: Any, value: int, device: Any) -> int:
    """Return the max of ``value`` across all ranks (identity for single process)."""

    if int(getattr(accelerator, "num_processes", 1)) <= 1 or not hasattr(accelerator, "gather"):
        return int(value)
    gathered = accelerator.gather(torch.tensor([int(value)], device=device))
    return int(gathered.max().item())


def _refresh_cache_if_needed(
    config: Any,
    sequences: list[dict[str, Any]],
    reusable: dict[str, float],
    signature: str,
    model_source: Any,
    results: list[Any],
) -> None:
    """Rewrite the manifest/parquet when the on-disk cache is absent or stale.

    Reached only when every required sequence is already covered by reusable
    entries. If the existing manifest already matches the current signature and
    covers exactly the current sequences, nothing is rewritten.
    """

    logger = get_logger("precompute_ref")
    current_hashes = {seq["render_hash"] for seq in sequences}
    entries = {render_hash: logp for render_hash, logp in reusable.items() if render_hash in current_hashes}

    manifest = read_cache_manifest(ref_cache_paths(config))
    already_current = (
        manifest is not None
        and manifest.get("signature") == signature
        and int(manifest.get("num_entries") or -1) == len(entries)
    )
    if already_current:
        logger.info("reference cache manifest already current; leaving it untouched")
        return

    write_ref_logp_cache(
        config,
        entries=entries,
        signature=signature,
        model_source=model_source,
        splits=[result.split for result in results],
    )


def _resolve_pad_token_id(tokenizer: Any) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    raise ValueError("tokenizer must define pad_token_id or eos_token_id for reference padding")


if __name__ == "__main__":
    main()

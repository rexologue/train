from __future__ import annotations

import math
from typing import Any, Iterable

import torch


def run_standard_eval(
    *,
    model: Any,
    dataloader: Iterable[dict[str, Any]] | None,
    trainer: Any,
    config: Any,
    accelerator: Any | None = None,
) -> dict[str, float]:
    if dataloader is None:
        return {}

    max_batches = _max_batches(config)
    loss_weighted_sum = 0.0
    loss_count = 0
    batch_count = 0
    supervised_tokens = 0
    total_tokens = 0

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch_count += 1
            loss = trainer.compute_loss(model, batch)
            loss_value = _to_float(loss)
            labels = batch.get("labels")
            batch_supervised_tokens = 0
            if labels is not None:
                batch_supervised_tokens = int((labels != config.ignore_index).sum().item())
                supervised_tokens += batch_supervised_tokens
                total_tokens += int(labels.numel())
            elif "chosen_labels" in batch and "rejected_labels" in batch:
                batch_supervised_tokens = int(
                    (batch["chosen_labels"] != config.ignore_index).sum().item()
                    + (batch["rejected_labels"] != config.ignore_index).sum().item()
                )
                supervised_tokens += batch_supervised_tokens
                total_tokens += int(batch["chosen_labels"].numel() + batch["rejected_labels"].numel())
            if labels is None and "chosen_labels" not in batch:
                loss_weighted_sum += loss_value
            elif "chosen_labels" in batch:
                sample_id = batch.get("sample_id")
                batch_pairs = len(sample_id) if isinstance(sample_id, list) else int(batch["chosen_labels"].shape[0])
                loss_weighted_sum += loss_value * batch_pairs
                loss_count += max(batch_pairs - 1, 0)
            elif batch_supervised_tokens > 0:
                loss_weighted_sum += loss_value * batch_supervised_tokens
                loss_count += max(batch_supervised_tokens - 1, 0)
            loss_count += 1

    if was_training and hasattr(model, "train"):
        model.train()

    local = {
        "loss_weighted_sum": loss_weighted_sum,
        "loss_count": loss_count,
        "batch_count": batch_count,
        "supervised_tokens": supervised_tokens,
        "tokens": total_tokens,
    }
    reduced = _reduce_eval_totals(local, accelerator)
    if reduced["loss_count"] <= 0:
        return {}

    loss = reduced["loss_weighted_sum"] / reduced["loss_count"]
    return {
        "eval/loss": loss,
        "eval/ppl": math.exp(min(loss, 20.0)),
        "eval/batches": float(reduced["batch_count"]),
        "eval/tokens": float(reduced["tokens"]),
        "eval/supervised_tokens": float(reduced["supervised_tokens"]),
    }


def _max_batches(config: Any) -> int | None:
    return config.eval.standard.max_batches


def _to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _reduce_eval_totals(local: dict[str, float | int], accelerator: Any | None) -> dict[str, float]:
    if accelerator is None:
        return {key: float(value) for key, value in local.items()}

    device = accelerator.device
    values = torch.tensor(
        [
            float(local["loss_weighted_sum"]),
            float(local["loss_count"]),
            float(local["batch_count"]),
            float(local["supervised_tokens"]),
            float(local["tokens"]),
        ],
        device=device,
    )
    gathered = accelerator.gather_for_metrics(values)
    if gathered.ndim == 1:
        totals = gathered.reshape(-1, 5).sum(dim=0)
    else:
        totals = gathered.sum(dim=0)
    return {
        "loss_weighted_sum": float(totals[0].item()),
        "loss_count": float(totals[1].item()),
        "batch_count": float(totals[2].item()),
        "supervised_tokens": float(totals[3].item()),
        "tokens": float(totals[4].item()),
    }

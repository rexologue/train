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
    route_totals = _empty_route_totals()

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
            loss_kind = str(batch.get("loss_kind") or "")
            labels = batch.get("labels")
            batch_supervised_tokens = 0
            if labels is not None:
                batch_supervised_tokens = int((labels != config.ignore_index).sum().item())
                supervised_tokens += batch_supervised_tokens
                total_tokens += int(labels.numel())
                if loss_kind in {"sft_target", "sft_tool"} and batch_supervised_tokens > 0:
                    _add_weighted_route_loss(route_totals, "sft", loss_value, batch_supervised_tokens)
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
                _add_weighted_route_loss(route_totals, "dpo", loss_value, batch_pairs)
                _add_weighted_route_metric(
                    route_totals,
                    "dpo",
                    "accuracy",
                    trainer.last_loss_metrics.get("dpo/accuracy"),
                    batch_pairs,
                )
                _add_weighted_route_metric(
                    route_totals,
                    "dpo",
                    "reward_margin",
                    trainer.last_loss_metrics.get("dpo/reward_margin"),
                    batch_pairs,
                )
            elif batch_supervised_tokens > 0:
                loss_weighted_sum += loss_value * batch_supervised_tokens
                loss_count += max(batch_supervised_tokens - 1, 0)
            loss_count += 1
            _add_route_counts(route_totals, loss_kind, batch, batch_supervised_tokens)

    if was_training and hasattr(model, "train"):
        model.train()

    local = {
        "loss_weighted_sum": loss_weighted_sum,
        "loss_count": loss_count,
        "batch_count": batch_count,
        "supervised_tokens": supervised_tokens,
        "tokens": total_tokens,
        **route_totals,
    }
    reduced = _reduce_eval_totals(local, accelerator)
    if reduced["loss_count"] <= 0:
        return {}

    metrics = {
        "eval/batches": float(reduced["batch_count"]),
        "eval/tokens": float(reduced["tokens"]),
        "eval/supervised_tokens": float(reduced["supervised_tokens"]),
    }

    # An aggregate eval/loss only has meaning when a single objective is present.
    # SFT cross-entropy (per token) and DPO logsigmoid (per pair) live on
    # different scales, so blending them into one eval/loss/eval/ppl is a
    # category error. When both routes appear, expose only the per-route metrics
    # below, which are computed on their own correct denominators.
    has_sft = reduced["sft_loss_count"] > 0
    has_dpo = reduced["dpo_loss_count"] > 0
    if has_sft != has_dpo:
        loss = reduced["loss_weighted_sum"] / reduced["loss_count"]
        metrics["eval/loss"] = loss
        if has_sft:
            metrics["eval/ppl"] = math.exp(min(loss, 20.0))

    metrics.update(_route_metrics(reduced))
    return metrics


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


def _empty_route_totals() -> dict[str, float]:
    return {
        "sft_loss_weighted_sum": 0.0,
        "sft_loss_count": 0.0,
        "sft_batch_count": 0.0,
        "sft_supervised_tokens": 0.0,
        "sft_tokens": 0.0,
        "dpo_loss_weighted_sum": 0.0,
        "dpo_loss_count": 0.0,
        "dpo_accuracy_weighted_sum": 0.0,
        "dpo_accuracy_count": 0.0,
        "dpo_reward_margin_weighted_sum": 0.0,
        "dpo_reward_margin_count": 0.0,
        "dpo_batch_count": 0.0,
        "dpo_pairs": 0.0,
        "dpo_supervised_tokens": 0.0,
        "dpo_tokens": 0.0,
    }


def _add_weighted_route_loss(totals: dict[str, float], route: str, value: float, weight: int) -> None:
    if weight <= 0:
        return
    totals[f"{route}_loss_weighted_sum"] += float(value) * float(weight)
    totals[f"{route}_loss_count"] += float(weight)


def _add_weighted_route_metric(
    totals: dict[str, float],
    route: str,
    metric: str,
    value: Any,
    weight: int,
) -> None:
    if value is None or weight <= 0:
        return
    totals[f"{route}_{metric}_weighted_sum"] += float(value) * float(weight)
    totals[f"{route}_{metric}_count"] += float(weight)


def _add_route_counts(
    totals: dict[str, float],
    loss_kind: str,
    batch: dict[str, Any],
    supervised_tokens: int,
) -> None:
    if loss_kind in {"sft_target", "sft_tool"}:
        totals["sft_batch_count"] += 1.0
        totals["sft_supervised_tokens"] += float(supervised_tokens)
        labels = batch.get("labels")
        if labels is not None:
            totals["sft_tokens"] += float(labels.numel())
    elif loss_kind == "dpo_target":
        totals["dpo_batch_count"] += 1.0
        sample_id = batch.get("sample_id")
        pairs = len(sample_id) if isinstance(sample_id, list) else int(batch["chosen_labels"].shape[0])
        totals["dpo_pairs"] += float(pairs)
        totals["dpo_supervised_tokens"] += float(supervised_tokens)
        totals["dpo_tokens"] += float(batch["chosen_labels"].numel() + batch["rejected_labels"].numel())


def _route_metrics(reduced: dict[str, float]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if reduced["sft_loss_count"] > 0:
        sft_loss = reduced["sft_loss_weighted_sum"] / reduced["sft_loss_count"]
        metrics.update(
            {
                "eval/sft/loss": sft_loss,
                "eval/sft/ppl": math.exp(min(sft_loss, 20.0)),
                "eval/sft/batches": float(reduced["sft_batch_count"]),
                "eval/sft/tokens": float(reduced["sft_tokens"]),
                "eval/sft/supervised_tokens": float(reduced["sft_supervised_tokens"]),
            }
        )
    if reduced["dpo_loss_count"] > 0:
        metrics.update(
            {
                "eval/dpo/loss": reduced["dpo_loss_weighted_sum"] / reduced["dpo_loss_count"],
                "eval/dpo/batches": float(reduced["dpo_batch_count"]),
                "eval/dpo/pairs": float(reduced["dpo_pairs"]),
                "eval/dpo/tokens": float(reduced["dpo_tokens"]),
                "eval/dpo/supervised_tokens": float(reduced["dpo_supervised_tokens"]),
            }
        )
    if reduced["dpo_accuracy_count"] > 0:
        metrics["eval/dpo/accuracy"] = reduced["dpo_accuracy_weighted_sum"] / reduced["dpo_accuracy_count"]
    if reduced["dpo_reward_margin_count"] > 0:
        metrics["eval/dpo/reward_margin"] = reduced["dpo_reward_margin_weighted_sum"] / reduced["dpo_reward_margin_count"]
    return metrics


def _reduce_eval_totals(local: dict[str, float | int], accelerator: Any | None) -> dict[str, float]:
    if accelerator is None:
        return {key: float(value) for key, value in local.items()}

    device = accelerator.device
    keys = sorted(local)
    values = torch.tensor([float(local[key]) for key in keys], device=device)
    gathered = accelerator.gather_for_metrics(values)
    if gathered.ndim == 1:
        totals = gathered.reshape(-1, len(keys)).sum(dim=0)
    else:
        totals = gathered.sum(dim=0)
    return {key: float(totals[index].item()) for index, key in enumerate(keys)}

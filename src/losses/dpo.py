from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DpoLossResult:
    """DPO loss plus detached diagnostics for route-level metrics."""

    loss: Any
    metrics: dict[str, float]


def sequence_logps(
    model: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
    labels: Any,
    ignore_index: int,
) -> Any:
    """Return summed log probabilities over label-selected completion tokens."""

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
    shift_logits = logits[:, :-1, :]
    shift_input_ids = input_ids[:, 1:]
    shift_labels = labels[:, 1:]
    loss_mask = shift_labels != int(ignore_index)
    token_logps = F.log_softmax(shift_logits, dim=-1)
    gathered = token_logps.gather(dim=-1, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)
    return (gathered.float() * loss_mask).sum(dim=-1)


def dpo_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    beta: float,
    ignore_index: int,
    accelerator: Any | None = None,
    reference_mode: str = "disable_adapter",
    cache_required: bool = False,
) -> DpoLossResult:
    """Compute one DPO batch loss with cached or online reference logprobs."""

    policy_chosen_logp = sequence_logps(
        model,
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch.get("chosen_attention_mask"),
        labels=batch["chosen_labels"],
        ignore_index=ignore_index,
    )
    policy_rejected_logp = sequence_logps(
        model,
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch.get("rejected_attention_mask"),
        labels=batch["rejected_labels"],
        ignore_index=ignore_index,
    )
    ref_chosen_logp, ref_rejected_logp = reference_logps(
        model,
        batch,
        ignore_index=ignore_index,
        accelerator=accelerator,
        reference_mode=reference_mode,
        cache_required=cache_required,
    )

    policy_logratio = policy_chosen_logp - policy_rejected_logp
    ref_logratio = ref_chosen_logp - ref_rejected_logp
    logits = float(beta) * (policy_logratio - ref_logratio)
    losses = -F.logsigmoid(logits)
    chosen_rewards = float(beta) * (policy_chosen_logp - ref_chosen_logp)
    rejected_rewards = float(beta) * (policy_rejected_logp - ref_rejected_logp)
    reward_margin = chosen_rewards - rejected_rewards
    metrics = {
        "dpo/policy_chosen_logp": tensor_mean(policy_chosen_logp),
        "dpo/policy_rejected_logp": tensor_mean(policy_rejected_logp),
        "dpo/ref_chosen_logp": tensor_mean(ref_chosen_logp),
        "dpo/ref_rejected_logp": tensor_mean(ref_rejected_logp),
        "dpo/reward_chosen": tensor_mean(chosen_rewards),
        "dpo/reward_rejected": tensor_mean(rejected_rewards),
        "dpo/reward_margin": tensor_mean(reward_margin),
        "dpo/accuracy": tensor_mean(chosen_rewards > rejected_rewards),
    }
    return DpoLossResult(loss=losses.mean(), metrics=metrics)


def reference_logps(
    model: Any,
    batch: dict[str, Any],
    *,
    ignore_index: int,
    accelerator: Any | None,
    reference_mode: str,
    cache_required: bool,
) -> tuple[Any, Any]:
    """Return reference logprobs from batch cache or by disabling the active adapter."""

    cached = cached_reference_logps(batch)
    if cached is not None:
        return cached
    if cache_required:
        raise ValueError("DPO reference logprob cache is required but missing from batch")
    if reference_mode != "disable_adapter":
        raise ValueError(f"unsupported DPO reference mode: {reference_mode!r}")

    disable_context = disable_adapter_context(model, accelerator)
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad(), disable_context:
            chosen = sequence_logps(
                model,
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch.get("chosen_attention_mask"),
                labels=batch["chosen_labels"],
                ignore_index=ignore_index,
            )
            rejected = sequence_logps(
                model,
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch.get("rejected_attention_mask"),
                labels=batch["rejected_labels"],
                ignore_index=ignore_index,
            )
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    return chosen, rejected


def cached_reference_logps(batch: dict[str, Any]) -> tuple[Any, Any] | None:
    """Return cached reference logprobs when both sides are present."""

    chosen = batch.get("chosen_ref_logp")
    rejected = batch.get("rejected_ref_logp")
    if chosen is None or rejected is None:
        return None
    return chosen, rejected


def disable_adapter_context(model: Any, accelerator: Any | None) -> Any:
    """Return a PEFT disable-adapter context without bypassing wrapped forward calls."""

    unwrapped = accelerator.unwrap_model(model) if accelerator is not None and hasattr(accelerator, "unwrap_model") else model
    if not hasattr(unwrapped, "disable_adapter"):
        raise TypeError("DPO reference_mode=disable_adapter requires a PEFT model with disable_adapter()")
    context = unwrapped.disable_adapter()
    return context if context is not None else nullcontext()


def tensor_mean(value: Any) -> float:
    """Return a detached float mean for metrics."""

    if isinstance(value, bool):
        return float(value)
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "mean"):
        value = value.mean()
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)

from __future__ import annotations

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

    shift_labels = labels[:, 1:]
    loss_mask = shift_labels != int(ignore_index)
    keep_positions = selected_logit_positions(loss_mask)
    if keep_positions.numel() == 0:
        return input_ids.new_zeros((input_ids.shape[0],), dtype=torch.float32)

    outputs = call_model_for_sequence_logps(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=keep_positions,
    )
    logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
    shift_logits, shift_input_ids, loss_mask = align_kept_logits(
        logits=logits,
        input_ids=input_ids,
        loss_mask=loss_mask,
        keep_positions=keep_positions,
    )
    target_logits = shift_logits.gather(dim=-1, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)
    log_normalizers = torch.logsumexp(shift_logits, dim=-1)
    gathered = target_logits - log_normalizers
    return (gathered.float() * loss_mask).sum(dim=-1)


def selected_logit_positions(loss_mask: Any) -> Any:
    """Return original-logit positions required for any selected label in the batch."""

    selected_columns = torch.nonzero(loss_mask.any(dim=0), as_tuple=False).flatten()
    return selected_columns.to(device=loss_mask.device, dtype=torch.long)


def call_model_for_sequence_logps(
    model: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
    logits_to_keep: Any,
) -> Any:
    """Call model with compact logits when supported, fallback for older test doubles/models."""

    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "logits_to_keep": logits_to_keep,
    }
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "logits_to_keep" not in str(exc):
            raise
        kwargs.pop("logits_to_keep")
        return model(**kwargs)


def align_kept_logits(
    *,
    logits: Any,
    input_ids: Any,
    loss_mask: Any,
    keep_positions: Any,
) -> tuple[Any, Any, Any]:
    """Align compact `logits_to_keep` outputs with target ids and loss mask.

    Qwen-style models support `logits_to_keep=<positions>` and return logits
    only for those original hidden-state positions. Some test doubles or older
    models may ignore the kwarg and return full sequence logits; keep that path
    working without changing the DPO math.
    """

    if logits.shape[1] == keep_positions.numel():
        target_positions = keep_positions + 1
        return logits, input_ids[:, target_positions], loss_mask[:, keep_positions]

    return logits[:, :-1, :], input_ids[:, 1:], loss_mask


def dpo_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    beta: float,
    ignore_index: int,
    accelerator: Any | None = None,
    cache_required: bool = False,
) -> DpoLossResult:
    """Compute one DPO batch loss with precomputed reference logprobs."""

    del accelerator, cache_required

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
) -> tuple[Any, Any]:
    """Return precomputed reference logprobs attached to the DPO batch."""

    del model, ignore_index
    cached = cached_reference_logps(batch)
    if cached is None:
        raise ValueError(
            "DPO reference logprobs are missing from batch; run reference precompute with "
            "loss_routing.dpo.reference.cache_enabled=true before training"
        )
    return cached


def cached_reference_logps(batch: dict[str, Any]) -> tuple[Any, Any] | None:
    """Return cached reference logprobs when both sides are present."""

    chosen = batch.get("chosen_ref_logp")
    rejected = batch.get("rejected_ref_logp")
    if chosen is None or rejected is None:
        return None
    return chosen, rejected


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

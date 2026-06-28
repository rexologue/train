from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
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

    Some causal LM implementations support `logits_to_keep=<positions>` and
    return logits only for those original hidden-state positions. Some test
    doubles or older models may ignore the kwarg and return full sequence
    logits; keep that path working without changing the DPO math.
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
) -> DpoLossResult:
    """Compute one DPO batch with an on-the-fly PEFT reference policy.

    There is intentionally no reference-logprob cache and no second reference
    model. The policy forward uses the active LoRA adapter. The reference forward
    runs the same FSDP-wrapped model under ``no_grad`` with PEFT adapters disabled.
    """

    policy_chosen_logp, policy_rejected_logp = concatenated_dpo_logps(
        model,
        batch,
        ignore_index=ignore_index,
    )

    with torch.no_grad(), peft_adapter_disabled(model, accelerator=accelerator):
        ref_chosen_logp, ref_rejected_logp = concatenated_dpo_logps(
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


def concatenated_dpo_logps(model: Any, batch: dict[str, Any], *, ignore_index: int) -> tuple[Any, Any]:
    """Run chosen and rejected continuations in one forward pass."""

    chosen_input_ids = batch["chosen_input_ids"]
    rejected_input_ids = batch["rejected_input_ids"]
    chosen_size = int(chosen_input_ids.shape[0])
    rejected_size = int(rejected_input_ids.shape[0])
    if chosen_size != rejected_size:
        raise ValueError(f"DPO chosen/rejected batch sizes differ: {chosen_size} != {rejected_size}")

    input_ids = concat_pair_padded(
        chosen_input_ids,
        rejected_input_ids,
        pad_value=0,
    )
    attention_mask = concat_pair_padded(
        batch["chosen_attention_mask"],
        batch["rejected_attention_mask"],
        pad_value=0,
    )
    labels = concat_pair_padded(
        batch["chosen_labels"],
        batch["rejected_labels"],
        pad_value=int(ignore_index),
    )

    logps = sequence_logps(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        ignore_index=ignore_index,
    )
    return logps[:chosen_size], logps[chosen_size:]


def concat_pair_padded(chosen: Any, rejected: Any, *, pad_value: int) -> Any:
    """Pad two [batch, seq] tensors to a shared length and concatenate by batch."""

    target_len = max(int(chosen.shape[1]), int(rejected.shape[1]))
    return torch.cat(
        [
            pad_to_length(chosen, target_len, pad_value=pad_value),
            pad_to_length(rejected, target_len, pad_value=pad_value),
        ],
        dim=0,
    )


def pad_to_length(tensor: Any, target_len: int, *, pad_value: int) -> Any:
    """Right-pad a [batch, seq] tensor to target_len."""

    if int(tensor.shape[1]) == int(target_len):
        return tensor
    if int(tensor.shape[1]) > int(target_len):
        raise ValueError(f"cannot pad seq_len={tensor.shape[1]} down to target_len={target_len}")
    pad_width = int(target_len) - int(tensor.shape[1])
    return F.pad(tensor, (0, pad_width), value=pad_value)


def peft_adapter_disabled(model: Any, *, accelerator: Any | None = None) -> Any:
    """Return PEFT's adapter-disable context for wrapped or unwrapped models."""

    target = find_disable_adapter_owner(model, accelerator=accelerator)
    if target is None:
        raise RuntimeError(
            "DPO on-the-fly reference requires a PEFT LoRA model with disable_adapter(). "
            "Build the model through PEFT/get_peft_model before FSDP wrapping."
        )
    return target.disable_adapter()


def find_disable_adapter_owner(model: Any, *, accelerator: Any | None = None) -> Any | None:
    """Find the object that owns PEFT's disable_adapter context method."""

    candidates: list[Any] = []
    if accelerator is not None and hasattr(accelerator, "unwrap_model"):
        try:
            candidates.append(accelerator.unwrap_model(model))
        except Exception:
            pass
    candidates.append(model)

    seen: set[int] = set()
    while candidates:
        candidate = candidates.pop(0)
        if candidate is None:
            continue
        marker = id(candidate)
        if marker in seen:
            continue
        seen.add(marker)
        if callable(getattr(candidate, "disable_adapter", None)):
            return candidate
        for attr in ("module", "model", "base_model"):
            child = getattr(candidate, attr, None)
            if child is not None:
                candidates.append(child)
    return None


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

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from losses.dpo import call_model_for_sequence_logps, selected_logit_positions


def sft_cross_entropy_loss(model: Any, batch: dict[str, Any], *, ignore_index: int = -100) -> Any:
    """Run the SFT route and apply the project-owned label mask."""

    shift_labels = batch["labels"][:, 1:]
    loss_mask = shift_labels != int(ignore_index)
    keep_positions = selected_logit_positions(loss_mask)
    if keep_positions.numel() == 0:
        return batch["input_ids"].new_zeros((), dtype=torch.float32)

    outputs = call_model_for_sequence_logps(
        model,
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        logits_to_keep=keep_positions,
    )
    logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
    if logits.shape[1] == keep_positions.numel():
        shift_logits = logits
        shift_labels = batch["labels"][:, keep_positions + 1]
        loss_mask = loss_mask[:, keep_positions]
    else:
        shift_logits = logits[:, :-1, :]

    return F.cross_entropy(shift_logits[loss_mask].float(), shift_labels[loss_mask])

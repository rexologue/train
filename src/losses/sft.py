from __future__ import annotations

from typing import Any

import torch.nn.functional as F


def sft_cross_entropy_loss(model: Any, batch: dict[str, Any], *, ignore_index: int = -100) -> Any:
    """Run the SFT route and apply the project-owned label mask."""

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
    )
    logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]).float(),
        shift_labels.view(-1),
        ignore_index=int(ignore_index),
    )

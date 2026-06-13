from __future__ import annotations

from typing import Any


def sft_cross_entropy_loss(model: Any, batch: dict[str, Any]) -> Any:
    """Run the causal-LM SFT route on tensor fields only."""

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        labels=batch["labels"],
    )
    if hasattr(outputs, "loss"):
        return outputs.loss
    if isinstance(outputs, dict) and "loss" in outputs:
        return outputs["loss"]
    raise TypeError("model output must expose a loss for the SFT route")

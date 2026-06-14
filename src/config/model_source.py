from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.schema import TrainingConfig


def effective_model_id(config: TrainingConfig) -> str:
    """Return the registry-resolved model directory used by all loaders."""

    resolved = config.section("model").get("resolved_model_id")
    if not resolved:
        raise RuntimeError("registry model must be resolved before loading model or tokenizer")
    return str(resolved)

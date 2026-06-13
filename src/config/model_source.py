from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config.schema import TrainingConfig


def effective_model_id(config: TrainingConfig) -> str:
    """Return the model path/id that downstream loaders should use."""

    model = config.section("model")
    resolved = model.get("resolved_model_id")
    if resolved:
        return str(resolved)

    source = model.get("source")
    if isinstance(source, dict) and source.get("kind", "local_or_hf") == "local_or_hf":
        local_dir = source.get("local_dir")
        if local_dir:
            return str(Path(str(local_dir)).expanduser().resolve())
    return str(model["base_model_id"])


def tokenizer_source_mode(tokenizer_config: dict[str, Any]) -> str:
    """Return tokenizer source mode with backward-compatible defaulting."""

    configured = tokenizer_config.get("source")
    if configured:
        return str(configured)
    return "explicit" if tokenizer_config.get("tokenizer_id") else "model"


def effective_tokenizer_id(config: TrainingConfig) -> str:
    """Return the tokenizer path/id according to tokenizer.source."""

    tokenizer = config.section("tokenizer")
    mode = tokenizer_source_mode(tokenizer)
    if mode == "model":
        return effective_model_id(config)
    if mode == "explicit":
        tokenizer_id = tokenizer.get("tokenizer_id")
        if not tokenizer_id:
            raise ValueError("tokenizer.tokenizer_id must be configured when tokenizer.source=explicit")
        return str(tokenizer_id)
    raise ValueError("tokenizer.source must be model or explicit")

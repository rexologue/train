from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUIRED_TOP_LEVEL_KEYS = {
    "project",
    "model",
    "tokenizer",
    "preprocessing",
    "loss_routing",
    "training",
    "checkpointing",
    "registry",
}


class ConfigError(ValueError):
    """Raised when YAML config is missing training-critical settings."""


@dataclass(frozen=True)
class TrainingConfig:
    """Thin validated wrapper around the loaded YAML mapping."""

    raw: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"Config section {name!r} must be a mapping")
        return value

    @property
    def preprocessing(self) -> dict[str, Any]:
        return self.section("preprocessing")

    @property
    def ignore_index(self) -> int:
        return int(self.preprocessing["masking"].get("ignore_index", -100))

    @property
    def reasoning(self) -> dict[str, Any]:
        return self.preprocessing["reasoning"]

    @property
    def rendering(self) -> dict[str, Any]:
        return self.preprocessing["rendering"]

    @property
    def sequence(self) -> dict[str, Any]:
        return self.preprocessing["sequence"]

    @property
    def masking_policies(self) -> dict[str, Any]:
        policies = self.preprocessing["masking"].get("policies")
        if not isinstance(policies, dict):
            raise ConfigError("preprocessing.masking.policies must be configured")
        return policies


def validate_config(raw: dict[str, Any]) -> TrainingConfig:
    """Validate invariants that can silently break preprocessing/training."""

    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(raw))
    if missing:
        raise ConfigError(f"Missing required top-level config sections: {missing}")

    preprocessing = raw["preprocessing"]
    allowed_preprocessing_keys = {"raw", "output", "sequence", "rendering", "reasoning", "masking"}
    unknown_preprocessing_keys = sorted(set(preprocessing) - allowed_preprocessing_keys)
    if unknown_preprocessing_keys:
        raise ConfigError(f"Unknown preprocessing config sections: {unknown_preprocessing_keys}")
    for key in sorted(allowed_preprocessing_keys):
        if not isinstance(preprocessing.get(key), dict):
            raise ConfigError(f"preprocessing.{key} must be configured")

    if not isinstance(preprocessing["reasoning"].get("enable_thinking"), bool):
        raise ConfigError("preprocessing.reasoning.enable_thinking must be true or false")

    model = raw["model"]
    if model.get("use_fp8_base") is not False:
        raise ConfigError("model.use_fp8_base must be false for the quality-first LoRA path")

    sequence = preprocessing["sequence"]
    if sequence.get("truncation") is not False:
        raise ConfigError("preprocessing.sequence.truncation must stay false until explicit turn-aware policy exists")
    if sequence.get("packing") is not False:
        raise ConfigError("preprocessing.sequence.packing must stay false until packing mask tests exist")

    training = raw["training"]
    if "drop_last" in training and not isinstance(training["drop_last"], bool):
        raise ConfigError("training.drop_last must be true or false")

    registry = raw["registry"]
    if registry.get("promote_best_to") is not None:
        raise ConfigError("registry.promote_best_to must be null during ordinary training")
    if not registry.get("candidate_alias_template"):
        raise ConfigError("registry.candidate_alias_template must be explicit")

    return TrainingConfig(raw)

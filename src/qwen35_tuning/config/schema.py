from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUIRED_TOP_LEVEL_KEYS = {
    "project",
    "model",
    "tokenizer",
    "reasoning",
    "data",
    "rendering",
    "masking",
    "loss_routing",
    "training",
    "checkpointing",
    "registry",
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingConfig:
    raw: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"Config section {name!r} must be a mapping")
        return value

    @property
    def ignore_index(self) -> int:
        return int(self.section("masking").get("ignore_index", -100))

    @property
    def reasoning(self) -> dict[str, Any]:
        return self.section("reasoning")

    @property
    def masking_policies(self) -> dict[str, Any]:
        policies = self.section("masking").get("policies")
        if not isinstance(policies, dict):
            raise ConfigError("masking.policies must be configured")
        return policies


def validate_config(raw: dict[str, Any]) -> TrainingConfig:
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(raw))
    if missing:
        raise ConfigError(f"Missing required top-level config sections: {missing}")

    reasoning = raw["reasoning"]
    if reasoning.get("enabled") is not False:
        raise ConfigError("reasoning.enabled must be false for this project")
    if reasoning.get("confirm_disabled_during_preprocessing") is not True:
        raise ConfigError("reasoning.confirm_disabled_during_preprocessing must be true")
    if reasoning.get("fail_if_supervised_think_tokens") is not True:
        raise ConfigError("reasoning.fail_if_supervised_think_tokens must be true")

    model = raw["model"]
    if model.get("use_fp8_base") is not False:
        raise ConfigError("model.use_fp8_base must be false for the quality-first LoRA path")

    data = raw["data"]
    if data.get("truncation") is not False:
        raise ConfigError("data.truncation must stay false until explicit turn-aware policy exists")
    if data.get("packing") is not False:
        raise ConfigError("data.packing must stay false until packing mask tests exist")

    registry = raw["registry"]
    if registry.get("promote_best_to") is not None:
        raise ConfigError("registry.promote_best_to must be null during ordinary training")
    if not registry.get("candidate_alias_template"):
        raise ConfigError("registry.candidate_alias_template must be explicit")

    return TrainingConfig(raw)


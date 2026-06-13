from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.model_source import tokenizer_source_mode


REQUIRED_TOP_LEVEL_KEYS = {
    "project",
    "model",
    "tokenizer",
    "preprocessing",
    "loss_routing",
    "training",
    "checkpointing",
    "mlflow",
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

    @property
    def mlflow(self) -> dict[str, Any]:
        return self.section("mlflow")


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
    validate_model_source(model)

    tokenizer = raw["tokenizer"]
    validate_tokenizer_source(tokenizer)

    sequence = preprocessing["sequence"]
    if sequence.get("truncation") is not False:
        raise ConfigError("preprocessing.sequence.truncation must stay false until explicit turn-aware policy exists")
    if sequence.get("packing") is not False:
        raise ConfigError("preprocessing.sequence.packing must stay false until packing mask tests exist")

    training = raw["training"]
    if "drop_last" in training and not isinstance(training["drop_last"], bool):
        raise ConfigError("training.drop_last must be true or false")

    mlflow = raw["mlflow"]
    if not isinstance(mlflow.get("enabled"), bool):
        raise ConfigError("mlflow.enabled must be true or false")
    if mlflow.get("enabled"):
        if not mlflow.get("tracking_uri"):
            raise ConfigError("mlflow.tracking_uri must be configured when mlflow.enabled=true")
        if not mlflow.get("experiment_name"):
            raise ConfigError("mlflow.experiment_name must be configured when mlflow.enabled=true")

    registry = raw["registry"]
    if registry.get("promote_best_to") is not None:
        raise ConfigError("registry.promote_best_to must be null during ordinary training")
    if not registry.get("candidate_alias_template"):
        raise ConfigError("registry.candidate_alias_template must be explicit")

    lineage = raw.get("lineage")
    if lineage is not None:
        validate_lineage(lineage)

    return TrainingConfig(raw)


def validate_model_source(model: dict[str, Any]) -> None:
    """Validate explicit model source fields without resolving remote state."""

    source = model.get("source")
    if source is None:
        return
    if not isinstance(source, dict):
        raise ConfigError("model.source must be a mapping")
    kind = source.get("kind", "local_or_hf")
    if kind not in {"local_or_hf", "registry"}:
        raise ConfigError("model.source.kind must be local_or_hf or registry")
    if kind == "local_or_hf":
        if "local_dir" in source and source["local_dir"] in ("", None):
            source["local_dir"] = None
        return

    if not source.get("model_name"):
        raise ConfigError("model.source.model_name must be configured for registry sources")
    has_alias = source.get("alias") not in (None, "")
    has_version = source.get("version") not in (None, "")
    if has_alias == has_version:
        raise ConfigError("model.source must configure exactly one of alias or version")
    if not source.get("local_dir"):
        raise ConfigError("model.source.local_dir must be configured for registry sources")
    if source.get("pull_policy", "if_local_empty") != "if_local_empty":
        raise ConfigError("model.source.pull_policy must be if_local_empty")
    for key in ("verify_local_hash", "verify_remote_ref", "require_registry_metadata"):
        if key in source and not isinstance(source[key], bool):
            raise ConfigError(f"model.source.{key} must be true or false")


def validate_lineage(lineage: dict[str, Any]) -> None:
    """Validate optional tracking lineage config."""

    if not isinstance(lineage, dict):
        raise ConfigError("lineage must be a mapping")
    dvc = lineage.get("dvc")
    if dvc is None:
        return
    if not isinstance(dvc, dict):
        raise ConfigError("lineage.dvc must be a mapping")
    if not isinstance(dvc.get("enabled"), bool):
        raise ConfigError("lineage.dvc.enabled must be true or false")
    if dvc.get("enabled"):
        if not dvc.get("repo_root"):
            raise ConfigError("lineage.dvc.repo_root must be configured when enabled")
        if not isinstance(dvc.get("targets"), dict) or not dvc["targets"]:
            raise ConfigError("lineage.dvc.targets must be a non-empty mapping when enabled")


def validate_tokenizer_source(tokenizer: dict[str, Any]) -> None:
    """Validate tokenizer source selection while keeping tokenizer params separate."""

    mode = tokenizer_source_mode(tokenizer)
    if mode not in {"model", "explicit"}:
        raise ConfigError("tokenizer.source must be model or explicit")
    if mode == "explicit" and not tokenizer.get("tokenizer_id"):
        raise ConfigError("tokenizer.tokenizer_id must be configured when tokenizer.source=explicit")

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
    "distributed",
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
    if "enabled" in training and not isinstance(training["enabled"], bool):
        raise ConfigError("training.enabled must be true or false")
    if "drop_last" in training and not isinstance(training["drop_last"], bool):
        raise ConfigError("training.drop_last must be true or false")
    if "adamw_betas" in training:
        validate_adamw_betas(training["adamw_betas"])
    validate_lora(raw.get("lora"), training_enabled=bool(training.get("enabled", True)))

    eval_config = raw.get("eval")
    if eval_config is not None:
        validate_eval(eval_config)

    validate_distributed(raw["distributed"])

    checkpointing = raw["checkpointing"]
    if "save_every_n_validations" in checkpointing:
        _require_positive_int(checkpointing["save_every_n_validations"], "checkpointing.save_every_n_validations")
    if checkpointing.get("save_total_limit") is not None:
        _require_positive_int(checkpointing["save_total_limit"], "checkpointing.save_total_limit")

    mlflow = raw["mlflow"]
    if not isinstance(mlflow.get("enabled"), bool):
        raise ConfigError("mlflow.enabled must be true or false")
    if mlflow.get("enabled"):
        if not mlflow.get("tracking_uri"):
            raise ConfigError("mlflow.tracking_uri must be configured when mlflow.enabled=true")
        if not mlflow.get("experiment_name"):
            raise ConfigError("mlflow.experiment_name must be configured when mlflow.enabled=true")
    validate_mlflow_async_logging(mlflow)

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


def validate_lora(lora: Any, *, training_enabled: bool) -> None:
    if lora is None:
        return
    if not isinstance(lora, dict):
        raise ConfigError("lora must be a mapping")
    if not isinstance(lora.get("enabled"), bool):
        raise ConfigError("lora.enabled must be true or false")
    if not training_enabled or not lora.get("enabled"):
        return
    if lora.get("target_modules_policy") != "whitelist":
        raise ConfigError("lora.target_modules_policy must be whitelist")
    target_modules = lora.get("target_modules")
    if not isinstance(target_modules, list) or not target_modules or not all(isinstance(item, str) and item for item in target_modules):
        raise ConfigError("lora.target_modules must be a non-empty list of module names when training is enabled")


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


def validate_eval(eval_config: dict[str, Any]) -> None:
    """Validate evaluation cadence and route-local limits."""

    if not isinstance(eval_config, dict):
        raise ConfigError("eval must be a mapping")
    if "enabled" in eval_config and not isinstance(eval_config["enabled"], bool):
        raise ConfigError("eval.enabled must be true or false")
    if "every_train_steps" in eval_config:
        _require_positive_int(eval_config["every_train_steps"], "eval.every_train_steps")

    standard = eval_config.get("standard")
    if standard is not None:
        if not isinstance(standard, dict):
            raise ConfigError("eval.standard must be a mapping")
        if standard.get("max_batches") is not None:
            _require_positive_int(standard["max_batches"], "eval.standard.max_batches")

    bfcl = eval_config.get("bfcl")
    if bfcl is not None:
        if not isinstance(bfcl, dict):
            raise ConfigError("eval.bfcl must be a mapping")
        if bfcl.get("run_every_n_validations") is not None:
            _require_positive_int(bfcl["run_every_n_validations"], "eval.bfcl.run_every_n_validations")


def validate_distributed(distributed: dict[str, Any]) -> None:
    """Validate the single supported distributed runtime: Accelerate/FSDP."""

    if not isinstance(distributed, dict):
        raise ConfigError("distributed must be a mapping")
    fsdp = distributed.get("fsdp")
    if not isinstance(fsdp, dict):
        raise ConfigError("distributed.fsdp must be configured")
    for key in (
        "cpu_offload",
        "activation_checkpointing",
        "use_orig_params",
        "limit_all_gathers",
        "cpu_ram_efficient_loading",
        "sync_module_states",
    ):
        if key in fsdp and not isinstance(fsdp[key], bool):
            raise ConfigError(f"distributed.fsdp.{key} must be true or false")
    class_names = fsdp.get("transformer_cls_names_to_wrap")
    if class_names is not None:
        if not isinstance(class_names, list) or not all(isinstance(item, str) and item for item in class_names):
            raise ConfigError("distributed.fsdp.transformer_cls_names_to_wrap must be a list of strings")


def validate_mlflow_async_logging(mlflow: dict[str, Any]) -> None:
    """Validate optional asynchronous MLflow/registry logging settings."""

    async_logging = mlflow.get("async_logging")
    if async_logging is None:
        return
    if not isinstance(async_logging, dict):
        raise ConfigError("mlflow.async_logging must be a mapping")
    if not isinstance(async_logging.get("enabled"), bool):
        raise ConfigError("mlflow.async_logging.enabled must be true or false")
    if "queue_max_items" in async_logging:
        _require_positive_int(async_logging["queue_max_items"], "mlflow.async_logging.queue_max_items")
    if "flush_timeout_seconds" in async_logging:
        _require_positive_int(async_logging["flush_timeout_seconds"], "mlflow.async_logging.flush_timeout_seconds")
    if "fail_on_worker_error" in async_logging and not isinstance(async_logging["fail_on_worker_error"], bool):
        raise ConfigError("mlflow.async_logging.fail_on_worker_error must be true or false")


def validate_adamw_betas(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError("training.adamw_betas must be a two-item list")
    for index, beta in enumerate(value):
        try:
            parsed = float(beta)
        except (TypeError, ValueError) as exc:
            raise ConfigError("training.adamw_betas values must be floats") from exc
        if parsed < 0.0 or parsed >= 1.0:
            raise ConfigError(f"training.adamw_betas[{index}] must be >= 0 and < 1")


def validate_tokenizer_source(tokenizer: dict[str, Any]) -> None:
    """Validate tokenizer source selection while keeping tokenizer params separate."""

    mode = tokenizer_source_mode(tokenizer)
    if mode not in {"model", "explicit"}:
        raise ConfigError("tokenizer.source must be model or explicit")
    if mode == "explicit" and not tokenizer.get("tokenizer_id"):
        raise ConfigError("tokenizer.tokenizer_id must be configured when tokenizer.source=explicit")


def _require_positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return parsed

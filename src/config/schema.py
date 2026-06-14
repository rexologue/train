from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL_KEYS = {
    "project",
    "model",
    "tokenizer",
    "lora",
    "preprocessing",
    "loss_routing",
    "training",
    "distributed",
    "eval",
    "checkpointing",
    "mlflow",
    "registry",
}

REGISTRY_SELECTION_METRICS = {
    "eval/loss",
    "eval/ppl",
    "eval/batches",
    "eval/tokens",
    "eval/supervised_tokens",
    "eval/bfcl/accuracy",
    "eval/bfcl/total",
    "eval/bfcl/passed",
    "eval/bfcl/failed",
}


class ConfigError(ValueError):
    """Raised when YAML config is missing training-critical settings."""


@dataclass(frozen=True)
class TrainingConfig:
    """Validated project configuration with deterministic derived paths."""

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

    @property
    def output_dir(self) -> Path:
        return Path(str(self.section("project")["output_dir"])).expanduser()

    @property
    def pretokenized_dir(self) -> Path:
        return self.output_dir / "pretokenized"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def bfcl_rows_path(self) -> Path:
        return self.output_dir / "eval" / "bfcl_rows.jsonl"


def validate_config(raw: dict[str, Any]) -> TrainingConfig:
    """Validate the single supported production training contour."""

    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(raw))
    if missing:
        raise ConfigError(f"Missing required top-level config sections: {missing}")
    _reject_unknown(raw, REQUIRED_TOP_LEVEL_KEYS | {"progress"}, "top-level config")

    project = _mapping(raw["project"], "project")
    _reject_unknown(project, {"name", "run_name", "seed", "output_dir"}, "project")
    for key in ("name", "output_dir"):
        if not project.get(key):
            raise ConfigError(f"project.{key} must be configured")

    model = _mapping(raw["model"], "model")
    validate_model(model)

    tokenizer = _mapping(raw["tokenizer"], "tokenizer")
    _reject_unknown(tokenizer, {"use_fast", "add_special_tokens", "padding_side"}, "tokenizer")

    preprocessing = _mapping(raw["preprocessing"], "preprocessing")
    allowed_preprocessing_keys = {"raw", "sequence", "rendering", "reasoning", "masking"}
    _reject_unknown(preprocessing, allowed_preprocessing_keys, "preprocessing")
    for key in sorted(allowed_preprocessing_keys):
        _mapping(preprocessing.get(key), f"preprocessing.{key}")
    if not isinstance(preprocessing["reasoning"].get("enable_thinking"), bool):
        raise ConfigError("preprocessing.reasoning.enable_thinking must be true or false")
    sequence = preprocessing["sequence"]
    if sequence.get("truncation") is not False:
        raise ConfigError("preprocessing.sequence.truncation must stay false until explicit turn-aware policy exists")
    if sequence.get("packing") is not False:
        raise ConfigError("preprocessing.sequence.packing must stay false until packing mask tests exist")

    loss_routing = _mapping(raw["loss_routing"], "loss_routing")
    _reject_unknown(loss_routing, {"routes"}, "loss_routing")
    if not isinstance(loss_routing.get("routes"), dict) or not loss_routing["routes"]:
        raise ConfigError("loss_routing.routes must be a non-empty mapping")
    for route_name, route in loss_routing["routes"].items():
        if route_name not in {"sft_target", "sft_tool"}:
            raise ConfigError(f"Unsupported active loss route: {route_name}")
        route = _mapping(route, f"loss_routing.routes.{route_name}")
        _reject_unknown(route, {"type"}, f"loss_routing.routes.{route_name}")
        if route.get("type") != "sft_ce":
            raise ConfigError(f"loss_routing.routes.{route_name}.type must be sft_ce")

    training = _mapping(raw["training"], "training")
    _reject_unknown(
        training,
        {
            "num_epochs",
            "per_device_train_batch_size",
            "drop_last",
            "gradient_accumulation_steps",
            "learning_rate",
            "adamw_betas",
            "weight_decay",
            "warmup_ratio",
            "lr_scheduler_type",
            "max_grad_norm",
        },
        "training",
    )
    if "drop_last" in training and not isinstance(training["drop_last"], bool):
        raise ConfigError("training.drop_last must be true or false")
    if "adamw_betas" in training:
        validate_adamw_betas(training["adamw_betas"])
    for key in ("num_epochs", "per_device_train_batch_size", "gradient_accumulation_steps"):
        _require_positive_int(training.get(key), f"training.{key}")
    _require_positive_float(training.get("learning_rate"), "training.learning_rate")
    warmup_ratio = float(training.get("warmup_ratio", 0.0))
    if warmup_ratio < 0.0 or warmup_ratio > 1.0:
        raise ConfigError("training.warmup_ratio must be between 0 and 1")
    if float(training.get("max_grad_norm", 0.0)) < 0.0:
        raise ConfigError("training.max_grad_norm must be >= 0")
    validate_lora(raw["lora"], model=model)

    validate_eval(raw["eval"])
    validate_distributed(raw["distributed"])

    checkpointing = _mapping(raw["checkpointing"], "checkpointing")
    _reject_unknown(checkpointing, {"save_every_n_validations", "save_total_limit", "resume"}, "checkpointing")
    if "save_every_n_validations" in checkpointing:
        _require_positive_int(checkpointing["save_every_n_validations"], "checkpointing.save_every_n_validations")
    if checkpointing.get("save_total_limit") is not None:
        _require_positive_int(checkpointing["save_total_limit"], "checkpointing.save_total_limit")
    resume = _mapping(checkpointing.get("resume"), "checkpointing.resume")
    _reject_unknown(resume, {"enabled", "strict_config", "strict_dataset_hash", "strict_template_hash"}, "checkpointing.resume")

    mlflow = _mapping(raw["mlflow"], "mlflow")
    _reject_unknown(
        mlflow,
        {
            "tracking_uri",
            "resume_run_id",
            "async_logging",
            "log_rendered_samples",
        },
        "mlflow",
    )
    if not mlflow.get("tracking_uri"):
        raise ConfigError("mlflow.tracking_uri must be configured")
    validate_mlflow_async_logging(mlflow)

    validate_registry(
        raw["registry"],
        project=project,
        eval_config=raw["eval"],
        checkpointing=raw["checkpointing"],
    )
    return TrainingConfig(raw)


def validate_model(model: dict[str, Any]) -> None:
    _reject_unknown(
        model,
        {
            "name",
            "alias",
            "cache_dir",
            "checks",
            "precision",
            "use_fp8_base",
            "trust_remote_code",
            "attn_implementation",
            "gradient_checkpointing",
            "freeze_router",
            "freeze_embeddings",
            "freeze_lm_head",
        },
        "model",
    )
    for key in ("name", "alias", "cache_dir"):
        if not model.get(key):
            raise ConfigError(f"model.{key} must be configured for registry source")
    checks = _mapping(model.get("checks"), "model.checks")
    for key in ("verify_local_hash", "verify_remote_ref", "require_registry_metadata"):
        if not isinstance(checks.get(key), bool):
            raise ConfigError(f"model.checks.{key} must be true or false")
    if model.get("use_fp8_base") is not False:
        raise ConfigError("model.use_fp8_base must be false for the quality-first LoRA path")


def validate_lora(lora: Any, *, model: dict[str, Any]) -> None:
    lora = _mapping(lora, "lora")
    _reject_unknown(lora, {"r", "alpha", "dropout", "bias", "target_modules", "modules_to_save"}, "lora")
    target_modules = lora.get("target_modules")
    if not isinstance(target_modules, list) or not target_modules or not all(isinstance(item, str) and item for item in target_modules):
        raise ConfigError("lora.target_modules must be a non-empty list of module names")
    modules_to_save = lora.get("modules_to_save") or []
    if not isinstance(modules_to_save, list) or not all(isinstance(item, str) and item for item in modules_to_save):
        raise ConfigError("lora.modules_to_save must be a list of module names")
    if bool(model.get("freeze_lm_head", False)) and "lm_head" in modules_to_save:
        raise ConfigError("lora.modules_to_save cannot include lm_head when model.freeze_lm_head=true")


def validate_eval(eval_config: Any) -> None:
    eval_config = _mapping(eval_config, "eval")
    _reject_unknown(eval_config, {"every_train_steps", "standard", "bfcl"}, "eval")
    if "every_train_steps" in eval_config:
        _require_positive_int(eval_config["every_train_steps"], "eval.every_train_steps")
    standard = _mapping(eval_config.get("standard"), "eval.standard")
    _reject_unknown(standard, {"max_batches"}, "eval.standard")
    if standard.get("max_batches") is not None:
        _require_positive_int(standard["max_batches"], "eval.standard.max_batches")
    bfcl = _mapping(eval_config.get("bfcl"), "eval.bfcl")
    _reject_unknown(
        bfcl,
        {"enabled", "run_every_n_validations", "include_multi_turn", "categories", "limit", "generation"},
        "eval.bfcl",
    )
    if not isinstance(bfcl.get("enabled"), bool):
        raise ConfigError("eval.bfcl.enabled must be true or false")
    if bfcl.get("run_every_n_validations") is not None:
        _require_positive_int(bfcl["run_every_n_validations"], "eval.bfcl.run_every_n_validations")


def validate_registry(
    registry: Any,
    *,
    project: dict[str, Any],
    eval_config: dict[str, Any],
    checkpointing: dict[str, Any],
) -> None:
    registry = _mapping(registry, "registry")
    _reject_unknown(registry, {"register_every_n_checkpoints", "selection"}, "registry")
    _require_positive_int(registry.get("register_every_n_checkpoints"), "registry.register_every_n_checkpoints")
    selection = _mapping(registry.get("selection"), "registry.selection")
    _reject_unknown(selection, {"metric", "mode"}, "registry.selection")
    metric = selection.get("metric")
    mode = selection.get("mode")
    if not isinstance(metric, str) or not (
        metric in REGISTRY_SELECTION_METRICS
        or metric.startswith("eval/bfcl/") and metric.endswith(("/accuracy", "/total"))
    ):
        raise ConfigError("registry.selection.metric must be an emitted ordinary/BFCL checkpoint metric")
    if mode not in {"min", "max"}:
        raise ConfigError("registry.selection.mode must be min or max")
    if metric.startswith("eval/bfcl/") and not bool(eval_config["bfcl"].get("enabled")):
        raise ConfigError("registry.selection.metric uses BFCL while eval.bfcl.enabled=false")
    if metric.startswith("eval/bfcl/"):
        bfcl_every = int(eval_config["bfcl"]["run_every_n_validations"])
        checkpoint_every = int(checkpointing["save_every_n_validations"])
        if checkpoint_every % bfcl_every != 0:
            raise ConfigError("BFCL registry selection requires every checkpoint boundary to run BFCL")
    if not project.get("name"):
        raise ConfigError("project.name is required as the registry destination model name")


def validate_distributed(distributed: Any) -> None:
    distributed = _mapping(distributed, "distributed")
    fsdp = _mapping(distributed.get("fsdp"), "distributed.fsdp")
    for key in ("cpu_offload", "activation_checkpointing", "use_orig_params", "limit_all_gathers", "cpu_ram_efficient_loading", "sync_module_states"):
        if key in fsdp and not isinstance(fsdp[key], bool):
            raise ConfigError(f"distributed.fsdp.{key} must be true or false")
    class_names = fsdp.get("transformer_cls_names_to_wrap")
    if class_names is not None and (not isinstance(class_names, list) or not all(isinstance(item, str) and item for item in class_names)):
        raise ConfigError("distributed.fsdp.transformer_cls_names_to_wrap must be a list of strings")
    if fsdp.get("cpu_ram_efficient_loading") and fsdp.get("sync_module_states") is not True:
        raise ConfigError("distributed.fsdp.cpu_ram_efficient_loading=true requires sync_module_states=true")
    if fsdp.get("state_dict_type") != "sharded_state_dict":
        raise ConfigError("distributed.fsdp.state_dict_type must be sharded_state_dict for distributed optimizer state")
    if fsdp.get("use_orig_params") is not True:
        raise ConfigError("distributed.fsdp.use_orig_params must be true for mixed frozen/trainable PEFT parameters")


def validate_mlflow_async_logging(mlflow: dict[str, Any]) -> None:
    async_logging = _mapping(mlflow.get("async_logging"), "mlflow.async_logging")
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


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _reject_unknown(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"Unknown {name} fields: {unknown}")


def _require_positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return parsed


def _require_positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a positive number") from exc
    if parsed <= 0:
        raise ConfigError(f"{name} must be a positive number")
    return parsed

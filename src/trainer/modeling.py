from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
import math
from pathlib import Path
from typing import Any

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from config import Config


@dataclass(frozen=True)
class TrainingObjects:
    tokenizer: Any
    model: Any
    optimizer: Any
    scheduler: Any


def load_tokenizer(config: Config) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.model.cache_dir),
        use_fast=config.tokenizer.use_fast,
        trust_remote_code=config.model.trust_remote_code,
        padding_side=config.tokenizer.padding_side,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer must define pad_token or eos_token")

        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_base_model(config: Config) -> Any:
    model_cfg = config.model
    dtype = precision_to_dtype(model_cfg.precision)

    kwargs = {
        "dtype": dtype,
        "trust_remote_code": model_cfg.trust_remote_code,
        "attn_implementation": model_cfg.attn_implementation,
    }

    if model_cfg.experts_implementation is not None:
        kwargs["experts_implementation"] = model_cfg.experts_implementation

    model = AutoModelForCausalLM.from_pretrained(
        str(model_cfg.cache_dir),
        **kwargs,
    )
    disable_model_kv_cache(model)
    return model


def disable_model_kv_cache(model: Any) -> None:
    """Disable generation KV cache for full-sequence train/eval forwards."""

    model_config = getattr(model, "config", None)
    if model_config is not None and hasattr(model_config, "use_cache"):
        model_config.use_cache = False

    base_model = getattr(model, "base_model", None)
    base_config = getattr(base_model, "config", None)
    if base_config is not None and hasattr(base_config, "use_cache"):
        base_config.use_cache = False


def validate_model_runtime_requirements(config: Config) -> None:
    if config.model.attn_implementation == "flash_attention_2" and find_spec("flash_attn") is None:
        raise ImportError(
            "model.attn_implementation=flash_attention_2 requires the flash_attn package; "
            "install it in the training environment or select another attention implementation"
        )


def precision_to_dtype(precision: str) -> Any:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision in {"fp32", "float32"}:
        return torch.float32

    raise ValueError(f"unsupported model.precision: {precision}")


def cast_floating_parameters(model: Any, dtype: Any) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.is_floating_point() and parameter.dtype != dtype:
                parameter.data = parameter.data.to(dtype)


def freeze_configured_modules(model: Any, config: Config) -> None:
    if config.model.freeze_embeddings:
        _freeze_module(getattr(model, "get_input_embeddings", lambda: None)())

    if config.model.freeze_lm_head:
        _freeze_module(getattr(model, "lm_head", None))

    if config.model.freeze_router:
        for name, parameter in model.named_parameters():
            lowered = name.lower()

            if "router" in lowered or ("gate" in lowered and "gate_proj" not in lowered):
                parameter.requires_grad = False


def configure_gradient_checkpointing(model: Any, config: Config) -> None:
    """Apply the YAML gradient-checkpointing contract to the prepared model."""

    if not config.model.gradient_checkpointing:
        return

    disable_model_kv_cache(model)

    if not hasattr(model, "gradient_checkpointing_enable"):
        raise TypeError("model.gradient_checkpointing=true requires gradient_checkpointing_enable()")

    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def apply_lora(config: Config, model: Any) -> Any:
    lora = config.lora

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=lora.bias,
        target_modules=list(lora.target_modules),
        modules_to_save=list(lora.modules_to_save),
    )
    
    return get_peft_model(model, peft_config)


def build_optimizer(config: Config, model: Any) -> Any:
    training = config.training
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    if not trainable_parameters:
        raise ValueError("model has no trainable parameters")

    return torch.optim.AdamW(
        trainable_parameters,
        lr=training.learning_rate,
        betas=training.adamw_betas,
        weight_decay=training.weight_decay,
    )


def training_steps_for_epochs(config: Config, train_dataloader: Any, *, num_processes: int = 1) -> int:
    """Resolve epoch count into optimizer steps for the distributed DataLoader."""

    micro_batches_per_process = math.ceil(len(train_dataloader) / max(int(num_processes), 1))
    steps_per_epoch = math.ceil(micro_batches_per_process / config.training.gradient_accumulation_steps)

    return steps_per_epoch * config.training.num_epochs


def build_scheduler(config: Config, optimizer: Any, *, total_steps: int) -> Any:
    training = config.training
    warmup_steps = int(total_steps * training.warmup_ratio)

    return get_scheduler(
        training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def load_lora_adapter(config: Config, model: Any, adapter_path: str | Path) -> Any:
    del config

    return PeftModel.from_pretrained(model, adapter_path, is_trainable=True)


def build_training_objects(
    config: Config,
    *,
    total_steps: int,
    resume_adapter_path: str | Path | None = None,
    tokenizer: Any | None = None,
) -> TrainingObjects:
    validate_model_runtime_requirements(config)
    tokenizer = load_tokenizer(config) if tokenizer is None else tokenizer
    base_model = load_base_model(config)

    model = (
        load_lora_adapter(config, base_model, resume_adapter_path)
        if resume_adapter_path is not None
        else apply_lora(config, base_model)
    )
    disable_model_kv_cache(model)
    configure_gradient_checkpointing(model, config)
    freeze_configured_modules(model, config)

    target_dtype = precision_to_dtype(config.model.precision)
    cast_floating_parameters(model, target_dtype)

    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, total_steps=total_steps)

    return TrainingObjects(tokenizer=tokenizer, model=model, optimizer=optimizer, scheduler=scheduler)


def _freeze_module(module: Any) -> None:
    if module is None:
        return

    for parameter in module.parameters():
        parameter.requires_grad = False

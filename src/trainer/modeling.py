from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import effective_tokenizer_id


@dataclass(frozen=True)
class TrainingObjects:
    tokenizer: Any
    model: Any
    optimizer: Any
    scheduler: Any


def load_tokenizer(config: Any) -> Any:
    from transformers import AutoTokenizer

    tokenizer_config = config.section("tokenizer")
    model_config = config.section("model")
    tokenizer = AutoTokenizer.from_pretrained(
        effective_tokenizer_id(config),
        revision=tokenizer_config.get("tokenizer_revision"),
        use_fast=bool(tokenizer_config.get("use_fast", True)),
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        padding_side=tokenizer_config.get("padding_side", "right"),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer must define pad_token or eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(config: Any) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model_config = config.section("model")
    kwargs: dict[str, Any] = {
        "revision": model_config.get("base_model_revision"),
        "trust_remote_code": bool(model_config.get("trust_remote_code", True)),
        "torch_dtype": precision_to_dtype(str(model_config.get("precision", "bf16"))),
    }
    if model_config.get("attn_implementation"):
        kwargs["attn_implementation"] = model_config["attn_implementation"]

    model = AutoModelForCausalLM.from_pretrained(model_config["resolved_model_id"], **kwargs)
    if bool(model_config.get("gradient_checkpointing", False)):
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        model.gradient_checkpointing_enable()
    freeze_configured_modules(model, model_config)
    return model


def precision_to_dtype(precision: str) -> Any:
    import torch

    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported model.precision: {precision}")


def freeze_configured_modules(model: Any, model_config: dict[str, Any]) -> None:
    if bool(model_config.get("freeze_embeddings", False)):
        _freeze_module(getattr(model, "get_input_embeddings", lambda: None)())
    if bool(model_config.get("freeze_lm_head", False)):
        _freeze_module(getattr(model, "lm_head", None))
    if bool(model_config.get("freeze_router", False)):
        for name, parameter in model.named_parameters():
            lowered = name.lower()
            if "router" in lowered or "gate" in lowered and "gate_proj" not in lowered:
                parameter.requires_grad = False


def apply_lora(config: Any, model: Any) -> Any:
    lora = config.section("lora")
    if not bool(lora.get("enabled", False)):
        return model
    from peft import LoraConfig, TaskType, get_peft_model

    if lora.get("target_modules_policy") != "whitelist":
        raise ValueError("only lora.target_modules_policy=whitelist is supported")
    target_modules = lora.get("target_modules")
    if not target_modules:
        raise ValueError("lora.target_modules must be configured for training")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora.get("dropout", 0.0)),
        bias=str(lora.get("bias", "none")),
        target_modules=list(target_modules),
        modules_to_save=list(lora.get("modules_to_save") or []),
    )
    return get_peft_model(model, peft_config)


def build_optimizer(config: Any, model: Any) -> Any:
    import torch

    training = config.section("training")
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training.get("adamw_betas", [0.9, 0.999])),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )


def build_scheduler(config: Any, optimizer: Any) -> Any:
    from transformers import get_scheduler

    training = config.section("training")
    max_steps = int(training["max_steps"])
    warmup_steps = int(max_steps * float(training.get("warmup_ratio", 0.0)))
    return get_scheduler(
        str(training.get("lr_scheduler_type", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )


def load_lora_adapter(config: Any, model: Any, adapter_path: str | Path) -> Any:
    del config
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path, is_trainable=True)


def build_training_objects(config: Any, *, resume_adapter_path: str | Path | None = None, tokenizer: Any | None = None) -> TrainingObjects:
    tokenizer = load_tokenizer(config) if tokenizer is None else tokenizer
    base_model = load_base_model(config)
    model = load_lora_adapter(config, base_model, resume_adapter_path) if resume_adapter_path is not None else apply_lora(config, base_model)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    return TrainingObjects(tokenizer=tokenizer, model=model, optimizer=optimizer, scheduler=scheduler)


def _freeze_module(module: Any) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = False

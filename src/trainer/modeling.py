from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import torch

from utils.cuda_runtime import preload_cuda_runtime

preload_cuda_runtime()

from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from config import Config


@dataclass(frozen=True, slots=True)
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
        "torch_dtype": dtype,
        "trust_remote_code": model_cfg.trust_remote_code,
        "attn_implementation": model_cfg.attn_implementation,
    }

    if model_cfg.experts_implementation is not None:
        kwargs["experts_implementation"] = model_cfg.experts_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_cfg.cache_dir), **kwargs)
    except TypeError as exc:
        if "torch_dtype" not in str(exc):
            raise
        kwargs["dtype"] = kwargs.pop("torch_dtype")
        model = AutoModelForCausalLM.from_pretrained(str(model_cfg.cache_dir), **kwargs)

    disable_model_kv_cache(model)
    disable_router_logits_when_frozen(model, config)
    return model


def disable_router_logits_when_frozen(model: Any, config: Config) -> None:
    """Avoid materializing MoE router logits when router loss is not trained."""

    if not config.model.freeze_router:
        return

    for cfg in (getattr(model, "config", None), getattr(getattr(model, "base_model", None), "config", None)):
        if cfg is not None and hasattr(cfg, "output_router_logits"):
            cfg.output_router_logits = False

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
    runtime_path = preload_cuda_runtime()

    if find_spec("causal_conv1d") is not None and runtime_path is None:
        raise ImportError(
            "causal_conv1d is installed, but libcudart.so.12 was not found in the environment. "
            "Install this project with constraints/cuda126_kernels.txt or install "
            "nvidia-cuda-runtime-cu12."
        )

    if config.model.attn_implementation == "flash_attention_2" and find_spec("flash_attn") is None:
        raise ImportError(
            "model.attn_implementation=flash_attention_2 requires the flash_attn package; "
            "install it with the project cuda-kernels extra or select another attention implementation"
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
    freeze_vocab_modules(model)

    if config.model.freeze_router:
        for name, parameter in model.named_parameters():
            lowered = name.lower()

            if "router" in lowered or ("gate" in lowered and "gate_proj" not in lowered):
                parameter.requires_grad = False



def freeze_vocab_modules(model: Any) -> None:
    """Always freeze input embeddings and output vocab projection.

    This is deliberately not configurable. The DPO reference pass is computed by
    disabling PEFT adapters on the same model, so vocab projection modules must
    stay outside the trainable adapter/state-switching surface.
    """

    seen: set[int] = set()

    for module in vocab_modules_from_names(model) + vocab_modules_from_accessors(model):
        if module is None:
            continue
        marker = id(module)
        if marker in seen:
            continue
        seen.add(marker)
        _freeze_module(module)


def assert_vocab_modules_frozen(model: Any) -> None:
    """Fail before optimizer/FSDP if any discovered vocab module is trainable."""

    modules = vocab_modules_from_names(model) + vocab_modules_from_accessors(model)
    seen: set[int] = set()
    trainable: list[str] = []

    for module in modules:
        if module is None:
            continue
        marker = id(module)
        if marker in seen:
            continue
        seen.add(marker)

        module_name = module_name_or_class(model, module)
        for parameter_name, parameter in module.named_parameters():
            if parameter.requires_grad:
                trainable.append(f"{module_name}.{parameter_name}")

    if trainable:
        raise RuntimeError(f"vocab modules must be frozen, but trainable parameters were found: {trainable[:10]}")


def vocab_modules_from_names(model: Any) -> list[Any]:
    modules: list[Any] = []
    for name, module in model.named_modules():
        if not name:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in {"embed_tokens", "lm_head"} and getattr(module, "weight", None) is not None:
            modules.append(module)
    return modules


def vocab_modules_from_accessors(model: Any) -> list[Any]:
    modules: list[Any] = []
    for root in iter_model_roots(model):
        for accessor_name in ("get_input_embeddings", "get_output_embeddings"):
            accessor = getattr(root, accessor_name, None)
            if not callable(accessor):
                continue
            try:
                module = accessor()
            except Exception:
                continue
            if module is not None and getattr(module, "weight", None) is not None:
                modules.append(module)
    return modules


def iter_model_roots(model: Any) -> tuple[Any, ...]:
    result: list[Any] = []
    queue: list[Any] = [model]
    seen: set[int] = set()

    while queue:
        current = queue.pop(0)
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(current)

        for attr in ("base_model", "model"):
            child = getattr(current, attr, None)
            if child is not None and id(child) not in seen:
                queue.append(child)

    return tuple(result)


def module_name_or_class(model: Any, target_module: Any) -> str:
    for name, module in model.named_modules():
        if module is target_module:
            return name or target_module.__class__.__name__
    return target_module.__class__.__name__


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


def validate_peft_reference_support(model: Any) -> None:
    """Fail before FSDP wrapping if DPO cannot compute its on-the-fly reference."""

    if not callable(getattr(model, "disable_adapter", None)):
        raise RuntimeError(
            "DPO on-the-fly reference requires PEFT disable_adapter(); "
            "the training model must be created with get_peft_model() or loaded as a trainable PeftModel."
        )


def summarize_trainable_parameters(model: Any) -> dict[str, Any]:
    """Return a compact startup audit for LoRA/FSDP runs."""

    total = 0
    trainable = 0
    trainable_names: list[str] = []
    router_trainable = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
            if len(trainable_names) < 20:
                trainable_names.append(name)
            lowered = name.lower()
            if "router" in lowered or ("gate" in lowered and "gate_proj" not in lowered):
                router_trainable += count

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_ratio": (trainable / total) if total else 0.0,
        "router_trainable_parameters": router_trainable,
        "trainable_name_sample": trainable_names,
        "has_disable_adapter": callable(getattr(model, "disable_adapter", None)),
    }


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
    """Resolve epoch count into full optimizer steps for the distributed DataLoader."""

    processes = max(int(num_processes), 1)
    total_micro_batches = int(len(train_dataloader))
    if total_micro_batches <= 0:
        raise ValueError("train dataloader must contain at least one micro-batch")
    if total_micro_batches % processes != 0:
        raise RuntimeError(
            "train dataloader length must be divisible by the number of processes before step counting: "
            f"len={total_micro_batches} num_processes={processes}"
        )

    micro_batches_per_process = total_micro_batches // processes
    grad_accum = int(config.training.gradient_accumulation_steps)
    if micro_batches_per_process % grad_accum != 0:
        raise RuntimeError(
            "per-process train dataloader length must be divisible by gradient_accumulation_steps: "
            f"micro_batches_per_process={micro_batches_per_process} gradient_accumulation_steps={grad_accum}. "
            "The train sampler must drop or pad the accumulation tail before training starts."
        )

    steps_per_epoch = micro_batches_per_process // grad_accum
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
    assert_vocab_modules_frozen(model)

    target_dtype = precision_to_dtype(config.model.precision)
    cast_floating_parameters(model, target_dtype)
    validate_peft_reference_support(model)

    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, total_steps=total_steps)

    return TrainingObjects(tokenizer=tokenizer, model=model, optimizer=optimizer, scheduler=scheduler)


def _freeze_module(module: Any) -> None:
    if module is None:
        return

    for parameter in module.parameters():
        parameter.requires_grad = False

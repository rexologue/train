from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import DistributedType

from config import Config


@dataclass(frozen=True)
class DistributedRuntime:
    accelerator: Any
    fsdp_plugin: Any


def build_fsdp_plugin(config: Config) -> Any:
    """Build Accelerate's FSDP plugin from YAML values."""

    fsdp = config.distributed.fsdp

    kwargs: dict[str, Any] = {
        "sharding_strategy": fsdp.sharding_strategy,
        "mixed_precision_policy": fsdp.mixed_precision,
        "cpu_offload": fsdp.cpu_offload,
        "state_dict_type": fsdp.state_dict_type,
        "activation_checkpointing": fsdp.activation_checkpointing,
        "use_orig_params": fsdp.use_orig_params,
        "limit_all_gathers": fsdp.limit_all_gathers,
    }
    
    kwargs["auto_wrap_policy"] = fsdp.auto_wrap_policy
    kwargs["transformer_cls_names_to_wrap"] = list(fsdp.transformer_cls_names_to_wrap)
    kwargs["cpu_ram_efficient_loading"] = fsdp.cpu_ram_efficient_loading
    kwargs["sync_module_states"] = fsdp.sync_module_states

    return FullyShardedDataParallelPlugin(**kwargs)


def create_accelerator(config: Config) -> DistributedRuntime:
    """Create the Accelerate runtime described by config."""

    fsdp_plugin = build_fsdp_plugin(config)
    fsdp = config.distributed.fsdp

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=fsdp.mixed_precision,
        fsdp_plugin=fsdp_plugin,
        project_dir=str(config.project.output_dir),
    )

    validate_training_runtime(accelerator)

    return DistributedRuntime(accelerator=accelerator, fsdp_plugin=fsdp_plugin)


def validate_training_runtime(accelerator: Any) -> None:
    """Fail before model loading when training was not launched through FSDP."""

    if accelerator.distributed_type != DistributedType.FSDP:
        raise RuntimeError(
            "training requires Accelerate/FSDP, but the active runtime is "
            f"{accelerator.distributed_type}; launch with `accelerate launch --use_fsdp ...`"
        )


def prepare_with_accelerator(runtime: DistributedRuntime, *objects: Any) -> tuple[Any, ...]:
    """Prepare model/optimizer/dataloaders/scheduler through Accelerator."""

    prepared = runtime.accelerator.prepare(*objects)
    
    if not isinstance(prepared, tuple):
        return (prepared,)

    return prepared

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DistributedRuntime:
    accelerator: Any
    fsdp_plugin: Any


def build_fsdp_plugin(config: Any) -> Any:
    """Build Accelerate's FSDP plugin from YAML values."""

    from accelerate import FullyShardedDataParallelPlugin

    fsdp = config.section("distributed")["fsdp"]

    kwargs: dict[str, Any] = {
        "sharding_strategy": fsdp.get("sharding_strategy"),
        "mixed_precision_policy": fsdp.get("mixed_precision"),
        "cpu_offload": bool(fsdp.get("cpu_offload", False)),
        "state_dict_type": fsdp.get("state_dict_type"),
        "activation_checkpointing": bool(fsdp.get("activation_checkpointing", False)),
        "use_orig_params": bool(fsdp.get("use_orig_params", True)),
        "limit_all_gathers": bool(fsdp.get("limit_all_gathers", True)),
    }
    if fsdp.get("auto_wrap_policy") is not None:
        kwargs["auto_wrap_policy"] = fsdp["auto_wrap_policy"]
    if fsdp.get("transformer_cls_names_to_wrap") is not None:
        kwargs["transformer_cls_names_to_wrap"] = list(fsdp["transformer_cls_names_to_wrap"])
    if fsdp.get("cpu_ram_efficient_loading") is not None:
        kwargs["cpu_ram_efficient_loading"] = bool(fsdp["cpu_ram_efficient_loading"])
    if fsdp.get("sync_module_states") is not None:
        kwargs["sync_module_states"] = bool(fsdp["sync_module_states"])
    return FullyShardedDataParallelPlugin(**kwargs)


def create_accelerator(config: Any) -> DistributedRuntime:
    """Create the Accelerate runtime described by config."""

    from accelerate import Accelerator

    distributed = config.section("distributed")
    fsdp_plugin = build_fsdp_plugin(config)
    training = config.section("training")
    fsdp = distributed["fsdp"]
    accelerator = Accelerator(
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        mixed_precision=fsdp.get("mixed_precision"),
        fsdp_plugin=fsdp_plugin,
        project_dir=str(config.section("project").get("output_dir")),
    )
    return DistributedRuntime(accelerator=accelerator, fsdp_plugin=fsdp_plugin)


def prepare_with_accelerator(runtime: DistributedRuntime, *objects: Any) -> tuple[Any, ...]:
    """Prepare model/optimizer/dataloaders/scheduler through Accelerator."""

    prepared = runtime.accelerator.prepare(*objects)
    if not isinstance(prepared, tuple):
        return (prepared,)
    return prepared

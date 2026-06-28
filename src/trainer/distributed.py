from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import DistributedType

from config import Config
from utils.logging import get_logger


# Accelerate resolves string ignored_modules with re.fullmatch(name).
# This pattern matches both plain HF names and PEFT-wrapped names, e.g.:
#   model.embed_tokens
#   lm_head
#   base_model.model.model.embed_tokens
#   base_model.model.lm_head
VOCAB_FSDP_IGNORE_PATTERN = r"(.*\.)?(embed_tokens|lm_head)"
VOCAB_MODULE_LEAVES = {"embed_tokens", "lm_head"}


@dataclass(frozen=True, slots=True)
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
        "ignored_modules": VOCAB_FSDP_IGNORE_PATTERN,
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

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision="no",
        fsdp_plugin=fsdp_plugin,
        project_dir=str(config.project.output_dir),
    )

    validate_training_runtime(accelerator)
    synchronize_runtime_fsdp_plugin(runtime_plugin=fsdp_plugin, accelerator=accelerator)

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

    model = first_model_like_object(objects)
    synchronize_runtime_fsdp_plugin(runtime_plugin=runtime.fsdp_plugin, accelerator=runtime.accelerator)

    if model is not None:
        audit_and_validate_vocab_fsdp_contract(model, runtime)
        validate_transformer_wrap_classes(model, runtime)

    prepared = runtime.accelerator.prepare(*objects)

    if not isinstance(prepared, tuple):
        return (prepared,)

    return prepared


def validate_transformer_wrap_classes(model: Any, runtime: DistributedRuntime) -> None:
    """Fail fast if `transformer_cls_names_to_wrap` matches no module class.

    With `transformer_based_wrap`, FSDP only shards modules whose class name is
    listed. A wrong/placeholder name (the shipped config uses
    `DecoderLayerClassName`) silently wraps nothing, so the whole 35B model
    becomes one FSDP unit and OOMs on the first forward. Verify the contract
    against the real module tree before `prepare()` builds FlatParameters.
    """

    plugin = getattr(getattr(runtime.accelerator, "state", None), "fsdp_plugin", None) or runtime.fsdp_plugin
    auto_wrap = str(getattr(plugin, "auto_wrap_policy", "") or "")
    configured = {name for name in getattr(plugin, "transformer_cls_names_to_wrap", None) or []}
    if not configured or "transformer" not in auto_wrap.lower():
        return

    present = {module.__class__.__name__ for _name, module in model.named_modules()}
    matched = configured & present
    if not matched:
        decoder_like = sorted(name for name in present if "DecoderLayer" in name or "Block" in name)
        raise RuntimeError(
            "distributed.fsdp.transformer_cls_names_to_wrap matches no module class in the model: "
            f"configured={sorted(configured)}. Set it to the model's decoder layer class. "
            f"Candidates found in the model: {decoder_like or sorted(present)[:20]}"
        )

    logger = get_logger("train")
    logger.info("FSDP transformer wrap classes matched: %s", sorted(matched))


def first_model_like_object(objects: tuple[Any, ...]) -> Any | None:
    for obj in objects:
        if callable(getattr(obj, "named_modules", None)) and callable(getattr(obj, "named_parameters", None)):
            return obj
    return None


def synchronize_runtime_fsdp_plugin(*, runtime_plugin: Any, accelerator: Any) -> None:
    """Keep both the plugin object and AcceleratorState plugin on the same ignore contract."""

    for plugin in unique_existing_plugins(runtime_plugin, getattr(getattr(accelerator, "state", None), "fsdp_plugin", None)):
        set_ignored_modules(plugin, VOCAB_FSDP_IGNORE_PATTERN)


def unique_existing_plugins(*plugins: Any) -> tuple[Any, ...]:
    result: list[Any] = []
    seen: set[int] = set()
    for plugin in plugins:
        if plugin is None:
            continue
        marker = id(plugin)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(plugin)
    return tuple(result)


def set_ignored_modules(plugin: Any, value: Any) -> None:
    try:
        setattr(plugin, "ignored_modules", value)
    except Exception as exc:
        raise RuntimeError("failed to set FSDP ignored_modules for vocab modules") from exc


def audit_and_validate_vocab_fsdp_contract(model: Any, runtime: DistributedRuntime) -> None:
    """Prove before prepare() that vocab modules are frozen and FSDP-ignored.

    The previous implementation tried to add module objects to ignored_modules after
    Accelerator creation. In recent Accelerate the supported robust path is to pass
    ignored_modules to FullyShardedDataParallelPlugin itself. Here we use a regex
    contract and verify it against the actual PEFT-wrapped module names before FSDP
    constructs FlatParameter objects.
    """

    logger = get_logger("train")
    plugin = getattr(getattr(runtime.accelerator, "state", None), "fsdp_plugin", None) or runtime.fsdp_plugin
    ignored_spec = getattr(plugin, "ignored_modules", None)

    named_vocab_modules = find_named_vocab_modules(model)
    if not named_vocab_modules:
        raise RuntimeError(
            "could not find vocab modules before FSDP prepare; expected modules named "
            "embed_tokens and/or lm_head in the PEFT-wrapped model"
        )

    trainable = trainable_vocab_parameter_names(named_vocab_modules)
    if trainable:
        raise RuntimeError(
            "vocab modules must be frozen before FSDP prepare; trainable parameters found: "
            f"{trainable[:10]}"
        )

    matched_names = resolve_ignored_module_names(model, ignored_spec)
    missing = [name for name, _module in named_vocab_modules if name not in matched_names]
    if missing:
        raise RuntimeError(
            "vocab modules were found but are not covered by FSDP ignored_modules. "
            f"ignored_modules={ignored_spec!r} missing={missing} matched={matched_names}"
        )

    tied_pairs = describe_tied_vocab_pairs(named_vocab_modules)
    logger.info(
        "FSDP vocab ignore audit: ignored_modules=%r matched=%s vocab=%s tied=%s",
        ignored_spec,
        matched_names,
        describe_vocab_modules(named_vocab_modules),
        tied_pairs,
    )


def find_named_vocab_modules(model: Any) -> list[tuple[str, Any]]:
    modules: list[tuple[str, Any]] = []
    seen: set[int] = set()

    for name, module in model.named_modules():
        if not name:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in VOCAB_MODULE_LEAVES:
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        marker = id(module)
        if marker in seen:
            continue
        seen.add(marker)
        modules.append((name, module))

    for module in modules_from_embedding_accessors(model):
        marker = id(module)
        if marker in seen:
            continue
        name = find_module_name(model, module)
        if name is None:
            continue
        seen.add(marker)
        modules.append((name, module))

    modules.sort(key=lambda item: item[0])
    return modules


def modules_from_embedding_accessors(model: Any) -> tuple[Any, ...]:
    modules: list[Any] = []
    seen: set[int] = set()

    for root in iter_model_roots(model):
        for accessor_name in ("get_input_embeddings", "get_output_embeddings"):
            accessor = getattr(root, accessor_name, None)
            if not callable(accessor):
                continue
            try:
                module = accessor()
            except Exception:
                continue
            if module is None or getattr(module, "weight", None) is None:
                continue
            marker = id(module)
            if marker in seen:
                continue
            seen.add(marker)
            modules.append(module)

    return tuple(modules)


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


def find_module_name(model: Any, target_module: Any) -> str | None:
    for name, module in model.named_modules():
        if module is target_module:
            return name
    return None


def trainable_vocab_parameter_names(named_vocab_modules: list[tuple[str, Any]]) -> list[str]:
    names: list[str] = []
    for module_name, module in named_vocab_modules:
        for parameter_name, parameter in module.named_parameters():
            if parameter.requires_grad:
                names.append(f"{module_name}.{parameter_name}")
    return names


def resolve_ignored_module_names(model: Any, ignored_spec: Any) -> list[str]:
    if ignored_spec is None:
        return []

    if isinstance(ignored_spec, str):
        regex = re.compile(ignored_spec)
        return [name for name, _module in model.named_modules() if name and regex.fullmatch(name)]

    modules = tuple(ignored_spec) if isinstance(ignored_spec, (tuple, list, set)) else (ignored_spec,)
    module_ids = {id(module) for module in modules}
    return [name for name, module in model.named_modules() if id(module) in module_ids]


def describe_vocab_modules(named_vocab_modules: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name, module in named_vocab_modules:
        weight = getattr(module, "weight", None)
        result.append(
            {
                "name": name,
                "class": module.__class__.__name__,
                "shape": list(getattr(weight, "shape", ())),
                "requires_grad": bool(getattr(weight, "requires_grad", False)),
            }
        )
    return result


def describe_tied_vocab_pairs(named_vocab_modules: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, (left_name, left_module) in enumerate(named_vocab_modules):
        left_weight = getattr(left_module, "weight", None)
        if left_weight is None:
            continue
        for right_name, right_module in named_vocab_modules[index + 1 :]:
            right_weight = getattr(right_module, "weight", None)
            if right_weight is None:
                continue
            if weights_share_identity_or_storage(left_weight, right_weight):
                result.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "shape": list(getattr(left_weight, "shape", ())),
                    }
                )
    return result


def weights_share_identity_or_storage(left: Any, right: Any) -> bool:
    if left is right:
        return True

    left_data = getattr(left, "data", None)
    right_data = getattr(right, "data", None)
    if left_data is not None and right_data is not None and left_data is right_data:
        return True

    try:
        return bool(left.data_ptr() == right.data_ptr())
    except Exception:
        return False

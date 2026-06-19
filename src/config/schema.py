from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


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

OPTIONAL_TOP_LEVEL_KEYS = {
    "progress",
}

TOP_LEVEL_KEYS = REQUIRED_TOP_LEVEL_KEYS | OPTIONAL_TOP_LEVEL_KEYS

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
class ProjectConfig:
    name: str
    run_name: str | None
    seed: int
    output_dir: Path

    @classmethod
    def from_dict(cls, raw: Any) -> ProjectConfig:
        data = _mapping(raw, "project")
        _reject_unknown(data, {"name", "run_name", "seed", "output_dir"}, "project")

        return cls(
            name=_require_non_empty_str(data.get("name"), "project.name"),
            run_name=_optional_str(data.get("run_name"), "project.run_name"),
            seed=_require_int(data.get("seed"), "project.seed"),
            output_dir=_path(data.get("output_dir"), "project.output_dir"),
        )


@dataclass(frozen=True)
class ModelChecksConfig:
    verify_local_hash: bool
    verify_remote_ref: bool
    require_registry_metadata: bool

    @classmethod
    def from_dict(cls, raw: Any) -> ModelChecksConfig:
        data = _mapping(raw, "model.checks")
        _reject_unknown(
            data,
            {"verify_local_hash", "verify_remote_ref", "require_registry_metadata"},
            "model.checks",
        )

        return cls(
            verify_local_hash=_require_bool(data.get("verify_local_hash"), "model.checks.verify_local_hash"),
            verify_remote_ref=_require_bool(data.get("verify_remote_ref"), "model.checks.verify_remote_ref"),
            require_registry_metadata=_require_bool(
                data.get("require_registry_metadata"),
                "model.checks.require_registry_metadata",
            ),
        )


@dataclass(frozen=True)
class ModelConfig:
    name: str
    alias: str
    cache_dir: Path
    checks: ModelChecksConfig
    precision: Literal["bf16", "fp16", "fp32"]
    trust_remote_code: bool
    attn_implementation: str | None
    experts_implementation: Literal["eager", "batched_mm", "grouped_mm"] | None
    gradient_checkpointing: bool
    freeze_router: bool
    freeze_embeddings: bool
    freeze_lm_head: bool

    @classmethod
    def from_dict(cls, raw: Any) -> ModelConfig:
        data = _mapping(raw, "model")
        _reject_unknown(
            data,
            {
                "name",
                "alias",
                "cache_dir",
                "checks",
                "precision",
                "trust_remote_code",
                "attn_implementation",
                "experts_implementation",
                "gradient_checkpointing",
                "freeze_router",
                "freeze_embeddings",
                "freeze_lm_head",
            },
            "model",
        )

        precision = _require_non_empty_str(data.get("precision"), "model.precision")
        if precision not in {"bf16", "fp16", "fp32"}:
            raise ConfigError("model.precision must be one of: bf16, fp16, fp32")

        experts_implementation = data.get("experts_implementation")
        if experts_implementation is not None and experts_implementation not in {"eager", "batched_mm", "grouped_mm"}:
            raise ConfigError("model.experts_implementation must be one of: eager, batched_mm, grouped_mm")

        return cls(
            name=_require_non_empty_str(data.get("name"), "model.name"),
            alias=_require_non_empty_str(data.get("alias"), "model.alias"),
            cache_dir=_path(data.get("cache_dir"), "model.cache_dir"),
            checks=ModelChecksConfig.from_dict(data.get("checks")),
            precision=precision,  # type: ignore[arg-type]
            trust_remote_code=_require_bool(data.get("trust_remote_code"), "model.trust_remote_code"),
            attn_implementation=_optional_str(data.get("attn_implementation"), "model.attn_implementation"),
            experts_implementation=experts_implementation,  # type: ignore[arg-type]
            gradient_checkpointing=_require_bool(data.get("gradient_checkpointing"), "model.gradient_checkpointing"),
            freeze_router=_require_bool(data.get("freeze_router"), "model.freeze_router"),
            freeze_embeddings=_require_bool(data.get("freeze_embeddings"), "model.freeze_embeddings"),
            freeze_lm_head=_require_bool(data.get("freeze_lm_head"), "model.freeze_lm_head"),
        )


@dataclass(frozen=True)
class TokenizerConfig:
    use_fast: bool
    add_special_tokens: bool
    padding_side: Literal["left", "right"]

    @classmethod
    def from_dict(cls, raw: Any) -> TokenizerConfig:
        data = _mapping(raw, "tokenizer")
        _reject_unknown(data, {"use_fast", "add_special_tokens", "padding_side"}, "tokenizer")

        padding_side = _require_non_empty_str(data.get("padding_side"), "tokenizer.padding_side")
        if padding_side not in {"left", "right"}:
            raise ConfigError("tokenizer.padding_side must be left or right")

        return cls(
            use_fast=_require_bool(data.get("use_fast"), "tokenizer.use_fast"),
            add_special_tokens=_require_bool(data.get("add_special_tokens"), "tokenizer.add_special_tokens"),
            padding_side=padding_side,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class LoraConfig:
    r: int
    alpha: int
    dropout: float
    bias: Literal["none", "all", "lora_only"]
    target_modules: tuple[str, ...]
    modules_to_save: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> LoraConfig:
        data = _mapping(raw, "lora")
        _reject_unknown(data, {"r", "alpha", "dropout", "bias", "target_modules", "modules_to_save"}, "lora")

        bias = _require_non_empty_str(data.get("bias"), "lora.bias")
        if bias not in {"none", "all", "lora_only"}:
            raise ConfigError("lora.bias must be one of: none, all, lora_only")

        dropout = _require_float(data.get("dropout"), "lora.dropout")
        if dropout < 0.0 or dropout >= 1.0:
            raise ConfigError("lora.dropout must be >= 0 and < 1")

        return cls(
            r=_require_positive_int(data.get("r"), "lora.r"),
            alpha=_require_positive_int(data.get("alpha"), "lora.alpha"),
            dropout=dropout,
            bias=bias,  # type: ignore[arg-type]
            target_modules=_string_tuple(data.get("target_modules"), "lora.target_modules", allow_empty=False),
            modules_to_save=_string_tuple(data.get("modules_to_save", []), "lora.modules_to_save", allow_empty=True),
        )


@dataclass(frozen=True)
class PreprocessingRawConfig:
    train_path: Path
    valid_path: Path
    test_path: Path | None
    test_required: bool

    @classmethod
    def from_dict(cls, raw: Any) -> PreprocessingRawConfig:
        data = _mapping(raw, "preprocessing.raw")
        _reject_unknown(data, {"train_path", "valid_path", "test_path", "test_required"}, "preprocessing.raw")

        test_path_value = data.get("test_path")

        return cls(
            train_path=_path(data.get("train_path"), "preprocessing.raw.train_path"),
            valid_path=_path(data.get("valid_path"), "preprocessing.raw.valid_path"),
            test_path=None if test_path_value is None else _path(test_path_value, "preprocessing.raw.test_path"),
            test_required=_require_bool(data.get("test_required"), "preprocessing.raw.test_required"),
        )


@dataclass(frozen=True)
class SequenceConfig:
    max_seq_len: int
    truncation: bool
    packing: bool

    @classmethod
    def from_dict(cls, raw: Any) -> SequenceConfig:
        data = _mapping(raw, "preprocessing.sequence")
        _reject_unknown(data, {"max_seq_len", "truncation", "packing"}, "preprocessing.sequence")

        truncation = _require_bool(data.get("truncation"), "preprocessing.sequence.truncation")
        packing = _require_bool(data.get("packing"), "preprocessing.sequence.packing")

        if truncation is not False:
            raise ConfigError("preprocessing.sequence.truncation must stay false until explicit turn-aware policy exists")
        if packing is not False:
            raise ConfigError("preprocessing.sequence.packing must stay false until packing mask tests exist")

        return cls(
            max_seq_len=_require_positive_int(data.get("max_seq_len"), "preprocessing.sequence.max_seq_len"),
            truncation=truncation,
            packing=packing,
        )


@dataclass(frozen=True)
class RenderingConfig:
    use_system: bool
    reject_raw_special_markers: bool

    @classmethod
    def from_dict(cls, raw: Any) -> RenderingConfig:
        data = _mapping(raw, "preprocessing.rendering")
        _reject_unknown(data, {"use_system", "reject_raw_special_markers"}, "preprocessing.rendering")

        return cls(
            use_system=_require_bool(data.get("use_system"), "preprocessing.rendering.use_system"),
            reject_raw_special_markers=_require_bool(
                data.get("reject_raw_special_markers"),
                "preprocessing.rendering.reject_raw_special_markers",
            )
        )


@dataclass(frozen=True)
class ReasoningConfig:
    enable_thinking: bool

    @classmethod
    def from_dict(cls, raw: Any) -> ReasoningConfig:
        data = _mapping(raw, "preprocessing.reasoning")
        _reject_unknown(data, {"enable_thinking"}, "preprocessing.reasoning")

        return cls(
            enable_thinking=_require_bool(data.get("enable_thinking"), "preprocessing.reasoning.enable_thinking")
        )


@dataclass(frozen=True)
class SftTargetMaskingPolicyConfig:
    min_guaranteed_assistant_chars: int
    loss_on_short_assistant_reply_prob: float
    short_response_sampling_seed: int

    @classmethod
    def from_dict(cls, raw: Any) -> SftTargetMaskingPolicyConfig:
        data = _mapping(raw, "preprocessing.masking.policies.sft_target")
        _reject_unknown(
            data,
            {
                "min_guaranteed_assistant_chars",
                "loss_on_short_assistant_reply_prob",
                "short_response_sampling_seed",
            },
            "preprocessing.masking.policies.sft_target",
        )

        return cls(
            min_guaranteed_assistant_chars=_require_non_negative_int(
                data.get("min_guaranteed_assistant_chars"),
                "preprocessing.masking.policies.sft_target.min_guaranteed_assistant_chars",
            ),
            loss_on_short_assistant_reply_prob=_require_probability(
                data.get("loss_on_short_assistant_reply_prob"),
                "preprocessing.masking.policies.sft_target.loss_on_short_assistant_reply_prob",
            ),
            short_response_sampling_seed=_require_int(
                data.get("short_response_sampling_seed"),
                "preprocessing.masking.policies.sft_target.short_response_sampling_seed",
            ),
        )


@dataclass(frozen=True)
class MaskingPoliciesConfig:
    sft_target: SftTargetMaskingPolicyConfig

    @classmethod
    def from_dict(cls, raw: Any) -> MaskingPoliciesConfig:
        data = _mapping(raw, "preprocessing.masking.policies")
        _reject_unknown(data, {"sft_target", "sft_tool"}, "preprocessing.masking.policies")

        return cls(
            sft_target=SftTargetMaskingPolicyConfig.from_dict(data.get("sft_target"))
        )


@dataclass(frozen=True)
class MaskingConfig:
    ignore_index: int
    require_positive_supervised_tokens: bool
    policies: MaskingPoliciesConfig

    @classmethod
    def from_dict(cls, raw: Any) -> MaskingConfig:
        data = _mapping(raw, "preprocessing.masking")
        _reject_unknown(
            data,
            {"ignore_index", "require_positive_supervised_tokens", "policies"},
            "preprocessing.masking",
        )

        return cls(
            ignore_index=_require_int(data.get("ignore_index"), "preprocessing.masking.ignore_index"),
            require_positive_supervised_tokens=_require_bool(
                data.get("require_positive_supervised_tokens"),
                "preprocessing.masking.require_positive_supervised_tokens",
            ),
            policies=MaskingPoliciesConfig.from_dict(data.get("policies")),
        )


@dataclass(frozen=True)
class PreprocessingConfig:
    raw: PreprocessingRawConfig
    sequence: SequenceConfig
    rendering: RenderingConfig
    reasoning: ReasoningConfig
    masking: MaskingConfig

    @classmethod
    def from_dict(cls, raw: Any) -> PreprocessingConfig:
        data = _mapping(raw, "preprocessing")
        _reject_unknown(data, {"raw", "sequence", "rendering", "reasoning", "masking"}, "preprocessing")

        return cls(
            raw=PreprocessingRawConfig.from_dict(data.get("raw")),
            sequence=SequenceConfig.from_dict(data.get("sequence")),
            rendering=RenderingConfig.from_dict(data.get("rendering")),
            reasoning=ReasoningConfig.from_dict(data.get("reasoning")),
            masking=MaskingConfig.from_dict(data.get("masking")),
        )


@dataclass(frozen=True)
class LossRouteConfig:
    type: Literal["sft_ce", "dpo"]

    @classmethod
    def from_dict(cls, raw: Any, *, route_name: str) -> LossRouteConfig:
        data = _mapping(raw, f"loss_routing.routes.{route_name}")
        _reject_unknown(data, {"type"}, f"loss_routing.routes.{route_name}")

        route_type = _require_non_empty_str(data.get("type"), f"loss_routing.routes.{route_name}.type")
        if route_type not in {"sft_ce", "dpo"}:
            raise ConfigError(f"loss_routing.routes.{route_name}.type must be `sft_ce` or `dpo`")

        return cls(type=route_type)  # type: ignore[arg-type]


@dataclass(frozen=True)
class DpoReferenceConfig:
    mode: Literal["disable_adapter"]
    cache_enabled: bool
    cache_refresh: bool
    cache_required: bool

    @classmethod
    def from_dict(cls, raw: Any) -> DpoReferenceConfig:
        data = {} if raw is None else _mapping(raw, "loss_routing.dpo.reference")
        _reject_unknown(
            data,
            {"mode", "cache_enabled", "cache_refresh", "cache_required"},
            "loss_routing.dpo.reference",
        )

        mode = data.get("mode", "disable_adapter")
        if mode != "disable_adapter":
            raise ConfigError("loss_routing.dpo.reference.mode must be disable_adapter")

        return cls(
            mode=mode,  # type: ignore[arg-type]
            cache_enabled=_optional_bool(data.get("cache_enabled"), "loss_routing.dpo.reference.cache_enabled", True),
            cache_refresh=_optional_bool(data.get("cache_refresh"), "loss_routing.dpo.reference.cache_refresh", False),
            cache_required=_optional_bool(data.get("cache_required"), "loss_routing.dpo.reference.cache_required", False),
        )


@dataclass(frozen=True)
class DpoConfig:
    beta: float
    reference: DpoReferenceConfig

    @classmethod
    def from_dict(cls, raw: Any) -> DpoConfig:
        data = {} if raw is None else _mapping(raw, "loss_routing.dpo")
        _reject_unknown(data, {"beta", "reference"}, "loss_routing.dpo")

        beta = _optional_float(data.get("beta"), "loss_routing.dpo.beta", 0.1)
        if beta <= 0.0:
            raise ConfigError("loss_routing.dpo.beta must be positive")

        return cls(
            beta=beta,
            reference=DpoReferenceConfig.from_dict(data.get("reference")),
        )



@dataclass(frozen=True)
class LossRoutingConfig:
    routes: dict[str, LossRouteConfig]
    dpo: DpoConfig

    @classmethod
    def from_dict(cls, raw: Any) -> LossRoutingConfig:
        data = _mapping(raw, "loss_routing")
        _reject_unknown(data, {"routes", "dpo"}, "loss_routing")

        routes_raw = _mapping(data.get("routes"), "loss_routing.routes")
        if not routes_raw:
            raise ConfigError("loss_routing.routes must be a non-empty mapping")

        routes: dict[str, LossRouteConfig] = {}
        for route_name, route_raw in routes_raw.items():
            if route_name not in {"sft_target", "sft_tool", "dpo_target"}:
                raise ConfigError(f"Unsupported active loss route: {route_name}")
            
            routes[route_name] = LossRouteConfig.from_dict(route_raw, route_name=route_name)

        return cls(routes=routes, dpo=DpoConfig.from_dict(data.get("dpo")))


@dataclass(frozen=True)
class TrainingLoopConfig:
    num_epochs: int
    per_device_train_batch_size: int
    drop_last: bool
    gradient_accumulation_steps: int
    learning_rate: float
    adamw_betas: tuple[float, float]
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    max_grad_norm: float

    @classmethod
    def from_dict(cls, raw: Any) -> TrainingLoopConfig:
        data = _mapping(raw, "training")
        _reject_unknown(
            data,
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

        warmup_ratio = _require_float(data.get("warmup_ratio"), "training.warmup_ratio")
        if warmup_ratio < 0.0 or warmup_ratio > 1.0:
            raise ConfigError("training.warmup_ratio must be between 0 and 1")

        max_grad_norm = _require_float(data.get("max_grad_norm"), "training.max_grad_norm")
        if max_grad_norm < 0.0:
            raise ConfigError("training.max_grad_norm must be >= 0")

        weight_decay = _require_float(data.get("weight_decay"), "training.weight_decay")
        if weight_decay < 0.0:
            raise ConfigError("training.weight_decay must be >= 0")

        return cls(
            num_epochs=_require_positive_int(data.get("num_epochs"), "training.num_epochs"),
            per_device_train_batch_size=_require_positive_int(
                data.get("per_device_train_batch_size"),
                "training.per_device_train_batch_size",
            ),
            drop_last=_require_bool(data.get("drop_last"), "training.drop_last"),
            gradient_accumulation_steps=_require_positive_int(
                data.get("gradient_accumulation_steps"),
                "training.gradient_accumulation_steps",
            ),
            learning_rate=_require_positive_float(data.get("learning_rate"), "training.learning_rate"),
            adamw_betas=_adamw_betas(data.get("adamw_betas")),
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=_require_non_empty_str(data.get("lr_scheduler_type"), "training.lr_scheduler_type"),
            max_grad_norm=max_grad_norm,
        )


@dataclass(frozen=True)
class FsdpConfig:
    sharding_strategy: str
    mixed_precision: str
    cpu_offload: bool
    activation_checkpointing: bool
    state_dict_type: str
    use_orig_params: bool
    limit_all_gathers: bool
    auto_wrap_policy: str
    transformer_cls_names_to_wrap: tuple[str, ...]
    cpu_ram_efficient_loading: bool
    sync_module_states: bool

    @classmethod
    def from_dict(cls, raw: Any) -> FsdpConfig:
        data = _mapping(raw, "distributed.fsdp")
        _reject_unknown(
            data,
            {
                "sharding_strategy",
                "mixed_precision",
                "cpu_offload",
                "activation_checkpointing",
                "state_dict_type",
                "use_orig_params",
                "limit_all_gathers",
                "auto_wrap_policy",
                "transformer_cls_names_to_wrap",
                "cpu_ram_efficient_loading",
                "sync_module_states",
            },
            "distributed.fsdp",
        )

        state_dict_type = _require_non_empty_str(data.get("state_dict_type"), "distributed.fsdp.state_dict_type")
        if state_dict_type != "sharded_state_dict":
            raise ConfigError("distributed.fsdp.state_dict_type must be sharded_state_dict for distributed optimizer state")

        cpu_ram_efficient_loading = _require_bool(
            data.get("cpu_ram_efficient_loading"),
            "distributed.fsdp.cpu_ram_efficient_loading",
        )
        sync_module_states = _require_bool(data.get("sync_module_states"), "distributed.fsdp.sync_module_states")

        if cpu_ram_efficient_loading and sync_module_states is not True:
            raise ConfigError("distributed.fsdp.cpu_ram_efficient_loading=true requires sync_module_states=true")

        return cls(
            sharding_strategy=_require_non_empty_str(data.get("sharding_strategy"), "distributed.fsdp.sharding_strategy"),
            mixed_precision=_require_non_empty_str(data.get("mixed_precision"), "distributed.fsdp.mixed_precision"),
            cpu_offload=_require_bool(data.get("cpu_offload"), "distributed.fsdp.cpu_offload"),
            activation_checkpointing=_require_bool(
                data.get("activation_checkpointing"),
                "distributed.fsdp.activation_checkpointing",
            ),
            state_dict_type=state_dict_type,
            use_orig_params=_require_bool(data.get("use_orig_params"), "distributed.fsdp.use_orig_params"),
            limit_all_gathers=_require_bool(data.get("limit_all_gathers"), "distributed.fsdp.limit_all_gathers"),
            auto_wrap_policy=_require_non_empty_str(data.get("auto_wrap_policy"), "distributed.fsdp.auto_wrap_policy"),
            transformer_cls_names_to_wrap=_string_tuple(
                data.get("transformer_cls_names_to_wrap"),
                "distributed.fsdp.transformer_cls_names_to_wrap",
                allow_empty=False,
            ),
            cpu_ram_efficient_loading=cpu_ram_efficient_loading,
            sync_module_states=sync_module_states,
        )


@dataclass(frozen=True)
class DistributedConfig:
    fsdp: FsdpConfig

    @classmethod
    def from_dict(cls, raw: Any) -> DistributedConfig:
        data = _mapping(raw, "distributed")
        _reject_unknown(data, {"fsdp"}, "distributed")

        return cls(fsdp=FsdpConfig.from_dict(data.get("fsdp")))


@dataclass(frozen=True)
class StandardEvalConfig:
    max_batches: int | None

    @classmethod
    def from_dict(cls, raw: Any) -> StandardEvalConfig:
        data = _mapping(raw, "eval.standard")
        _reject_unknown(data, {"max_batches"}, "eval.standard")

        return cls(
            max_batches=_optional_positive_int(data.get("max_batches"), "eval.standard.max_batches")
        )


@dataclass(frozen=True)
class BfclGenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool

    @classmethod
    def from_dict(cls, raw: Any) -> BfclGenerationConfig:
        data = _mapping(raw, "eval.bfcl.generation")
        _reject_unknown(
            data,
            {"max_new_tokens", "temperature", "top_p", "do_sample"},
            "eval.bfcl.generation",
        )

        top_p = _require_float(data.get("top_p"), "eval.bfcl.generation.top_p")
        if top_p <= 0.0 or top_p > 1.0:
            raise ConfigError("eval.bfcl.generation.top_p must be > 0 and <= 1")

        temperature = _require_float(data.get("temperature"), "eval.bfcl.generation.temperature")
        if temperature < 0.0:
            raise ConfigError("eval.bfcl.generation.temperature must be >= 0")

        return cls(
            max_new_tokens=_require_positive_int(data.get("max_new_tokens"), "eval.bfcl.generation.max_new_tokens"),
            temperature=temperature,
            top_p=top_p,
            do_sample=_require_bool(data.get("do_sample"), "eval.bfcl.generation.do_sample"),
        )


@dataclass(frozen=True)
class BfclEvalConfig:
    enabled: bool
    run_every_n_validations: int
    include_multi_turn: bool
    categories: tuple[str, ...] | None
    limit: int | None
    generation: BfclGenerationConfig

    @classmethod
    def from_dict(cls, raw: Any) -> BfclEvalConfig:
        data = _mapping(raw, "eval.bfcl")
        _reject_unknown(
            data,
            {"enabled", "run_every_n_validations", "include_multi_turn", "categories", "limit", "generation"},
            "eval.bfcl",
        )

        categories = data.get("categories")
        if categories is None:
            parsed_categories = None
        else:
            parsed_categories = _string_tuple(categories, "eval.bfcl.categories", allow_empty=False)

        return cls(
            enabled=_require_bool(data.get("enabled"), "eval.bfcl.enabled"),
            run_every_n_validations=_require_positive_int(
                data.get("run_every_n_validations"),
                "eval.bfcl.run_every_n_validations",
            ),
            include_multi_turn=_require_bool(data.get("include_multi_turn"), "eval.bfcl.include_multi_turn"),
            categories=parsed_categories,
            limit=_optional_positive_int(data.get("limit"), "eval.bfcl.limit"),
            generation=BfclGenerationConfig.from_dict(data.get("generation")),
        )


@dataclass(frozen=True)
class EvalConfig:
    every_train_steps: int
    standard: StandardEvalConfig
    bfcl: BfclEvalConfig

    @classmethod
    def from_dict(cls, raw: Any) -> EvalConfig:
        data = _mapping(raw, "eval")
        _reject_unknown(data, {"every_train_steps", "standard", "bfcl"}, "eval")

        return cls(
            every_train_steps=_require_positive_int(data.get("every_train_steps"), "eval.every_train_steps"),
            standard=StandardEvalConfig.from_dict(data.get("standard")),
            bfcl=BfclEvalConfig.from_dict(data.get("bfcl")),
        )


@dataclass(frozen=True)
class ResumeConfig:
    enabled: bool
    strict_config: bool
    strict_dataset_hash: bool
    strict_template_hash: bool
    strict_model_source_hash: bool

    @classmethod
    def from_dict(cls, raw: Any) -> ResumeConfig:
        data = _mapping(raw, "checkpointing.resume")
        _reject_unknown(
            data,
            {"enabled", "strict_config", "strict_dataset_hash", "strict_template_hash", "strict_model_source_hash"},
            "checkpointing.resume",
        )

        return cls(
            enabled=_require_bool(data.get("enabled"), "checkpointing.resume.enabled"),
            strict_config=_require_bool(data.get("strict_config"), "checkpointing.resume.strict_config"),
            strict_dataset_hash=_require_bool(
                data.get("strict_dataset_hash"),
                "checkpointing.resume.strict_dataset_hash",
            ),
            strict_template_hash=_require_bool(
                data.get("strict_template_hash"),
                "checkpointing.resume.strict_template_hash",
            ),
            strict_model_source_hash=_optional_bool(
                data.get("strict_model_source_hash"),
                "checkpointing.resume.strict_model_source_hash",
                True,
            ),
        )


@dataclass(frozen=True)
class CheckpointingConfig:
    save_every_n_validations: int
    save_total_limit: int | None
    resume: ResumeConfig

    @classmethod
    def from_dict(cls, raw: Any) -> CheckpointingConfig:
        data = _mapping(raw, "checkpointing")
        _reject_unknown(data, {"save_every_n_validations", "save_total_limit", "resume"}, "checkpointing")

        return cls(
            save_every_n_validations=_require_positive_int(
                data.get("save_every_n_validations"),
                "checkpointing.save_every_n_validations",
            ),
            save_total_limit=_optional_positive_int(data.get("save_total_limit"), "checkpointing.save_total_limit"),
            resume=ResumeConfig.from_dict(data.get("resume")),
        )


@dataclass(frozen=True)
class MlflowAsyncLoggingConfig:
    enabled: bool
    queue_max_items: int
    flush_timeout_seconds: int
    fail_on_worker_error: bool

    @classmethod
    def from_dict(cls, raw: Any) -> MlflowAsyncLoggingConfig:
        data = _mapping(raw, "mlflow.async_logging")
        _reject_unknown(
            data,
            {"enabled", "queue_max_items", "flush_timeout_seconds", "fail_on_worker_error"},
            "mlflow.async_logging",
        )

        return cls(
            enabled=_require_bool(data.get("enabled"), "mlflow.async_logging.enabled"),
            queue_max_items=_require_positive_int(data.get("queue_max_items"), "mlflow.async_logging.queue_max_items"),
            flush_timeout_seconds=_require_positive_int(
                data.get("flush_timeout_seconds"),
                "mlflow.async_logging.flush_timeout_seconds",
            ),
            fail_on_worker_error=_require_bool(
                data.get("fail_on_worker_error"),
                "mlflow.async_logging.fail_on_worker_error",
            ),
        )


@dataclass(frozen=True)
class MlflowConfig:
    tracking_uri: str
    resume_run_id: str | None
    async_logging: MlflowAsyncLoggingConfig
    log_rendered_samples: bool

    @classmethod
    def from_dict(cls, raw: Any) -> MlflowConfig:
        data = _mapping(raw, "mlflow")
        _reject_unknown(
            data,
            {"tracking_uri", "resume_run_id", "async_logging", "log_rendered_samples"},
            "mlflow",
        )

        return cls(
            tracking_uri=_require_non_empty_str(data.get("tracking_uri"), "mlflow.tracking_uri"),
            resume_run_id=_optional_str(data.get("resume_run_id"), "mlflow.resume_run_id"),
            async_logging=MlflowAsyncLoggingConfig.from_dict(data.get("async_logging")),
            log_rendered_samples=_require_bool(data.get("log_rendered_samples"), "mlflow.log_rendered_samples"),
        )


@dataclass(frozen=True)
class RegistrySelectionConfig:
    metric: str
    mode: Literal["min", "max"]

    @classmethod
    def from_dict(cls, raw: Any) -> RegistrySelectionConfig:
        data = _mapping(raw, "registry.selection")
        _reject_unknown(data, {"metric", "mode"}, "registry.selection")

        metric = _require_non_empty_str(data.get("metric"), "registry.selection.metric")
        mode = _require_non_empty_str(data.get("mode"), "registry.selection.mode")

        if not (
            metric in REGISTRY_SELECTION_METRICS
            or metric.startswith("eval/bfcl/")
            and metric.endswith(("/accuracy", "/total"))
        ):
            raise ConfigError("registry.selection.metric must be an emitted ordinary/BFCL checkpoint metric")

        if mode not in {"min", "max"}:
            raise ConfigError("registry.selection.mode must be min or max")

        return cls(metric=metric, mode=mode)  # type: ignore[arg-type]


@dataclass(frozen=True)
class RegistryConfig:
    register_every_n_checkpoints: int
    selection: RegistrySelectionConfig

    @classmethod
    def from_dict(cls, raw: Any) -> RegistryConfig:
        data = _mapping(raw, "registry")
        _reject_unknown(data, {"register_every_n_checkpoints", "selection"}, "registry")

        return cls(
            register_every_n_checkpoints=_require_positive_int(
                data.get("register_every_n_checkpoints"),
                "registry.register_every_n_checkpoints",
            ),
            selection=RegistrySelectionConfig.from_dict(data.get("selection")),
        )


@dataclass(frozen=True)
class ProgressConfig:
    enabled: bool = True

    @classmethod
    def from_dict(cls, raw: Any) -> ProgressConfig:
        if raw is None:
            return cls()
        data = _mapping(raw, "progress")
        _reject_unknown(data, {"enabled"}, "progress")
        return cls(enabled=_require_bool(data.get("enabled"), "progress.enabled"))


@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    model: ModelConfig
    tokenizer: TokenizerConfig
    lora: LoraConfig
    preprocessing: PreprocessingConfig
    loss_routing: LossRoutingConfig
    training: TrainingLoopConfig
    distributed: DistributedConfig
    eval: EvalConfig
    checkpointing: CheckpointingConfig
    mlflow: MlflowConfig
    registry: RegistryConfig
    progress: ProgressConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        data = _mapping(raw, "top-level config")

        missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(data))
        if missing:
            raise ConfigError(f"Missing required top-level config sections: {missing}")

        _reject_unknown(data, TOP_LEVEL_KEYS, "top-level config")

        config = cls(
            project=ProjectConfig.from_dict(data.get("project")),
            model=ModelConfig.from_dict(data.get("model")),
            tokenizer=TokenizerConfig.from_dict(data.get("tokenizer")),
            lora=LoraConfig.from_dict(data.get("lora")),
            preprocessing=PreprocessingConfig.from_dict(data.get("preprocessing")),
            loss_routing=LossRoutingConfig.from_dict(data.get("loss_routing")),
            training=TrainingLoopConfig.from_dict(data.get("training")),
            distributed=DistributedConfig.from_dict(data.get("distributed")),
            eval=EvalConfig.from_dict(data.get("eval")),
            checkpointing=CheckpointingConfig.from_dict(data.get("checkpointing")),
            mlflow=MlflowConfig.from_dict(data.get("mlflow")),
            registry=RegistryConfig.from_dict(data.get("registry")),
            progress=ProgressConfig.from_dict(data.get("progress")),
        )

        config._validate_cross_fields()
        return config

    def to_dict(self) -> dict[str, Any]:
        return _to_plain_data(asdict(self))

    @property
    def output_dir(self) -> Path:
        return self.project.output_dir

    @property
    def pretokenized_dir(self) -> Path:
        return self.output_dir / "pretokenized"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def bfcl_rows_path(self) -> Path:
        return self.output_dir / "eval" / "bfcl_rows.jsonl"

    @property
    def ignore_index(self) -> int:
        return self.preprocessing.masking.ignore_index

    def _validate_cross_fields(self) -> None:
        if self.model.freeze_lm_head and "lm_head" in self.lora.modules_to_save:
            raise ConfigError("lora.modules_to_save cannot include lm_head when model.freeze_lm_head=true")

        metric = self.registry.selection.metric

        if metric.startswith("eval/bfcl/") and not self.eval.bfcl.enabled:
            raise ConfigError("registry.selection.metric uses BFCL while eval.bfcl.enabled=false")

        if metric.startswith("eval/bfcl/"):
            bfcl_every = self.eval.bfcl.run_every_n_validations
            checkpoint_every = self.checkpointing.save_every_n_validations

            if checkpoint_every % bfcl_every != 0:
                raise ConfigError("BFCL registry selection requires every checkpoint boundary to run BFCL")


def _to_plain_data(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain_data(item) for item in value]
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return dict(value)


def _reject_unknown(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"Unknown {name} fields: {unknown}")


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false")
    return value


def _optional_bool(value: Any, name: str, default: bool) -> bool:
    if value is None:
        return default
    return _require_bool(value, name)


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _require_non_negative_int(value: Any, name: str) -> int:
    parsed = _require_int(value, name)
    if parsed < 0:
        raise ConfigError(f"{name} must be >= 0")
    return parsed


def _require_positive_int(value: Any, name: str) -> int:
    parsed = _require_int(value, name)
    if parsed <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return parsed


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, name)


def _require_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a number")

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number") from exc


def _optional_float(value: Any, name: str, default: float) -> float:
    if value is None:
        return default
    return _require_float(value, name)


def _require_positive_float(value: Any, name: str) -> float:
    parsed = _require_float(value, name)
    if parsed <= 0:
        raise ConfigError(f"{name} must be a positive number")
    return parsed


def _require_probability(value: Any, name: str) -> float:
    parsed = _require_float(value, name)
    if parsed < 0.0 or parsed > 1.0:
        raise ConfigError(f"{name} must be between 0 and 1")
    return parsed


def _require_non_empty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _optional_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be a string or null")
    return value


def _path(value: Any, name: str) -> Path:
    text = _require_non_empty_str(value, name)
    return Path(text).expanduser()


def _string_tuple(value: Any, name: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{name} must be a list of strings")

    if not allow_empty and not value:
        raise ConfigError(f"{name} must be a non-empty list of strings")

    parsed: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ConfigError(f"{name}[{index}] must be a non-empty string")
        parsed.append(item)

    return tuple(parsed)


def _adamw_betas(value: Any) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError("training.adamw_betas must be a two-item list")

    betas = tuple(_require_float(item, f"training.adamw_betas[{index}]") for index, item in enumerate(value))

    for index, beta in enumerate(betas):
        if beta < 0.0 or beta >= 1.0:
            raise ConfigError(f"training.adamw_betas[{index}] must be >= 0 and < 1")

    return betas  # type: ignore[return-value]

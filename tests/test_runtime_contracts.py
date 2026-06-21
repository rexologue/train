from __future__ import annotations

from types import SimpleNamespace

import torch

from config import ConfigError, load_config
from checkpointing.save import trainable_state_dict
from conftest import example_config
from trainer.distributed import configure_ignored_tied_embeddings, tied_frozen_embedding_modules
from trainer.modeling import configure_gradient_checkpointing, disable_model_kv_cache, freeze_configured_modules


def test_example_config_loads_with_dpo_route() -> None:
    config = load_config("configs/config.example.yaml")

    assert config.loss_routing.routes["dpo_target"].type == "dpo"
    assert config.loss_routing.dpo.reference.cache_enabled is True
    assert config.model.gradient_checkpointing is False
    assert config.distributed.fsdp.activation_checkpointing is True
    assert config.eval.bfcl.enabled is False
    assert config.eval.bfcl.limit == 100
    assert config.registry.selection.metric == "eval/sft/loss"
    assert config.output_dir.as_posix().endswith("qwen35-a3b-lora-sft-dpo-v1")


def test_config_rejects_double_activation_checkpointing() -> None:
    try:
        example_config(
            model={"gradient_checkpointing": True},
            distributed={"fsdp": {"activation_checkpointing": True}},
        )
    except ConfigError as exc:
        assert "gradient_checkpointing" in str(exc)
        assert "activation_checkpointing" in str(exc)
    else:
        raise AssertionError("expected double activation checkpointing config to fail")


def test_config_rejects_accelerator_style_fsdp_mixed_precision() -> None:
    try:
        example_config(distributed={"fsdp": {"mixed_precision": "no"}})
    except ConfigError as exc:
        assert "distributed.fsdp.mixed_precision" in str(exc)
        assert "fp32" in str(exc)
    else:
        raise AssertionError("expected invalid FSDP mixed_precision config to fail")


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 2)
        self.lm_head = torch.nn.Linear(2, 4)
        self.router = torch.nn.Linear(2, 2)
        self.gate = torch.nn.Linear(2, 2)
        self.gate_proj = torch.nn.Linear(2, 2)
        self.config = SimpleNamespace(use_cache=True)
        self.gradient_checkpointing_kwargs = None
        self.input_grads_enabled = False

    def get_input_embeddings(self):
        return self.embedding

    def gradient_checkpointing_enable(self, **kwargs):
        self.gradient_checkpointing_kwargs = kwargs

    def enable_input_require_grads(self):
        self.input_grads_enabled = True


class TiedEmbeddingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 2)
        self.lm_head = torch.nn.Linear(2, 4, bias=False)
        self.lm_head.weight = self.embedding.weight

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head


def test_frozen_tied_embeddings_are_ignored_for_fsdp() -> None:
    model = TiedEmbeddingModel()
    model.embedding.weight.requires_grad = False

    ignored = tied_frozen_embedding_modules(model)

    assert ignored == (model.embedding, model.lm_head)


def test_trainable_tied_embeddings_are_not_ignored_for_fsdp() -> None:
    model = TiedEmbeddingModel()

    assert tied_frozen_embedding_modules(model) == ()


def test_prepare_configures_fsdp_ignore_for_tied_embeddings() -> None:
    model = TiedEmbeddingModel()
    model.embedding.weight.requires_grad = False
    plugin = SimpleNamespace(ignored_modules=None)

    ignored = configure_ignored_tied_embeddings(plugin, (model,))

    assert plugin.ignored_modules == (model.embedding, model.lm_head)
    assert ignored == (model.embedding, model.lm_head)


def test_adapter_state_dict_contains_only_trainable_parameters() -> None:
    model = TiedEmbeddingModel()
    model.embedding.weight.requires_grad = False
    model.adapter = torch.nn.Linear(2, 2, bias=False)

    state = trainable_state_dict(model)

    assert sorted(state) == ["adapter.weight"]
    assert state["adapter.weight"].device.type == "cpu"


def test_model_runtime_flags_are_applied() -> None:
    model = TinyModel()
    config = SimpleNamespace(
        model=SimpleNamespace(
            gradient_checkpointing=True,
            freeze_embeddings=True,
            freeze_lm_head=True,
            freeze_router=True,
        )
    )

    configure_gradient_checkpointing(model, config)
    freeze_configured_modules(model, config)

    assert model.config.use_cache is False
    assert model.gradient_checkpointing_kwargs == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    assert model.input_grads_enabled is True
    assert model.embedding.weight.requires_grad is False
    assert model.lm_head.weight.requires_grad is False
    assert model.router.weight.requires_grad is False
    assert model.gate.weight.requires_grad is False
    assert model.gate_proj.weight.requires_grad is True


def test_model_kv_cache_is_disabled_without_gradient_checkpointing() -> None:
    model = TinyModel()

    disable_model_kv_cache(model)

    assert model.config.use_cache is False

from __future__ import annotations

from types import SimpleNamespace

import torch

from config import ConfigError, load_config
from conftest import example_config
from trainer.modeling import configure_gradient_checkpointing, freeze_configured_modules


def test_example_config_loads_with_dpo_route() -> None:
    config = load_config("configs/config.example.yaml")

    assert config.loss_routing.routes["dpo_target"].type == "dpo"
    assert config.loss_routing.dpo.reference.cache_enabled is True
    assert config.model.gradient_checkpointing is False
    assert config.distributed.fsdp.activation_checkpointing is True
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

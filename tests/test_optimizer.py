from __future__ import annotations

import torch

from config import load_config
from trainer.modeling import build_optimizer, training_steps_for_epochs


def test_build_optimizer_uses_configured_adamw_betas():
    config = load_config("configs/config.example.yaml")
    config.raw["training"]["adamw_betas"] = [0.8, 0.95]
    model = torch.nn.Linear(2, 1)

    optimizer = build_optimizer(config, model)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["betas"] == (0.8, 0.95)


def test_training_steps_are_derived_per_epoch_without_cross_epoch_padding():
    config = load_config("configs/config.example.yaml")
    config.raw["training"]["num_epochs"] = 3
    config.raw["training"]["gradient_accumulation_steps"] = 16

    assert training_steps_for_epochs(config, list(range(10)), num_processes=2) == 3

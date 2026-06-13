from __future__ import annotations

import torch

from config import load_config
from trainer.modeling import build_optimizer


def test_build_optimizer_uses_configured_adamw_betas():
    config = load_config("configs/config.preprocess.yaml")
    config.raw["training"]["adamw_betas"] = [0.8, 0.95]
    model = torch.nn.Linear(2, 1)

    optimizer = build_optimizer(config, model)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["betas"] == (0.8, 0.95)

from __future__ import annotations

from types import SimpleNamespace

import torch

from losses.sft import sft_cross_entropy_loss


class DummyCausalLm:
    def __init__(self):
        self.seen_kwargs = None

    def __call__(self, **kwargs):
        self.seen_kwargs = kwargs
        return SimpleNamespace(loss=torch.tensor(1.25))


def test_sft_loss_passes_only_model_inputs():
    model = DummyCausalLm()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": torch.tensor([[-100, 2, 3]]),
        "loss_kind": "sft_target",
        "sample_id": ["sample_0"],
    }

    loss = sft_cross_entropy_loss(model, batch)

    assert loss.item() == 1.25
    assert model.seen_kwargs == {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
    }

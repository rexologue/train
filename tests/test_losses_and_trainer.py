from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import losses.dpo as dpo_module
from losses.dpo import dpo_loss, sequence_logps
from losses.sft import sft_cross_entropy_loss
from trainer.trainer import RoutedTrainer
from conftest import example_config


class DummySftModel:
    def __init__(self) -> None:
        self.seen_kwargs = None

    def __call__(self, **kwargs):
        self.seen_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        logits = torch.zeros((*input_ids.shape, 8), dtype=torch.float32)
        logits[:, 0, 2] = 4.0
        logits[:, 1, 3] = 4.0
        return SimpleNamespace(logits=logits)


class TinyLogitModel:
    def __call__(self, *, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        logits = torch.zeros((*input_ids.shape, 16), dtype=torch.float32, device=input_ids.device)
        logits[..., 5] = 3.0
        logits[..., 6] = 1.0
        return SimpleNamespace(logits=logits)


class DummyAccelerator:
    device = torch.device("cpu")

    def unwrap_model(self, model):
        return model


def test_sft_loss_computes_masked_ce_without_model_labels() -> None:
    model = DummySftModel()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": torch.tensor([[-100, 2, 3]]),
        "loss_kind": "sft_target",
        "sample_id": ["sample_0"],
    }

    loss = sft_cross_entropy_loss(model, batch)

    assert loss.item() < 0.2
    assert model.seen_kwargs == {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "use_cache": False,
    }


def test_dpo_loss_uses_cached_reference_logprobs() -> None:
    model = TinyLogitModel()
    batch = {
        "loss_kind": "dpo_target",
        "chosen_input_ids": torch.tensor([[1, 5]]),
        "chosen_attention_mask": torch.tensor([[1, 1]]),
        "chosen_labels": torch.tensor([[-100, 5]]),
        "rejected_input_ids": torch.tensor([[1, 6]]),
        "rejected_attention_mask": torch.tensor([[1, 1]]),
        "rejected_labels": torch.tensor([[-100, 6]]),
        "chosen_ref_logp": torch.tensor([-1.0]),
        "rejected_ref_logp": torch.tensor([-1.0]),
    }

    result = dpo_loss(
        model,
        batch,
        beta=0.1,
        ignore_index=-100,
        accelerator=DummyAccelerator(),
        cache_required=True,
    )

    assert torch.isfinite(result.loss)
    assert result.metrics["dpo/policy_chosen_logp"] > result.metrics["dpo/policy_rejected_logp"]
    assert result.metrics["dpo/accuracy"] == 1.0


def test_sequence_logps_does_not_materialize_full_log_softmax(monkeypatch) -> None:
    def fail_log_softmax(*args, **kwargs):
        del args, kwargs
        raise AssertionError("sequence_logps must avoid full [batch, seq, vocab] log_softmax")

    monkeypatch.setattr(dpo_module.F, "log_softmax", fail_log_softmax)
    logps = sequence_logps(
        TinyLogitModel(),
        input_ids=torch.tensor([[1, 5, 6]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        labels=torch.tensor([[-100, 5, 6]]),
        ignore_index=-100,
    )

    assert torch.isfinite(logps).all()
    assert logps.shape == (1,)


def test_dpo_loss_requires_precomputed_reference_logprobs() -> None:
    batch = {
        "loss_kind": "dpo_target",
        "chosen_input_ids": torch.tensor([[1, 5]]),
        "chosen_attention_mask": torch.tensor([[1, 1]]),
        "chosen_labels": torch.tensor([[-100, 5]]),
        "rejected_input_ids": torch.tensor([[1, 6]]),
        "rejected_attention_mask": torch.tensor([[1, 1]]),
        "rejected_labels": torch.tensor([[-100, 6]]),
    }

    with pytest.raises(ValueError, match="reference logprobs are missing"):
        dpo_loss(
            TinyLogitModel(),
            batch,
            beta=0.1,
            ignore_index=-100,
            accelerator=DummyAccelerator(),
            cache_required=False,
        )


def test_routed_trainer_dispatches_dpo_and_records_route_metrics() -> None:
    trainer = RoutedTrainer(config=example_config(), accelerator=DummyAccelerator(), cadence=None)
    batch = {
        "loss_kind": "dpo_target",
        "chosen_input_ids": torch.tensor([[1, 5]]),
        "chosen_attention_mask": torch.tensor([[1, 1]]),
        "chosen_labels": torch.tensor([[-100, 5]]),
        "rejected_input_ids": torch.tensor([[1, 6]]),
        "rejected_attention_mask": torch.tensor([[1, 1]]),
        "rejected_labels": torch.tensor([[-100, 6]]),
        "chosen_ref_logp": torch.tensor([-1.0]),
        "rejected_ref_logp": torch.tensor([-1.0]),
    }

    loss = trainer.compute_loss(TinyLogitModel(), batch)

    assert torch.isfinite(loss)
    assert "dpo/reward_margin" in trainer.last_loss_metrics

from __future__ import annotations

from types import SimpleNamespace

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
        logits_to_keep = kwargs.get("logits_to_keep")
        positions = logits_to_keep if logits_to_keep is not None else torch.arange(input_ids.shape[1])
        logits = torch.zeros((input_ids.shape[0], positions.numel(), 8), dtype=torch.float32)
        logits[:, 0, 2] = 4.0
        if logits.shape[1] > 1:
            logits[:, 1, 3] = 4.0
        return SimpleNamespace(logits=logits)


class TinyLogitModel:
    def __call__(self, *, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        logits = torch.zeros((*input_ids.shape, 16), dtype=torch.float32, device=input_ids.device)
        logits[..., 5] = 3.0
        logits[..., 6] = 1.0
        return SimpleNamespace(logits=logits)



class AdapterAwareTinyLogitModel:
    def __init__(self) -> None:
        self.adapter_enabled = True
        self.call_input_shapes = []

    def disable_adapter(self):
        model = self

        class DisableAdapterContext:
            def __enter__(self):
                self.previous = model.adapter_enabled
                model.adapter_enabled = False
                return model

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                model.adapter_enabled = self.previous
                return False

        return DisableAdapterContext()

    def __call__(self, *, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        self.call_input_shapes.append(tuple(input_ids.shape))
        logits = torch.zeros((*input_ids.shape, 16), dtype=torch.float32, device=input_ids.device)
        if self.adapter_enabled:
            logits[..., 5] = 3.0
            logits[..., 6] = 1.0
        else:
            logits[..., 5] = 2.0
            logits[..., 6] = 2.0
        return SimpleNamespace(logits=logits)

class CompactLogitModel:
    def __init__(self) -> None:
        self.seen_logits_to_keep = None

    def __call__(self, *, input_ids, attention_mask=None, logits_to_keep=None, **kwargs):
        del attention_mask, kwargs
        self.seen_logits_to_keep = logits_to_keep.detach().cpu().tolist()
        positions = logits_to_keep if logits_to_keep is not None else torch.arange(input_ids.shape[1], device=input_ids.device)
        logits = torch.zeros((input_ids.shape[0], positions.numel(), 16), dtype=torch.float32, device=input_ids.device)
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
    assert model.seen_kwargs["input_ids"] is batch["input_ids"]
    assert model.seen_kwargs["attention_mask"] is batch["attention_mask"]
    assert model.seen_kwargs["use_cache"] is False
    assert model.seen_kwargs["logits_to_keep"].tolist() == [0, 1]


def test_sft_loss_requests_only_selected_label_positions() -> None:
    model = CompactLogitModel()
    batch = {
        "input_ids": torch.tensor([[1, 5, 2, 6, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        "labels": torch.tensor([[-100, 5, -100, 6, -100]]),
        "loss_kind": "sft_target",
        "sample_id": ["sample_0"],
    }

    loss = sft_cross_entropy_loss(model, batch)

    assert torch.isfinite(loss)
    assert model.seen_logits_to_keep == [0, 2]


def test_dpo_loss_uses_on_the_fly_disabled_adapter_reference() -> None:
    model = AdapterAwareTinyLogitModel()
    batch = {
        "loss_kind": "dpo_target",
        "chosen_input_ids": torch.tensor([[1, 5]]),
        "chosen_attention_mask": torch.tensor([[1, 1]]),
        "chosen_labels": torch.tensor([[-100, 5]]),
        "rejected_input_ids": torch.tensor([[1, 6, 0]]),
        "rejected_attention_mask": torch.tensor([[1, 1, 0]]),
        "rejected_labels": torch.tensor([[-100, 6, -100]]),
    }

    result = dpo_loss(
        model,
        batch,
        beta=0.1,
        ignore_index=-100,
        accelerator=DummyAccelerator(),
    )

    assert torch.isfinite(result.loss)
    assert model.adapter_enabled is True
    assert model.call_input_shapes == [(2, 3), (2, 3)]
    assert result.metrics["dpo/policy_chosen_logp"] > result.metrics["dpo/policy_rejected_logp"]
    assert result.metrics["dpo/ref_chosen_logp"] == result.metrics["dpo/ref_rejected_logp"]
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


def test_sequence_logps_requests_only_selected_label_positions() -> None:
    model = CompactLogitModel()

    logps = sequence_logps(
        model,
        input_ids=torch.tensor([[1, 2, 5, 6, 7], [1, 5, 6, 7, 8]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        labels=torch.tensor([[-100, -100, 5, 6, -100], [-100, 5, -100, -100, -100]]),
        ignore_index=-100,
    )

    assert model.seen_logits_to_keep == [0, 1, 2]
    assert torch.isfinite(logps).all()


def test_dpo_loss_requires_peft_disable_adapter_for_reference() -> None:
    batch = {
        "loss_kind": "dpo_target",
        "chosen_input_ids": torch.tensor([[1, 5]]),
        "chosen_attention_mask": torch.tensor([[1, 1]]),
        "chosen_labels": torch.tensor([[-100, 5]]),
        "rejected_input_ids": torch.tensor([[1, 6]]),
        "rejected_attention_mask": torch.tensor([[1, 1]]),
        "rejected_labels": torch.tensor([[-100, 6]]),
    }

    try:
        dpo_loss(
            TinyLogitModel(),
            batch,
            beta=0.1,
            ignore_index=-100,
            accelerator=DummyAccelerator(),
        )
    except RuntimeError as exc:
        assert "disable_adapter" in str(exc)
    else:
        raise AssertionError("expected non-PEFT DPO model to fail")


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
    }

    loss = trainer.compute_loss(AdapterAwareTinyLogitModel(), batch)

    assert torch.isfinite(loss)
    assert "dpo/reward_margin" in trainer.last_loss_metrics

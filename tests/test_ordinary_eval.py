from __future__ import annotations

from types import SimpleNamespace
from contextlib import nullcontext

import torch

from config import load_config
from eval.ordinary import run_standard_eval
from eval.ordinary import _reduce_eval_totals
from trainer.trainer import RoutedTrainer, TrainerCadence


class FixedLossModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None, labels=None):
        del input_ids, attention_mask, labels
        return SimpleNamespace(loss=torch.tensor(0.5))


class InputMeanLossModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask, labels
        return SimpleNamespace(loss=input_ids.float().mean())


class FakeTrainerAccelerator:
    def accumulate(self, model):
        del model
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def test_standard_eval_respects_max_batches_and_counts_supervised_tokens():
    config = load_config("configs/config.example.yaml")
    config.raw["eval"]["standard"]["max_batches"] = 1
    trainer = RoutedTrainer(
        accelerator=FakeTrainerAccelerator(),
        cadence=TrainerCadence(
            eval_every_train_steps=10,
            checkpoint_every_n_validations=1,
            bfcl_every_n_validations=1,
        )
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": torch.tensor([[-100, 2, 3]]),
        "loss_kind": "sft_target",
    }

    metrics = run_standard_eval(
        model=FixedLossModel(),
        dataloader=[batch, batch],
        trainer=trainer,
        config=config,
    )

    assert metrics["eval/loss"] == 0.5
    assert metrics["eval/batches"] == 1.0
    assert metrics["eval/supervised_tokens"] == 2.0


def test_standard_eval_weights_loss_by_supervised_tokens():
    config = load_config("configs/config.example.yaml")
    config.raw["eval"]["standard"]["max_batches"] = None
    trainer = RoutedTrainer(
        accelerator=FakeTrainerAccelerator(),
        cadence=TrainerCadence(
            eval_every_train_steps=10,
            checkpoint_every_n_validations=1,
            bfcl_every_n_validations=1,
        )
    )
    short_batch = {
        "input_ids": torch.tensor([[2, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "labels": torch.tensor([[2, -100]]),
        "loss_kind": "sft_target",
    }
    long_batch = {
        "input_ids": torch.tensor([[4, 4, 4, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "labels": torch.tensor([[4, 4, 4, 4]]),
        "loss_kind": "sft_target",
    }

    metrics = run_standard_eval(
        model=InputMeanLossModel(),
        dataloader=[short_batch, long_batch],
        trainer=trainer,
        config=config,
    )

    assert metrics["eval/loss"] == 3.6
    assert metrics["eval/supervised_tokens"] == 5.0


def test_reduce_eval_totals_sums_flat_gathered_vectors():
    class FakeAccelerator:
        device = torch.device("cpu")

        def gather_for_metrics(self, values):
            return torch.cat([values, torch.tensor([2.0, 1.0, 5.0, 6.0])])

    reduced = _reduce_eval_totals(
        {"loss_weighted_sum": 1.0, "loss_count": 1, "supervised_tokens": 3, "tokens": 4},
        FakeAccelerator(),
    )

    assert reduced == {
        "loss_weighted_sum": 3.0,
        "loss_count": 2.0,
        "supervised_tokens": 8.0,
        "tokens": 10.0,
    }

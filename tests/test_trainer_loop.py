from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from config import load_config
from trainer.callbacks import TrainerHooks
from trainer.state import TrainerState
from trainer.trainer import RoutedTrainer, TrainerCadence, _gather_step_totals


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask, labels
        return SimpleNamespace(loss=self.weight * input_ids.float().mean())


class FakeAccelerator:
    def accumulate(self, model):
        del model
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def batch(value: int):
    return {
        "input_ids": torch.tensor([[value]], dtype=torch.long),
        "attention_mask": torch.tensor([[1]], dtype=torch.long),
        "labels": torch.tensor([[value]], dtype=torch.long),
        "loss_kind": "sft_target",
        "sample_id": [f"sample_{value}"],
    }


def test_routed_trainer_validation_checkpoint_cadence():
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    events: list[tuple[str, int, int, int]] = []
    metrics_log: list[tuple[int, dict]] = []

    def on_phase(name, state):
        events.append((name, state.global_step, state.validation_index, state.checkpoint_index))

    def standard_eval(model, dataloader, state):
        del model, dataloader, state
        return {"eval/loss": 0.5}

    def bfcl_eval(model, dataloader, state):
        del model, dataloader, state
        return {"eval/bfcl/accuracy": 1.0}

    def save_checkpoint(model, optimizer, state, metrics):
        del model, optimizer, metrics
        return f"checkpoint-{state.checkpoint_index}"

    def log_metrics(metrics, state):
        metrics_log.append((state.global_step, dict(metrics)))

    trainer = RoutedTrainer(
        accelerator=FakeAccelerator(),
        hooks=TrainerHooks(
            on_phase=on_phase,
            run_standard_eval=standard_eval,
            run_bfcl_eval=bfcl_eval,
            save_checkpoint=save_checkpoint,
            log_metrics=log_metrics,
        ),
        cadence=TrainerCadence(
            eval_every_train_steps=2,
            checkpoint_every_n_validations=2,
            bfcl_every_n_validations=1,
        ),
    )

    state = trainer.fit(
        model,
        optimizer,
        [batch(1), batch(2), batch(3)],
        total_steps=5,
        gradient_accumulation_steps=2,
    )

    assert state.global_step == 5
    assert state.validation_index == 2
    assert state.checkpoint_index == 1
    assert state.consumed_batches == 10
    assert events == [
        ("validation:standard:start", 2, 1, 0),
        ("validation:bfcl:start", 2, 1, 0),
        ("validation:end", 2, 1, 0),
        ("validation:standard:start", 4, 2, 0),
        ("validation:bfcl:start", 4, 2, 0),
        ("checkpoint:save:start", 4, 2, 1),
        ("validation:end", 4, 2, 1),
    ]
    assert any(item[1].get("checkpoint/path") == "checkpoint-1" for item in metrics_log)


def test_routed_trainer_does_not_predivide_loss_for_accelerator():
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    seen_backward_losses: list[float] = []

    class RecordingAccelerator(FakeAccelerator):
        def accumulate(self, model):
            del model
            return nullcontext()

        def backward(self, loss):
            seen_backward_losses.append(float(loss.detach().item()))
            (loss / 2).backward()

    trainer = RoutedTrainer(
        accelerator=RecordingAccelerator(),
        cadence=TrainerCadence(
            eval_every_train_steps=10,
            checkpoint_every_n_validations=1,
            bfcl_every_n_validations=1,
        ),
    )

    trainer.fit(
        model,
        optimizer,
        [batch(2), batch(4)],
        total_steps=1,
        gradient_accumulation_steps=2,
    )

    assert seen_backward_losses == [2.0, 4.0]


def test_routed_trainer_fast_forwards_consumed_batches_on_resume():
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics_log: list[dict] = []
    trainer = RoutedTrainer(
        accelerator=FakeAccelerator(),
        hooks=TrainerHooks(log_metrics=lambda metrics, state: metrics_log.append(dict(metrics))),
        cadence=TrainerCadence(
            eval_every_train_steps=10,
            checkpoint_every_n_validations=1,
            bfcl_every_n_validations=1,
        ),
    )

    state = trainer.fit(
        model,
        optimizer,
        [batch(1), batch(2), batch(3)],
        state=TrainerState(global_step=0, consumed_batches=1),
        total_steps=1,
        gradient_accumulation_steps=1,
    )

    assert state.consumed_batches == 2
    assert metrics_log[0]["train/loss"] == 2.0


def test_routed_trainer_consumes_exact_epoch_micro_batches():
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = RoutedTrainer(
        accelerator=FakeAccelerator(),
        cadence=TrainerCadence(
            eval_every_train_steps=10,
            checkpoint_every_n_validations=1,
            bfcl_every_n_validations=None,
        ),
    )

    state = trainer.fit(
        model,
        optimizer,
        [batch(1), batch(2), batch(3)],
        total_steps=4,
        total_micro_batches=6,
        gradient_accumulation_steps=2,
    )

    assert state.global_step == 4
    assert state.consumed_batches == 6


def test_partial_epoch_accumulation_rescales_accelerate_backward_loss():
    seen_backward_losses: list[float] = []

    class ScalingAccelerator(FakeAccelerator):
        gradient_accumulation_steps = 4

        def backward(self, loss):
            seen_backward_losses.append(float(loss.detach().item()))
            (loss / self.gradient_accumulation_steps).backward()

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = RoutedTrainer(
        accelerator=ScalingAccelerator(),
        cadence=TrainerCadence(
            eval_every_train_steps=10,
            checkpoint_every_n_validations=1,
            bfcl_every_n_validations=None,
        ),
    )

    trainer.fit(
        model,
        optimizer,
        [batch(1), batch(2)],
        total_steps=1,
        total_micro_batches=2,
        gradient_accumulation_steps=4,
    )

    assert seen_backward_losses == [2.0, 4.0]


def test_training_step_metrics_are_aggregated_across_ranks():
    class GatherAccelerator:
        device = torch.device("cpu")

        def gather_for_metrics(self, values):
            remote = torch.tensor([4.0, 3.0, 7.0, 5.0, 2.0])
            return torch.cat([values, remote])

    totals = _gather_step_totals(
        GatherAccelerator(),
        accumulated_loss=2.0,
        samples=1,
        tokens=3,
        supervised_tokens=2,
        elapsed=1.0,
    )

    assert totals == {
        "accumulated_loss": 6.0,
        "samples": 4.0,
        "tokens": 10.0,
        "supervised_tokens": 7.0,
        "elapsed": 2.0,
        "processes": 2.0,
    }

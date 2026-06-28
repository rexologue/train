from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable

import torch

from losses.dpo import dpo_loss
from losses.sft import sft_cross_entropy_loss
from trainer.callbacks import TrainerHooks
from trainer.state import TrainerState


@dataclass(frozen=True, slots=True)
class TrainerCadence:
    """Validation/checkpoint cadence expressed in training-loop units."""

    eval_every_train_steps: int
    checkpoint_every_n_validations: int
    bfcl_every_n_validations: int | None

    @classmethod
    def from_config(cls, config: Any) -> "TrainerCadence":
        bfcl_every = config.eval.bfcl.run_every_n_validations if config.eval.bfcl.enabled else None

        return cls(
            eval_every_train_steps=_positive_int(config.eval.every_train_steps, "eval.every_train_steps"),
            checkpoint_every_n_validations=_positive_int(
                config.checkpointing.save_every_n_validations,
                "checkpointing.save_every_n_validations",
            ),
            bfcl_every_n_validations=(
                _positive_int(bfcl_every, "eval.bfcl.run_every_n_validations") if bfcl_every is not None else None
            ),
        )

    def should_validate(self, global_step: int) -> bool:
        return global_step > 0 and global_step % self.eval_every_train_steps == 0

    def should_checkpoint(self, validation_index: int) -> bool:
        return validation_index > 0 and validation_index % self.checkpoint_every_n_validations == 0

    def should_run_bfcl(self, validation_index: int) -> bool:
        return (
            self.bfcl_every_n_validations is not None
            and validation_index > 0
            and validation_index % self.bfcl_every_n_validations == 0
        )


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _cycle_next(iterator: Any, iterable: Iterable[Any]) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(iterable)
        return next(iterator), iterator


def _detach_item(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


class RoutedTrainer:
    def __init__(
        self,
        config: Any | None = None,
        *,
        accelerator: Any,
        hooks: TrainerHooks | None = None,
        cadence: TrainerCadence | None = None,
    ):
        self.config = config
        self.accelerator = accelerator
        self.hooks = hooks or TrainerHooks()
        self.cadence = cadence or (TrainerCadence.from_config(config) if config is not None else None)
        self.last_loss_metrics: dict[str, float] = {}

    def compute_loss(self, model, batch):
        self.last_loss_metrics = {}
        loss_kind = batch.get("loss_kind")
        if loss_kind in {"sft_target", "sft_tool"}:
            ignore_index = int(getattr(self.config, "ignore_index", -100)) if self.config is not None else -100
            return sft_cross_entropy_loss(model, batch, ignore_index=ignore_index)
        if loss_kind == "dpo_target":
            if self.config is None:
                raise ValueError("DPO loss requires trainer config")
            result = dpo_loss(
                model,
                batch,
                beta=float(self.config.loss_routing.dpo.beta),
                ignore_index=int(self.config.ignore_index),
                accelerator=self.accelerator,
            )
            self.last_loss_metrics = result.metrics
            return result.loss
        raise ValueError(f"unknown loss_kind {loss_kind!r}")

    def fit(
        self,
        model: Any,
        optimizer: Any,
        train_dataloader: Iterable[dict[str, Any]],
        *,
        scheduler: Any | None = None,
        valid_dataloader: Iterable[dict[str, Any]] | None = None,
        state: TrainerState | None = None,
        total_steps: int | None = None,
        total_micro_batches: int | None = None,
        gradient_accumulation_steps: int | None = None,
    ) -> TrainerState:
        """Run the routed training loop.

        `global_step` is an optimizer step, not a micro-batch. Validation,
        checkpointing, and registry hooks are expected to hang off this
        boundary, which keeps the training flow linear and resumable.
        """

        if self.cadence is None:
            raise ValueError("trainer cadence must be configured")
        if total_steps is None:
            raise ValueError("total_steps must be provided")

        total_steps = _positive_int(total_steps, "total_steps")
        grad_accum = _positive_int(
            gradient_accumulation_steps
            if gradient_accumulation_steps is not None
            else self.config.training.gradient_accumulation_steps
            if self.config is not None
            else 1,
            "training.gradient_accumulation_steps",
        )
        max_grad_norm = float(self.config.training.max_grad_norm if self.config is not None else 0.0)
        ignore_index = int(getattr(self.config, "ignore_index", -100)) if self.config is not None else -100
        state = state or TrainerState()
        exact_epoch_mode = total_micro_batches is not None
        target_micro_batches = (
            _positive_int(total_micro_batches, "total_micro_batches")
            if total_micro_batches is not None
            else state.consumed_batches + (total_steps - state.global_step) * grad_accum
        )
        batches_per_epoch = len(train_dataloader) if exact_epoch_mode else 0  # type: ignore[arg-type]
        if exact_epoch_mode:
            if batches_per_epoch <= 0:
                raise ValueError("train dataloader must contain at least one micro-batch")
            if batches_per_epoch % grad_accum != 0:
                raise RuntimeError(
                    "exact epoch training requires len(train_dataloader) to be divisible by "
                    "training.gradient_accumulation_steps: "
                    f"len={batches_per_epoch} gradient_accumulation_steps={grad_accum}"
                )
            if target_micro_batches % grad_accum != 0:
                raise RuntimeError(
                    "target_micro_batches must be divisible by training.gradient_accumulation_steps: "
                    f"target_micro_batches={target_micro_batches} gradient_accumulation_steps={grad_accum}"
                )
            if state.consumed_batches % grad_accum != 0:
                raise RuntimeError(
                    "resume state is not on an optimizer-step boundary: "
                    f"consumed_batches={state.consumed_batches} gradient_accumulation_steps={grad_accum}"
                )
            _set_epoch(train_dataloader, state.consumed_batches // batches_per_epoch)
        train_iterator = iter(train_dataloader)
        train_iterator = _advance_iterator(train_iterator, train_dataloader, state.consumed_batches)

        accumulated_loss = 0.0
        step_samples = 0
        step_tokens = 0
        step_supervised_tokens = 0
        accumulated_batches = 0
        accumulated_route_metrics: dict[str, float] = {}
        accumulated_route_metric_count = 0
        accumulation_target = grad_accum
        step_started_at = time.monotonic()
        if hasattr(model, "train"):
            model.train()

        optimizer.zero_grad(set_to_none=True)

        while state.global_step < total_steps and state.consumed_batches < target_micro_batches:
            with self.accelerator.accumulate(model):
                batch, train_iterator = _cycle_next(train_iterator, train_dataloader)
                loss = self.compute_loss(model, batch)
                if self.last_loss_metrics:
                    for key, value in self.last_loss_metrics.items():
                        accumulated_route_metrics[key] = accumulated_route_metrics.get(key, 0.0) + float(value)
                    accumulated_route_metric_count += 1
                accumulated_loss += _detach_item(loss)
                batch_stats = _batch_stats(batch, ignore_index=ignore_index)
                step_samples += batch_stats["samples"]
                step_tokens += batch_stats["tokens"]
                step_supervised_tokens += batch_stats["supervised_tokens"]
                self.accelerator.backward(loss)
                state.consumed_batches += 1
                accumulated_batches += 1

            epoch_boundary = exact_epoch_mode and state.consumed_batches % batches_per_epoch == 0
            if epoch_boundary:
                _set_epoch(train_dataloader, state.consumed_batches // batches_per_epoch)
            should_step = accumulated_batches >= accumulation_target
            if not should_step:
                continue
            if hasattr(self.accelerator, "sync_gradients") and not self.accelerator.sync_gradients:
                raise RuntimeError("Accelerate did not synchronize gradients at an optimizer-step boundary")

            if max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)

            state.global_step += 1
            elapsed = max(time.monotonic() - step_started_at, 1e-9)
            step_totals = _gather_step_totals(
                self.accelerator,
                accumulated_loss=accumulated_loss,
                samples=step_samples,
                tokens=step_tokens,
                supervised_tokens=step_supervised_tokens,
                elapsed=elapsed,
            )
            metrics = {
                "train/loss": step_totals["accumulated_loss"] / (accumulated_batches * step_totals["processes"]),
                "train/samples_per_second": step_totals["samples"] / step_totals["elapsed"],
                "train/tokens_per_second": step_totals["tokens"] / step_totals["elapsed"],
                "train/supervised_tokens_per_second": step_totals["supervised_tokens"] / step_totals["elapsed"],
            }
            lr = _current_lr(optimizer)
            if lr is not None:
                metrics["train/lr"] = lr
            if accumulated_route_metric_count:
                for key, value in accumulated_route_metrics.items():
                    metrics[f"train/{key}"] = value / accumulated_route_metric_count
            self.hooks.metrics(metrics, state)

            if self.cadence.should_validate(state.global_step):
                self.run_validation_boundary(model, optimizer, valid_dataloader, state)

            accumulated_loss = 0.0
            step_samples = 0
            step_tokens = 0
            step_supervised_tokens = 0
            accumulated_batches = 0
            accumulated_route_metrics = {}
            accumulated_route_metric_count = 0
            step_started_at = time.monotonic()
            if hasattr(model, "train"):
                model.train()

        return state

    def run_validation_boundary(
        self,
        model: Any,
        optimizer: Any,
        valid_dataloader: Iterable[dict[str, Any]] | None,
        state: TrainerState,
    ) -> dict[str, Any]:
        state.validation_index += 1
        metrics: dict[str, Any] = {}

        self.hooks.phase("validation:standard:start", state)
        if hasattr(model, "eval"):
            model.eval()
        metrics.update(self.hooks.standard_eval(model, valid_dataloader, state))

        if self.cadence is not None and self.cadence.should_run_bfcl(state.validation_index):
            self.hooks.phase("validation:bfcl:start", state)
            metrics.update(self.hooks.bfcl_eval(model, valid_dataloader, state))

        if self.cadence is not None and self.cadence.should_checkpoint(state.validation_index):
            state.checkpoint_index += 1
            self.hooks.phase("checkpoint:save:start", state)
            checkpoint_path = self.hooks.checkpoint(model, optimizer, state, metrics)
            if checkpoint_path is not None:
                metrics["checkpoint/path"] = checkpoint_path

        if metrics:
            self.hooks.metrics(metrics, state)
        self.hooks.phase("validation:end", state)
        return metrics


def _batch_stats(batch: dict[str, Any], *, ignore_index: int) -> dict[str, int]:
    sample_id = batch.get("sample_id")
    samples = len(sample_id) if isinstance(sample_id, list) else int(batch["input_ids"].shape[0]) if "input_ids" in batch else 0
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        tokens = int(attention_mask.sum().item())
    elif "chosen_attention_mask" in batch and "rejected_attention_mask" in batch:
        tokens = int(batch["chosen_attention_mask"].sum().item() + batch["rejected_attention_mask"].sum().item())
    else:
        tokens = 0
    labels = batch.get("labels")
    if labels is not None:
        supervised_tokens = int((labels != ignore_index).sum().item())
    elif "chosen_labels" in batch and "rejected_labels" in batch:
        supervised_tokens = int(
            (batch["chosen_labels"] != ignore_index).sum().item()
            + (batch["rejected_labels"] != ignore_index).sum().item()
        )
    else:
        supervised_tokens = 0
    return {"samples": samples, "tokens": tokens, "supervised_tokens": supervised_tokens}


def _current_lr(optimizer: Any) -> float | None:
    param_groups = getattr(optimizer, "param_groups", None)
    if not param_groups:
        return None
    return float(param_groups[0].get("lr", 0.0))


def _gather_step_totals(
    accelerator: Any,
    *,
    accumulated_loss: float,
    samples: int,
    tokens: int,
    supervised_tokens: int,
    elapsed: float,
) -> dict[str, float]:
    """Aggregate training metrics across ranks without changing gradient semantics."""

    if not hasattr(accelerator, "gather_for_metrics"):
        return {
            "accumulated_loss": float(accumulated_loss),
            "samples": float(samples),
            "tokens": float(tokens),
            "supervised_tokens": float(supervised_tokens),
            "elapsed": float(elapsed),
            "processes": 1.0,
        }

    values = torch.tensor(
        [float(accumulated_loss), float(samples), float(tokens), float(supervised_tokens), float(elapsed)],
        device=accelerator.device,
    )
    gathered = accelerator.gather_for_metrics(values).reshape(-1, 5)
    return {
        "accumulated_loss": float(gathered[:, 0].sum().item()),
        "samples": float(gathered[:, 1].sum().item()),
        "tokens": float(gathered[:, 2].sum().item()),
        "supervised_tokens": float(gathered[:, 3].sum().item()),
        "elapsed": max(float(gathered[:, 4].max().item()), 1e-9),
        "processes": float(gathered.shape[0]),
    }


def _advance_iterator(iterator: Any, iterable: Iterable[Any], consumed_batches: int) -> Any:
    if consumed_batches <= 0:
        return iterator
    to_skip = int(consumed_batches)
    try:
        length = len(iterable)  # type: ignore[arg-type]
    except TypeError:
        length = 0
    if length:
        to_skip %= int(length)
    for _ in range(to_skip):
        _batch, iterator = _cycle_next(iterator, iterable)
    return iterator


def _set_epoch(iterable: Iterable[Any], epoch: int) -> None:
    """Propagate the epoch to the routed sampler for per-epoch reshuffle.

    After ``accelerator.prepare`` the object is an Accelerate ``DataLoaderShard``
    whose ``.batch_sampler`` is Accelerate's ``BatchSamplerShard`` wrapper, which
    does not expose ``set_epoch``. The shard itself does, and it forwards the
    epoch to the underlying ``RoutedBatchSampler``. Try every layer so the epoch
    reaches the real sampler in both the prepared and unprepared cases.
    """

    applied = False
    for target in (iterable, getattr(iterable, "batch_sampler", None), getattr(iterable, "sampler", None)):
        if hasattr(target, "set_epoch"):
            target.set_epoch(epoch)
            applied = True
    if not applied:
        # Reach through Accelerate's wrapper to the wrapped routed sampler.
        wrapped = getattr(getattr(iterable, "batch_sampler", None), "batch_sampler", None)
        if hasattr(wrapped, "set_epoch"):
            wrapped.set_epoch(epoch)

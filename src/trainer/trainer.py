from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

from trainer.callbacks import TrainerHooks
from trainer.state import TrainerState


@dataclass(frozen=True)
class TrainerCadence:
    """Validation/checkpoint cadence expressed in training-loop units."""

    eval_every_train_steps: int
    checkpoint_every_n_validations: int
    bfcl_every_n_validations: int

    @classmethod
    def from_config(cls, config: Any) -> "TrainerCadence":
        eval_config = config.section("eval") if "eval" in config.raw else {}
        checkpointing = config.section("checkpointing")

        eval_every = eval_config.get("every_train_steps")
        if eval_every is None:
            raise ValueError("eval.every_train_steps must be configured")

        checkpoint_every = checkpointing.get("save_every_n_validations")
        if checkpoint_every is None:
            raise ValueError("checkpointing.save_every_n_validations must be configured")

        bfcl = eval_config.get("bfcl") if isinstance(eval_config.get("bfcl"), dict) else {}
        bfcl_every = bfcl.get("run_every_n_validations", 1)

        return cls(
            eval_every_train_steps=_positive_int(eval_every, "eval.every_train_steps"),
            checkpoint_every_n_validations=_positive_int(
                checkpoint_every,
                "checkpointing.save_every_n_validations",
            ),
            bfcl_every_n_validations=_positive_int(bfcl_every, "eval.bfcl.run_every_n_validations"),
        )

    def should_validate(self, global_step: int) -> bool:
        return global_step > 0 and global_step % self.eval_every_train_steps == 0

    def should_checkpoint(self, validation_index: int) -> bool:
        return validation_index > 0 and validation_index % self.checkpoint_every_n_validations == 0

    def should_run_bfcl(self, validation_index: int) -> bool:
        return validation_index > 0 and validation_index % self.bfcl_every_n_validations == 0


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

    def compute_loss(self, model, batch):
        loss_kind = batch.get("loss_kind")
        if loss_kind in {"sft_target", "sft_tool"}:
            from losses.sft import sft_cross_entropy_loss

            return sft_cross_entropy_loss(model, batch)
        if loss_kind == "dpo_target":
            raise NotImplementedError("DPO route requires chosen/rejected logprob plumbing")
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
        max_steps: int | None = None,
        gradient_accumulation_steps: int | None = None,
    ) -> TrainerState:
        """Run the routed training loop.

        `global_step` is an optimizer step, not a micro-batch. Validation,
        checkpointing, and registry hooks are expected to hang off this
        boundary, which keeps the training flow linear and resumable.
        """

        if self.cadence is None:
            raise ValueError("trainer cadence must be configured")
        if self.config is None and max_steps is None:
            raise ValueError("max_steps must be provided when trainer has no config")

        training = self.config.section("training") if self.config is not None else {}
        max_steps = _positive_int(max_steps if max_steps is not None else training["max_steps"], "training.max_steps")
        grad_accum = _positive_int(
            gradient_accumulation_steps
            if gradient_accumulation_steps is not None
            else training.get("gradient_accumulation_steps", 1),
            "training.gradient_accumulation_steps",
        )
        max_grad_norm = float(training.get("max_grad_norm", 0.0) or 0.0)
        ignore_index = int(getattr(self.config, "ignore_index", -100)) if self.config is not None else -100
        eval_enabled = bool(self.config.section("eval").get("enabled", True)) if self.config is not None else True

        state = state or TrainerState()
        train_iterator = iter(train_dataloader)
        train_iterator = _advance_iterator(train_iterator, train_dataloader, state.consumed_batches)

        while state.global_step < max_steps:
            if hasattr(model, "train"):
                model.train()
            optimizer.zero_grad(set_to_none=True)

            accumulated_loss = 0.0
            step_samples = 0
            step_tokens = 0
            step_supervised_tokens = 0
            step_started_at = time.monotonic()
            for _ in range(grad_accum):
                with self.accelerator.accumulate(model):
                    batch, train_iterator = _cycle_next(train_iterator, train_dataloader)
                    loss = self.compute_loss(model, batch)
                    accumulated_loss += _detach_item(loss)
                    batch_stats = _batch_stats(batch, ignore_index=ignore_index)
                    step_samples += batch_stats["samples"]
                    step_tokens += batch_stats["tokens"]
                    step_supervised_tokens += batch_stats["supervised_tokens"]
                    self.accelerator.backward(loss)
                    state.consumed_batches += 1

            if max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

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
                "train/loss": step_totals["accumulated_loss"] / (grad_accum * step_totals["processes"]),
                "train/samples_per_second": step_totals["samples"] / step_totals["elapsed"],
                "train/tokens_per_second": step_totals["tokens"] / step_totals["elapsed"],
                "train/supervised_tokens_per_second": step_totals["supervised_tokens"] / step_totals["elapsed"],
            }
            lr = _current_lr(optimizer)
            if lr is not None:
                metrics["train/lr"] = lr
            self.hooks.metrics(metrics, state)

            if eval_enabled and self.cadence.should_validate(state.global_step):
                self.run_validation_boundary(model, optimizer, valid_dataloader, state)

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
    tokens = int(attention_mask.sum().item()) if attention_mask is not None else 0
    labels = batch.get("labels")
    supervised_tokens = int((labels != ignore_index).sum().item()) if labels is not None else 0
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

    import torch

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

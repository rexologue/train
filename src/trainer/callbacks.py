from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from trainer.state import TrainerState


PhaseHook = Callable[[str, TrainerState], None]
EvalHook = Callable[[Any, Any, TrainerState], dict[str, float]]
CheckpointHook = Callable[[Any, Any, TrainerState, dict[str, float]], str | None]
MetricsHook = Callable[[dict[str, Any], TrainerState], None]


@dataclass
class TrainerHooks:
    """Optional side-effect hooks used by the routed training loop."""

    on_phase: PhaseHook | None = None
    run_standard_eval: EvalHook | None = None
    run_bfcl_eval: EvalHook | None = None
    save_checkpoint: CheckpointHook | None = None
    log_metrics: MetricsHook | None = None

    def phase(self, name: str, state: TrainerState) -> None:
        if self.on_phase is not None:
            self.on_phase(name, state)

    def standard_eval(self, model: Any, dataloader: Any, state: TrainerState) -> dict[str, float]:
        if self.run_standard_eval is None:
            return {}
        return self.run_standard_eval(model, dataloader, state)

    def bfcl_eval(self, model: Any, dataloader: Any, state: TrainerState) -> dict[str, float]:
        if self.run_bfcl_eval is None:
            return {}
        return self.run_bfcl_eval(model, dataloader, state)

    def checkpoint(self, model: Any, optimizer: Any, state: TrainerState, metrics: dict[str, float]) -> str | None:
        if self.save_checkpoint is None:
            return None
        return self.save_checkpoint(model, optimizer, state, metrics)

    def metrics(self, metrics: dict[str, Any], state: TrainerState) -> None:
        if self.log_metrics is not None:
            self.log_metrics(metrics, state)

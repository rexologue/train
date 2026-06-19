from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from registry.tags import candidate_alias


def better_metric(current: float | None, candidate: float, greater_is_better: bool) -> bool:
    if current is None:
        return True
    return candidate > current if greater_is_better else candidate < current


@dataclass(frozen=True)
class CheckpointCandidate:
    path: Path
    checkpoint_index: int
    global_step: int
    metric_value: float
    metrics: dict[str, Any]


@dataclass(frozen=True)
class RegistrationDecision:
    checkpoint: CheckpointCandidate
    candidate_index: int
    aliases: list[str]


class CandidateWindowSelector:
    """Select the best checkpoint in each registry window."""

    def __init__(
        self,
        *,
        register_every_n_checkpoints: int,
        selection_metric: str,
        greater_is_better: bool,
        candidate_alias_template: str,
        rolling_candidate_alias: str | None = None,
        next_candidate_index: int = 1,
    ):
        if register_every_n_checkpoints <= 0:
            raise ValueError("registry.register_every_n_checkpoints must be positive")
        self.register_every_n_checkpoints = int(register_every_n_checkpoints)
        self.selection_metric = selection_metric
        self.greater_is_better = bool(greater_is_better)
        self.candidate_alias_template = candidate_alias_template
        self.rolling_candidate_alias = rolling_candidate_alias
        self.next_candidate_index = int(next_candidate_index)
        self._window: list[CheckpointCandidate] = []

    @classmethod
    def from_config(cls, config: Any, *, next_candidate_index: int = 1) -> "CandidateWindowSelector":
        return cls(
            register_every_n_checkpoints=config.registry.register_every_n_checkpoints,
            selection_metric=config.registry.selection.metric,
            greater_is_better=config.registry.selection.mode == "max",
            candidate_alias_template="candidate-{candidate_index:06d}",
            rolling_candidate_alias="candidate-latest",
            next_candidate_index=next_candidate_index,
        )

    def observe_checkpoint(
        self,
        *,
        checkpoint_path: str | Path,
        checkpoint_index: int,
        global_step: int,
        metrics: dict[str, Any],
    ) -> RegistrationDecision | None:
        if self.selection_metric not in metrics:
            raise KeyError(f"missing registry selection metric: {self.selection_metric}")
        candidate = CheckpointCandidate(
            path=Path(checkpoint_path),
            checkpoint_index=int(checkpoint_index),
            global_step=int(global_step),
            metric_value=float(metrics[self.selection_metric]),
            metrics=dict(metrics),
        )
        self._window.append(candidate)
        if len(self._window) < self.register_every_n_checkpoints:
            return None

        best = self._best_in_window()
        aliases = [candidate_alias(self.candidate_alias_template, self.next_candidate_index)]
        if self.rolling_candidate_alias:
            aliases.append(str(self.rolling_candidate_alias))
        decision = RegistrationDecision(
            checkpoint=best,
            candidate_index=self.next_candidate_index,
            aliases=aliases,
        )
        self.next_candidate_index += 1
        self._window.clear()
        return decision

    def _best_in_window(self) -> CheckpointCandidate:
        best: CheckpointCandidate | None = None
        for candidate in self._window:
            if best is None or better_metric(best.metric_value, candidate.metric_value, self.greater_is_better):
                best = candidate
        if best is None:
            raise RuntimeError("cannot select from an empty registry window")
        return best

    def window_checkpoint_paths(self) -> set[Path]:
        return {candidate.path for candidate in self._window}

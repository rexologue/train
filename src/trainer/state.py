from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TrainerState:
    """Serializable counters that define strict resume position."""

    global_step: int = 0
    validation_index: int = 0
    checkpoint_index: int = 0
    consumed_batches: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainerState":
        return cls(
            global_step=int(data.get("global_step", 0)),
            validation_index=int(data.get("validation_index", 0)),
            checkpoint_index=int(data.get("checkpoint_index", 0)),
            consumed_batches=int(data.get("consumed_batches", 0)),
        )

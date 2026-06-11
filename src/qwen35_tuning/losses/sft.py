from __future__ import annotations

from typing import Any


def sft_cross_entropy_loss(model: Any, batch: dict[str, Any]) -> Any:
    return model(**batch).loss


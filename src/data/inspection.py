from __future__ import annotations

import random
from typing import Any

from data.dataloaders import DataLoaderBundle


def _shape_summary(value: Any) -> dict[str, Any]:
    """Describe a batch field without materializing full token arrays."""

    if hasattr(value, "shape"):
        return {"type": value.__class__.__name__, "shape": list(value.shape), "dtype": str(getattr(value, "dtype", ""))}
    if isinstance(value, list):
        return {"type": "list", "length": len(value)}
    return {"type": type(value).__name__}


def _batch_size(batch: dict[str, Any]) -> int:
    """Infer batch size from the first tensor-like or list batch field."""

    for value in batch.values():
        if hasattr(value, "shape") and len(value.shape) > 0:
            return int(value.shape[0])
        if isinstance(value, list):
            return len(value)
    raise ValueError("cannot infer batch size from empty or scalar-only batch")


def _preview_tensor(value: Any, *, token_limit: int) -> dict[str, Any]:
    """Return a bounded preview for one tensor-valued example field."""

    shape = list(value.shape)
    flat = value.detach().cpu().reshape(-1)
    preview_len = min(int(token_limit), int(flat.numel()))
    return {
        "type": value.__class__.__name__,
        "shape": shape,
        "dtype": str(value.dtype),
        "values": flat[:preview_len].tolist(),
        "num_values": int(flat.numel()),
        "truncated": int(flat.numel()) > preview_len,
    }


def _preview_value(value: Any, *, element_index: int, batch_size: int, token_limit: int) -> Any:
    """Preview the selected example from one batch field."""

    if hasattr(value, "shape"):
        if len(value.shape) == 0:
            return value.detach().cpu().item()
        return _preview_tensor(value[element_index], token_limit=token_limit)
        
    if isinstance(value, list) and len(value) == batch_size:
        return value[element_index]

    return value


def inspect_random_batch(
    bundle: DataLoaderBundle,
    *,
    split: str,
    seed: int,
    token_limit: int,
) -> dict[str, Any]:
    """Return keys, shapes, and one bounded example preview from a random batch.

    The function consumes at most one DataLoader pass up to the chosen batch.
    It intentionally reports token arrays with a preview limit so inspection is
    useful for long-context samples without dumping tens of thousands of ids.
    """

    if token_limit <= 0:
        raise ValueError("token_limit must be positive")
    if split not in bundle.splits:
        raise ValueError(f"split {split!r} was not built; available={sorted(bundle.splits)}")

    split_loader = bundle.splits[split]
    if len(split_loader.sampler) == 0:
        raise ValueError(f"split {split!r} has no batches to inspect")

    rng = random.Random(seed)
    batch_index = rng.randrange(len(split_loader.sampler))
    batch = None
    for current_index, candidate in enumerate(split_loader.dataloader):
        if current_index == batch_index:
            batch = candidate
            break
    if batch is None:
        raise RuntimeError(f"failed to read batch index {batch_index} from split {split!r}")

    size = _batch_size(batch)
    element_index = rng.randrange(size)
    return {
        "split": split,
        "batch_index": batch_index,
        "element_index": element_index,
        "batch_size": size,
        "loss_kind": batch.get("loss_kind"),
        "keys": list(batch.keys()),
        "shapes": {key: _shape_summary(value) for key, value in batch.items()},
        "element": {
            key: _preview_value(value, element_index=element_index, batch_size=size, token_limit=token_limit)
            for key, value in batch.items()
        },
    }

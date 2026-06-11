from __future__ import annotations

from pathlib import Path

from .atomic import atomic_checkpoint_dir


def prepare_atomic_save(path: str | Path) -> Path:
    tmp, _ = atomic_checkpoint_dir(path)
    tmp.mkdir(parents=True, exist_ok=False)
    return tmp


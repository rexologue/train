from __future__ import annotations

from pathlib import Path


def atomic_checkpoint_dir(final_dir: str | Path) -> tuple[Path, Path]:
    final_path = Path(final_dir)
    return final_path.with_name(final_path.name + ".tmp"), final_path


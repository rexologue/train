from __future__ import annotations

from pathlib import Path
import shutil


def atomic_checkpoint_dir(final_dir: str | Path) -> tuple[Path, Path]:
    final_path = Path(final_dir)
    return final_path.with_name(final_path.name + ".tmp"), final_path


def reset_tmp_checkpoint_dir(tmp_dir: str | Path) -> Path:
    tmp_path = Path(tmp_dir)
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=False)
    return tmp_path


def commit_atomic_checkpoint_dir(tmp_dir: str | Path, final_dir: str | Path) -> Path:
    tmp_path = Path(tmp_dir)
    final_path = Path(final_dir)
    if final_path.exists():
        raise FileExistsError(f"checkpoint already exists: {final_path}")
    tmp_path.replace(final_path)
    return final_path

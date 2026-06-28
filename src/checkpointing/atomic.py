from __future__ import annotations

from pathlib import Path
import shutil


READY_MARKER = "READY"


def atomic_checkpoint_dir(final_dir: str | Path) -> tuple[Path, Path]:
    final_path = Path(final_dir)
    return final_path.with_name(final_path.name + ".tmp"), final_path


def reset_tmp_checkpoint_dir(tmp_dir: str | Path) -> Path:
    tmp_path = Path(tmp_dir)
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=False)
    return tmp_path


def checkpoint_is_ready(checkpoint_dir: str | Path) -> bool:
    """A checkpoint is only complete once its READY marker exists."""

    return (Path(checkpoint_dir) / READY_MARKER).is_file()


def commit_atomic_checkpoint_dir(tmp_dir: str | Path, final_dir: str | Path) -> Path:
    tmp_path = Path(tmp_dir)
    final_path = Path(final_dir)

    # The READY marker is written last, inside the tmp dir, so that after the
    # atomic rename a reader either sees no final dir at all or sees a fully
    # written, READY-marked one. There is no in-between visible state.
    (tmp_path / READY_MARKER).write_text("ok", encoding="utf-8")

    if final_path.exists():
        # A fully committed checkpoint with the same name is a real duplicate
        # and must not be silently overwritten. A leftover from a crashed save
        # (no READY) is corrupt and is replaced.
        if checkpoint_is_ready(final_path):
            raise FileExistsError(f"checkpoint already exists and is READY: {final_path}")
        shutil.rmtree(final_path)

    tmp_path.replace(final_path)
    return final_path

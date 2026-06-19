from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import yaml

from config import Config
from utils.hashing import file_sha256


def collect_tracking_lineage(config: Config) -> dict[str, Any]:
    """Auto-discover data.dvc next to the configured train dataset."""

    train_path = config.preprocessing.raw.train_path.expanduser().resolve()
    candidates = (train_path.parent / "data.dvc", train_path.parent.parent / "data.dvc")
    dvc_path = next((path for path in candidates if path.is_file()), None)
    if dvc_path is None:
        return {
            "dvc": {
                "enabled": False,
                "train_path": str(train_path),
                "searched": [str(path) for path in candidates],
            }
        }
    repo_root = dvc_path.parent

    return {
        "dvc": {
            "enabled": True,
            "repo_root": str(repo_root),
            "git": collect_git_metadata(repo_root),
            "targets": {"data": collect_dvc_target(repo_root, dvc_path.name)},
        }
    }


def collect_git_metadata(repo_root: Path) -> dict[str, Any]:
    """Collect reproducibility metadata from a git repository."""

    commit = run_git(repo_root, "rev-parse", "HEAD")
    branch = run_git(repo_root, "branch", "--show-current")
    status = run_git(repo_root, "status", "--short") or ""
    remote_url = run_git(repo_root, "remote", "get-url", "origin")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status.strip()),
        "status_short": [line for line in status.splitlines() if line.strip()],
        "remote_origin": remote_url,
    }


def collect_code_metadata(repo_root: Path) -> dict[str, Any]:
    """Collect git metadata for the training code repository."""

    return collect_git_metadata(repo_root)


def run_git(repo_root: Path, *args: str) -> str | None:
    """Run a git command and return stripped stdout, or None outside git repos."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def collect_dvc_target(repo_root: Path, dvc_file: str) -> dict[str, Any]:
    """Read one .dvc metadata file without requiring the DVC Python package."""

    dvc_path = (repo_root / dvc_file).resolve()
    with dvc_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"DVC metadata must be a mapping: {dvc_path}")
    outs = raw.get("outs")
    if not isinstance(outs, list):
        raise ValueError(f"DVC metadata has no outs list: {dvc_path}")
    return {
        "dvc_file": str(dvc_path),
        "dvc_file_sha256": file_sha256(dvc_path),
        "outs": [normalize_dvc_out(item) for item in outs],
    }


def normalize_dvc_out(value: Any) -> dict[str, Any]:
    """Normalize one DVC out entry into a stable JSON object."""

    if not isinstance(value, dict):
        raise ValueError("DVC outs entries must be mappings")
    return {
        "path": value.get("path"),
        "hash": value.get("hash"),
        "md5": value.get("md5"),
        "size": value.get("size"),
        "nfiles": value.get("nfiles"),
    }

from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen35_tuning.config.hashing import file_sha256


def collect_dvc_lineage(repo: str, targets: list[str]) -> dict[str, Any]:
    repo_path = Path(repo)
    target_hashes = {}
    for target in targets:
        path = repo_path / target
        target_hashes[target] = file_sha256(path) if path.exists() and path.is_file() else None
    return {"repo": str(repo_path), "targets": targets, "target_hashes": target_hashes}


from __future__ import annotations

from pathlib import Path

from config.hashing import file_sha256


def directory_checksums(root: str | Path) -> dict[str, str]:
    root_path = Path(root)
    return {
        str(path.relative_to(root_path)): file_sha256(path)
        for path in sorted(root_path.rglob("*"))
        if path.is_file()
    }


from __future__ import annotations

from pathlib import Path

from utils.hashing import file_sha256


# Files that are written after the checksum manifest itself and therefore must
# be excluded both when computing and when verifying it.
CHECKSUM_EXCLUDED_NAMES = {"checksums.json", "READY"}


def directory_checksums(root: str | Path) -> dict[str, str]:
    root_path = Path(root)
    return {
        str(path.relative_to(root_path)): file_sha256(path)
        for path in sorted(root_path.rglob("*"))
        if path.is_file() and path.name not in CHECKSUM_EXCLUDED_NAMES
    }


def verify_directory_checksums(root: str | Path, expected: dict[str, str]) -> list[str]:
    """Return a list of human-readable integrity problems; empty means intact.

    A checkpoint is only safe to resume from when every file recorded at save
    time is present and byte-identical. Missing files, extra files, and content
    drift are all reported so resume can fail loudly instead of training from a
    corrupt optimizer/adapter shard.
    """

    actual = directory_checksums(root)
    problems: list[str] = []
    for name, digest in expected.items():
        if name not in actual:
            problems.append(f"missing file: {name}")
        elif actual[name] != digest:
            problems.append(f"checksum mismatch: {name}")
    for name in actual:
        if name not in expected:
            problems.append(f"unexpected file: {name}")
    return problems

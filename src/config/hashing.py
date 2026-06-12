from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json_dumps(value: Any) -> str:
    """Serialize values deterministically for hashes and manifests."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    """Return a prefixed SHA256 for UTF-8 text."""

    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return a prefixed SHA256 for bytes."""

    return "sha256:" + hashlib.sha256(data).hexdigest()


def stable_hash(value: Any) -> str:
    """Return a stable SHA256 for JSON-serializable values."""

    return sha256_text(stable_json_dumps(value))


def file_sha256(path: str | Path) -> str:
    """Stream a file and return its prefixed SHA256."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()

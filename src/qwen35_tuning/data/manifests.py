from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen35_tuning.config.hashing import stable_json_dumps


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")


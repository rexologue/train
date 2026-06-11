from __future__ import annotations

from typing import Any

from qwen35_tuning.config.hashing import stable_hash, stable_json_dumps


def canonical_tools_json(tools: list[dict[str, Any]] | None) -> str | None:
    if tools is None:
        return None
    return stable_json_dumps(tools)


def tools_hash(tools: list[dict[str, Any]] | None) -> str | None:
    if tools is None:
        return None
    return stable_hash(tools)


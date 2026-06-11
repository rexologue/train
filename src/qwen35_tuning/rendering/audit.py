from __future__ import annotations

FORBIDDEN_RAW_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<think>",
    "</think>",
)


class RenderingAuditError(ValueError):
    pass


def reject_forbidden_raw_markers(messages: list[dict], enabled: bool = True) -> None:
    if not enabled:
        return
    for index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for marker in FORBIDDEN_RAW_MARKERS:
            if marker in content:
                raise RenderingAuditError(f"raw message {index} contains forbidden marker {marker!r}")


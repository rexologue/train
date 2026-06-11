from __future__ import annotations

import re
from dataclasses import dataclass


THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
THINK_MARKER_RE = re.compile(r"</?think>", re.IGNORECASE)


class ReasoningAuditError(ValueError):
    pass


@dataclass(frozen=True)
class ReasoningAudit:
    has_think_markers: bool
    nonempty_think_blocks: int
    supervised_think_tokens: int


def audit_reasoning(
    rendered_text: str,
    supervised_char_ranges: list[tuple[int, int]],
    config: dict,
) -> ReasoningAudit:
    blocks = THINK_BLOCK_RE.finditer(rendered_text)
    nonempty = 0
    supervised_think_tokens = 0
    for match in blocks:
        content_start, content_end = match.span(1)
        if rendered_text[content_start:content_end].strip():
            nonempty += 1
        for start, end in supervised_char_ranges:
            if start < match.end() and end > match.start():
                supervised_think_tokens += 1

    has_markers = THINK_MARKER_RE.search(rendered_text) is not None
    if nonempty and config.get("fail_on_nonempty_think_blocks", True):
        raise ReasoningAuditError("non-empty <think> block found while reasoning is disabled")
    if supervised_think_tokens and config.get("fail_if_supervised_think_tokens", True):
        raise ReasoningAuditError("supervised span intersects <think> block")
    if has_markers and not config.get("allow_empty_template_think_markers", False):
        raise ReasoningAuditError("think markers are present and allow_empty_template_think_markers=false")

    return ReasoningAudit(has_markers, nonempty, supervised_think_tokens)


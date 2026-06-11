from __future__ import annotations

from qwen35_tuning.data.schemas import AssistantSpan


def spans_to_ranges(spans: list[AssistantSpan]) -> list[tuple[int, int]]:
    return [(span.start, span.end) for span in spans]


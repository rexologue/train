from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from qwen35_tuning.data.schemas import AssistantSpan, CanonicalRow, TargetSpan


@dataclass(frozen=True)
class TargetSelectionSummary:
    num_target_candidates: int
    num_long_targets_kept: int
    num_short_targets_total: int
    num_short_targets_kept: int
    num_targets_dropped_by_policy: int


def stable_uniform_0_1(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / float(2**64)


def select_sft_target_spans(
    row: CanonicalRow,
    assistant_spans: list[AssistantSpan],
    policy: dict[str, Any],
) -> tuple[list[TargetSpan], TargetSelectionSummary]:
    span_by_index = {span.message_index: span for span in assistant_spans}
    selected: list[TargetSpan] = []
    candidates = 0
    long_kept = 0
    short_total = 0
    short_kept = 0

    min_chars = int(policy["min_guaranteed_assistant_chars"])
    keep_probability = float(policy.get("loss_on_short_assistant_reply_prob", 0.3))
    seed = int(policy["short_response_sampling_seed"])

    for index, message in enumerate(row.messages):
        if message.get("role") != "assistant" or index not in span_by_index:
            continue

        candidates += 1
        span = span_by_index[index]
        chars = max(0, span.end - span.start)
        if chars > min_chars:
            selected.append(TargetSpan(index, span.start, span.end, "long_response"))
            long_kept += 1
            continue

        short_total += 1
        key = f"{row.sample_id}:{index}:{seed}"
        if stable_uniform_0_1(key) < keep_probability:
            selected.append(TargetSpan(index, span.start, span.end, "short_sampled"))
            short_kept += 1

    summary = TargetSelectionSummary(
        num_target_candidates=candidates,
        num_long_targets_kept=long_kept,
        num_short_targets_total=short_total,
        num_short_targets_kept=short_kept,
        num_targets_dropped_by_policy=candidates - len(selected),
    )
    return selected, summary


def select_sft_tool_spans(
    row: CanonicalRow,
    assistant_spans: list[AssistantSpan],
    policy: dict[str, Any],
) -> tuple[list[TargetSpan], TargetSelectionSummary]:
    selected: list[TargetSpan] = [
        TargetSpan(span.message_index, span.start, span.end, "all_assistant")
        for span in assistant_spans
    ]
    summary = TargetSelectionSummary(
        num_target_candidates=len(selected),
        num_long_targets_kept=len(selected),
        num_short_targets_total=0,
        num_short_targets_kept=0,
        num_targets_dropped_by_policy=0,
    )
    return selected, summary

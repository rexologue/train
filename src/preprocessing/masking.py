from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

from config import stable_hash


LossKind = Literal["sft_target", "sft_tool", "dpo_target"]
Split = Literal["train", "valid", "test"]


class CanonicalizationError(ValueError):
    """Raised when a parsed raw sample is not valid for its authoritative loss kind."""


class MaskingError(ValueError):
    """Raised when token labels cannot be built safely."""


@dataclass(frozen=True)
class CanonicalRow:
    """Canonical in-memory sample after parsing the parquet wrapper."""

    sample_id: str
    split: Split
    loss_kind: LossKind
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    parallel_tool_calls: bool | None = None
    prompt: list[dict[str, Any]] | None = None
    chosen: dict[str, Any] | None = None
    rejected: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssistantSpan:
    """Character span for an assistant completion body in rendered text."""

    message_index: int
    start: int
    end: int
    kind: str


@dataclass(frozen=True)
class TargetSpan:
    """Character span selected for supervised loss."""

    message_index: int
    start: int
    end: int
    reason: str


@dataclass(frozen=True)
class RenderedSample:
    """Rendered chat text plus assistant spans and renderer metadata."""

    rendered_text: str
    assistant_spans: list[AssistantSpan]
    render_metadata: dict[str, Any]


@dataclass(frozen=True)
class TargetSelectionSummary:
    """Counters emitted by deterministic target selection."""

    num_target_candidates: int
    num_long_targets_kept: int
    num_short_targets_total: int
    num_short_targets_kept: int
    num_targets_dropped_by_policy: int


def spans_to_ranges(spans: list[AssistantSpan]) -> list[tuple[int, int]]:
    """Convert assistant spans to plain `(start, end)` ranges."""

    return [(span.start, span.end) for span in spans]


def canonicalize_row(row: dict[str, Any], split: Split, row_index: int) -> CanonicalRow:
    """Convert a parsed raw row to the canonical shape used by legacy tests/helpers.

    `loss_kind` must already be injected from the parquet `type`/`target` column.
    The function intentionally does not infer it from payload contents because
    `sft_tool` may contain ordinary assistant text and no explicit tools.
    """

    if not isinstance(row, dict):
        raise CanonicalizationError("raw row must be a mapping")

    loss_kind = row.get("loss_kind")
    if loss_kind not in {"sft_target", "sft_tool", "dpo_target"}:
        raise CanonicalizationError("canonical row requires explicit loss_kind from parquet type/target column")

    sample_id = str(row.get("sample_id") or row.get("id") or stable_hash({"split": split, "row_index": row_index, "row": row}))
    metadata = dict(row.get("metadata") or {})
    metadata.setdefault("source_hash", stable_hash(row))

    if loss_kind == "dpo_target":
        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not isinstance(prompt, list) or not isinstance(chosen, dict) or not isinstance(rejected, dict):
            raise CanonicalizationError("dpo_target row requires prompt list and chosen/rejected mappings")
        return CanonicalRow(
            sample_id=sample_id,
            split=split,
            loss_kind=loss_kind,
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            metadata=metadata,
        )

    messages = row.get("messages")
    if not isinstance(messages, list):
        raise CanonicalizationError(f"{loss_kind} row requires messages list")
    tools = row.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise CanonicalizationError("tools must be a list when present")

    return CanonicalRow(
        sample_id=sample_id,
        split=split,
        loss_kind=loss_kind,
        messages=messages,
        tools=tools,
        parallel_tool_calls=row.get("parallel_tool_calls"),
        metadata=metadata,
    )


def stable_uniform_0_1(key: str) -> float:
    """Map a string key to a deterministic float in `[0, 1)`."""

    digest = hashlib.sha256(key.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / float(2**64)


def select_sft_target_spans(
    row: CanonicalRow,
    assistant_spans: list[AssistantSpan],
    policy: dict[str, Any],
) -> tuple[list[TargetSpan], TargetSelectionSummary]:
    """Select `sft_target` assistant spans by length plus deterministic short sampling.

    All assistant messages are candidates. Long replies are always kept. Short
    replies are kept with `loss_on_short_assistant_reply_prob` using a stable
    hash over `(sample_id, turn_index, seed)` so the decision is independent of
    dataloader order and runtime RNG state.
    """

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
        if stable_uniform_0_1(f"{row.sample_id}:{index}:{seed}") < keep_probability:
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
    """Select every assistant span for `sft_tool`."""

    selected = [TargetSpan(span.message_index, span.start, span.end, "all_assistant") for span in assistant_spans]
    summary = TargetSelectionSummary(
        num_target_candidates=len(selected),
        num_long_targets_kept=len(selected),
        num_short_targets_total=0,
        num_short_targets_kept=0,
        num_targets_dropped_by_policy=0,
    )
    return selected, summary


def build_labels(
    input_ids: list[int],
    offsets: list[tuple[int, int]],
    supervised_char_ranges: list[tuple[int, int]],
    ignore_index: int,
    require_positive: bool = True,
) -> tuple[list[int], int]:
    """Build token labels from rendered-text character supervision ranges.

    The renderer/masker decides supervision in character space because that is
    where chat-template assistant spans are observable. Tokenization then gives
    offsets back into the same rendered string. A token is supervised if its
    offset overlaps any selected assistant completion range; every other token
    is set to `ignore_index`, including system/user/tool text, role headers,
    padding-like zero-width offsets, and unselected assistant turns.
    """

    if len(input_ids) != len(offsets):
        raise MaskingError("input_ids and offsets must have the same length")

    labels: list[int] = []
    supervised = 0
    for token_id, (start, end) in zip(input_ids, offsets):
        # Fast tokenizers may emit zero-width offsets for special/control tokens.
        # They are never supervised because there is no rendered character span
        # proving they belong to a selected completion.
        is_supervised = end > start and any(start < span_end and end > span_start for span_start, span_end in supervised_char_ranges)
        labels.append(token_id if is_supervised else ignore_index)
        supervised += int(is_supervised)

    if require_positive and supervised == 0:
        raise MaskingError("accepted training sample has zero supervised tokens")
    return labels, supervised


def tokenize_with_offsets(tokenizer: Any, text: str, add_special_tokens: bool = False) -> dict[str, Any]:
    """Tokenize rendered text and require character offsets for label construction."""

    encoded = tokenizer(text, add_special_tokens=add_special_tokens, return_offsets_mapping=True)
    if "offset_mapping" not in encoded:
        raise MaskingError("tokenizer must return offset_mapping for preprocessing")
    return {
        "input_ids": list(encoded["input_ids"]),
        "attention_mask": list(encoded.get("attention_mask", [1] * len(encoded["input_ids"]))),
        "offset_mapping": [tuple(item) for item in encoded["offset_mapping"]],
    }

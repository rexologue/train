from __future__ import annotations

from typing import Any


class MaskingError(ValueError):
    pass


def build_labels(
    input_ids: list[int],
    offsets: list[tuple[int, int]],
    supervised_char_ranges: list[tuple[int, int]],
    ignore_index: int,
    require_positive: bool = True,
) -> tuple[list[int], int]:
    if len(input_ids) != len(offsets):
        raise MaskingError("input_ids and offsets must have the same length")

    labels: list[int] = []
    supervised = 0
    for token_id, (start, end) in zip(input_ids, offsets):
        is_supervised = end > start and any(start < span_end and end > span_start for span_start, span_end in supervised_char_ranges)
        if is_supervised:
            labels.append(token_id)
            supervised += 1
        else:
            labels.append(ignore_index)

    if require_positive and supervised == 0:
        raise MaskingError("accepted training sample has zero supervised tokens")
    return labels, supervised


def tokenize_with_offsets(tokenizer: Any, text: str, add_special_tokens: bool = False) -> dict[str, Any]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens, return_offsets_mapping=True)
    if "offset_mapping" not in encoded:
        raise MaskingError("tokenizer must return offset_mapping for preprocessing")
    return {
        "input_ids": list(encoded["input_ids"]),
        "attention_mask": list(encoded.get("attention_mask", [1] * len(encoded["input_ids"]))),
        "offset_mapping": [tuple(item) for item in encoded["offset_mapping"]],
    }


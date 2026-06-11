from __future__ import annotations

from typing import Any


def pad_sequences(rows: list[list[int]], pad_value: int) -> list[list[int]]:
    max_len = max(len(row) for row in rows)
    return [row + [pad_value] * (max_len - len(row)) for row in rows]


class SFTCollator:
    def __init__(self, pad_token_id: int, ignore_index: int):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        loss_kinds = {example["loss_kind"] for example in examples}
        if len(loss_kinds) != 1:
            raise ValueError(f"batch must be homogeneous by loss_kind, got {sorted(loss_kinds)}")
        return {
            "input_ids": pad_sequences([example["input_ids"] for example in examples], self.pad_token_id),
            "attention_mask": pad_sequences([example["attention_mask"] for example in examples], 0),
            "labels": pad_sequences([example["labels"] for example in examples], self.ignore_index),
            "loss_kind": examples[0]["loss_kind"],
            "sample_id": [example["sample_id"] for example in examples],
        }


from __future__ import annotations

from typing import Any

import torch


SFT_LOSS_KINDS = {"sft_target", "sft_tool"}
DPO_LOSS_KIND = "dpo_target"


def pad_sequences(rows: list[list[int]], pad_value: int) -> list[list[int]]:
    """Pad variable-length token sequences to the longest row in the batch."""

    if not rows:
        return []
    max_len = max(len(row) for row in rows)
    return [row + [pad_value] * (max_len - len(row)) for row in rows]


class RoutedCollator:
    """Collate pretokenized homogeneous route batches.

    `RoutedBatchSampler` is expected to prevent mixed batches. The collator
    repeats that check because mixed `loss_kind` tensors would be ambiguous for
    loss dispatch and would make DPO/SFT padding rules conflict.
    """

    def __init__(self, pad_token_id: int, ignore_index: int):
        """Create a collator with tokenizer padding and label ignore ids."""

        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Pad and tensorize examples according to their shared loss route."""

        if not examples:
            raise ValueError("batch must contain at least one example")

        loss_kinds = {example["loss_kind"] for example in examples}

        if len(loss_kinds) != 1:
            raise ValueError(f"batch must be homogeneous by loss_kind, got {sorted(loss_kinds)}")

        loss_kind = examples[0]["loss_kind"]

        if loss_kind in SFT_LOSS_KINDS:
            return self._collate_sft(examples)
        if loss_kind == DPO_LOSS_KIND:
            return self._collate_dpo(examples)

        raise ValueError(f"unknown loss_kind {loss_kind!r}")

    def _tensorize(self, rows: list[list[int]]) -> Any:
        """Create long tensors."""

        return torch.tensor(rows, dtype=torch.long)

    def _collate_sft(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Pad SFT/tool single-sequence fields."""

        return {
            "input_ids": self._tensorize(
                pad_sequences([example["input_ids"] for example in examples], self.pad_token_id)
            ),
            "attention_mask": self._tensorize(
                pad_sequences([example["attention_mask"] for example in examples], 0)
            ),
            "labels": self._tensorize(
                pad_sequences([example["labels"] for example in examples], self.ignore_index)
            ),
            "loss_kind": examples[0]["loss_kind"],
            "sample_id": [example["sample_id"] for example in examples],
            "row_index": [example["row_index"] for example in examples],
        }

    def _collate_dpo(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Pad chosen and rejected branches independently for DPO loss."""

        batch = {
            "chosen_input_ids": self._tensorize(
                pad_sequences([example["chosen_input_ids"] for example in examples], self.pad_token_id)
            ),
            "chosen_attention_mask": self._tensorize(
                pad_sequences([example["chosen_attention_mask"] for example in examples], 0)
            ),
            "chosen_labels": self._tensorize(
                pad_sequences([example["chosen_labels"] for example in examples], self.ignore_index)
            ),
            "rejected_input_ids": self._tensorize(
                pad_sequences([example["rejected_input_ids"] for example in examples], self.pad_token_id)
            ),
            "rejected_attention_mask": self._tensorize(
                pad_sequences([example["rejected_attention_mask"] for example in examples], 0)
            ),
            "rejected_labels": self._tensorize(
                pad_sequences([example["rejected_labels"] for example in examples], self.ignore_index)
            ),
            "loss_kind": DPO_LOSS_KIND,
            "sample_id": [example["sample_id"] for example in examples],
            "row_index": [example["row_index"] for example in examples],
            "chosen_render_hash": [example.get("chosen_render_hash") for example in examples],
            "rejected_render_hash": [example.get("rejected_render_hash") for example in examples],
        }
        return batch
